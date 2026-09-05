import json

import crafter
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):
  """Crafter adapter with optional score-aligned achievement rewards.

  Every ``achievement_update_interval`` completed episodes, achievement
  weights are recomputed from that block's success rates. Weights are
  proportional to ``1 / (epsilon + success_rate)`` and then normalized and
  clipped. The reward stored in replay is

    train_reward = env_reward + scale * sum((weight_i - 1) * unlock_i).

  The original environment reward and achievements remain available as logs.
  """

  def __init__(
      self, task, size=(64, 64), logs=False, logdir=None, seed=None,
      achievement_reweight=False, achievement_scale=1.0,
      achievement_warmup=100, achievement_epsilon=0.01,
      achievement_weight_min=0.5, achievement_weight_max=2.0,
      achievement_freeze=None, achievement_update_interval=None):
    assert task in ('reward', 'noreward')
    assert not achievement_reweight or task == 'reward', (
        'Achievement reweighting requires task="reward".')
    assert achievement_scale >= 0
    assert achievement_warmup > 0
    assert achievement_epsilon > 0
    assert 0 < achievement_weight_min <= achievement_weight_max
    achievement_update_interval = (
        achievement_warmup
        if achievement_update_interval is None
        else achievement_update_interval)
    assert achievement_update_interval > 0
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._train_reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._achievement_reweight = achievement_reweight
    self._achievement_scale = float(achievement_scale)
    # ``achievement_warmup`` is kept as a backward-compatible name for the
    # first 100-episode block. The new method uses the same-sized blocks for
    # every later update as well.
    self._achievement_update_interval = int(achievement_update_interval)
    self._achievement_epsilon = float(achievement_epsilon)
    self._achievement_weight_min = float(achievement_weight_min)
    self._achievement_weight_max = float(achievement_weight_max)
    # Kept only so older configs that pass achievement_freeze still load. The
    # periodic method intentionally never freezes the weights.
    del achievement_freeze
    self._completed_episodes = 0
    self._achievement_successes = {
        name: 0 for name in self._achievements}
    self._window_episodes = 0
    self._window_achievement_successes = {
        name: 0 for name in self._achievements}
    self._achievement_weights = {
        name: 1.0 for name in self._achievements}
    self._weight_updates = 0
    self._previous_achievements = {
        name: 0 for name in self._achievements}
    self._restore_stats()
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
        'log/train_reward': elements.Space(np.float32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements})
      spaces.update({
          f'log/achievement_weight_{k}': elements.Space(np.float32)
          for k in self._achievements})
      spaces.update({
          f'log/achievement_success_rate_{k}': elements.Space(np.float32)
          for k in self._achievements})
      spaces['log/achievement_weight_updates'] = elements.Space(np.int32)
      spaces['log/health'] = elements.Space(np.int32)
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0.0
      self._train_reward = 0.0
      self._previous_achievements = {
          name: 0 for name in self._achievements}
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, raw_reward=0.0, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    raw_reward = float(reward)
    newly_unlocked = self._newly_unlocked(info['achievements'])
    train_reward = self._reweight_reward(raw_reward, newly_unlocked)
    self._reward += raw_reward
    self._train_reward += train_reward
    self._length += 1
    if self._done:
      weights_used = self._achievement_weights.copy()
      self._finish_episode(info['achievements'])
      if self._logdir:
        self._write_stats(
            self._length, self._reward, self._train_reward, info,
            weights_used)
    return self._obs(
        image, train_reward, info, raw_reward=raw_reward,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _newly_unlocked(self, achievements):
    # Achievement counters can increase repeatedly within an episode. Crafter's
    # sparse reward and score only care about the first unlock in each episode.
    unlocked = {
        name: int(
            self._previous_achievements[name] == 0 and
            achievements[name] > 0)
        for name in self._achievements}
    self._previous_achievements = {
        name: achievements[name] for name in self._achievements}
    return unlocked

  def _reweight_reward(self, reward, newly_unlocked):
    if not self._achievement_reweight:
      return reward
    bonus = sum(
        (self._achievement_weights[name] - 1.0) * unlocked
        for name, unlocked in newly_unlocked.items())
    return reward + self._achievement_scale * bonus

  def _finish_episode(self, achievements):
    self._completed_episodes += 1
    self._window_episodes += 1
    for name in self._achievements:
      success = int(achievements[name] > 0)
      self._achievement_successes[name] += success
      self._window_achievement_successes[name] += success
    self._update_achievement_weights()

  def _update_achievement_weights(self):
    if (
        not self._achievement_reweight or
        self._window_episodes < self._achievement_update_interval):
      return
    rates = np.array([
        self._window_achievement_successes[name] / self._window_episodes
        for name in self._achievements
    ], np.float64)
    weights = 1.0 / (self._achievement_epsilon + rates)
    weights /= weights.mean()
    weights = np.clip(
        weights,
        self._achievement_weight_min,
        self._achievement_weight_max)
    self._achievement_weights = {
        name: float(weight)
        for name, weight in zip(self._achievements, weights)}
    self._weight_updates += 1
    self._window_episodes = 0
    self._window_achievement_successes = {
        name: 0 for name in self._achievements}

  def _restore_stats(self):
    if not self._logdir:
      return
    filename = self._logdir / 'stats.jsonl'
    if not filename.exists():
      return
    for line in filename.read().splitlines():
      try:
        stats = json.loads(line)
      except json.JSONDecodeError:
        continue
      if not all(
          f'achievement_{name}' in stats for name in self._achievements):
        continue
      self._episode = max(self._episode, int(stats.get('episode', 0)))
      self._completed_episodes += 1
      self._window_episodes += 1
      for name in self._achievements:
        success = int(stats[f'achievement_{name}'] > 0)
        self._achievement_successes[name] += success
        self._window_achievement_successes[name] += success
      self._update_achievement_weights()
    if self._completed_episodes:
      print(
          f'Restored achievement statistics from {filename}: '
          f'{self._completed_episodes} episodes')

  def _obs(
      self, image, reward, info,
      raw_reward=None, is_first=False, is_last=False, is_terminal=False):
    raw_reward = reward if raw_reward is None else raw_reward
    logged_reward = info['reward'] if info else raw_reward
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{
            'log/reward': np.float32(logged_reward),
            'log/train_reward': np.float32(reward),
        },
    )
    if self._logs:
      log_achievements = {
          # Binary unlocked-or-not per episode for official success rates.
          f'log/achievement_{k}': int(info['achievements'][k] > 0) if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
      obs.update({
          f'log/achievement_weight_{k}': np.float32(
              self._achievement_weights[k])
          for k in self._achievements})
      obs.update({
          f'log/achievement_success_rate_{k}': np.float32(
              self._achievement_successes[k] / self._completed_episodes
              if self._completed_episodes else 0.0)
          for k in self._achievements})
      obs['log/achievement_weight_updates'] = np.int32(
          self._weight_updates)
      health = info['inventory']['health'] if info else 0
      obs['log/health'] = np.int32(health)
    return obs

  def _write_stats(self, length, reward, train_reward, info, weights_used):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        'train_reward': round(train_reward, 3),
        'achievement_reweight': self._achievement_reweight,
        'achievement_weight_updates': self._weight_updates,
        'achievement_update_interval': self._achievement_update_interval,
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
        **{
            f'achievement_success_rate_{k}': round(
                self._achievement_successes[k] / self._completed_episodes, 6)
            for k in self._achievements},
        **{
            f'achievement_weight_{k}': round(weights_used[k], 6)
            for k in self._achievements},
        **{
            f'achievement_next_weight_{k}': round(
                self._achievement_weights[k], 6)
            for k in self._achievements},
    }
    filename = self._logdir / 'stats.jsonl'
    filename.write(json.dumps(stats) + '\n', mode='a')
    print(f'Wrote stats: {filename}')
