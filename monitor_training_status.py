#!/usr/bin/env python3
"""Continuously refreshes a human-readable training status file for a
DreamerV3 Crafter run, without touching the training process itself.

Reads metrics.jsonl / scores.jsonl in this logdir, computes stats over the
most recent WINDOW_STEPS env steps, and rewrites STATUS_FILE every
REFRESH_SEC seconds.
"""

import argparse
import json
import os
import time
import traceback
from bisect import bisect_right
from collections import deque
from datetime import datetime, timedelta
from math import exp, log
from pathlib import Path

import yaml

LOGDIR = Path(__file__).resolve().parent
METRICS_FILE = LOGDIR / "metrics.jsonl"
SCORES_FILE = LOGDIR / "scores.jsonl"
CONFIG_FILE = LOGDIR / "config.yaml"
STATUS_FILE = LOGDIR / "training_status.md"
BENCHMARK_PLOT = LOGDIR / "crafter_benchmark.png"
REWEIGHTING_PLOT = LOGDIR / "achievement_reweighting.png"
ENV_STATS_FILE = LOGDIR / "env0" / "stats.jsonl"
BASELINE_LOGDIR = Path(
    "/home/aikusrv01/hyejineom/logdir/dreamer/"
    "20260806T070402_dreamer3_size166m_seed0_8gpu")
BASELINE_METRICS_FILE = BASELINE_LOGDIR / "metrics.jsonl"
BASELINE_SCORES_FILE = BASELINE_LOGDIR / "scores.jsonl"

WINDOW_STEPS = 10_000
BENCHMARK_BUDGET = 1_000_000
REFRESH_SEC = 30
TAIL_START_BYTES = 2 * 1024 * 1024
TAIL_MAX_BYTES = 128 * 1024 * 1024
_LAST_PLOT_STEP = None
_LAST_REWEIGHTING_PLOT_STEP = None
_BASELINE_METRICS = None
_BASELINE_SCORES = None

# This fresh run starts with binary achievement logging enabled, so all
# achievement/health records in metrics.jsonl are valid from byte 0.
ACHIEVEMENT_VALID_FROM_BYTE = 0

# This is a fresh 8GPU run, so all performance rows are valid from byte 0.
PERFORMANCE_VALID_FROM_BYTE = 0

ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]


def configure_logdir(path):
    """Point the monitor at a run without copying this script into it."""
    global LOGDIR, METRICS_FILE, SCORES_FILE, CONFIG_FILE, STATUS_FILE
    global BENCHMARK_PLOT, REWEIGHTING_PLOT, ENV_STATS_FILE
    LOGDIR = Path(path).resolve()
    METRICS_FILE = LOGDIR / "metrics.jsonl"
    SCORES_FILE = LOGDIR / "scores.jsonl"
    CONFIG_FILE = LOGDIR / "config.yaml"
    STATUS_FILE = LOGDIR / "training_status.md"
    BENCHMARK_PLOT = LOGDIR / "crafter_benchmark.png"
    REWEIGHTING_PLOT = LOGDIR / "achievement_reweighting.png"
    ENV_STATS_FILE = LOGDIR / "env0" / "stats.jsonl"


def tail_records(path, min_step_span, key="step"):
    """Read growing chunks from the end of a jsonl file until the parsed
    records span at least `min_step_span` in `key`, or the whole file/size
    cap is reached. Avoids re-reading a multi-hundred-MB file every cycle."""
    if not path.exists():
        return []
    chunk = TAIL_START_BYTES
    with open(path, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        while True:
            read_size = min(chunk, filesize)
            f.seek(filesize - read_size)
            data = f.read(read_size)
            lines = data.split(b"\n")
            if read_size < filesize:
                lines = lines[1:]  # drop partial first line
            records = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            steps = [r[key] for r in records if key in r]
            span = (max(steps) - min(steps)) if steps else 0
            covered_all = read_size >= filesize
            if span >= min_step_span or covered_all or read_size >= TAIL_MAX_BYTES:
                return records
            chunk *= 4


def read_records_from_offset(path, min_offset):
    """Read all jsonl records starting at a fixed byte offset. Used to skip
    over rows written before the achievement-logging fix, regardless of
    what 'step' value they claim."""
    if not path.exists():
        return []
    records = []
    with open(path, "rb") as f:
        f.seek(min_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_records(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def baseline_records():
    """Load the completed baseline once; it no longer changes."""
    global _BASELINE_METRICS, _BASELINE_SCORES
    if _BASELINE_METRICS is None:
        _BASELINE_METRICS = read_records(BASELINE_METRICS_FILE)
    if _BASELINE_SCORES is None:
        _BASELINE_SCORES = read_records(BASELINE_SCORES_FILE)
    return _BASELINE_METRICS, _BASELINE_SCORES


def avg(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def benchmark_reward(row):
    """Return untouched Crafter reward, falling back for baseline logs."""
    value = row.get("episode/benchmark_score")
    if value is None:
        value = row.get("episode/score")
    return float(value) if value is not None else None


def attach_official_rewards(scores):
    """Use the environment's per-episode raw reward as ground truth."""
    stats = read_records(ENV_STATS_FILE)
    for row, episode in zip(scores, stats):
        if episode.get("reward") is not None:
            row["episode/benchmark_score"] = float(episode["reward"])
    return scores


def fmt(x, digits=3, suffix=""):
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}{suffix}"


def fmt_duration(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))


def build_achievement_intervals(metrics, scores, end_step):
    """Recover exact achievement totals for each completed logger interval.

    Dreamer stores the mean per-episode unlocked flag for each logger
    interval. Multiplying that mean by the number of episode endings in the
    same interval recovers the number of successful episodes exactly.
    """
    episode_steps = sorted(
        r["step"] for r in scores
        if "episode/score" in r and r.get("step", -1) <= end_step)
    intervals = []
    episode_index = 0

    for row in sorted(metrics, key=lambda r: r.get("step", -1)):
        step = row.get("step", -1)
        if step < 0 or step > end_step:
            continue
        if not any(
                f"epstats/log/achievement_{name}/max" in row
                for name in ACHIEVEMENTS):
            continue

        start_index = episode_index
        while (episode_index < len(episode_steps) and
               episode_steps[episode_index] <= step):
            episode_index += 1
        steps = episode_steps[start_index:episode_index]
        if not steps:
            continue

        count = len(steps)
        totals = {}
        for name in ACHIEVEMENTS:
            key = f"epstats/log/achievement_{name}/max"
            totals[name] = float(row.get(key, 0.0)) * count
        intervals.append({
            "step": step,
            "first_episode_step": steps[0],
            "last_episode_step": steps[-1],
            "episodes": count,
            "totals": totals,
        })

    covered_step = intervals[-1]["step"] if intervals else 0
    pending = sum(covered_step < step <= end_step for step in episode_steps)
    return intervals, pending, covered_step


def score_from_totals(totals, episodes):
    if episodes <= 0:
        return None, None
    rates = {
        name: 100.0 * totals[name] / episodes
        for name in ACHIEVEMENTS}
    score = exp(
        sum(log(1.0 + rates[name]) for name in ACHIEVEMENTS) /
        len(ACHIEVEMENTS)) - 1.0
    return score, rates


def summarize_benchmark(metrics, scores, current_step):
    """Use 0..current below 1M steps and a rolling 1M window afterwards."""
    start_step = max(0, current_step - BENCHMARK_BUDGET)
    reward_rows = sorted(
        (r for r in scores if benchmark_reward(r) is not None),
        key=lambda r: r.get("step", -1))
    reward_values = [
        benchmark_reward(r) for r in reward_rows
        if start_step < r.get("step", -1) <= current_step]

    intervals, pending, covered_step = build_achievement_intervals(
        metrics, scores, current_step)
    selected = [
        item for item in intervals
        if item["first_episode_step"] > start_step]
    boundary_excluded = sum(
        item["episodes"] for item in intervals
        if item["first_episode_step"] <= start_step < item["last_episode_step"])
    totals = {name: 0.0 for name in ACHIEVEMENTS}
    episodes = 0
    for item in selected:
        episodes += item["episodes"]
        for name in ACHIEVEMENTS:
            totals[name] += item["totals"][name]
    score, rates = score_from_totals(totals, episodes)

    return {
        "start_step": start_step,
        "end_step": current_step,
        "reward": (
            sum(reward_values) / len(reward_values)
            if reward_values else None),
        "reward_episodes": len(reward_values),
        "score": score,
        "rates": rates,
        "successes": totals,
        "achievement_episodes": episodes,
        "pending": pending,
        "boundary_excluded": boundary_excluded,
        "covered_step": covered_step,
        "intervals": intervals,
        "reward_rows": reward_rows,
    }


def benchmark_history(summary):
    """Return cumulative curves until 1M and rolling-1M curves afterwards."""
    intervals = summary["intervals"]
    reward_rows = summary["reward_rows"]
    reward_steps = [r["step"] for r in reward_rows]
    reward_prefix = [0.0]
    for row in reward_rows:
        reward_prefix.append(reward_prefix[-1] + benchmark_reward(row))

    active = deque()
    totals = {name: 0.0 for name in ACHIEVEMENTS}
    episodes = 0
    history = []
    for item in intervals:
        active.append(item)
        episodes += item["episodes"]
        for name in ACHIEVEMENTS:
            totals[name] += item["totals"][name]

        cutoff = max(0, item["step"] - BENCHMARK_BUDGET)
        # Exclude the entire logger interval that straddles the rolling
        # boundary; its per-episode achievement values cannot be separated.
        while active and active[0]["first_episode_step"] <= cutoff:
            old = active.popleft()
            episodes -= old["episodes"]
            for name in ACHIEVEMENTS:
                totals[name] -= old["totals"][name]

        crafter_score, _ = score_from_totals(totals, episodes)
        left = bisect_right(reward_steps, cutoff)
        right = bisect_right(reward_steps, item["step"])
        reward = None
        if right > left:
            reward = (
                reward_prefix[right] - reward_prefix[left]) / (right - left)
        history.append((item["step"], reward, crafter_score))
    return history


def write_benchmark_plot(summary):
    global _LAST_PLOT_STEP
    upgrade_history = benchmark_history(summary)
    if not upgrade_history:
        return
    latest_step = upgrade_history[-1][0]
    if latest_step == _LAST_PLOT_STEP and BENCHMARK_PLOT.exists():
        return

    baseline_metrics, baseline_scores = baseline_records()
    baseline_summary = summarize_benchmark(
        baseline_metrics, baseline_scores, summary["end_step"])
    baseline_history = benchmark_history(baseline_summary)

    os.environ.setdefault("MPLCONFIGDIR", str(LOGDIR / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row[0] for row in upgrade_history]
    rewards = [row[1] for row in upgrade_history]
    crafter_scores = [row[2] for row in upgrade_history]
    fig, left = plt.subplots(figsize=(10, 4.8))
    right = left.twinx()
    lines = []
    if baseline_history:
        baseline_steps = [row[0] for row in baseline_history]
        lines += left.plot(
            baseline_steps, [row[1] for row in baseline_history],
            color="#e07a1f", linestyle="--", linewidth=1.8,
            alpha=0.75, label="Baseline Reward")
        lines += right.plot(
            baseline_steps, [row[2] for row in baseline_history],
            color="#2878b5", linestyle="--", linewidth=1.8,
            alpha=0.75, label="Baseline Crafter Score")
    lines += left.plot(
        steps, rewards, color="#e07a1f", linewidth=2.3,
        label="Curious Reweighting Reward")
    lines += right.plot(
        steps, crafter_scores, color="#2878b5", linewidth=2.3,
        label="Curious Reweighting Crafter Score")
    left.set_xlabel("Environment steps")
    left.set_ylabel("Reward", color="#e07a1f")
    right.set_ylabel("Crafter Score (%)", color="#2878b5")
    left.tick_params(axis="y", labelcolor="#e07a1f")
    right.tick_params(axis="y", labelcolor="#2878b5")
    left.grid(True, alpha=0.25)
    left.ticklabel_format(axis="x", style="plain")
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.suptitle(
        "Crafter benchmark comparison: cumulative to 1M, then rolling 1M")
    fig.tight_layout()
    tmp = BENCHMARK_PLOT.with_suffix(".png.tmp")
    fig.savefig(tmp, format="png", dpi=150)
    plt.close(fig)
    tmp.replace(BENCHMARK_PLOT)
    _LAST_PLOT_STEP = latest_step


def reweighting_snapshot(benchmark, current_step):
    """Current adaptive weights and equal-step baseline success rates."""
    stats = read_records(ENV_STATS_FILE)
    latest = stats[-1] if stats else {}
    weights = {
        name: float(latest.get(f"achievement_next_weight_{name}", 1.0))
        for name in ACHIEVEMENTS}
    updates = int(latest.get("achievement_weight_updates", 0))

    baseline_metrics, baseline_scores = baseline_records()
    baseline_summary = summarize_benchmark(
        baseline_metrics, baseline_scores, current_step)
    return {
        "weights": weights,
        "updates": updates,
        "episodes": len(stats),
        "rates": benchmark.get("rates") or {
            name: 0.0 for name in ACHIEVEMENTS},
        "baseline_rates": baseline_summary.get("rates") or {
            name: 0.0 for name in ACHIEVEMENTS},
    }


def achievement_breadth(rates):
    rates = rates or {}
    return {
        "nonzero": sum(value > 0.0 for value in rates.values()),
        "over_1": sum(value >= 1.0 for value in rates.values()),
        "over_10": sum(value >= 10.0 for value in rates.values()),
    }


def sum_values(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return sum(values) if values else None


def curious_replay_snapshot(metrics, current_step, config):
    """Summarize priority signal health and actual sampler fractions."""
    window_start = current_step - WINDOW_STEPS
    window = [r for r in metrics if r.get("step", -1) >= window_start]
    replay_config = config.get("replay", {})
    fractions = replay_config.get("fracs", {})
    priority_count = sum_values(window, "replay/sample_priority_count")
    recency_count = sum_values(window, "replay/sample_recency_count")
    sample_total = (priority_count or 0) + (recency_count or 0)
    return {
        "configured_uniform": float(fractions.get("uniform", 0.0)),
        "configured_priority": float(fractions.get("priority", 0.0)),
        "configured_recency": float(fractions.get("recency", 0.0)),
        "sample_priority_count": priority_count,
        "sample_recency_count": recency_count,
        "sample_total": sample_total,
        "sample_priority_fraction": (
            priority_count / sample_total if sample_total else None),
        "sample_recency_fraction": (
            recency_count / sample_total if sample_total else None),
        "priority_count": sum_values(window, "replay/priority_count"),
        "priority_mean": avg(window, "replay/priority_mean"),
        "priority_std": avg(window, "replay/priority_std"),
        "priority_nonfinite": sum_values(
            window, "replay/priority_nonfinite"),
        "model_priority_mean": avg(
            window, "train/curious_replay/model_priority_mean"),
        "model_priority_std": avg(
            window, "train/curious_replay/model_priority_std"),
    }


def write_reweighting_plot(snapshot, current_step):
    """Show both the intended mechanism and its equal-step outcome."""
    global _LAST_REWEIGHTING_PLOT_STEP
    if (_LAST_REWEIGHTING_PLOT_STEP == current_step and
            REWEIGHTING_PLOT.exists()):
        return

    weights = snapshot["weights"]
    current_rates = snapshot["rates"]
    baseline_rates = snapshot["baseline_rates"]
    names = sorted(
        ACHIEVEMENTS,
        key=lambda name: (
            -weights[name], current_rates[name], baseline_rates[name], name))

    os.environ.setdefault("MPLCONFIGDIR", str(LOGDIR / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.arange(len(names))
    fig, (success_axis, weight_axis) = plt.subplots(
        1, 2, figsize=(14, 9), sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]})

    bar_height = 0.38
    success_axis.barh(
        y - bar_height / 2,
        [baseline_rates[name] for name in names],
        height=bar_height, color="#9aa3a4", label="Baseline 166M")
    success_axis.barh(
        y + bar_height / 2,
        [current_rates[name] for name in names],
        height=bar_height, color="#7b61b3",
        label="Curious Reweighting 166M")
    success_axis.set_xscale("symlog", linthresh=1.0)
    success_axis.set_xlim(0, 100)
    success_axis.set_xticks([0, 1, 5, 10, 25, 50, 100])
    success_axis.set_xticklabels(["0", "1", "5", "10", "25", "50", "100"])
    success_axis.set_xlabel("Achievement success rate (%)")
    success_axis.set_yticks(y)
    success_axis.set_yticklabels(names, fontsize=9)
    success_axis.invert_yaxis()
    success_axis.grid(True, axis="x", alpha=0.25)
    success_axis.legend(loc="lower right")

    weight_values = [weights[name] for name in names]
    colors = [
        "#6f42c1" if value > 1.0 else
        "#d58a2a" if value < 1.0 else "#7f8c8d"
        for value in weight_values]
    weight_axis.barh(y, weight_values, color=colors, height=0.62)
    weight_axis.axvline(
        1.0, color="#303030", linestyle="--", linewidth=1.4,
        label="Baseline weight = 1")
    weight_axis.set_xlim(0, 2.1)
    weight_axis.set_xlabel("Current achievement weight")
    weight_axis.grid(True, axis="x", alpha=0.25)
    weight_axis.legend(loc="lower right")

    fig.suptitle(
        "Achievement reweighting: equal-step success rates and current weights\n"
        f"Environment step {current_step:,} · weight updates {snapshot['updates']}")
    fig.tight_layout()
    tmp = REWEIGHTING_PLOT.with_suffix(".png.tmp")
    fig.savefig(tmp, format="png", dpi=150)
    plt.close(fig)
    tmp.replace(REWEIGHTING_PLOT)
    _LAST_REWEIGHTING_PLOT_STEP = current_step


def build_status():
    config = yaml.safe_load(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    target_steps = float(config.get("run", {}).get("steps", 0) or 0)
    task = config.get("task", "unknown")

    metrics = tail_records(METRICS_FILE, WINDOW_STEPS)
    all_scores = attach_official_rewards(read_records(SCORES_FILE))

    if not metrics:
        replay_fracs = config.get("replay", {}).get("fracs", {})
        env_config = config.get("env", {}).get("crafter", {})
        lines = [
            "# Curious Reweighting Crafter 학습 상태\n",
            f"_마지막 갱신: {datetime.now():%Y-%m-%d %H:%M:%S}_  ",
            f"_logdir: `{LOGDIR}`_\n",
            "## 전체 진행 상태",
            f"- Task: `{task}`",
            f"- 현재 스텝: **0** / 목표 {int(target_steps):,}",
            "- 초기 컴파일 또는 replay warm-up 중입니다.",
            "",
            "## Crafter 벤치마크",
            "> 아직 완료된 episode metric이 없어 Reward와 Crafter Score를 집계하지 않았습니다.",
            "",
            "### Achievement Success Rates (22개)",
            "",
            "최초 완료 episode 후 22개 achievement 성공률 표가 생성됩니다.",
            "",
            "### Reward · Crafter Score 추이",
            "",
            "최초 metric 로그 후 baseline 비교 그래프가 생성됩니다.",
            "",
            "### 성공률별 Achievement 개수",
            "",
            "| 기준 | Curious Reweighting | 동일 step 166M Baseline |",
            "|---|---:|---:|",
            "| 한 번이라도 달성 | 대기 중 | 대기 중 |",
            "| 성공률 1% 이상 | 대기 중 | 대기 중 |",
            "| 성공률 10% 이상 | 대기 중 | 대기 중 |",
            "",
            "## Achievement reweighting 작동 상태",
            "",
            f"- 활성화: **{bool(env_config.get('achievement_reweight', False))}**",
            f"- scale: **{env_config.get('achievement_scale', 'N/A')}**",
            f"- update interval: **{env_config.get('achievement_update_interval', 'N/A')} episodes**",
            "- 현재 상태: 첫 weight update를 기다리는 중",
            "",
            "## Curious Replay 작동 상태",
            "",
            "| 지표 | 값 | 기대/판단 |",
            "|---|---:|---|",
            f"| Uniform / Priority / Recency 설정 | "
            f"{float(replay_fracs.get('uniform', 0)):.0%} / "
            f"{float(replay_fracs.get('priority', 0)):.0%} / "
            f"{float(replay_fracs.get('recency', 0)):.0%} | 0% / 50% / 50% |",
            "| 실제 sampler 선택률 | 대기 중 | replay warm-up 후 집계 |",
            "",
        ]
        return "\n".join(lines)

    performance_metrics = read_records_from_offset(
        METRICS_FILE, PERFORMANCE_VALID_FROM_BYTE)
    if not performance_metrics:
        performance_metrics = metrics

    last_step = max(r["step"] for r in performance_metrics if "step" in r)
    window_start = last_step - WINDOW_STEPS
    window_metrics = [
        r for r in performance_metrics if r.get("step", -1) >= window_start]

    clean_metrics = read_records_from_offset(
        METRICS_FILE, ACHIEVEMENT_VALID_FROM_BYTE)

    # config.yaml is written once when the training process starts and never
    # touched again, so its mtime is a reliable process-start timestamp
    # (unlike the logdir's own ctime, which changes whenever any file in it
    # is created or modified).
    started = datetime.fromtimestamp(CONFIG_FILE.stat().st_mtime)
    elapsed = datetime.now() - started

    fps_policy = avg(window_metrics, "fps/policy")
    fps_train = avg(window_metrics, "fps/train")
    eta_sec = None
    if fps_policy and fps_policy > 0 and target_steps:
        eta_sec = max(0.0, (target_steps - last_step) / fps_policy)

    pct = (last_step / target_steps * 100) if target_steps else None

    benchmark = summarize_benchmark(clean_metrics, all_scores, last_step)
    write_benchmark_plot(benchmark)
    reweighting = reweighting_snapshot(benchmark, last_step)
    write_reweighting_plot(reweighting, last_step)
    curious = curious_replay_snapshot(performance_metrics, last_step, config)

    gpu_idx = config.get("jax", {}).get("train_devices", [0, 1, 2, 3])
    gpu_lines = []
    for i in gpu_idx:
        v = avg(window_metrics, f"usage/nvsmi/compute_avg/gpu{i}")
        gpu_lines.append(f"  - GPU{i}: {fmt(v, 2)}" if v is not None else f"  - GPU{i}: N/A")

    lines = []
    lines.append("# Curious Reweighting Crafter 학습 상태\n")
    lines.append(f"_마지막 갱신: {datetime.now():%Y-%m-%d %H:%M:%S}_  ")
    lines.append(f"_logdir: `{LOGDIR}`_\n")

    lines.append("## 전체 진행 상태")
    lines.append(f"- Task: `{task}`")
    lines.append(f"- 현재 스텝: **{int(last_step):,}** / 목표 {int(target_steps):,} ({fmt(pct, 1, '%') if pct is not None else 'N/A'})")
    lines.append(f"- 학습 시작 이후 경과 시간: {fmt_duration(elapsed.total_seconds())} (시작: {started:%Y-%m-%d %H:%M})")
    lines.append(f"- 예상 남은 시간 (최근 fps 기준): {fmt_duration(eta_sec)}")
    lines.append("")

    lines.append("## Crafter 벤치마크")
    if benchmark["start_step"] == 0:
        range_label = f"0 ~ {int(benchmark['end_step']):,} step (현재까지 누적)"
    else:
        range_label = (
            f"{int(benchmark['start_step']):,} ~ "
            f"{int(benchmark['end_step']):,} step (최근 1,000,000 step)")
    lines.append(f"> **집계 범위:** `{range_label}`  ")
    lines.append(
        f"> **완료된 에피소드:** `{benchmark['reward_episodes']:,}`  ")
    lines.append(f"> **Reward:** `{fmt(benchmark['reward'], 2)}`  ")
    lines.append(
        f"> **Crafter Score:** `{fmt(benchmark['score'], 2, '%')}`")
    lines.append("")
    lines.append(
        "Agents are allowed a budget of **1M environment steps** and are "
        "evaluated by the success rates of the 22 achievements and their "
        "geometric mean score. 이 상태표는 1M 미만에서는 0부터 현재까지, "
        "1M 이후에는 최근 1M step을 집계합니다.")
    lines.append("")
    lines.append(
        "- **Reward:** achievement 최초 해제 시 `+1`, 체력 감소/회복 시 "
        "`-0.1/+0.1`인 에피소드 reward의 평균입니다. 공식 비교에서는 reward보다 "
        "success rates와 Crafter Score를 우선합니다.")
    lines.append(
        "- **Success rates:** 각 achievement를 한 번 이상 달성한 에피소드의 "
        "비율입니다.")
    lines.append(
        "- **Crafter Score:** 22개 success rate에 1% offset을 적용한 "
        "기하평균입니다: `exp(mean(log(1 + success_rate))) - 1`.")
    lines.append("")
    lines.append("### 핵심 점수")
    lines.append("")
    lines.append("| 지표 | 현재 값 | 집계 에피소드 |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Reward | **{fmt(benchmark['reward'], 2)}** | "
        f"{benchmark['reward_episodes']:,} |")
    lines.append(
        f"| Crafter Score | **{fmt(benchmark['score'], 2, '%')}** | "
        f"{benchmark['achievement_episodes']:,} |")
    lines.append("")

    if benchmark["rates"]:
        lines.append("### Achievement Success Rates (22개)")
        lines.append("")
        lines.append("| Achievement | 성공률 | 달성 / 전체 에피소드 |")
        lines.append("|---|---:|---:|")
        ach_rows = sorted(
            benchmark["rates"].items(), key=lambda item: (-item[1], item[0]))
        for name, rate in ach_rows:
            successes = int(round(benchmark["successes"][name]))
            lines.append(
                f"| `{name}` | **{rate:.2f}%** | "
                f"{successes:,} / {benchmark['achievement_episodes']:,} |")
        lines.append("")
    else:
        lines.append("아직 achievement score를 계산할 완료 에피소드가 없습니다.")
        lines.append("")

    notes = []
    if benchmark["pending"]:
        notes.append(
            f"logger interval에 아직 반영되지 않은 완료 에피소드 "
            f"{benchmark['pending']:,}개")
    if benchmark["boundary_excluded"]:
        notes.append(
            f"최근 1M 경계에 걸친 logger interval의 에피소드 "
            f"{benchmark['boundary_excluded']:,}개 제외")
    if notes:
        lines.append(f"> **집계 참고:** {'; '.join(notes)}")
        lines.append("")
    lines.append(
        f"> Achievement 최종 반영 step: "
        f"`{int(benchmark['covered_step']):,}` · 단일 seed 0 run")
    lines.append("")
    lines.append("### Reward · Crafter Score 추이")
    lines.append("")
    lines.append("![Crafter Reward and Score](crafter_benchmark.png)")
    lines.append("")

    current_breadth = achievement_breadth(benchmark.get("rates"))
    baseline_breadth = achievement_breadth(reweighting["baseline_rates"])
    lines.append("### 성공률별 Achievement 개수")
    lines.append("")
    lines.append("| 기준 | Curious Reweighting | 동일 step 166M Baseline |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| 한 번이라도 달성 | **{current_breadth['nonzero']} / 22** | "
        f"{baseline_breadth['nonzero']} / 22 |")
    lines.append(
        f"| 성공률 1% 이상 | **{current_breadth['over_1']} / 22** | "
        f"{baseline_breadth['over_1']} / 22 |")
    lines.append(
        f"| 성공률 10% 이상 | **{current_breadth['over_10']} / 22** | "
        f"{baseline_breadth['over_10']} / 22 |")
    lines.append("")

    lines.append("## Achievement reweighting 작동 상태")
    lines.append("")
    lines.append(
        "> **설정:** 최근 100개 완료 episode마다 success rate의 역수로 "
        "weight 갱신 · `epsilon=0.01` · `weight range=0.5~2.0` · `scale=1.0`  ")
    lines.append(
        f"> **완료 episode:** `{reweighting['episodes']:,}` · "
        f"**weight 갱신 횟수:** `{reweighting['updates']:,}` · "
        f"**현재 weight 범위:** "
        f"`{min(reweighting['weights'].values()):.3f} ~ "
        f"{max(reweighting['weights'].values()):.3f}`")
    if reweighting["updates"] == 0:
        lines.append(
            "> 첫 100개 episode warm-up 전이므로 모든 weight가 1.0입니다.")
    lines.append("")
    lines.append(
        "왼쪽은 동일 학습 step에서 baseline 대비 achievement별 success "
        "rate를, 오른쪽은 현재 adaptive weight를 보여줍니다.")
    lines.append("")
    lines.append(
        "![Achievement Reweighting](achievement_reweighting.png)")
    lines.append("")

    lines.append("## Curious Replay 작동 상태")
    lines.append("")
    lines.append(
        "> 월드모델 prediction error priority와 recency sampling이 "
        "설정한 비율대로 동작하는지 최근 10,000 step에서 확인합니다.")
    lines.append("")
    lines.append("| 지표 | 값 | 기대/판단 |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Uniform / Priority / Recency 설정 | "
        f"{curious['configured_uniform']:.0%} / "
        f"**{curious['configured_priority']:.0%}** / "
        f"**{curious['configured_recency']:.0%}** | 0% / 50% / 50% |")
    priority_fraction = curious["sample_priority_fraction"]
    recency_fraction = curious["sample_recency_fraction"]
    lines.append(
        f"| Priority 실제 선택률 | "
        f"**{fmt(priority_fraction * 100 if priority_fraction is not None else None, 2, '%')}** "
        f"({fmt(curious['sample_priority_count'], 0)}회) | 설정 50% |")
    lines.append(
        f"| Recency 실제 선택률 | "
        f"**{fmt(recency_fraction * 100 if recency_fraction is not None else None, 2, '%')}** "
        f"({fmt(curious['sample_recency_count'], 0)}회) | 설정 50% |")
    lines.append(
        f"| Replay priority 수 | **{fmt(curious['priority_count'], 0)}** | "
        "priority update 활성 여부 |")
    lines.append(
        f"| Priority 평균 / 표준편차 | "
        f"**{fmt(curious['priority_mean'], 4)} / "
        f"{fmt(curious['priority_std'], 4)}** | 신호 변별력 |")
    lines.append(
        f"| 비정상 priority (NaN/Inf) | "
        f"**{fmt(curious['priority_nonfinite'], 0)}** | 반드시 0 |")
    lines.append(
        f"| Agent model priority 평균 / 표준편차 | "
        f"**{fmt(curious['model_priority_mean'], 4)} / "
        f"{fmt(curious['model_priority_std'], 4)}** | 원본 surprise 신호 |")
    lines.append("")
    if priority_fraction is None:
        lines.append(
            "> 아직 sampler 지표가 없어 replay warm-up을 기다리는 중입니다.")
    elif (curious["priority_nonfinite"] or 0) > 0:
        lines.append(
            "> **주의:** NaN/Inf priority가 검출됐습니다.")
    elif curious["sample_total"] >= 1000 and abs(
            priority_fraction - curious["configured_priority"]) <= 0.05:
        lines.append(
            "> Priority/Recency 혼합과 priority 신호가 설정대로 동작 중입니다.")
    else:
        lines.append(
            "> 초기 표본 구간입니다. 선택률이 50:50으로 수렴하는지 확인합니다.")
    lines.append("")

    lines.append("## 학습 손실/최적화 지표 (구간 평균)")
    loss_keys = [
        ("dyn", "train/loss/dyn"), ("rep", "train/loss/rep"),
        ("image", "train/loss/image"), ("reward", "train/loss/rew"),
        ("continue", "train/loss/con"), ("policy", "train/loss/policy"),
        ("value", "train/loss/value"),
    ]
    for label, key in loss_keys:
        lines.append(f"- {label}: {fmt(avg(window_metrics, key))}")
    lines.append(f"- return (train/ret): {fmt(avg(window_metrics, 'train/ret'))}")
    lines.append(f"- reward (train/rew): {fmt(avg(window_metrics, 'train/rew'))}")
    lines.append(f"- grad_norm: {fmt(avg(window_metrics, 'train/opt/grad_norm'))}")
    lines.append(f"- policy entropy: {fmt(avg(window_metrics, 'train/ent/action'))}")
    lines.append("")

    lines.append("## Replay 버퍼")
    lines.append(f"- 저장된 아이템 수: {fmt(avg(window_metrics, 'replay/items'), 0)}")
    lines.append(f"- RAM 사용량: {fmt(avg(window_metrics, 'replay/ram_gb'), 2)} GB")
    lines.append(f"- replay ratio: {fmt(avg(window_metrics, 'replay/replay_ratio'), 1)}")
    lines.append("")

    lines.append("## 리소스 사용량 (구간 평균)")
    lines.append("- GPU 사용률 (train devices):")
    lines.extend(gpu_lines)
    lines.append(f"- CPU 사용률 (전체): {fmt(avg(window_metrics, 'usage/psutil/total_cpu_frac'), 3, '')}")
    lines.append(f"- RAM 사용률 (전체): {fmt(avg(window_metrics, 'usage/psutil/total_ram_frac'), 3, '')}")
    lines.append(f"- 학습 프로세스 RAM: {fmt(avg(window_metrics, 'usage/psutil/proc_ram_gb'), 2)} GB")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.logdir:
        configure_logdir(args.logdir)
    print(f"[monitor] watching {LOGDIR}, refreshing {STATUS_FILE.name} every {REFRESH_SEC}s")
    while True:
        try:
            content = build_status()
            tmp = STATUS_FILE.with_suffix(".md.tmp")
            tmp.write_text(content)
            tmp.replace(STATUS_FILE)
        except Exception:
            traceback.print_exc()
        if args.once:
            break
        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
