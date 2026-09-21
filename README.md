# Steve the Dreamer

**Curious Reweighting: DreamerV3의 경험 재학습과 보상 재가중을 통한 Crafter 성능 개선**

📢 2026년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.

## 소개

DreamerV3는 환경에서 얻은 경험으로 world model을 학습하고, 모델이 상상한 궤적으로 행동 정책을 개선하는 강화학습 알고리즘입니다. 이 프로젝트는 Crafter에서 **어떤 경험을 다시 학습할지**와 **어떤 achievement에 보상을 더 줄지**를 함께 조정해, 드물게 달성되는 행동의 학습을 개선합니다.

최종 방법인 **Curious Replay + Achievement Reweighting**은 실험 기록에서 Crafter Score **21.63%**를 기록했습니다. Baseline 25M의 **12.21%**보다 **9.42%p** 높으며, 팀 내 비교 실험 중 가장 높은 점수입니다.

이 저장소는 [DreamerV3](https://github.com/danijar/dreamerv3)를 기반으로 하며, DreamerV3 baseline, Curious Replay, Achievement Reweighting 및 두 방법의 조합을 실행할 수 있습니다. Survival Shaping, Latent Novelty, Replay Self-Imitation은 비교 실험 결과로 소개하며, 해당 실험의 별도 구현은 이 저장소에 포함되어 있지 않습니다.

## 방법론

### 1. Crafter와 평가 지표

[Crafter](https://github.com/danijar/crafter)는 자원 수집, 도구 제작, 전투 등 22개 achievement를 통해 에이전트의 다양한 능력을 평가하는 환경입니다. Crafter Score는 각 achievement 성공률의 기하평균에 기반합니다.

$$
\mathrm{Crafter\ Score}=\exp\left(\frac{1}{22}\sum_{i=1}^{22}\log(1+s_i)\right)-1
$$

여기서 $s_i$는 각 achievement의 에피소드별 성공률을 **0–100의 백분율**로 나타낸 값입니다. 이미 높은 성공률을 더 높이는 것보다, 성공률이 낮은 achievement를 개선하는 것이 점수 향상에 더 크게 기여할 수 있습니다.

### 2. Curious Replay: 경험 샘플링 변경

[Curious Replay](https://proceedings.mlr.press/v202/kauvar23a.html)의 아이디어를 참고하여, world model이 예측하기 어려운 경험과 최근 경험을 더 자주 학습하도록 replay sampler를 변경했습니다.

| 설정 | 이 저장소의 구현 |
| --- | --- |
| Priority 신호 | reconstruction, reward, continuation, KL 항을 포함하는 world-model loss |
| Sampler 혼합 비율 | priority 0.5 / recency 0.5 / uniform 0.0 |
| Recency | 최근 경험에 더 높은 확률 부여 (`recexp=1.0`) |
| 활성화 방법 | `--configs crafter size25m curious_replay` |

혼합 비율은 selector 설정값입니다. 기본 `replay.online=True`에서는 새 경험을 별도로 공급하는 경로도 사용합니다. 이 구현의 세부 설정과 결과를 원 논문의 실험과 동일한 것으로 간주하지 않습니다.

### 3. Achievement Reweighting: 학습 보상 변경

완료된 **100개 에피소드마다** 해당 구간의 achievement 성공률을 집계합니다. 성공률의 역수에 비례하는 가중치를 평균으로 정규화한 뒤, **0.5–2.0** 범위로 제한합니다.

$$
u_i=\frac{1}{\epsilon+p_i},\qquad
w_i=\operatorname{clip}\left(\frac{u_i}{\operatorname{mean}_j(u_j)},\ 0.5,\ 2.0\right)
$$

여기서 $p_i$는 **0–1 범위**의 성공률이며, 기본값은 $\epsilon=0.01$입니다. 가중치는 1로 시작하고 첫 100개 에피소드 이후부터 갱신됩니다. 각 에피소드에서 처음 달성한 achievement에만 다음 보상을 적용합니다.

$$
r_{\mathrm{train}}=r_{\mathrm{env}}+\alpha\sum_i(w_i-1)\,\mathbf{1}[\text{achievement }i\text{ first unlocked}]
$$

기본값은 $\alpha=1.0$입니다. 변경된 보상은 replay에 저장되어 reward model과 value 학습에 사용되고, actor는 상상 궤적에서 예측한 보상으로 학습합니다. 원래 환경 보상은 별도로 기록해 학습 보상과 평가를 구분합니다.

### 4. 두 방법의 결합

Curious Replay는 **경험을 선택하는 분포**를, Achievement Reweighting은 **경험에 저장되는 학습 보상**을 바꿉니다. 두 개입을 결합하면 드문 경험을 다시 학습하면서, 해당 achievement를 달성하는 행동에도 더 높은 보상을 줄 수 있습니다. 실제 결합 효과는 아래 비교 실험으로 확인했습니다.

## 실험 결과

아래 수치는 프로젝트의 [실험 기록](https://app.notion.com/p/3ce855b4e56580c7979df9da86c056dd)에 기재된 집계 결과입니다. `25M`과 `166M`은 원문 실험명을 그대로 사용했습니다. 결합 실험과 Replay SIL에는 원문에 없는 모델 크기나 step 수를 추가하지 않았습니다.

### 전체 실험 비교

| 실험 | Crafter Score ↑ | Reward | Score 집계 에피소드 | Reward 집계 에피소드 |
| --- | ---: | ---: | ---: | ---: |
| Baseline 166M | 18.41% | 8.98 | 4,705 | 4,706 |
| Baseline 25M | 12.21% | 7.57 | 4,945 | 4,950 |
| Survival Shaping 25M | 8.05% | 10.54 | 4,986 | 4,994 |
| Achievement Reweighting 25M | 12.92% | 7.75 | 4,622 | 4,626 |
| Latent Novelty 25M | 11.18% | 6.90 | 4,704 | 4,708 |
| Curious Replay 25M | 18.83% | 8.30 | 4,635 | 4,636 |
| **Curious Replay + Achievement Reweighting** | **21.63%** | **9.40** | **4,412** | **4,413** |
| Replay Self-Imitation (Replay SIL) | 11.10% | 6.94 | 미기재 | 미기재 |

결합 방법은 **이 프로젝트에서 비교한 실험 중 가장 높은 Crafter Score인 21.63%**를 기록했습니다. Baseline 25M의 12.21%보다 **9.42%p**, Curious Replay 단독의 18.83%보다 **2.80%p** 높았습니다. 이는 제공된 집계의 비교이며, 반복 실험을 통한 통계적 유의성은 확인하지 않았습니다.

Baseline 25M과 비교하면 Achievement Reweighting은 **+0.71%p**, Curious Replay는 **+6.62%p**를 기록했습니다. Survival Shaping, Latent Novelty, Replay SIL은 각각 **−4.16%p**, **−1.03%p**, **−1.11%p**였습니다. Survival Shaping은 기록된 Reward가 가장 높았지만 Crafter Score는 가장 낮았습니다. 보상값의 상승이 다양한 achievement의 달성으로 바로 이어지지는 않았습니다.

Reward와 Score의 집계 에피소드 수가 서로 달라 원문 값을 각각 표시했습니다. 자료에는 그 차이의 원인과 Reward에 포함된 보상 항의 범위가 명시되어 있지 않습니다.

### Achievement별 변화

결합 방법에서는 Baseline 25M 대비 `make_stone_sword`가 **0.12% → 26.59%**, `make_stone_pickaxe`가 **0.30% → 23.93%**, `collect_iron`이 **0.06% → 4.01%**로 높아졌습니다. 반면 `collect_diamond`, `make_iron_pickaxe`, `make_iron_sword`는 세 실험 모두 **0%**로 남았습니다. 철 수집의 개선이 철 도구 제작까지 이어지지는 않았습니다.

| 집계 기준 | Baseline 25M | Curious Replay 25M | Curious Replay + Reweighting |
| --- | ---: | ---: | ---: |
| 한 번 이상 달성한 achievement | 19 / 22 | 18 / 22 | 19 / 22 |
| 성공률 1% 이상 | 15 / 22 | 18 / 22 | 18 / 22 |
| 성공률 10% 이상 | 13 / 22 | 16 / 22 | 16 / 22 |
| 성공률 1% 미만 | 7 / 22 | 4 / 22 | 4 / 22 |
| 22개 성공률의 산술평균 | 38.62% | 41.85% | 46.86% |

결합 방법은 한 번이라도 달성한 achievement의 총수를 늘리기보다, 기존에 드물게 달성하던 일부 achievement의 성공률을 높였습니다. 산술평균은 아래에 표시된 성공률로 계산한 값이며 Crafter Score와는 다른 지표입니다.

<details>
<summary>22개 achievement 성공률과 달성 에피소드 수 전체 비교</summary>

각 셀은 **성공률 (달성 에피소드 / 집계 에피소드)**입니다.

| Achievement | Baseline 25M | Curious Replay 25M | Curious Replay + Reweighting |
| --- | ---: | ---: | ---: |
| `collect_sapling` | 96.78% (4,786 / 4,945) | 88.82% (4,117 / 4,635) | 95.69% (4,222 / 4,412) |
| `collect_wood` | 95.73% (4,734 / 4,945) | 98.47% (4,564 / 4,635) | 98.37% (4,340 / 4,412) |
| `wake_up` | 94.72% (4,684 / 4,945) | 92.62% (4,293 / 4,635) | 92.68% (4,089 / 4,412) |
| `place_plant` | 94.32% (4,664 / 4,945) | 86.08% (3,990 / 4,635) | 94.90% (4,187 / 4,412) |
| `place_table` | 85.80% (4,243 / 4,945) | 93.16% (4,318 / 4,635) | 93.88% (4,142 / 4,412) |
| `collect_drink` | 80.04% (3,958 / 4,945) | 83.97% (3,892 / 4,635) | 80.19% (3,538 / 4,412) |
| `make_wood_sword` | 62.99% (3,115 / 4,945) | 37.65% (1,745 / 4,635) | 80.01% (3,530 / 4,412) |
| `make_wood_pickaxe` | 60.91% (3,012 / 4,945) | 63.02% (2,921 / 4,635) | 67.66% (2,985 / 4,412) |
| `collect_stone` | 47.68% (2,358 / 4,945) | 50.85% (2,357 / 4,635) | 60.15% (2,654 / 4,412) |
| `defeat_zombie` | 46.65% (2,307 / 4,945) | 63.06% (2,923 / 4,635) | 60.81% (2,683 / 4,412) |
| `place_stone` | 38.69% (1,913 / 4,945) | 42.09% (1,951 / 4,635) | 48.80% (2,153 / 4,412) |
| `collect_coal` | 16.99% (840 / 4,945) | 22.57% (1,046 / 4,635) | 30.76% (1,357 / 4,412) |
| `place_furnace` | 15.25% (754 / 4,945) | 16.35% (758 / 4,635) | 44.58% (1,967 / 4,412) |
| `eat_cow` | 9.08% (449 / 4,945) | 25.22% (1,169 / 4,635) | 20.67% (912 / 4,412) |
| `defeat_skeleton` | 3.42% (169 / 4,945) | 4.57% (212 / 4,635) | 7.25% (320 / 4,412) |
| `make_stone_pickaxe` | 0.30% (15 / 4,945) | 22.01% (1,020 / 4,635) | 23.93% (1,056 / 4,412) |
| `make_stone_sword` | 0.12% (6 / 4,945) | 26.77% (1,241 / 4,635) | 26.59% (1,173 / 4,412) |
| `collect_iron` | 0.06% (3 / 4,945) | 3.43% (159 / 4,635) | 4.01% (177 / 4,412) |
| `eat_plant` | 0.02% (1 / 4,945) | 0.00% (0 / 4,635) | 0.02% (1 / 4,412) |
| `collect_diamond` | 0.00% (0 / 4,945) | 0.00% (0 / 4,635) | 0.00% (0 / 4,412) |
| `make_iron_pickaxe` | 0.00% (0 / 4,945) | 0.00% (0 / 4,635) | 0.00% (0 / 4,412) |
| `make_iron_sword` | 0.00% (0 / 4,945) | 0.00% (0 / 4,635) | 0.00% (0 / 4,412) |

</details>

## 환경 설정

저장소를 복제한 뒤 프로젝트 루트에서 실행합니다. 아래 설치 예시는 **Linux / NVIDIA GPU / CUDA 12 / Python 3.11** 환경을 기준으로 합니다. `requirements.txt`에 GPU용 JAX가 고정되어 있으므로 GPU 드라이버와의 호환성을 확인해야 합니다.

```sh
git clone https://github.com/maejinee/curious_reweighting.git
cd curious_reweighting

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install crafter ruamel.yaml PyYAML matplotlib
```

마지막 명령은 Crafter 환경, 설정 파일 로딩 및 모니터링에 필요한 추가 패키지를 설치합니다. CPU/macOS 환경에는 위 GPU 의존성을 그대로 설치하지 말고 별도 JAX 환경을 구성해야 합니다.

## 사용 방법

### 학습

아래는 현재 코드의 `size25m` 프로필로 네 가지 설정을 실행하는 예시입니다. **`25M`은 모델 크기 프로필이며 학습 step 수가 아닙니다.** `crafter` 기본 설정은 **1,100,000 environment steps**, 환경 1개, `train_ratio=512`입니다. 이는 현재 코드의 기본값이며, Notion에 기록된 모든 실험의 실행 조건을 복원한 명령은 아닙니다.

```sh
# 1. Baseline
python dreamerv3/main.py \
  --configs crafter size25m \
  --logdir ./logdir/baseline_seed0 \
  --seed 0

# 2. Curious Replay
python dreamerv3/main.py \
  --configs crafter size25m curious_replay \
  --logdir ./logdir/curious_replay_seed0 \
  --seed 0

# 3. Achievement Reweighting
python dreamerv3/main.py \
  --configs crafter size25m \
  --env.crafter.achievement_reweight True \
  --logdir ./logdir/achievement_reweighting_seed0 \
  --seed 0

# 4. Curious Replay + Achievement Reweighting
python dreamerv3/main.py \
  --configs crafter size25m curious_replay \
  --env.crafter.achievement_reweight True \
  --logdir ./logdir/curious_reweighting_seed0 \
  --seed 0
```

각 설정은 별도의 `logdir`에서 실행합니다. 중단한 실행을 이어갈 때에는 같은 설정과 `logdir`를 사용합니다. 전체 옵션은 [`dreamerv3/configs.yaml`](dreamerv3/configs.yaml)에 있으며, 예를 들어 `--run.steps 1100000`으로 학습 길이를 지정할 수 있습니다.

### 학습 결과 확인

```sh
# 상태 요약과 그래프를 한 번 갱신
python monitor_training_status.py \
  --logdir ./logdir/curious_reweighting_seed0 --once

# Scope 대시보드
python -m scope.viewer --basedir ./logdir --port 8000
```

모니터에서 `--once`를 생략하면 30초마다 갱신합니다. 학습 로그가 쌓이면 실행 디렉터리에서 다음 파일을 확인할 수 있습니다.

| 파일 | 내용 |
| --- | --- |
| `config.yaml` | 실제 실행에 사용한 설정 |
| `metrics.jsonl`, `scores.jsonl` | 학습 지표와 에피소드 점수 |
| `env0/stats.jsonl` | 원래 보상, 학습 보상, achievement 및 가중치 |
| `training_status.md` | 학습 상태와 Crafter 평가 요약 |
| `crafter_benchmark.png` | Reward / Crafter Score 추이 |
| `achievement_reweighting.png` | Achievement 가중치와 성공률 |

`episode/score`는 학습에 사용한 보상의 합계입니다. 보상 재가중을 사용한 실행의 원래 환경 보상은 `env0/stats.jsonl`의 `reward` 등 별도 기록을 기준으로 확인합니다. 모니터의 Crafter Score는 초기에는 누적 구간, 이후에는 최근 1,000,000 environment steps 구간을 집계합니다.

> 모니터의 baseline 비교 경로는 [`monitor_training_status.py`](monitor_training_status.py)의 `BASELINE_LOGDIR`에 고정되어 있습니다. 자신의 baseline과 비교하려면 이 값을 바꿔야 하며, `--logdir`는 비교 대상 baseline 경로를 변경하지 않습니다.

### 코드 구조

```text
.
├── dreamerv3/
│   ├── agent.py                 # World model, actor-critic, replay priority
│   ├── configs.yaml             # 모델 및 실험 설정
│   └── main.py                  # 학습 진입점, replay 구성
├── embodied/
│   ├── core/replay.py           # Replay buffer
│   ├── core/selectors.py        # Uniform / priority / recency sampler
│   ├── envs/crafter.py          # Achievement reweighting과 원래 보상 기록
│   └── run/                    # 학습·평가 루프
├── monitor_training_status.py  # 상태 요약과 실험 그래프 생성
├── docs/DREAMERV3.md            # 원본 DreamerV3 README
├── requirements.txt
└── LICENSE
```

`scores/`와 `baselines.yaml`은 원본 DreamerV3 코드에서 제공하는 참고 자료입니다. 위 프로젝트 실험의 원시 로그나 checkpoint는 이 저장소에 포함되어 있지 않습니다.

## 한계 및 향후 과제

- 최종 조합에서도 `make_iron_pickaxe`, `make_iron_sword`, `collect_diamond`의 성공률은 0%였습니다. 철 수집 이후 제작 단계로 진행하지 못하는 원인을 확인할 필요가 있습니다.
- 실험별 전체 설정, 평가 구간 및 여러 seed의 분산이 함께 제공되지 않아, 기록된 차이를 통계적으로 확정된 효과나 외부 벤치마크 최고 성능으로 해석하기는 어렵습니다.
- Survival Shaping은 생존 자원 반복 수집으로 치우친 행동이 관찰되었습니다. 보상 상한 또는 생존 임계값을 이용해 이 유인을 줄이는 후속 실험이 필요합니다.
- Replay Self-Imitation은 실제 성공 경험을 강화하는 항의 가중치와 warmup을 조정하고, 상상 기반 학습과의 간섭 여부를 분석할 필요가 있습니다.

## 팀원

Notion 실험 기록의 이름과 담당 실험을 기준으로 정리했습니다.

| 팀원 | 담당 실험 |
| --- | --- |
| 혜진 | Survival Shaping |
| 성진 | Achievement Reweighting |
| 주은 | Latent Novelty |
| 정훈 | Curious Replay |
| 찬 | Replay Self-Imitation |

## 참고자료 및 라이선스

- [프로젝트 실험 기록: 스티브 실험 결과](https://app.notion.com/p/3ce855b4e56580c7979df9da86c056dd) — 본 README의 실험 수치와 역할 분담 출처. 페이지 접근 권한이 필요할 수 있습니다.
- [DreamerV3 논문](https://arxiv.org/abs/2301.04104) / [공식 코드](https://github.com/danijar/dreamerv3)
- [Curious Replay for Model-based Adaptation, ICML 2023](https://proceedings.mlr.press/v202/kauvar23a.html) / [공식 코드](https://github.com/AutonomousAgentsLab/curiousreplay)
- [Crafter benchmark](https://github.com/danijar/crafter)

DreamerV3 기반 코드의 MIT 라이선스 및 원 저작권 표시는 [`LICENSE`](LICENSE)에 보존되어 있습니다. 원본 실행 안내와 DreamerV3 인용 정보는 [`docs/DREAMERV3.md`](docs/DREAMERV3.md)를 참고하세요.
