# Task 19 Learned Fixed-Budget Replay

## Purpose

Task 19 separates two ideas that were mixed together in Task 18:

```text
1. use a learned predictor to rank risky examples
2. skip replay when examples do not look risky enough
```

Task 18 failed as a sparse replay gate, but that did not tell us whether the
learned predictor was bad at choosing examples or whether skipping replay was
too aggressive.

Task 19 removes the skip decision. It asks:

```text
If we replay the same number of examples as random replay, does learned risk
choose better examples?
```

## Implementation

Main code:

- [src/baselines/learned_fixed_budget_replay.py](../src/baselines/learned_fixed_budget_replay.py)
- [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
- [configs/experiments/learned_fixed_budget_replay_smoke.yaml](../configs/experiments/learned_fixed_budget_replay_smoke.yaml)
- [configs/experiments/learned_fixed_budget_replay_split_cifar100.yaml](../configs/experiments/learned_fixed_budget_replay_split_cifar100.yaml)
- [tests/baselines/test_learned_fixed_budget_replay.py](../tests/baselines/test_learned_fixed_budget_replay.py)
- [tests/replay/test_spaced_scheduler.py](../tests/replay/test_spaced_scheduler.py)

The scheduler now has a `risk_ranked` budget mode. In this mode it:

- scores memory examples with the learned logistic forgetting-risk predictor;
- sorts examples from highest predicted risk to lowest predicted risk;
- fills the full replay batch whenever the buffer has enough examples;
- does not skip replay because risk is low;
- logs `risk_ranked` as the selection reason.

The Split CIFAR-100 config is matched to the random replay seed-0 config:

- same task split;
- same model size;
- same learning rate;
- same current batch size;
- same replay buffer capacity;
- same replay batch size;
- same total replay sample budget.

## Leakage Guard

The active Task 19 run does not train its predictor from its own future labels.
It uses the same prior random-replay seed-0 source artifacts as Task 18:

```text
experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals
```

The run metadata records the source signal and label files and their hashes, so
future experiments can swap in cross-seed or stricter prior-run predictors
without changing the replay-selection contract.

## Smoke Result

Artifact:

```text
.tmp/baseline_runs/learned_fixed_budget_replay/learned_fixed_budget_replay/learned_fixed_budget_replay_smoke
```

Result:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.16666666666666666` |
| average forgetting | `0.5` |
| replay samples | `24` |
| replay-augmented batches | `6` |
| skipped replay opportunities | `0` |
| scheduler trace rows | `24` |

This verifies the intended mechanism: the learned-risk selector fills the replay
budget and does not skip low-risk replay.

## Split CIFAR-100 Pilot

Artifact:

```text
experiments/runs/learned_fixed_budget_replay/learned_fixed_budget_replay/learned_fixed_budget_replay_split_cifar100_seed0_prior_random
```

Result:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.0759` |
| average forgetting | `0.3587777777777778` |
| total replay samples | `45216` |
| replay-augmented batches | `1413` |
| replay batch size | `32` |
| skipped replay opportunities | `0` |
| scheduler trace rows | `45216` |
| unique replayed samples | `1761` |
| never replayed buffer samples | `239` |
| mean replay count | `11.141` |
| training time seconds | `43.47675380000146` |

The replay budget matches random replay exactly:

```text
1413 replay batches * 32 examples = 45216 replay samples
```

## Comparison Context

Seed-0 references:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |
| cheap risk-gated replay, threshold `0.75` | `0.046` | `0.43377777777777776` | `2071` |
| learned risk-gated replay, threshold `0.90` | `0.0379` | `0.40311111111111114` | `14425` |

Three-seed references:

| Method | Final accuracy mean | Avg forgetting mean |
| --- | ---: | ---: |
| fine-tuning | `0.0470` | `0.4310` |
| random replay | `0.1016` | `0.3027` |
| spaced replay proxy | `0.0986` | `0.3131` |
| MIR replay | `0.11636666666666667` | `0.2167037037037037` |

## Scientific Interpretation

Task 19 is a negative but useful intervention result.

The learned predictor is better than the cheap heuristic in offline predictor
reports, but using it to rank replay examples under the same replay budget does
not beat random replay. It also remains far behind MIR.

In plain language:

```text
The predictor can often identify examples that look risky, but replaying only
the riskiest examples is not the same as preserving a broad memory of old tasks.
```

This means the Task 18 failure was not caused only by skipping too many replay
examples. Even when replay volume is restored to the random-replay budget, pure
learned-risk ranking is still worse than random selection under this seed-0
test.

The most likely issue is loss of diversity. Risk-only ranking may over-focus on
hard, unstable, or noisy examples, while random replay naturally spreads replay
across many classes and tasks.

## Next Research Move

Task 20 has now implemented a balanced hybrid selector:

```text
some learned-risk examples + some random or class-balanced examples
```

That test answered whether learned risk has value once diversity is protected.
The 50/50 learned-risk plus class-balanced hybrid improved over pure learned
fixed-budget replay, but still did not beat random replay. See
[LEARNED_HYBRID_REPLAY_TASK20.md](./LEARNED_HYBRID_REPLAY_TASK20.md).

MIR remains the strongest implemented replay method. It likely works better
because it scores examples by expected interference from the current update,
which is closer to the actual replay decision than offline future-forgetting
probability.

## Verification

Focused verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\replay\test_spaced_scheduler.py tests\baselines\test_learned_fixed_budget_replay.py tests\predictors\test_online_forgetting.py -q
```

Result:

```text
7 passed
```

Full repository verification after Task 19:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
76 passed
```

Experiment commands:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.learned_fixed_budget_replay --config configs\experiments\learned_fixed_budget_replay_smoke.yaml
.\.venv\Scripts\python.exe -m src.baselines.learned_fixed_budget_replay --config configs\experiments\learned_fixed_budget_replay_split_cifar100.yaml
```
