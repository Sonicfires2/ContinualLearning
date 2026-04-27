# Task 18 Learned-Predictor Risk-Gated Replay

## Purpose

Task 18 tests whether the stronger offline predictor from Tasks 16 and 17 can
improve the replay intervention from Task 15.

The question is:

```text
If a logistic predictor can rank likely-to-be-forgotten examples offline, does
using that predictor as the online replay gate improve continual-learning
retention?
```

This is an intervention test, not just another predictor report.

## Implementation

Main code:

- [src/predictors/online_forgetting.py](../src/predictors/online_forgetting.py)
- [src/baselines/learned_risk_gated_replay.py](../src/baselines/learned_risk_gated_replay.py)
- [configs/experiments/learned_risk_gated_replay_smoke.yaml](../configs/experiments/learned_risk_gated_replay_smoke.yaml)
- [configs/experiments/learned_risk_gated_replay_split_cifar100.yaml](../configs/experiments/learned_risk_gated_replay_split_cifar100.yaml)
- [tests/predictors/test_online_forgetting.py](../tests/predictors/test_online_forgetting.py)
- [tests/baselines/test_learned_risk_gated_replay_baseline.py](../tests/baselines/test_learned_risk_gated_replay_baseline.py)

The learned gate uses logistic regression over the Task 17 `all_features`
feature group. The online scheduler computes the same feature columns from
current and past observations, then uses the predictor's positive-class
probability as the replay risk score.

## Leakage Guard

The active replay run does not read its own future labels while making replay
decisions.

For the Split CIFAR-100 pilot, the learned predictor is trained before the run
from this prior artifact:

```text
experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals
```

The saved run config records SHA-256 hashes for both source artifacts:

- `sample_signals.json`
- `forgetting_labels.json`

This makes the learned gate auditable and future-proof. A better later version
can swap in different prior-run artifacts or a stricter cross-seed training
rule without changing the scheduler contract.

## Smoke Result

Artifact:

```text
.tmp/baseline_runs/learned_risk_gated_replay/learned_risk_gated_replay/learned_risk_gated_replay_smoke
```

The smoke fixture uses a very high threshold because the toy predictor is
nearly perfectly separable.

```text
risk_threshold = 0.9998
scheduler_budget_mode = risk_only
```

Result:

| Metric | Value |
| --- | ---: |
| replay samples | `14` |
| replay-augmented batches | `5` |
| skipped replay opportunities | `1` |
| effective replay ratio | `0.5555555555555556` |
| risk score source | `learned` |

This verifies that the learned gate can both select replay and skip replay.

## Split CIFAR-100 Pilots

These are seed-0 diagnostics, not repeated-seed claims.

| Gate | Threshold | Final accuracy | Avg forgetting | Replay samples | Replay batches | Skipped events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned `risk_only` | `0.70` | `0.0376` | `0.4286666666666667` | `21423` | `672` | `39` |
| learned `risk_only` | `0.90` | `0.0379` | `0.40311111111111114` | `14425` | `466` | `245` |

Primary artifact for the stricter threshold:

```text
experiments/runs/learned_risk_gated_replay/learned_risk_gated_replay/learned_risk_gated_replay_split_cifar100_seed0_prior_random_threshold090
```

## Comparison Context

Seed-0 references:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |
| cheap risk-gated replay, threshold `0.75` | `0.046` | `0.43377777777777776` | `2071` |
| learned risk-gated replay, threshold `0.90` | `0.0379` | `0.40311111111111114` | `14425` |

Three-seed references:

| Method | Final accuracy mean | Avg forgetting mean |
| --- | ---: | ---: |
| random replay | `0.1016` | `0.3027` |
| spaced replay proxy | `0.0986` | `0.3131` |
| MIR replay | `0.11636666666666667` | `0.2167037037037037` |

## Scientific Interpretation

Task 18 is a negative intervention result.

The learned predictor improved offline ranking in Tasks 16 and 17, but using it
as the online replay gate did not improve retention. It performed worse than
random replay and far worse than MIR. It also did not rescue the sparse replay
idea from Task 15.

The most likely explanation is an online-distribution mismatch:

- offline predictor rows come from evaluation anchors;
- online gate rows come from current training and replay batches;
- the predictor often assigns high risk to many online observations;
- selecting only those high-risk examples may over-focus on hard or unstable
  samples instead of preserving a balanced memory of old tasks.

So the project should not claim:

```text
learned risk-gated replay improves retention
```

It can claim:

```text
offline forgetting prediction improves, but directly turning that predictor
into an online replay gate failed under the tested setup
```

## Next Research Move

The next useful step is not just raising or lowering thresholds. Task 18 mixed
two questions:

```text
1. Is learned risk useful for ranking examples?
2. Is learned risk safe for skipping replay?
```

The failure shows that this sparse gate is not enough. It does not yet show
that learned risk is useless for replay selection.

Good next options:

1. Compare a learned-risk ranking selector under a fixed replay budget before
   using it as a sparse skip gate.
2. Combine learned risk with diversity or class/task balancing.
3. Train the gate on online-style rows, not only evaluation-anchor rows.
4. Calibrate probabilities on a held-out prior run before choosing thresholds.

This plan is now recorded in
[FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md).
Task 19 has now tested fixed-budget learned-risk replay; see
[LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](./LEARNED_FIXED_BUDGET_REPLAY_TASK19.md).
That result is also negative under seed 0, so the next implementation should
test a balanced hybrid before adding expensive signals.

MIR remains the strongest implemented method because it asks a different and
more directly intervention-aligned question: which memory examples would be
hurt most by the current update?

## Verification

Focused verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\predictors\test_online_forgetting.py tests\replay\test_spaced_scheduler.py tests\baselines\test_learned_risk_gated_replay_baseline.py -q
```

Result:

```text
6 passed
```

Full repository verification after Task 18:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
73 passed
```
