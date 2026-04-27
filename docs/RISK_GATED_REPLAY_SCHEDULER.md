# Task 15 Event-Triggered Risk-Gated Replay

## Purpose

Task 15 adds the replay mechanism suggested after the first spaced-replay
result:

```text
Only replay when the model is predicted to be near forgetting; otherwise skip replay.
```

The goal is not only higher retention. It is a stricter retention/compute test:
can the project avoid unnecessary replay while preserving old-task accuracy?

## Implementation

Scheduler support:

- [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)

Baseline entry point:

- [src/baselines/risk_gated_replay.py](../src/baselines/risk_gated_replay.py)

Configs:

- [configs/experiments/risk_gated_replay_smoke.yaml](../configs/experiments/risk_gated_replay_smoke.yaml)
- [configs/experiments/risk_gated_replay_split_cifar100.yaml](../configs/experiments/risk_gated_replay_split_cifar100.yaml)

Tests:

- [tests/replay/test_spaced_scheduler.py](../tests/replay/test_spaced_scheduler.py)
- [tests/baselines/test_risk_gated_replay_baseline.py](../tests/baselines/test_risk_gated_replay_baseline.py)

The scheduler now supports these selection modes:

| Mode | Behavior |
| --- | --- |
| `match_random_replay` | Original budget-matched spaced replay: select due samples first, then fill by nearest due time. |
| `due_only` | Replay only samples whose due step has arrived. |
| `risk_only` | Replay only samples with `risk_score >= risk_threshold`. |
| `risk_and_due` | Replay only samples that are both high-risk and due. |
| `risk_or_due` | Replay samples that are high-risk or due. |

The Task 15 baseline uses:

```text
scheduler_budget_mode = risk_and_due
risk_threshold = 0.75
min_replay_interval_steps = 1
max_replay_interval_steps = 64
```

The risk score is the same online cheap-signal score used by the first spaced
scheduler:

```text
loss, uncertainty, 1 - target_probability, loss increase from previous observation
```

No forgetting labels, future evaluations, or offline reports are used during
online replay selection.

## Trace Artifact

The scheduler trace is still saved as:

```text
scheduler_trace.json
```

It now contains both:

- `rows`: replay selections;
- `skipped_rows`: replay opportunities skipped because no item passed the
  configured gate.

The scheduler summary records:

- `trace_row_count`
- `skipped_selection_event_count`
- `selection_reason_counts`
- `skip_reason_counts`
- `risk_threshold`
- `budget_mode`
- mean risk score
- mean estimated forgetting time

This keeps later ablations compatible with the earlier spaced-replay trace while
adding the efficiency evidence needed for event-triggered replay.

## Smoke Calibration

The first smoke run with `risk_threshold = 0.6` skipped all replay on the small
fixture, so the smoke config was lowered to `0.5` to verify that the mechanism
can both select and skip replay.

Smoke result:

| Metric | Value |
| --- | ---: |
| Final accuracy | `0.5` |
| Average forgetting | `0.0` |
| Replay samples | `7` |
| Replay-augmented batches | `2` |
| Effective replay ratio | `0.2222222222222222` |
| Skipped selection events | `4` |
| Scheduler trace rows | `7` |

Smoke artifact:

```text
.tmp/baseline_runs/risk_gated_replay/risk_gated_replay_smoke
```

## Split CIFAR-100 One-Seed Pilot

Several one-seed thresholds were tested because the event-triggered mechanism is
sensitive to threshold calibration. These are diagnostic pilot runs, not final
three-seed evidence.

| Run | Gate | Threshold | Final accuracy | Avg forgetting | Replay samples | Replay batches | Skipped events | Effective replay ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration loose gate | `risk_and_due` | `0.50` | `0.0569` | `0.3831111111111111` | `45216` | `1413` | `0` | `0.9` |
| middle gate | `risk_and_due` | `0.70` | `0.0438` | `0.4357777777777778` | `11219` | `1406` | `7` | `0.8955414012738854` |
| primary event-triggered pilot | `risk_and_due` | `0.75` | `0.046` | `0.43377777777777776` | `2071` | `735` | `678` | `0.4681528662420382` |
| risk-only diagnostic | `risk_only` | `0.75` | `0.0454` | `0.43` | `1871` | `342` | `1071` | `0.2178343949044586` |

Primary artifact:

```text
experiments/runs/risk_gated_replay/risk_gated_replay_split_cifar100_threshold075_seed0
```

## Comparison Context

Single-seed context from earlier runs:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| fine-tuning, seed 0 | `0.0455` | `0.434` | `0` |
| random replay, seed 0 | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| spaced replay due-time proxy, seed 0 | `0.0962` | `0.315` | `45216` |
| MIR replay, seed 0 | `0.1183` | `0.21400000000000002` | `45216` |
| risk-gated replay, primary pilot | `0.046` | `0.43377777777777776` | `2071` |

The primary risk-gated pilot used about `95.4%` fewer replay samples than random
replay, but it did not preserve retention. It behaved much closer to
fine-tuning than to random replay.

## Scientific Interpretation

Task 15 is complete as an implementation task: the repository can now run and
audit event-triggered replay that skips low-risk replay opportunities.

The first scientific result is negative:

- the mechanism can greatly reduce replay volume;
- the current cheap online risk gate is not sufficient to preserve retention
  when replay is sparse;
- loosening the threshold enough to preserve more replay removes the compute
  savings and still does not beat random replay;
- the due-time gate appears too brittle with the current heuristic `T_i` proxy.

This does not disprove event-triggered replay. It says the current online
heuristic is not strong enough as the sole replay trigger. Task 16 has now
tested the proposal's learned predictors offline; the next step is to ablate
signals and threshold behavior before using a learned model as an online replay
gate.

## Verification

Focused tests:

```text
.\.venv\Scripts\python.exe -m pytest tests\replay\test_spaced_scheduler.py tests\baselines\test_risk_gated_replay_baseline.py -q
```

Result:

```text
5 passed
```

Full repository verification:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
62 passed
```
