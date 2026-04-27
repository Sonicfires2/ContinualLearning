# Spaced Replay Scheduler

## Purpose

Task 12 implements the first operational spaced replay scheduler. The scheduler
uses online sample signals to estimate a due-time proxy for each memory item and
then replay samples that are due or nearest due.

This is not a final proof of cognitive spacing. It is the first controlled
implementation of the proposal's mechanism:

```text
observe sample signals -> estimate replay interval -> schedule replay near due step
```

## Implementation

Scheduler:

- [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)

Experiment entry point:

- [src/baselines/spaced_replay.py](../src/baselines/spaced_replay.py)

Configs:

- [configs/experiments/spaced_replay_smoke.yaml](../configs/experiments/spaced_replay_smoke.yaml)
- [configs/experiments/spaced_replay_split_cifar100.yaml](../configs/experiments/spaced_replay_split_cifar100.yaml)

The scheduler maintains one state per observed `sample_id`:

- last observed loss
- last observed uncertainty
- last target probability
- replay count
- risk score
- estimated forgetting interval in optimizer steps
- next scheduled replay step

Risk is computed only from online information available at the current step:

```text
loss, uncertainty, 1 - target_probability, loss increase from previous observation
```

The current due-time proxy maps high risk to a shorter interval and low risk to
a longer interval:

```text
estimated_T_i = max_interval - risk_score * (max_interval - min_interval)
next_due_step = current_step + estimated_T_i
```

## Selection Rule

The scheduler selects:

1. due samples first, sorted by how overdue they are and then by risk;
2. if `budget_mode = match_random_replay`, fills remaining replay slots with
   samples nearest their due step.

The default full-run mode is:

```text
budget_mode = match_random_replay
```

This keeps the replay volume comparable to the random replay baseline. It means
the first scheduler tests selection and ordering under a matched replay budget,
not replay-frequency reduction.

## Leakage Guard

The scheduler does not read:

- `forgetting_labels.json`
- `time_to_forgetting_targets.json`
- future evaluation rows
- predictor reports

It updates state only from current training/replay logits observed during the
online run. Offline labels and `T_i` targets are for evaluation and later model
development only.

## Scheduler Trace

Every replay selection is logged in:

```text
scheduler_trace.json
```

Trace rows include:

- `global_step`
- `sample_id`
- `risk_score`
- `estimated_forgetting_time_steps`
- `next_scheduled_replay_step`
- `selection_reason`
- `overdue_steps`
- replay count before selection

This trace is required for later ablations: risk-only, timing-only,
due-only, and budget-matched variants.

## Full Split CIFAR-100 Single-Seed Result

Run:

```text
experiments/runs/spaced_replay/spaced_replay_split_cifar100_seed0
```

Config:

```text
min_replay_interval_steps = 1
max_replay_interval_steps = 64
scheduler_budget_mode = match_random_replay
replay_batch_size = 32
replay_capacity = 2000
```

Result:

| Metric | Value |
| --- | ---: |
| Final accuracy | `0.0962` |
| Average forgetting | `0.315` |
| Training time seconds | `40.18328460000339` |
| Replay-augmented batches | `1413` |
| Total replay samples | `45216` |
| Scheduler trace rows | `45216` |
| Mean estimated forgetting time | `22.7363` |
| Mean risk score | `0.6549832268742224` |

Single-seed comparison against existing pilot runs:

| Method | Replay samples | Final accuracy | Average forgetting |
| --- | ---: | ---: | ---: |
| fine-tuning | `0` | `0.0455` | `0.434` |
| fixed-periodic replay, interval ablation | `22624` | `0.0657` | `0.4083333333333333` |
| random replay | `45216` | `0.10129999999999999` | `0.30433333333333334` |
| spaced replay, due-time proxy | `45216` | `0.0962` | `0.315` |

Interpretation:

- The scheduler runs end-to-end and produces the required trace artifact.
- Replay volume is matched to the existing random replay pilot.
- On this single seed, the first spaced replay proxy is slightly worse than
  random replay.
- This is not a final result because repeated seeds and final budget controls
  belong to Task 13.

## Current Scientific Judgment

The first scheduler is useful but not yet strong. It proves the infrastructure
for spacing-aware replay exists, but it does not yet prove the proposal's main
claim. The likely reason is consistent with Task 11: crude risk-to-interval
mapping is not yet a good `T_i` estimator.

Task 13 has now run the first fair three-seed comparison, documented in
[CORE_COMPARISON_TASK13.md](./CORE_COMPARISON_TASK13.md). That comparison found
that this first spaced replay proxy is slightly worse than random replay under
matched replay budget. The next methodological improvement should test whether a
better `T_i` estimator or a due-only budget mode improves over this first proxy.

## Verification

Focused tests:

```text
.\.venv\Scripts\python.exe -m pytest tests\replay\test_spaced_scheduler.py tests\baselines\test_spaced_replay_baseline.py -q
```

Full test suite:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
46 passed
```
