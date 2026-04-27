# Fixed-Periodic Replay Baseline

## Purpose

Task 10 adds the proposal's fixed-schedule comparator: replay uniformly from
memory every `k` optimizer steps. This method is intentionally not
forgetting-aware. It does not estimate `T_i`, does not use risk scores, and does
not implement cognitive spacing. Its job is to answer a simpler control
question: how much of replay's benefit comes from replaying on a fixed cadence?

## Implementation

Entry point:

- [src/baselines/fixed_periodic_replay.py](../src/baselines/fixed_periodic_replay.py)

Configs:

- [configs/experiments/fixed_periodic_replay_smoke.yaml](../configs/experiments/fixed_periodic_replay_smoke.yaml)
- [configs/experiments/fixed_periodic_replay_split_cifar100.yaml](../configs/experiments/fixed_periodic_replay_split_cifar100.yaml)

The baseline uses the same reservoir buffer and task-end insertion policy as the
random replay baseline. Replay is triggered when:

```text
(global_step + 1) % replay_interval == 0
```

This treats optimizer steps as one-indexed for the cadence rule. The stored
`global_step` remains zero-indexed in signal logs and artifacts.

## Artifact Fields

The replay metrics include:

- `schedule`
- `budget_mode`
- `replay_interval`
- `replay_batch_size`
- `current_batch_count`
- `replay_augmented_batch_count`
- `replay_event_steps`
- `replay_event_optimizer_steps`
- `skipped_replay_steps_due_to_interval`
- `skipped_replay_steps_due_to_empty_buffer`
- `effective_replay_ratio`
- `total_replay_samples`

The current implementation uses:

```text
budget_mode = interval_ablation
```

That means the method obeys the fixed interval literally. It is not yet
compute-matched to random replay. The later core comparison must either choose a
budget-matched interval or explicitly report this as a frequency ablation.

## Full Split CIFAR-100 Single-Seed Result

Run:

```text
experiments/runs/fixed_periodic_replay/fixed_periodic_replay_split_cifar100_seed0
```

Config:

```text
replay_interval = 2
replay_batch_size = 32
replay_capacity = 2000
budget_mode = interval_ablation
```

Result:

| Metric | Value |
| --- | ---: |
| Final accuracy | `0.0657` |
| Average forgetting | `0.4083333333333333` |
| Training time seconds | `19.60763469999074` |
| Current batches | `1570` |
| Replay-augmented batches | `707` |
| Total replay samples | `22624` |
| Effective replay ratio | `0.45031847133757963` |
| Signal rows | `127624` |

Interpretation:

- The baseline runs and logs the fixed cadence correctly.
- It performs better than no-replay fine-tuning in final accuracy, but worse
  than the current random replay run.
- This is not yet a fair final comparison against random replay because the
  fixed-periodic run used fewer replay samples than random replay.
- The result is a useful Task 10 artifact, not a final claim about fixed replay.

## Verification

Focused checks:

```text
.\.venv\Scripts\python.exe -m pytest tests\baselines\test_fixed_periodic_replay_baseline.py -q
```

Result:

```text
3 passed
```

Full test suite:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
35 passed
```

The tests verify that replay only occurs on due optimizer steps and that replay
samples come from earlier tasks, not future tasks.
