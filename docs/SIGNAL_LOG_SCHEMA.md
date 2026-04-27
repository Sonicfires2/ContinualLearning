# Sample Signal Log Schema

## Purpose

Task 7 adds an append-only sample-level signal artifact named `sample_signals.json`.
Its job is to preserve the raw observations needed to test whether cheap
sample-level signals predict later forgetting.

These logs are not forgetting labels and are not scheduler decisions. They are
the evidence source for the next phase.

## Artifact Location

Every run with signal logging enabled writes:

```text
<run_dir>/sample_signals.json
```

The artifact is included in `manifest.json` with a SHA-256 hash, so later
analysis can detect accidental edits.

## Payload Shape

```json
{
  "schema_version": 1,
  "fields": ["sample_id", "..."],
  "summary": {},
  "rows": []
}
```

## Row Semantics

Each row is one model observation for one sample at one point in training or
evaluation.

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable sample identifier from the task stream. This is the primary join key. |
| `source_task_id` | Task that owns the sample according to dataset metadata. |
| `original_class_id` | CIFAR-100 fine class ID used for class-incremental targets. |
| `within_task_label` | Task-local label for the sample. |
| `original_index` | Index in the original CIFAR-100 train or test split. |
| `split` | Dataset split, usually `train` or `test`. |
| `target` | Target used for the observed loss and correctness. |
| `observation_type` | One of `current_train`, `replay_train`, or `seen_task_eval`. |
| `trained_task_id` | Task being trained or most recently completed when the row was logged. |
| `evaluated_task_id` | Seen task being evaluated; `null` for training rows. |
| `epoch` | Epoch within the current task; `null` for evaluation rows. |
| `global_step` | Monotonic training step counter at observation time. |
| `is_replay` | Whether the row came from replay memory. |
| `replay_count` | Number of times that replay-buffer item has been replayed so far. Current-train and evaluation rows use `0`. |
| `last_replay_step` | Last step when the replay-buffer item was sampled. Current-train and evaluation rows use `null`. |
| `loss` | Per-sample cross-entropy loss. |
| `predicted_class` | Argmax class from the model logits. |
| `correct` | Whether `predicted_class == target`. |
| `confidence` | Maximum softmax probability. |
| `target_probability` | Softmax probability assigned to the target class. |
| `uncertainty` | `1 - confidence`. |
| `entropy` | Predictive entropy from the softmax distribution. |

## Observation Types

- `current_train`: current task training examples before the optimizer step.
- `replay_train`: replay-memory examples included in a replay-augmented batch
  before the optimizer step.
- `seen_task_eval`: evaluation examples from all seen tasks after a task is
  completed.

## Research Constraints

- Signal rows must be joined by `sample_id`, not by dataloader position.
- Signal logs are append-only evidence. Forgetting labels should be derived in a
  separate artifact during task 8.
- Scheduler code must not use future `seen_task_eval` rows when making replay
  choices.
- Any future expensive signal, such as gradient norm or representation drift,
  should extend this schema with a new schema version rather than changing the
  meaning of existing fields.

## Verified Artifact Counts

Signal logging was verified on both offline smoke baselines and full
Split CIFAR-100 single-seed runs.

| Run | Row count | Current-train rows | Replay-train rows | Seen-task eval rows |
| --- | ---: | ---: | ---: | ---: |
| `fine_tuning_smoke` | `120` | `72` | `0` | `48` |
| `random_replay_smoke` | `144` | `72` | `24` | `48` |
| `fine_tuning_split_cifar100_seed0_signals` | `105000` | `50000` | `0` | `55000` |
| `random_replay_split_cifar100_seed0_signals` | `150216` | `50000` | `45216` | `55000` |

The full signal runs reproduce the earlier task-5 and task-6 single-seed
baseline metrics:

| Run | Final accuracy | Average forgetting |
| --- | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `0.0455` | `0.434` |
| `random_replay_split_cifar100_seed0_signals` | `0.10129999999999999` | `0.30433333333333334` |

These are still single-seed baselines, not final statistical claims.

The future-forgetting labels derived from these signal artifacts are documented
in [FORGETTING_LABEL_DEFINITIONS.md](./FORGETTING_LABEL_DEFINITIONS.md).
