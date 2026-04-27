# Forgetting Label Definitions

## Purpose

Task 8 turns `sample_signals.json` into `forgetting_labels.json`.

The goal is to define what the project will later try to predict. This is a
separate step from building the predictor and from building the replay scheduler.
Keeping those steps separate protects the research from future-information
leakage and from vague claims about "forgetting" that are not actually measured.

## Field Context

Continual-learning work usually measures forgetting as a drop in performance on
previously learned tasks after later tasks are trained. GEM formalized
continual-learning metrics around sequential tasks and transfer/forgetting
behavior. iCaRL made class-incremental CIFAR-100 a central setting for exemplar
memory and replay-style evaluation. MIR later made a sample-selection argument:
which memory items are likely to be most negatively affected by an update?

This project follows that logic but makes the unit of analysis smaller:

- standard continual-learning metrics ask whether a task was forgotten;
- this project asks whether an individual sample is likely to be forgotten;
- later replay policies can use that estimate to decide what to rehearse.

References used for design context:

- GEM: https://papers.neurips.cc/paper/7225-gradient-episodic-memory-for-continual-lea
- iCaRL: https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
- MIR: https://papers.nips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval

## Plain-English Explanation

Imagine the model learns CIFAR-100 in chunks. First it learns a group of
classes, then another group, then another. After each chunk, we test the model
on all chunks it has already seen.

For one test image, we can ask:

1. Was the model correct on this image at this point?
2. After later tasks were trained, did the model become wrong on this same image?
3. Even if it stayed correct, did its loss go up or did confidence in the true
   class go down?

That is the core of task 8. We are not yet predicting forgetting. We are only
creating the target labels that future prediction code must predict.

## Anchor Rows

An anchor row is a `seen_task_eval` signal row with at least one later
`seen_task_eval` row for the same `sample_id`.

Example:

```text
After task 2, evaluate sample 123.
Later, after tasks 3, 4, ..., 9, evaluate sample 123 again.
The task-2 evaluation is an anchor because it has future outcomes.
```

Rows after the final task are not anchors because there is no future evaluation
left to define forgetting.

## Eligibility

Binary forgetting is only meaningful when the model has the sample correct at
the anchor.

```text
eligible_for_binary_forgetting = anchor_correct == true
```

If the model is already wrong at the anchor, it may still be a hard or
unlearned sample, but it was not retained at that moment. We do not count it as
a binary forgetting event.

## Primary Definition

The primary label is:

```text
forgot_any_future
```

Definition:

```text
anchor_correct == true
and at least one later evaluation for the same sample_id is incorrect
```

Why this is primary:

- it is the closest sample-level analog of task-level forgetting;
- it captures temporary forgetting, not only final-state collapse;
- it is useful for replay because a scheduler should care if a sample becomes
  vulnerable at any later point, even if it eventually recovers.

## Secondary Binary Definitions

Task 8 also records two stricter binary labels.

| Label | Definition | Why Keep It |
| --- | --- | --- |
| `forgot_next_eval` | Correct at the anchor, wrong at the next later evaluation. | Measures immediate interference from the next task. |
| `forgot_final_eval` | Correct at the anchor, wrong at the final evaluation. | Matches the final-retention view used in many aggregate reports. |

These are not replacements for the primary label. They are diagnostic windows.
If a future predictor only works for `forgot_final_eval`, it may be predicting
permanent collapse but missing temporary interference. If it only works for
`forgot_next_eval`, it may be predicting short-term instability rather than
long-term forgetting.

## Continuous Targets

Correct/wrong labels are easy to audit, but they are coarse. The artifact also
stores continuous deterioration targets:

| Target | Meaning |
| --- | --- |
| `max_future_loss_increase` | Worst later increase in per-sample loss relative to the anchor. |
| `final_loss_delta` | Final loss minus anchor loss. |
| `max_future_target_probability_drop` | Worst later drop in probability assigned to the true class. |
| `final_target_probability_drop` | Anchor true-class probability minus final true-class probability. |
| `max_future_confidence_drop` | Worst later drop in max softmax confidence. |
| `final_confidence_drop` | Anchor confidence minus final confidence. |

The true-class probability targets are especially important. A model can become
confident in the wrong class, so max confidence alone is not always the cleanest
measure of memory for the correct label.

## Leakage Guard

Each label row stores:

```text
label_uses_future_after_task_id
leakage_safe
```

The builder enforces:

```text
future_row.trained_task_id > anchor_trained_task_id
```

That means the label is allowed to look into the future, but future rows must
not be used as predictor inputs. In task 9, features must come only from signal
history available at or before the anchor.

## Artifact

The task-8 script writes:

```text
forgetting_labels.json
```

The artifact includes:

- the definition text used to generate labels;
- a SHA-256 hash of the source `sample_signals.json`;
- label rows keyed by `sample_id`;
- counts by anchor task, source task, original class, and split;
- the primary label and secondary target values.

## Verified Full Single-Seed Counts

These counts were generated from the task-7 full Split CIFAR-100 signal runs.
They are useful diagnostics, not final statistical claims.

| Run | Anchor rows | Eligible anchors | `forgot_any_future` positives | Positive rate over eligible | `forgot_next_eval` positives | `forgot_final_eval` positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `45000` | `3938` | `3938` | `1.0` | `3907` | `3938` |
| `random_replay_split_cifar100_seed0_signals` | `45000` | `7497` | `6131` | `0.8177937841803388` | `4035` | `4973` |

Interpretation:

- fine-tuning retains relatively few eligible sample-checkpoints, and every
  retained eligible checkpoint eventually becomes incorrect later in this
  single-seed run;
- random replay creates more eligible retained checkpoints and lowers the
  any-future forgetting rate, but forgetting remains substantial;
- this supports moving to task 9 because the labels are nontrivial and aligned
  with the research question.

## Next Use

Task 9 should build a predictor or heuristic using only pre-anchor information.
The predictor should report precision-recall behavior because the positive
class can be imbalanced depending on the method, task, and anchor window.

## Time-To-Forgetting Note

The proposal also requires estimating an explicit sample-specific forgetting
time `T_i` and scheduling replay near `t + T_i`. The task-8 label artifact
defines binary forgetting windows and continuous deterioration targets. The
dedicated timing target is now implemented and documented in
[TIME_TO_FORGETTING_TARGETS.md](./TIME_TO_FORGETTING_TARGETS.md).

The `T_i` target adds:

- first future evaluation task where the sample becomes incorrect;
- task delta from anchor to first forgetting;
- step delta from anchor to first forgetting;
- right-censoring flag for samples not forgotten by the final evaluation;
- interval-censoring note, because exact forgetting may occur between two
  evaluation checkpoints.
