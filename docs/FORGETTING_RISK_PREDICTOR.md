# Forgetting-Risk Predictor Report

## Purpose

Task 9 tests whether the cheap sample-level signals from task 7 have predictive
value for the forgetting labels from task 8.

This is not the replay scheduler and it is not the full `T_i` estimator yet. It
is the diagnostic stage that asks:

```text
Given only what was known at an evaluation anchor, can we rank samples by later
forgetting risk better than chance?
```

## Inputs

Each report uses two artifacts from the same run directory:

- `sample_signals.json`
- `forgetting_labels.json`

The output is:

```text
forgetting_risk_report.json
```

## Leakage Rule

Features are built only from `seen_task_eval` signal rows at or before the
anchor row:

```text
feature_row.trained_task_id <= anchor_trained_task_id
feature_row.global_step <= anchor_global_step when trained_task_id is equal
```

Future rows are allowed to define labels, but never features.

## Features

The first feature set is intentionally cheap:

- anchor loss
- anchor uncertainty
- anchor confidence
- anchor target probability
- previous evaluation loss, uncertainty, and target probability
- loss increase from previous evaluation
- target-probability drop from previous evaluation
- max loss so far
- min target probability so far
- max uncertainty so far
- correctness history rate
- number of evaluations seen for that sample
- tasks since the sample's source task

These features are deliberately available from normal evaluation telemetry. No
gradient norms, representation drift, or future labels are used.

## Scores Evaluated

The report evaluates simple heuristic scores:

| Score | Meaning |
| --- | --- |
| `anchor_loss` | Higher current loss means higher predicted risk. |
| `anchor_uncertainty` | Higher uncertainty means higher predicted risk. |
| `low_target_probability` | Lower probability on the true class means higher predicted risk. |
| `loss_increase_from_previous` | Recent loss increase means higher predicted risk. |
| `target_probability_drop_from_previous` | Recent drop in true-class probability means higher predicted risk. |
| `combined_signal` | A simple min-max scaled combination of cheap signals. |

It also fits a lightweight logistic regression predictor when the temporal
training split contains both positive and negative examples.

## Temporal Evaluation

The report uses earlier anchor tasks for scaling or fitting and later anchor
tasks for testing.

For the full 10-task Split CIFAR-100 runs, the default split is:

```text
train anchors: anchor task <= 4
test anchors:  anchor task >= 5
```

Metrics are reported only over `eligible_for_binary_forgetting == true` rows,
because binary forgetting is only meaningful when the sample was correct at the
anchor.

## Metrics

The main metric is average precision, because forgetting labels can be
imbalanced. A random ranking has expected average precision close to the
positive rate in the test split.

The report also stores ROC-AUC when the test split has both classes, plus
precision and recall at the top 10% and 20% ranked samples.

## Full Single-Seed Results

These are diagnostics from one seed, not final claims.

| Run | Train eligible | Train positives | Test eligible | Test positives | Test positive rate | Best cheap heuristic AP | Logistic AP | Logistic ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `2162` | `2162` | `1776` | `1776` | `1.0` | `1.0` | skipped | skipped |
| `random_replay_split_cifar100_seed0_signals` | `3792` | `3369` | `3705` | `2762` | `0.7454790823211876` | `0.8471253916174554` | `0.9083240127221096` | `0.7978139160228613` |

Interpretation:

- Fine-tuning is not a useful discrimination test in this run because every
  held-out eligible anchor is positive. The predictor cannot separate forgotten
  from retained examples when there are no retained examples in the test split.
- Random replay gives a meaningful prediction test because the held-out split
  contains both positives and negatives.
- On random replay, cheap signals outperform the test positive-rate baseline.
  The logistic predictor improves further, but this is still a single-seed
  diagnostic result.

## Research Consequence

Task 9 supports moving forward, but cautiously:

- There is evidence that cheap signals carry predictive value under random
  replay.
- The fine-tuning run mostly confirms severe collapse, not predictor quality.
- The scheduler should initially use cheap signal heuristics and log every
  selection reason.
- The proposal still requires a time-to-forgetting estimate `T_i`; risk ranking
  alone does not satisfy the full spaced-replay mechanism.
- Repeated-seed evaluation is still required before making a research claim.

## Next Use

Task 10, Task 11, and Task 12 are now implemented:

- fixed-periodic replay is documented in [FIXED_PERIODIC_REPLAY_BASELINE.md](./FIXED_PERIODIC_REPLAY_BASELINE.md)
- `T_i` targets are documented in [TIME_TO_FORGETTING_TARGETS.md](./TIME_TO_FORGETTING_TARGETS.md)
- the first spaced replay scheduler is documented in [SPACED_REPLAY_SCHEDULER.md](./SPACED_REPLAY_SCHEDULER.md)

The next use of this predictor work is the Task 13 controlled comparison and
later ablations that test whether risk scoring, due-time estimation, or their
combination actually improves replay.
