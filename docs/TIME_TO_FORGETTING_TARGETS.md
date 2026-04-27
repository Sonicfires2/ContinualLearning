# Time-To-Forgetting Targets

## Purpose

Task 11 defines the proposal's sample-specific forgetting-time target `T_i`.
Earlier work measured whether a retained sample is eventually forgotten. This
task adds the timing question:

```text
After a retained anchor observation at step t, when is the sample first observed forgotten?
```

This is the offline target needed before any scheduler can honestly claim to
replay near `t + T_i`.

## Literature Grounding

The definition follows three ideas from the literature:

- Continual-learning forgetting should be measured after later learning has
  interfered with earlier tasks. This is why the target uses future seen-task
  evaluations after an anchor checkpoint, consistent with standard continual
  forgetting metrics used in works such as Chaudhry et al.,
  [Continual Learning with Tiny Episodic Memories](https://arxiv.org/abs/1902.10486).
- Spacing effects depend on timing, not just item choice. Cepeda et al.,
  [Spacing Effects in Learning](https://journals.sagepub.com/doi/10.1111/j.1467-9280.2008.02209.x),
  show that retention changes with the interval between study/review and test.
- Trainable spaced repetition models estimate memory strength or recall
  half-life from prior observations. Settles and Meeder,
  [A Trainable Spaced Repetition Model for Language Learning](https://aclanthology.org/P16-1174/),
  motivate the idea that review timing can be predicted from history.

The continual-learning replay literature also supports the broader premise that
memory selection matters. MIR shows random replay is not always optimal
([Aljundi et al., NeurIPS 2019](https://papers.nips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval)),
and later work studies example-level forgetting dynamics
([Hacohen and Tuytelaars, ICML 2025](https://proceedings.mlr.press/v267/hacohen25a.html)).

## Definition

Artifact:

```text
time_to_forgetting_targets.json
```

Builder:

- [src/signals/time_to_forgetting.py](../src/signals/time_to_forgetting.py)
- [scripts/build_time_to_forgetting_targets.py](../scripts/build_time_to_forgetting_targets.py)

Primary target:

```text
first_observed_forgetting_step_delta
```

For each `seen_task_eval` anchor row with at least one later evaluation for the
same `sample_id`:

- `eligible_for_time_to_forgetting = anchor_correct == true`
- `event_observed = anchor_correct == true` and at least one later evaluation is
  incorrect
- `first_observed_forgetting_task_delta` is the first later incorrect
  `trained_task_id` minus the anchor `trained_task_id`
- `first_observed_forgetting_step_delta` is the first later incorrect
  `global_step` minus the anchor `global_step`

If the sample stays correct through the final future evaluation, the row is
right-censored. If the anchor is already incorrect, `T_i` is undefined because
the sample was not retained at the anchor.

## Censoring Rule

The exact forgetting moment is not observed continuously. It is only checked at
evaluation checkpoints. Therefore an observed event is interval-censored:

```text
last observed correct step < true forgetting time <= first observed incorrect step
```

The artifact stores both interval bounds:

- `interval_lower_step_delta`
- `interval_upper_step_delta`
- `interval_lower_task_delta`
- `interval_upper_task_delta`

For right-censored rows, only the lower bound is known: the sample survived at
least until the final available future evaluation.

## Evaluation Report

Artifact:

```text
time_to_forgetting_report.json
```

Evaluator:

- [src/predictors/time_to_forgetting.py](../src/predictors/time_to_forgetting.py)
- [scripts/evaluate_time_to_forgetting.py](../scripts/evaluate_time_to_forgetting.py)

The first timing report evaluates simple due-time proxies on a temporal split:
earlier anchor tasks are used for fitting/scaling, and later anchor tasks are
held out. Metrics are reported on observed events only for MAE, while
right-censored rows are counted separately.

## Single-Seed Results

These are diagnostic, not final statistical claims.

| Run | Eligible anchors | Observed events | Right-censored | Best step-delta MAE | Best estimator |
| --- | ---: | ---: | ---: | ---: | --- |
| `fine_tuning_split_cifar100_seed0_signals` | `3938` | `3938` | `0` | `0.5304054054054054` | `constant_train_median` |
| `random_replay_split_cifar100_seed0_signals` | `7497` | `6131` | `1366` | `30.069876900796523` | `constant_train_median` |
| `fixed_periodic_replay_split_cifar100_seed0` | `5095` | `5006` | `89` | `7.298861480075901` | `constant_train_median` |
| `spaced_replay_split_cifar100_seed0` | `7282` | `6026` | `1256` | `30.52126865671642` | `constant_train_median` |

The uncomfortable but important finding is that the crude risk-scaled timing
heuristics did not beat the constant median timing baseline on these pilot runs.
That means the first scheduler should be described as an online due-time proxy,
not as a validated learned `T_i` estimator.

## Scientific Interpretation

Task 11 successfully defines and evaluates the timing target the proposal needs.
It also reveals that predicting exact timing is harder than predicting binary
future forgetting risk in the current setup. This is not a failure of the
project; it is a constraint that must shape the claims.

The final report should separate:

- binary forgetting-risk prediction;
- time-to-first-observed-forgetting estimation;
- replay-scheduler intervention performance.

If later work improves `T_i` estimation, it should beat the constant median
baseline on held-out temporal anchors and handle right-censored samples
explicitly.
