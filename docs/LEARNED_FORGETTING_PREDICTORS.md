# Task 16 Learned Forgetting Predictors

## Purpose

Task 16 implements the learned predictor families named in the proposal and
compares them against the current cheap forgetting-risk heuristic before any
learned model is trusted as a replay gate.

This task evaluates prediction, not intervention:

```text
given leakage-safe sample history at an anchor, can learned models rank or
estimate later forgetting better than cheap signal scores?
```

## Implementation

Main code:

- [src/predictors/learned_forgetting.py](../src/predictors/learned_forgetting.py)
- [scripts/evaluate_learned_forgetting_predictors.py](../scripts/evaluate_learned_forgetting_predictors.py)
- [tests/predictors/test_learned_forgetting.py](../tests/predictors/test_learned_forgetting.py)

Output artifact:

```text
learned_forgetting_predictor_report.json
```

The report uses the same leakage-safe feature rows as Task 9:

- signal rows come only from `seen_task_eval` observations at or before the
  anchor;
- labels and continuous deterioration targets come from future evaluations;
- temporal evaluation trains on earlier anchor tasks and tests on later anchor
  tasks.

For the full Split CIFAR-100 runs, the default split is:

```text
train anchors: anchor task <= 4
test anchors:  anchor task >= 5
```

## Models Added

Binary future-forgetting classification:

- `logistic_regression`
- `linear_svm_classifier`

Continuous future-deterioration regression:

- `linear_regression`
- `ridge_regression`
- `linear_svm_regressor`
- `constant_train_mean` baseline

Optional observed time-to-forgetting regression:

- `linear_regression`
- `ridge_regression`
- `linear_svm_regressor`
- `constant_train_mean` baseline

The SVM models record non-fatal fit warnings inside the JSON report. This keeps
CLI runs clean while preserving convergence information for later analysis.

## Metrics

Binary classifiers report:

- average precision;
- ROC-AUC when the held-out split contains both classes;
- precision and recall at the top 10%, 20%, and 30% of ranked samples;
- probability-threshold behavior for logistic regression.

Regression models report:

- MAE;
- RMSE;
- R2 when valid;
- target mean and standard deviation.

## Single-Seed Results

These are diagnostics from existing seed-0 artifacts, not repeated-seed claims.

| Run | Best cheap heuristic AP | Best learned binary AP | Best learned model | Learned beats heuristic? | Notes |
| --- | ---: | ---: | --- | --- | --- |
| `random_replay_split_cifar100_seed0_signals` | `0.8471253916174554` | `0.9083240127221096` | `logistic_regression` | `true` | Held-out positive rate is `0.7454790823211875`, so this is a meaningful discrimination test. |
| `spaced_replay_split_cifar100_seed0` | `0.8267285181601283` | `0.9188844673560967` | `logistic_regression` | `true` | Learned predictors improve ranking on the spaced-replay artifact as well. |
| `fine_tuning_split_cifar100_seed0_signals` | `1.0` | skipped | none | n/a | Training split has only one target class, so supervised classifiers cannot fit. |
| `fixed_periodic_replay_split_cifar100_seed0` | `0.9618410015616607` | skipped | none | n/a | Training split has only one target class; this run is mostly a collapse diagnostic. |

For random replay, `linear_svm_classifier` is close to logistic regression:

```text
linear_svm_classifier AP = 0.9073454366577642
linear_svm_classifier ROC-AUC = 0.7982654307857817
```

For spaced replay:

```text
linear_svm_classifier AP = 0.917337540083114
linear_svm_classifier ROC-AUC = 0.8182767119523211
```

## Regression Diagnostics

The continuous and time-to-forgetting regressors are now implemented, but they
should be treated as diagnostics.

On the random-replay artifact:

```text
ridge_regression on max_future_loss_increase:
MAE = 0.6791658883584177
RMSE = 0.9195014677415336
R2 = 0.32926868857339087
```

For observed-event time-to-forgetting on the same artifact:

```text
constant_train_mean MAE = 158.73290602286167
ridge_regression MAE = 129.16222919724063
```

This is promising as a diagnostic, but it excludes right-censored rows. It is
not yet a validated online `T_i` estimator.

## Scientific Interpretation

Task 16 strengthens the predictive part of the proposal. On the replay runs
with usable class balance, learned binary predictors outperform the best cheap
single heuristic by average precision, and logistic regression is currently the
best simple learned classifier.

This does not overturn Task 15's scheduler result. The risk-gated scheduler
used a cheap online heuristic, and learned predictors have only been evaluated
offline so far. Task 17 now adds signal ablations showing `all_features` is the
strongest gate input. The next scheduler improvement should train a predictor
only from prior runs or prior anchors and test it online against the cheap
risk-gated scheduler.

Until then, the project can claim:

```text
learned predictors improve offline forgetting-risk ranking on informative
seed-0 replay artifacts
```

It should not yet claim:

```text
learned risk-gated replay improves continual-learning retention
```

## Verification

Focused predictor verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\predictors -q
```

Result:

```text
12 passed
```

Full repository verification after Task 16:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
67 passed
```
