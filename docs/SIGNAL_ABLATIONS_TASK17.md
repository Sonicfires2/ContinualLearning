# Task 17 Signal Ablations and Predictor-Quality Analysis

## Purpose

Task 17 asks which cheap sample-level signals are actually responsible for the
learned predictor gains from Task 16.

This matters because the replay scheduler should not use a learned predictor
just because it has a nicer average precision number. The project needs to know
whether the predictor is relying on interpretable, available-at-scheduling-time
signals and whether its threshold behavior is plausible for a future online
replay gate.

## Implementation

Main code:

- [src/predictors/signal_ablations.py](../src/predictors/signal_ablations.py)
- [scripts/evaluate_signal_ablations.py](../scripts/evaluate_signal_ablations.py)
- [tests/predictors/test_signal_ablations.py](../tests/predictors/test_signal_ablations.py)

Output artifact:

```text
signal_ablation_report.json
```

The report keeps the same leakage rule as the earlier predictor reports:

- features use only `seen_task_eval` signal rows at or before the anchor;
- labels use future evaluations;
- earlier anchor tasks train or scale predictors;
- later anchor tasks are held out.

For the full Split CIFAR-100 seed-0 artifacts:

```text
train anchors: anchor task <= 4
test anchors:  anchor task >= 5
```

## Feature Groups

The ablation report evaluates these groups:

| Feature group | Meaning |
| --- | --- |
| `loss_only` | Anchor, previous, delta, increase, and max loss features. |
| `uncertainty_only` | Anchor, previous, delta, and max uncertainty features. |
| `target_probability_only` | Anchor, previous, delta, drop, and minimum true-class probability features. |
| `anchor_state` | Current loss, uncertainty, confidence, and target probability. |
| `history_delta` | Short-term changes between the previous and anchor evaluations. |
| `history_summary` | Evaluation count, previous correctness, correctness rate, tasks since source, and anchor progress. |
| `all_features` | The full Task 16 feature set. |

Each feature group is evaluated with logistic regression and a linear SVM
classifier. The report also keeps the best cheap heuristic score as a baseline.

## Results

These are single-seed diagnostics from existing artifacts, not repeated-seed
claims.

### Random replay seed 0

Artifact:

```text
experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals/signal_ablation_report.json
```

| Rank | Feature group | Best model | Average precision |
| ---: | --- | --- | ---: |
| 1 | `all_features` | `logistic_regression` | `0.9083240127221096` |
| 2 | `history_summary` | `linear_svm_classifier` | `0.8951623697996727` |
| 3 | `uncertainty_only` | `logistic_regression` | `0.8683205257744547` |
| 4 | `target_probability_only` | `logistic_regression` | `0.8665784039155886` |
| 5 | `loss_only` | `logistic_regression` | `0.8611969396771237` |
| 6 | `anchor_state` | `logistic_regression` | `0.8471253916174554` |
| 7 | `history_delta` | `linear_svm_classifier` | `0.8059718959561837` |

Best cheap heuristic:

```text
anchor_loss AP = 0.8471253916174554
```

Logistic threshold behavior for `all_features`:

| Threshold | Selected fraction | Precision | Recall |
| ---: | ---: | ---: | ---: |
| `0.3` | `0.7889338731443994` | `0.8450222374273008` | `0.8942795076031861` |
| `0.5` | `0.6402159244264507` | `0.8827993254637436` | `0.7581462708182476` |
| `0.7` | `0.43967611336032386` | `0.9134438305709024` | `0.5387400434467777` |
| `0.9` | `0.11336032388663968` | `0.9523809523809523` | `0.14482259232440262` |

### Spaced replay seed 0

Artifact:

```text
experiments/runs/spaced_replay/spaced_replay_split_cifar100_seed0/signal_ablation_report.json
```

| Rank | Feature group | Best model | Average precision |
| ---: | --- | --- | ---: |
| 1 | `all_features` | `logistic_regression` | `0.9188844673560967` |
| 2 | `history_summary` | `linear_svm_classifier` | `0.9070827647828695` |
| 3 | `target_probability_only` | `linear_svm_classifier` | `0.8504351775572876` |
| 4 | `uncertainty_only` | `logistic_regression` | `0.8500353140223217` |
| 5 | `loss_only` | `logistic_regression` | `0.8429752580361802` |
| 6 | `anchor_state` | `logistic_regression` | `0.8267285181601283` |
| 7 | `history_delta` | `logistic_regression` | `0.8069975088982162` |

Best cheap heuristic:

```text
anchor_loss AP = 0.8267285181601283
```

Logistic threshold behavior for `all_features`:

| Threshold | Selected fraction | Precision | Recall |
| ---: | ---: | ---: | ---: |
| `0.3` | `0.7737390814313891` | `0.8652585579024035` | `0.8865671641791045` |
| `0.5` | `0.5781910397295013` | `0.9108187134502924` | `0.6973880597014925` |
| `0.7` | `0.34657650042265425` | `0.9390243902439024` | `0.43097014925373134` |
| `0.9` | `0.020005635390250773` | `0.9154929577464789` | `0.024253731343283583` |

## Degenerate Runs

The fine-tuning and fixed-periodic seed-0 artifacts also produce reports, but
the learned classifiers cannot fit because the training temporal split has only
one target class.

```text
fine_tuning_split_cifar100_seed0_signals:
use_learned_online_gate_next = false

fixed_periodic_replay_split_cifar100_seed0:
use_learned_online_gate_next = false
```

These runs are still useful as collapse diagnostics, but they should not drive
supervised predictor selection.

## Interpretation

The ablations support three useful conclusions.

First, the learned predictor gain is not coming from one isolated scalar. The
full feature set is best on both informative replay artifacts.

Second, `history_summary` is consistently the strongest compact group. That
means sample age, previous correctness, correctness history rate, and task
progress appear to matter. This is aligned with the proposal's spacing
intuition: history and elapsed interference are predictive, not just current
loss.

Third, `history_delta` alone is weak. The scheduler should not rely only on
recent loss or probability changes.

## Recommendation

Task 17 recommends implementing a learned-predictor replay gate next, but with
careful safeguards:

- fit the predictor from prior-run artifacts or from strictly earlier anchors;
- start with logistic regression over `all_features`;
- log probability thresholds, selected fractions, precision/recall proxy
  assumptions, skipped replay events, and replay-sample savings;
- compare against Task 15's cheap risk-gated replay, random replay, fixed
  replay, spaced replay, and MIR.

This should be treated as the next intervention test, not as a final claim.

## Verification

Focused predictor verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\predictors -q
```

Result:

```text
15 passed
```

Full repository verification after Task 17:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
70 passed
```
