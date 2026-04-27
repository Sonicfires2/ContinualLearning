# Task 21 Gradient Signal Diagnostic

## Purpose

Task 21 tests one expensive signal family from the proposal: gradient norms.

The previous learned replay interventions failed even though cheap signals
predicted forgetting offline. This task asks:

```text
Do gradient-family signals improve forgetting prediction enough to justify a
new replay scheduler?
```

This is a diagnostic task, not a new replay intervention.

## Implementation

Main code:

- [src/signals/gradient_signals.py](../src/signals/gradient_signals.py)
- [src/baselines/gradient_signal_diagnostic.py](../src/baselines/gradient_signal_diagnostic.py)
- [src/predictors/expensive_signal_diagnostics.py](../src/predictors/expensive_signal_diagnostics.py)
- [scripts/evaluate_expensive_signal_diagnostics.py](../scripts/evaluate_expensive_signal_diagnostics.py)
- [configs/experiments/gradient_signal_diagnostic_smoke.yaml](../configs/experiments/gradient_signal_diagnostic_smoke.yaml)
- [configs/experiments/gradient_signal_diagnostic_split_cifar100.yaml](../configs/experiments/gradient_signal_diagnostic_split_cifar100.yaml)
- [tests/signals/test_gradient_signals.py](../tests/signals/test_gradient_signals.py)
- [tests/predictors/test_expensive_signal_diagnostics.py](../tests/predictors/test_expensive_signal_diagnostics.py)
- [tests/baselines/test_gradient_signal_diagnostic.py](../tests/baselines/test_gradient_signal_diagnostic.py)

The implemented signal is the exact per-sample cross-entropy gradient norm for
the final linear classifier layer. It is cheaper than a full-model per-sample
backward pass, but it is still a real gradient signal:

```text
grad_W loss_i = outer(probabilities_i - one_hot(target_i), penultimate_features_i)
```

The diagnostic logs:

- logit-gradient L2 norm;
- penultimate activation L2 norm;
- classifier weight-gradient L2 norm;
- classifier bias-gradient L2 norm;
- combined final-layer gradient L2 norm.

## Smoke Result

Artifact:

```text
.tmp/baseline_runs/gradient_signal_diagnostic/gradient_signal_diagnostic/gradient_signal_diagnostic_smoke
```

The smoke run verified the artifact path:

| Metric | Value |
| --- | ---: |
| gradient rows | `48` |
| observation type | `seen_task_eval` |
| forgotten positives | `0` |

The smoke fixture has no positive forgetting labels, so it verifies plumbing but
cannot evaluate predictive value.

## Split CIFAR-100 Diagnostic

Artifact:

```text
experiments/runs/gradient_signal_diagnostic/gradient_signal_diagnostic/gradient_signal_diagnostic_split_cifar100_seed0_random_replay
```

The diagnostic run matches random replay seed 0:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.10129999999999999` |
| average forgetting | `0.30433333333333334` |
| replay samples | `45216` |
| sample-signal rows | `150216` |
| gradient-signal rows | `55000` |
| gradient unique samples | `10000` |
| mean final-layer gradient L2 | `20.583347188570553` |
| max final-layer gradient L2 | `65.6209716796875` |

## Predictor Result

Report:

```text
experiments/runs/gradient_signal_diagnostic/gradient_signal_diagnostic/gradient_signal_diagnostic_split_cifar100_seed0_random_replay/expensive_signal_diagnostic_report.json
```

The temporal split is still:

```text
train anchors <= task 4
test anchors >= task 5
```

Average precision results:

| Feature group | Best model | Average precision |
| --- | --- | ---: |
| cheap all features | logistic regression | `0.9083240127221096` |
| cheap plus gradient | logistic regression | `0.9080918805327551` |
| gradient only | linear SVM classifier | `0.8386703932509996` |
| best cheap heuristic | anchor loss | `0.8471253916174554` |

The gradient-only group is worse than the cheap learned feature set and even
slightly worse than the best cheap heuristic. Adding gradient features to the
cheap features does not improve average precision.

## Cost

Compared with the existing random replay seed-0 run:

| Cost item | Value |
| --- | ---: |
| random replay training time | `20.07671979998122` seconds |
| gradient diagnostic training time | `31.610530899997684` seconds |
| added time | `11.533811100016464` seconds |
| relative overhead | `0.5744868292691545` |
| gradient artifact size | `42909732` bytes |

The diagnostic adds about `57%` measured training-time overhead and writes a
large extra artifact.

## Scientific Interpretation

Task 21 is a negative diagnostic result.

In plain language:

```text
The gradient signal is measurable, but it does not tell us more than the cheap
signals already told us.
```

This does not mean all gradient or interference ideas are bad. MIR is still the
strongest implemented replay method, and MIR is interference-based. The result
only says that this particular final-layer gradient norm is not enough to
justify building another replay scheduler.

The project should not build a gradient-norm replay policy from this signal.

## Next Research Move

The original next step was stretch benchmarks, but the current evidence does
not justify expanding to new datasets yet. The Split CIFAR-100 core still has no
forgetting-aware method that beats random replay.

The better next step is a decision checkpoint:

1. Treat the current work as a clean negative result for risk-based replay.
2. Run only small targeted rescue ablations if needed, such as a lower
   learned-risk fraction (`0.25`) or random-diversity hybrid.
3. If continuing method development, pivot toward MIR-like current-interference
   or representation-drift diagnostics, not final-layer gradient norm alone.
4. Move stretch benchmarks behind this decision point.

## Verification

Focused verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\signals\test_gradient_signals.py tests\predictors\test_expensive_signal_diagnostics.py tests\baselines\test_gradient_signal_diagnostic.py tests\baselines\test_random_replay_baseline.py -q
```

Result:

```text
9 passed
```

Full repository verification after Task 21:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
86 passed
```

Experiment and report commands:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.gradient_signal_diagnostic --config configs\experiments\gradient_signal_diagnostic_smoke.yaml
.\.venv\Scripts\python.exe scripts\build_forgetting_labels.py --run-dir .tmp\baseline_runs\gradient_signal_diagnostic\gradient_signal_diagnostic\gradient_signal_diagnostic_smoke
.\.venv\Scripts\python.exe scripts\evaluate_expensive_signal_diagnostics.py --run-dir .tmp\baseline_runs\gradient_signal_diagnostic\gradient_signal_diagnostic\gradient_signal_diagnostic_smoke
.\.venv\Scripts\python.exe -m src.baselines.gradient_signal_diagnostic --config configs\experiments\gradient_signal_diagnostic_split_cifar100.yaml
.\.venv\Scripts\python.exe scripts\build_forgetting_labels.py --run-dir experiments\runs\gradient_signal_diagnostic\gradient_signal_diagnostic\gradient_signal_diagnostic_split_cifar100_seed0_random_replay
.\.venv\Scripts\python.exe scripts\evaluate_expensive_signal_diagnostics.py --run-dir experiments\runs\gradient_signal_diagnostic\gradient_signal_diagnostic\gradient_signal_diagnostic_split_cifar100_seed0_random_replay --reference-metrics experiments\runs\random_replay\random_replay_split_cifar100_seed0_signals\metrics.json
```
