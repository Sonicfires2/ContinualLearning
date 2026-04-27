# Progress Log

## Purpose

This document records implementation progress in a way that stays tied to the research goal, not just to code activity. Each entry should explain what was changed, why it was changed at that point in the sequence, and how the change strengthens the final scientific claim.

## Progress Entry: 2026-04-27

Task: `Repository handoff and GitHub readiness`  
Status: `complete`

### What I did

- added [TEAMMATE_HANDOFF.md](./TEAMMATE_HANDOFF.md) as the first document a
  teammate should read before continuing the project
- added Task-22 rescue-ablation configs for `25%` learned-risk plus
  class-balanced replay, `25%` learned-risk plus random replay, and pure
  class-balanced replay
- added config-loader coverage so the staged Task-22 configs are checked by
  the test suite
- updated [README.md](../README.md) with a handoff pointer and the current test
  verification command
- updated [experiments/README.md](../experiments/README.md) to make clear that
  generated run outputs should not be committed
- updated `.gitignore` so generated experiment result directories stay out of
  Git while source code, configs, tests, and documentation remain commit-ready

### Why this change is needed

The project has moved from early implementation into a handoff stage. A
teammate now needs to understand not only how to run code, but also what the
results mean and why the next task is a decision checkpoint instead of another
large benchmark.

### Current status

The repository has implemented Tasks 1 through 21. The main result so far is:

```text
forgetting can be predicted offline, but the tested risk-guided replay policies
do not yet beat random replay online
```

The next research task remains Task 22: a small targeted rescue ablation and
decision checkpoint.

### Evidence of completion

- handoff document: [TEAMMATE_HANDOFF.md](./TEAMMATE_HANDOFF.md)
- status document: [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md)
- next-task source of truth: [ACTION_PLAN.md](./ACTION_PLAN.md)
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `87 passed`

## Progress Entry: 2026-04-25

Task: `21 - Add expensive signals such as gradient norms or representation drift only if runtime allows`  
Status: `complete`

### What I did

- added [src/signals/gradient_signals.py](../src/signals/gradient_signals.py)
  to compute exact per-sample final-layer gradient norms
- added [src/baselines/gradient_signal_diagnostic.py](../src/baselines/gradient_signal_diagnostic.py)
  to run random replay while logging `gradient_signals.json`
- added [src/predictors/expensive_signal_diagnostics.py](../src/predictors/expensive_signal_diagnostics.py)
  and [scripts/evaluate_expensive_signal_diagnostics.py](../scripts/evaluate_expensive_signal_diagnostics.py)
  to test whether gradient features improve forgetting prediction
- added smoke and Split CIFAR-100 configs:
  [gradient_signal_diagnostic_smoke.yaml](../configs/experiments/gradient_signal_diagnostic_smoke.yaml)
  and
  [gradient_signal_diagnostic_split_cifar100.yaml](../configs/experiments/gradient_signal_diagnostic_split_cifar100.yaml)
- added tests for gradient signal logging, expensive-signal reports, and the
  diagnostic baseline
- ran smoke and Split CIFAR-100 seed-0 diagnostics
- added [GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md)
- added [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md), [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md),
  and [README.md](../README.md)

### Why this change is needed

Tasks 18 through 20 showed that cheap learned-risk replay did not beat random
replay. Task 21 tests whether adding a proposal-listed expensive signal,
gradient norm, improves predictor quality enough to justify another scheduler.

### Results

Split CIFAR-100 seed-0 diagnostic:

| Feature group | Best model | Average precision |
| --- | --- | ---: |
| cheap all features | logistic regression | `0.9083240127221096` |
| cheap plus gradient | logistic regression | `0.9080918805327551` |
| gradient only | linear SVM classifier | `0.8386703932509996` |
| best cheap heuristic | anchor loss | `0.8471253916174554` |

Cost:

| Cost item | Value |
| --- | ---: |
| reference random replay time | `20.07671979998122` seconds |
| gradient diagnostic time | `31.610530899997684` seconds |
| relative overhead | `0.5744868292691545` |
| gradient artifact size | `42909732` bytes |

### Scientific interpretation

Task 21 is a negative diagnostic result. The final-layer gradient norm is
measurable and proposal-aligned, but it does not improve the leakage-safe
forgetting predictor over cheap features. It should not be turned into a replay
scheduler.

The broader retrospective is now:

```text
forgetting can be predicted offline, but the tested risk-guided replay policies
do not yet beat random replay online
```

MIR remains the best evidence for what kind of signal matters: current-update
interference, not general future-risk ranking.

### Evidence of completion

- method document:
  [GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md)
- retrospective:
  [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\signals\test_gradient_signals.py tests\predictors\test_expensive_signal_diagnostics.py tests\baselines\test_gradient_signal_diagnostic.py tests\baselines\test_random_replay_baseline.py -q`
- focused verification result: `9 passed`
- full verification command:
  `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `86 passed`
- Split CIFAR-100 diagnostic command:
  `.\.venv\Scripts\python.exe -m src.baselines.gradient_signal_diagnostic --config configs\experiments\gradient_signal_diagnostic_split_cifar100.yaml`
- diagnostic report command:
  `.\.venv\Scripts\python.exe scripts\evaluate_expensive_signal_diagnostics.py --run-dir experiments\runs\gradient_signal_diagnostic\gradient_signal_diagnostic\gradient_signal_diagnostic_split_cifar100_seed0_random_replay --reference-metrics experiments\runs\random_replay\random_replay_split_cifar100_seed0_signals\metrics.json`

### Next task unlocked by this work

The plan should be modified before moving to stretch benchmarks. The next
research step should be a decision checkpoint plus small targeted rescue
ablations, not Split CUB or DistilBERT.

## Progress Entry: 2026-04-25

Task: `20 - Implement a balanced hybrid of learned-risk replay plus random or class-balanced replay`  
Status: `complete`

### What I did

- added [src/baselines/learned_hybrid_replay.py](../src/baselines/learned_hybrid_replay.py)
  as a fixed-budget hybrid replay baseline
- added smoke and Split CIFAR-100 configs:
  [learned_hybrid_replay_smoke.yaml](../configs/experiments/learned_hybrid_replay_smoke.yaml)
  and
  [learned_hybrid_replay_split_cifar100.yaml](../configs/experiments/learned_hybrid_replay_split_cifar100.yaml)
- added [tests/baselines/test_learned_hybrid_replay.py](../tests/baselines/test_learned_hybrid_replay.py)
  for class-balanced diversity selection, smoke artifacts, and config loading
- ran the smoke and seed-0 Split CIFAR-100 pilots
- added [LEARNED_HYBRID_REPLAY_TASK20.md](./LEARNED_HYBRID_REPLAY_TASK20.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md), [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md),
  [FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md),
  and [README.md](../README.md)

### Why this change is needed

Task 19 showed that pure learned-risk replay was worse than random replay even
with the same replay budget. Task 20 tests the next most plausible explanation:
maybe learned risk needs diversity protection.

The implemented pilot uses:

```text
50% learned-risk-ranked replay
50% class-balanced replay
```

### Results

Smoke result:

| Metric | Value |
| --- | ---: |
| replay samples | `24` |
| learned-risk selections | `12` |
| class-balanced selections | `12` |
| skipped replay opportunities | `0` |

Split CIFAR-100 seed-0 result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid replay, 50/50 class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

### Scientific interpretation

Task 20 is mixed but still negative. The hybrid improves over pure learned-risk
replay, which supports the idea that diversity matters. But it still does not
beat random replay, so the current learned-risk score is not yet an effective
online replay selector.

The project should now say:

```text
cheap signals predict forgetting offline, but the implemented learned-risk
selectors do not yet beat random replay as interventions
```

### Evidence of completion

- method document:
  [LEARNED_HYBRID_REPLAY_TASK20.md](./LEARNED_HYBRID_REPLAY_TASK20.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\baselines\test_learned_hybrid_replay.py tests\baselines\test_learned_fixed_budget_replay.py tests\replay\test_spaced_scheduler.py tests\predictors\test_online_forgetting.py -q`
- focused verification result: `10 passed`
- full verification command:
  `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `79 passed`
- smoke command:
  `.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_smoke.yaml`
- Split CIFAR-100 pilot command:
  `.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_split_cifar100.yaml`

### Next task unlocked by this work

Task `21 - Add expensive signals such as gradient norms or representation drift only if runtime allows`

## Progress Entry: 2026-04-24

Task: `19 - Implement fixed-budget learned-risk replay`  
Status: `complete`

### What I did

- added [src/baselines/learned_fixed_budget_replay.py](../src/baselines/learned_fixed_budget_replay.py)
  as a fixed-budget learned-risk replay baseline
- extended [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
  with a `risk_ranked` mode that fills the replay batch by learned risk instead
  of skipping low-risk replay
- added smoke and Split CIFAR-100 configs:
  [learned_fixed_budget_replay_smoke.yaml](../configs/experiments/learned_fixed_budget_replay_smoke.yaml)
  and
  [learned_fixed_budget_replay_split_cifar100.yaml](../configs/experiments/learned_fixed_budget_replay_split_cifar100.yaml)
- added tests in
  [tests/baselines/test_learned_fixed_budget_replay.py](../tests/baselines/test_learned_fixed_budget_replay.py)
  and extended scheduler coverage in
  [tests/replay/test_spaced_scheduler.py](../tests/replay/test_spaced_scheduler.py)
- ran the smoke and seed-0 Split CIFAR-100 pilots
- added [LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](./LEARNED_FIXED_BUDGET_REPLAY_TASK19.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md), [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md),
  [FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md),
  and [README.md](../README.md)

### Why this change is needed

Task 18 mixed learned-risk ranking with sparse replay skipping. Because that
result was negative, the next clean test was to remove skipping and ask whether
the learned predictor can choose better replay examples under the same replay
budget as random replay.

### Results

Smoke result:

| Metric | Value |
| --- | ---: |
| replay samples | `24` |
| replay-augmented batches | `6` |
| skipped replay opportunities | `0` |
| scheduler trace rows | `24` |

Split CIFAR-100 seed-0 result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

### Scientific interpretation

Task 19 is a negative but clarifying result. Learned fixed-budget replay uses
the same replay volume as random replay, so the weaker result cannot be blamed
on replay skipping. Pure learned-risk ranking is not yet a better replay
selection policy than random replay under this seed-0 test.

This does not invalidate the proposal's broader idea. It narrows the next
question: learned risk may need diversity protection. Task 20 should combine
some learned-risk selections with random or class-balanced selections.

### Evidence of completion

- method document:
  [LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](./LEARNED_FIXED_BUDGET_REPLAY_TASK19.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\replay\test_spaced_scheduler.py tests\baselines\test_learned_fixed_budget_replay.py tests\predictors\test_online_forgetting.py -q`
- focused verification result: `7 passed`
- full verification command:
  `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `76 passed`
- smoke command:
  `.\.venv\Scripts\python.exe -m src.baselines.learned_fixed_budget_replay --config configs\experiments\learned_fixed_budget_replay_smoke.yaml`
- Split CIFAR-100 pilot command:
  `.\.venv\Scripts\python.exe -m src.baselines.learned_fixed_budget_replay --config configs\experiments\learned_fixed_budget_replay_split_cifar100.yaml`

### Next task unlocked by this work

Task `20 - Implement a balanced hybrid of learned-risk replay plus random or class-balanced replay`

## Progress Entry: 2026-04-24

Task: `Post-Task-18 research plan adjustment: fixed-budget learned replay before expensive signals`  
Status: `complete`

### What I changed

- added [FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so Task 19 is now
  `Implement fixed-budget learned-risk replay`
- inserted Task 20, `Implement a balanced hybrid of learned-risk replay plus
  random or class-balanced replay`
- moved expensive signals back to Task 21
- updated [LEARNED_RISK_GATED_REPLAY_TASK18.md](./LEARNED_RISK_GATED_REPLAY_TASK18.md)
  and [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md)

### Why this change is needed

Task 18 failed as an intervention, but it tested two ideas at once:

```text
learned-risk ranking + sparse replay skipping
```

The result tells us sparse learned gating did not work. It does not yet tell us
whether learned risk is useful for choosing examples when the replay budget is
held fixed.

The next test should therefore remove the skip decision:

```text
same replay budget as random replay, but select examples by learned risk
```

This keeps the project aligned with the proposal because the proposal is about
using forgetting signals to improve replay timing/selection. It also keeps the
research honest because random replay and MIR remain the standards to beat.

### Updated near-term sequence

1. Task 19: fixed-budget learned-risk replay.
2. Task 20: balanced hybrid replay using learned risk plus random or
   class-balanced diversity.
3. Task 21: expensive signals such as gradient norms or representation drift,
   only if the fixed-budget and hybrid diagnostics justify more signal cost.

### What success or failure will mean

If Task 19 beats random replay, learned risk is useful for selection and sparse
gating was the weak part.

If Task 19 fails but Task 20 helps, learned risk needs diversity protection.

If both fail, the project should conclude that the current cheap signal set can
predict forgetting offline but is not yet operationally useful for replay
selection.

### Next task unlocked by this work

Task `19 - Implement fixed-budget learned-risk replay`

## Progress Entry: 2026-04-24

Task: `18 - Implement a learned-predictor risk-gated replay variant`  
Status: `complete`

### What I did

- added [src/predictors/online_forgetting.py](../src/predictors/online_forgetting.py)
  to train a logistic online risk scorer from prior signal/label artifacts
- extended [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
  so it can use either the original cheap heuristic risk score or a learned
  risk scorer
- added [src/baselines/learned_risk_gated_replay.py](../src/baselines/learned_risk_gated_replay.py)
  as the Task 18 learned-gate intervention
- added smoke and Split CIFAR-100 configs:
  [learned_risk_gated_replay_smoke.yaml](../configs/experiments/learned_risk_gated_replay_smoke.yaml)
  and
  [learned_risk_gated_replay_split_cifar100.yaml](../configs/experiments/learned_risk_gated_replay_split_cifar100.yaml)
- added [tests/predictors/test_online_forgetting.py](../tests/predictors/test_online_forgetting.py)
  and
  [tests/baselines/test_learned_risk_gated_replay_baseline.py](../tests/baselines/test_learned_risk_gated_replay_baseline.py)
- ran smoke and seed-0 Split CIFAR-100 pilots
- added [LEARNED_RISK_GATED_REPLAY_TASK18.md](./LEARNED_RISK_GATED_REPLAY_TASK18.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md), [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md),
  and [README.md](../README.md)

### Leakage guard

The active learned-gate replay run does not read its own future labels. For the
Split CIFAR-100 pilot, the logistic scorer is trained before the run from the
prior random-replay seed-0 signal and label artifacts:

```text
experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals
```

The run config saves the source artifact paths and SHA-256 hashes.

### Results

Smoke result:

| Metric | Value |
| --- | ---: |
| replay samples | `14` |
| replay-augmented batches | `5` |
| skipped replay opportunities | `1` |
| effective replay ratio | `0.5555555555555556` |

Split CIFAR-100 seed-0 pilots:

| Gate | Threshold | Final accuracy | Avg forgetting | Replay samples | Replay batches | Skipped events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned `risk_only` | `0.70` | `0.0376` | `0.4286666666666667` | `21423` | `672` | `39` |
| learned `risk_only` | `0.90` | `0.0379` | `0.40311111111111114` | `14425` | `466` | `245` |

Reference context:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay seed 0 | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| MIR replay seed 0 | `0.1183` | `0.21400000000000002` | `45216` |
| cheap risk-gated replay threshold `0.75` | `0.046` | `0.43377777777777776` | `2071` |
| learned risk-gated replay threshold `0.90` | `0.0379` | `0.40311111111111114` | `14425` |

### Scientific interpretation

Task 18 is a negative intervention result. The learned predictor is better
offline, but directly using it as an online sparse replay gate does not improve
retention. It performs worse than random replay and much worse than MIR.

The likely issue is that offline evaluation-anchor features and online
training-batch features are not the same distribution. The learned gate often
assigns high risk to many online observations, and replaying those examples can
over-focus on hard or unstable samples rather than maintaining a balanced memory
of old tasks.

The project should now say:

```text
learned offline forgetting prediction improved, but the first learned online
replay gate failed under the tested setup
```

### Evidence of completion

- method document:
  [LEARNED_RISK_GATED_REPLAY_TASK18.md](./LEARNED_RISK_GATED_REPLAY_TASK18.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\predictors\test_online_forgetting.py tests\replay\test_spaced_scheduler.py tests\baselines\test_learned_risk_gated_replay_baseline.py -q`
- focused verification result: `6 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `73 passed`

### Next task unlocked by this work

Originally this pointed to expensive signals. The post-Task-18 plan adjustment
above supersedes that: the next task is now
`19 - Implement fixed-budget learned-risk replay`.

## Progress Entry: 2026-04-24

Task: `17 - Run signal ablations and predictor-quality analysis`  
Status: `complete`

### What I did

- added [src/predictors/signal_ablations.py](../src/predictors/signal_ablations.py)
  to evaluate feature-group ablations for learned forgetting-risk predictors
- added [scripts/evaluate_signal_ablations.py](../scripts/evaluate_signal_ablations.py)
  to build `signal_ablation_report.json` from a run directory
- extended `evaluate_binary_learned_models` so Task 17 can evaluate custom
  feature subsets while preserving Task 16's full-feature report
- added [tests/predictors/test_signal_ablations.py](../tests/predictors/test_signal_ablations.py)
- generated ablation reports for random replay, spaced replay, fine-tuning, and
  fixed-periodic seed-0 artifacts
- added [SIGNAL_ABLATIONS_TASK17.md](./SIGNAL_ABLATIONS_TASK17.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the next task is a
  learned-predictor replay gate before expensive signals

### Feature groups tested

- `loss_only`
- `uncertainty_only`
- `target_probability_only`
- `anchor_state`
- `history_delta`
- `history_summary`
- `all_features`

Each group is tested with logistic regression and a linear SVM classifier on
the same leakage-safe temporal holdout used by the earlier predictor reports.

### Single-seed diagnostic results

Random replay seed 0:

| Rank | Feature group | Best model | Average precision |
| ---: | --- | --- | ---: |
| 1 | `all_features` | `logistic_regression` | `0.9083240127221096` |
| 2 | `history_summary` | `linear_svm_classifier` | `0.8951623697996727` |
| 3 | `uncertainty_only` | `logistic_regression` | `0.8683205257744547` |
| 4 | `target_probability_only` | `logistic_regression` | `0.8665784039155886` |
| 5 | `loss_only` | `logistic_regression` | `0.8611969396771237` |
| 6 | `anchor_state` | `logistic_regression` | `0.8471253916174554` |
| 7 | `history_delta` | `linear_svm_classifier` | `0.8059718959561837` |

Spaced replay seed 0:

| Rank | Feature group | Best model | Average precision |
| ---: | --- | --- | ---: |
| 1 | `all_features` | `logistic_regression` | `0.9188844673560967` |
| 2 | `history_summary` | `linear_svm_classifier` | `0.9070827647828695` |
| 3 | `target_probability_only` | `linear_svm_classifier` | `0.8504351775572876` |
| 4 | `uncertainty_only` | `logistic_regression` | `0.8500353140223217` |
| 5 | `loss_only` | `logistic_regression` | `0.8429752580361802` |
| 6 | `anchor_state` | `logistic_regression` | `0.8267285181601283` |
| 7 | `history_delta` | `logistic_regression` | `0.8069975088982162` |

Threshold behavior for `all_features` logistic regression:

| Run | Threshold | Selected fraction | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| random replay seed 0 | `0.7` | `0.43967611336032386` | `0.9134438305709024` | `0.5387400434467777` |
| random replay seed 0 | `0.9` | `0.11336032388663968` | `0.9523809523809523` | `0.14482259232440262` |
| spaced replay seed 0 | `0.5` | `0.5781910397295013` | `0.9108187134502924` | `0.6973880597014925` |
| spaced replay seed 0 | `0.7` | `0.34657650042265425` | `0.9390243902439024` | `0.43097014925373134` |

The fine-tuning and fixed-periodic seed-0 reports are generated, but supervised
learned models are skipped because the temporal training split has only one
target class.

### Scientific interpretation

The learned predictor gain is not explained by one isolated current-state
feature. The full cheap feature set is best on both informative replay
artifacts. The strongest compact group is `history_summary`, which means sample
age, previous correctness, correctness history rate, and task progress are
important. That supports the proposal's spacing intuition more than the weak
`history_delta` group does.

Task 17 recommends implementing a learned-predictor risk-gated replay variant
next, using logistic regression over `all_features` with strict temporal
safeguards. This is the right intervention test before adding expensive signals.

### Evidence of completion

- method document: [SIGNAL_ABLATIONS_TASK17.md](./SIGNAL_ABLATIONS_TASK17.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\predictors -q`
- focused verification result: `15 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `70 passed`

### Next task unlocked by this work

Task `18 - Implement a learned-predictor risk-gated replay variant`

## Progress Entry: 2026-04-24

Task: `16 - Add learned forgetting predictors from the proposal: logistic regression, linear models, and support-vector machines`  
Status: `complete`

### What I did

- added [src/predictors/learned_forgetting.py](../src/predictors/learned_forgetting.py)
  with proposal-suggested learned predictor families
- added [scripts/evaluate_learned_forgetting_predictors.py](../scripts/evaluate_learned_forgetting_predictors.py)
  to build `learned_forgetting_predictor_report.json` from a run directory
- extended feature rows with continuous future-deterioration targets from
  `forgetting_labels.json`
- added [tests/predictors/test_learned_forgetting.py](../tests/predictors/test_learned_forgetting.py)
  for binary classifiers, continuous regressors, time-to-forgetting regressors,
  full reports, and JSON saving
- added [LEARNED_FORGETTING_PREDICTORS.md](./LEARNED_FORGETTING_PREDICTORS.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md), [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md),
  and [README.md](../README.md)

### Models implemented

Binary future-forgetting classification:

- `logistic_regression`
- `linear_svm_classifier`

Continuous deterioration and observed-event timing regression:

- `linear_regression`
- `ridge_regression`
- `linear_svm_regressor`
- `constant_train_mean` baseline

The SVM fits record non-fatal fit warnings inside the report rather than
printing noisy warnings to the command line.

### Single-seed diagnostic results

These results use existing seed-0 artifacts and the same temporal split:

```text
train anchors <= task 4
test anchors >= task 5
```

| Run | Best cheap heuristic AP | Best learned binary AP | Best learned model | Learned beats heuristic? |
| --- | ---: | ---: | --- | --- |
| `random_replay_split_cifar100_seed0_signals` | `0.8471253916174554` | `0.9083240127221096` | `logistic_regression` | `true` |
| `spaced_replay_split_cifar100_seed0` | `0.8267285181601283` | `0.9188844673560967` | `logistic_regression` | `true` |
| `fine_tuning_split_cifar100_seed0_signals` | `1.0` | skipped | none | n/a |
| `fixed_periodic_replay_split_cifar100_seed0` | `0.9618410015616607` | skipped | none | n/a |

The fine-tuning and fixed-periodic learned classifiers are skipped because the
training temporal split has only one target class. Those runs are collapse or
near-collapse diagnostics, not useful supervised discrimination tests.

For random replay, the linear SVM classifier is close to logistic regression:

```text
linear_svm_classifier AP = 0.9073454366577642
linear_svm_classifier ROC-AUC = 0.7982654307857817
```

The random-replay continuous and timing regressions also run:

```text
ridge_regression on max_future_loss_increase:
MAE = 0.6791658883584177
RMSE = 0.9195014677415336
R2 = 0.32926868857339087

observed-event time-to-forgetting:
constant_train_mean MAE = 158.73290602286167
ridge_regression MAE = 129.16222919724063
```

The timing result excludes right-censored rows, so it is not yet a validated
online `T_i` estimator.

### Scientific interpretation

Task 16 improves the predictive evidence: on replay artifacts with usable class
balance, learned binary predictors beat the best cheap heuristic by average
precision. Logistic regression is currently the best simple learned binary
predictor.

This does not yet solve the scheduler problem. The learned models are evaluated
offline from completed run artifacts. The next step should run feature ablations
and threshold analysis before using the learned predictor as an online
risk-gated replay mechanism.

### Evidence of completion

- method document: [LEARNED_FORGETTING_PREDICTORS.md](./LEARNED_FORGETTING_PREDICTORS.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\predictors -q`
- focused verification result: `12 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `67 passed`

### Next task unlocked by this work

Task `17 - Run signal ablations and predictor-quality analysis`

## Progress Entry: 2026-04-24

Task: `15 - Implement and test event-triggered risk-gated replay`  
Status: `complete`

### What I did

- extended [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
  with risk-gated modes: `risk_only`, `risk_and_due`, and `risk_or_due`
- added skipped replay opportunity logging through `skipped_rows` in
  `scheduler_trace.json`
- added [src/baselines/risk_gated_replay.py](../src/baselines/risk_gated_replay.py)
  as the event-triggered replay baseline
- added smoke and Split CIFAR-100 configs:
  [risk_gated_replay_smoke.yaml](../configs/experiments/risk_gated_replay_smoke.yaml)
  and
  [risk_gated_replay_split_cifar100.yaml](../configs/experiments/risk_gated_replay_split_cifar100.yaml)
- added tests in
  [tests/baselines/test_risk_gated_replay_baseline.py](../tests/baselines/test_risk_gated_replay_baseline.py)
  and extended scheduler tests
- ran smoke calibration and one-seed Split CIFAR-100 pilots
- added [RISK_GATED_REPLAY_SCHEDULER.md](./RISK_GATED_REPLAY_SCHEDULER.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so Task 16 is now the next step

### Method definition

The new method reuses the online cheap-signal risk score:

```text
loss, uncertainty, 1 - target_probability, loss increase from previous observation
```

The primary Task 15 pilot uses:

```text
scheduler_budget_mode = risk_and_due
risk_threshold = 0.75
min_replay_interval_steps = 1
max_replay_interval_steps = 64
```

A replay sample is selected only if it is both due and above the risk threshold.
If no sample passes the gate, replay is skipped and the skipped opportunity is
logged.

### Smoke calibration

The first smoke threshold, `0.6`, skipped all replay on the tiny fixture. The
smoke config now uses `0.5`, which verifies both selection and skipping:

- final accuracy: `0.5`
- average forgetting: `0.0`
- replay samples: `7`
- replay-augmented batches: `2`
- skipped selection events: `4`

### Split CIFAR-100 one-seed pilots

These runs are diagnostic and should not be treated as repeated-seed evidence.

| Gate | Threshold | Final accuracy | Avg forgetting | Replay samples | Replay batches | Skipped events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `risk_and_due` | `0.50` | `0.05689999999999999` | `0.3831111111111111` | `45216` | `1413` | `0` |
| `risk_and_due` | `0.70` | `0.0438` | `0.4357777777777778` | `11219` | `1406` | `7` |
| `risk_and_due` | `0.75` | `0.046` | `0.43377777777777776` | `2071` | `735` | `678` |
| `risk_only` | `0.75` | `0.0454` | `0.43` | `1871` | `342` | `1071` |

Primary artifact:

```text
experiments/runs/risk_gated_replay/risk_gated_replay_split_cifar100_threshold075_seed0
```

### Scientific interpretation

Task 15 is implemented and auditable. The scheduler can skip replay when no
sample appears near forgetting, and it records skipped opportunities.

The first result is negative: sparse risk-gated replay greatly reduces replay
volume, but it does not preserve retention. At threshold `0.75`, replay samples
drop from the random replay budget of `45216` to `2071`, but final accuracy is
`0.046`, near the fine-tuning seed-0 result of `0.0455`.

This suggests the current cheap online heuristic is too weak as the sole replay
trigger. Task 16 should test the proposal's learned predictors before using a
learned model as the online gate.

### Evidence of completion

- method document: [RISK_GATED_REPLAY_SCHEDULER.md](./RISK_GATED_REPLAY_SCHEDULER.md)
- focused verification command:
  `.\.venv\Scripts\python.exe -m pytest tests\replay\test_spaced_scheduler.py tests\baselines\test_risk_gated_replay_baseline.py -q`
- focused verification result: `5 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `62 passed`

### Next task unlocked by this work

Task `16 - Add learned forgetting predictors from the proposal: logistic regression, linear models, and support-vector machines`

## Progress Entry: 2026-04-24

Task: `Action plan update: event-triggered replay and learned predictors`  
Status: `complete`

### What I did

- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the next implementation task is
  now Task 15, `Implement and test event-triggered risk-gated replay`
- added Task 16, `Add learned forgetting predictors from the proposal:
  logistic regression, linear models, and support-vector machines`
- moved the earlier ablation, expensive-signal, stretch-benchmark, and final
  synthesis tasks back one or more positions
- recorded that the proposal's learned predictor families are feasible in this
  repository because scikit-learn is already a dependency and the project
  already has leakage-safe signal, label, and temporal-split artifacts

### Scientific decision

The next scheduler should test an event-triggered policy:

```text
Only replay a memory item when the current predictor says it is near forgetting;
otherwise skip replay and save compute.
```

This is a sharper test than the current budget-matched spaced replay proxy,
which fills the replay budget even when no sample is clearly due. The new task
should report both retention and efficiency: final accuracy, average
forgetting, total replay samples, replay-sample savings, effective replay ratio,
and training time.

The learned-predictor task follows immediately after because the proposal
explicitly suggests logistic regression, linear models, support-vector machines,
and related predictors. Those models should be evaluated on temporal splits
before being trusted as online scheduler inputs.

### Next task unlocked by this work

Task `15 - Implement and test event-triggered risk-gated replay`

## Progress Entry: 2026-04-24

Task: `14 - Add a stronger replay baseline such as MIR`  
Status: `complete`

### What I did

- added [src/replay/mir.py](../src/replay/mir.py) with ER-MIR MI-1 scoring
- added [src/baselines/mir_replay.py](../src/baselines/mir_replay.py)
- extended [src/replay/buffer.py](../src/replay/buffer.py) with candidate sampling that does not update replay utilization counters
- added [src/experiments/mir_replay_comparison.py](../src/experiments/mir_replay_comparison.py)
  and [scripts/run_mir_replay_comparison.py](../scripts/run_mir_replay_comparison.py)
- added MIR smoke/full configs and three-seed comparison configs
- added tests for parameter restoration, replay-budget accounting, smoke artifacts, config loading, and comparison aggregation
- ran a full three-seed Split CIFAR-100 MIR comparison
- added [MIR_REPLAY_BASELINE.md](./MIR_REPLAY_BASELINE.md)

### Method definition

The implemented MIR score is:

```text
pre_loss_i = CE(f_theta(x_i), y_i)
theta_prime = theta - alpha * grad_theta CE(f_theta(x_current), y_current)
post_loss_i = CE(f_theta_prime(x_i), y_i)
interference_i = post_loss_i - pre_loss_i
```

The baseline samples `128` replay candidates, scores their predicted
interference under a virtual current-batch update, restores the model
parameters, and replays the top `32` candidates. Candidate scoring is selection
overhead; only selected examples count as replay samples.

### Fairness controls

The full comparison used the same Split CIFAR-100 protocol as Task 13:

- protocol: `core_split_cifar100_v2`
- seeds: `0`, `1`, `2`
- fixed task split seed: `0`
- same model, optimizer, batch size, epochs per task, replay capacity, and replay batch size
- replay samples per seed: `45216`

### Aggregate result

| Method | Final accuracy mean | Final accuracy std | Avg forgetting mean | Avg forgetting std | Replay samples mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIR replay | `0.11636666666666667` | `0.0020033305601755646` | `0.2167037037037037` | `0.003425275213477844` | `45216` |

Task 14 deltas relative to Task 13:

- versus random replay: final accuracy `+0.014800000000000021`, average forgetting `-0.08603703703703705`
- versus fixed-periodic replay, `k=1`: final accuracy `+0.014800000000000021`, average forgetting `-0.08603703703703705`
- versus spaced replay due-time proxy: final accuracy `+0.017733333333333337`, average forgetting `-0.09640740740740741`

### Scientific interpretation

MIR is now the strongest implemented baseline in the repository. It improves
retention over random replay and the current spaced-replay due-time proxy under
the same replay sample budget.

This strengthens the evaluation but does not validate the proposal's spacing
mechanism. MIR is interference-aware sample selection, not a cognitive-spacing
`T_i` scheduler. The current evidence says sample choice matters and that the
first spaced scheduler must be improved before the project can claim
spacing-inspired replay helps continual learning.

### Evidence of completion

- comparison document: [MIR_REPLAY_BASELINE.md](./MIR_REPLAY_BASELINE.md)
- full summary artifact: `experiments/task14_mir_replay_comparison/task14_mir_replay_comparison_split_cifar100_summary.json`
- focused verification command: `.\.venv\Scripts\python.exe -m pytest tests\replay\test_mir.py tests\baselines\test_mir_replay_baseline.py tests\experiments\test_mir_replay_comparison.py -q`
- focused verification result: `8 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `59 passed`
- smoke command: `.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_smoke.yaml`
- full command: `.\.venv\Scripts\python.exe scripts\run_mir_replay_comparison.py --config configs\experiments\mir_replay_comparison_split_cifar100.yaml`

### Next task unlocked by this work

Task `15 - Implement and test event-triggered risk-gated replay`

The next task should test whether skipping replay when no sample is predicted
to be near forgetting improves the retention/compute trade-off. MIR should
remain the strong replay reference after any new spaced or risk-gated variant
beats random and fixed-periodic replay.

## Progress Entry: 2026-04-24

Task: `13 - Run the core controlled experiment`  
Status: `complete`

### What I did

- added [src/experiments/core_comparison.py](../src/experiments/core_comparison.py)
  to run and aggregate the four required core methods across a shared seed list
- added [scripts/run_core_comparison.py](../scripts/run_core_comparison.py)
- added [configs/experiments/core_comparison_smoke.yaml](../configs/experiments/core_comparison_smoke.yaml)
  and [configs/experiments/core_comparison_split_cifar100.yaml](../configs/experiments/core_comparison_split_cifar100.yaml)
- added [tests/experiments/test_core_comparison.py](../tests/experiments/test_core_comparison.py)
- updated fixed-periodic replay so `budget_mode = budget_matched` is accepted
- ran the smoke comparison and the full three-seed Split CIFAR-100 comparison
- added [CORE_COMPARISON_TASK13.md](./CORE_COMPARISON_TASK13.md)

### Fairness controls

The full comparison used:

- protocol: `core_split_cifar100_v2`
- seeds: `0`, `1`, `2`
- fixed task split seed: `0`
- same model, optimizer, batch size, epochs per task, replay capacity, and replay batch size
- replay methods all used `45216` replay samples on each seed

Budget-matched fixed-periodic replay used `k = 1`. This makes it fair in replay
volume, but it also makes it effectively equivalent to the current random replay
baseline because both replay uniformly every current-task batch once memory is
available. The earlier `k = 2` fixed-periodic result remains an interval
ablation, not the core budget-matched comparator.

### Aggregate result

| Method | Final accuracy mean | Final accuracy std | Avg forgetting mean | Avg forgetting std | Replay samples mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine-tuning | `0.04703333333333334` | `0.0017897858344878411` | `0.43103703703703705` | `0.004659165046447831` | `0` |
| random replay | `0.10156666666666665` | `0.004206344414492624` | `0.30274074074074075` | `0.0014330031201594854` | `45216` |
| fixed-periodic replay, `k=1` | `0.10156666666666665` | `0.004206344414492624` | `0.30274074074074075` | `0.0014330031201594854` | `45216` |
| spaced replay, due-time proxy | `0.09863333333333334` | `0.003957692930652066` | `0.3131111111111111` | `0.0028043176586942256` | `45216` |

### Scientific interpretation

Replay clearly improves over fine-tuning. The first spaced replay due-time proxy
does not improve over random replay. This is a clean negative result for the
current scheduler version and is consistent with Task 11, where crude
risk-scaled timing heuristics failed to beat a constant median timing baseline.

The project should not claim that cognitive-spacing replay improves continual
learning yet. The immediate next work at that time was either to add a stronger
baseline such as MIR or run ablations; after Task 14, the updated plan is to
test event-triggered risk-gated replay before broader ablations.

### Evidence of completion

- comparison runner: [src/experiments/core_comparison.py](../src/experiments/core_comparison.py)
- comparison document: [CORE_COMPARISON_TASK13.md](./CORE_COMPARISON_TASK13.md)
- full summary artifact: `experiments/task13_core_comparison/task13_core_comparison_split_cifar100_summary.json`
- smoke command: `.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_smoke.yaml`
- full command: `.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_split_cifar100.yaml`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `50 passed`

### Next task unlocked by this work

Task `14 - Add a stronger replay baseline such as MIR, and optionally uncertainty-guided replay`

Given the negative spaced-replay result, future ablations are also scientifically
valuable after the next scheduler and learned-predictor tasks.

## Progress Entry: 2026-04-24

Task: `12 - Implement the spaced replay scheduler driven by risk and estimated T_i proxy`  
Status: `complete`

### What I did

- added [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
  with online sample state, risk scoring, due-step estimation, and scheduler trace rows
- added [src/baselines/spaced_replay.py](../src/baselines/spaced_replay.py)
  as the first spacing-inspired replay method
- added [configs/experiments/spaced_replay_smoke.yaml](../configs/experiments/spaced_replay_smoke.yaml)
  and [configs/experiments/spaced_replay_split_cifar100.yaml](../configs/experiments/spaced_replay_split_cifar100.yaml)
- extended [src/replay/buffer.py](../src/replay/buffer.py) so replay batches can be drawn by explicit sample IDs
- added [tests/replay/test_spaced_scheduler.py](../tests/replay/test_spaced_scheduler.py)
  and [tests/baselines/test_spaced_replay_baseline.py](../tests/baselines/test_spaced_replay_baseline.py)
- added [SPACED_REPLAY_SCHEDULER.md](./SPACED_REPLAY_SCHEDULER.md)
  to document the scheduler and pilot result

### What has been done to the project

The project now has a first implementation of the proposal's replay
intervention. The scheduler observes only online current/replay logits, converts
loss, uncertainty, low target probability, and loss increase into a risk score,
maps that score to an estimated replay interval, and logs each selected replay
sample in `scheduler_trace.json`.

The implementation deliberately calls this an online due-time proxy. It does not
claim that the current heuristic is a validated learned `T_i` estimator.

### Full Split CIFAR-100 single-seed result

Run:

```text
experiments/runs/spaced_replay/spaced_replay_split_cifar100_seed0
```

Measured result:

- final accuracy: `0.0962`
- average forgetting: `0.315`
- training time: `40.18328460000339` seconds
- replay samples: `45216`
- replay-augmented batches: `1413`
- scheduler trace rows: `45216`
- mean estimated forgetting time: `22.7363` optimizer steps
- mean risk score: `0.6549832268742224`

Single-seed interpretation:

- the scheduler is functional and budget-matched to the existing random replay
  pilot in replay sample count
- it is slightly worse than random replay on this seed
- this is not final evidence because Task 13 must repeat methods across seeds
  under locked controls

### Evidence of completion

- scheduler module: [src/replay/spaced_scheduler.py](../src/replay/spaced_scheduler.py)
- spaced replay baseline: [src/baselines/spaced_replay.py](../src/baselines/spaced_replay.py)
- scheduler document: [SPACED_REPLAY_SCHEDULER.md](./SPACED_REPLAY_SCHEDULER.md)
- smoke artifacts: `.tmp/baseline_runs/spaced_replay/spaced_replay_smoke`
- full artifacts: `experiments/runs/spaced_replay/spaced_replay_split_cifar100_seed0`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `46 passed`

### Next task unlocked by this work

Task `13 - Run the core controlled experiment`

The next task should create the fair repeated-seed comparison across
fine-tuning, random replay, fixed-periodic replay, and spaced replay.

## Progress Entry: 2026-04-24

Task: `11 - Define and evaluate sample-specific time-to-forgetting T_i targets`  
Status: `complete`

### What I did

- added [src/signals/time_to_forgetting.py](../src/signals/time_to_forgetting.py)
  to derive time-to-first-observed-forgetting targets from `sample_signals.json`
- added [scripts/build_time_to_forgetting_targets.py](../scripts/build_time_to_forgetting_targets.py)
- added [src/predictors/time_to_forgetting.py](../src/predictors/time_to_forgetting.py)
  to evaluate simple timing heuristics without future-feature leakage
- added [scripts/evaluate_time_to_forgetting.py](../scripts/evaluate_time_to_forgetting.py)
- added tests in [tests/signals/test_time_to_forgetting.py](../tests/signals/test_time_to_forgetting.py)
  and [tests/predictors/test_time_to_forgetting_predictor.py](../tests/predictors/test_time_to_forgetting_predictor.py)
- added [TIME_TO_FORGETTING_TARGETS.md](./TIME_TO_FORGETTING_TARGETS.md)
  to document definitions, censoring, literature grounding, and results

### Definitions chosen

Primary timing target:

```text
first_observed_forgetting_step_delta
```

For each retained anchor:

```text
eligible_for_time_to_forgetting = anchor_correct == true
event_observed = at least one later evaluation for the same sample_id is incorrect
```

The exact forgetting time is interval-censored because evaluation happens only
after tasks. The artifact therefore stores both the last observed correct
checkpoint and the first observed incorrect checkpoint. Samples that remain
correct through the final future evaluation are right-censored.

### Full Split CIFAR-100 single-seed results

| Run | Eligible anchors | Observed events | Right-censored | Best step-delta MAE | Best estimator |
| --- | ---: | ---: | ---: | ---: | --- |
| `fine_tuning_split_cifar100_seed0_signals` | `3938` | `3938` | `0` | `0.5304054054054054` | `constant_train_median` |
| `random_replay_split_cifar100_seed0_signals` | `7497` | `6131` | `1366` | `30.069876900796523` | `constant_train_median` |
| `fixed_periodic_replay_split_cifar100_seed0` | `5095` | `5006` | `89` | `7.298861480075901` | `constant_train_median` |
| `spaced_replay_split_cifar100_seed0` | `7282` | `6026` | `1256` | `30.52126865671642` | `constant_train_median` |

### Scientific interpretation

The target definition is now rigorous enough to support the proposal's timing
question. The first timing result is sobering: crude risk-scaled timing
heuristics do not beat a constant median timing baseline on the pilot runs. That
means the project can proceed with a scheduler, but it must not overclaim that
the current heuristic precisely predicts `T_i`.

### Evidence of completion

- timing target module: [src/signals/time_to_forgetting.py](../src/signals/time_to_forgetting.py)
- timing evaluator: [src/predictors/time_to_forgetting.py](../src/predictors/time_to_forgetting.py)
- timing document: [TIME_TO_FORGETTING_TARGETS.md](./TIME_TO_FORGETTING_TARGETS.md)
- generated artifacts: `time_to_forgetting_targets.json`
- generated reports: `time_to_forgetting_report.json`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `46 passed`

### Next task unlocked by this work

Task `12 - Implement the spaced replay scheduler driven by predicted risk, estimated T_i, and spacing rules`

## Progress Entry: 2026-04-24

Task: `10 - Add the fixed-periodic replay baseline`  
Status: `complete`

### What I did

- added [src/baselines/fixed_periodic_replay.py](../src/baselines/fixed_periodic_replay.py)
  as a fixed-cadence replay baseline
- added [configs/experiments/fixed_periodic_replay_smoke.yaml](../configs/experiments/fixed_periodic_replay_smoke.yaml)
  and [configs/experiments/fixed_periodic_replay_split_cifar100.yaml](../configs/experiments/fixed_periodic_replay_split_cifar100.yaml)
- added [tests/baselines/test_fixed_periodic_replay_baseline.py](../tests/baselines/test_fixed_periodic_replay_baseline.py)
  to verify cadence behavior, artifact fields, and no future-task replay
- added [FIXED_PERIODIC_REPLAY_BASELINE.md](./FIXED_PERIODIC_REPLAY_BASELINE.md)
  to document the method and result
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next task is now
  Task 11, time-to-forgetting target definition

### What has been done to the project

The project now has the fixed schedule comparator required by the proposal. The
baseline reuses the same reservoir replay buffer and task-end insertion policy
as random replay, but samples replay only when:

```text
(global_step + 1) % replay_interval == 0
```

This method is deliberately not forgetting-aware. It does not estimate `T_i`,
does not use risk scores, and does not implement the final spaced replay
scheduler. It exists so the final scheduler can be compared against a fixed
timing rule instead of only against random replay.

### Full Split CIFAR-100 single-seed result

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

Measured result:

- final accuracy: `0.0657`
- average forgetting: `0.4083333333333333`
- training time: `19.60763469999074` seconds
- replay-augmented batches: `707`
- total replay samples: `22624`
- effective replay ratio: `0.45031847133757963`
- signal rows: `127624`

This is a valid Task 10 artifact, but not a final random-vs-fixed conclusion.
The current fixed-periodic run uses fewer replay samples than the current random
replay run, so it should be read as an interval ablation until the final
budget-matched core comparison is configured.

### Evidence of completion

- baseline module: [src/baselines/fixed_periodic_replay.py](../src/baselines/fixed_periodic_replay.py)
- method document: [FIXED_PERIODIC_REPLAY_BASELINE.md](./FIXED_PERIODIC_REPLAY_BASELINE.md)
- smoke artifacts: `.tmp/baseline_runs/fixed_periodic_replay/fixed_periodic_replay_smoke`
- full artifacts: `experiments/runs/fixed_periodic_replay/fixed_periodic_replay_split_cifar100_seed0`
- focused verification command: `.\.venv\Scripts\python.exe -m pytest tests\baselines\test_fixed_periodic_replay_baseline.py -q`
- focused verification result: `3 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `35 passed`

### Next task unlocked by this work

Task `11 - Define and evaluate sample-specific time-to-forgetting T_i targets`

The next implementation should derive first-forgetting task/step deltas and
censoring flags from the signal logs. It should not yet build the spaced replay
scheduler.

## Progress Entry: 2026-04-24

Task: `Proposal coverage assessment after task 9`  
Status: `complete`

### What I did

- reviewed the proposal concepts against the implemented work through Task 9
- recorded that Tasks 1 through 9 are valid pilot and pipeline-validation work,
  but not final proposal-valid evidence
- documented that representation drift, latent drift, gradient norms,
  fixed-periodic replay, explicit `T_i` estimation, and the spaced scheduler are
  still future work
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) and
  [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md) with the rerun decision

### Scientific decision

The project does not need to rerun Tasks 1 through 9 immediately. The right
sequence is to implement Task 10, Task 11, and Task 12 first, then rerun the
controlled core comparison under protocol `core_split_cifar100_v2`.

Current single-seed runs are useful diagnostics because they validate the task
stream, baseline execution, signal logging, label generation, and risk
prediction. They should not be used as final evidence for cognitive-spacing
claims because the project has not yet implemented fixed-periodic replay,
time-to-forgetting estimation, or the spaced scheduler.

### Next task unlocked by this work

Task `10 - Add the fixed-periodic replay baseline: replay uniformly from memory every k optimizer steps`

## Progress Entry: 2026-04-24

Task: `Proposal source extraction and T_i alignment correction`  
Status: `complete`

### What I did

- confirmed that [Project_Proposal-2.tex](../Project_Proposal-2.tex) is readable
- added [PROJECT_PROPOSAL_TEXT.md](./PROJECT_PROPOSAL_TEXT.md), a cleaned Markdown extraction of the proposal text
- updated [CORE_EXPERIMENT_PROTOCOL.md](./CORE_EXPERIMENT_PROTOCOL.md) and [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml) from `core_split_cifar100_v1` to `core_split_cifar100_v2`
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the proposal's `T_i` forgetting-time requirement is explicit
- updated [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md), [FORGETTING_LABEL_DEFINITIONS.md](./FORGETTING_LABEL_DEFINITIONS.md), [FORGETTING_RISK_PREDICTOR.md](./FORGETTING_RISK_PREDICTOR.md), and [README.md](../README.md) to distinguish risk prediction from time-to-forgetting estimation
- updated experiment configs and protocol tests to reference the v2 protocol for future runs

### What changed scientifically

The earlier plan correctly implemented risk prediction, but the proposal is sharper than that. It says the scheduler should estimate a sample-specific forgetting time:

```text
hat(T_i)
```

and schedule the next replay at:

```text
t + hat(T_i)
```

Therefore, risk ranking alone is not enough to claim the proposed spaced-replay mechanism. The project now treats `T_i` estimation as a required stage between fixed-periodic replay and the final spaced scheduler.

### Current implication for the action plan

- Task 10 remains the fixed-periodic replay baseline.
- A new required stage after that defines and evaluates time-to-forgetting `T_i` targets.
- The spaced replay scheduler must log estimated `T_i`, due steps, risk scores, spacing state, and selection reasons.

### Evidence of completion

- extracted proposal text: [PROJECT_PROPOSAL_TEXT.md](./PROJECT_PROPOSAL_TEXT.md)
- protocol doc: [CORE_EXPERIMENT_PROTOCOL.md](./CORE_EXPERIMENT_PROTOCOL.md)
- machine-readable protocol: [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml)
- action plan: [ACTION_PLAN.md](./ACTION_PLAN.md)

## Progress Entry: 2026-04-24

Task: `9 - Build a simple forgetting-risk predictor or heuristic score`  
Status: `complete`

### What I did

- added [src/predictors/forgetting_risk.py](../src/predictors/forgetting_risk.py) with leakage-safe feature construction, heuristic scoring, temporal evaluation, and a lightweight logistic predictor
- added [src/predictors/__init__.py](../src/predictors/__init__.py) to expose predictor APIs
- added [scripts/evaluate_forgetting_risk.py](../scripts/evaluate_forgetting_risk.py) as the CLI for creating `forgetting_risk_report.json`
- added [tests/predictors/test_forgetting_risk.py](../tests/predictors/test_forgetting_risk.py) for pre-anchor feature construction, temporal holdout behavior, heuristic metrics, and logistic fitting
- added [FORGETTING_RISK_PREDICTOR.md](./FORGETTING_RISK_PREDICTOR.md) to document the features, scores, temporal split, and result interpretation
- generated forgetting-risk reports for the smoke and full Split CIFAR-100 signal runs
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 10, the fixed-periodic replay baseline

### What has been done to the project

- the project can now build predictor features from `sample_signals.json` and `forgetting_labels.json`
- features use only signal rows available at or before the anchor, never the future rows that define forgetting labels
- the first heuristic scores are `anchor_loss`, `anchor_uncertainty`, `low_target_probability`, `loss_increase_from_previous`, `target_probability_drop_from_previous`, and `combined_signal`
- the report uses temporal evaluation: earlier anchor tasks for scaling or fitting, later anchor tasks for testing
- average precision is reported as the main metric because forgetting labels can be imbalanced

### Full Split CIFAR-100 single-seed result

These are diagnostics, not final statistical claims.

| Run | Train eligible | Train positives | Test eligible | Test positives | Test positive rate | Best cheap heuristic AP | Logistic AP | Logistic ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `2162` | `2162` | `1776` | `1776` | `1.0` | `1.0` | skipped | skipped |
| `random_replay_split_cifar100_seed0_signals` | `3792` | `3369` | `3705` | `2762` | `0.7454790823211876` | `0.8471253916174554` | `0.9083240127221096` | `0.7978139160228613` |

Interpretation:

- the fine-tuning run is degenerate for prediction because every held-out eligible anchor is later forgotten
- the random replay run is the useful prediction diagnostic because the held-out split has both forgotten and retained examples
- on random replay, cheap signals beat the positive-rate baseline, and the logistic predictor improves average precision further
- this supports the predictive claim enough to continue, but repeated seeds are still required before claiming a stable effect

### Leakage safeguards

- feature rows are built only from `seen_task_eval` rows at or before each label anchor
- temporal holdout uses earlier anchor tasks for scaling/fitting and later anchor tasks for testing
- the report stores source hashes for both `sample_signals.json` and `forgetting_labels.json`
- the implementation keeps prediction separate from replay scheduling, so task 11 cannot accidentally use labels as online scheduler inputs

### Evidence of completion

- predictor module: [src/predictors/forgetting_risk.py](../src/predictors/forgetting_risk.py)
- predictor script: [scripts/evaluate_forgetting_risk.py](../scripts/evaluate_forgetting_risk.py)
- predictor document: [FORGETTING_RISK_PREDICTOR.md](./FORGETTING_RISK_PREDICTOR.md)
- smoke fine-tuning report: `.tmp/baseline_runs/fine_tuning/fine_tuning_smoke/forgetting_risk_report.json`
- smoke random replay report: `.tmp/baseline_runs/random_replay/random_replay_smoke/forgetting_risk_report.json`
- full fine-tuning report: `experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0_signals/forgetting_risk_report.json`
- full random replay report: `experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals/forgetting_risk_report.json`
- focused verification command: `.\.venv\Scripts\python.exe -m pytest tests\predictors\test_forgetting_risk.py -q`
- focused verification result: `4 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `32 passed`

### Next task unlocked by this work

Task `10 - Add the fixed-periodic replay baseline: replay uniformly from memory every k optimizer steps`

The next implementation should add the fixed replay schedule comparator from the proposal before the spaced scheduler is implemented.

## Progress Entry: 2026-04-24

Task: `8 - Define forgetting labels or targets from future performance changes`  
Status: `complete`

### What I did

- added [src/signals/forgetting_labels.py](../src/signals/forgetting_labels.py) to derive future-forgetting labels from `sample_signals.json`
- added [scripts/build_forgetting_labels.py](../scripts/build_forgetting_labels.py) as the CLI for producing `forgetting_labels.json`
- updated [src/signals/__init__.py](../src/signals/__init__.py) to expose the label-generation API
- added [tests/signals/test_forgetting_labels.py](../tests/signals/test_forgetting_labels.py) for primary labels, continuous targets, leakage-shape rejection, and source hash recording
- added [FORGETTING_LABEL_DEFINITIONS.md](./FORGETTING_LABEL_DEFINITIONS.md) to document every label definition and the research rationale
- linked the label definitions from [SIGNAL_LOG_SCHEMA.md](./SIGNAL_LOG_SCHEMA.md)
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 9
- generated forgetting-label artifacts for the smoke and full Split CIFAR-100 signal runs

### Definitions chosen

Primary binary label: `forgot_any_future`

```text
anchor_correct == true
and at least one later seen-task evaluation for the same sample_id is incorrect
```

Secondary binary labels:

- `forgot_next_eval`: correct at the anchor, incorrect at the next later evaluation
- `forgot_final_eval`: correct at the anchor, incorrect at the final available evaluation

Continuous targets:

- `max_future_loss_increase`
- `final_loss_delta`
- `max_future_target_probability_drop`
- `final_target_probability_drop`
- `max_future_confidence_drop`
- `final_confidence_drop`

The artifact also records `eligible_for_binary_forgetting`. A sample-checkpoint is eligible only if it was correct at the anchor. If the model is already wrong, the sample may be difficult or unlearned, but it was not forgotten from a retained state.

### How this directly correlates with the research goal

The proposal asks two questions: whether forgetting can be predicted and whether prediction-guided replay improves continual learning. Task 8 handles the first necessary condition for the prediction question: it defines the target.

The primary label is a sample-level version of standard continual-learning forgetting: a model performs correctly on old knowledge, then later loses that performance after learning more tasks. This keeps the project aligned with the field while giving the later scheduler a sample-level target.

### Leakage safeguards

- labels are built only from `seen_task_eval` rows
- every label has an anchor task id and future rows must satisfy `future.trained_task_id > anchor.trained_task_id`
- rows after the final task are excluded as anchors because they have no future outcome
- `forgetting_labels.json` stores the SHA-256 hash of the source `sample_signals.json`
- task 9 must use only pre-anchor or at-anchor signal history as predictor input

### Generated artifacts

Smoke artifacts:

| Run | Anchor rows | Eligible anchors | `forgot_any_future` positives |
| --- | ---: | ---: | ---: |
| `fine_tuning_smoke` | `24` | `8` | `4` |
| `random_replay_smoke` | `24` | `12` | `0` |

Full Split CIFAR-100 single-seed artifacts:

| Run | Anchor rows | Eligible anchors | `forgot_any_future` positives | Positive rate over eligible | `forgot_next_eval` positives | `forgot_final_eval` positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `45000` | `3938` | `3938` | `1.0` | `3907` | `3938` |
| `random_replay_split_cifar100_seed0_signals` | `45000` | `7497` | `6131` | `0.8177937841803388` | `4035` | `4973` |

These are still single-seed diagnostics, not final statistical claims. They show the label pipeline is producing nontrivial targets and that random replay changes the sample-level forgetting profile.

### Future-proof design decisions

- label generation is post-hoc and does not mutate `sample_signals.json`
- each label row stores both binary and continuous targets so later analyses can compare coarse forgetting events with softer deterioration
- the primary label captures temporary forgetting, while final and next-step labels remain available as diagnostics
- the artifact stores definition text, source hashes, and sanity counts, making it auditable without reading the implementation
- the CLI accepts either a run directory or a direct signal path, so future analysis jobs can operate on copied artifacts

### Evidence of completion

- label module: [src/signals/forgetting_labels.py](../src/signals/forgetting_labels.py)
- label script: [scripts/build_forgetting_labels.py](../scripts/build_forgetting_labels.py)
- definitions document: [FORGETTING_LABEL_DEFINITIONS.md](./FORGETTING_LABEL_DEFINITIONS.md)
- smoke fine-tuning labels: `.tmp/baseline_runs/fine_tuning/fine_tuning_smoke/forgetting_labels.json`
- smoke random replay labels: `.tmp/baseline_runs/random_replay/random_replay_smoke/forgetting_labels.json`
- full fine-tuning labels: `experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0_signals/forgetting_labels.json`
- full random replay labels: `experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals/forgetting_labels.json`
- focused verification command: `.\.venv\Scripts\python.exe -m pytest tests\signals\test_forgetting_labels.py -q`
- focused verification result: `3 passed`
- broader verification command: `.\.venv\Scripts\python.exe -m pytest tests\signals tests\baselines tests\experiment_tracking -q`
- broader verification result: `16 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `28 passed`

### Next task unlocked by this work

Task `9 - Build a simple forgetting-risk predictor or heuristic score`

The next implementation should use only information available at or before each anchor and report precision-recall behavior for `forgot_any_future`.

## Progress Entry: 2026-04-24

Task: `7 - Log the cheapest sample-level signals first: loss history, confidence or uncertainty, replay count, and last replay step`  
Status: `complete`

### What I did

- added [src/signals/sample_signals.py](../src/signals/sample_signals.py) with a schema-versioned sample signal logger
- added [src/signals/__init__.py](../src/signals/__init__.py) to expose signal APIs
- extended [src/training/continual.py](../src/training/continual.py) so the shared trainer can log current-task training rows and seen-task evaluation rows
- extended [src/replay/buffer.py](../src/replay/buffer.py) so replay samples expose `is_replay`, `replay_count`, `last_replay_step`, and split metadata
- extended [src/baselines/fine_tuning.py](../src/baselines/fine_tuning.py) and [src/baselines/random_replay.py](../src/baselines/random_replay.py) so both baselines can save `sample_signals.json`
- extended [src/experiment_tracking/artifacts.py](../src/experiment_tracking/artifacts.py) so run artifacts can include hashed extra JSON artifacts
- updated experiment configs so signal logging is explicit through `log_signals: true`
- added [tests/signals/test_sample_signals.py](../tests/signals/test_sample_signals.py) and strengthened replay, artifact, and baseline tests
- added [SIGNAL_LOG_SCHEMA.md](./SIGNAL_LOG_SCHEMA.md) to document the signal artifact contract
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 8

### What has been done to the project

- the project now records sample-level loss, predicted class, correctness, confidence, target probability, uncertainty, and entropy
- every signal row is keyed by stable `sample_id` and carries task, class, original index, split, and target metadata
- random replay signal rows now include replay count and last replay step
- signal artifacts are saved as manifest-hashed JSON files beside config, metrics, accuracy matrix, train losses, and environment metadata
- signal logging covers `current_train`, `replay_train`, and `seen_task_eval` observations

### Smoke verification result

Run: `fine_tuning_smoke`

- signal rows: `120`
- current-train rows: `72`
- replay-train rows: `0`
- seen-task evaluation rows: `48`
- unique samples observed: `48`

Run: `random_replay_smoke`

- signal rows: `144`
- current-train rows: `72`
- replay-train rows: `24`
- seen-task evaluation rows: `48`
- replay observations: `24`
- unique samples observed: `48`

These are fixture counts for pipeline verification. Full single-seed signal
artifacts were also generated under new run names, listed below.

### How this directly correlates with the research goal

The research question asks whether sample-level signals can predict future forgetting. That question cannot be answered from aggregate accuracy alone. This task creates the raw evidence needed to test the predictive claim without yet building the predictor or scheduler.

The key scientific safeguard is temporal separation: the log records what the model knew at each training or evaluation point, while forgetting labels will be derived later from future changes. This lets the next stage build labels and predictors while checking for future-information leakage.

### Future-proof design decisions

- signal logging is a separate module rather than method-specific inline JSON construction
- the trainer emits observations, but the signal module owns schema and serialization
- extra artifacts are now first-class manifest entries, so future labels, predictor metrics, and scheduler traces can use the same artifact mechanism
- existing fields have documented semantics and a schema version, which gives future expensive signals a safe extension path
- the logger fails fast when stable sample metadata is missing, protecting the project from unusable sample-level evidence

### Evidence of completion

- signal module: [src/signals/sample_signals.py](../src/signals/sample_signals.py)
- schema doc: [SIGNAL_LOG_SCHEMA.md](./SIGNAL_LOG_SCHEMA.md)
- fine-tuning smoke artifact: `.tmp/baseline_runs/fine_tuning/fine_tuning_smoke/sample_signals.json`
- random replay smoke artifact: `.tmp/baseline_runs/random_replay/random_replay_smoke/sample_signals.json`
- focused verification command: `.\.venv\Scripts\python.exe -m pytest tests\signals\test_sample_signals.py tests\replay\test_replay_buffer.py tests\experiment_tracking\test_artifacts.py -q`
- focused verification result: `10 passed`
- baseline verification command: `.\.venv\Scripts\python.exe -m pytest tests\baselines\test_fine_tuning_baseline.py tests\baselines\test_random_replay_baseline.py -q`
- baseline verification result: `5 passed`
- full verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- full verification result: `25 passed`
- full fine-tuning signal artifact: `experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0_signals/sample_signals.json`
- full fine-tuning signal rows: `105000`
- full random replay signal artifact: `experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals/sample_signals.json`
- full random replay signal rows: `150216`

### Full Split CIFAR-100 signal run check

The task-7 full signal runs used new run names so the earlier baseline artifacts were not overwritten.

| Run | Final accuracy | Average forgetting | Signal rows | Current-train rows | Replay-train rows | Seen-task eval rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fine_tuning_split_cifar100_seed0_signals` | `0.0455` | `0.434` | `105000` | `50000` | `0` | `55000` |
| `random_replay_split_cifar100_seed0_signals` | `0.10129999999999999` | `0.30433333333333334` | `150216` | `50000` | `45216` | `55000` |

The metrics match the earlier single-seed baseline results, so the signal instrumentation did not change the measured baseline behavior.

### Next task unlocked by this work

Task `8 - Define forgetting labels or targets from future performance changes`

The next implementation should read `sample_signals.json` and produce an auditable forgetting-label artifact without using future information as predictor input.

## Progress Entry: 2026-04-24

Task: `6 - Add a bounded replay buffer and the random replay baseline`  
Status: `complete`

### What I did

- added [src/replay/buffer.py](../src/replay/buffer.py) with a bounded reservoir replay buffer
- added [src/replay/__init__.py](../src/replay/__init__.py) to expose replay APIs
- added [src/baselines/random_replay.py](../src/baselines/random_replay.py) as the bounded random replay baseline entry point
- added [configs/experiments/random_replay_smoke.yaml](../configs/experiments/random_replay_smoke.yaml) for offline replay verification
- added [configs/experiments/random_replay_split_cifar100.yaml](../configs/experiments/random_replay_split_cifar100.yaml) for the real Split CIFAR-100 random replay baseline
- added [tests/replay/test_replay_buffer.py](../tests/replay/test_replay_buffer.py) for buffer capacity and replay utilization behavior
- added [tests/baselines/test_random_replay_baseline.py](../tests/baselines/test_random_replay_baseline.py) for replay baseline artifact checks
- extended [src/training/continual.py](../src/training/continual.py) so training results can carry method-specific metrics
- extended [src/experiment_tracking/artifacts.py](../src/experiment_tracking/artifacts.py) so replay metrics are validated and saved with run artifacts
- ran the offline random replay smoke baseline and the real Split CIFAR-100 random replay baseline
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 7

### What has been done to the project

- the project now has the required standard replay comparator
- the replay buffer enforces a fixed capacity and uses seeded reservoir insertion
- replay sampling is uniform random under a fixed replay seed
- replay examples preserve stable sample metadata needed for later replay analysis
- random replay uses the same task order, model family, target key, optimizer family, and evaluation schedule as the fine-tuning baseline
- replay utilization is saved in the artifact metrics

### Baseline result

Run: `random_replay_split_cifar100_seed0`  
Method: `random_replay`  
Replay buffer capacity: `2000`  
Replay batch size: `32`  
Replay insertion policy: `reservoir_task_end`  
Replay sampling policy: `uniform_random`  
Target: `original_class_id`  
Tasks: `10`  
Classes per task: `10`  
Epochs per task: `1`  
Model: `flatten_mlp` with `812388` trainable parameters

Measured result:

- final accuracy: `0.10129999999999999`
- average forgetting: `0.30433333333333334`
- training time: `19.213716600002954` seconds
- train loss count: `1570`
- replay-augmented batches: `1413`
- total replay samples: `45216`
- final buffer size: `2000`
- unique replayed buffer samples: `1773`
- buffer samples never replayed by the end of the run: `227`

Accuracy matrix:

```text
[
  [0.442, null,  null,  null,  null,  null,  null,  null,  null,  null],
  [0.468, 0.241, null,  null,  null,  null,  null,  null,  null,  null],
  [0.289, 0.220, 0.343, null,  null,  null,  null,  null,  null,  null],
  [0.277, 0.190, 0.142, 0.271, null,  null,  null,  null,  null,  null],
  [0.174, 0.167, 0.095, 0.046, 0.427, null,  null,  null,  null,  null],
  [0.138, 0.131, 0.088, 0.083, 0.093, 0.369, null,  null,  null,  null],
  [0.140, 0.135, 0.099, 0.020, 0.065, 0.030, 0.369, null,  null,  null],
  [0.119, 0.106, 0.080, 0.024, 0.068, 0.062, 0.030, 0.496, null,  null],
  [0.100, 0.123, 0.072, 0.048, 0.102, 0.040, 0.026, 0.094, 0.355, null],
  [0.093, 0.145, 0.070, 0.023, 0.082, 0.039, 0.035, 0.107, 0.006, 0.413],
]
```

### Comparison to fine-tuning

Single-seed comparison against the task-5 fine-tuning run:

| Method | Final accuracy | Average forgetting |
| --- | --- | --- |
| fine-tuning | `0.0455` | `0.434` |
| random replay | `0.10129999999999999` | `0.30433333333333334` |

This is not yet a final statistical claim because it is one seed, but it is a valid first controlled comparator. Random replay improves retention over no replay under the current setup and therefore gives the future spaced replay method a meaningful baseline to beat.

### How this directly correlates with the research goal

The research goal asks whether a forgetting-aware spaced replay policy improves continual learning. That claim is not credible unless the project first compares against ordinary random replay under a fixed memory and replay budget.

This task establishes that comparator and records replay utilization. Future spaced replay must use the same model, task order, replay capacity, replay batch size, target key, and evaluation schedule before any improvement can be attributed to the scheduler.

### Future-proof design decisions

- replay storage is metadata-rich enough to support future signal logging and spaced scheduling
- replay insertion and sampling policies are named in the run config
- replay metrics are saved as method-specific artifact metrics instead of being hidden in console output
- current-task examples are inserted after a task completes, so replay during a task only uses earlier-task memory
- the random replay baseline is a separate entry point from fine-tuning, reducing the risk that later methods accidentally change the no-replay baseline

### Evidence of completion

- replay buffer: [src/replay/buffer.py](../src/replay/buffer.py)
- random replay baseline: [src/baselines/random_replay.py](../src/baselines/random_replay.py)
- smoke config: [configs/experiments/random_replay_smoke.yaml](../configs/experiments/random_replay_smoke.yaml)
- real Split CIFAR-100 config: [configs/experiments/random_replay_split_cifar100.yaml](../configs/experiments/random_replay_split_cifar100.yaml)
- replay tests: [tests/replay/test_replay_buffer.py](../tests/replay/test_replay_buffer.py)
- baseline tests: [tests/baselines/test_random_replay_baseline.py](../tests/baselines/test_random_replay_baseline.py)
- full baseline artifacts: `experiments/runs/random_replay/random_replay_split_cifar100_seed0`
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `23 passed`
- artifact integrity check: `load_experiment_artifacts('experiments/runs/random_replay/random_replay_split_cifar100_seed0')`

### Next task unlocked by this work

Task `7 - Log the cheapest sample-level signals first: loss history, confidence or uncertainty, replay count, and last replay step`

The next implementation should add a sample-level signal log keyed by `sample_id` and record loss, predicted class, confidence, uncertainty, replay count, and last replay step.

## Progress Entry: 2026-04-24

Task: `Dataset inspection and documentation for Split CIFAR-100`  
Status: `complete`

### What I did

- added [scripts/inspect_cifar100.py](../scripts/inspect_cifar100.py) to inspect the raw CIFAR-100 pickle files directly
- generated local inspection artifacts under `.tmp/dataset_preview/cifar100`
- added [docs/DATASET_CIFAR100.md](./DATASET_CIFAR100.md) with dataset facts, sample-ID rules, task composition, preprocessing notes, and research caveats
- linked the dataset document from [README.md](../README.md)

### What has been done to the project

- the project now has a reproducible way to inspect raw CIFAR-100 without training transforms
- the inspection script writes `summary.json`, `task_class_order.csv`, and `train_preview_grid.png`
- dataset counts, sample-ID offsets, class order, and task composition are documented
- the dataset documentation now explains how this benchmark supports the forgetting and replay research question

### Evidence of completion

- inspection script: [scripts/inspect_cifar100.py](../scripts/inspect_cifar100.py)
- dataset document: [docs/DATASET_CIFAR100.md](./DATASET_CIFAR100.md)
- inspection command: `.\.venv\Scripts\python.exe scripts\inspect_cifar100.py --data-root data --output-dir .tmp\dataset_preview\cifar100 --task-count 10 --classes-per-task 10 --split-seed 0`
- inspection result: train examples `50000`, test examples `10000`, fine classes `100`, coarse classes `20`

## Progress Entry: 2026-04-24

Task: `5 - Run the naive sequential fine-tuning baseline`  
Status: `complete`

### What I did

- added [src/models/simple.py](../src/models/simple.py) with a lightweight flattening MLP model factory
- added [src/models/__init__.py](../src/models/__init__.py) to expose model helpers
- added [src/baselines/fine_tuning.py](../src/baselines/fine_tuning.py) as the no-replay baseline entry point
- added [src/baselines/__init__.py](../src/baselines/__init__.py) for the baseline package
- added [configs/experiments/fine_tuning_smoke.yaml](../configs/experiments/fine_tuning_smoke.yaml) for an offline smoke baseline
- added [configs/experiments/fine_tuning_split_cifar100.yaml](../configs/experiments/fine_tuning_split_cifar100.yaml) for the real Split CIFAR-100 baseline
- added [tests/baselines/test_fine_tuning_baseline.py](../tests/baselines/test_fine_tuning_baseline.py) for smoke-run artifacts, config loading, and clean missing-data failure
- added `experiments/runs/` to [.gitignore](../.gitignore) so generated experiment artifacts are not accidentally committed
- ran the offline smoke baseline and the real Split CIFAR-100 fine-tuning baseline
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 6

### What has been done to the project

- the repository now has an executable no-replay fine-tuning baseline
- the baseline uses `original_class_id` targets by default, giving a class-incremental no-replay reference point
- the baseline saves complete artifacts through the experiment tracking layer
- the real Split CIFAR-100 dataset was downloaded into `data/` after explicit approval and used for the full baseline run
- the full baseline artifact bundle is saved under `experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0`
- a smoke baseline remains available for fast offline verification without requiring CIFAR-100

### Baseline result

Run: `fine_tuning_split_cifar100_seed0`  
Method: `fine_tuning`  
Replay: disabled  
Target: `original_class_id`  
Tasks: `10`  
Classes per task: `10`  
Epochs per task: `1`  
Model: `flatten_mlp` with `812388` trainable parameters

Measured result:

- final accuracy: `0.0455`
- average forgetting: `0.434`
- training time: `11.997672299999977` seconds
- train loss count: `1570`

Accuracy matrix:

```text
[
  [0.442, null,  null,  null,  null,  null,  null,  null,  null,  null],
  [0.013, 0.375, null,  null,  null,  null,  null,  null,  null,  null],
  [0.000, 0.000, 0.446, null,  null,  null,  null,  null,  null,  null],
  [0.000, 0.000, 0.007, 0.370, null,  null,  null,  null,  null,  null],
  [0.000, 0.000, 0.006, 0.000, 0.503, null,  null,  null,  null,  null],
  [0.000, 0.000, 0.000, 0.000, 0.000, 0.405, null,  null,  null,  null],
  [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.443, null,  null,  null],
  [0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.001, 0.541, null,  null],
  [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.381, null],
  [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.455],
]
```

### How this directly correlates with the research goal

The research goal asks whether replay can reduce catastrophic forgetting. This task establishes the no-replay condition against which replay must be judged.

The result shows severe forgetting: final accuracy falls to `0.0455`, and most earlier task accuracies collapse toward zero after later tasks. This confirms that the selected Split CIFAR-100 setup produces the forgetting problem the project is meant to study.

### Future-proof design decisions

- the fine-tuning baseline is an explicit method entry point rather than a special case hidden in the trainer
- smoke and real-data paths share the same trainer, metrics, and artifact writer
- the baseline refuses non-`original_class_id` targets unless changed deliberately, reducing the risk of accidentally switching evaluation regimes
- missing real data now fails with a clear `BaselineDataUnavailableError`
- generated artifacts are ignored by git while still being reproducible from configs and code

### Evidence of completion

- baseline module: [src/baselines/fine_tuning.py](../src/baselines/fine_tuning.py)
- model module: [src/models/simple.py](../src/models/simple.py)
- smoke config: [configs/experiments/fine_tuning_smoke.yaml](../configs/experiments/fine_tuning_smoke.yaml)
- real Split CIFAR-100 config: [configs/experiments/fine_tuning_split_cifar100.yaml](../configs/experiments/fine_tuning_split_cifar100.yaml)
- regression tests: [tests/baselines/test_fine_tuning_baseline.py](../tests/baselines/test_fine_tuning_baseline.py)
- full baseline artifacts: `experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0`
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `19 passed`
- artifact integrity check: `load_experiment_artifacts('experiments/runs/fine_tuning/fine_tuning_split_cifar100_seed0')`

### Next task unlocked by this work

Task `6 - Add a bounded replay buffer and the random replay baseline`

The next implementation should add replay-buffer capacity control, random replay sampling, replay utilization logging, and a matched-budget random replay baseline.

## Progress Entry: 2026-04-24

Task: `4 - Add reproducible metrics and experiment logging`  
Status: `complete`

### What I did

- added [src/experiment_tracking/artifacts.py](../src/experiment_tracking/artifacts.py) with schema-versioned experiment artifact writing and loading
- added [src/experiment_tracking/__init__.py](../src/experiment_tracking/__init__.py) to expose the artifact API
- added [configs/experiments/core_fine_tuning_smoke.yaml](../configs/experiments/core_fine_tuning_smoke.yaml) as the first experiment-level config stub
- added [tests/experiment_tracking/test_artifacts.py](../tests/experiment_tracking/test_artifacts.py) for round-trip artifact tests, overwrite protection, accuracy-matrix validation, and tamper detection
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 5

### What has been done to the project

- run artifacts can now save the run config, summary metrics, accuracy matrix, train losses, environment snapshot, and manifest
- artifact bundles are schema-versioned through `ARTIFACT_SCHEMA_VERSION`
- summary metrics are derived from the same accuracy-matrix definitions used by the trainer and metrics modules
- saved artifacts include hashes in `manifest.json`, and loading can verify that files have not been modified after writing
- artifact writing refuses to overwrite an existing run directory unless explicitly allowed
- invalid accuracy matrices are rejected before they can become experiment evidence

### How this directly correlates with the research goal

The research goal depends on fair comparison between fine-tuning, random replay, and spaced replay under matched controls. That comparison becomes scientifically weak if runs cannot be reconstructed or if metrics are computed differently across methods.

This task makes every future run leave a reproducible evidence trail before baseline numbers begin. It also creates fail-safes against two dangerous research mistakes: accidentally including future-task accuracy in a seen-task matrix and silently overwriting or modifying run results.

### Future-proof design decisions

- artifacts are JSON-first, which keeps them readable by humans, notebooks, scripts, and future analysis tools
- the run config schema stores dataset, model, trainer, evaluation, method, replay, signal, and task-split sections even though replay and signal methods are not implemented yet
- environment snapshots record Python, platform, PyTorch, CUDA availability, and git metadata when available
- manifests store file hashes so later analysis can detect accidental edits or stale copied results
- tests use repo-local `.tmp` artifacts because the Windows system temp directory can be permission-restricted in this environment

### Evidence of completion

- artifact module: [src/experiment_tracking/artifacts.py](../src/experiment_tracking/artifacts.py)
- experiment config stub: [configs/experiments/core_fine_tuning_smoke.yaml](../configs/experiments/core_fine_tuning_smoke.yaml)
- regression tests: [tests/experiment_tracking/test_artifacts.py](../tests/experiment_tracking/test_artifacts.py)
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `16 passed`

### Next task unlocked by this work

Task `5 - Run the naive sequential fine-tuning baseline`

The next implementation should add the minimal model and experiment entry point needed to run a no-replay baseline, save its artifacts through the new tracking layer, and establish the first measured reference point for catastrophic forgetting.

## Progress Entry: 2026-04-24

Task: `3 - Build the core continual training and evaluation loop`  
Status: `complete`

### What I did

- added [src/training/continual.py](../src/training/continual.py) with a method-agnostic sequential training loop
- added [src/training/__init__.py](../src/training/__init__.py) to expose the training API
- added [src/metrics/continual.py](../src/metrics/continual.py) with accuracy-matrix metric helpers
- added [src/metrics/__init__.py](../src/metrics/__init__.py) to expose the metrics API
- added [tests/training/test_continual_training.py](../tests/training/test_continual_training.py) with a tiny in-memory smoke run
- added [tests/metrics/test_continual_metrics.py](../tests/metrics/test_continual_metrics.py) for final accuracy and average forgetting definitions
- updated [ACTION_PLAN.md](./ACTION_PLAN.md) so the immediate next implementation step now points to task 4

### What has been done to the project

- the project can now train sequentially over task datasets produced by the task-stream layer
- the trainer evaluates every seen task after each completed task
- the trainer emits a lower-triangular accuracy matrix where unseen future tasks remain `None`
- the trainer supports CPU execution and an `auto` device mode that uses CUDA when available
- the training target is configurable through `target_key`, so later experiments can choose task-local labels or original class labels explicitly
- metric helpers now compute average accuracy after a task, final accuracy, per-task forgetting, and average forgetting from the same accuracy matrix contract

### How this directly correlates with the research goal

The research goal requires comparing fine-tuning, random replay, and spaced replay under the same sequential evaluation protocol. That comparison is impossible without one shared training loop and one shared evaluation format.

This task creates the common experimental spine. Future replay methods should plug into this structure instead of inventing their own evaluation path, which reduces the risk that method differences are actually measurement differences.

### Future-proof design decisions

- the loop is method-agnostic and contains no replay, signal, or scheduler assumptions
- model outputs may be tensors, dictionaries with `logits`, or tuple/list outputs, which leaves room for later model wrappers
- unseen task entries are represented as `None`, which makes it harder for metrics to accidentally include tasks that should not be evaluated yet
- the trainer accepts separate train and evaluation streams, so later work can use train/test splits without changing the accuracy-matrix contract
- metric formulas live outside the trainer, which keeps future logging and reporting code from duplicating definitions

### Evidence of completion

- training loop: [src/training/continual.py](../src/training/continual.py)
- metrics: [src/metrics/continual.py](../src/metrics/continual.py)
- training test: [tests/training/test_continual_training.py](../tests/training/test_continual_training.py)
- metrics test: [tests/metrics/test_continual_metrics.py](../tests/metrics/test_continual_metrics.py)
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `10 passed`

### Next task unlocked by this work

Task `4 - Add reproducible metrics and experiment logging`

The next implementation should save run configs, accuracy matrices, losses, timing, and summary metrics in a predictable artifact format before baseline experiments begin.

## Progress Entry: 2026-04-24

Task: `2 - Implement the Split CIFAR-100 task stream with stable sample IDs`  
Status: `complete`

### What I did

- added [src/data/split_cifar100.py](../src/data/split_cifar100.py) with deterministic Split CIFAR-100 task-stream construction
- added [src/data/__init__.py](../src/data/__init__.py) to expose the data module API
- added [configs/datasets/split_cifar100.yaml](../configs/datasets/split_cifar100.yaml) for the first dataset-level configuration values
- added [tests/data/test_split_cifar100_task_stream.py](../tests/data/test_split_cifar100_task_stream.py) with offline fixture-based tests

### What has been done to the project

- the project can now construct class-incremental task specs for Split CIFAR-100-style targets
- task construction is deterministic under a fixed split seed
- each task exposes disjoint original class IDs
- each task dataset returns stable metadata: `sample_id`, task id, original class id, within-task label, original dataset index, and split name
- train and test splits can use different sample-ID offsets so their IDs do not collide
- unit tests do not require CIFAR-100 downloads or network access

### How this directly correlates with the research goal

The research question depends on tracking which examples become risky, forgotten, replayed, or retained over time. That is impossible without stable sample identity.

This task creates the identity layer needed for later signal logging, forgetting-label construction, replay-buffer metadata, and scheduler analysis. It also prevents a common failure mode where the model appears to have sample-level telemetry but the IDs cannot actually be traced across dataloader shuffling, replay, and evaluation.

### Future-proof design decisions

- the task-stream builder works with any CIFAR-style dataset exposing `targets`, which keeps tests lightweight and leaves room for fixtures or alternate vision datasets
- the real CIFAR-100 loader is isolated behind `load_cifar100_dataset`, so unit tests do not need `torchvision` downloads
- task specs can be built from an explicit `class_order` for audited runs or from a seeded shuffle for normal experiments
- sample metadata is returned with every item, so future trainers, loggers, replay buffers, and predictors can consume the same identity contract
- dataset configuration has been separated from the protocol manifest so implementation details can evolve without changing the locked research contract

### Evidence of completion

- data module: [src/data/split_cifar100.py](../src/data/split_cifar100.py)
- dataset config: [configs/datasets/split_cifar100.yaml](../configs/datasets/split_cifar100.yaml)
- regression tests: [tests/data/test_split_cifar100_task_stream.py](../tests/data/test_split_cifar100_task_stream.py)
- verification command: `.\.venv\Scripts\python.exe -m pytest -q`
- verification result: `8 passed`

### Next task unlocked by this work

Task `3 - Build the core continual training and evaluation loop`

The next implementation should consume these task datasets, train sequentially over tasks, evaluate all seen tasks after each task, and emit the accuracy matrix required for final accuracy and average forgetting.

## Progress Entry: 2026-04-24

Task: `Action plan research hardening`  
Status: `complete`

### What I did

- reviewed [ACTION_PLAN.md](./ACTION_PLAN.md) against the locked research question
- added a plan-quality verdict that identifies execution rigor as the main risk
- separated the proposal into two testable claims: prediction of forgetting risk and intervention through replay scheduling
- added recommended implementation module boundaries for data, training, replay, signals, predictors, metrics, logging, and utilities
- added phase exit checkpoints so later work advances only when evidence artifacts exist
- strengthened the immediate next step by requiring offline-testable Split CIFAR-100 task-stream construction

### Why this was needed now

The project had already locked its scope, but the action plan still needed more detail about how evidence should accumulate. Without that detail, the team could build a spaced replay feature before proving that the benchmark, baselines, metrics, and sample-level logs are trustworthy.

### How this strengthens the research

The project now has clearer guardrails for the two central questions:

- whether sample-level signals predict future forgetting
- whether using that prediction improves continual learning under matched budgets

This makes it harder for future results to overclaim. A failed predictor, a failed scheduler, or a mixed result can now be interpreted cleanly instead of being blurred into one vague success or failure.

### Next task unlocked by this work

Task `2 - Implement the Split CIFAR-100 task stream with stable sample IDs`

The next implementation should focus on deterministic task construction, stable sample metadata, and tests that can run without downloading CIFAR-100.

## Progress Entry: 2026-04-23

Task: `1 - Lock the scope, benchmark, and evaluation protocol`  
Status: `complete`

### What I did

- created a locked human-readable protocol in [CORE_EXPERIMENT_PROTOCOL.md](./CORE_EXPERIMENT_PROTOCOL.md)
- created a machine-readable protocol manifest in [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml)
- added protocol loading and validation logic in [src/research_protocol.py](../src/research_protocol.py)
- updated [src/main.py](../src/main.py) so protocol files can be loaded and summarized instead of only being treated as placeholders
- updated [README.md](../README.md) and [experiments/README.md](../experiments/README.md) so repository guidance matches the locked scope
- added [tests/test_research_protocol.py](../tests/test_research_protocol.py) to keep the protocol stable as the repository grows

### What has been done to the project

- the project now has one explicit core experiment contract instead of several partially overlapping descriptions
- the required benchmark is now formally locked to Split CIFAR-100 for the first valid study
- the required comparison set is now formally locked to fine-tuning, random replay, and spaced replay
- the required metrics are now formally locked to final accuracy and average forgetting
- stretch work such as Split CUB, DistilBERT, gradient norms, and representation drift is now recorded without being allowed to overtake the main path

### How this directly correlates with the research goal

The research goal is to evaluate whether spacing-inspired replay reduces catastrophic forgetting under limited memory and compute. That claim only becomes trustworthy if the project compares methods under the same benchmark and the same evaluation rules.

This task directly supports that goal by making the benchmark, core methods, controls, and required metrics explicit before more implementation begins. In practical terms, it reduces the risk that future code answers a different question than the one the proposal set out to study.

### Future-proof design decisions

- the protocol is versioned, so later expansions can be added without overwriting the original study definition
- the protocol exists in YAML and Markdown, so both humans and future code can rely on the same source of truth
- required, recommended, and stretch items are separated, which makes it safer to add new baselines or datasets later
- validation logic was added now so future experiment code can consume the protocol without silently accepting incomplete or conflicting settings
- repository docs were updated together, which avoids future ambiguity caused by stale instructions

### Evidence of completion

- locked protocol doc: [docs/CORE_EXPERIMENT_PROTOCOL.md](./CORE_EXPERIMENT_PROTOCOL.md)
- machine-readable contract: [configs/protocols/core_experiment.yaml](../configs/protocols/core_experiment.yaml)
- validation module: [src/research_protocol.py](../src/research_protocol.py)
- regression test: [tests/test_research_protocol.py](../tests/test_research_protocol.py)

### Next task unlocked by this work

Task `2 - Implement the Split CIFAR-100 task stream with stable sample IDs`

This next task is now safer to implement because the benchmark, task granularity, and required evaluation outputs are no longer ambiguous.
