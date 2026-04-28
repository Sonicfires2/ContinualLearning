# Task 22 Decision Checkpoint and Rescue Ablations

## Purpose

Task 22 asks whether the project should keep trying small learned-risk replay
variants, pivot toward a different signal, or stop with a clean negative result.

The specific rescue question was:

```text
If pure learned-risk replay over-focuses on risky examples, does lowering the
learned-risk fraction and protecting diversity rescue performance?
```

## What Was Run

All runs use the same Split CIFAR-100 seed-0 controls as the prior learned
replay pilots:

- replay capacity: `2000`
- replay batch size: `32`
- replay samples: `45216`
- prior predictor artifact:
  `experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals`
- learned predictor: logistic regression over `all_features`

Commands:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_task22_frac025_class_balanced_split_cifar100.yaml
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_task22_frac025_random_split_cifar100.yaml
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_task22_class_balanced_only_split_cifar100.yaml
```

## Results

These are seed-0 rescue ablations, not repeated-seed final claims.

| Method, seed 0 | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid, 50% learned-risk + 50% class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| Task 22, 25% learned-risk + 75% random | `0.0879` | `0.33144444444444443` | `45216` |
| Task 22, 25% learned-risk + 75% class-balanced | `0.0986` | `0.3268888888888889` | `45216` |
| Task 22, class-balanced only | `0.10500000000000001` | `0.2962222222222222` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

## Predictor Accuracy Right Now

The current learned forgetting predictor is still good as an offline predictor.
It predicts sample-level future forgetting, not whole-class forgetting.

For the random replay seed-0 artifact:

| Metric | Value |
| --- | ---: |
| held-out rows | `3705` |
| held-out positive rate | `0.7454790823211875` |
| best model | `logistic_regression` |
| average precision | `0.9083240127221096` |
| ROC-AUC | `0.7978139160228613` |
| precision in top 10% highest-risk samples | `0.954177897574124` |
| precision in top 20% highest-risk samples | `0.9473684210526315` |
| precision in top 30% highest-risk samples | `0.9316546762589928` |

Threshold behavior for the same logistic predictor:

| Probability threshold | Selected fraction | Precision | Recall | Approx. accuracy |
| ---: | ---: | ---: | ---: | ---: |
| `0.3` | `0.7889338731443994` | `0.8450222374273008` | `0.8942795076031861` | `0.7989203778677463` |
| `0.5` | `0.6402159244264507` | `0.8827993254637436` | `0.7581462708182476` | `0.7446693657219973` |
| `0.7` | `0.43967611336032386` | `0.9134438305709024` | `0.5387400434467777` | `0.6180836707152497` |
| `0.9` | `0.11336032388663968` | `0.9523809523809523` | `0.14482259232440262` | `0.357085020242915` |

Plain interpretation:

```text
The predictor is good at ranking examples by future forgetting risk.
But a good ranking of future risk still has not made a better replay policy.
```

Average precision is the most useful metric here because many held-out rows are
positive. Plain accuracy can be misleading: predicting "will forget" for
everything would already get about `0.745` accuracy on this held-out split.

## Interpretation

Task 22 mostly confirms the previous diagnosis.

Lowering the learned-risk fraction from `0.50` to `0.25` helps when the other
`0.75` is class-balanced, but it still does not clearly beat random replay:

```text
0.0879 -> 0.0986 final accuracy
0.3428888888888889 -> 0.3268888888888889 average forgetting
```

The pure class-balanced run is the only Task 22 rescue ablation that beats
random replay on seed 0:

```text
final accuracy: 0.1013 random -> 0.1050 class-balanced
average forgetting: 0.3043 random -> 0.2962 class-balanced
```

That is useful, but it is not evidence that the learned forgetting predictor is
helping replay. It points in the other direction:

```text
diversity/class coverage helps more than the current learned-risk score
```

MIR still remains the strongest implemented method:

```text
MIR final accuracy: 0.1183
MIR average forgetting: 0.2140
```

## Decision

The project should not spend more time on minor learned-risk fraction tuning.

Recommended next step:

1. Treat class-balanced replay as a useful stronger non-learned replay baseline.
2. If more method work is required, pivot to Task 23:
   MIR-like current-interference or representation-drift diagnostics.
3. If a final report is the goal, move toward Task 25 and write the result as:

```text
Offline forgetting prediction works, but the tested learned-risk replay
interventions do not beat random replay; simple diversity protection is more
useful than the learned-risk score in the current implementation.
```

