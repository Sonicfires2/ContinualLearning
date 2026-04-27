# Task 20 Learned Hybrid Replay

## Purpose

Task 20 tests whether learned forgetting risk helps once replay diversity is
protected.

Task 19 showed that pure learned-risk ranking was worse than random replay even
with the same replay budget. A likely explanation was over-focus: the learned
selector may repeatedly choose hard or unstable examples instead of keeping a
broad memory of old tasks.

Task 20 therefore asks:

```text
Can learned risk help if only part of the replay batch is risk-ranked and the
rest is random or class-balanced?
```

## Implementation

Main code:

- [src/baselines/learned_hybrid_replay.py](../src/baselines/learned_hybrid_replay.py)
- [configs/experiments/learned_hybrid_replay_smoke.yaml](../configs/experiments/learned_hybrid_replay_smoke.yaml)
- [configs/experiments/learned_hybrid_replay_split_cifar100.yaml](../configs/experiments/learned_hybrid_replay_split_cifar100.yaml)
- [tests/baselines/test_learned_hybrid_replay.py](../tests/baselines/test_learned_hybrid_replay.py)

The implemented selector fills each replay batch with:

```text
50% learned-risk-ranked examples
50% class-balanced diversity examples
```

The diversity half is selected across classes where possible. The code also
supports `hybrid_diversity_mode: random` for a later random-diversity ablation.

The Split CIFAR-100 pilot uses the same replay budget and training settings as
random replay and Task 19:

```text
1413 replay batches * 32 examples = 45216 replay samples
```

## Leakage Guard

The active hybrid run does not train its predictor from its own future labels.
It uses the same prior random-replay seed-0 source artifacts used by Tasks 18
and 19:

```text
experiments/runs/random_replay/random_replay_split_cifar100_seed0_signals
```

The saved run config records the source predictor artifact metadata.

## Smoke Result

Artifact:

```text
.tmp/baseline_runs/learned_hybrid_replay/learned_hybrid_replay/learned_hybrid_replay_smoke
```

Result:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.3333333333333333` |
| average forgetting | `0.0` |
| replay samples | `24` |
| replay-augmented batches | `6` |
| learned-risk selections | `12` |
| class-balanced selections | `12` |
| skipped replay opportunities | `0` |

This verifies that the hybrid fills the budget and logs the intended 50/50 mix.

## Split CIFAR-100 Pilot

Artifact:

```text
experiments/runs/learned_hybrid_replay/learned_hybrid_replay/learned_hybrid_replay_split_cifar100_seed0_prior_random_frac050_class_balanced
```

Result:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.0879` |
| average forgetting | `0.3428888888888889` |
| total replay samples | `45216` |
| replay-augmented batches | `1413` |
| learned-risk selections | `22608` |
| class-balanced selections | `22608` |
| actual learned-risk fraction | `0.5` |
| unique replayed samples | `1789` |
| never replayed buffer samples | `211` |
| training time seconds | `44.392299700004514` |

## Comparison Context

Seed-0 references:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid replay, 50/50 class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

Three-seed references:

| Method | Final accuracy mean | Avg forgetting mean |
| --- | ---: | ---: |
| fine-tuning | `0.0470` | `0.4310` |
| random replay | `0.1016` | `0.3027` |
| spaced replay proxy | `0.0986` | `0.3131` |
| MIR replay | `0.11636666666666667` | `0.2167037037037037` |

## Scientific Interpretation

Task 20 is mixed but still negative.

The hybrid improves over pure learned fixed-budget replay:

```text
final accuracy: 0.0759 -> 0.0879
average forgetting: 0.3588 -> 0.3429
```

That supports the diversity hypothesis. Protecting part of the replay batch for
class-balanced examples helped.

However, the hybrid still does not beat random replay:

```text
random replay final accuracy: 0.1013
hybrid final accuracy: 0.0879
```

So the current learned-risk score is still not operationally useful as a replay
selector under this setup. It may predict future forgetting offline, but it does
not yet choose better training examples than random replay online.

In plain language:

```text
Adding balanced review helped the learned selector, but plain random review is
still better at keeping old knowledge alive.
```

MIR remains the strongest implemented method because it chooses replay examples
based on estimated interference from the current update, which is more directly
tied to the training step than the current offline forgetting-risk score.

## Next Research Move

The next task should not claim success for learned-risk replay. The clean
conclusion so far is:

```text
The current cheap signals can predict forgetting offline, but they have not yet
produced a replay selector that beats random replay.
```

Task 21 moved to expensive signal diagnostics, starting with final-layer
gradient norms, framed as a diagnosis:

```text
Do stronger signals improve replay-relevant prediction enough to beat random
replay, and are they worth the added cost?
```

The result is documented in
[GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md):
gradient features did not improve average precision over the cheap feature set.

Before spending heavily, a small follow-up ablation could also try lower
learned-risk fractions such as `0.25`, because the 50/50 hybrid moved in the
right direction but still carried too much weak learned-risk selection.

## Verification

Focused verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\baselines\test_learned_hybrid_replay.py tests\baselines\test_learned_fixed_budget_replay.py tests\replay\test_spaced_scheduler.py tests\predictors\test_online_forgetting.py -q
```

Result:

```text
10 passed
```

Full repository verification after Task 20:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
79 passed
```

Experiment commands:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_smoke.yaml
.\.venv\Scripts\python.exe -m src.baselines.learned_hybrid_replay --config configs\experiments\learned_hybrid_replay_split_cifar100.yaml
```
