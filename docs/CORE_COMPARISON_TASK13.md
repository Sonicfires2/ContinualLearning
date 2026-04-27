# Task 13 Core Controlled Comparison

## Purpose

Task 13 is the first fair method-comparison pass for the core proposal. It
compares:

- fine-tuning
- random replay
- fixed-periodic replay
- spaced replay

under the same Split CIFAR-100 task order, model, optimizer settings, epochs,
replay capacity, replay batch size, and seed list.

## Runner

Implementation:

- [src/experiments/core_comparison.py](../src/experiments/core_comparison.py)
- [scripts/run_core_comparison.py](../scripts/run_core_comparison.py)

Configs:

- [configs/experiments/core_comparison_smoke.yaml](../configs/experiments/core_comparison_smoke.yaml)
- [configs/experiments/core_comparison_split_cifar100.yaml](../configs/experiments/core_comparison_split_cifar100.yaml)

Full summary artifact:

```text
experiments/task13_core_comparison/task13_core_comparison_split_cifar100_summary.json
```

## Fairness Controls

Shared controls:

- protocol: `core_split_cifar100_v2`
- benchmark: Split CIFAR-100
- task count: `10`
- classes per task: `10`
- split seed: `0`
- model: flatten MLP, hidden dimension `256`
- epochs per task: `1`
- batch size: `32`
- learning rate: `0.01`
- replay capacity: `2000`
- replay batch size: `32`
- seeds: `0`, `1`, `2`

Replay budget matching:

- random replay: replay every current-task batch after the buffer is nonempty
- fixed-periodic replay: `k = 1`, replay every current-task batch after the
  buffer is nonempty
- spaced replay: `budget_mode = match_random_replay`, replay every current-task
  batch after the buffer is nonempty, selecting by due-time priority

Important caveat: in this implementation, budget-matched fixed-periodic replay
with `k = 1` is effectively the same cadence and selection distribution as the
random replay baseline. The earlier `k = 2` fixed-periodic run remains an
interval ablation, not a budget-matched comparison.

## Aggregate Results

These are three-seed means and sample standard deviations.

| Method | Final accuracy mean | Final accuracy std | Avg forgetting mean | Avg forgetting std | Replay samples mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine-tuning | `0.04703333333333334` | `0.0017897858344878411` | `0.43103703703703705` | `0.004659165046447831` | `0` |
| random replay | `0.10156666666666665` | `0.004206344414492624` | `0.30274074074074075` | `0.0014330031201594854` | `45216` |
| fixed-periodic replay, `k=1` | `0.10156666666666665` | `0.004206344414492624` | `0.30274074074074075` | `0.0014330031201594854` | `45216` |
| spaced replay, due-time proxy | `0.09863333333333334` | `0.003957692930652066` | `0.3131111111111111` | `0.0028043176586942256` | `45216` |

## Per-Seed Results

| Method | Seed | Final accuracy | Average forgetting | Replay samples |
| --- | ---: | ---: | ---: | ---: |
| fine-tuning | `0` | `0.0455` | `0.434` | `0` |
| fine-tuning | `1` | `0.0466` | `0.4334444444444444` | `0` |
| fine-tuning | `2` | `0.049` | `0.42566666666666664` | `0` |
| random replay | `0` | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| random replay | `1` | `0.1059` | `0.30233333333333334` | `45216` |
| random replay | `2` | `0.0975` | `0.3015555555555556` | `45216` |
| fixed-periodic replay, `k=1` | `0` | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| fixed-periodic replay, `k=1` | `1` | `0.1059` | `0.30233333333333334` | `45216` |
| fixed-periodic replay, `k=1` | `2` | `0.0975` | `0.3015555555555556` | `45216` |
| spaced replay, due-time proxy | `0` | `0.0962` | `0.315` | `45216` |
| spaced replay, due-time proxy | `1` | `0.1032` | `0.3098888888888889` | `45216` |
| spaced replay, due-time proxy | `2` | `0.0965` | `0.31444444444444447` | `45216` |

## Scientific Interpretation

The fair comparison supports three claims:

1. Replay helps. All replay methods substantially improve final accuracy and
   average forgetting compared with fine-tuning.
2. The budget-matched fixed-periodic comparator is equivalent to random replay
   in this implementation because `k=1` gives the same replay cadence and
   uniform sampling.
3. The current spaced replay due-time proxy does not beat random replay. It is
   slightly worse in final accuracy and average forgetting across the three
   seeds.

This is a clean negative result for the first scheduler version, not a failure
of the whole project. Task 11 already showed that crude risk-scaled timing
heuristics do not estimate `T_i` better than a constant median baseline. Task 13
now shows the operational consequence: the current due-time proxy is not enough
to outperform random replay.

## Claim Boundary

The project can now claim:

- sample-level logging and timing targets are implemented;
- replay improves over no replay;
- the first spaced replay proxy does not improve over random replay.

The project cannot yet claim:

- cognitive-spacing replay improves continual learning;
- the current scheduler accurately estimates sample-specific `T_i`;
- representation drift or gradient norms have been evaluated.

## Post-Task-14 Stronger Baseline Note

Task 14 added MIR as a stronger replay baseline, documented in
[MIR_REPLAY_BASELINE.md](./MIR_REPLAY_BASELINE.md). Under the same Split
CIFAR-100 replay budget and seed list, MIR reached final accuracy mean
`0.11636666666666667` and average forgetting mean `0.2167037037037037`.

This does not change the Task 13 conclusion: the first spaced replay proxy still
fails against random replay. It does raise the future bar for the project. A
revised spaced replay method should first beat random and fixed-periodic replay,
then be checked against MIR.

## Verification

Smoke comparison:

```text
.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_smoke.yaml
```

Full comparison:

```text
.\.venv\Scripts\python.exe scripts\run_core_comparison.py --config configs\experiments\core_comparison_split_cifar100.yaml
```

Full test suite:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
50 passed
```
