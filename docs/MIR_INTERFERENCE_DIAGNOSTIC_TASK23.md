# Task 23 MIR-Like Current-Interference Diagnostic

## Purpose

Task 23 tests the most important lesson from the previous results:

```text
Maybe the useful signal is not "will this example be forgotten later?"
Maybe the useful signal is "will the current update damage this example now?"
```

MIR works by asking that second question. For each replay opportunity, it takes
a candidate pool from memory, performs a virtual update with the current batch,
and scores each memory candidate by:

```text
post-virtual-update loss - pre-virtual-update loss
```

Higher score means the current update would hurt that old example more.

Task 23 does not build a new scheduler. It is a diagnostic that compares:

- the prior-artifact learned future-forgetting risk score;
- MIR's current-update interference score on the same candidate pool.

## Implementation

Main code:

- [src/replay/mir.py](../src/replay/mir.py)
  - added all-candidate MIR scoring via `score_mir_replay_candidates`
- [src/predictors/mir_interference_diagnostics.py](../src/predictors/mir_interference_diagnostics.py)
  - logs MIR candidate rows and computes agreement metrics
- [src/baselines/mir_interference_diagnostic.py](../src/baselines/mir_interference_diagnostic.py)
  - runs random replay while scoring each candidate pool with both learned risk
    and MIR interference

Configs:

- [configs/experiments/mir_interference_diagnostic_smoke.yaml](../configs/experiments/mir_interference_diagnostic_smoke.yaml)
- [configs/experiments/mir_interference_diagnostic_split_cifar100.yaml](../configs/experiments/mir_interference_diagnostic_split_cifar100.yaml)

Tests:

- [tests/replay/test_mir.py](../tests/replay/test_mir.py)
- [tests/predictors/test_mir_interference_diagnostics.py](../tests/predictors/test_mir_interference_diagnostics.py)
- [tests/baselines/test_mir_interference_diagnostic.py](../tests/baselines/test_mir_interference_diagnostic.py)

## What Was Run

Smoke command:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.mir_interference_diagnostic --config configs\experiments\mir_interference_diagnostic_smoke.yaml
```

Split CIFAR-100 seed-0 command:

```powershell
.\.venv\Scripts\python.exe -m src.baselines.mir_interference_diagnostic --config configs\experiments\mir_interference_diagnostic_split_cifar100.yaml
```

The Split CIFAR-100 run used:

- candidate pool size: `128`
- replay batch size: `32`
- positive target: "MIR would put this candidate in the top 32"
- actual replay policy: uniform replay from the candidate pool
- learned-risk scorer: prior-artifact logistic predictor from random replay seed 0

## Results

Split CIFAR-100 seed 0:

| Metric | Value |
| --- | ---: |
| final accuracy | `0.10090000000000002` |
| average forgetting | `0.30144444444444446` |
| candidate rows scored | `180864` |
| candidate events | `1413` |
| candidate count per event | `128` |
| MIR top-k target rate | `0.25` |
| learned-risk AP for MIR top-k | `0.21600508010478187` |
| learned-risk ROC-AUC for MIR top-k | `0.42531155059460235` |
| learned-risk top-k overlap with MIR top-k | `0.1792949398443029` |
| random expected top-k overlap | `0.25` |
| overlap minus random expected | `-0.0707050601556971` |
| mean learned risk for MIR top-k candidates | `0.7074673223586053` |
| mean learned risk for non-top-k candidates | `0.7701525397229256` |
| mean MIR interference for top-k candidates | `0.19340795277708828` |
| mean MIR interference for non-top-k candidates | `0.028875874144746707` |

## Plain-English Explanation

The learned forgetting predictor is good at its original job:

```text
From saved evaluation histories, rank examples by whether they may be forgotten later.
```

But Task 23 asks a different question:

```text
From the current training batch, which memory examples is this next update about to hurt?
```

The answer is bad for the current learned-risk replay idea. If the learned-risk
score agreed with MIR, it would often pick the same examples MIR puts in the top
32 out of 128 candidates. It does not.

In fact:

```text
random expected overlap: 0.25
learned-risk overlap:   0.1793
```

So the learned risk score overlaps with MIR less than random selection would.
The average precision is also below the base rate:

```text
base rate: 0.25
average precision: 0.2160
```

That means the learned future-forgetting score is not just weak at predicting
MIR's current-interference choices; in this diagnostic it is actively pointing
away from them.

## Interpretation

Task 23 explains why the earlier results looked contradictory.

The earlier predictor result was strong:

```text
future-forgetting AP = 0.9083
```

But Task 23 shows that this score does not line up with MIR's replay decision:

```text
MIR-top-k AP from learned risk = 0.2160
```

Those are not measuring the same thing. The learned predictor is answering a
future-risk question. MIR is answering a current-update-interference question.
Replay selection appears to need the second one.

## Decision

Do not build another scheduler that simply tunes the current learned-risk
fraction.

The next sensible research move is:

1. write the final result as a clean diagnostic/negative study; or
2. if more method work is required, build a new method around current-update
   interference rather than around offline future-forgetting risk.

The recommended next task is Task 25: final synthesis.

