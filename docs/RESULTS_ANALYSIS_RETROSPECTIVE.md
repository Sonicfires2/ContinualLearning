# Results Analysis and Retrospective

## Short Version

The project is still answering the original research question, but the answer
so far is mostly negative for the proposed replay interventions.

In simple terms:

```text
Replay helps.
MIR helps more.
Our forgetting-risk predictors can predict future forgetting offline.
But using those predictors to choose replay examples has not beaten random replay.
```

That is still a valid research result. It means prediction and intervention are
not the same thing.

For the full Task-25 final synthesis, including background, method
descriptions, benchmark definitions, metric explanations, and result tables, see
[FINAL_SYNTHESIS_TASK25.md](./FINAL_SYNTHESIS_TASK25.md).

## What The Project Has Shown

### 1. The benchmark is real

Fine-tuning forgets badly:

| Method | Final accuracy | Avg forgetting |
| --- | ---: | ---: |
| fine-tuning, 3-seed mean | `0.0470` | `0.4310` |

This confirms the Split CIFAR-100 setup creates the problem the project is
supposed to study.

### 2. Ordinary replay helps

Random replay is much better than no replay:

| Method | Final accuracy | Avg forgetting |
| --- | ---: | ---: |
| fine-tuning, 3-seed mean | `0.0470` | `0.4310` |
| random replay, 3-seed mean | `0.1016` | `0.3027` |

Plain review of old examples already reduces forgetting a lot.

### 3. MIR is the strongest implemented method

MIR beats random replay:

| Method | Final accuracy | Avg forgetting |
| --- | ---: | ---: |
| random replay, 3-seed mean | `0.1016` | `0.3027` |
| MIR replay, 3-seed mean | `0.11636666666666667` | `0.2167037037037037` |

Plain explanation:

```text
MIR chooses old examples that the current new-task update would damage most.
That is closer to the actual cause of forgetting than just asking which examples
look risky in general.
```

### 4. Offline prediction works

The learned predictor improves over cheap heuristics in offline reports:

| Artifact | Best cheap heuristic AP | Best learned AP |
| --- | ---: | ---: |
| random replay seed 0 | `0.8471253916174554` | `0.9083240127221096` |
| spaced replay seed 0 | `0.8267285181601283` | `0.9188844673560967` |

This supports part of the proposal:

```text
sample-level signals contain information about future forgetting
```

### 5. Online replay interventions do not work yet

The replay policies that use the learned risk score have not beaten random
replay:

| Method, seed 0 | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned risk-gated, threshold `0.90` | `0.0379` | `0.40311111111111114` | `14425` |
| learned fixed-budget | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid 50/50 class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| Task 22, 25% learned-risk + 75% class-balanced | `0.0986` | `0.3268888888888889` | `45216` |
| Task 22, 25% learned-risk + 75% random | `0.0879` | `0.33144444444444443` | `45216` |
| Task 22, class-balanced only | `0.10500000000000001` | `0.2962222222222222` | `45216` |
| MIR replay | `0.1183` | `0.21400000000000002` | `45216` |

Plain explanation:

```text
The predictor can say, "this example might be forgotten later."
But replay needs to answer, "which examples should I review right now so the
next update does less damage?"
```

Those are related, but they are not identical.

Task 22 adds one important nuance: pure class-balanced replay slightly beats
random replay on seed 0, while adding learned-risk selection back in does not.
That suggests the useful ingredient in the rescue ablation is diversity, not
the current learned-risk score.

### 6. Final-layer gradient norm did not rescue the predictor

Task 21 tested a gradient-family signal:

| Feature group | Average precision |
| --- | ---: |
| cheap all features | `0.9083240127221096` |
| cheap plus gradient | `0.9080918805327551` |
| gradient only | `0.8386703932509996` |

The gradient diagnostic added about `57%` measured training-time overhead and a
`42909732` byte artifact, but did not improve prediction.

Plain explanation:

```text
This gradient signal costs more, but does not tell us more.
```

### 7. MIR-style current-interference explains the mismatch

Task 23 compared the learned future-forgetting score against MIR's
current-update interference ranking on the same replay candidate pools:

| Diagnostic | Value |
| --- | ---: |
| candidate rows | `180864` |
| MIR top-k base rate | `0.25` |
| learned-risk AP for MIR top-k | `0.21600508010478187` |
| learned-risk ROC-AUC for MIR top-k | `0.42531155059460235` |
| learned-risk top-k overlap with MIR | `0.1792949398443029` |
| random expected top-k overlap | `0.25` |

Plain explanation:

```text
The learned predictor is good at "which examples may be forgotten later?"
MIR is good at "which examples is this current update about to damage?"
Those are different jobs.
```

The learned-risk score overlaps with MIR less than random selection would. That
strongly explains why learned-risk replay has not worked as an intervention.

## What This Means

The main lesson is:

```text
Predicting forgetting is easier than preventing forgetting.
```

The project has evidence that cheap sample-level signals can forecast which
examples will later be forgotten. But the replay scheduler built from those
signals does not yet improve learning. That means the weak part is not only the
predictor. The weak part is the connection between the predictor and the replay
decision.

Random replay works because it preserves broad coverage of old tasks. MIR works
because it estimates immediate interference from the current update. The current
learned-risk methods sit between those ideas and get the best of neither:

- they are less broad than random replay;
- they are less update-aware than MIR.

Task 23 makes that explanation concrete: the learned-risk score does not pick
the same candidates that MIR's current-interference calculation picks.

## Are The Research Goals Still Valid?

Yes, but the expected conclusion has changed.

The original goal was:

```text
Develop and evaluate spacing-inspired replay policies that reduce catastrophic
forgetting during continual fine-tuning.
```

That goal is still valid because the project has evaluated those policies under
a controlled Split CIFAR-100 setup. But the evidence no longer supports a
positive claim that the current spacing-inspired or risk-guided policies reduce
forgetting better than random replay.

The current defensible research conclusion is:

```text
Sample-level forgetting can be predicted offline, but the tested risk-guided
and spacing-inspired replay policies do not yet translate that prediction into
better retention than random replay.
```

That is a real research answer.

## Retrospective

### What went well

- The benchmark and artifact pipeline are strong.
- Every major method saves metrics, configs, and sample-level traces.
- Negative results are interpretable because budgets and seeds are controlled.
- MIR gives a credible stronger baseline.
- The project did not overclaim when learned replay failed.

### What did not work

- The first spaced scheduler was only a weak due-time proxy and did not beat
  random replay.
- Sparse replay gates saved compute but lost too much retention.
- Learned-risk replay over-focused on examples that looked risky.
- Adding class-balanced diversity helped but did not close the gap.
- Final-layer gradient norm added cost without improving prediction.

### Why this is probably happening

The current predictor is trained to identify examples that will be forgotten at
future evaluation anchors. Replay selection needs a more direct signal:

```text
which old examples are being damaged by the current update?
```

MIR asks that more direct question. The learned-risk scheduler does not.

## Plan Modification

The next steps should change.

The old plan moved from expensive signals to stretch benchmarks. That should
not happen yet. A stretch dataset will not fix the core issue; it would only
make the negative result larger and harder to debug.

Recommended modified sequence:

1. Treat the Task 22 rescue ablation as complete:
   class-balanced replay is a useful baseline, but learned-risk selection still
   has not been rescued.
2. Treat the Task 23 MIR-interference diagnostic as complete:
   the current learned-risk score does not agree with MIR's replay choices.
3. Write the final report as a clean diagnostic result unless the team wants a
   new project phase built around current-interference methods.
4. Move Split CUB, DistilBERT, and other stretch benchmarks behind this
   decision point.

The research goals are still there. The project has simply learned that the
current path is not enough to achieve the intervention goal.
