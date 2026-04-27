# Fixed-Budget Learned Replay Plan

## Why This Plan Change Exists

Task 18 tested a learned predictor as a sparse replay gate. That experiment was
important, but it bundled two different ideas together:

```text
1. choose examples using learned forgetting risk
2. skip replay when risk is below a threshold
```

The result was negative. The learned gate performed worse than random replay,
MIR, and even the cheap risk-gated replay pilot.

That does not prove learned risk is useless. It proves that this version of
sparse learned gating is not enough. The next step should separate the two
questions:

```text
Does learned risk help choose replay examples when the replay budget is fixed?
```

Only after that should the project ask:

```text
Can learned risk safely skip replay and save compute?
```

## Research Alignment

This preserves the original proposal idea. The proposal asks whether
sample-level forgetting signals can predict when examples should be replayed
better than random or fixed schedules.

The revised plan still tests that idea, but in a cleaner order:

1. First test whether learned risk is useful for **ranking/selecting** examples.
2. Then test whether learned risk is useful for **skipping** replay.
3. Then add expensive signals only if the current cheap signals are not enough.

This avoids overreacting to one failed sparse-gating result.

## What Has Worked

- Random replay improves strongly over fine-tuning.
- MIR is currently the strongest implemented replay method.
- Cheap signals can predict forgetting better than chance in offline reports.
- Learned predictors improve offline average precision over simple heuristics.
- Signal ablations show the full cheap feature set is best, and
  `history_summary` is the strongest compact feature group.

## What Has Failed

- The first spaced replay due-time proxy did not beat random replay.
- Cheap sparse risk-gated replay saved replay samples but did not preserve
  retention.
- Learned sparse risk-gated replay also failed. It used a better offline
  predictor, but online replay performance was worse than random replay.

## What We Still Do Not Know

After Task 19, the project now knows that sparse gating was not the only weak
part. Pure learned-risk ranking still underperforms random replay when replay
volume is held fixed.

In plain language:

```text
The predictor can identify risky-looking examples,
but replaying only risky-looking examples is not enough.
```

Task 20 has now tested whether learned risk can help when diversity is
protected by class-balanced replay. Diversity helped, but it was still not
enough to beat random replay.

## Task 19: Fixed-Budget Learned-Risk Replay

Status: `complete`; see
[LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](./LEARNED_FIXED_BUDGET_REPLAY_TASK19.md).

Implemented a replay method that uses the learned predictor to rank memory items,
but still replays the same number of examples as random replay.

The method should:

- train the logistic risk scorer from prior artifacts, as in Task 18;
- score memory items online;
- select the highest-risk items for replay;
- fill every replay batch to the same size as random replay;
- never skip replay because risk is low;
- log risk scores, selected sample IDs, replay counts, class/task coverage, and
  selection reasons.

This directly tested:

```text
Is learned risk useful for choosing examples under the same replay budget?
```

Seed-0 comparison:

| Method | Replay budget | Purpose |
| --- | --- | --- |
| random replay | full matched budget | baseline |
| learned-risk fixed-budget replay | full matched budget | test learned ranking |
| MIR | full matched budget | strong replay reference |

Result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned-risk fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| MIR | `0.1183` | `0.21400000000000002` | `45216` |

Interpretation:

Learned-risk fixed-budget replay did not beat random replay and did not
approach MIR. The learned predictor is not yet operationally useful as a pure
replay ranker under this setup.

## Task 20: Balanced Hybrid Replay

Status: `complete`; see
[LEARNED_HYBRID_REPLAY_TASK20.md](./LEARNED_HYBRID_REPLAY_TASK20.md).

Task 19 was not clearly better than random replay, so Task 20 implemented a
hybrid selector:

```text
part learned-risk replay + part random or class-balanced replay
```

The reason is diversity. Random replay works partly because it samples broadly.
Risk-only replay can over-focus on hard, unstable, or noisy examples.

The hybrid should test variants such as:

- 50% high learned-risk examples, 50% random examples;
- 50% high learned-risk examples, 50% class-balanced examples;
- per-task or per-class caps so one task cannot dominate replay.

This asked:

```text
Can learned risk help if diversity is protected?
```

Seed-0 result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid replay, 50/50 class-balanced | `0.0879` | `0.3428888888888889` | `45216` |
| MIR | `0.1183` | `0.21400000000000002` | `45216` |

The hybrid improved over pure learned-risk replay, but it still did not beat
random replay. This means diversity protection helped, but the current learned
risk signal is still not strong enough as an online replay selector.

## Task 21: Expensive Signals

Status: `complete`; see
[GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md).

After Tasks 19 and 20, the project tested a proposal-aligned expensive signal:
final-layer gradient norm.

Those signals are still aligned with the proposal, but they should answer a
specific question:

```text
Do stronger signals improve replay-relevant prediction enough to justify their
runtime cost?
```

Seed-0 diagnostic result:

| Feature group | Average precision |
| --- | ---: |
| cheap all features | `0.9083240127221096` |
| cheap plus gradient | `0.9080918805327551` |
| gradient only | `0.8386703932509996` |

The gradient signal is measurable, but it does not improve prediction enough to
justify a new replay scheduler.

## Updated Near-Term Sequence

The near-term research sequence is now:

```text
Task 19: fixed-budget learned-risk replay - complete, negative
Task 20: balanced hybrid learned-risk replay - complete, mixed but negative
Task 21: gradient signal diagnostic - complete, negative
Task 22: decision checkpoint and targeted rescue ablations - next
Task 23: optional MIR-like interference or representation-drift diagnostics
Task 24: stretch benchmarks only after the Split CIFAR-100 conclusion is stable
Task 25: final synthesis
```

## Success Criteria

Task 19 would have succeeded if it had beaten random replay under the same
replay budget. It did not, so the study should not claim that learned-risk
ranking alone improves retention.

Task 20 improved over pure learned-risk replay but did not improve over random
replay. It therefore supports the diversity diagnosis without supporting a
positive intervention claim for learned-risk replay.

Task 21 did not show a useful gain from final-layer gradient norms. This means
the project should not build a gradient-norm replay intervention from the
current signal. Stretch benchmarks should stay behind a decision checkpoint.

The final report should be honest either way. A clean negative result is still
valuable if it shows:

```text
offline forgetting prediction does not automatically translate into better
online replay selection
```

That is a meaningful answer to the original proposal.
