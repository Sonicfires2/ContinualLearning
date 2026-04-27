# Proposal Assessment

Date: 2026-04-22

## Purpose

This document records a focused review of the current project proposal in [Project_Proposal-2.pdf](../Project_Proposal-2.pdf) and the text-friendly source [Project_Proposal-2.tex](../Project_Proposal-2.tex). A cleaned extraction is available in [PROJECT_PROPOSAL_TEXT.md](./PROJECT_PROPOSAL_TEXT.md). It answers four practical questions:

1. What is the research project about?
2. What is the research question?
3. What does the proposal suggest the team should do?
4. Is the proposal's recommended information correct and well-supported?

## Short Answer

The proposal is about replay-based continual learning, specifically a spacing-inspired replay strategy for reducing catastrophic forgetting during sequential fine-tuning. The core idea is sound and aligns with recent continual-learning literature. The proposal is strongest when interpreted as a vision-focused continual-learning study built around Split CIFAR-100 and replay scheduling. It becomes less grounded when it expands into broad LLM motivation, multi-benchmark scope, and hardware assumptions that do not match the current Windows setup.

Overall judgment:

- Core research direction: correct and worthwhile
- Main research question: clear and reasonable
- Proposed experiment shape: mostly correct
- Recommended scope: ambitious and should be trimmed to a strong core
- Hardware assumptions: correct for Apple MPS in general, but incorrect for the current local machine
- Citation list: core ideas are supported, but every citation should still be checked before final submission

## What the Research Project Is About

The project studies **continual learning**, where a model is trained on tasks sequentially and tends to forget earlier tasks when learning later ones. The proposal focuses on **replay-based continual learning**, where a small memory buffer stores old examples and reuses them during later training.

The proposal's main contribution is not simply "use replay," but:

- estimate which samples are most at risk of being forgotten
- estimate when those samples will be forgotten, represented in the proposal as an estimated forgetting time `T_i`
- use that risk and timing estimate to decide when and what to replay
- compare this strategy against standard replay baselines

In plain language, the project is asking whether replay can be made smarter by paying attention to forgetting signals instead of replaying old data randomly or on a fixed schedule.

## Research Question

The central research question can be stated as:

> Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

The proposal source makes "when" concrete: the scheduler should compute an estimated forgetting time `T_i` and replay near `t + T_i`.

This is a good research question because it is:

- specific
- testable
- connected to measurable outcomes
- grounded in a real design choice in continual learning systems

## What the Proposal Suggests the Team Should Do

The proposal recommends the following research program.

### 1. Use continual-learning benchmarks

Primary benchmark:

- Split CIFAR-100

Secondary or stretch benchmarks:

- Split CUB
- sequential domain adaptation with DistilBERT if time permits

### 2. Build baseline methods

The proposal suggests comparing the new idea against standard baselines such as:

- naive sequential fine-tuning
- experience replay
- MIR
- ESMER
- uncertainty-based replay

### 3. Track sample-level forgetting signals

The proposal highlights:

- loss trajectories
- uncertainty
- gradient norms
- representation drift

These signals are meant to help estimate which examples are likely to be forgotten.

### 4. Build a forgetting predictor

The project then proposes turning those signals into a prediction of forgetting risk.

### 5. Use that prediction to drive spaced replay

The main proposed method is a spacing-inspired scheduler that decides when to replay examples from memory. In the proposal source, this is described as deriving a sample-specific interval from the probability of forgetting within a horizon, computing an estimated forgetting time `T_i`, and scheduling replay at `t + T_i`.

### 6. Evaluate both predictor quality and task performance

The proposal suggests evaluating:

- final accuracy
- average forgetting
- computational cost
- replay utilization
- predictor precision-recall behavior
- time-to-forgetting or due-time estimation quality
- signal ablations

## Assessment of Proposal Quality

## 1. Is the main direction correct?

Yes.

Replay-based continual learning is a well-established area, and recent work supports the idea that **which samples are replayed** can matter significantly. The proposal is well aligned with the field when it focuses on:

- replay-based learning
- sample selection
- forgetting prediction
- representation drift or uncertainty as useful signals

## 2. Is the research question correct and meaningful?

Yes.

It is meaningful because replay buffers are small in practice, so the problem of which examples to store and when to replay them is important. It is also experimentally tractable because the project can compare:

- random replay
- fixed replay schedules, such as replay once every `k` iterations
- heuristic replay
- prediction-guided replay

under the same memory budget.

## 3. Are the proposed baselines reasonable?

Mostly yes.

The proposal's baseline set is directionally correct:

- fine-tuning gives the no-replay lower baseline
- experience replay gives the standard replay baseline
- MIR is a credible stronger replay baseline
- ESMER and uncertainty-guided methods are useful if time permits

The only caution is scope. A student project should treat the baselines in tiers:

Required:

- fine-tuning
- random experience replay
- fixed-periodic replay

Recommended:

- MIR

Stretch:

- ESMER
- uncertainty-based replay

## 4. Are the proposed forgetting signals reasonable?

Yes, but with an important caveat.

The listed signals are all plausible and supported by recent work:

- uncertainty is a reasonable proxy for instability
- loss history may reveal samples that are getting harder over time
- gradient-based signals can indicate interference
- representation drift can identify internal feature instability

However, the proposal should treat them as **hypotheses to test**, not as already-proven truths. Some signals may work better than others, and some may be too expensive to compute relative to their benefit.

## 5. Are the proposed datasets and scope reasonable?

Partly.

### Good core benchmark

Split CIFAR-100 is a strong and realistic primary benchmark for this project.

### Acceptable but harder extension

Split CUB is plausible, but more expensive and harder to prepare well.

### Risky stretch goal

DistilBERT sequential domain adaptation is possible, but it broadens the project significantly. It should only be attempted after the vision pipeline is already stable.

Practical conclusion:

- Split CIFAR-100 should be the required benchmark
- Split CUB should be optional
- DistilBERT should be a stretch goal only

## 6. Is the proposal's hardware recommendation correct?

Only partly.

The proposal's statement that PyTorch can use Apple's MPS backend on Apple silicon is correct in general. That is a real and supported backend.

However, it does **not** match the current local project setup:

- the current environment is on Windows
- the current local PyTorch install is CPU-only unless changed to a CUDA build
- MPS-specific wording should not be treated as the execution plan for this repository

So the hardware section is:

- technically correct in isolation
- operationally incorrect for the current machine

## 7. Is the broad LLM framing accurate?

Not really, at least not as written.

The proposal motivates the work partly as an AI or LLM problem, but the actual experimental plan is mostly:

- vision benchmarks
- continual classification
- replay scheduling

That means the current proposal is best understood as a **continual learning project with optional later language experiments**, not as an LLM project from the start.

## 8. Is the recommended information in the proposal correct overall?

The best overall answer is:

**Mostly yes, but unevenly.**

### Correct and well-supported

- catastrophic forgetting is a central problem in continual learning
- replay is a strong family of baselines
- replay-buffer composition matters
- uncertainty, learning dynamics, and representational change are sensible replay signals
- MPS support on Apple hardware exists

### Correct but should be framed as hypotheses

- loss, uncertainty, gradient norm, and drift can predict forgetting well enough to improve replay
- a spacing-inspired schedule based on estimated forgetting time `T_i` will outperform simpler replay policies

These are exactly the claims the project should test.

### Weak or over-ambitious

- combining several strong baselines, multiple datasets, and optional language experiments in one short project
- presenting the project as broadly LLM-driven when the core work is still vision continual learning

### Incorrect for the current local environment

- assuming the project will run on the proposal's Mac M3 / MPS setup

## Recommended Interpretation for This Repository

The cleanest reading of the project is:

1. Build a strong continual-learning pipeline on Split CIFAR-100.
2. Compare fine-tuning, random replay, and fixed-periodic replay.
3. Add forgetting signals, starting with the cheapest ones first.
4. Define risk prediction and time-to-forgetting estimation targets.
5. Implement a spaced replay scheduler that uses estimated `T_i` or a documented due-time proxy.
6. Test whether that scheduler improves average forgetting under the same memory budget.
7. Only then consider stronger baselines such as MIR, Split CUB, or DistilBERT.

This interpretation keeps the proposal research-valid while reducing the chance of scope failure.

## Current Implementation Coverage After Task 9

As of the Task 9 implementation pass, the repository has covered the proposal's
infrastructure and first prediction stage, but not the full spaced-replay
intervention.

Covered at pilot quality:

- Split CIFAR-100 task stream and stable sample identity.
- Fine-tuning baseline.
- Random replay baseline.
- Sample-level loss, confidence, uncertainty, replay-count, and last-replay
  telemetry.
- Future-forgetting labels derived from later seen-task evaluations.
- A leakage-safe first forgetting-risk predictor.

Not yet covered:

- fixed-periodic replay every `k` iterations;
- explicit `T_i` or due-time estimation;
- a spacing scheduler that replays near `t + T_i`;
- cognitive-spacing claims tested through an actual timing rule;
- gradient-norm instrumentation;
- representation or latent drift instrumentation.

Therefore, the existing Task 1 through Task 9 runs are useful pilot evidence and
pipeline validation, but they are not final proposal-valid runs. The project
does not need to rerun them immediately. It should first implement fixed-periodic
replay, define/evaluate `T_i`, implement the spaced scheduler, and then rerun
the controlled core comparison under protocol `core_split_cifar100_v2`.

## Current Implementation Coverage After Task 18

The repository has now moved beyond the Task 9 pilot stage. Tasks 10 through 14
added fixed-periodic replay, explicit time-to-forgetting targets, the first
spaced replay scheduler, a three-seed core comparison, and a three-seed MIR
baseline. Task 15 added an event-triggered risk-gated replay scheduler that can
skip replay when no buffered sample passes the risk/due gate. Task 16 added the
proposal's learned predictor families for offline comparison: logistic
regression, linear regression/ridge regression, and linear SVM classifiers and
regressors. Task 17 added signal ablations and threshold analysis. Task 18
added a learned-predictor risk-gated replay intervention.

Current evidence:

- random replay and fixed-periodic replay improve substantially over
  fine-tuning;
- the first spaced replay due-time proxy does not beat random replay;
- MIR is currently the strongest implemented replay method under the same
  Split CIFAR-100 budget, with higher final accuracy and lower average
  forgetting than random replay and the first spaced scheduler.
- the first sparse risk-gated replay pilot saves replay samples but does not
  preserve retention; with `risk_threshold = 0.75`, it used `2071` replay
  samples on seed 0 and reached final accuracy `0.046`.
- on informative seed-0 replay artifacts, learned binary predictors improve
  average precision over the best cheap heuristic: random replay improves from
  `0.8471253916174554` to `0.9083240127221096`, and spaced replay improves from
  `0.8267285181601283` to `0.9188844673560967`.
- signal ablations show `all_features` is best and `history_summary` is the
  strongest compact group on the informative replay artifacts.
- the learned online replay gate is implemented but negative: threshold `0.90`
  reaches final accuracy `0.0379` on seed 0, worse than random replay, MIR, and
  the cheap risk-gated replay pilot.
- the post-Task-18 plan now separates learned-risk ranking from sparse replay
  skipping: fixed-budget learned-risk replay and balanced hybrid replay should
  be tested before expensive signals.

Research interpretation:

- the core pipeline is proposal-valid enough for controlled method comparison;
- the project has not yet validated the proposal's cognitive-spacing mechanism;
- event-triggered replay is implemented, but the current cheap online heuristic
  is not strong enough as the sole replay trigger;
- learned predictors improve offline ranking on the usable replay artifacts,
  but have not yet been validated as online replay gates;
- the first learned-predictor online replay gate failed, suggesting an
  online/offline feature-distribution mismatch;
- the next step should test fixed-budget learned-risk replay, because Task 18
  did not isolate learned-risk ranking from replay skipping;
- expensive signals such as gradient norms or representation drift remain
  useful, but should follow the fixed-budget and balanced-hybrid diagnostics.

## Current Implementation Coverage After Task 19

Task 19 implemented the fixed-budget learned-risk replay diagnostic proposed
after Task 18. It used the same prior-artifact logistic scorer, but removed
low-risk replay skipping and matched random replay's seed-0 replay budget.

Result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay seed 0 | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay seed 0 | `0.0759` | `0.3587777777777778` | `45216` |
| MIR replay seed 0 | `0.1183` | `0.21400000000000002` | `45216` |

Interpretation:

- learned-risk ranking alone does not yet beat random replay under matched
  replay volume;
- the Task 18 failure was not caused only by sparse replay skipping;
- the current learned predictor remains more convincing as an offline
  forgetting diagnostic than as an online replay selector;
- the next aligned test is a balanced hybrid that combines learned-risk
  examples with random or class-balanced diversity;
- expensive signals remain useful later, but should follow the hybrid
  diagnostic rather than replacing it.

## Current Implementation Coverage After Task 20

Task 20 implemented the balanced hybrid diagnostic recommended after Task 19.
The pilot used a 50/50 split:

```text
50% learned-risk-ranked examples
50% class-balanced examples
```

Result:

| Method | Final accuracy | Avg forgetting | Replay samples |
| --- | ---: | ---: | ---: |
| random replay seed 0 | `0.10129999999999999` | `0.30433333333333334` | `45216` |
| learned fixed-budget replay seed 0 | `0.0759` | `0.3587777777777778` | `45216` |
| learned hybrid replay seed 0 | `0.0879` | `0.3428888888888889` | `45216` |
| MIR replay seed 0 | `0.1183` | `0.21400000000000002` | `45216` |

Interpretation:

- protecting diversity improves learned-risk replay compared with pure
  learned-risk ranking;
- the hybrid still does not beat random replay or MIR;
- the current cheap learned-risk signal is not yet an intervention-effective
  replay selector;
- the project can now justify expensive signal diagnostics, but only as a
  targeted attempt to improve replay-relevant prediction, not as a guaranteed
  fix.

## Current Implementation Coverage After Task 21

Task 21 implemented a gradient-norm diagnostic. The signal is the exact
per-sample final-layer cross-entropy gradient norm, logged at seen-task
evaluation anchors during a random-replay diagnostic run.

Result:

| Feature group | Average precision |
| --- | ---: |
| cheap all features | `0.9083240127221096` |
| cheap plus gradient | `0.9080918805327551` |
| gradient only | `0.8386703932509996` |

Cost:

- measured training-time overhead versus random replay seed 0:
  `0.5744868292691545`;
- gradient artifact size: `42909732` bytes.

Interpretation:

- the implemented gradient signal is proposal-aligned and auditable;
- it does not improve forgetting prediction enough to justify a new replay
  scheduler;
- the strongest current lesson is that prediction quality does not automatically
  translate into replay-intervention quality;
- stretch benchmarks should be moved behind a decision checkpoint, because the
  core Split CIFAR-100 method has not yet beaten random replay;
- if method development continues, MIR-like current-interference or
  representation-drift diagnostics are more promising than final-layer gradient
  norm alone.

## Recommended Scope Adjustment

The proposal should be operationalized in this priority order.

### Phase 1: required

- Split CIFAR-100
- naive fine-tuning baseline
- random replay baseline
- fixed-periodic replay baseline
- logging for loss and uncertainty
- average forgetting and final accuracy metrics

### Phase 2: strong contribution

- forgetting-risk heuristic or simple predictor
- time-to-forgetting `T_i` estimator
- spaced replay scheduler
- controlled comparison under fixed memory budget

### Phase 3: stronger evaluation

- MIR baseline
- event-triggered risk-gated replay scheduler
- learned forgetting predictors such as logistic regression, linear regression,
  and support-vector machines
- signal ablations
- learned-predictor replay gate
- fixed-budget learned-risk replay
- balanced hybrid replay
- predictor precision-recall AUC

### Phase 4: stretch work

- Split CUB
- DistilBERT
- heavier signals such as representation drift or gradient norms if runtime allows

## Verified External Support

The following external sources support the main assessment above.

### Replay-based continual learning baseline

- Aljundi et al., **Online Continual Learning with Maximal Interfered Retrieval**, NeurIPS 2019  
  https://papers.nips.cc/paper_files/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html

Why it matters:

- confirms MIR is a real and relevant replay baseline
- supports the claim that random replay is not always optimal

### Forgetting-aware replay buffer composition

- Hacohen and Tuytelaars, **Predicting the Susceptibility of Examples to Catastrophic Forgetting**, ICML 2025 / PMLR 267  
  https://proceedings.mlr.press/v267/hacohen25a.html

Why it matters:

- supports the idea that example-level forgetting dynamics can guide replay selection

### Representation-drift motivated replay

- Sarfraz et al., **Error Sensitivity Modulation based Experience Replay**, ICLR 2023  
  https://openreview.net/forum?id=zlbci7019Z3

Why it matters:

- supports the use of representation-drift-related reasoning and replay improvements beyond plain random sampling

### Uncertainty-guided continual learning

- Serra et al., **How to Leverage Predictive Uncertainty Estimates for Reducing Catastrophic Forgetting in Online Continual Learning**, TMLR 2025 / OpenReview  
  https://openreview.net/forum?id=dczXe0S1oL

Why it matters:

- supports uncertainty as a legitimate signal family for replay decisions

### Apple MPS backend support

- PyTorch documentation, `torch.mps`  
  https://docs.pytorch.org/docs/stable/mps.html

Why it matters:

- confirms the proposal's Apple MPS claim is technically valid in general, even though it does not describe the current Windows environment

## Final Verdict

The proposal is **good in its core idea** and **usable as the scientific basis for the project**, but it should be interpreted carefully.

Best one-sentence summary:

> The proposal is strongest as a vision-based continual-learning study on smarter replay scheduling, and weakest when it stretches into broad LLM framing, multi-benchmark ambition, and machine-specific assumptions that do not match the current setup.

If the team keeps the scope centered on Split CIFAR-100, random replay, forgetting signals, and a spaced replay scheduler, the proposal is not only reasonable but genuinely promising.
