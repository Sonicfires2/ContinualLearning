# Action Plan: Research-Aligned Execution Sequence

## Purpose

This document converts the proposal into an implementation sequence that stays tightly aligned with the strongest, most defensible version of the project: a vision-based continual-learning study on Split CIFAR-100 that tests whether forgetting-aware spaced replay improves retention better than fine-tuning, random replay, and fixed-periodic replay under the same memory budget.

This plan is based on the proposal assessment in [PROPOSAL_ASSESSMENT.md](./PROPOSAL_ASSESSMENT.md).

## Research Goal

Develop and evaluate spacing-inspired replay policies that reduce catastrophic forgetting during continual fine-tuning under limited memory and compute.

## Research Question

Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

In the proposal source, "when" has a concrete timing meaning: estimate each
buffered sample's forgetting time `T_i` and schedule its next replay near
`t + T_i`. Risk ranking alone is not sufficient to satisfy the full proposal.

## Alignment Rules

The plan follows four alignment rules from the proposal assessment:

1. Keep Split CIFAR-100 as the required benchmark.
2. Establish fine-tuning, random replay, and fixed-periodic replay before making final claims about the new method.
3. Add forgetting signals and prediction before building the spaced replay scheduler.
4. Treat MIR as a post-core stronger baseline, and treat expensive signals, Split CUB, and DistilBERT as later-stage additions, not core requirements.

## Research Goals This Plan Must Satisfy

1. Build a reliable continual-learning benchmark on Split CIFAR-100.
2. Measure forgetting clearly with final accuracy and average forgetting.
3. Test whether sample-level signals carry predictive value for forgetting risk.
4. Estimate or approximate sample-specific forgetting time `T_i` from those signals.
5. Use the risk estimate and `T_i` estimate to drive a spaced replay policy.
6. Show whether the policy improves retention under a fixed memory and compute budget.

## Plan Quality Verdict

The current plan is good because it asks a research question that can be tested with concrete experiments, keeps the core benchmark narrow, and puts baselines before the proposed method. Its main risk is not the idea; its main risk is sloppy execution. This project will fail scientifically if it produces a scheduler before it has a trustworthy task stream, accuracy matrix, random replay baseline, and sample-level logs.

The plan should therefore be treated as a staged evidence pipeline, not as a feature checklist. Each stage must produce artifacts that make the next stage harder to fool.

## Proposal Alignment Correction

The text-friendly proposal source [PROJECT_PROPOSAL_TEXT.md](./PROJECT_PROPOSAL_TEXT.md) makes one requirement sharper than the earlier PDF-based assessment:

```text
estimate forgetting time T_i -> schedule next replay at t + T_i
```

Therefore, the project must not treat a binary forgetting-risk score as the complete spaced-replay method. Task 9's risk predictor is a useful prerequisite, but the scheduler must either estimate `T_i` directly or define a defensible approximation from risk-over-horizon predictions.

## Current Proposal-Coverage Status After Task 21

Tasks 1 through 21 have built the full core scaffold, the first spacing-inspired
intervention, the first fair core comparison, a stronger MIR replay baseline,
the first event-triggered risk-gated replay variant, and the first systematic
learned-predictor comparison, signal-ablation analysis, and a learned-predictor
risk-gated replay intervention. Task 19 also tested learned-risk ranking under
the same replay budget as random replay. Task 20 tested a 50/50 learned-risk
plus class-balanced hybrid. Task 21 tested a gradient-norm diagnostic. The
project has not yet completed the proposal's full scientific mechanism.

What is already covered well enough for the current stage:

- Split CIFAR-100 task construction with stable sample IDs.
- Shared continual-training and seen-task evaluation loop.
- Fine-tuning and random replay pilot baselines.
- Fixed-periodic replay as an interval-ablation baseline.
- Loss, confidence, uncertainty, replay count, and last-replay-step logging.
- Binary and continuous sample-level forgetting labels.
- A leakage-safe first forgetting-risk predictor.
- Time-to-first-observed-forgetting targets with censoring metadata.
- A first spaced replay scheduler using an online due-time proxy.
- A three-seed core comparison across fine-tuning, random replay,
  budget-matched fixed-periodic replay, and spaced replay.
- A three-seed MIR stronger replay baseline under the same Split CIFAR-100
  model, memory budget, replay batch size, task split, and seed list.
- An event-triggered replay scheduler that skips replay unless samples satisfy
  configured risk and/or due-time gates.
- A one-seed Split CIFAR-100 risk-gated replay pilot showing that sparse replay
  saves replay samples but does not yet preserve retention.
- Learned binary forgetting predictors from the proposal. On informative
  seed-0 replay artifacts, logistic regression and linear SVM classifiers beat
  the best cheap heuristic by average precision.
- Signal ablations showing that `all_features` is best and `history_summary` is
  the strongest compact group on the informative seed-0 replay artifacts.
- A learned-predictor replay gate that uses prior-artifact logistic regression
  probabilities online and logs the source artifacts used to train the gate.
- A fixed-budget learned-risk replay selector that uses the learned predictor
  to rank memory examples while matching random replay's replay sample count.
- A balanced hybrid replay selector that combines learned-risk replay with
  class-balanced replay under the same replay sample count.
- A final-layer gradient-norm diagnostic that measures an expensive
  proposal-listed signal and evaluates whether it improves leakage-safe
  forgetting prediction.
- A consolidated results analysis and retrospective in
  [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md).

What is not yet covered:

- a learned or empirically validated `T_i` estimator that beats a constant
  median timing baseline;
- a successful direct test that cognitive-spacing style replay timing improves
  retention over random replay;
- a successful online scheduler variant that turns the learned predictor into
  better retention;
- representation or latent drift signals.

Task 13 is the first fair repeated-seed comparison under protocol
`core_split_cifar100_v2`. It is final enough to support the limited claim that
the first spaced replay proxy does not beat random replay under the tested
controls. It is not enough to claim that all spacing-inspired replay ideas fail,
because the learned or validated `T_i` estimator is still weak.

Task 14 adds MIR as a stronger replay comparator. MIR beats random replay and
the first spaced replay proxy on the current three-seed Split CIFAR-100
comparison, so the project should now treat MIR as the strongest implemented
performance reference. This strengthens the study, but it also raises the bar:
future spaced-replay variants should be compared against MIR after they first
beat random and fixed-periodic replay.

Task 15 added event-triggered risk-gated replay. The primary one-seed pilot
used `2071` replay samples, a large reduction from the random replay budget of
`45216`, but final accuracy stayed near fine-tuning. This is a clean negative
or diagnostic result: the mechanism can skip replay, but the current cheap
online risk gate is not enough to retain old tasks.

Task 16 added learned predictor comparisons. On the random replay seed-0
artifact, logistic regression improves average precision from the best cheap
heuristic's `0.8471253916174554` to `0.9083240127221096`; on the spaced replay
seed-0 artifact, it improves from `0.8267285181601283` to
`0.9188844673560967`. These are offline predictor diagnostics, not yet online
scheduler evidence.

Task 17 added signal ablations. The full feature set is best on both
informative replay artifacts, while `history_summary` is the strongest compact
group. This supports testing a learned-predictor replay gate next, with strict
temporal safeguards.

Task 18 implemented that learned-predictor replay gate. The result is negative:
the learned gate runs without future-label leakage, but seed-0 pilots reach only
`0.0376` to `0.0379` final accuracy, below random replay, MIR, and the cheap
risk-gated replay result.

After Task 18, the plan was adjusted in
[FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md).
The key correction is to test learned-risk ranking under a fixed replay budget
before adding expensive signals. This preserves the proposal's forgetting-aware
replay idea while avoiding a premature conclusion from sparse gating alone.

Task 19 implemented that fixed-budget learned-risk test. The seed-0 pilot used
the same `45216` replay samples as random replay, selected all replay examples
by learned risk, and skipped no replay opportunities. It reached final accuracy
`0.0759` and average forgetting `0.3587777777777778`, worse than random replay
seed 0 (`0.10129999999999999`, `0.30433333333333334`) and MIR seed 0
(`0.1183`, `0.21400000000000002`). This is a clean diagnostic result: learned
risk ranking alone is not yet a better replay selector than random replay.
Task 20 implemented a 50/50 learned-risk plus class-balanced replay policy. The
seed-0 pilot used the same `45216` replay samples as random replay and improved
over pure learned fixed-budget replay, with final accuracy moving from `0.0759`
to `0.0879` and average forgetting moving from `0.3587777777777778` to
`0.3428888888888889`. It still did not beat random replay seed 0
(`0.10129999999999999`, `0.30433333333333334`) or MIR seed 0 (`0.1183`,
`0.21400000000000002`). This is mixed but still negative: diversity helps the
learned selector, but the current learned-risk signal is still not a better
online replay intervention than random replay.

Task 21 implemented a final-layer gradient-norm diagnostic. It produced `55000`
gradient signal rows on random replay seed 0 and added about `57%` measured
training-time overhead plus a `42909732` byte artifact. Gradient-only prediction
reached average precision `0.8386703932509996`; cheap all-features prediction
reached `0.9083240127221096`; cheap plus gradient reached
`0.9080918805327551`. This is a negative diagnostic: this gradient signal is
measurable, but it does not improve the predictor enough to justify a new
gradient-based replay scheduler.

## Two Research Claims Under Test

The proposal contains two related but separate claims. The action plan must preserve this separation through implementation and reporting.

| Claim | Question | Required Evidence | Valid Conclusion If It Fails |
| --- | --- | --- | --- |
| Predictive claim | Can cheap sample-level signals predict future forgetting risk better than chance? | Forgetting labels derived from later evaluations, temporally valid train/test splits for the predictor, precision-recall analysis, and signal ablations. | The signals used in this project are not sufficient predictors under the tested setup. Do not claim forgetting-aware scheduling is supported by prediction evidence. |
| Timing claim | Can those signals estimate or approximate when a retained sample will first be forgotten? | Time-to-first-forgetting targets, censoring rules for samples not forgotten by the final evaluation, temporal validation, and timing-error metrics. | The project can still study risk-guided replay, but it must not claim the proposal's `T_i` scheduling mechanism is supported. |
| Intervention claim | If forgetting risk or `T_i` can be estimated, does using it for replay improve continual learning? | Matched-budget comparison of fine-tuning, random replay, fixed-periodic replay, and spaced replay across repeated seeds using final accuracy and average forgetting. | Prediction may be real but not operationally useful, or the scheduler may need redesign. Do not hide this by emphasizing only predictor metrics. |

The strongest outcome is when both claims hold. A mixed outcome is still publishable as a clean negative or diagnostic result if the evidence is honest.

## Research Rigor Rules

These rules are non-negotiable. They exist to keep the project from producing a polished but scientifically weak result.

1. Every method comparison must use the same task order, model architecture, optimizer, replay memory budget, replay batch budget, epochs per task, preprocessing, and seed list unless the comparison explicitly studies one of those variables.
2. No result is valid unless the run config, seed, task split, method name, metric outputs, and accuracy matrix are saved together.
3. The project must report negative or mixed results honestly. If spaced replay fails to beat random replay, the result is still useful if the evidence is clean.
4. The project must separate three claims: whether signals predict forgetting, whether the scheduler improves retention, and whether the improvement is worth the extra compute.
5. Stretch work cannot redefine success. Split CUB, DistilBERT, gradient norms, and representation drift only strengthen the study after the Split CIFAR-100 core result exists.
6. Do not claim a cognitive-spacing mechanism unless the implemented scheduler actually tests a spacing rule. Otherwise, describe the method as risk-guided replay with spacing-inspired constraints.

## Decisions To Freeze Before Core Runs

The protocol has locked the research shape, but the following implementation choices still need fixed values before any baseline result can be treated as evidence:

- class order and task split seed
- model architecture
- optimizer, learning rate, weight decay, scheduler, and batch size
- data preprocessing and augmentation policy
- epochs or update steps per task
- replay buffer capacity
- replay batch size or replay ratio
- memory insertion policy
- seed list for repeated runs
- device policy, including CPU/CUDA fallback behavior
- output directory structure and artifact schema

These values should live in versioned config files, not only in code or notebook cells. Once baseline runs begin, changing any of them requires either rerunning all compared methods or creating a new protocol/config version.

## Core Measurement Definitions

The first implementation should use one consistent accuracy matrix `A`, where `A[t, i]` is the accuracy on task `i` after finishing training task `t`. Entries where `i > t` are not part of the seen-task evaluation.

- `final_accuracy`: mean accuracy over all tasks after the final task is trained.
- `average_accuracy_after_task_t`: mean accuracy over tasks seen by step `t`.
- `forgetting_for_task_i`: best previous accuracy on task `i` after it was learned minus final accuracy on task `i`.
- `average_forgetting`: mean `forgetting_for_task_i` over tasks that have a later task after them. The final task is excluded because it has no later interference period.
- `replay_utilization`: distribution of how often stored examples are replayed, including examples never replayed.
- `training_time_seconds`: wall-clock training time measured consistently per method.

If later work adds class-incremental versus task-incremental evaluation variants, those variants must be named explicitly and not mixed in one table.

## Recommended Implementation Shape

The code should grow around stable research interfaces rather than one-off scripts. The first implementation can stay simple, but the module boundaries should leave room for stronger baselines and later datasets.

```text
src/
  data/             Split CIFAR-100 task stream and sample metadata
  models/           model factory and device-safe model construction
  training/         method-agnostic continual training and evaluation loop
  replay/           replay buffers and replay selection policies
  signals/          loss, uncertainty, and optional expensive signal extractors
  predictors/       forgetting-risk heuristics and learned predictors
  metrics/          accuracy matrix, forgetting, utilization, timing
  logging/          run artifact writers and schema helpers
  utils/            seeds, devices, config helpers
configs/
  protocols/        locked research contracts
  experiments/      runnable experiment configs
tests/
  data/             deterministic task stream and sample-id tests
  metrics/          metric formula tests
  replay/           buffer capacity and sampling tests
```

Every major method should be selectable by config. The trainer should not need method-specific branches beyond calling a replay policy or signal extractor interface.

## Sample-Level Data Contract

Sample-level tracking is central to the research question, so it must be designed before signal logging begins.

- Each original CIFAR-100 example needs a stable `sample_id` that survives task filtering, dataloader shuffling, replay-buffer storage, and evaluation.
- Each training or replay observation should be able to record `sample_id`, task id, class id, global step, loss, predicted class, confidence or uncertainty, whether it came from replay, replay count, and last replay step.
- Replay-buffer entries should store enough metadata to support later scheduler changes without rewriting the dataset pipeline.
- Signal logs should be append-only run artifacts. Analysis code should derive labels and features from logs rather than mutating the original logs.

The minimum first-pass forgetting target should be binary and auditable: an example was previously classified correctly at an evaluation point and becomes incorrectly classified after later task training. A continuous target, such as increase in loss or drop in confidence, can be added as a secondary analysis.

## Method Acceptance Gates

| Method or Component | Minimum Acceptance Criteria Before Moving On |
| --- | --- |
| Split CIFAR-100 stream | Produces 10 tasks with 10 disjoint classes each; exposes stable sample IDs; has tests for class coverage, no class overlap across tasks, deterministic ordering under a fixed seed, and expected train/test counts. |
| Core trainer | Can train sequentially over task loaders; evaluates all seen tasks after each task; emits an accuracy matrix; runs on CPU and uses CUDA if available without changing results format. |
| Metrics and logging | Saves run config, seed, method name, accuracy matrix, final accuracy, average forgetting, and timing in a predictable artifact directory. |
| Fine-tuning baseline | Runs without replay and establishes measurable forgetting on Split CIFAR-100. If it does not show forgetting, the benchmark or training setup must be investigated before replay methods are trusted. |
| Random replay baseline | Uses the same training budget and memory budget as spaced replay; exposes buffer occupancy and replay utilization; is strong enough to be a fair baseline, not a strawman. |
| Fixed-periodic replay baseline | Replays from the same buffer at a fixed interval such as every `k` optimizer steps; exposes interval, replay batch size, replay utilization, and total replay samples. It must be budget-matched before being used as a scheduler comparator. |
| Signal logging | Logs loss history and predictive uncertainty with sample IDs; verifies that logs can be joined back to task, class, and replay metadata. |
| Forgetting labels | Produces reproducible labels or continuous targets from future evaluation changes; documents exactly which evaluation windows define the target. |
| Risk predictor or heuristic | Beats chance on a held-out temporal split or clearly fails; reports precision-recall behavior, not only accuracy, because forgetting-risk labels may be imbalanced. If it fails, the scheduler can still be studied as an ablation, but the project must not claim successful forgetting prediction. |
| Time-to-forgetting estimator | Defines first-forgetting time targets, handles right-censored samples that do not forget by the final evaluation, and estimates `T_i` or a due-time proxy without future leakage. |
| Spaced replay scheduler | Uses only information available at scheduling time; respects the same buffer and replay budget as random replay; logs why each replay sample was selected, including estimated `T_i` or replay due step. |
| Core comparison | Runs fine-tuning, random replay, fixed-periodic replay, and spaced replay with matched controls and repeated seeds; reports mean and variability, not only best runs. |
| Stronger replay baseline | Implements MIR or another literature-grounded replay selector with the same memory budget, replay batch size, task split, and seed list; logs selection scores and candidate counts without counting candidates as replay samples. |
| Event-triggered risk-gated replay | Replays only when a sample is predicted to be near forgetting, otherwise skips replay. It must log risk thresholds, skipped replay opportunities, selected sample IDs, replay-sample savings, final accuracy, average forgetting, and training time. |
| Learned forgetting predictor suite | Evaluates proposal-suggested predictors such as logistic regression, linear models, and support-vector machines on leakage-safe temporal splits. It must report average precision, ROC-AUC where valid, calibration or threshold behavior, and whether any learned model beats the current heuristic. |
| Ablations | Tests whether improvement comes from risk scoring, spacing constraints, or extra replay frequency. |

## Phase Exit Checkpoints

These checkpoints are the practical version of the action plan. Do not move to a later phase just because code exists; move when the phase can produce its required evidence.

| Phase | Exit Criteria | Evidence Artifact |
| --- | --- | --- |
| Protocol lock | Research scope, required methods, metrics, and scope guards are versioned and validated. | Protocol YAML, protocol Markdown, validation tests. |
| Benchmark | Split CIFAR-100 tasks are deterministic, class-disjoint, sample-ID stable, and testable without relying on network downloads during unit tests. | Task-stream tests and a small synthetic fixture path. |
| Trainer and metrics | A method-agnostic trainer can evaluate all seen tasks and produce a valid accuracy matrix. | Metric tests plus a tiny smoke run or fixture run. |
| Baselines | Fine-tuning, random replay, and fixed-periodic replay run under matched controls and expose measurable forgetting behavior. | Saved configs, accuracy matrices, replay utilization logs, and replay cadence metadata. |
| Signals and labels | Loss/uncertainty logs can be joined to stable sample IDs and converted into future-forgetting targets. | Signal log schema, label-generation script, label sanity report. |
| Predictor | Predictor evaluation uses only past information to predict later forgetting and reports PR behavior. | Predictor metrics, signal ablations, temporal split description. |
| Time-to-forgetting | First-forgetting time or due-step targets are defined and evaluated with leakage-safe features. | `T_i` target artifact, censoring report, timing-error metrics. |
| Scheduler | Spaced replay uses predicted risk, estimated `T_i`, and spacing state without future leakage. | Scheduler selection logs and matched-budget comparison configs. |
| Core experiment | All core methods run across the same seed list and report mean and variability. | Final tables, plots, configs, and reproducibility notes. |
| Stronger baseline | MIR or an equivalent strong replay selector runs under the same core controls and is compared against random and spaced replay. | Stronger-baseline summary, selection trace, and reproducibility notes. |
| Event-triggered scheduler | A risk-gated replay variant skips replay when no memory item is predicted to be near forgetting and reports the retention/compute trade-off. | Risk-gated scheduler trace, skipped-step counts, replay-sample savings, and comparison summary. |
| Learned predictors | Proposal-suggested learned predictors are compared against heuristic risk scores before they are trusted as scheduler inputs. | Predictor comparison report with temporal splits and leakage checks. |
| Fixed-budget learned replay | Learned risk is used to rank/select replay items while keeping the same replay sample budget as random replay. | Matched-budget learned-risk replay report with final accuracy, average forgetting, replay utilization, and comparison against random replay and MIR. |
| Balanced hybrid replay | Learned-risk selection is combined with random or class-balanced replay so diversity is protected. | Hybrid replay report separating learned-risk benefit from diversity benefit. |

## Ordered Action Plan

| Order | Action Item | Why It Is In This Position | How It Contributes to the Research Goals |
| --- | --- | --- | --- |
| 1 | Lock the scope, benchmark, and evaluation protocol | This must come first because the proposal is strongest when scoped to Split CIFAR-100, fine-tuning, random replay, forgetting signals, and spaced replay. If the scope is not fixed at the start, later engineering work can drift into stretch work before the core question is answered. | It protects Goal 1 and Goal 5 by ensuring every later decision serves the main experiment instead of diluting it. |
| 2 | Implement the Split CIFAR-100 task stream with stable sample IDs | This comes after scope lock because the benchmark definition determines the task order, class partitioning, and sample tracking strategy. It comes before trainer work because the trainer cannot be designed correctly without knowing how tasks and samples are represented. | It directly supports Goal 1 and Goal 3 by creating the sequential benchmark and the sample identity needed for signal logging and forgetting analysis. The implementation must include deterministic split construction and tests for task count, class coverage, class disjointness, and stable IDs. |
| 3 | Build the core continual training and evaluation loop | This comes after the task stream because the trainer must consume task-specific loaders. It comes before baselines because every baseline and proposed method will reuse the same loop, so this shared machinery should exist before method-specific logic is added. | It supports Goal 1 and Goal 2 by making sequential training and task-by-task evaluation possible. The loop must produce an accuracy matrix after each task and must not contain method-specific replay assumptions. |
| 4 | Add reproducible metrics and experiment logging | This comes after the trainer because metrics need model outputs and evaluation checkpoints. It comes before the first baseline run so that all evidence is collected consistently from the start instead of being added later in an inconsistent way. | It supports Goal 2 and Goal 5 by making final accuracy, average forgetting, runtime, and replay behavior comparable across methods. This step must define artifact schemas before large runs begin. |
| 5 | Run the naive sequential fine-tuning baseline | This comes after the benchmark and metrics are ready because it is the cleanest first measurement of catastrophic forgetting. It comes before any replay method because the team must first observe the unmitigated problem before claiming to reduce it. | It supports Goal 2 by establishing the no-replay reference point that shows how much forgetting the task stream actually induces. If forgetting is weak or absent, the training schedule and benchmark setup must be investigated before replay methods are added. |
| 6 | Add a bounded replay buffer and the random replay baseline | This comes after fine-tuning because random replay is the main standard baseline the proposal's method must beat. It comes before signal-driven methods because the project should first prove it can run a fair replay comparison under a fixed memory budget. | It supports Goal 2 and Goal 5 by establishing the first required replay comparator for the proposed spaced replay method. The replay buffer must expose insertion policy, capacity, sample IDs, replay counts, and deterministic behavior under a fixed seed. |
| 7 | Log the cheapest sample-level signals first: loss history, confidence or uncertainty, replay count, and last replay step | This comes after random replay because signal logging should be attached to a working continual-learning pipeline, not built in isolation. It comes before forgetting labels and prediction because these measurements are the raw material from which forgetting risk will be estimated. | It supports Goal 3 by creating the observable features needed to test whether forgetting can be predicted at the sample level. Logs must be joinable by `sample_id` and must distinguish current-task training from replay observations. |
| 8 | Define forgetting labels or targets from future performance changes | This comes after signal logging because forgetting labels depend on seeing whether examples become harder, less confident, or incorrect after later tasks. It comes before predictor design because a predictor cannot be evaluated unless forgetting has been defined operationally. | It supports Goal 3 by turning the abstract idea of forgetting into a measurable prediction target. The first label should be binary and auditable; continuous loss or confidence deltas can be secondary targets. |
| 9 | Build a simple forgetting-risk predictor or heuristic score | This comes after signals and labels because prediction requires both input features and a target definition. It comes before the scheduler because the spaced replay policy is supposed to act on estimated forgetting risk rather than on raw, unstructured telemetry. | It directly supports Goal 3 and Goal 4 by testing whether the proposed signals are informative enough to guide replay decisions. The predictor must be evaluated on temporally later data so it does not learn from the future. |
| 10 | Add the fixed-periodic replay baseline: replay uniformly from memory every `k` optimizer steps | This comes after the predictor because the predictor clarifies what the proposed scheduler will try to improve, but before the spaced scheduler comparison because the proposal explicitly asks whether the new method beats fixed replay schedules. | It supports Goal 6 by adding the missing timing baseline. The implementation must log `replay_interval`, replay events, skipped steps, total replay samples, and whether the run is compute-matched or frequency-ablation mode. |
| 11 | Define and evaluate sample-specific time-to-forgetting `T_i` targets | This comes after risk prediction and fixed-periodic replay because the proposal's scheduler is not merely a risk ranker; it schedules replay near an estimated forgetting time. It comes before the scheduler so the due-time rule has a documented target. | It supports Goal 4 by converting future forgetting observations into first-forgetting time, step delta, task delta, and censoring labels. It must document how `T_i` is estimated when exact failure time is interval-censored between evaluations. |
| 12 | Implement the spaced replay scheduler driven by predicted risk, estimated `T_i`, and spacing rules | This comes after the predictor, fixed-periodic baseline, and `T_i` estimator because the scheduler is the intervention that uses the risk and timing estimates. It comes before broader comparisons because the core proposed method has to exist before it can be evaluated against the baselines. | It directly supports Goal 5 and Goal 6 by turning forgetting prediction into an actual replay policy. The scheduler must log selected sample IDs, risk scores, estimated `T_i`, due steps, spacing state, and selection reason. |
| 13 | Run the core controlled experiment: fine-tuning vs random replay vs fixed-periodic replay vs spaced replay under the same budget | This comes after all four core methods exist because this is the first decisive test of the proposal's main claim. It comes before stronger baselines and stretch work so that the project answers the primary research question as early as possible. | It is the main test of Goal 6 because it determines whether spaced replay improves retention under matched conditions. The report must include mean and variability across seeds, not only a single run. |
| 14 | Add a stronger replay baseline such as MIR, and optionally uncertainty-guided replay | This comes after the core comparison because stronger baselines increase credibility, but they should not delay the first answer to the main question. It comes before stretch datasets because methodological strength matters more than benchmark expansion. | It strengthens Goal 6 by showing whether spaced replay is only better than random or fixed-periodic replay, or also competitive with stronger replay policies. Any stronger baseline must obey the same artifact and budget rules as the core methods. |
| 15 | Implement and test event-triggered risk-gated replay | This comes immediately after MIR because the first spaced scheduler failed while using the full replay budget, and MIR showed that sample choice matters. The next useful scheduler should ask whether replay can be skipped unless a sample is predicted to be near forgetting. | It supports Goal 5 and Goal 6 by testing a compute-efficient intervention: replay only high-risk or due-and-high-risk examples, log skipped replay opportunities, and compare retention against random replay, fixed-periodic replay, spaced replay, and MIR under the same Split CIFAR-100 controls. |
| 16 | Add learned forgetting predictors from the proposal: logistic regression, linear models, and support-vector machines | This comes after the risk-gated scheduler because the first implementation can use the existing cheap online heuristic, while this task tests whether the proposal's learned predictors provide better risk estimates for future scheduler variants. These models are feasible because the repository already depends on scikit-learn and already has leakage-safe feature/label artifacts. | It supports Goal 3 and Goal 4 by comparing learned predictors against the current heuristic on temporal splits. Logistic regression should predict binary forgetting risk, linear regression can target continuous deterioration or observed time-to-forgetting, and SVM variants can test nonlinear margins if runtime remains practical. |
| 17 | Run signal ablations and predictor-quality analysis | This now follows the event-triggered scheduler and learned-predictor suite so ablations can diagnose both the first heuristic scheduler and any learned model improvements. | It supports Goal 3, Goal 4, and Goal 5 by revealing which signals matter, whether the predictor has real precision-recall value, whether `T_i` estimates are useful, and how much each component contributes. Required ablations should separate loss-only, uncertainty-only, target-probability-only, loss-increase-only, combined signal, risk-only replay, due-only replay, risk-gated replay, fixed-periodic timing, and spacing-only scheduling where feasible. |
| 18 | Implement a learned-predictor risk-gated replay variant | This comes after Task 17 because ablations show the learned offline ranking is stronger than the cheap heuristic and identify `all_features` as the best gate input. It should come before expensive signals so the project first tests whether the current cheap signals are operationally useful. | It directly tests Goal 5 and Goal 6 by replacing the cheap online risk score in Task 15 with a temporally safe learned predictor, then measuring replay savings, final accuracy, average forgetting, and comparison against random replay, fixed replay, spaced replay, and MIR. |
| 19 | Implement fixed-budget learned-risk replay | This comes immediately after the failed learned sparse gate because Task 18 mixed two ideas: learned example ranking and replay skipping. Before adding expensive signals, the project must test whether learned risk helps choose examples when the replay budget is held equal to random replay. | It directly supports Goal 5 and Goal 6 by isolating the value of learned-risk ranking under matched compute. If it beats random replay, the learned predictor is operationally useful and sparse gating was the weak link. If it fails, the predictor is not yet useful for replay selection. |
| 20 | Implement a balanced hybrid of learned-risk replay plus random or class-balanced replay | This follows fixed-budget learned-risk replay because pure risk ranking may over-focus on hard or noisy examples. The hybrid tests whether learned risk helps when replay diversity is protected. | It supports Goal 5 and Goal 6 by separating risk-scoring benefit from diversity and class/task balance. It should log the learned-risk fraction, random or balanced fraction, class/task coverage, replay utilization, final accuracy, and average forgetting. |
| 21 | Add expensive signals such as gradient norms or representation drift only if runtime allows | This comes after fixed-budget and hybrid learned-risk tests because expensive instrumentation should be justified by a remaining prediction or scheduling weakness. It should refine the study, not bypass the simpler replay-selection diagnosis. | It deepens Goal 3 and Goal 4 by testing whether more complex signals improve forgetting prediction or `T_i` estimation enough to justify their cost. These signals must report added runtime or memory overhead so the project can judge whether they are worth using. |
| 22 | Run a decision checkpoint and small targeted rescue ablations before any stretch benchmark | This is inserted after Task 21 because the gradient diagnostic did not improve prediction and no risk-guided replay method beats random replay yet. Stretch benchmarks would not answer the core failure. | It protects Goal 6 by deciding whether to stop with a clean negative result or run cheap final ablations such as lower learned-risk fractions, random-diversity hybrids, or a pure class-balanced replay baseline. |
| 23 | If continuing method development, test MIR-like current-interference or representation-drift diagnostics | This follows the decision checkpoint because MIR is the strongest method and suggests that update-specific interference is more relevant than general future-risk ranking. | It refines Goal 3 and Goal 5 by testing signals more directly tied to the replay decision. This should remain diagnostic until it beats cheap features or explains MIR's advantage. |
| 24 | Attempt stretch benchmarks such as Split CUB or DistilBERT only after the Split CIFAR-100 conclusion is stable | This moves behind the decision checkpoint because the assessment identified stretch benchmarks as scope risk. They should expand the claim only after the project has a clear primary-benchmark conclusion. | It broadens Goal 1 and Goal 6 only if the core result is worth generalizing. A stretch benchmark must create a new protocol version instead of silently altering the core study. |
| 25 | Synthesize the final results into tables, plots, and a written argument tied back to the research question | This comes last because interpretation should follow evidence, not lead it. It depends on the earlier steps because only a complete set of baselines, metrics, predictor results, timing estimates, replay interventions, expensive-signal diagnostics, and retrospective decisions can support a convincing conclusion. | It closes all six goals by turning the experimental evidence into a clear answer to the research question and a defensible final project report. The final writeup must distinguish supported conclusions from plausible explanations. |

## Sequence Logic in One Line

The action order is intentionally:

`scope -> benchmark -> trainer -> metrics -> fine-tuning baseline -> random replay baseline -> signal logging -> forgetting labels -> risk predictor -> fixed-periodic replay -> T_i estimator -> spaced replay -> core comparison -> stronger baselines -> event-triggered replay -> learned predictors -> ablations -> learned replay gate -> fixed-budget learned replay -> balanced hybrid replay -> expensive signals -> decision checkpoint -> optional interference or drift diagnostics -> stretch benchmarks -> final report`

That order mirrors the causal structure of the research question:

1. First create a trustworthy continual-learning setting.
2. Then measure forgetting without and with standard replay.
3. Then test whether forgetting can be predicted.
4. Then estimate when samples are likely to fail.
5. Then use risk and time estimates to drive spaced replay.
6. Then test whether replay can be skipped when no sample appears near forgetting.
7. Then check whether learned predictors improve the risk signal.
8. Then test whether the learned predictor improves replay gating.
9. Then test whether learned risk helps selection under a fixed replay budget.
10. Then test whether learned risk helps when diversity is protected.
11. Then test whether expensive signals improve prediction enough to justify
    another scheduler.
12. Then decide whether to stop with a clean negative result or run small,
    targeted rescue ablations.
13. Then check whether any new policy actually improves retention.

## Priority Tiers

### Required Core

- Split CIFAR-100
- fine-tuning baseline
- random replay baseline
- fixed-periodic replay baseline
- cheap signals such as loss and uncertainty
- spaced replay scheduler
- final accuracy and average forgetting

Required core work is the minimum publishable project for this repository. It is not optional, and no stretch result can compensate for a missing or weak required core comparison.

### Strong Contribution

- forgetting-risk heuristic or lightweight predictor
- time-to-forgetting `T_i` estimator
- event-triggered risk-gated replay scheduler
- learned forgetting predictors such as logistic regression, linear regression, and support-vector machines
- learned-predictor replay gate
- fixed-budget learned-risk replay
- balanced hybrid replay
- final-layer gradient-norm diagnostic
- controlled memory-budget comparison
- predictor precision-recall analysis
- signal ablations

Strong-contribution work is what turns the project from an engineering demo into a research answer. In particular, predictor analysis is necessary because a scheduler can improve by accident, but the proposal specifically asks whether sample-level signals are predictive.

### Credibility Boosters

- MIR baseline (implemented in Task 14)
- uncertainty-guided replay baseline
- replay utilization analysis

Credibility boosters should be added only after the random replay comparison is working. They are valuable because beating random replay is useful but not enough to show competitiveness against stronger replay literature.

### Stretch Only

- additional representation drift or full-model gradient diagnostics if runtime permits
- Split CUB
- DistilBERT sequential domain adaptation

Stretch work must remain isolated from the core claim. If it is attempted, it should be reported as preliminary unless it receives the same control, seed, and artifact discipline as the core Split CIFAR-100 experiment.

## Failure Conditions

The plan should be treated as failing, or at least not yet ready for final claims, if any of the following happen:

- comparisons use different memory budgets or training budgets
- random replay is underimplemented or weaker than a normal experience-replay baseline
- fixed-periodic replay is omitted from the main scheduler comparison
- spaced replay does not estimate or approximate `T_i`
- spaced replay uses future information when selecting samples
- metrics are computed from inconsistent evaluation schedules
- the project reports only best runs instead of repeated-seed summaries
- sample-level logs cannot be traced back to stable sample IDs
- expensive signals or stretch benchmarks delay the core comparison
- the final writeup claims generalization to language or broad AI systems without evidence from those settings

## Immediate Next Implementation Step

Task 9 is implemented and documented in [FORGETTING_RISK_PREDICTOR.md](./FORGETTING_RISK_PREDICTOR.md).
Task 10 is implemented and documented in [FIXED_PERIODIC_REPLAY_BASELINE.md](./FIXED_PERIODIC_REPLAY_BASELINE.md).
Task 11 is implemented and documented in [TIME_TO_FORGETTING_TARGETS.md](./TIME_TO_FORGETTING_TARGETS.md).
Task 12 is implemented and documented in [SPACED_REPLAY_SCHEDULER.md](./SPACED_REPLAY_SCHEDULER.md).
Task 13 is implemented and documented in [CORE_COMPARISON_TASK13.md](./CORE_COMPARISON_TASK13.md).
Task 14 is implemented and documented in [MIR_REPLAY_BASELINE.md](./MIR_REPLAY_BASELINE.md).
Task 15 is implemented and documented in [RISK_GATED_REPLAY_SCHEDULER.md](./RISK_GATED_REPLAY_SCHEDULER.md).
Task 16 is implemented and documented in [LEARNED_FORGETTING_PREDICTORS.md](./LEARNED_FORGETTING_PREDICTORS.md).
Task 17 is implemented and documented in [SIGNAL_ABLATIONS_TASK17.md](./SIGNAL_ABLATIONS_TASK17.md).
Task 18 is implemented and documented in [LEARNED_RISK_GATED_REPLAY_TASK18.md](./LEARNED_RISK_GATED_REPLAY_TASK18.md).
Task 19 is implemented and documented in [LEARNED_FIXED_BUDGET_REPLAY_TASK19.md](./LEARNED_FIXED_BUDGET_REPLAY_TASK19.md).
Task 20 is implemented and documented in [LEARNED_HYBRID_REPLAY_TASK20.md](./LEARNED_HYBRID_REPLAY_TASK20.md).
Task 21 is implemented and documented in [GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md](./GRADIENT_SIGNAL_DIAGNOSTIC_TASK21.md).
The current cross-task analysis and retrospective are documented in [RESULTS_ANALYSIS_RETROSPECTIVE.md](./RESULTS_ANALYSIS_RETROSPECTIVE.md).
The post-Task-18 plan adjustment is documented in [FIXED_BUDGET_LEARNED_REPLAY_PLAN.md](./FIXED_BUDGET_LEARNED_REPLAY_PLAN.md).

The next task is now:

`22 - Run a decision checkpoint and small targeted rescue ablations before any stretch benchmark`

Task-22 rescue-ablation configs have been staged in
[configs/experiments](../configs/experiments):

- `learned_hybrid_replay_task22_frac025_class_balanced_split_cifar100.yaml`
- `learned_hybrid_replay_task22_frac025_random_split_cifar100.yaml`
- `learned_hybrid_replay_task22_class_balanced_only_split_cifar100.yaml`

The next implementation unit should run the following:

1. a small ablation config for lower learned-risk fractions, starting with
   `0.25` learned risk and `0.75` class-balanced or random replay;
2. optionally, a pure class-balanced replay baseline to separate diversity from
   learned risk;
3. a one-seed comparison against random replay, learned fixed-budget replay,
   learned hybrid 50/50, and MIR;
4. a decision note stating whether to stop with the clean negative result or
   pivot toward MIR-like current-interference or representation-drift
   diagnostics.

Because Task 13 produced a clean negative result for the first spaced scheduler
and Task 14 shows MIR is stronger than random replay, Task 15 tested whether
skipping low-risk replay can improve the retention/compute trade-off. The first
result was negative: sparse risk-gated replay saved replay samples but did not
preserve retention. Task 16 showed learned predictors can improve offline risk
ranking on informative replay artifacts. Task 17 found that the full cheap
feature set is best and recommended testing a learned replay gate next. Task 18
implemented that learned sparse gate, but the seed-0 pilots were negative. The
next research step was to separate learned-risk ranking from sparse replay
skipping. Task 19 completed that test and found that pure learned-risk ranking
is still worse than random replay under the same replay budget. Task 20 tested a
balanced hybrid selector and found that diversity helps, but the hybrid still
does not beat random replay. The next research step can now move to expensive
signals, but only as a diagnostic for why the current cheap learned-risk signal
is not intervention-effective. Task 21 tested final-layer gradient norms and
found that they do not improve prediction over cheap features enough to justify
a new replay scheduler. The plan should therefore insert a decision checkpoint
before stretch benchmarks.

## Definition of Done

The plan is complete when the repository can produce a documented experiment that:

1. Trains sequentially on Split CIFAR-100.
2. Compares fine-tuning, random replay, fixed-periodic replay, and spaced replay under the same memory budget.
3. Reports final accuracy and average forgetting.
4. Shows whether the chosen sample-level signals predict forgetting better than chance.
5. Shows whether the project can estimate or approximate sample-specific forgetting time `T_i`.
6. Explains, through ablations or stronger baselines, why the final result is scientifically credible.

The stronger definition of done is that another researcher can rerun the core experiment from the saved configs and reproduce the same tables within expected seed variability.
