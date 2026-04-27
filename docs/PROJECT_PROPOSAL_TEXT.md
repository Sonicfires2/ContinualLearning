# Extracted Project Proposal Text

Source: [Project_Proposal-2.tex](../Project_Proposal-2.tex)  
Extraction date: 2026-04-24  
Note: This is a cleaned Markdown extraction of the proposal text. LaTeX commands,
formatting, and the bibliography block were simplified; the scientific content
and section order were preserved.

# Spaced-Replay Continual Finetuning: Predicting Forgetting and Scheduling Adaptive Replay in Large Language System

Authors: Minh Nguyen and Ngoc Mai Pham

## Abstract

Catastrophic forgetting remains a fundamental obstacle for AI systems in general
and LLM-system specifically trained on non-stationary data streams. Although
experience replay buffers reduce forgetting, they scale poorly and often rely on
naive sampling. Inspired by cognitive science, where spaced practice and
forgetting curves guide review schedules, we propose to study continual
finetuning under limited compute and memory.

The project seeks to:

1. measure sample-level forgetting curves during sequential finetuning; and
2. design spacing-inspired replay policies that predict when each sample or
   concept is at risk of being forgotten using uncertainty, gradient norms, loss
   trajectories, and representation drift.

We will evaluate whether adaptive spaced replay on small models can deliver
better long-term retention than existing baselines with the same memory and
limited compute budget.

## Introduction

Continual learning aims to endow neural networks with the ability to accumulate
knowledge over a sequence of tasks without catastrophic forgetting. Most deep
networks are trained in a static, one-shot setting and struggle to maintain
performance when exposed to non-IID data streams. The stability and plasticity
trade-off is still an open challenge.

Rehearsal-based methods, such as Experience Replay, alleviate forgetting by
maintaining a buffer of past examples and interleaving them with new data. While
effective, these methods typically rely on random sampling and require large
buffers to guarantee good coverage, which is impractical for
resource-constrained devices.

In contrast, cognitive psychology has long demonstrated that spaced retrieval
practice, the art of reviewing material at increasing intervals, produces
durable and long-lasting learning and reduces the need for massed review.
Complementary learning systems theory posits that rapid encoding and slow
neocortical consolidation rely on replay of old memories to integrate new
knowledge without interference.

Recent continual-learning research has begun to explore representation drift,
gradient interference, uncertainty-based sample selection, and learning-speed
signals to prioritize which data point should be retrained. However, the
proposal argues that these techniques have not explicitly targeted forgetting
curves or the temporal spacing of retraining data points.

## Scope

This project focuses on continual finetuning: the stage where a pretrained
model is sequentially adapted to downstream tasks or domains. The proposal
restricts experiments to moderately sized models, such as ResNet-18 or
DistilBERT, that can be trained efficiently on an Apple M3 Pro laptop.

Rather than pursuing generative replay, the proposal assumes access to a
bounded buffer and investigates whether sample-level signals can predict future
forgetting under non-IID streams.

The proposal further states the central scheduling idea:

> Schedule replay sessions using a spacing rule: revisit each sample just
> before its predicted forgetting time, so computation is minimized while data
> retention is maximized.

## Proposed Contributions

1. Introduce a methodology to measure forgetting curves of individual samples
   and concepts during continual finetuning, characterizing how loss,
   confidence, and representations decay over time.
2. Evaluate four predictive signals: uncertainty, gradient norms, loss
   trajectories, and representation drift, to assess their ability to forecast
   imminent forgetting.
3. Propose a spaced-replay scheduler that adapts the replay interval of each
   memory based on predicted forgetting times and demonstrate improved accuracy
   and memory utilization on image and language benchmarks under fixed buffer
   and compute budgets.
4. Provide an open-source implementation and analysis of continual finetuning on
   Apple hardware, highlighting practical considerations for edge devices with
   limited capabilities.

## Visual Abstract

The proposal describes the pipeline as:

```text
Non-IID task stream -> Replay buffer -> Signals L,U,G,D -> Predict forgetting -> Spaced replay
```

Goal:

```text
lower forgetting, better buffer use, same compute and memory budget
```

The figure caption states that the system should monitor replay-buffer samples,
predict when they will be forgotten, and replay them just before failure.

## Background

### Continual Learning

Continual learning is a paradigm where AI models learn new information, events,
and knowledge that did not exist in their original training data. This often
occurs from non-IID data streams where incoming samples arrive sequentially and
do not follow a fixed shuffled distribution.

Such forgetting can be studied at the sample level, where individual examples
are tracked, or at the concept level, where related samples such as classes or
semantic clusters are monitored together.

### Experience Replay and Representation Drift

Experience replay stores a small subset of previous data and jointly trains on
current and buffered samples. It often outperforms more complicated
regularization-based approaches.

However, experience replay does not explicitly mitigate representation drift:
when new classes arrive, latent features for older classes shift, causing data
to overlap and overall accuracy to degrade. The proposal cites work on ACE,
ESMER, and latent-drift replay as related approaches for reducing abrupt
representation change and prioritizing replay.

### Sample Selection Signals

The proposal identifies several signal families for selecting memories to
replay:

- Maximally Interfered Retrieval estimates how much a sample's loss would
  increase after an upcoming parameter update and retrieves samples with high
  gradient interference.
- Predictive uncertainty can measure a sample's position in decision space.
- Speed-based sampling observes that quickly learned examples are less prone to
  forgetting and selects samples based on learning speed.

The proposal's key claim is that these studies hint at forgetting curves, but do
not explicitly model or forecast when a sample will fail.

### Spacing and Complementary Learning Systems

Cognitive science shows that spaced practice, reviewing material after a delay,
often leads to better long-term retention than massed practice.

Optimal spacing intervals depend on desired retention time: subsequent reviews
should occur when retrieval becomes effortful but before memory is lost.

The proposal seeks to bring these insights into replay buffers for continual
finetuning.

### Generative Replay and Taxonomies

Generative replay uses a generator to synthesize past data rather than storing
raw samples. The proposal treats diffusion and federated generative replay as
large-scale but computationally heavy work outside its core scope.

The proposal focuses on replay as a complement to regularization and parameter
isolation methods.

## Approach

The proposal frames continual finetuning as a stream of tasks:

```text
D_1, ..., D_T
```

At time `t`, model parameters are updated by gradient descent on the current
minibatch and samples from replay buffer `B`.

Each buffered example `i` is associated with four signals:

- Loss trajectory `L_i(t)`: cross-entropy loss evaluated periodically during
  training. Rising loss signals potential forgetting.
- Uncertainty `U_i(t)`: predictive entropy or margin. Lower confidence than at
  learning time indicates increasing uncertainty.
- Gradient norm `G_i(t)`: L2 norm of the gradient for the sample without
  updating parameters. A large norm suggests retaining the sample requires
  significant parameter change.
- Representation drift `D_i(t)`: cosine distance between hidden activation at
  insertion time and current activation. High drift indicates internal
  representation movement.

## Predicting Forgetting

For each sample, the proposal defines a binary target variable:

```text
delta_i(t)
```

This indicates whether the sample will be forgotten within a fixed future
horizon `tau`, meaning it becomes misclassified or its loss increases beyond a
threshold.

The proposal collects feature vectors:

```text
(L_i(t), U_i(t), G_i(t), D_i(t))
```

and corresponding targets by monitoring the buffer during finetuning.

Proposed prediction methods include:

- linear regression;
- logistic regression;
- support-vector machines;
- gradient boosting models.

The proposal evaluates precision-recall on held-out streams and also considers
concept-level aggregation by class or semantic cluster.

## Spacing Replay Scheduler

This is the key scheduler definition from the proposal.

Given a predictor that outputs the probability of forgetting within `tau`, the
system derives a sample-specific review interval. From this, it computes an
estimated forgetting time:

```text
hat(T_i)
```

Then it schedules the next replay at:

```text
t + hat(T_i)
```

When the buffer is full, samples with the highest forgetting probability are
prioritized. Samples with low forgetting risk can have replay delayed or can
free capacity for new data.

The proposal says this adaptive scheduler should be compared to simple
heuristics, including sampling once every `k` iterations and random selection.

## Experiments

The proposal evaluates on continual-finetuning benchmarks:

- Split CIFAR-100;
- Split CUB;
- sequential domain adaptation of DistilBERT on sentiment datasets, if time
  permits.

Each task contains distinct classes or domains and is treated sequentially.

The proposal mentions implementing:

- Experience Replay;
- MIR;
- ESMER;
- uncertainty-based baselines;
- spaced replay scheduler.

## Evaluation Metrics

The proposal reports:

- final accuracy on all tasks;
- average forgetting, defined as the difference between best and last
  performance;
- computational cost, such as training time per task;
- buffer utilization, such as replay operations per iteration;
- area under the precision-recall curve for each forgetting predictor signal;
- ablations of each signal's contribution to the scheduler.

## Computational Resources

The proposal assumes a MacBook Pro with Apple M3 Pro hardware and PyTorch's
Metal backend. It proposes ResNet-18 for vision and DistilBERT for language,
with half-precision training and a replay buffer of a few hundred samples.

For this repository, the hardware assumption is treated as proposal context
rather than the execution environment because the current implementation is
Windows-first.

## Timeline

- Week 1-2: Reproduce Experience Replay baseline on Split CIFAR-100 and collect
  initial forgetting curves. Implement logging of loss, uncertainty, gradient
  norm, and representation drift.
- Week 3: Train predictive models for sample-level forgetting on collected data.
  Explore concept-level aggregation. Conduct ablation for individual signals.
- Week 4: Implement spacing-inspired replay scheduler. Integrate forgetting
  predictors into buffer management. Benchmark against MIR,
  uncertainty-based, and random sampling baselines. Conduct additional language
  experiments if time permits.
- Week 5: Prepare final report, including analysis of forgetting curves,
  predictor performance, and scheduling benefits.

## Extracted Proposal Requirements Most Relevant To The Repository

1. The project must measure sample-level forgetting curves, not only aggregate
   task forgetting.
2. The predictor should estimate whether a sample will be forgotten within a
   future horizon `tau`.
3. The scheduler should estimate a sample-specific forgetting time `hat(T_i)`.
4. The next replay should be scheduled at `t + hat(T_i)`.
5. The adaptive scheduler must be compared against random replay and fixed
   schedules such as replay once every `k` iterations.
6. The final claim is better long-term retention under fixed memory and compute
   budgets.
