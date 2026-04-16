# Action Plan: Spaced-Replay Continual Finetuning

## Purpose

This document turns the project proposal into an implementation and research plan. It is meant to be practical enough to guide coding work, but stable enough that the project can grow from a small Split CIFAR-100 experiment into harder vision and language experiments without rewriting the core system.

## Research Goal

Develop and evaluate spacing-inspired replay policies that reduce catastrophic forgetting during continual finetuning under limited memory and compute.

## Research Question

Can sample-level forgetting signals, such as uncertainty, loss history, gradient norms, and representation drift, predict when examples should be replayed better than random or fixed replay schedules?

## Why This Question Matters

Continual learning systems need to adapt to new tasks without losing useful knowledge from earlier tasks. Experience replay helps, but random replay often wastes memory and compute. If the system can predict which examples are at risk of being forgotten, it can replay fewer examples at better times and improve retention under the same resource budget.

## Working Hypotheses

1. Samples with rising loss, rising uncertainty, high gradient norm, or large representation drift are more likely to be forgotten later.
2. A replay policy that schedules samples using these signals will reduce average forgetting compared with random replay at the same buffer size.
3. The best signal set may depend on dataset difficulty, so the system should support ablations instead of hard-coding one signal.

## Core Terms

- Task stream: An ordered sequence of tasks, such as Split CIFAR-100 class groups.
- Sample ID: A stable identifier for an example across training, replay, logging, and analysis.
- Forgetting: The drop from a task's best historical accuracy to its later accuracy.
- Sample-level forgetting label: A binary or continuous target estimating whether a sample became harder or misclassified after later tasks.
- Replay buffer: A bounded memory of previous examples or features.
- Replay policy: The algorithm that decides which stored examples to replay.
- Spaced replay: A replay policy that schedules review times based on forgetting risk and spacing rules.

## Success Criteria

Minimum viable success:

- Run sequential finetuning on Split CIFAR-100.
- Compare fine-tuning, random experience replay, and spaced replay.
- Report final accuracy and average forgetting.
- Log at least two sample-level signals, preferably loss history and uncertainty.

Strong project success:

- Add predictor precision-recall AUC for forgetting risk.
- Add ablations for individual signals.
- Add MIR, ESMER, or uncertainty replay as stronger baselines.
- Add Split CUB or a small DistilBERT domain adaptation experiment.

## Recommended Repository Shape

```text
configs/
  dataset/
  model/
  replay/
  experiment/
data/
docs/
experiments/
notebooks/
scripts/
src/
  data/
  models/
  training/
  replay/
  signals/
  predictors/
  metrics/
  logging/
  utils/
tests/
```

This structure keeps experiment configuration, reusable code, and research analysis separate.

## Ordered Action Plan

The order below is chosen so each step produces something testable and helps answer the research question.

| Order | Action Item | Deliverable | How It Helps Answer the Research Question |
| --- | --- | --- | --- |
| 0 | Stabilize environment and project layout | Working `.venv`, requirements, `.gitignore`, pytest config, folders | Makes results reproducible and prevents environment issues from being confused with research failures. |
| 1 | Define config system | Hydra or dataclass configs for dataset, model, replay, training, seed, device | Allows fair comparisons where only the replay policy or signal set changes. |
| 2 | Build task stream abstraction | `TaskStream` for Split CIFAR-100 with stable sample IDs | The research question depends on tracking examples across sequential tasks. |
| 3 | Implement device and model factory | ResNet-18 or small CNN, CPU/CUDA detection, optional pretrained support | Keeps the project runnable on Windows CPU while leaving room for CUDA or other backends. |
| 4 | Implement basic continual trainer | Train task 1, then task 2, and evaluate all seen tasks after each task | Produces the accuracy matrix needed to measure forgetting. |
| 5 | Implement metrics | Accuracy matrix, average accuracy, average forgetting, training time | Turns training runs into comparable evidence. |
| 6 | Add fine-tuning baseline | No replay, sequential training only | Establishes the amount of catastrophic forgetting without mitigation. |
| 7 | Add replay buffer and random replay | Bounded buffer with uniform sampling | Establishes the main baseline that spaced replay must beat. |
| 8 | Add sample telemetry | Per-sample loss, confidence, entropy, replay count, last replay step | Creates the raw data needed to ask whether signals predict forgetting. |
| 9 | Define forgetting labels | Sample or class-level label based on future correctness or loss increase | Gives the predictor a target and enables precision-recall evaluation. |
| 10 | Implement signal extractors | Loss trajectory, uncertainty, gradient norm, representation drift | Directly tests which proposed signals contain useful forgetting information. |
| 11 | Build forgetting predictor | Heuristic score first, logistic regression or small MLP second | Measures whether signals can predict forgetting risk better than chance. |
| 12 | Implement spaced replay policy | Scheduler using risk score, due time, and buffer constraints | Tests the core claim that predicted forgetting can guide replay timing. |
| 13 | Add stronger baselines | MIR, ESMER, uncertainty-based replay as time permits | Makes the comparison more credible than only random replay. |
| 14 | Run controlled experiments | Same model, tasks, buffer size, seed set, and compute budget across policies | Ensures improvements are due to replay policy rather than extra resources. |
| 15 | Run ablations | Loss only, uncertainty only, drift only, combined signals | Explains why the method works or fails. |
| 16 | Analyze and report | Tables, plots, error analysis, final write-up | Connects results back to the research question and proposal. |

## Future-Proof API Interfaces

These interfaces are intentionally small. The goal is to let the first implementation be simple while still supporting later baselines, datasets, and model types.

### Data Types

```python
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    class_ids: tuple[int, ...] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SampleBatch:
    x: torch.Tensor
    y: torch.Tensor
    sample_ids: torch.Tensor
    task_id: int
    metadata: dict[str, Any] | None = None


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    features: torch.Tensor | None = None
```

Why this helps: Stable `sample_ids` are the bridge between training, replay, signal logging, and forgetting prediction.

### Task Stream

```python
from typing import Protocol, Iterable


class TaskStream(Protocol):
    def task_specs(self) -> list[TaskSpec]:
        ...

    def train_loader(self, task_id: int) -> Iterable[SampleBatch]:
        ...

    def eval_loader(self, task_id: int) -> Iterable[SampleBatch]:
        ...
```

Why this helps: Split CIFAR-100, Split CUB, and language domain adaptation can share the same trainer.

### Model Factory

```python
class ModelFactory(Protocol):
    def build(self, num_classes: int, device: str) -> torch.nn.Module:
        ...
```

Why this helps: The project can start with a small CNN and later switch to ResNet-18 or DistilBERT without changing replay code.

### Replay Buffer and Policy

```python
@dataclass
class ReplayItem:
    x: torch.Tensor
    y: torch.Tensor
    sample_id: int
    task_id: int
    added_step: int
    last_replayed_step: int | None = None
    replay_count: int = 0
    score: float = 0.0


class ReplayBuffer(Protocol):
    def add(self, batch: SampleBatch, step: int) -> None:
        ...

    def items(self) -> list[ReplayItem]:
        ...

    def __len__(self) -> int:
        ...


class ReplayPolicy(Protocol):
    name: str

    def select(self, buffer: ReplayBuffer, batch_size: int, step: int) -> list[ReplayItem]:
        ...

    def observe(
        self,
        batch: SampleBatch,
        output: ModelOutput,
        step: int,
    ) -> None:
        ...
```

Why this helps: Random replay, MIR, uncertainty replay, and spaced replay can all plug into the same trainer.

### Signal Extraction

```python
@dataclass
class SampleSignal:
    sample_id: int
    task_id: int
    step: int
    loss: float | None = None
    confidence: float | None = None
    entropy: float | None = None
    grad_norm: float | None = None
    representation_drift: float | None = None


class SignalExtractor(Protocol):
    name: str

    def update(
        self,
        batch: SampleBatch,
        output: ModelOutput,
        model: torch.nn.Module,
        step: int,
    ) -> list[SampleSignal]:
        ...
```

Why this helps: The research question is about whether signals predict forgetting. This keeps each signal independently testable.

### Forgetting Predictor

```python
class ForgettingPredictor(Protocol):
    name: str

    def fit(self, signals: list[SampleSignal], labels: dict[int, float]) -> None:
        ...

    def predict_risk(self, signals: list[SampleSignal]) -> dict[int, float]:
        ...
```

Why this helps: The scheduler can start with a heuristic predictor and later use a learned model without changing replay selection.

### Metrics and Logging

```python
@dataclass
class ContinualMetrics:
    accuracy_matrix: list[list[float]]
    average_accuracy: float
    average_forgetting: float
    training_time_seconds: float
    replay_operations: int


class ExperimentLogger(Protocol):
    def log_scalar(self, name: str, value: float, step: int) -> None:
        ...

    def log_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        ...

    def save_artifact(self, path: str) -> None:
        ...
```

Why this helps: Every method must produce the same evidence, making comparisons fair and easier to debug.

### Trainer Entry Point

```python
@dataclass
class TrainerConfig:
    epochs_per_task: int
    batch_size: int
    replay_batch_size: int
    learning_rate: float
    seed: int
    device: str


def train_continual(
    model: torch.nn.Module,
    task_stream: TaskStream,
    replay_buffer: ReplayBuffer,
    replay_policy: ReplayPolicy,
    signal_extractors: list[SignalExtractor],
    logger: ExperimentLogger,
    config: TrainerConfig,
) -> ContinualMetrics:
    ...
```

Why this helps: This is the main integration point for both people. If this signature stays stable, baseline work and scheduler work can proceed in parallel.

## Future Extension Points

The first version should stay small, but the interfaces above leave room for later work:

- New datasets should implement `TaskStream`.
- New models should implement `ModelFactory`.
- New baselines should implement `ReplayPolicy`.
- New forgetting signals should implement `SignalExtractor`.
- New prediction methods should implement `ForgettingPredictor`.
- New loggers should implement `ExperimentLogger`.
- CUDA, CPU, and future device backends should be selected through config, not hard-coded in training logic.
- Language experiments should reuse the same task, signal, predictor, and metric interfaces even if the batch fields include token IDs instead of images.

This keeps future work additive. A new method should mostly add a module and a config file, not require changes across the whole project.

## Experiment Protocol

1. Use the same task order for all methods.
2. Use the same model architecture and optimizer settings for all methods.
3. Use the same replay buffer capacity and replay batch size when comparing replay methods.
4. Run multiple seeds when time permits.
5. Evaluate all seen tasks after each task finishes.
6. Save the accuracy matrix, replay counts, signal logs, predictor scores, and config for every run.

## Division of Work for Two People

### Person 1: Infrastructure, Data, Baselines, Evaluation

Primary goal: Make the experiment pipeline trustworthy.

Owns:

- `src/data/`
- `src/models/`
- `src/training/`
- `src/metrics/`
- `configs/`
- baseline experiment scripts

Main tasks:

1. Implement Split CIFAR-100 task stream with stable sample IDs.
2. Add model factory and Windows CPU/CUDA device selection.
3. Build the basic continual training loop.
4. Implement fine-tuning and random experience replay baselines.
5. Implement accuracy matrix, average forgetting, average accuracy, and training time metrics.
6. Add TensorBoard or CSV logging.
7. Add tests for data splits, metrics, config loading, and replay buffer capacity.

Research contribution:

- Establishes the baseline amount of forgetting.
- Ensures spaced replay is compared against fair and reproducible alternatives.
- Produces the metrics needed to answer whether replay timing improves retention.

### Person 2: Signals, Forgetting Prediction, Spaced Replay

Primary goal: Make the research idea testable.

Owns:

- `src/replay/`
- `src/signals/`
- `src/predictors/`
- `src/analysis/`
- scheduler ablation scripts

Main tasks:

1. Implement replay buffer metadata and sample telemetry.
2. Track loss history, confidence, entropy, replay count, and last replay step.
3. Add gradient norm and representation drift if time permits.
4. Define sample-level or concept-level forgetting labels.
5. Build heuristic forgetting risk predictor.
6. Add learned predictor if enough signal data exists.
7. Implement spaced replay selection using risk score and due time.
8. Run signal ablations and predictor precision-recall AUC analysis.

Research contribution:

- Tests whether the proposed forgetting signals actually predict forgetting.
- Turns prediction into an actionable replay schedule.
- Explains which signals matter and whether the scheduling idea is better than random replay.

### Shared Responsibilities

- Agree on the API contracts before implementing major modules.
- Keep the trainer API stable.
- Review each other's changes at integration points.
- Keep experiment configs reproducible.
- Write down every run's dataset, seed, method, buffer size, and commit hash.

### Integration Checkpoints

| Checkpoint | Person 1 Delivers | Person 2 Delivers | Combined Outcome |
| --- | --- | --- | --- |
| A | Split CIFAR-100 task stream | Replay buffer API draft | Trainer can see task IDs and sample IDs. |
| B | Fine-tuning baseline | Telemetry logger | First forgetting curves can be collected. |
| C | Random replay baseline | Loss and uncertainty signals | Random replay can be compared with signal statistics. |
| D | Metrics and logging | Heuristic spaced scheduler | First core experiment can run. |
| E | Stronger baseline or Split CUB | Predictor and ablations | Final report has credible comparisons and explanation. |

## Suggested Timeline

Week 1:

- Person 1: environment, configs, Split CIFAR-100 stream, simple model.
- Person 2: replay buffer design, sample telemetry schema, signal storage format.

Week 2:

- Person 1: fine-tuning baseline, evaluation after each task, accuracy matrix.
- Person 2: loss and uncertainty tracking, forgetting label definition.

Week 3:

- Person 1: random replay baseline, metrics, logging.
- Person 2: heuristic forgetting predictor, first spaced replay policy.

Week 4:

- Person 1: stronger baseline or reproducibility sweeps.
- Person 2: predictor PR-AUC, signal ablations, scheduler ablations.

Week 5:

- Both: final experiments, plots, written analysis, cleanup, and presentation.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Python 3.14 package compatibility changes | Keep requirements broad but tested; record installed versions from the venv. |
| Windows CPU training is slow | Start with small models, fewer tasks, fewer epochs, and smaller buffers before scaling. |
| Split CUB setup takes too long | Treat Split CIFAR-100 as the required benchmark and Split CUB as stretch work. |
| Signal extraction slows training | Make signals optional by config and start with cheap loss and uncertainty signals. |
| Predictor labels are noisy | Report predictor PR-AUC and scheduler performance separately. |
| Too many baselines for the timeline | Prioritize fine-tuning, random replay, and spaced replay before MIR or ESMER. |

## Definition of Done

The project is done when the team can run a documented experiment that:

1. Trains sequentially on Split CIFAR-100.
2. Compares fine-tuning, random replay, and spaced replay.
3. Uses the same model, task stream, memory budget, and evaluation metrics for all methods.
4. Reports final accuracy, average forgetting, and replay utilization.
5. Shows whether sample-level signals predict forgetting better than chance.
6. Includes plots or tables that directly answer the research question.

## Decision Log Template

Use this format whenever the plan changes:

```text
Date:
Decision:
Reason:
Impact on research question:
Owner:
Follow-up:
```

Keeping decisions explicit helps future readers understand whether a change was made for scientific reasons, engineering constraints, or timeline pressure.
