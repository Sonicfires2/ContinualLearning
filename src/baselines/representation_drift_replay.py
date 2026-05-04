"""Replay schedulers that use representation drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

from omegaconf import OmegaConf
import torch
from torch import nn

from src.baselines.fine_tuning import (
    BaselineDataUnavailableError,
    build_fixture_streams,
    build_real_split_cifar100_streams,
)
from src.baselines.random_replay import (
    RandomReplayBaselineConfig,
    RandomReplayBaselineRun,
    _batch_to_tensors,
    _make_loader,
    _resolve_device,
)
from src.experiment_tracking import ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.replay import ReplayItem, ReservoirReplayBuffer
from src.signals import (
    REPRESENTATION_SIGNAL_FIELDS,
    SIGNAL_FIELDS,
    RepresentationDriftLogger,
    SampleSignalLogger,
    representation_signal_tensors,
)
from src.training import ContinualTrainerConfig, ContinualTrainingResult, evaluate_task_accuracy


@dataclass(frozen=True)
class DriftReference:
    """Reference representation for one replay-buffer item."""

    vector: torch.Tensor
    l2: float
    added_at_step: int
    added_at_task: int


@dataclass
class DriftReplayState:
    """Online replay state derived from representation drift."""

    sample_id: int
    drift_score: float
    risk_score: float
    next_due_step: int
    last_scored_step: int
    due_step_before_update: int
    score_count: int = 0


@dataclass(frozen=True)
class DriftSelection:
    """One selected replay item."""

    global_step: int
    sample_id: int
    source_task_id: int
    original_class_id: int
    drift_score: float
    risk_score: float
    next_due_step: int
    overdue_steps: int
    selection_reason: str
    variant: str


@dataclass(frozen=True)
class RepresentationDriftReplayConfig(RandomReplayBaselineConfig):
    """Configuration for representation-drift replay variants."""

    method_name: str = "representation_drift_replay"
    run_name: str = "representation_drift_replay_baseline"
    drift_variant: str = "drift_ranked"
    drift_candidate_count: int = 128
    drift_scoring_batch_size: int = 128
    drift_scale: float = 1.0
    min_replay_interval_steps: int = 1
    max_replay_interval_steps: int = 64
    drift_hybrid_fraction: float = 0.5
    drift_hybrid_diversity_mode: str = "class_balanced"
    log_representation_signals: bool = True


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first_norm = torch.linalg.vector_norm(first, ord=2)
    second_norm = torch.linalg.vector_norm(second, ord=2)
    first_norm_value = float(first_norm.item())
    second_norm_value = float(second_norm.item())
    if first_norm_value == 0.0 and second_norm_value == 0.0:
        return 1.0
    if first_norm_value == 0.0 or second_norm_value == 0.0:
        return 0.0
    similarity = torch.dot(first, second) / (first_norm * second_norm)
    return max(-1.0, min(1.0, float(similarity.item())))


def _estimated_interval_from_risk(
    *,
    risk_score: float,
    min_interval_steps: int,
    max_interval_steps: int,
) -> int:
    risk = _clamp(risk_score)
    span = max_interval_steps - min_interval_steps
    return int(round(max_interval_steps - risk * span))


def _candidate_items(
    *,
    items: tuple[ReplayItem, ...],
    candidate_count: int,
    rng: Random,
) -> list[ReplayItem]:
    if candidate_count < 1:
        raise ValueError("drift_candidate_count must be positive")
    if candidate_count >= len(items):
        return list(items)
    return rng.sample(list(items), candidate_count)


def _representations_for_items(
    *,
    model: nn.Module,
    items: list[ReplayItem],
    device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    if not items:
        return {}
    was_training = model.training
    model.eval()
    representations: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            inputs = torch.stack([item.x for item in batch_items]).to(device)
            tensors = representation_signal_tensors(model=model, inputs=inputs)
            reps = tensors["representations"].detach().float().cpu()
            for item, representation in zip(batch_items, reps):
                representations[item.sample_id] = representation.clone()
    model.train(was_training)
    return representations


def _ensure_references(
    *,
    model: nn.Module,
    items: tuple[ReplayItem, ...],
    references: dict[int, DriftReference],
    device: torch.device,
    batch_size: int,
) -> None:
    missing = [item for item in items if item.sample_id not in references]
    if not missing:
        return
    reps = _representations_for_items(
        model=model,
        items=missing,
        device=device,
        batch_size=batch_size,
    )
    for item in missing:
        vector = reps[item.sample_id]
        references[item.sample_id] = DriftReference(
            vector=vector,
            l2=float(torch.linalg.vector_norm(vector, ord=2).item()),
            added_at_step=item.added_at_step,
            added_at_task=item.added_at_task,
        )


def _score_candidates(
    *,
    model: nn.Module,
    candidates: list[ReplayItem],
    references: dict[int, DriftReference],
    states: dict[int, DriftReplayState],
    global_step: int,
    device: torch.device,
    batch_size: int,
    drift_scale: float,
    min_interval_steps: int,
    max_interval_steps: int,
) -> list[tuple[ReplayItem, DriftReplayState]]:
    if drift_scale <= 0:
        raise ValueError("drift_scale must be positive")
    current_reps = _representations_for_items(
        model=model,
        items=candidates,
        device=device,
        batch_size=batch_size,
    )
    scored: list[tuple[ReplayItem, DriftReplayState]] = []
    for item in candidates:
        reference = references[item.sample_id]
        current = current_reps[item.sample_id]
        drift = 1.0 - _cosine_similarity(current, reference.vector)
        risk = _clamp(drift / drift_scale)
        interval = _estimated_interval_from_risk(
            risk_score=risk,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
        )
        previous = states.get(item.sample_id)
        due_step_before_update = (
            previous.next_due_step
            if previous is not None
            else reference.added_at_step
        )
        states[item.sample_id] = DriftReplayState(
            sample_id=item.sample_id,
            drift_score=drift,
            risk_score=risk,
            next_due_step=global_step + interval,
            last_scored_step=global_step,
            due_step_before_update=due_step_before_update,
            score_count=(previous.score_count + 1 if previous is not None else 1),
        )
        scored.append((item, states[item.sample_id]))
    return scored


def _select_class_balanced(
    *,
    items: tuple[ReplayItem, ...],
    excluded_sample_ids: set[int],
    quota: int,
    rng: Random,
) -> list[ReplayItem]:
    candidates = [item for item in items if item.sample_id not in excluded_sample_ids]
    if quota <= 0 or not candidates:
        return []
    by_class: dict[int, list[ReplayItem]] = {}
    for item in candidates:
        by_class.setdefault(item.original_class_id, []).append(item)
    for class_items in by_class.values():
        rng.shuffle(class_items)
    class_ids = list(by_class)
    rng.shuffle(class_ids)
    selected: list[ReplayItem] = []
    offsets = {class_id: 0 for class_id in class_ids}
    while len(selected) < quota and class_ids:
        for class_id in list(class_ids):
            offset = offsets[class_id]
            class_items = by_class[class_id]
            if offset >= len(class_items):
                class_ids.remove(class_id)
                continue
            selected.append(class_items[offset])
            offsets[class_id] = offset + 1
            if len(selected) >= quota:
                break
    return selected


def _select_random(
    *,
    items: tuple[ReplayItem, ...],
    excluded_sample_ids: set[int],
    quota: int,
    rng: Random,
) -> list[ReplayItem]:
    candidates = [item for item in items if item.sample_id not in excluded_sample_ids]
    if quota <= 0 or not candidates:
        return []
    return rng.sample(candidates, min(quota, len(candidates)))


def _select_drift_replay_items(
    *,
    variant: str,
    model: nn.Module,
    items: tuple[ReplayItem, ...],
    references: dict[int, DriftReference],
    states: dict[int, DriftReplayState],
    global_step: int,
    replay_batch_size: int,
    candidate_count: int,
    scoring_batch_size: int,
    drift_scale: float,
    min_interval_steps: int,
    max_interval_steps: int,
    hybrid_fraction: float,
    hybrid_diversity_mode: str,
    device: torch.device,
    rng: Random,
) -> tuple[list[ReplayItem], list[DriftSelection]]:
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    if not items:
        return [], []
    candidates = _candidate_items(
        items=items,
        candidate_count=candidate_count,
        rng=rng,
    )
    scored = _score_candidates(
        model=model,
        candidates=candidates,
        references=references,
        states=states,
        global_step=global_step,
        device=device,
        batch_size=scoring_batch_size,
        drift_scale=drift_scale,
        min_interval_steps=min_interval_steps,
        max_interval_steps=max_interval_steps,
    )
    scored_sample_ids = {item.sample_id for item, _state in scored}
    target_count = min(replay_batch_size, len(items))

    if variant == "drift_ranked":
        selected_records = sorted(
            scored,
            key=lambda record: (
                -record[1].risk_score,
                -record[1].drift_score,
                record[0].replay_count,
                record[0].sample_id,
            ),
        )[:target_count]
        selected = [item for item, _state in selected_records]
        selected_reasons = {item.sample_id: "drift_ranked" for item in selected}
    elif variant == "drift_due_time":
        selected_records = sorted(
            scored,
            key=lambda record: (
                int(global_step < record[1].due_step_before_update),
                -(global_step - record[1].due_step_before_update),
                -record[1].risk_score,
                record[1].due_step_before_update,
                record[0].sample_id,
            ),
        )[:target_count]
        selected = [item for item, _state in selected_records]
        selected_reasons = {
            item.sample_id: (
                "due"
                if global_step >= state.due_step_before_update
                else "budget_fill_near_due"
            )
            for item, state in selected_records
        }
    elif variant == "drift_hybrid":
        if not 0.0 <= hybrid_fraction <= 1.0:
            raise ValueError("drift_hybrid_fraction must be in [0, 1]")
        drift_quota = min(target_count, int(round(target_count * hybrid_fraction)))
        diversity_quota = target_count - drift_quota
        selected_records = sorted(
            scored,
            key=lambda record: (
                -record[1].risk_score,
                -record[1].drift_score,
                record[0].replay_count,
                record[0].sample_id,
            ),
        )[:drift_quota]
        selected = [item for item, _state in selected_records]
        selected_ids = {item.sample_id for item in selected}
        if hybrid_diversity_mode == "class_balanced":
            diversity_items = _select_class_balanced(
                items=items,
                excluded_sample_ids=selected_ids,
                quota=diversity_quota,
                rng=rng,
            )
        elif hybrid_diversity_mode == "random":
            diversity_items = _select_random(
                items=items,
                excluded_sample_ids=selected_ids,
                quota=diversity_quota,
                rng=rng,
            )
        else:
            raise ValueError("drift_hybrid_diversity_mode must be class_balanced or random")
        selected.extend(diversity_items)
        selected_ids.update(item.sample_id for item in diversity_items)
        if len(selected) < target_count:
            selected.extend(
                _select_random(
                    items=items,
                    excluded_sample_ids=selected_ids,
                    quota=target_count - len(selected),
                    rng=rng,
                )
            )
        selected_reasons = {
            item.sample_id: "drift_ranked"
            for item, _state in selected_records
        }
        selected_reasons.update(
            {item.sample_id: f"diversity_{hybrid_diversity_mode}" for item in diversity_items}
        )
    else:
        raise ValueError("drift_variant must be drift_ranked, drift_due_time, or drift_hybrid")

    unscored_selected = [
        item for item in selected if item.sample_id not in scored_sample_ids
    ]
    if unscored_selected:
        _score_candidates(
            model=model,
            candidates=unscored_selected,
            references=references,
            states=states,
            global_step=global_step,
            device=device,
            batch_size=scoring_batch_size,
            drift_scale=drift_scale,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
        )

    selections = []
    for item in selected:
        state = states.get(item.sample_id)
        if state is None:
            state = DriftReplayState(
                sample_id=item.sample_id,
                drift_score=0.0,
                risk_score=0.0,
                next_due_step=global_step,
                last_scored_step=global_step,
                due_step_before_update=global_step,
            )
        selections.append(
            DriftSelection(
                global_step=int(global_step),
                sample_id=item.sample_id,
                source_task_id=item.task_id,
                original_class_id=item.original_class_id,
                drift_score=state.drift_score,
                risk_score=state.risk_score,
                next_due_step=state.next_due_step,
                overdue_steps=max(0, global_step - state.due_step_before_update),
                selection_reason=selected_reasons.get(item.sample_id, "fallback_random"),
                variant=variant,
            )
        )
    return selected, selections


def _experiment_run_config(
    *,
    config: RepresentationDriftReplayConfig,
    model_parameter_count: int,
    class_order: tuple[int, ...],
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        protocol_id=config.protocol_id,
        method_name=config.method_name,
        seed=config.seed,
        run_name=config.run_name,
        dataset={
            "name": "fixture_split_cifar100" if config.smoke else "split_cifar100",
            "data_root": config.data_root,
            "download": config.download,
            "task_count": config.task_count,
            "classes_per_task": config.classes_per_task,
            "split_seed": config.split_seed,
            "target_key": config.target_key,
            "smoke": config.smoke,
        },
        model={
            "name": "flatten_mlp",
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "output_dim": config.task_count * config.classes_per_task,
            "trainable_parameters": model_parameter_count,
        },
        trainer={
            "epochs_per_task": config.epochs_per_task,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "device": config.device,
            "target_key": config.target_key,
        },
        evaluation={
            "schedule": "evaluate_all_seen_tasks_after_each_task",
            "metrics": [
                "final_accuracy",
                "average_forgetting",
                "average_accuracy",
                "total_replay_samples",
            ],
        },
        task_split={"class_order": list(class_order)},
        method={
            "description": "representation-drift replay scheduler",
            "uses_replay": True,
            "forgetting_aware": True,
            "drift_variant": config.drift_variant,
            "matched_replay_budget_to_random_replay": True,
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": config.drift_variant,
            "candidate_count": config.drift_candidate_count,
            "drift_scale": config.drift_scale,
            "min_replay_interval_steps": config.min_replay_interval_steps,
            "max_replay_interval_steps": config.max_replay_interval_steps,
            "drift_hybrid_fraction": config.drift_hybrid_fraction,
            "drift_hybrid_diversity_mode": config.drift_hybrid_diversity_mode,
        },
        signals={
            "enabled": config.log_signals,
            "artifact": "sample_signals.json" if config.log_signals else None,
            "fields": list(SIGNAL_FIELDS) if config.log_signals else [],
            "representation_signals_enabled": config.log_representation_signals,
            "representation_fields": (
                list(REPRESENTATION_SIGNAL_FIELDS)
                if config.log_representation_signals
                else []
            ),
            "observation_types": (
                ["current_train", "replay_train", "seen_task_eval"]
                if config.log_signals
                else []
            ),
        },
    )


def _selection_summary(selections: list[DriftSelection]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    drift_values = []
    risk_values = []
    for selection in selections:
        reason_counts[selection.selection_reason] = (
            reason_counts.get(selection.selection_reason, 0) + 1
        )
        drift_values.append(selection.drift_score)
        risk_values.append(selection.risk_score)
    return {
        "trace_row_count": len(selections),
        "selection_reason_counts": reason_counts,
        "mean_selected_drift_score": (
            sum(drift_values) / len(drift_values) if drift_values else None
        ),
        "mean_selected_risk_score": (
            sum(risk_values) / len(risk_values) if risk_values else None
        ),
        "max_selected_drift_score": max(drift_values) if drift_values else None,
    }


def _train_representation_drift_replay(
    *,
    model: nn.Module,
    train_stream,
    eval_stream,
    optimizer: torch.optim.Optimizer,
    trainer_config: ContinualTrainerConfig,
    replay_buffer: ReservoirReplayBuffer,
    config: RepresentationDriftReplayConfig,
    signal_logger: SampleSignalLogger | None = None,
    representation_signal_logger: RepresentationDriftLogger | None = None,
) -> tuple[ContinualTrainingResult, list[DriftSelection]]:
    device = _resolve_device(trainer_config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    task_count = len(train_stream)
    accuracy_matrix = [[None for _ in range(task_count)] for _ in range(task_count)]
    train_losses: list[float] = []
    replay_loss_values: list[float] = []
    current_batch_count = 0
    replay_augmented_batch_count = 0
    global_step = 0
    rng = Random(config.replay_seed + 31_000)
    references: dict[int, DriftReference] = {}
    states: dict[int, DriftReplayState] = {}
    selections: list[DriftSelection] = []

    start = perf_counter()
    for task_id in range(task_count):
        train_loader = _make_loader(
            train_stream.task_dataset(task_id),
            batch_size=trainer_config.batch_size,
            shuffle=trainer_config.shuffle_train,
            seed=trainer_config.seed + task_id,
        )

        model.train()
        for _epoch in range(trainer_config.epochs_per_task):
            for batch in train_loader:
                current_x, current_y = _batch_to_tensors(
                    batch,
                    target_key=trainer_config.target_key,
                    device=device,
                )
                train_x = current_x
                train_y = current_y
                replay_batch = None
                replay_y = None
                buffer_items = replay_buffer.items()
                if buffer_items:
                    _ensure_references(
                        model=model,
                        items=buffer_items,
                        references=references,
                        device=device,
                        batch_size=config.drift_scoring_batch_size,
                    )
                    selected_items, batch_selections = _select_drift_replay_items(
                        variant=config.drift_variant,
                        model=model,
                        items=buffer_items,
                        references=references,
                        states=states,
                        global_step=global_step,
                        replay_batch_size=config.replay_batch_size,
                        candidate_count=config.drift_candidate_count,
                        scoring_batch_size=config.drift_scoring_batch_size,
                        drift_scale=config.drift_scale,
                        min_interval_steps=config.min_replay_interval_steps,
                        max_interval_steps=config.max_replay_interval_steps,
                        hybrid_fraction=config.drift_hybrid_fraction,
                        hybrid_diversity_mode=config.drift_hybrid_diversity_mode,
                        device=device,
                        rng=rng,
                    )
                    selections.extend(batch_selections)
                    if selected_items:
                        replay_batch = replay_buffer.sample_batch_by_sample_ids(
                            sample_ids=[item.sample_id for item in selected_items],
                            target_key=trainer_config.target_key,
                            replay_step=global_step,
                        )
                        replay_x, replay_y = _batch_to_tensors(
                            replay_batch,
                            target_key=trainer_config.target_key,
                            device=device,
                        )
                        train_x = torch.cat([current_x, replay_x], dim=0)
                        train_y = torch.cat([current_y, replay_y], dim=0)
                        replay_augmented_batch_count += 1

                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x)
                loss = criterion(logits, train_y)
                if signal_logger is not None:
                    signal_logger.log_batch(
                        logits=logits[: len(current_y)],
                        targets=current_y,
                        batch=batch,
                        observation_type="current_train",
                        trained_task_id=task_id,
                        epoch=_epoch,
                        global_step=global_step,
                        is_replay=False,
                    )
                    if replay_batch is not None and replay_y is not None:
                        signal_logger.log_batch(
                            logits=logits[len(current_y) :],
                            targets=replay_y,
                            batch=replay_batch,
                            observation_type="replay_train",
                            trained_task_id=task_id,
                            epoch=_epoch,
                            global_step=global_step,
                            is_replay=True,
                        )
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                if replay_batch is not None and replay_y is not None:
                    replay_loss = criterion(logits[len(current_y) :], replay_y)
                    replay_loss_values.append(float(replay_loss.detach().cpu().item()))
                current_batch_count += 1
                global_step += 1

        add_loader = _make_loader(
            train_stream.task_dataset(task_id),
            batch_size=trainer_config.batch_size,
            shuffle=False,
            seed=trainer_config.seed + 10_000 + task_id,
        )
        for add_batch in add_loader:
            replay_buffer.add_batch(
                add_batch,
                target_key=trainer_config.target_key,
                added_at_task=task_id,
                added_at_step=global_step,
            )
        _ensure_references(
            model=model,
            items=replay_buffer.items(),
            references=references,
            device=device,
            batch_size=config.drift_scoring_batch_size,
        )

        for eval_task_id in range(task_id + 1):
            accuracy_matrix[task_id][eval_task_id] = evaluate_task_accuracy(
                model=model,
                dataset=eval_stream.task_dataset(eval_task_id),
                config=trainer_config,
                device=device,
                signal_logger=signal_logger,
                representation_signal_logger=representation_signal_logger,
                trained_task_id=task_id,
                evaluated_task_id=eval_task_id,
                global_step=global_step,
            )

    replay_summary = replay_buffer.utilization_summary()
    replay_summary.update(
        {
            "schedule": config.drift_variant,
            "replay_batch_size": config.replay_batch_size,
            "current_batch_count": current_batch_count,
            "replay_augmented_batch_count": replay_augmented_batch_count,
            "effective_replay_ratio": (
                replay_augmented_batch_count / current_batch_count
                if current_batch_count
                else 0.0
            ),
            "mean_replay_loss": (
                sum(replay_loss_values) / len(replay_loss_values)
                if replay_loss_values
                else None
            ),
        }
    )
    result = ContinualTrainingResult(
        accuracy_matrix=accuracy_matrix,
        train_losses=train_losses,
        training_time_seconds=perf_counter() - start,
        task_count=task_count,
        method_metrics={
            "replay": replay_summary,
            "drift_scheduler": {
                "variant": config.drift_variant,
                "candidate_count": config.drift_candidate_count,
                "reference_count": len(references),
                "state_count": len(states),
                **_selection_summary(selections),
            },
        },
    )
    return result, selections


def run_representation_drift_replay(
    config: RepresentationDriftReplayConfig,
) -> RandomReplayBaselineRun:
    """Run a representation-drift replay scheduler and save artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "representation-drift replay defaults to original_class_id targets"
        )
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if config.drift_variant not in {"drift_ranked", "drift_due_time", "drift_hybrid"}:
        raise ValueError("drift_variant must be drift_ranked, drift_due_time, or drift_hybrid")

    if config.smoke:
        train_stream, eval_stream, input_shape, class_order = build_fixture_streams(config)
    else:
        train_stream, eval_stream, input_shape, class_order = build_real_split_cifar100_streams(config)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = build_mlp(
        input_shape=input_shape,
        output_dim=config.task_count * config.classes_per_task,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    trainer_config = ContinualTrainerConfig(
        epochs_per_task=config.epochs_per_task,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        device=config.device,
        seed=config.seed,
        target_key=config.target_key,
    )
    replay_buffer = ReservoirReplayBuffer(
        capacity=config.replay_capacity,
        seed=config.replay_seed,
    )
    signal_logger = SampleSignalLogger() if config.log_signals else None
    representation_signal_logger = (
        RepresentationDriftLogger() if config.log_representation_signals else None
    )
    result, selections = _train_representation_drift_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        config=config,
        signal_logger=signal_logger,
        representation_signal_logger=representation_signal_logger,
    )
    if signal_logger is not None:
        result.method_metrics["signals"] = signal_logger.summary()
    if representation_signal_logger is not None:
        result.method_metrics["representation_signals"] = (
            representation_signal_logger.summary()
        )
    run_config = _experiment_run_config(
        config=config,
        model_parameter_count=count_parameters(model),
        class_order=class_order,
    )
    extra_json_artifacts = {}
    if signal_logger is not None:
        extra_json_artifacts["sample_signals"] = signal_logger.to_json_payload()
    if representation_signal_logger is not None:
        extra_json_artifacts["representation_signals"] = (
            representation_signal_logger.to_json_payload()
        )
    extra_json_artifacts["scheduler_trace"] = {
        "schema_version": 1,
        "variant": config.drift_variant,
        "records": [asdict(selection) for selection in selections],
    }
    artifacts = save_experiment_artifacts(
        output_root=config.output_root,
        run_config=run_config,
        result=result,
        overwrite=config.overwrite,
        extra_metadata={
            "baseline_status": "smoke" if config.smoke else "real_split_cifar100",
            "research_role": "representation_drift_replay_scheduler",
            "drift_variant": config.drift_variant,
        },
        extra_json_artifacts=extra_json_artifacts,
    )
    return RandomReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> RepresentationDriftReplayConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "representation_drift_replay" not in raw:
        raise ValueError("config must contain a representation_drift_replay section")
    section: dict[str, Any] = raw["representation_drift_replay"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return RepresentationDriftReplayConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run representation-drift replay")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--variant", type=str, default=None, help="Variant override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = dict(config.__dict__)
    if args.variant is not None:
        overrides["drift_variant"] = args.variant
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
        overrides["replay_seed"] = int(args.seed)
    if args.variant is not None or args.seed is not None:
        variant = overrides["drift_variant"]
        seed = overrides["seed"]
        if not config.smoke:
            overrides["run_name"] = (
                f"representation_drift_replay_{variant}_split_cifar100_seed{seed}"
            )
        else:
            overrides["run_name"] = (
                f"representation_drift_replay_{variant}_smoke_seed{seed}"
            )
        config = RepresentationDriftReplayConfig(**overrides)
    try:
        run = run_representation_drift_replay(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved representation-drift replay artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
