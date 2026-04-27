"""Hybrid replay that combines learned risk ranking with diversity sampling."""

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
from src.baselines.learned_fixed_budget_replay import LearnedFixedBudgetReplayConfig
from src.baselines.learned_risk_gated_replay import _build_risk_scorer
from src.baselines.random_replay import (
    RandomReplayBaselineRun,
    _batch_to_tensors,
    _make_loader,
    _resolve_device,
)
from src.baselines.spaced_replay import _scheduler_config
from src.experiment_tracking import ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.replay import ReplayItem, ReservoirReplayBuffer, SpacedReplayScheduler, SpacedReplaySchedulerConfig
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult, evaluate_task_accuracy


@dataclass(frozen=True)
class LearnedHybridReplayConfig(LearnedFixedBudgetReplayConfig):
    """Configuration for learned-risk plus diversity replay."""

    method_name: str = "learned_hybrid_replay"
    run_name: str = "learned_hybrid_replay_baseline"
    scheduler_budget_mode: str = "risk_ranked"
    learned_risk_fraction: float = 0.5
    hybrid_diversity_mode: str = "class_balanced"


def _experiment_run_config(
    *,
    config: LearnedHybridReplayConfig,
    scheduler_config: SpacedReplaySchedulerConfig,
    risk_scorer,
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
                "effective_replay_ratio",
            ],
        },
        task_split={"class_order": list(class_order)},
        method={
            "description": "hybrid replay using learned-risk ranking plus diversity sampling",
            "uses_replay": True,
            "forgetting_aware": True,
            "event_triggered": False,
            "skips_low_risk_replay": False,
            "learned_predictor_ranker": True,
            "protects_replay_diversity": True,
            "matched_replay_budget_to_random_replay": True,
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "learned_risk_plus_diversity_fixed_budget",
            "schedule": "learned_hybrid_replay",
            "learned_risk_fraction": config.learned_risk_fraction,
            "hybrid_diversity_mode": config.hybrid_diversity_mode,
            "scheduler": asdict(scheduler_config),
            "risk_scorer": risk_scorer.to_json_metadata(),
        },
        signals={
            "enabled": config.log_signals,
            "artifact": "sample_signals.json" if config.log_signals else None,
            "fields": list(SIGNAL_FIELDS) if config.log_signals else [],
            "observation_types": (
                ["current_train", "replay_train", "seen_task_eval"]
                if config.log_signals
                else []
            ),
        },
    )


def _select_diversity_items(
    *,
    items: tuple[ReplayItem, ...],
    excluded_sample_ids: set[int],
    quota: int,
    mode: str,
    rng: Random,
) -> list[ReplayItem]:
    if quota <= 0:
        return []
    candidates = [item for item in items if item.sample_id not in excluded_sample_ids]
    if not candidates:
        return []
    if mode == "random":
        return rng.sample(candidates, min(quota, len(candidates)))
    if mode != "class_balanced":
        raise ValueError("hybrid_diversity_mode must be random or class_balanced")

    by_class: dict[int, list[ReplayItem]] = {}
    for item in candidates:
        by_class.setdefault(item.original_class_id, []).append(item)
    for class_items in by_class.values():
        rng.shuffle(class_items)

    class_ids = list(by_class)
    rng.shuffle(class_ids)
    selected: list[ReplayItem] = []
    class_offsets = {class_id: 0 for class_id in class_ids}
    while len(selected) < quota and class_ids:
        progressed = False
        for class_id in list(class_ids):
            offset = class_offsets[class_id]
            class_items = by_class[class_id]
            if offset >= len(class_items):
                class_ids.remove(class_id)
                continue
            selected.append(class_items[offset])
            class_offsets[class_id] = offset + 1
            progressed = True
            if len(selected) >= quota:
                break
        if not progressed:
            break
    return selected


def _train_learned_hybrid_replay(
    *,
    model: nn.Module,
    train_stream,
    eval_stream,
    optimizer: torch.optim.Optimizer,
    trainer_config: ContinualTrainerConfig,
    replay_buffer: ReservoirReplayBuffer,
    replay_batch_size: int,
    scheduler: SpacedReplayScheduler,
    learned_risk_fraction: float,
    hybrid_diversity_mode: str,
    replay_seed: int,
    signal_logger: SampleSignalLogger | None = None,
) -> ContinualTrainingResult:
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    if not 0.0 <= learned_risk_fraction <= 1.0:
        raise ValueError("learned_risk_fraction must be in [0, 1]")
    if hybrid_diversity_mode not in {"random", "class_balanced"}:
        raise ValueError("hybrid_diversity_mode must be random or class_balanced")

    device = _resolve_device(trainer_config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    task_count = len(train_stream)
    accuracy_matrix = [[None for _ in range(task_count)] for _ in range(task_count)]
    train_losses: list[float] = []
    replay_loss_values: list[float] = []
    replay_event_steps: list[int] = []
    current_batch_count = 0
    replay_augmented_batch_count = 0
    learned_risk_selected_count = 0
    diversity_selected_count = 0
    fallback_selected_count = 0
    global_step = 0
    rng = Random(replay_seed + 20_000)

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
                    learned_quota = int(round(replay_batch_size * learned_risk_fraction))
                    learned_quota = max(0, min(replay_batch_size, learned_quota))
                    diversity_quota = replay_batch_size - learned_quota
                    risk_selections = scheduler.select(
                        items=buffer_items,
                        global_step=global_step,
                        batch_size=learned_quota,
                    ) if learned_quota > 0 else []
                    selected_ids = [selection.sample_id for selection in risk_selections]
                    selected_id_set = set(selected_ids)
                    diversity_items = _select_diversity_items(
                        items=buffer_items,
                        excluded_sample_ids=selected_id_set,
                        quota=diversity_quota,
                        mode=hybrid_diversity_mode,
                        rng=rng,
                    )
                    selected_ids.extend(item.sample_id for item in diversity_items)
                    selected_id_set.update(item.sample_id for item in diversity_items)

                    if len(selected_ids) < min(replay_batch_size, len(buffer_items)):
                        fallback_items = _select_diversity_items(
                            items=buffer_items,
                            excluded_sample_ids=selected_id_set,
                            quota=min(replay_batch_size, len(buffer_items)) - len(selected_ids),
                            mode="random",
                            rng=rng,
                        )
                        selected_ids.extend(item.sample_id for item in fallback_items)
                        fallback_selected_count += len(fallback_items)

                    if selected_ids:
                        replay_batch = replay_buffer.sample_batch_by_sample_ids(
                            sample_ids=selected_ids,
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
                        replay_event_steps.append(global_step)
                        learned_risk_selected_count += len(risk_selections)
                        diversity_selected_count += len(diversity_items)

                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x)
                current_logits = logits[: len(current_y)]
                scheduler.observe_batch(
                    logits=current_logits,
                    targets=current_y,
                    batch=batch,
                    global_step=global_step,
                    is_replay=False,
                    trained_task_id=task_id,
                )
                if replay_batch is not None and replay_y is not None:
                    replay_logits = logits[len(current_y) :]
                    scheduler.observe_batch(
                        logits=replay_logits,
                        targets=replay_y,
                        batch=replay_batch,
                        global_step=global_step,
                        is_replay=True,
                        trained_task_id=task_id,
                    )

                loss = criterion(logits, train_y)
                if signal_logger is not None:
                    signal_logger.log_batch(
                        logits=current_logits,
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

        for eval_task_id in range(task_id + 1):
            accuracy_matrix[task_id][eval_task_id] = evaluate_task_accuracy(
                model=model,
                dataset=eval_stream.task_dataset(eval_task_id),
                config=trainer_config,
                device=device,
                signal_logger=signal_logger,
                trained_task_id=task_id,
                evaluated_task_id=eval_task_id,
                global_step=global_step,
            )

    total_hybrid_selections = (
        learned_risk_selected_count + diversity_selected_count + fallback_selected_count
    )
    replay_summary = replay_buffer.utilization_summary()
    replay_summary.update(
        {
            "schedule": "learned_hybrid_replay",
            "replay_batch_size": replay_batch_size,
            "current_batch_count": current_batch_count,
            "replay_augmented_batch_count": replay_augmented_batch_count,
            "replay_event_steps": replay_event_steps,
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
            "hybrid": {
                "learned_risk_fraction": learned_risk_fraction,
                "diversity_mode": hybrid_diversity_mode,
                "learned_risk_selected_count": learned_risk_selected_count,
                "diversity_selected_count": diversity_selected_count,
                "fallback_selected_count": fallback_selected_count,
                "actual_learned_risk_fraction": (
                    learned_risk_selected_count / total_hybrid_selections
                    if total_hybrid_selections
                    else 0.0
                ),
            },
        }
    )
    return ContinualTrainingResult(
        accuracy_matrix=accuracy_matrix,
        train_losses=train_losses,
        training_time_seconds=perf_counter() - start,
        task_count=task_count,
        method_metrics={
            "replay": replay_summary,
            "scheduler": scheduler.summary(),
        },
    )


def run_learned_hybrid_replay(config: LearnedHybridReplayConfig) -> RandomReplayBaselineRun:
    """Run learned-risk plus diversity replay under a fixed replay budget."""

    if config.target_key != "original_class_id":
        raise ValueError("learned hybrid replay defaults to class-incremental targets")
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if config.scheduler_budget_mode != "risk_ranked":
        raise ValueError("learned hybrid replay requires risk_ranked budget mode")

    risk_scorer = _build_risk_scorer(config)
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
    scheduler_config = _scheduler_config(config)
    scheduler_config = SpacedReplaySchedulerConfig(
        **{**asdict(scheduler_config), "risk_score_source": "learned"}
    )
    scheduler = SpacedReplayScheduler(scheduler_config, risk_scorer=risk_scorer)
    signal_logger = SampleSignalLogger() if config.log_signals else None
    result = _train_learned_hybrid_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        scheduler=scheduler,
        learned_risk_fraction=config.learned_risk_fraction,
        hybrid_diversity_mode=config.hybrid_diversity_mode,
        replay_seed=config.replay_seed,
        signal_logger=signal_logger,
    )
    if signal_logger is not None:
        result.method_metrics["signals"] = signal_logger.summary()
    run_config = _experiment_run_config(
        config=config,
        scheduler_config=scheduler_config,
        risk_scorer=risk_scorer,
        model_parameter_count=count_parameters(model),
        class_order=class_order,
    )
    artifacts = save_experiment_artifacts(
        output_root=config.output_root,
        run_config=run_config,
        result=result,
        overwrite=config.overwrite,
        extra_metadata={
            "baseline_status": "smoke" if config.smoke else "real_split_cifar100",
            "research_role": "learned_risk_diversity_hybrid_replay",
            "risk_scorer": risk_scorer.to_json_metadata(),
        },
        extra_json_artifacts={
            **(
                {"sample_signals": signal_logger.to_json_payload()}
                if signal_logger is not None
                else {}
            ),
            "scheduler_trace": scheduler.to_json_payload(),
        },
    )
    return RandomReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> LearnedHybridReplayConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "learned_hybrid_replay" not in raw:
        raise ValueError("config must contain a learned_hybrid_replay section")
    section = raw["learned_hybrid_replay"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return LearnedHybridReplayConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run learned-risk hybrid replay")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_learned_hybrid_replay(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved learned hybrid replay artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
