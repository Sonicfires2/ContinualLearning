"""Spacing-inspired replay scheduler baseline.

This method uses only online signals observed at or before the current training
step. It is the first operational version of the proposal's scheduler, using a
documented due-time proxy rather than future labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
    _batch_to_tensors,
    _make_loader,
    _resolve_device,
)
from src.experiment_tracking import ArtifactPaths, ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.replay import ReservoirReplayBuffer, SpacedReplayScheduler, SpacedReplaySchedulerConfig
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult, evaluate_task_accuracy


@dataclass(frozen=True)
class SpacedReplayBaselineConfig(RandomReplayBaselineConfig):
    """Configuration for spacing-inspired replay."""

    method_name: str = "spaced_replay"
    run_name: str = "spaced_replay_baseline"
    min_replay_interval_steps: int = 1
    max_replay_interval_steps: int = 64
    risk_threshold: float = 0.7
    scheduler_loss_scale: float = 5.0
    scheduler_loss_weight: float = 1.0
    scheduler_uncertainty_weight: float = 1.0
    scheduler_target_probability_weight: float = 1.0
    scheduler_loss_increase_weight: float = 0.5
    scheduler_budget_mode: str = "match_random_replay"


@dataclass(frozen=True)
class SpacedReplayBaselineRun:
    """In-memory result plus saved artifact locations."""

    result: ContinualTrainingResult
    artifacts: ArtifactPaths
    run_config: ExperimentRunConfig


def _scheduler_config(config: SpacedReplayBaselineConfig) -> SpacedReplaySchedulerConfig:
    return SpacedReplaySchedulerConfig(
        min_interval_steps=config.min_replay_interval_steps,
        max_interval_steps=config.max_replay_interval_steps,
        risk_threshold=config.risk_threshold,
        loss_scale=config.scheduler_loss_scale,
        loss_weight=config.scheduler_loss_weight,
        uncertainty_weight=config.scheduler_uncertainty_weight,
        target_probability_weight=config.scheduler_target_probability_weight,
        loss_increase_weight=config.scheduler_loss_increase_weight,
        budget_mode=config.scheduler_budget_mode,
        max_anchor_task_id=max(1, config.task_count - 1),
    )


def _experiment_run_config(
    *,
    config: SpacedReplayBaselineConfig,
    scheduler_config: SpacedReplaySchedulerConfig,
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
            "metrics": ["final_accuracy", "average_forgetting", "average_accuracy"],
        },
        task_split={
            "class_order": list(class_order),
        },
        method={
            "description": "spacing-inspired replay using online risk-to-interval due-time proxy",
            "uses_replay": True,
            "forgetting_aware": True,
            "estimates_t_i": True,
            "ti_estimator": "online_risk_to_interval_proxy",
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "due_time_then_nearest_due",
            "schedule": "spaced_replay",
            "scheduler": asdict(scheduler_config),
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


def _train_spaced_replay(
    *,
    model: nn.Module,
    train_stream,
    eval_stream,
    optimizer: torch.optim.Optimizer,
    trainer_config: ContinualTrainerConfig,
    replay_buffer: ReservoirReplayBuffer,
    replay_batch_size: int,
    scheduler: SpacedReplayScheduler,
    schedule_name: str = "spaced_replay",
    signal_logger: SampleSignalLogger | None = None,
) -> ContinualTrainingResult:
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")

    device = _resolve_device(trainer_config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    task_count = len(train_stream)
    accuracy_matrix = [[None for _ in range(task_count)] for _ in range(task_count)]
    train_losses: list[float] = []
    replay_loss_values: list[float] = []
    current_batch_count = 0
    replay_augmented_batch_count = 0
    replay_event_steps: list[int] = []
    global_step = 0

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

                selections = scheduler.select(
                    items=replay_buffer.items(),
                    global_step=global_step,
                    batch_size=replay_batch_size,
                )
                if selections:
                    replay_batch = replay_buffer.sample_batch_by_sample_ids(
                        sample_ids=[selection.sample_id for selection in selections],
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

    replay_summary = replay_buffer.utilization_summary()
    replay_summary.update(
        {
            "schedule": schedule_name,
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


def run_spaced_replay_baseline(config: SpacedReplayBaselineConfig) -> SpacedReplayBaselineRun:
    """Run spacing-inspired replay and save experiment artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "spaced replay defaults to class-incremental original_class_id targets; "
            "change this only with an explicit research justification"
        )
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")

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
    scheduler = SpacedReplayScheduler(scheduler_config)
    signal_logger = SampleSignalLogger() if config.log_signals else None
    result = _train_spaced_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        scheduler=scheduler,
        signal_logger=signal_logger,
    )
    if signal_logger is not None:
        result.method_metrics["signals"] = signal_logger.summary()
    run_config = _experiment_run_config(
        config=config,
        scheduler_config=scheduler_config,
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
            "research_role": "spacing_inspired_replay_scheduler",
            "ti_estimation": "online_due_time_proxy",
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
    return SpacedReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> SpacedReplayBaselineConfig:
    """Load a spaced replay config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "spaced_replay_baseline" not in raw:
        raise ValueError("config must contain a spaced_replay_baseline section")
    section = raw["spaced_replay_baseline"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return SpacedReplayBaselineConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the spacing-inspired replay baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_spaced_replay_baseline(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved spaced replay artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
