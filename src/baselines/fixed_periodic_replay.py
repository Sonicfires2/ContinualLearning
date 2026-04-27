"""Fixed-periodic replay baseline.

This baseline is the proposal's timing comparator: it replays uniformly from the
same bounded memory as random replay, but only on every k-th optimizer step.
It is deliberately not forgetting-aware and does not estimate T_i.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from src.replay import ReservoirReplayBuffer
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult, evaluate_task_accuracy


@dataclass(frozen=True)
class FixedPeriodicReplayBaselineConfig(RandomReplayBaselineConfig):
    """Configuration for replay every k optimizer steps."""

    method_name: str = "fixed_periodic_replay"
    run_name: str = "fixed_periodic_replay_baseline"
    replay_interval: int = 4
    budget_mode: str = "interval_ablation"


@dataclass(frozen=True)
class FixedPeriodicReplayBaselineRun:
    """In-memory result plus saved artifact locations."""

    result: ContinualTrainingResult
    artifacts: ArtifactPaths
    run_config: ExperimentRunConfig


def _experiment_run_config(
    *,
    config: FixedPeriodicReplayBaselineConfig,
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
            "description": "bounded replay at a fixed optimizer-step interval",
            "uses_replay": True,
            "forgetting_aware": False,
            "estimates_t_i": False,
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "uniform_random",
            "schedule": "fixed_periodic",
            "replay_interval": config.replay_interval,
            "budget_mode": config.budget_mode,
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


def _should_replay_on_step(global_step: int, replay_interval: int) -> bool:
    """Return true for every k-th optimizer step using one-indexed cadence."""

    if replay_interval < 1:
        raise ValueError("replay_interval must be positive")
    return (global_step + 1) % replay_interval == 0


def _train_fixed_periodic_replay(
    *,
    model: nn.Module,
    train_stream,
    eval_stream,
    optimizer: torch.optim.Optimizer,
    trainer_config: ContinualTrainerConfig,
    replay_buffer: ReservoirReplayBuffer,
    replay_batch_size: int,
    replay_interval: int,
    budget_mode: str,
    signal_logger: SampleSignalLogger | None = None,
) -> ContinualTrainingResult:
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    if replay_interval < 1:
        raise ValueError("replay_interval must be positive")
    if budget_mode not in {"interval_ablation", "budget_matched"}:
        raise ValueError("budget_mode must be interval_ablation or budget_matched")

    device = _resolve_device(trainer_config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    task_count = len(train_stream)
    accuracy_matrix = [[None for _ in range(task_count)] for _ in range(task_count)]
    train_losses: list[float] = []
    replay_loss_values: list[float] = []
    replay_event_steps: list[int] = []
    replay_event_optimizer_steps: list[int] = []
    skipped_due_to_interval = 0
    skipped_due_to_empty_buffer = 0
    current_batch_count = 0
    replay_augmented_batch_count = 0
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
                due_for_replay = _should_replay_on_step(global_step, replay_interval)

                if due_for_replay and len(replay_buffer) > 0:
                    replay_batch = replay_buffer.sample_batch(
                        batch_size=replay_batch_size,
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
                    replay_event_optimizer_steps.append(global_step + 1)
                elif due_for_replay:
                    skipped_due_to_empty_buffer += 1
                else:
                    skipped_due_to_interval += 1

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

                if len(train_y) > len(current_y):
                    replay_logits = logits[len(current_y) :]
                    replay_loss = criterion(replay_logits, train_y[len(current_y) :])
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
            "schedule": "fixed_periodic",
            "budget_mode": budget_mode,
            "replay_interval": replay_interval,
            "replay_batch_size": replay_batch_size,
            "current_batch_count": current_batch_count,
            "replay_augmented_batch_count": replay_augmented_batch_count,
            "replay_event_steps": replay_event_steps,
            "replay_event_optimizer_steps": replay_event_optimizer_steps,
            "skipped_replay_steps_due_to_interval": skipped_due_to_interval,
            "skipped_replay_steps_due_to_empty_buffer": skipped_due_to_empty_buffer,
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
        },
    )


def run_fixed_periodic_replay_baseline(
    config: FixedPeriodicReplayBaselineConfig,
) -> FixedPeriodicReplayBaselineRun:
    """Run the fixed-periodic replay baseline and save experiment artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "fixed-periodic replay defaults to class-incremental original_class_id targets; "
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
    signal_logger = SampleSignalLogger() if config.log_signals else None
    result = _train_fixed_periodic_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        replay_interval=config.replay_interval,
        budget_mode=config.budget_mode,
        signal_logger=signal_logger,
    )
    if signal_logger is not None:
        result.method_metrics["signals"] = signal_logger.summary()
    run_config = _experiment_run_config(
        config=config,
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
            "research_role": "fixed_timing_replay_comparator",
            "ti_estimation": "not_used",
        },
        extra_json_artifacts=(
            {"sample_signals": signal_logger.to_json_payload()}
            if signal_logger is not None
            else None
        ),
    )
    return FixedPeriodicReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> FixedPeriodicReplayBaselineConfig:
    """Load a fixed-periodic replay baseline config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "fixed_periodic_replay_baseline" not in raw:
        raise ValueError("config must contain a fixed_periodic_replay_baseline section")
    section = raw["fixed_periodic_replay_baseline"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return FixedPeriodicReplayBaselineConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the fixed-periodic replay baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_fixed_periodic_replay_baseline(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved fixed-periodic replay baseline artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
