"""Random replay diagnostic run with final-layer gradient signal logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch

from src.baselines.fine_tuning import (
    BaselineDataUnavailableError,
    build_fixture_streams,
    build_real_split_cifar100_streams,
)
from src.baselines.random_replay import (
    RandomReplayBaselineConfig,
    RandomReplayBaselineRun,
    _train_random_replay,
)
from src.experiment_tracking import ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.replay import ReservoirReplayBuffer
from src.signals import (
    GRADIENT_SIGNAL_FIELDS,
    SIGNAL_FIELDS,
    GradientSignalLogger,
    SampleSignalLogger,
)
from src.training import ContinualTrainerConfig


@dataclass(frozen=True)
class GradientSignalDiagnosticConfig(RandomReplayBaselineConfig):
    """Configuration for the Task 21 gradient signal diagnostic."""

    method_name: str = "gradient_signal_diagnostic"
    run_name: str = "gradient_signal_diagnostic_baseline"
    log_gradient_signals: bool = True


def _experiment_run_config(
    *,
    config: GradientSignalDiagnosticConfig,
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
        task_split={"class_order": list(class_order)},
        method={
            "description": "random replay diagnostic with exact final-layer gradient norm signals",
            "uses_replay": True,
            "diagnostic_only": True,
            "expensive_signal_family": "gradient_norm",
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "uniform_random",
        },
        signals={
            "enabled": config.log_signals,
            "artifact": "sample_signals.json" if config.log_signals else None,
            "fields": list(SIGNAL_FIELDS) if config.log_signals else [],
            "gradient_signals_enabled": config.log_gradient_signals,
            "gradient_artifact": (
                "gradient_signals.json" if config.log_gradient_signals else None
            ),
            "gradient_fields": (
                list(GRADIENT_SIGNAL_FIELDS) if config.log_gradient_signals else []
            ),
            "observation_types": (
                ["current_train", "replay_train", "seen_task_eval"]
                if config.log_signals
                else []
            ),
            "gradient_observation_types": (
                ["seen_task_eval"] if config.log_gradient_signals else []
            ),
        },
    )


def run_gradient_signal_diagnostic(
    config: GradientSignalDiagnosticConfig,
) -> RandomReplayBaselineRun:
    """Run random replay while logging final-layer gradient signals at evaluation."""

    if config.target_key != "original_class_id":
        raise ValueError("gradient diagnostic defaults to original_class_id targets")
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if not config.log_signals:
        raise ValueError("gradient diagnostics require sample signal logging")
    if not config.log_gradient_signals:
        raise ValueError("gradient diagnostics require gradient signal logging")

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
    signal_logger = SampleSignalLogger()
    gradient_signal_logger = GradientSignalLogger()
    result = _train_random_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        signal_logger=signal_logger,
        gradient_signal_logger=gradient_signal_logger,
    )
    result.method_metrics["signals"] = signal_logger.summary()
    result.method_metrics["gradient_signals"] = gradient_signal_logger.summary()
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
            "research_role": "expensive_gradient_signal_diagnostic",
        },
        extra_json_artifacts={
            "sample_signals": signal_logger.to_json_payload(),
            "gradient_signals": gradient_signal_logger.to_json_payload(),
        },
    )
    return RandomReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> GradientSignalDiagnosticConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "gradient_signal_diagnostic" not in raw:
        raise ValueError("config must contain a gradient_signal_diagnostic section")
    section: dict[str, Any] = raw["gradient_signal_diagnostic"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return GradientSignalDiagnosticConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run gradient signal diagnostic")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_gradient_signal_diagnostic(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved gradient signal diagnostic artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
