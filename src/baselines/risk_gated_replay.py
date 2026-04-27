"""Event-triggered replay baseline gated by online forgetting risk."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import OmegaConf
import torch

from src.baselines.fine_tuning import (
    BaselineDataUnavailableError,
    build_fixture_streams,
    build_real_split_cifar100_streams,
)
from src.baselines.spaced_replay import (
    SpacedReplayBaselineConfig,
    _scheduler_config,
    _train_spaced_replay,
)
from src.experiment_tracking import ArtifactPaths, ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.replay import ReservoirReplayBuffer, SpacedReplayScheduler, SpacedReplaySchedulerConfig
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult


@dataclass(frozen=True)
class RiskGatedReplayBaselineConfig(SpacedReplayBaselineConfig):
    """Configuration for replay only when examples are predicted near forgetting."""

    method_name: str = "risk_gated_replay"
    run_name: str = "risk_gated_replay_baseline"
    risk_threshold: float = 0.75
    scheduler_budget_mode: str = "risk_and_due"


@dataclass(frozen=True)
class RiskGatedReplayBaselineRun:
    """In-memory result plus saved artifact locations."""

    result: ContinualTrainingResult
    artifacts: ArtifactPaths
    run_config: ExperimentRunConfig


def _experiment_run_config(
    *,
    config: RiskGatedReplayBaselineConfig,
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
            "metrics": [
                "final_accuracy",
                "average_forgetting",
                "average_accuracy",
                "total_replay_samples",
                "effective_replay_ratio",
            ],
        },
        task_split={
            "class_order": list(class_order),
        },
        method={
            "description": "event-triggered replay gated by online forgetting-risk threshold and optional due-time state",
            "uses_replay": True,
            "forgetting_aware": True,
            "estimates_t_i": True,
            "ti_estimator": "online_risk_to_interval_proxy",
            "event_triggered": True,
            "skips_low_risk_replay": True,
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "risk_threshold_and_due"
            if config.scheduler_budget_mode == "risk_and_due"
            else config.scheduler_budget_mode,
            "schedule": "risk_gated_replay",
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


def run_risk_gated_replay_baseline(
    config: RiskGatedReplayBaselineConfig,
) -> RiskGatedReplayBaselineRun:
    """Run event-triggered replay and save experiment artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "risk-gated replay defaults to class-incremental original_class_id targets; "
            "change this only with an explicit research justification"
        )
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if config.scheduler_budget_mode not in {"risk_only", "risk_and_due", "risk_or_due"}:
        raise ValueError(
            "risk-gated replay requires scheduler_budget_mode to be "
            "risk_only, risk_and_due, or risk_or_due"
        )

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
        schedule_name="risk_gated_replay",
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
            "research_role": "event_triggered_risk_gated_replay",
            "ti_estimation": "online_due_time_proxy",
            "risk_threshold": config.risk_threshold,
            "scheduler_budget_mode": config.scheduler_budget_mode,
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
    return RiskGatedReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> RiskGatedReplayBaselineConfig:
    """Load a risk-gated replay config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "risk_gated_replay_baseline" not in raw:
        raise ValueError("config must contain a risk_gated_replay_baseline section")
    section = raw["risk_gated_replay_baseline"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return RiskGatedReplayBaselineConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run event-triggered risk-gated replay")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_risk_gated_replay_baseline(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved risk-gated replay artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
