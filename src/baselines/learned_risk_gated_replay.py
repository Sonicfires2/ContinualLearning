"""Risk-gated replay using a prior-artifact learned forgetting predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch

from src.baselines.fine_tuning import (
    BaselineDataUnavailableError,
    build_fixture_streams,
    build_real_split_cifar100_streams,
)
from src.baselines.risk_gated_replay import (
    RiskGatedReplayBaselineRun,
)
from src.baselines.spaced_replay import (
    SpacedReplayBaselineConfig,
    _scheduler_config,
    _train_spaced_replay,
)
from src.experiment_tracking import ArtifactPaths, ExperimentRunConfig, save_experiment_artifacts
from src.models import build_mlp, count_parameters
from src.predictors import (
    OnlineForgettingRiskScorer,
    train_online_forgetting_risk_scorer,
    train_online_forgetting_risk_scorer_from_paths,
)
from src.replay import ReservoirReplayBuffer, SpacedReplayScheduler, SpacedReplaySchedulerConfig
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult


@dataclass(frozen=True)
class LearnedRiskGatedReplayBaselineConfig(SpacedReplayBaselineConfig):
    """Configuration for replay gated by a learned prior-artifact predictor."""

    method_name: str = "learned_risk_gated_replay"
    run_name: str = "learned_risk_gated_replay_baseline"
    risk_threshold: float = 0.7
    scheduler_budget_mode: str = "risk_only"
    learned_predictor_signal_path: str | None = None
    learned_predictor_label_path: str | None = None
    learned_predictor_feature_group: str = "all_features"
    learned_predictor_train_anchor_task_max: int | None = None


def _fixture_predictor_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    signal_rows = []
    label_rows = []
    for task_id in range(4):
        for offset in range(4):
            sample_id = task_id * 10 + offset
            high_risk = offset in {0, 1}
            if task_id > 0:
                signal_rows.append(
                    _fixture_signal_row(
                        sample_id=sample_id,
                        trained_task_id=task_id - 1,
                        correct=True,
                        loss=0.6 if high_risk else 0.2,
                        target_probability=0.4 if high_risk else 0.8,
                        confidence=0.7 if high_risk else 0.8,
                    )
                )
            signal_rows.append(
                _fixture_signal_row(
                    sample_id=sample_id,
                    trained_task_id=task_id,
                    correct=True,
                    loss=1.4 if high_risk else 0.25,
                    target_probability=0.15 if high_risk else 0.75,
                    confidence=0.55 if high_risk else 0.75,
                )
            )
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "split": "test",
                    "source_task_id": 0,
                    "original_class_id": sample_id,
                    "within_task_label": 0,
                    "original_index": sample_id,
                    "target": sample_id,
                    "anchor_trained_task_id": task_id,
                    "anchor_evaluated_task_id": 0,
                    "anchor_global_step": task_id * 10,
                    "anchor_correct": True,
                    "anchor_loss": 0.5,
                    "anchor_confidence": 0.5,
                    "anchor_target_probability": 0.5,
                    "anchor_uncertainty": 0.5,
                    "eligible_for_binary_forgetting": True,
                    "future_eval_count": 1,
                    "next_trained_task_id": task_id + 1,
                    "final_trained_task_id": task_id + 1,
                    "forgot_next_eval": high_risk,
                    "forgot_final_eval": high_risk,
                    "forgot_any_future": high_risk,
                    "label_uses_future_after_task_id": task_id,
                    "leakage_safe": True,
                }
            )
    return {"rows": signal_rows}, {"rows": label_rows}


def _fixture_signal_row(
    *,
    sample_id: int,
    trained_task_id: int,
    correct: bool,
    loss: float,
    target_probability: float,
    confidence: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_task_id": 0,
        "original_class_id": sample_id,
        "within_task_label": 0,
        "original_index": sample_id,
        "split": "test",
        "target": sample_id,
        "observation_type": "seen_task_eval",
        "trained_task_id": trained_task_id,
        "evaluated_task_id": 0,
        "epoch": None,
        "global_step": trained_task_id * 10,
        "is_replay": False,
        "replay_count": 0,
        "last_replay_step": None,
        "loss": loss,
        "predicted_class": sample_id if correct else 99,
        "correct": correct,
        "confidence": confidence,
        "target_probability": target_probability,
        "uncertainty": 1.0 - confidence,
        "entropy": 0.5,
    }


def _build_risk_scorer(
    config: LearnedRiskGatedReplayBaselineConfig,
) -> OnlineForgettingRiskScorer:
    if config.learned_predictor_signal_path and config.learned_predictor_label_path:
        return train_online_forgetting_risk_scorer_from_paths(
            signal_path=config.learned_predictor_signal_path,
            label_path=config.learned_predictor_label_path,
            feature_group=config.learned_predictor_feature_group,
            train_anchor_task_max=config.learned_predictor_train_anchor_task_max,
        )
    if not config.smoke:
        raise ValueError(
            "learned replay gate requires learned_predictor_signal_path and "
            "learned_predictor_label_path for non-smoke runs"
        )
    signal_payload, label_payload = _fixture_predictor_payloads()
    return train_online_forgetting_risk_scorer(
        signal_payload=signal_payload,
        label_payload=label_payload,
        feature_group=config.learned_predictor_feature_group,
        train_anchor_task_max=config.learned_predictor_train_anchor_task_max,
    )


def _experiment_run_config(
    *,
    config: LearnedRiskGatedReplayBaselineConfig,
    scheduler_config: SpacedReplaySchedulerConfig,
    risk_scorer: OnlineForgettingRiskScorer,
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
            "description": "event-triggered replay gated by a prior-artifact learned forgetting-risk predictor",
            "uses_replay": True,
            "forgetting_aware": True,
            "estimates_t_i": True,
            "ti_estimator": "learned_risk_to_interval_proxy",
            "event_triggered": True,
            "skips_low_risk_replay": True,
            "learned_predictor_gate": True,
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": config.scheduler_budget_mode,
            "schedule": "learned_risk_gated_replay",
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


def run_learned_risk_gated_replay_baseline(
    config: LearnedRiskGatedReplayBaselineConfig,
) -> RiskGatedReplayBaselineRun:
    """Run event-triggered replay using a learned forgetting-risk gate."""

    if config.target_key != "original_class_id":
        raise ValueError(
            "learned risk-gated replay defaults to class-incremental original_class_id targets"
        )
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if config.scheduler_budget_mode not in {"risk_only", "risk_and_due", "risk_or_due", "risk_ranked"}:
        raise ValueError(
            "learned risk-gated replay requires scheduler_budget_mode to be "
            "risk_only, risk_and_due, risk_or_due, or risk_ranked"
        )

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
    result = _train_spaced_replay(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        scheduler=scheduler,
        schedule_name="learned_risk_gated_replay",
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
            "research_role": "learned_predictor_event_triggered_replay",
            "ti_estimation": "learned_risk_to_interval_proxy",
            "risk_threshold": config.risk_threshold,
            "scheduler_budget_mode": config.scheduler_budget_mode,
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
    return RiskGatedReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> LearnedRiskGatedReplayBaselineConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "learned_risk_gated_replay_baseline" not in raw:
        raise ValueError(
            "config must contain a learned_risk_gated_replay_baseline section"
        )
    section = raw["learned_risk_gated_replay_baseline"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return LearnedRiskGatedReplayBaselineConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run learned-predictor risk-gated replay"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_learned_risk_gated_replay_baseline(config)
    except BaselineDataUnavailableError as exc:
        print(f"Baseline data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved learned risk-gated replay artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
