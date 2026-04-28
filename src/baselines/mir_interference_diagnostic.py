"""Diagnostic run comparing learned risk with MIR current interference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from time import perf_counter

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
from src.predictors import MIRInterferenceDiagnosticLogger
from src.replay import ReservoirReplayBuffer, SpacedReplayScheduler, SpacedReplaySchedulerConfig, score_mir_replay_candidates
from src.signals import SIGNAL_FIELDS, SampleSignalLogger
from src.training import ContinualTrainerConfig, ContinualTrainingResult, evaluate_task_accuracy


@dataclass(frozen=True)
class MIRInterferenceDiagnosticConfig(LearnedFixedBudgetReplayConfig):
    """Configuration for the MIR-vs-learned-risk diagnostic."""

    method_name: str = "mir_interference_diagnostic"
    run_name: str = "mir_interference_diagnostic"
    scheduler_budget_mode: str = "risk_ranked"
    mir_candidate_size: int = 128
    mir_virtual_lr: float | None = None


def _experiment_run_config(
    *,
    config: MIRInterferenceDiagnosticConfig,
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
                "learned_risk_vs_mir_topk_average_precision",
                "learned_risk_vs_mir_topk_overlap",
            ],
        },
        task_split={"class_order": list(class_order)},
        method={
            "description": "random replay diagnostic that logs MIR candidate interference and learned forgetting risk",
            "uses_replay": True,
            "actual_replay_policy": "uniform_from_mir_candidate_pool",
            "diagnostic_only": True,
            "compares_signals": [
                "prior-artifact learned future-forgetting risk",
                "MIR current-update interference",
            ],
        },
        replay={
            "enabled": True,
            "buffer_capacity": config.replay_capacity,
            "replay_batch_size": config.replay_batch_size,
            "replay_seed": config.replay_seed,
            "insertion_policy": config.replay_insertion_policy,
            "sampling_policy": "uniform_random_from_candidate_pool",
            "mir_candidate_size": config.mir_candidate_size,
            "mir_virtual_lr": (
                config.mir_virtual_lr
                if config.mir_virtual_lr is not None
                else config.learning_rate
            ),
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


def _train_mir_interference_diagnostic(
    *,
    model: nn.Module,
    train_stream,
    eval_stream,
    optimizer: torch.optim.Optimizer,
    trainer_config: ContinualTrainerConfig,
    replay_buffer: ReservoirReplayBuffer,
    replay_batch_size: int,
    mir_candidate_size: int,
    mir_virtual_lr: float,
    replay_seed: int,
    scheduler: SpacedReplayScheduler,
    diagnostic_logger: MIRInterferenceDiagnosticLogger,
    signal_logger: SampleSignalLogger | None = None,
) -> ContinualTrainingResult:
    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    if mir_candidate_size < replay_batch_size:
        raise ValueError("mir_candidate_size must be >= replay_batch_size")
    if mir_virtual_lr <= 0:
        raise ValueError("mir_virtual_lr must be positive")

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
    rng = Random(replay_seed + 30_000)

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

                if len(replay_buffer) > 0:
                    candidate_items = replay_buffer.sample_items(
                        batch_size=mir_candidate_size,
                    )
                    learned_risk_scores = scheduler.score_items(candidate_items)
                    candidate_scores = score_mir_replay_candidates(
                        model=model,
                        current_x=current_x,
                        current_y=current_y,
                        candidate_items=candidate_items,
                        virtual_lr=mir_virtual_lr,
                        global_step=global_step,
                        device=device,
                    )
                    diagnostic_logger.record(
                        candidate_scores=candidate_scores,
                        learned_risk_scores=learned_risk_scores,
                        replay_batch_size=replay_batch_size,
                    )
                    actual_replay_items = rng.sample(
                        candidate_items,
                        min(replay_batch_size, len(candidate_items)),
                    )
                    replay_batch = replay_buffer.sample_batch_by_sample_ids(
                        sample_ids=[item.sample_id for item in actual_replay_items],
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
            "schedule": "mir_interference_diagnostic",
            "replay_batch_size": replay_batch_size,
            "mir_candidate_size": mir_candidate_size,
            "mir_virtual_lr": mir_virtual_lr,
            "actual_replay_policy": "uniform_random_from_candidate_pool",
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
    return ContinualTrainingResult(
        accuracy_matrix=accuracy_matrix,
        train_losses=train_losses,
        training_time_seconds=perf_counter() - start,
        task_count=task_count,
        method_metrics={
            "replay": replay_summary,
            "scheduler": scheduler.summary(),
            "mir_interference": diagnostic_logger.summary(),
        },
    )


def run_mir_interference_diagnostic(
    config: MIRInterferenceDiagnosticConfig,
) -> RandomReplayBaselineRun:
    """Run the MIR-interference diagnostic and save artifacts."""

    if config.target_key != "original_class_id":
        raise ValueError("MIR interference diagnostic defaults to class-incremental targets")
    if config.replay_insertion_policy != "reservoir_task_end":
        raise ValueError("only reservoir_task_end insertion is currently implemented")
    if config.mir_candidate_size < config.replay_batch_size:
        raise ValueError("mir_candidate_size must be >= replay_batch_size")

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
    diagnostic_logger = MIRInterferenceDiagnosticLogger()
    signal_logger = SampleSignalLogger() if config.log_signals else None
    mir_virtual_lr = (
        config.mir_virtual_lr
        if config.mir_virtual_lr is not None
        else config.learning_rate
    )
    result = _train_mir_interference_diagnostic(
        model=model,
        train_stream=train_stream,
        eval_stream=eval_stream,
        optimizer=optimizer,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        replay_batch_size=config.replay_batch_size,
        mir_candidate_size=config.mir_candidate_size,
        mir_virtual_lr=mir_virtual_lr,
        replay_seed=config.replay_seed,
        scheduler=scheduler,
        diagnostic_logger=diagnostic_logger,
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
            "research_role": "mir_interference_diagnostic",
            "risk_scorer": risk_scorer.to_json_metadata(),
        },
        extra_json_artifacts={
            **(
                {"sample_signals": signal_logger.to_json_payload()}
                if signal_logger is not None
                else {}
            ),
            "scheduler_trace": scheduler.to_json_payload(),
            "mir_interference_diagnostic": diagnostic_logger.to_json_payload(),
        },
    )
    return RandomReplayBaselineRun(
        result=result,
        artifacts=artifacts,
        run_config=run_config,
    )


def load_config(path: str | Path) -> MIRInterferenceDiagnosticConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "mir_interference_diagnostic" not in raw:
        raise ValueError("config must contain a mir_interference_diagnostic section")
    section = raw["mir_interference_diagnostic"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    return MIRInterferenceDiagnosticConfig(**section)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run MIR interference diagnostic")
    parser.add_argument("--config", type=str, required=True, help="Path to diagnostic YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        run = run_mir_interference_diagnostic(config)
    except BaselineDataUnavailableError as exc:
        print(f"Diagnostic data unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Saved MIR interference diagnostic artifacts to: {run.artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

