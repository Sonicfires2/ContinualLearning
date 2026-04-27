"""Task 14 MIR replay comparison runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from omegaconf import OmegaConf

from src.baselines.mir_replay import MIRReplayBaselineConfig, run_mir_replay_baseline


@dataclass(frozen=True)
class MIRReplayComparisonConfig:
    """Shared controls for the Task 14 MIR replay comparison."""

    protocol_id: str = "core_split_cifar100_v2"
    comparison_name: str = "task14_mir_replay_comparison"
    output_root: str = "experiments/task14_mir_replay_comparison"
    overwrite: bool = False
    smoke: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
    reference_summary_path: str | None = None
    data_root: str = "data"
    download: bool = False
    task_count: int = 10
    classes_per_task: int = 10
    split_seed: int = 0
    epochs_per_task: int = 1
    batch_size: int = 32
    eval_batch_size: int = 128
    learning_rate: float = 0.01
    device: str = "auto"
    hidden_dims: tuple[int, ...] = (256,)
    dropout: float = 0.0
    target_key: str = "original_class_id"
    log_signals: bool = True
    replay_capacity: int = 2000
    replay_batch_size: int = 32
    replay_insertion_policy: str = "reservoir_task_end"
    mir_candidate_size: int = 128
    mir_virtual_lr: float | None = None


@dataclass(frozen=True)
class PlannedMIRRun:
    """One MIR run planned for a seed."""

    method_name: str
    seed: int
    run_name: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        newline="\n",
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, sort_keys=True, default=_json_default)
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _safe_mean(values)
    assert mean is not None
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _metric(run_payload: dict[str, Any], key: str) -> float:
    return float(run_payload["metrics"][key])


def planned_runs(config: MIRReplayComparisonConfig) -> list[PlannedMIRRun]:
    """Return the seed plan in deterministic execution order."""

    return [
        PlannedMIRRun(
            method_name="mir_replay",
            seed=seed,
            run_name=f"{config.comparison_name}_mir_replay_seed{seed}",
        )
        for seed in config.seeds
    ]


def baseline_config_for(
    config: MIRReplayComparisonConfig,
    *,
    seed: int,
) -> MIRReplayBaselineConfig:
    """Build the concrete MIR baseline config for one seed."""

    return MIRReplayBaselineConfig(
        protocol_id=config.protocol_id,
        method_name="mir_replay",
        run_name=f"{config.comparison_name}_mir_replay_seed{seed}",
        seed=seed,
        output_root=config.output_root,
        overwrite=config.overwrite,
        smoke=config.smoke,
        data_root=config.data_root,
        download=config.download,
        task_count=config.task_count,
        classes_per_task=config.classes_per_task,
        split_seed=config.split_seed,
        epochs_per_task=config.epochs_per_task,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        target_key=config.target_key,
        log_signals=config.log_signals,
        replay_capacity=config.replay_capacity,
        replay_batch_size=config.replay_batch_size,
        replay_seed=seed,
        replay_insertion_policy=config.replay_insertion_policy,
        mir_candidate_size=config.mir_candidate_size,
        mir_virtual_lr=config.mir_virtual_lr,
    )


def _aggregate_mir_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    final_accuracies = [_metric(run, "final_accuracy") for run in runs]
    average_forgetting_values = [_metric(run, "average_forgetting") for run in runs]
    training_times = [_metric(run, "training_time_seconds") for run in runs]
    replay_samples = [
        float(run["metrics"]["method_metrics"]["replay"]["total_replay_samples"])
        for run in runs
    ]
    unique_replayed_samples = [
        float(run["metrics"]["method_metrics"]["replay"]["unique_replayed_samples"])
        for run in runs
    ]
    selected_interference_scores = [
        float(
            run["metrics"]["method_metrics"]["mir"][
                "mean_selected_interference_score"
            ]
        )
        for run in runs
    ]

    return {
        "run_count": len(runs),
        "final_accuracy_mean": _safe_mean(final_accuracies),
        "final_accuracy_std": _safe_std(final_accuracies),
        "average_forgetting_mean": _safe_mean(average_forgetting_values),
        "average_forgetting_std": _safe_std(average_forgetting_values),
        "training_time_seconds_mean": _safe_mean(training_times),
        "training_time_seconds_std": _safe_std(training_times),
        "total_replay_samples_mean": _safe_mean(replay_samples),
        "total_replay_samples_std": _safe_std(replay_samples),
        "unique_replayed_samples_mean": _safe_mean(unique_replayed_samples),
        "unique_replayed_samples_std": _safe_std(unique_replayed_samples),
        "mean_selected_interference_score_mean": _safe_mean(
            selected_interference_scores
        ),
        "mean_selected_interference_score_std": _safe_std(
            selected_interference_scores
        ),
    }


def _reference_deltas(
    *,
    mir_aggregate: dict[str, Any],
    reference_summary_path: str | None,
) -> dict[str, Any]:
    if not reference_summary_path:
        return {}

    path = Path(reference_summary_path)
    if not path.exists():
        return {
            "reference_summary_path": str(path),
            "reference_summary_loaded": False,
        }

    with path.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)

    deltas: dict[str, Any] = {
        "reference_summary_path": str(path),
        "reference_summary_loaded": True,
        "reference_comparison_name": reference.get("comparison_name"),
    }
    reference_aggregates = reference.get("aggregates", {})
    for method_name, method_metrics in reference_aggregates.items():
        final_accuracy_mean = method_metrics.get("final_accuracy_mean")
        average_forgetting_mean = method_metrics.get("average_forgetting_mean")
        if final_accuracy_mean is not None:
            deltas[f"final_accuracy_mean_delta_mir_minus_{method_name}"] = (
                mir_aggregate["final_accuracy_mean"] - final_accuracy_mean
            )
        if average_forgetting_mean is not None:
            deltas[f"average_forgetting_mean_delta_mir_minus_{method_name}"] = (
                mir_aggregate["average_forgetting_mean"] - average_forgetting_mean
            )
    return deltas


def aggregate_mir_replay_comparison(
    *,
    config: MIRReplayComparisonConfig,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate MIR seed runs into summary statistics."""

    mir_aggregate = _aggregate_mir_runs(runs)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "comparison_name": config.comparison_name,
        "protocol_id": config.protocol_id,
        "config": asdict(config),
        "fairness_controls": {
            "same_task_order_as_core_comparison": True,
            "same_split_seed": config.split_seed,
            "same_model_architecture": True,
            "same_optimizer_settings": True,
            "same_epochs_per_task": True,
            "same_replay_capacity": config.replay_capacity,
            "same_replay_batch_size": config.replay_batch_size,
            "same_replay_sample_budget_as_random_replay": True,
            "candidate_pool_counted_as_selection_overhead_not_replay_budget": True,
        },
        "methods": ["mir_replay"],
        "seeds": list(config.seeds),
        "aggregates": {"mir_replay": mir_aggregate},
        "reference_deltas": _reference_deltas(
            mir_aggregate=mir_aggregate,
            reference_summary_path=config.reference_summary_path,
        ),
        "runs": runs,
    }


def run_mir_replay_comparison(config: MIRReplayComparisonConfig) -> dict[str, Any]:
    """Run MIR across the configured seed list."""

    runs: list[dict[str, Any]] = []
    for planned in planned_runs(config):
        baseline_config = baseline_config_for(config, seed=planned.seed)
        run = run_mir_replay_baseline(baseline_config)
        with run.artifacts.metrics.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        runs.append(
            {
                "method_name": planned.method_name,
                "seed": planned.seed,
                "run_name": planned.run_name,
                "run_dir": str(run.artifacts.run_dir),
                "metrics": metrics,
            }
        )

    summary = aggregate_mir_replay_comparison(config=config, runs=runs)
    output_path = Path(config.output_root) / f"{config.comparison_name}_summary.json"
    _atomic_write_json(output_path, summary)
    summary["summary_path"] = str(output_path)
    return summary


def load_config(path: str | Path) -> MIRReplayComparisonConfig:
    """Load a MIR replay comparison config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "mir_replay_comparison" not in raw:
        raise ValueError("config must contain a mir_replay_comparison section")
    section = raw["mir_replay_comparison"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    if "seeds" in section:
        section["seeds"] = tuple(section["seeds"])
    return MIRReplayComparisonConfig(**section)
