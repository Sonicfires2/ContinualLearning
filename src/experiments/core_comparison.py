"""Task 13 controlled core comparison runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from omegaconf import OmegaConf

from src.baselines.fine_tuning import FineTuningBaselineConfig, run_fine_tuning_baseline
from src.baselines.random_replay import RandomReplayBaselineConfig, run_random_replay_baseline
from src.baselines.fixed_periodic_replay import (
    FixedPeriodicReplayBaselineConfig,
    run_fixed_periodic_replay_baseline,
)
from src.baselines.spaced_replay import SpacedReplayBaselineConfig, run_spaced_replay_baseline


CORE_METHODS = (
    "fine_tuning",
    "random_replay",
    "fixed_periodic_replay",
    "spaced_replay",
)


@dataclass(frozen=True)
class CoreComparisonConfig:
    """Shared controls for the Task 13 core comparison."""

    protocol_id: str = "core_split_cifar100_v2"
    comparison_name: str = "task13_core_comparison"
    output_root: str = "experiments/task13_core_comparison"
    overwrite: bool = False
    smoke: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
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
    fixed_periodic_replay_interval: int = 1
    fixed_periodic_budget_mode: str = "budget_matched"
    min_replay_interval_steps: int = 1
    max_replay_interval_steps: int = 64
    scheduler_loss_scale: float = 5.0
    scheduler_loss_weight: float = 1.0
    scheduler_uncertainty_weight: float = 1.0
    scheduler_target_probability_weight: float = 1.0
    scheduler_loss_increase_weight: float = 0.5
    scheduler_budget_mode: str = "match_random_replay"


@dataclass(frozen=True)
class PlannedRun:
    """One method/seed run planned for the comparison."""

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


def planned_runs(config: CoreComparisonConfig) -> list[PlannedRun]:
    """Return the method/seed plan in deterministic execution order."""

    return [
        PlannedRun(
            method_name=method_name,
            seed=seed,
            run_name=f"{config.comparison_name}_{method_name}_seed{seed}",
        )
        for seed in config.seeds
        for method_name in CORE_METHODS
    ]


def _common_kwargs(
    config: CoreComparisonConfig,
    *,
    method_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "protocol_id": config.protocol_id,
        "method_name": method_name,
        "run_name": f"{config.comparison_name}_{method_name}_seed{seed}",
        "seed": seed,
        "output_root": config.output_root,
        "overwrite": config.overwrite,
        "smoke": config.smoke,
        "data_root": config.data_root,
        "download": config.download,
        "task_count": config.task_count,
        "classes_per_task": config.classes_per_task,
        "split_seed": config.split_seed,
        "epochs_per_task": config.epochs_per_task,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "device": config.device,
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "target_key": config.target_key,
        "log_signals": config.log_signals,
    }


def baseline_config_for(
    config: CoreComparisonConfig,
    *,
    method_name: str,
    seed: int,
):
    """Build the concrete baseline config for one method and seed."""

    common = _common_kwargs(config, method_name=method_name, seed=seed)
    if method_name == "fine_tuning":
        return FineTuningBaselineConfig(**common)

    replay_kwargs = {
        **common,
        "replay_capacity": config.replay_capacity,
        "replay_batch_size": config.replay_batch_size,
        "replay_seed": seed,
        "replay_insertion_policy": config.replay_insertion_policy,
    }
    if method_name == "random_replay":
        return RandomReplayBaselineConfig(**replay_kwargs)
    if method_name == "fixed_periodic_replay":
        return FixedPeriodicReplayBaselineConfig(
            **replay_kwargs,
            replay_interval=config.fixed_periodic_replay_interval,
            budget_mode=config.fixed_periodic_budget_mode,
        )
    if method_name == "spaced_replay":
        return SpacedReplayBaselineConfig(
            **replay_kwargs,
            min_replay_interval_steps=config.min_replay_interval_steps,
            max_replay_interval_steps=config.max_replay_interval_steps,
            scheduler_loss_scale=config.scheduler_loss_scale,
            scheduler_loss_weight=config.scheduler_loss_weight,
            scheduler_uncertainty_weight=config.scheduler_uncertainty_weight,
            scheduler_target_probability_weight=config.scheduler_target_probability_weight,
            scheduler_loss_increase_weight=config.scheduler_loss_increase_weight,
            scheduler_budget_mode=config.scheduler_budget_mode,
        )
    raise ValueError(f"unknown core method {method_name!r}")


def _runner_for(method_name: str) -> Callable[[Any], Any]:
    if method_name == "fine_tuning":
        return run_fine_tuning_baseline
    if method_name == "random_replay":
        return run_random_replay_baseline
    if method_name == "fixed_periodic_replay":
        return run_fixed_periodic_replay_baseline
    if method_name == "spaced_replay":
        return run_spaced_replay_baseline
    raise ValueError(f"unknown core method {method_name!r}")


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


def aggregate_core_comparison(
    *,
    config: CoreComparisonConfig,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate method/seed run payloads into summary statistics."""

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in CORE_METHODS}
    for run in runs:
        by_method[run["method_name"]].append(run)

    aggregates: dict[str, Any] = {}
    for method_name, method_runs in by_method.items():
        final_accuracies = [_metric(run, "final_accuracy") for run in method_runs]
        average_forgetting_values = [
            _metric(run, "average_forgetting") for run in method_runs
        ]
        training_times = [
            _metric(run, "training_time_seconds") for run in method_runs
        ]
        replay_samples = [
            float(
                run["metrics"]
                .get("method_metrics", {})
                .get("replay", {})
                .get("total_replay_samples", 0)
            )
            for run in method_runs
        ]
        aggregates[method_name] = {
            "run_count": len(method_runs),
            "final_accuracy_mean": _safe_mean(final_accuracies),
            "final_accuracy_std": _safe_std(final_accuracies),
            "average_forgetting_mean": _safe_mean(average_forgetting_values),
            "average_forgetting_std": _safe_std(average_forgetting_values),
            "training_time_seconds_mean": _safe_mean(training_times),
            "training_time_seconds_std": _safe_std(training_times),
            "total_replay_samples_mean": _safe_mean(replay_samples),
            "total_replay_samples_std": _safe_std(replay_samples),
        }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "comparison_name": config.comparison_name,
        "protocol_id": config.protocol_id,
        "config": asdict(config),
        "fairness_controls": {
            "same_task_order": True,
            "same_split_seed": config.split_seed,
            "same_model_architecture": True,
            "same_optimizer_settings": True,
            "same_epochs_per_task": True,
            "same_replay_capacity_for_replay_methods": config.replay_capacity,
            "same_replay_batch_size_for_replay_methods": config.replay_batch_size,
            "budget_matched_replay_methods": (
                config.fixed_periodic_replay_interval == 1
                and config.scheduler_budget_mode == "match_random_replay"
            ),
        },
        "methods": list(CORE_METHODS),
        "seeds": list(config.seeds),
        "aggregates": aggregates,
        "runs": runs,
    }


def run_core_comparison(config: CoreComparisonConfig) -> dict[str, Any]:
    """Run all core methods across the configured seed list."""

    runs: list[dict[str, Any]] = []
    for planned in planned_runs(config):
        baseline_config = baseline_config_for(
            config,
            method_name=planned.method_name,
            seed=planned.seed,
        )
        run = _runner_for(planned.method_name)(baseline_config)
        runs.append(
            {
                "method_name": planned.method_name,
                "seed": planned.seed,
                "run_name": planned.run_name,
                "run_dir": str(run.artifacts.run_dir),
                "metrics": {
                    "final_accuracy": run.result.method_metrics.get(
                        "final_accuracy",
                        None,
                    ),
                },
            }
        )
        metrics_path = run.artifacts.metrics
        with metrics_path.open("r", encoding="utf-8") as handle:
            runs[-1]["metrics"] = json.load(handle)

    summary = aggregate_core_comparison(config=config, runs=runs)
    output_path = Path(config.output_root) / f"{config.comparison_name}_summary.json"
    _atomic_write_json(output_path, summary)
    summary["summary_path"] = str(output_path)
    return summary


def load_config(path: str | Path) -> CoreComparisonConfig:
    """Load a core comparison config from YAML."""

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "core_comparison" not in raw:
        raise ValueError("config must contain a core_comparison section")
    section = raw["core_comparison"]
    if "hidden_dims" in section:
        section["hidden_dims"] = tuple(section["hidden_dims"])
    if "seeds" in section:
        section["seeds"] = tuple(section["seeds"])
    return CoreComparisonConfig(**section)
