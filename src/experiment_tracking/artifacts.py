"""Schema-versioned experiment artifacts for reproducible runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any

from src.metrics.continual import (
    AccuracyMatrix,
    average_accuracy_after_task,
    average_forgetting,
    final_accuracy,
    forgetting_by_task,
)
from src.training.continual import ContinualTrainingResult


ARTIFACT_SCHEMA_VERSION = 1
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Minimum run configuration required for a defensible experiment artifact."""

    protocol_id: str
    method_name: str
    seed: int
    dataset: dict[str, Any]
    model: dict[str, Any]
    trainer: dict[str, Any]
    evaluation: dict[str, Any]
    run_name: str | None = None
    task_split: dict[str, Any] = field(default_factory=dict)
    method: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths written for one experiment run."""

    run_dir: Path
    manifest: Path
    config: Path
    metrics: Path
    accuracy_matrix: Path
    train_losses: Path
    environment: Path
    extra: dict[str, Path] = field(default_factory=dict)


def _safe_name(value: str) -> str:
    normalized = _SAFE_NAME_PATTERN.sub("-", value.strip())
    return normalized.strip("-") or "run"


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
        json.dump(
            payload,
            tmp_file,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def collect_environment_snapshot() -> dict[str, Any]:
    """Collect lightweight environment metadata needed for reproducibility."""

    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": _torch_environment(),
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "is_dirty": _git_value(["status", "--porcelain"]) not in (None, ""),
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _is_finite_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def validate_accuracy_matrix(accuracy_matrix: AccuracyMatrix) -> None:
    """Validate the lower-triangular seen-task accuracy matrix contract."""

    if not accuracy_matrix:
        raise ValueError("accuracy_matrix must not be empty")

    task_count = len(accuracy_matrix)
    for row_index, row in enumerate(accuracy_matrix):
        if len(row) != task_count:
            raise ValueError("accuracy_matrix must be square")
        for column_index, value in enumerate(row):
            if column_index <= row_index:
                if value is None:
                    raise ValueError(
                        "accuracy_matrix is missing a seen-task value at "
                        f"row {row_index}, column {column_index}"
                    )
                if not _is_finite_probability(float(value)):
                    raise ValueError(
                        "accuracy_matrix values must be finite probabilities "
                        f"in [0, 1], got {value!r}"
                    )
            elif value is not None:
                raise ValueError(
                    "accuracy_matrix contains a future-task value at "
                    f"row {row_index}, column {column_index}"
                )


def _validate_train_losses(train_losses: list[float]) -> None:
    for index, value in enumerate(train_losses):
        if not math.isfinite(float(value)):
            raise ValueError(f"train_losses[{index}] is not finite: {value!r}")


def _validate_json_safe(value: Any, *, path: str = "method_metrics") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key: {key!r}")
            _validate_json_safe(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON-safe value: {type(value).__name__}")


def validate_run_config(run_config: ExperimentRunConfig) -> None:
    """Validate that a run config contains the minimum evidence fields."""

    if not run_config.protocol_id.strip():
        raise ValueError("run_config.protocol_id must be set")
    if not run_config.method_name.strip():
        raise ValueError("run_config.method_name must be set")
    if not isinstance(run_config.seed, int):
        raise ValueError("run_config.seed must be an int")

    required_sections = {
        "dataset": run_config.dataset,
        "model": run_config.model,
        "trainer": run_config.trainer,
        "evaluation": run_config.evaluation,
    }
    for section_name, section in required_sections.items():
        if not isinstance(section, dict) or not section:
            raise ValueError(f"run_config.{section_name} must be a non-empty dict")


def summarize_training_result(result: ContinualTrainingResult) -> dict[str, Any]:
    """Compute summary metrics from the trainer result and validate them."""

    validate_accuracy_matrix(result.accuracy_matrix)
    _validate_train_losses(result.train_losses)
    _validate_json_safe(result.method_metrics)
    if result.training_time_seconds < 0 or not math.isfinite(result.training_time_seconds):
        raise ValueError("training_time_seconds must be finite and non-negative")

    task_count = len(result.accuracy_matrix)
    if result.task_count not in (0, task_count):
        raise ValueError(
            f"result.task_count={result.task_count} does not match matrix size {task_count}"
        )

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "task_count": task_count,
        "final_accuracy": final_accuracy(result.accuracy_matrix),
        "average_forgetting": average_forgetting(result.accuracy_matrix),
        "forgetting_by_task": {
            str(task_id): value
            for task_id, value in forgetting_by_task(result.accuracy_matrix).items()
        },
        "average_accuracy_after_task": [
            average_accuracy_after_task(result.accuracy_matrix, task_id)
            for task_id in range(task_count)
        ],
        "training_time_seconds": result.training_time_seconds,
        "train_loss_count": len(result.train_losses),
        "first_train_loss": result.train_losses[0] if result.train_losses else None,
        "last_train_loss": result.train_losses[-1] if result.train_losses else None,
        "method_metrics": result.method_metrics,
    }


def _run_dir_for(
    *,
    output_root: Path,
    run_config: ExperimentRunConfig,
    created_at: str,
) -> Path:
    run_name = run_config.run_name or (
        f"{created_at}_{run_config.method_name}_seed-{run_config.seed}"
    )
    return output_root / _safe_name(run_config.method_name) / _safe_name(run_name)


def _artifact_paths(run_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        run_dir=run_dir,
        manifest=run_dir / "manifest.json",
        config=run_dir / "config.json",
        metrics=run_dir / "metrics.json",
        accuracy_matrix=run_dir / "accuracy_matrix.json",
        train_losses=run_dir / "train_losses.json",
        environment=run_dir / "environment.json",
    )


def _write_manifest(paths: ArtifactPaths, *, metadata: dict[str, Any]) -> None:
    artifact_files = {
        "config": paths.config,
        "metrics": paths.metrics,
        "accuracy_matrix": paths.accuracy_matrix,
        "train_losses": paths.train_losses,
        "environment": paths.environment,
    }
    artifact_files.update(paths.extra)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": metadata,
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _sha256_file(path),
            }
            for name, path in artifact_files.items()
        },
    }
    _atomic_write_json(paths.manifest, payload)


def save_experiment_artifacts(
    *,
    output_root: str | Path,
    run_config: ExperimentRunConfig,
    result: ContinualTrainingResult,
    extra_metadata: dict[str, Any] | None = None,
    extra_json_artifacts: dict[str, Any] | None = None,
    overwrite: bool = False,
    environment: dict[str, Any] | None = None,
) -> ArtifactPaths:
    """Save a complete, reloadable run artifact bundle.

    The function fails fast on invalid metrics and refuses to overwrite an
    existing run directory unless explicitly requested.
    """

    validate_run_config(run_config)
    metrics = summarize_training_result(result)
    created_at = _utc_timestamp()
    output_root_path = Path(output_root)
    run_dir = _run_dir_for(
        output_root=output_root_path,
        run_config=run_config,
        created_at=created_at,
    )
    extra_artifacts = extra_json_artifacts or {}
    for artifact_name, payload in extra_artifacts.items():
        if artifact_name in {
            "config",
            "metrics",
            "accuracy_matrix",
            "train_losses",
            "environment",
            "manifest",
        }:
            raise ValueError(f"extra artifact name {artifact_name!r} is reserved")
        _validate_json_safe(payload, path=f"extra_json_artifacts.{artifact_name}")

    paths = _artifact_paths(run_dir)
    if extra_artifacts:
        paths = replace(
            paths,
            extra={
                artifact_name: run_dir / f"{_safe_name(artifact_name)}.json"
                for artifact_name in extra_artifacts
            },
        )

    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"run artifact directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": asdict(run_config),
    }
    matrix_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "accuracy_matrix": result.accuracy_matrix,
    }
    losses_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "train_losses": result.train_losses,
    }
    environment_payload = environment or collect_environment_snapshot()

    _atomic_write_json(paths.config, config_payload)
    _atomic_write_json(paths.metrics, metrics)
    _atomic_write_json(paths.accuracy_matrix, matrix_payload)
    _atomic_write_json(paths.train_losses, losses_payload)
    _atomic_write_json(paths.environment, environment_payload)
    for artifact_name, payload in extra_artifacts.items():
        _atomic_write_json(paths.extra[artifact_name], payload)
    _write_manifest(
        paths,
        metadata={
            "created_at_utc": created_at,
            "protocol_id": run_config.protocol_id,
            "method_name": run_config.method_name,
            "seed": run_config.seed,
            "extra": extra_metadata or {},
        },
    )
    return paths


def load_experiment_artifacts(run_dir: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Load a saved artifact bundle and optionally verify manifest hashes."""

    run_dir_path = Path(run_dir)
    paths = _artifact_paths(run_dir_path)
    manifest = _read_json(paths.manifest)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")

    loaded = {"manifest": manifest}
    for artifact_name, artifact_info in manifest["artifacts"].items():
        artifact_path = run_dir_path / artifact_info["path"]
        if verify_hashes:
            actual_hash = _sha256_file(artifact_path)
            expected_hash = artifact_info["sha256"]
            if actual_hash != expected_hash:
                raise ValueError(
                    f"artifact hash mismatch for {artifact_name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        loaded[artifact_name] = _read_json(artifact_path)

    validate_accuracy_matrix(loaded["accuracy_matrix"]["accuracy_matrix"])
    return loaded
