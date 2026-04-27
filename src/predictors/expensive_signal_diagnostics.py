"""Diagnostics for expensive sample-level signals such as gradient norms."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from src.predictors.forgetting_risk import (
    FEATURE_COLUMNS,
    PRIMARY_TARGET,
    build_feature_rows,
    sha256_file,
)
from src.predictors.signal_ablations import evaluate_signal_ablations


EXPENSIVE_SIGNAL_DIAGNOSTIC_SCHEMA_VERSION = 1
GRADIENT_FEATURE_COLUMNS = (
    "anchor_last_layer_gradient_l2",
    "anchor_logit_gradient_l2",
    "anchor_penultimate_activation_l2",
    "anchor_classifier_weight_gradient_l2",
    "anchor_classifier_bias_gradient_l2",
    "previous_last_layer_gradient_l2",
    "last_layer_gradient_delta_from_previous",
    "last_layer_gradient_increase_from_previous",
    "max_last_layer_gradient_l2_so_far",
)
EXPENSIVE_SIGNAL_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "cheap_all_features": FEATURE_COLUMNS,
    "gradient_only": GRADIENT_FEATURE_COLUMNS,
    "cheap_plus_gradient": FEATURE_COLUMNS + GRADIENT_FEATURE_COLUMNS,
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
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


def _as_int(row: dict[str, Any], key: str) -> int:
    if key not in row:
        raise KeyError(f"row is missing required key {key!r}")
    return int(row[key])


def _as_float(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise KeyError(f"row is missing required key {key!r}")
    return float(row[key])


def _gradient_history(
    gradient_payload: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    rows = gradient_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("gradient payload must contain rows")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("observation_type") != "seen_task_eval":
            continue
        grouped.setdefault(_as_int(row, "sample_id"), []).append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(
            key=lambda row: (
                _as_int(row, "trained_task_id"),
                _as_int(row, "global_step"),
            )
        )
    return grouped


def _find_gradient_anchor(
    *,
    feature_row: dict[str, Any],
    sample_history: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_task = _as_int(feature_row, "anchor_trained_task_id")
    anchor_step = _as_int(feature_row, "anchor_global_step")
    source_task = _as_int(feature_row, "source_task_id")
    history = [
        row
        for row in sample_history
        if (
            _as_int(row, "trained_task_id") < anchor_task
            or (
                _as_int(row, "trained_task_id") == anchor_task
                and _as_int(row, "global_step") <= anchor_step
            )
        )
    ]
    matches = [
        row
        for row in history
        if _as_int(row, "trained_task_id") == anchor_task
        and _as_int(row, "global_step") == anchor_step
        and _as_int(row, "evaluated_task_id") == source_task
    ]
    if len(matches) != 1:
        raise ValueError(
            "could not uniquely match gradient anchor for "
            f"sample_id={feature_row.get('sample_id')!r}, "
            f"anchor_task={anchor_task}, anchor_step={anchor_step}"
        )
    return matches[0], history


def augment_feature_rows_with_gradient_signals(
    *,
    feature_rows: list[dict[str, Any]],
    gradient_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join gradient history onto leakage-safe feature rows at each anchor."""

    histories = _gradient_history(gradient_payload)
    augmented_rows = []
    for feature_row in feature_rows:
        sample_id = _as_int(feature_row, "sample_id")
        if sample_id not in histories:
            continue
        anchor, history = _find_gradient_anchor(
            feature_row=feature_row,
            sample_history=histories[sample_id],
        )
        previous = history[-2] if len(history) >= 2 else None
        anchor_grad = _as_float(anchor, "last_layer_gradient_l2")
        previous_grad = (
            _as_float(previous, "last_layer_gradient_l2")
            if previous is not None
            else anchor_grad
        )
        history_grads = [_as_float(row, "last_layer_gradient_l2") for row in history]
        row = dict(feature_row)
        row.update(
            {
                "anchor_last_layer_gradient_l2": anchor_grad,
                "anchor_logit_gradient_l2": _as_float(anchor, "logit_gradient_l2"),
                "anchor_penultimate_activation_l2": _as_float(
                    anchor,
                    "penultimate_activation_l2",
                ),
                "anchor_classifier_weight_gradient_l2": _as_float(
                    anchor,
                    "classifier_weight_gradient_l2",
                ),
                "anchor_classifier_bias_gradient_l2": _as_float(
                    anchor,
                    "classifier_bias_gradient_l2",
                ),
                "previous_last_layer_gradient_l2": previous_grad,
                "last_layer_gradient_delta_from_previous": anchor_grad - previous_grad,
                "last_layer_gradient_increase_from_previous": max(
                    0.0,
                    anchor_grad - previous_grad,
                ),
                "max_last_layer_gradient_l2_so_far": max(history_grads),
                "gradient_feature_uses_rows_up_to_trained_task_id": _as_int(
                    feature_row,
                    "anchor_trained_task_id",
                ),
                "gradient_feature_uses_rows_up_to_global_step": _as_int(
                    feature_row,
                    "anchor_global_step",
                ),
            }
        )
        augmented_rows.append(row)
    if not augmented_rows:
        raise ValueError("no feature rows could be matched with gradient signals")
    return augmented_rows


def _artifact_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _runtime_overhead(
    *,
    reference_metrics_payload: dict[str, Any] | None,
    diagnostic_metrics_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if reference_metrics_payload is None or diagnostic_metrics_payload is None:
        return None
    reference = float(reference_metrics_payload["training_time_seconds"])
    diagnostic = float(diagnostic_metrics_payload["training_time_seconds"])
    return {
        "reference_training_time_seconds": reference,
        "diagnostic_training_time_seconds": diagnostic,
        "absolute_overhead_seconds": diagnostic - reference,
        "relative_overhead": (diagnostic / reference) - 1.0 if reference > 0 else None,
    }


def _best_group_ap(ablation_report: dict[str, Any], group_name: str) -> float | None:
    payload = ablation_report["feature_group_reports"].get(group_name)
    if not payload:
        return None
    return payload["best_model"]["average_precision"]


def _recommendation(ablation_report: dict[str, Any]) -> dict[str, Any]:
    cheap_ap = _best_group_ap(ablation_report, "cheap_all_features")
    gradient_ap = _best_group_ap(ablation_report, "gradient_only")
    combined_ap = _best_group_ap(ablation_report, "cheap_plus_gradient")
    if cheap_ap is None or combined_ap is None:
        return {
            "build_replay_intervention_next": False,
            "reason": "diagnostic did not produce comparable cheap and gradient-augmented AP",
        }
    if combined_ap > cheap_ap + 0.01:
        return {
            "build_replay_intervention_next": True,
            "reason": "gradient-augmented features improve AP by more than 0.01",
            "cheap_all_features_average_precision": cheap_ap,
            "gradient_only_average_precision": gradient_ap,
            "cheap_plus_gradient_average_precision": combined_ap,
        }
    return {
        "build_replay_intervention_next": False,
        "reason": "gradient-augmented features do not improve AP enough to justify a replay intervention",
        "cheap_all_features_average_precision": cheap_ap,
        "gradient_only_average_precision": gradient_ap,
        "cheap_plus_gradient_average_precision": combined_ap,
    }


def build_expensive_signal_diagnostic_report(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
    gradient_payload: dict[str, Any],
    signal_path: str | Path | None = None,
    label_path: str | Path | None = None,
    gradient_path: str | Path | None = None,
    reference_metrics_payload: dict[str, Any] | None = None,
    diagnostic_metrics_payload: dict[str, Any] | None = None,
    reference_metrics_path: str | Path | None = None,
    diagnostic_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    base_rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    augmented_rows = augment_feature_rows_with_gradient_signals(
        feature_rows=base_rows,
        gradient_payload=gradient_payload,
    )
    ablation_report = evaluate_signal_ablations(
        augmented_rows,
        feature_groups=EXPENSIVE_SIGNAL_FEATURE_GROUPS,
    )
    return {
        "schema_version": EXPENSIVE_SIGNAL_DIAGNOSTIC_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": PRIMARY_TARGET,
        "source_signal_artifact": _artifact_info(signal_path),
        "source_label_artifact": _artifact_info(label_path),
        "source_gradient_artifact": _artifact_info(gradient_path),
        "source_reference_metrics_artifact": _artifact_info(reference_metrics_path),
        "source_diagnostic_metrics_artifact": _artifact_info(diagnostic_metrics_path),
        "definition": {
            "signal_family": "gradient_norm",
            "gradient_scope": "final linear layer",
            "feature_rule": "gradient features use seen_task_eval rows at or before each anchor only",
            "target_rule": "forgot_any_future from forgetting_labels.json",
            "purpose": "diagnose whether gradient-family signals improve forgetting prediction before building another replay scheduler",
        },
        "feature_summary": {
            "base_row_count": len(base_rows),
            "gradient_augmented_row_count": len(augmented_rows),
            "cheap_feature_columns": list(FEATURE_COLUMNS),
            "gradient_feature_columns": list(GRADIENT_FEATURE_COLUMNS),
        },
        "gradient_signal_summary": gradient_payload.get("summary", {}),
        "runtime_overhead": _runtime_overhead(
            reference_metrics_payload=reference_metrics_payload,
            diagnostic_metrics_payload=diagnostic_metrics_payload,
        ),
        "ablation_report": ablation_report,
        "recommendation": _recommendation(ablation_report),
    }


def build_expensive_signal_diagnostic_report_from_paths(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    gradient_path: str | Path,
    reference_metrics_path: str | Path | None = None,
    diagnostic_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    signal_path = Path(signal_path)
    label_path = Path(label_path)
    gradient_path = Path(gradient_path)
    reference_payload = _read_json(Path(reference_metrics_path)) if reference_metrics_path else None
    diagnostic_payload = _read_json(Path(diagnostic_metrics_path)) if diagnostic_metrics_path else None
    return build_expensive_signal_diagnostic_report(
        signal_payload=_read_json(signal_path),
        label_payload=_read_json(label_path),
        gradient_payload=_read_json(gradient_path),
        signal_path=signal_path,
        label_path=label_path,
        gradient_path=gradient_path,
        reference_metrics_payload=reference_payload,
        diagnostic_metrics_payload=diagnostic_payload,
        reference_metrics_path=reference_metrics_path,
        diagnostic_metrics_path=diagnostic_metrics_path,
    )


def save_expensive_signal_diagnostic_report(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    gradient_path: str | Path,
    output_path: str | Path,
    reference_metrics_path: str | Path | None = None,
    diagnostic_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_expensive_signal_diagnostic_report_from_paths(
        signal_path=signal_path,
        label_path=label_path,
        gradient_path=gradient_path,
        reference_metrics_path=reference_metrics_path,
        diagnostic_metrics_path=diagnostic_metrics_path,
    )
    _atomic_write_json(Path(output_path), report)
    return report
