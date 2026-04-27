"""Feature-group ablations for forgetting-risk predictors."""

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
    TemporalSplit,
    build_feature_rows,
    evaluate_heuristics,
    sha256_file,
)
from src.predictors.learned_forgetting import evaluate_binary_learned_models


SIGNAL_ABLATION_REPORT_SCHEMA_VERSION = 1
SIGNAL_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "loss_only": (
        "anchor_loss",
        "previous_loss",
        "loss_delta_from_previous",
        "loss_increase_from_previous",
        "max_loss_so_far",
    ),
    "uncertainty_only": (
        "anchor_uncertainty",
        "previous_uncertainty",
        "uncertainty_delta_from_previous",
        "max_uncertainty_so_far",
    ),
    "target_probability_only": (
        "anchor_target_probability",
        "previous_target_probability",
        "target_probability_delta_from_previous",
        "target_probability_drop_from_previous",
        "min_target_probability_so_far",
    ),
    "anchor_state": (
        "anchor_loss",
        "anchor_uncertainty",
        "anchor_confidence",
        "anchor_target_probability",
    ),
    "history_delta": (
        "loss_delta_from_previous",
        "uncertainty_delta_from_previous",
        "target_probability_delta_from_previous",
        "loss_increase_from_previous",
        "target_probability_drop_from_previous",
    ),
    "history_summary": (
        "history_eval_count",
        "has_previous_eval",
        "previous_correct",
        "correct_history_rate",
        "tasks_since_source",
        "anchor_task_progress",
    ),
    "all_features": FEATURE_COLUMNS,
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


def _eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["eligible_for_binary_forgetting"]]


def _default_split(rows: list[dict[str, Any]]) -> TemporalSplit:
    anchor_tasks = sorted({int(row["anchor_trained_task_id"]) for row in rows})
    if len(anchor_tasks) < 2:
        raise ValueError("temporal evaluation requires at least two anchor tasks")
    split_index = max(1, int(len(anchor_tasks) * 0.6)) - 1
    split_index = min(split_index, len(anchor_tasks) - 2)
    return TemporalSplit(
        train_anchor_task_max=anchor_tasks[split_index],
        test_anchor_task_min=anchor_tasks[split_index + 1],
    )


def _best_heuristic(heuristic_report: dict[str, Any]) -> dict[str, Any]:
    best_name = None
    best_ap = None
    for name, metrics in heuristic_report["heuristics"].items():
        ap = metrics.get("average_precision")
        if ap is not None and (best_ap is None or ap > best_ap):
            best_name = name
            best_ap = ap
    return {"name": best_name, "average_precision": best_ap}


def _best_model(model_report: dict[str, Any]) -> dict[str, Any]:
    best_name = None
    best_ap = None
    for name, payload in model_report["models"].items():
        metrics = payload.get("metrics")
        if not metrics:
            continue
        ap = metrics.get("average_precision")
        if ap is not None and (best_ap is None or ap > best_ap):
            best_name = name
            best_ap = ap
    return {"name": best_name, "average_precision": best_ap}


def _threshold_recommendation(
    threshold_rows: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(threshold_rows, list):
        return {"threshold": None, "reason": "no calibrated probability thresholds"}
    candidates = [
        row
        for row in threshold_rows
        if row.get("selected_count", 0) > 0
        and row.get("precision") is not None
        and row.get("recall") is not None
    ]
    if not candidates:
        return {"threshold": None, "reason": "no threshold selected positives"}
    best = max(
        candidates,
        key=lambda row: (
            row["precision"],
            row["recall"],
            -abs((row["selected_fraction"] or 0.0) - 0.2),
        ),
    )
    return {
        "threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "selected_fraction": best["selected_fraction"],
        "selection_rule": "highest precision among fixed logistic probability thresholds",
    }


def _ablation_recommendation(
    *,
    best_heuristic: dict[str, Any],
    best_group: dict[str, Any],
    all_features: dict[str, Any],
) -> dict[str, Any]:
    best_group_ap = best_group.get("average_precision")
    heuristic_ap = best_heuristic.get("average_precision")
    all_ap = all_features.get("best_model", {}).get("average_precision")
    if best_group_ap is None:
        return {
            "use_learned_online_gate_next": False,
            "reason": "no learned feature group produced average precision",
        }
    if heuristic_ap is not None and best_group_ap <= heuristic_ap:
        return {
            "use_learned_online_gate_next": False,
            "reason": "learned feature groups did not beat the best cheap heuristic",
        }
    if all_ap is not None and best_group_ap + 0.01 < all_ap:
        return {
            "use_learned_online_gate_next": False,
            "reason": "best compact group trails all features by more than 0.01 AP; inspect feature interactions first",
        }
    return {
        "use_learned_online_gate_next": True,
        "reason": "learned feature groups beat the best cheap heuristic on the temporal holdout",
    }


def evaluate_signal_ablations(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_name: str = PRIMARY_TARGET,
    feature_groups: dict[str, tuple[str, ...]] = SIGNAL_FEATURE_GROUPS,
) -> dict[str, Any]:
    """Evaluate learned predictor feature groups on a temporal holdout."""

    eligible_rows = _eligible_rows(feature_rows)
    if split is None:
        split = _default_split(eligible_rows)

    heuristic_report = evaluate_heuristics(
        feature_rows,
        split=split,
        target_name=target_name,
    )
    best_heuristic = _best_heuristic(heuristic_report)
    group_reports = {}
    for group_name, columns in feature_groups.items():
        missing = [column for column in columns if column not in feature_rows[0]]
        if missing:
            group_reports[group_name] = {
                "status": "skipped",
                "reason": f"missing feature columns: {missing}",
                "feature_columns": list(columns),
                "best_model": {"name": None, "average_precision": None},
            }
            continue
        report = evaluate_binary_learned_models(
            feature_rows,
            split=split,
            target_name=target_name,
            feature_columns=columns,
        )
        group_reports[group_name] = {
            "status": "fit",
            "feature_columns": list(columns),
            "best_model": _best_model(report),
            "model_report": report,
        }

    ranked_groups = sorted(
        (
            {
                "feature_group": name,
                "best_model": payload["best_model"]["name"],
                "average_precision": payload["best_model"]["average_precision"],
            }
            for name, payload in group_reports.items()
            if payload["best_model"]["average_precision"] is not None
        ),
        key=lambda row: row["average_precision"],
        reverse=True,
    )
    best_group = ranked_groups[0] if ranked_groups else {
        "feature_group": None,
        "best_model": None,
        "average_precision": None,
    }
    all_features_payload = group_reports.get("all_features", {})
    logistic_thresholds = (
        all_features_payload.get("model_report", {})
        .get("models", {})
        .get("logistic_regression", {})
        .get("threshold_behavior", [])
    )

    return {
        "target": target_name,
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "feature_group_definitions": {
            name: list(columns) for name, columns in feature_groups.items()
        },
        "best_heuristic": best_heuristic,
        "ranked_feature_groups": ranked_groups,
        "feature_group_reports": group_reports,
        "all_features_logistic_threshold_recommendation": _threshold_recommendation(
            logistic_thresholds,
        ),
        "recommendation": _ablation_recommendation(
            best_heuristic=best_heuristic,
            best_group=best_group,
            all_features=all_features_payload,
        ),
    }


def build_signal_ablation_report(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
    signal_path: str | Path | None = None,
    label_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete Task 17 signal-ablation report."""

    feature_rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    ablation_report = evaluate_signal_ablations(feature_rows)

    source_signal = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source_signal = {"path": str(signal_path), "sha256": sha256_file(signal_path)}
    source_labels = None
    if label_path is not None:
        label_path = Path(label_path)
        source_labels = {"path": str(label_path), "sha256": sha256_file(label_path)}

    return {
        "schema_version": SIGNAL_ABLATION_REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": PRIMARY_TARGET,
        "source_signal_artifact": source_signal,
        "source_label_artifact": source_labels,
        "definition": {
            "feature_rule": "features use seen_task_eval rows at or before each anchor only",
            "target_rule": "forgot_any_future from forgetting_labels.json",
            "temporal_evaluation": "earlier anchor tasks are used for fitting/scaling; later anchors are held out",
            "purpose": "identify which cheap signal groups support learned forgetting-risk ranking before using them in an online replay gate",
        },
        "feature_summary": {
            "row_count": len(feature_rows),
            "eligible_binary_row_count": len(_eligible_rows(feature_rows)),
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_groups": {
                name: list(columns) for name, columns in SIGNAL_FEATURE_GROUPS.items()
            },
        },
        "ablation_report": ablation_report,
    }


def build_signal_ablation_report_from_paths(
    *,
    signal_path: str | Path,
    label_path: str | Path,
) -> dict[str, Any]:
    signal_path = Path(signal_path)
    label_path = Path(label_path)
    return build_signal_ablation_report(
        signal_payload=_read_json(signal_path),
        label_payload=_read_json(label_path),
        signal_path=signal_path,
        label_path=label_path,
    )


def save_signal_ablation_report(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = build_signal_ablation_report_from_paths(
        signal_path=signal_path,
        label_path=label_path,
    )
    _atomic_write_json(Path(output_path), report)
    return report
