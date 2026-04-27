"""Learned forgetting-predictor comparisons from leakage-safe artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

from src.predictors.forgetting_risk import (
    FEATURE_COLUMNS,
    HEURISTIC_SCORE_NAMES,
    PRIMARY_TARGET,
    TemporalSplit,
    build_feature_rows,
    evaluate_heuristics,
    sha256_file,
)
from src.predictors.time_to_forgetting import (
    PRIMARY_TIME_TARGET,
    build_time_feature_rows,
)


LEARNED_PREDICTOR_REPORT_SCHEMA_VERSION = 1
CONTINUOUS_FORGETTING_TARGETS = (
    "max_future_loss_increase",
    "final_loss_delta",
    "max_future_target_probability_drop",
    "final_target_probability_drop",
    "max_future_confidence_drop",
    "final_confidence_drop",
)


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


def _eligible_binary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["eligible_for_binary_forgetting"]]


def _split_rows(
    rows: list[dict[str, Any]],
    split: TemporalSplit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [
        row
        for row in rows
        if int(row["anchor_trained_task_id"]) <= split.train_anchor_task_max
    ]
    test_rows = [
        row
        for row in rows
        if int(row["anchor_trained_task_id"]) >= split.test_anchor_task_min
    ]
    return train_rows, test_rows


def _feature_matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    return np.array([[float(row[column]) for column in feature_columns] for row in rows])


def _precision_at_fraction(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"fraction": fraction, "k": 0, "precision": None, "recall": None}
    k = max(1, int(math.ceil(len(y_true) * fraction)))
    order = np.argsort(scores)[::-1]
    selected = order[:k]
    positives = int(np.sum(y_true))
    true_positives = int(np.sum(y_true[selected]))
    return {
        "fraction": fraction,
        "k": k,
        "precision": true_positives / k,
        "recall": true_positives / positives if positives else None,
    }


def _binary_metric_payload(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    payload: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": positives / len(y_true) if len(y_true) else None,
        "average_precision": None,
        "roc_auc": None,
        "precision_at_10_percent": _precision_at_fraction(y_true, scores, 0.10),
        "precision_at_20_percent": _precision_at_fraction(y_true, scores, 0.20),
        "precision_at_30_percent": _precision_at_fraction(y_true, scores, 0.30),
    }
    if len(y_true) and positives > 0:
        payload["average_precision"] = float(average_precision_score(y_true, scores))
    if positives > 0 and negatives > 0:
        payload["roc_auc"] = float(roc_auc_score(y_true, scores))
        precision, recall, _ = precision_recall_curve(y_true, scores)
        payload["pr_curve_point_count"] = int(len(precision))
        payload["max_precision"] = float(np.max(precision))
        payload["max_recall"] = float(np.max(recall))
    return payload


def _probability_threshold_payload(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    positives = int(np.sum(y_true))
    rows: list[dict[str, Any]] = []
    for threshold in (0.3, 0.5, 0.7, 0.9):
        selected = probabilities >= threshold
        selected_count = int(np.sum(selected))
        true_positives = int(np.sum(y_true[selected])) if selected_count else 0
        rows.append(
            {
                "threshold": threshold,
                "selected_count": selected_count,
                "selected_fraction": (
                    selected_count / len(y_true) if len(y_true) else None
                ),
                "precision": (
                    true_positives / selected_count if selected_count else None
                ),
                "recall": true_positives / positives if positives else None,
            }
        )
    return rows


def _regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)) if len(y_true) else None,
        "target_std": float(np.std(y_true)) if len(y_true) else None,
        "mae": None,
        "rmse": None,
        "r2": None,
    }
    if len(y_true):
        payload["mae"] = float(mean_absolute_error(y_true, predictions))
        payload["rmse"] = float(math.sqrt(mean_squared_error(y_true, predictions)))
        if len(y_true) >= 2 and float(np.var(y_true)) > 0.0:
            payload["r2"] = float(r2_score(y_true, predictions))
    return payload


def _fit_model(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> list[str]:
    """Fit a scikit-learn model and return non-fatal warning messages."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    return [str(warning.message) for warning in caught]


def _fit_status(warnings_: list[str]) -> str:
    return "fit_with_warnings" if warnings_ else "fit"


def _empty_model_payload(
    *,
    model_family: str,
    target: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    reason: str,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "target": target,
        "feature_columns": list(feature_columns),
        "train": {"n": len(train_rows)},
        "test": {"n": len(test_rows)},
        "status": "skipped",
        "reason": reason,
        "metrics": None,
    }


def evaluate_binary_learned_models(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_name: str = PRIMARY_TARGET,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Fit proposal-suggested binary forgetting predictors on a temporal split."""

    eligible_rows = _eligible_binary_rows(feature_rows)
    if split is None:
        split = _default_split(eligible_rows)
    train_rows, test_rows = _split_rows(eligible_rows, split)
    y_train = np.array([int(row[target_name]) for row in train_rows], dtype=int)
    y_test = np.array([int(row[target_name]) for row in test_rows], dtype=int)

    models: dict[str, dict[str, Any]] = {}
    base_payload = {
        "target": target_name,
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "train": {
            "n": len(train_rows),
            "positive_count": int(np.sum(y_train)),
        },
        "test": {
            "n": len(test_rows),
            "positive_count": int(np.sum(y_test)),
        },
    }
    if not train_rows or not test_rows:
        reason = "empty temporal split"
        for name in ("logistic_regression", "linear_svm_classifier"):
            models[name] = _empty_model_payload(
                model_family=name,
                target=target_name,
                train_rows=train_rows,
                test_rows=test_rows,
                reason=reason,
                feature_columns=feature_columns,
            )
        return {**base_payload, "models": models}
    if len(set(y_train.tolist())) < 2:
        reason = "training split has only one target class"
        for name in ("logistic_regression", "linear_svm_classifier"):
            models[name] = _empty_model_payload(
                model_family=name,
                target=target_name,
                train_rows=train_rows,
                test_rows=test_rows,
                reason=reason,
                feature_columns=feature_columns,
            )
        return {**base_payload, "models": models}

    x_train = _feature_matrix(train_rows, feature_columns)
    x_test = _feature_matrix(test_rows, feature_columns)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    logistic_warnings = _fit_model(logistic, x_train, y_train)
    logistic_scores = logistic.predict_proba(x_test)[:, 1]
    models["logistic_regression"] = {
        "model_family": "logistic_regression",
        "target": target_name,
        "feature_columns": list(feature_columns),
        "train": {
            "n": len(train_rows),
            "positive_count": int(np.sum(y_train)),
        },
        "test": {
            "n": len(test_rows),
            "positive_count": int(np.sum(y_test)),
        },
        "status": _fit_status(logistic_warnings),
        "reason": None,
        "fit_warnings": logistic_warnings,
        "score_kind": "predict_proba_positive_class",
        "metrics": _binary_metric_payload(y_test, logistic_scores),
        "threshold_behavior": _probability_threshold_payload(y_test, logistic_scores),
    }

    linear_svm = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", dual="auto", random_state=0, max_iter=20000),
    )
    svm_warnings = _fit_model(linear_svm, x_train, y_train)
    svm_scores = linear_svm.decision_function(x_test)
    models["linear_svm_classifier"] = {
        "model_family": "linear_svm_classifier",
        "target": target_name,
        "feature_columns": list(feature_columns),
        "train": {
            "n": len(train_rows),
            "positive_count": int(np.sum(y_train)),
        },
        "test": {
            "n": len(test_rows),
            "positive_count": int(np.sum(y_test)),
        },
        "status": _fit_status(svm_warnings),
        "reason": None,
        "fit_warnings": svm_warnings,
        "score_kind": "decision_function_margin",
        "metrics": _binary_metric_payload(y_test, svm_scores),
        "threshold_behavior": {
            "note": "LinearSVC margins are not calibrated probabilities; use top-fraction precision or calibrate before scheduler use."
        },
    }
    return {**base_payload, "models": models}


def evaluate_continuous_forgetting_models(
    feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_names: tuple[str, ...] = CONTINUOUS_FORGETTING_TARGETS,
) -> dict[str, Any]:
    """Fit linear models for continuous future deterioration targets."""

    eligible_rows = _eligible_binary_rows(feature_rows)
    if split is None:
        split = _default_split(eligible_rows)
    train_rows, test_rows = _split_rows(eligible_rows, split)
    payload: dict[str, Any] = {
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "train": {"n": len(train_rows)},
        "test": {"n": len(test_rows)},
        "targets": {},
    }
    if not train_rows or not test_rows:
        for target_name in target_names:
            payload["targets"][target_name] = {
                "status": "skipped",
                "reason": "empty temporal split",
                "models": {},
            }
        return payload

    x_train = _feature_matrix(train_rows)
    x_test = _feature_matrix(test_rows)
    model_factories = {
        "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "ridge_regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "linear_svm_regressor": lambda: make_pipeline(
            StandardScaler(),
            LinearSVR(random_state=0, max_iter=20000),
        ),
    }
    for target_name in target_names:
        if target_name not in train_rows[0] or target_name not in test_rows[0]:
            payload["targets"][target_name] = {
                "status": "skipped",
                "reason": f"target {target_name!r} is not present in feature rows",
                "models": {},
            }
            continue
        y_train = np.array([float(row[target_name]) for row in train_rows], dtype=float)
        y_test = np.array([float(row[target_name]) for row in test_rows], dtype=float)
        target_payload = {
            "status": "fit",
            "reason": None,
            "train": {
                "n": len(train_rows),
                "target_mean": float(np.mean(y_train)),
            },
            "test": {
                "n": len(test_rows),
                "target_mean": float(np.mean(y_test)),
            },
            "models": {},
        }
        constant_prediction = np.full(len(y_test), float(np.mean(y_train)))
        target_payload["models"]["constant_train_mean"] = {
            "model_family": "constant_train_mean",
            "status": "fit",
            "metrics": _regression_metrics(y_test, constant_prediction),
        }
        for model_name, factory in model_factories.items():
            model = factory()
            fit_warnings = _fit_model(model, x_train, y_train)
            predictions = model.predict(x_test)
            target_payload["models"][model_name] = {
                "model_family": model_name,
                "status": _fit_status(fit_warnings),
                "fit_warnings": fit_warnings,
                "metrics": _regression_metrics(y_test, predictions),
            }
        payload["targets"][target_name] = target_payload
    return payload


def evaluate_time_to_forgetting_learned_models(
    time_feature_rows: list[dict[str, Any]],
    *,
    split: TemporalSplit | None = None,
    target_name: str = PRIMARY_TIME_TARGET,
) -> dict[str, Any]:
    """Fit simple regressors for observed time-to-forgetting events."""

    eligible_rows = [
        row
        for row in time_feature_rows
        if row["eligible_for_time_to_forgetting"] and row["event_observed"]
    ]
    if split is None:
        split = _default_split(eligible_rows)
    train_rows, test_rows = _split_rows(eligible_rows, split)
    payload: dict[str, Any] = {
        "target": target_name,
        "target_scope": "observed events only; right-censored rows are excluded from regression fitting",
        "temporal_split": {
            "train_anchor_task_max": split.train_anchor_task_max,
            "test_anchor_task_min": split.test_anchor_task_min,
        },
        "train": {"n": len(train_rows)},
        "test": {"n": len(test_rows)},
        "models": {},
    }
    if not train_rows or not test_rows:
        payload["status"] = "skipped"
        payload["reason"] = "empty temporal split after filtering observed events"
        return payload

    x_train = _feature_matrix(train_rows)
    x_test = _feature_matrix(test_rows)
    y_train = np.array([float(row[target_name]) for row in train_rows], dtype=float)
    y_test = np.array([float(row[target_name]) for row in test_rows], dtype=float)
    model_factories = {
        "constant_train_mean": None,
        "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "ridge_regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "linear_svm_regressor": lambda: make_pipeline(
            StandardScaler(),
            LinearSVR(random_state=0, max_iter=20000),
        ),
    }
    for model_name, factory in model_factories.items():
        if factory is None:
            predictions = np.full(len(y_test), float(np.mean(y_train)))
        else:
            model = factory()
            fit_warnings = _fit_model(model, x_train, y_train)
            predictions = model.predict(x_test)
        payload["models"][model_name] = {
            "model_family": model_name,
            "status": "fit" if factory is None else _fit_status(fit_warnings),
            "fit_warnings": [] if factory is None else fit_warnings,
            "metrics": _regression_metrics(y_test, predictions),
        }
    payload["status"] = "fit"
    payload["reason"] = None
    return payload


def _best_average_precision(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_name = None
    best_ap = None
    for name, payload in models.items():
        metrics = payload.get("metrics")
        if not metrics:
            continue
        ap = metrics.get("average_precision")
        if ap is not None and (best_ap is None or ap > best_ap):
            best_name = name
            best_ap = ap
    return {"model": best_name, "average_precision": best_ap}


def _best_heuristic_average_precision(
    heuristic_report: dict[str, Any],
) -> dict[str, Any]:
    best_name = None
    best_ap = None
    for name, metrics in heuristic_report["heuristics"].items():
        ap = metrics.get("average_precision")
        if ap is not None and (best_ap is None or ap > best_ap):
            best_name = name
            best_ap = ap
    return {"heuristic": best_name, "average_precision": best_ap}


def build_learned_predictor_report(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
    time_payload: dict[str, Any] | None = None,
    signal_path: str | Path | None = None,
    label_path: str | Path | None = None,
    time_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the Task 16 learned-predictor comparison report."""

    feature_rows = build_feature_rows(
        signal_payload=signal_payload,
        label_payload=label_payload,
    )
    eligible_rows = _eligible_binary_rows(feature_rows)
    split = _default_split(eligible_rows)
    heuristic_report = evaluate_heuristics(feature_rows, split=split)
    binary_report = evaluate_binary_learned_models(feature_rows, split=split)
    continuous_report = evaluate_continuous_forgetting_models(
        feature_rows,
        split=split,
    )
    time_report = None
    if time_payload is not None:
        time_rows = build_time_feature_rows(
            signal_payload=signal_payload,
            time_payload=time_payload,
        )
        time_report = evaluate_time_to_forgetting_learned_models(
            time_rows,
            split=split,
        )

    source_signal = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source_signal = {"path": str(signal_path), "sha256": sha256_file(signal_path)}
    source_labels = None
    if label_path is not None:
        label_path = Path(label_path)
        source_labels = {"path": str(label_path), "sha256": sha256_file(label_path)}
    source_time = None
    if time_path is not None:
        time_path = Path(time_path)
        source_time = {"path": str(time_path), "sha256": sha256_file(time_path)}

    best_heuristic = _best_heuristic_average_precision(heuristic_report)
    best_binary_model = _best_average_precision(binary_report["models"])
    beats_best_heuristic = None
    if (
        best_heuristic["average_precision"] is not None
        and best_binary_model["average_precision"] is not None
    ):
        beats_best_heuristic = (
            best_binary_model["average_precision"]
            > best_heuristic["average_precision"]
        )

    return {
        "schema_version": LEARNED_PREDICTOR_REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": PRIMARY_TARGET,
        "source_signal_artifact": source_signal,
        "source_label_artifact": source_labels,
        "source_time_to_forgetting_artifact": source_time,
        "definition": {
            "feature_rule": "features use seen_task_eval rows at or before each anchor only",
            "binary_target_rule": "forgot_any_future from forgetting_labels.json",
            "continuous_target_rule": "future deterioration fields from forgetting_labels.json",
            "time_target_rule": "optional observed first_observed_forgetting_step_delta from time_to_forgetting_targets.json",
            "temporal_evaluation": "earlier anchor tasks are used for fitting and scaling; later anchor tasks are held out",
            "scheduler_use_warning": "offline labels evaluate predictors only; online schedulers must fit from prior runs or prior anchors without reading future labels",
        },
        "feature_summary": {
            "row_count": len(feature_rows),
            "eligible_binary_row_count": len(eligible_rows),
            "feature_columns": list(FEATURE_COLUMNS),
            "heuristic_scores": list(HEURISTIC_SCORE_NAMES),
            "continuous_targets": list(CONTINUOUS_FORGETTING_TARGETS),
        },
        "comparison_summary": {
            "best_heuristic_by_average_precision": best_heuristic,
            "best_binary_model_by_average_precision": best_binary_model,
            "best_binary_model_beats_best_heuristic": beats_best_heuristic,
        },
        "heuristic_report": heuristic_report,
        "binary_classification_report": binary_report,
        "continuous_forgetting_report": continuous_report,
        "time_to_forgetting_report": time_report,
    }


def build_learned_predictor_report_from_paths(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    time_path: str | Path | None = None,
) -> dict[str, Any]:
    signal_path = Path(signal_path)
    label_path = Path(label_path)
    time_payload = None
    resolved_time_path = None
    if time_path is not None:
        resolved_time_path = Path(time_path)
        time_payload = _read_json(resolved_time_path)
    return build_learned_predictor_report(
        signal_payload=_read_json(signal_path),
        label_payload=_read_json(label_path),
        time_payload=time_payload,
        signal_path=signal_path,
        label_path=label_path,
        time_path=resolved_time_path,
    )


def save_learned_predictor_report(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
    time_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_learned_predictor_report_from_paths(
        signal_path=signal_path,
        label_path=label_path,
        time_path=time_path,
    )
    _atomic_write_json(Path(output_path), report)
    return report
