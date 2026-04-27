"""Online forgetting-risk scorer trained from prior artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.predictors.forgetting_risk import (
    FEATURE_COLUMNS,
    PRIMARY_TARGET,
    build_feature_rows,
    sha256_file,
)
from src.predictors.signal_ablations import SIGNAL_FEATURE_GROUPS


@dataclass(frozen=True)
class OnlineForgettingRiskScorerMetadata:
    """Auditable metadata for the offline-trained online scorer."""

    model_family: str
    target: str
    feature_group: str
    feature_columns: tuple[str, ...]
    training_row_count: int
    positive_count: int
    positive_rate: float
    train_anchor_task_max: int | None
    source_signal_artifact: dict[str, str] | None
    source_label_artifact: dict[str, str] | None


class OnlineForgettingRiskScorer:
    """Predict future-forgetting probability from online scheduler features."""

    def __init__(
        self,
        *,
        model: Any,
        metadata: OnlineForgettingRiskScorerMetadata,
    ) -> None:
        self.model = model
        self.metadata = metadata

    def score(self, feature_row: dict[str, Any]) -> float:
        """Return the positive-class forgetting probability for one feature row."""

        x = np.array(
            [[float(feature_row[column]) for column in self.metadata.feature_columns]],
            dtype=float,
        )
        probability = float(self.model.predict_proba(x)[0, 1])
        return max(0.0, min(1.0, probability))

    def to_json_metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.metadata.model_family,
            "target": self.metadata.target,
            "feature_group": self.metadata.feature_group,
            "feature_columns": list(self.metadata.feature_columns),
            "training_row_count": self.metadata.training_row_count,
            "positive_count": self.metadata.positive_count,
            "positive_rate": self.metadata.positive_rate,
            "train_anchor_task_max": self.metadata.train_anchor_task_max,
            "source_signal_artifact": self.metadata.source_signal_artifact,
            "source_label_artifact": self.metadata.source_label_artifact,
        }


def train_online_forgetting_risk_scorer(
    *,
    signal_payload: dict[str, Any],
    label_payload: dict[str, Any],
    feature_group: str = "all_features",
    train_anchor_task_max: int | None = None,
    signal_path: str | Path | None = None,
    label_path: str | Path | None = None,
) -> OnlineForgettingRiskScorer:
    """Train a logistic online scorer from prior signal and label artifacts."""

    if feature_group not in SIGNAL_FEATURE_GROUPS:
        raise ValueError(f"unknown feature_group {feature_group!r}")
    feature_columns = SIGNAL_FEATURE_GROUPS[feature_group]
    rows = [
        row
        for row in build_feature_rows(
            signal_payload=signal_payload,
            label_payload=label_payload,
        )
        if row["eligible_for_binary_forgetting"]
        and (
            train_anchor_task_max is None
            or int(row["anchor_trained_task_id"]) <= train_anchor_task_max
        )
    ]
    if not rows:
        raise ValueError("online scorer training split is empty")

    missing = [column for column in feature_columns if column not in rows[0]]
    if missing:
        raise ValueError(f"online scorer feature columns are missing: {missing}")

    y = np.array([int(row[PRIMARY_TARGET]) for row in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("online scorer training split has only one target class")

    x = np.array([[float(row[column]) for column in feature_columns] for row in rows])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    model.fit(x, y)

    source_signal = None
    if signal_path is not None:
        signal_path = Path(signal_path)
        source_signal = {"path": str(signal_path), "sha256": sha256_file(signal_path)}
    source_labels = None
    if label_path is not None:
        label_path = Path(label_path)
        source_labels = {"path": str(label_path), "sha256": sha256_file(label_path)}

    metadata = OnlineForgettingRiskScorerMetadata(
        model_family="logistic_regression",
        target=PRIMARY_TARGET,
        feature_group=feature_group,
        feature_columns=tuple(feature_columns),
        training_row_count=len(rows),
        positive_count=int(np.sum(y)),
        positive_rate=float(np.mean(y)),
        train_anchor_task_max=train_anchor_task_max,
        source_signal_artifact=source_signal,
        source_label_artifact=source_labels,
    )
    return OnlineForgettingRiskScorer(model=model, metadata=metadata)


def train_online_forgetting_risk_scorer_from_paths(
    *,
    signal_path: str | Path,
    label_path: str | Path,
    feature_group: str = "all_features",
    train_anchor_task_max: int | None = None,
) -> OnlineForgettingRiskScorer:
    import json

    signal_path = Path(signal_path)
    label_path = Path(label_path)
    with signal_path.open("r", encoding="utf-8") as handle:
        signal_payload = json.load(handle)
    with label_path.open("r", encoding="utf-8") as handle:
        label_payload = json.load(handle)
    return train_online_forgetting_risk_scorer(
        signal_payload=signal_payload,
        label_payload=label_payload,
        feature_group=feature_group,
        train_anchor_task_max=train_anchor_task_max,
        signal_path=signal_path,
        label_path=label_path,
    )
