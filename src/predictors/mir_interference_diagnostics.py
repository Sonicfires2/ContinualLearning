"""Diagnostics comparing learned forgetting risk with MIR interference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.replay import MIRCandidateScore


MIR_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MIRInterferenceDiagnosticRow:
    """One replay-memory candidate scored by MIR and learned forgetting risk."""

    global_step: int
    optimizer_step: int
    sample_id: int
    source_task_id: int
    original_class_id: int
    replay_count_before_selection: int
    last_replay_step_before_selection: int | None
    learned_risk_score: float
    mir_pre_update_loss: float
    mir_post_update_loss: float
    mir_interference_score: float
    mir_candidate_rank: int
    candidate_count: int
    is_mir_topk: bool


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _metric_payload(rows: list[MIRInterferenceDiagnosticRow]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "positive_count": 0,
            "positive_rate": None,
            "average_precision": None,
            "roc_auc": None,
        }
    y_true = np.array([int(row.is_mir_topk) for row in rows], dtype=int)
    scores = np.array([row.learned_risk_score for row in rows], dtype=float)
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    payload: dict[str, Any] = {
        "n": int(len(rows)),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": positives / len(rows),
        "average_precision": None,
        "roc_auc": None,
    }
    if positives > 0:
        payload["average_precision"] = float(average_precision_score(y_true, scores))
    if positives > 0 and negatives > 0:
        payload["roc_auc"] = float(roc_auc_score(y_true, scores))
    return payload


def _event_overlap(rows: list[MIRInterferenceDiagnosticRow]) -> dict[str, Any]:
    rows_by_step: dict[int, list[MIRInterferenceDiagnosticRow]] = {}
    for row in rows:
        rows_by_step.setdefault(row.global_step, []).append(row)

    overlaps: list[float] = []
    random_expected_overlaps: list[float] = []
    for event_rows in rows_by_step.values():
        mir_top = [row for row in event_rows if row.is_mir_topk]
        k = len(mir_top)
        if k == 0:
            continue
        mir_ids = {row.sample_id for row in mir_top}
        learned_top = sorted(
            event_rows,
            key=lambda row: (-row.learned_risk_score, row.sample_id),
        )[:k]
        learned_ids = {row.sample_id for row in learned_top}
        overlaps.append(len(mir_ids & learned_ids) / k)
        random_expected_overlaps.append(k / len(event_rows))

    return {
        "event_count": len(rows_by_step),
        "mean_topk_overlap": _mean(overlaps),
        "mean_random_expected_topk_overlap": _mean(random_expected_overlaps),
        "mean_overlap_minus_random_expected": (
            _mean([overlap - expected for overlap, expected in zip(overlaps, random_expected_overlaps, strict=True)])
            if overlaps
            else None
        ),
    }


@dataclass
class MIRInterferenceDiagnosticLogger:
    """Append-only diagnostic logger for MIR candidate pools."""

    rows: list[MIRInterferenceDiagnosticRow] = field(default_factory=list)

    def record(
        self,
        *,
        candidate_scores: Sequence[MIRCandidateScore],
        learned_risk_scores: dict[int, float],
        replay_batch_size: int,
    ) -> None:
        if replay_batch_size < 1:
            raise ValueError("replay_batch_size must be positive")
        top_rank = min(replay_batch_size, len(candidate_scores))
        for candidate in candidate_scores:
            if candidate.sample_id not in learned_risk_scores:
                raise KeyError(
                    f"missing learned risk score for sample_id={candidate.sample_id}"
                )
            self.rows.append(
                MIRInterferenceDiagnosticRow(
                    global_step=candidate.global_step,
                    optimizer_step=candidate.optimizer_step,
                    sample_id=candidate.sample_id,
                    source_task_id=candidate.source_task_id,
                    original_class_id=candidate.original_class_id,
                    replay_count_before_selection=candidate.replay_count_before_selection,
                    last_replay_step_before_selection=candidate.last_replay_step_before_selection,
                    learned_risk_score=float(learned_risk_scores[candidate.sample_id]),
                    mir_pre_update_loss=candidate.pre_update_loss,
                    mir_post_update_loss=candidate.post_update_loss,
                    mir_interference_score=candidate.interference_score,
                    mir_candidate_rank=candidate.candidate_rank,
                    candidate_count=candidate.candidate_count,
                    is_mir_topk=candidate.candidate_rank <= top_rank,
                )
            )

    def summary(self) -> dict[str, Any]:
        top_rows = [row for row in self.rows if row.is_mir_topk]
        non_top_rows = [row for row in self.rows if not row.is_mir_topk]
        candidate_counts = [row.candidate_count for row in self.rows]
        return {
            "schema_version": MIR_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION,
            "candidate_row_count": len(self.rows),
            "event_count": len({row.global_step for row in self.rows}),
            "mean_candidate_count": _mean(candidate_counts),
            "learned_risk_predicts_mir_topk": _metric_payload(self.rows),
            "event_topk_overlap": _event_overlap(self.rows),
            "mean_learned_risk_mir_topk": _mean(
                [row.learned_risk_score for row in top_rows]
            ),
            "mean_learned_risk_non_topk": _mean(
                [row.learned_risk_score for row in non_top_rows]
            ),
            "mean_mir_interference_topk": _mean(
                [row.mir_interference_score for row in top_rows]
            ),
            "mean_mir_interference_non_topk": _mean(
                [row.mir_interference_score for row in non_top_rows]
            ),
        }

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MIR_INTERFERENCE_DIAGNOSTIC_SCHEMA_VERSION,
            "definition": {
                "question": "Does the learned future-forgetting risk score agree with MIR's current-update interference ranking?",
                "mir_target": "is_mir_topk indicates candidates that MIR would replay from the same candidate pool",
                "learned_score": "online scheduler risk score from the prior-artifact logistic forgetting predictor",
                "leakage_guard": "MIR scores use only the current batch and memory candidates before the actual optimizer step",
            },
            "summary": self.summary(),
            "rows": [asdict(row) for row in self.rows],
        }

