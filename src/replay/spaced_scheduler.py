"""Online spacing-inspired replay scheduler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from src.replay.buffer import ReplayItem


SCHEDULER_TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpacedReplaySchedulerConfig:
    """Configuration for the online due-time proxy."""

    min_interval_steps: int = 1
    max_interval_steps: int = 64
    risk_threshold: float = 0.7
    loss_scale: float = 5.0
    loss_weight: float = 1.0
    uncertainty_weight: float = 1.0
    target_probability_weight: float = 1.0
    loss_increase_weight: float = 0.5
    budget_mode: str = "match_random_replay"
    max_anchor_task_id: int = 9
    risk_score_source: str = "heuristic"


@dataclass
class ReplayScheduleState:
    """Current scheduling state for one sample."""

    sample_id: int
    source_task_id: int
    original_class_id: int
    last_observed_step: int
    last_observed_loss: float
    last_observed_uncertainty: float
    last_observed_confidence: float
    last_observed_target_probability: float
    last_observed_correct: bool
    risk_score: float
    estimated_forgetting_time_steps: int
    next_scheduled_replay_step: int
    observation_count: int = 1
    replay_observation_count: int = 0
    last_replay_step: int | None = None
    previous_loss: float | None = None
    previous_uncertainty: float | None = None
    previous_target_probability: float | None = None
    previous_correct: bool | None = None
    correct_observation_count: int = 0
    max_loss_so_far: float = 0.0
    min_target_probability_so_far: float = 0.5
    max_uncertainty_so_far: float = 0.5


@dataclass(frozen=True)
class SchedulerSelection:
    """One scheduler decision for one replay item."""

    global_step: int
    optimizer_step: int
    sample_id: int
    source_task_id: int
    original_class_id: int
    replay_count_before_selection: int
    last_replay_step_before_selection: int | None
    risk_score: float
    estimated_forgetting_time_steps: int
    next_scheduled_replay_step: int
    overdue_steps: int
    selection_reason: str
    budget_mode: str


@dataclass(frozen=True)
class SchedulerSkip:
    """One replay opportunity skipped by an event-triggered mode."""

    global_step: int
    optimizer_step: int
    candidate_count: int
    due_candidate_count: int
    high_risk_candidate_count: int
    max_risk_score: float | None
    risk_threshold: float
    skip_reason: str
    budget_mode: str


def _as_sequence(value: Any, *, expected_len: int, field_name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != expected_len:
        raise ValueError(
            f"scheduler field {field_name!r} has length {len(values)}, expected {expected_len}"
        )
    return values


def _int_field(batch: dict[str, Any], key: str, index: int, batch_size: int) -> int:
    if key not in batch:
        raise KeyError(f"scheduler observation requires batch metadata key {key!r}")
    return int(_as_sequence(batch[key], expected_len=batch_size, field_name=key)[index])


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class SpacedReplayScheduler:
    """Maintain online sample state and select replay by due time plus risk."""

    def __init__(
        self,
        config: SpacedReplaySchedulerConfig | None = None,
        *,
        risk_scorer: Any | None = None,
    ) -> None:
        self.config = config or SpacedReplaySchedulerConfig()
        if self.config.min_interval_steps < 1:
            raise ValueError("min_interval_steps must be positive")
        if self.config.max_interval_steps < self.config.min_interval_steps:
            raise ValueError("max_interval_steps must be >= min_interval_steps")
        if not 0.0 <= self.config.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0, 1]")
        if self.config.loss_scale <= 0:
            raise ValueError("loss_scale must be positive")
        if self.config.budget_mode not in {
            "match_random_replay",
            "due_only",
            "risk_only",
            "risk_and_due",
            "risk_or_due",
            "risk_ranked",
        }:
            raise ValueError(
                "budget_mode must be one of match_random_replay, due_only, "
                "risk_only, risk_and_due, risk_or_due, or risk_ranked"
            )
        if self.config.max_anchor_task_id < 1:
            raise ValueError("max_anchor_task_id must be positive")
        if self.config.risk_score_source not in {"heuristic", "learned"}:
            raise ValueError("risk_score_source must be heuristic or learned")
        if self.config.risk_score_source == "learned" and risk_scorer is None:
            raise ValueError("learned risk_score_source requires a risk_scorer")
        total_weight = (
            self.config.loss_weight
            + self.config.uncertainty_weight
            + self.config.target_probability_weight
            + self.config.loss_increase_weight
        )
        if total_weight <= 0:
            raise ValueError("at least one scheduler signal weight must be positive")
        self._states: dict[int, ReplayScheduleState] = {}
        self._trace: list[SchedulerSelection] = []
        self._skipped: list[SchedulerSkip] = []
        self._risk_scorer = risk_scorer

    def state_for(self, sample_id: int) -> ReplayScheduleState | None:
        return self._states.get(int(sample_id))

    def score_items(self, items: Iterable[ReplayItem]) -> dict[int, float]:
        """Return current risk scores for replay items without selecting them."""

        item_list = list(items)
        for item in item_list:
            self._ensure_state_for_item(item)
        return {
            item.sample_id: self._states[item.sample_id].risk_score
            for item in item_list
        }

    def observe_batch(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch: dict[str, Any],
        global_step: int,
        is_replay: bool,
        trained_task_id: int | None = None,
    ) -> None:
        """Update scheduling state from information available at this step."""

        if logits.ndim != 2:
            raise ValueError("scheduler expects logits with shape [batch, classes]")
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets have different batch sizes")

        batch_size = int(targets.shape[0])
        if batch_size == 0:
            return

        detached_logits = logits.detach()
        detached_targets = targets.detach().long().to(detached_logits.device)
        losses = F.cross_entropy(detached_logits, detached_targets, reduction="none")
        probabilities = F.softmax(detached_logits, dim=1)
        confidence_values, _ = probabilities.max(dim=1)
        target_probabilities = probabilities.gather(
            1,
            detached_targets.view(-1, 1),
        ).squeeze(1)

        for index in range(batch_size):
            sample_id = _int_field(batch, "sample_id", index, batch_size)
            loss = float(losses[index].detach().cpu().item())
            confidence = float(confidence_values[index].detach().cpu().item())
            target_probability = float(
                target_probabilities[index].detach().cpu().item()
            )
            uncertainty = 1.0 - confidence
            correct = int(torch.argmax(detached_logits[index]).detach().cpu().item()) == int(
                detached_targets[index].detach().cpu().item()
            )
            previous = self._states.get(sample_id)
            previous_loss = previous.last_observed_loss if previous is not None else None
            source_task_id = _int_field(batch, "task_id", index, batch_size)
            anchor_task = int(trained_task_id) if trained_task_id is not None else source_task_id
            feature_row = self._feature_row(
                sample_id=sample_id,
                source_task_id=source_task_id,
                anchor_task_id=anchor_task,
                loss=loss,
                uncertainty=uncertainty,
                confidence=confidence,
                target_probability=target_probability,
                correct=correct,
                previous=previous,
            )
            risk_score = self._score_feature_row(
                feature_row=feature_row,
                loss=loss,
                uncertainty=uncertainty,
                target_probability=target_probability,
                previous_loss=previous_loss,
            )
            estimated_t = self._estimated_time_from_risk(risk_score)
            next_due = int(global_step + estimated_t)
            replay_count = (
                previous.replay_observation_count + int(is_replay)
                if previous is not None
                else int(is_replay)
            )
            observation_count = (
                previous.observation_count + 1 if previous is not None else 1
            )
            correct_observation_count = (
                previous.correct_observation_count + int(correct)
                if previous is not None
                else int(correct)
            )
            self._states[sample_id] = ReplayScheduleState(
                sample_id=sample_id,
                source_task_id=source_task_id,
                original_class_id=_int_field(
                    batch,
                    "original_class_id",
                    index,
                    batch_size,
                ),
                last_observed_step=int(global_step),
                last_observed_loss=loss,
                last_observed_uncertainty=uncertainty,
                last_observed_confidence=confidence,
                last_observed_target_probability=target_probability,
                last_observed_correct=correct,
                risk_score=risk_score,
                estimated_forgetting_time_steps=estimated_t,
                next_scheduled_replay_step=next_due,
                observation_count=observation_count,
                replay_observation_count=replay_count,
                last_replay_step=int(global_step) if is_replay else (
                    previous.last_replay_step if previous is not None else None
                ),
                previous_loss=previous_loss,
                previous_uncertainty=(
                    previous.last_observed_uncertainty if previous is not None else None
                ),
                previous_target_probability=(
                    previous.last_observed_target_probability
                    if previous is not None
                    else None
                ),
                previous_correct=(
                    previous.last_observed_correct if previous is not None else None
                ),
                correct_observation_count=correct_observation_count,
                max_loss_so_far=(
                    max(previous.max_loss_so_far, loss)
                    if previous is not None
                    else loss
                ),
                min_target_probability_so_far=(
                    min(previous.min_target_probability_so_far, target_probability)
                    if previous is not None
                    else target_probability
                ),
                max_uncertainty_so_far=(
                    max(previous.max_uncertainty_so_far, uncertainty)
                    if previous is not None
                    else uncertainty
                ),
            )

    def select(
        self,
        *,
        items: Iterable[ReplayItem],
        global_step: int,
        batch_size: int,
    ) -> list[SchedulerSelection]:
        """Select replay items using due state, filling budget if configured."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        item_list = list(items)
        if not item_list:
            return []
        for item in item_list:
            self._ensure_state_for_item(item)

        records = []
        due_items = []
        future_items = []
        high_risk_items = []
        for item in item_list:
            state = self._states[item.sample_id]
            overdue = int(global_step - state.next_scheduled_replay_step)
            record = (item, state, overdue)
            records.append(record)
            if overdue >= 0:
                due_items.append(record)
            else:
                future_items.append(record)
            if state.risk_score >= self.config.risk_threshold:
                high_risk_items.append(record)

        due_items.sort(
            key=lambda record: (
                -record[2],
                -record[1].risk_score,
                record[1].next_scheduled_replay_step,
                record[0].sample_id,
            )
        )
        high_risk_items.sort(
            key=lambda record: (
                -record[1].risk_score,
                -record[2],
                record[1].next_scheduled_replay_step,
                record[0].sample_id,
            )
        )

        if self.config.budget_mode == "match_random_replay":
            selected_records = due_items[:batch_size]
            selected_ids = {record[0].sample_id for record in selected_records}
            remaining_future_items = [
                record for record in future_items if record[0].sample_id not in selected_ids
            ]
            remaining_future_items.sort(
                key=lambda record: (
                    record[1].next_scheduled_replay_step,
                    -record[1].risk_score,
                    record[0].sample_id,
                )
            )
            selected_records.extend(
                remaining_future_items[: batch_size - len(selected_records)]
            )
        elif self.config.budget_mode == "due_only":
            selected_records = due_items[:batch_size]
        elif self.config.budget_mode == "risk_only":
            selected_records = high_risk_items[:batch_size]
        elif self.config.budget_mode == "risk_and_due":
            selected_records = [
                record
                for record in due_items
                if record[1].risk_score >= self.config.risk_threshold
            ][:batch_size]
        elif self.config.budget_mode == "risk_or_due":
            eligible_records = [
                record
                for record in records
                if record[2] >= 0 or record[1].risk_score >= self.config.risk_threshold
            ]
            eligible_records.sort(
                key=lambda record: (
                    int(record[2] < 0),
                    -record[2],
                    -record[1].risk_score,
                    record[1].next_scheduled_replay_step,
                    record[0].sample_id,
                )
            )
            selected_records = eligible_records[:batch_size]
        elif self.config.budget_mode == "risk_ranked":
            selected_records = sorted(
                records,
                key=lambda record: (
                    -record[1].risk_score,
                    -record[2],
                    record[1].next_scheduled_replay_step,
                    record[0].sample_id,
                ),
            )[:batch_size]
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown budget_mode {self.config.budget_mode!r}")

        if not selected_records:
            self._record_skip(
                global_step=global_step,
                candidate_count=len(records),
                due_candidate_count=len(due_items),
                high_risk_candidate_count=len(high_risk_items),
                max_risk_score=(
                    max(record[1].risk_score for record in records) if records else None
                ),
            )
            return []

        selections: list[SchedulerSelection] = []
        due_ids = {record[0].sample_id for record in due_items}
        high_risk_ids = {record[0].sample_id for record in high_risk_items}
        for item, state, overdue in selected_records:
            reason = self._selection_reason(
                sample_id=item.sample_id,
                due_ids=due_ids,
                high_risk_ids=high_risk_ids,
            )
            selection = SchedulerSelection(
                global_step=int(global_step),
                optimizer_step=int(global_step + 1),
                sample_id=item.sample_id,
                source_task_id=item.task_id,
                original_class_id=item.original_class_id,
                replay_count_before_selection=item.replay_count,
                last_replay_step_before_selection=item.last_replayed_step,
                risk_score=state.risk_score,
                estimated_forgetting_time_steps=state.estimated_forgetting_time_steps,
                next_scheduled_replay_step=state.next_scheduled_replay_step,
                overdue_steps=max(0, overdue),
                selection_reason=reason,
                budget_mode=self.config.budget_mode,
            )
            selections.append(selection)
            self._trace.append(selection)
        return selections

    def summary(self) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        for row in self._trace:
            reason_counts[row.selection_reason] = reason_counts.get(row.selection_reason, 0) + 1
        skip_reason_counts: dict[str, int] = {}
        for row in self._skipped:
            skip_reason_counts[row.skip_reason] = (
                skip_reason_counts.get(row.skip_reason, 0) + 1
            )
        estimates = [
            state.estimated_forgetting_time_steps for state in self._states.values()
        ]
        risks = [state.risk_score for state in self._states.values()]
        return {
            "schema_version": SCHEDULER_TRACE_SCHEMA_VERSION,
            "state_count": len(self._states),
            "trace_row_count": len(self._trace),
            "skipped_selection_event_count": len(self._skipped),
            "selection_reason_counts": reason_counts,
            "skip_reason_counts": skip_reason_counts,
            "budget_mode": self.config.budget_mode,
            "risk_threshold": self.config.risk_threshold,
            "risk_score_source": self.config.risk_score_source,
            "min_interval_steps": self.config.min_interval_steps,
            "max_interval_steps": self.config.max_interval_steps,
            "mean_estimated_forgetting_time_steps": (
                sum(estimates) / len(estimates) if estimates else None
            ),
            "mean_risk_score": sum(risks) / len(risks) if risks else None,
        }

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEDULER_TRACE_SCHEMA_VERSION,
            "definition": {
                "policy": "selection depends on budget_mode: match_random_replay fills the replay budget by due/near-due items; due_only selects only due items; risk-gated modes select only items satisfying configured risk and/or due gates",
                "risk_score": "weighted combination of normalized loss, uncertainty, low target probability, and loss increase",
                "learned_risk_score": "when configured, risk_score is the positive-class probability from a prior-artifact logistic forgetting predictor",
                "estimated_forgetting_time_steps": "linear due-time proxy: high risk maps to min_interval_steps, low risk maps to max_interval_steps",
                "risk_gated_modes": "risk_only, risk_and_due, and risk_or_due skip replay when no sample satisfies the configured risk/due gate",
                "fixed_budget_mode": "risk_ranked always fills the replay batch with the highest-risk available items and does not skip for low risk",
                "future_leakage_guard": "scheduler uses only current/past train or replay logits observed online",
            },
            "config": asdict(self.config),
            "summary": self.summary(),
            "rows": [asdict(row) for row in self._trace],
            "skipped_rows": [asdict(row) for row in self._skipped],
        }

    def _record_skip(
        self,
        *,
        global_step: int,
        candidate_count: int,
        due_candidate_count: int,
        high_risk_candidate_count: int,
        max_risk_score: float | None,
    ) -> None:
        self._skipped.append(
            SchedulerSkip(
                global_step=int(global_step),
                optimizer_step=int(global_step + 1),
                candidate_count=int(candidate_count),
                due_candidate_count=int(due_candidate_count),
                high_risk_candidate_count=int(high_risk_candidate_count),
                max_risk_score=max_risk_score,
                risk_threshold=self.config.risk_threshold,
                skip_reason=self._skip_reason(
                    due_candidate_count=due_candidate_count,
                    high_risk_candidate_count=high_risk_candidate_count,
                ),
                budget_mode=self.config.budget_mode,
            )
        )

    def _skip_reason(
        self,
        *,
        due_candidate_count: int,
        high_risk_candidate_count: int,
    ) -> str:
        if self.config.budget_mode == "due_only":
            return "no_due_items"
        if self.config.budget_mode == "risk_only":
            return "no_high_risk_items"
        if self.config.budget_mode == "risk_and_due":
            if due_candidate_count == 0 and high_risk_candidate_count == 0:
                return "no_due_or_high_risk_items"
            if due_candidate_count == 0:
                return "no_due_items"
            if high_risk_candidate_count == 0:
                return "no_high_risk_items"
            return "no_items_passing_risk_and_due"
        if self.config.budget_mode == "risk_or_due":
            return "no_due_or_high_risk_items"
        if self.config.budget_mode == "risk_ranked":
            return "no_items_available"
        return "no_selected_items"

    def _selection_reason(
        self,
        *,
        sample_id: int,
        due_ids: set[int],
        high_risk_ids: set[int],
    ) -> str:
        is_due = sample_id in due_ids
        is_high_risk = sample_id in high_risk_ids
        if self.config.budget_mode == "match_random_replay":
            return "due" if is_due else "budget_fill_near_due"
        if self.config.budget_mode == "due_only":
            return "due"
        if self.config.budget_mode == "risk_ranked":
            return "risk_ranked"
        if is_due and is_high_risk:
            return "risk_threshold_and_due"
        if is_high_risk:
            return "risk_threshold"
        if is_due:
            return "due"
        return "selected"

    def _ensure_state_for_item(self, item: ReplayItem) -> None:
        if item.sample_id in self._states:
            return
        risk = 0.5
        estimated_t = self._estimated_time_from_risk(risk)
        self._states[item.sample_id] = ReplayScheduleState(
            sample_id=item.sample_id,
            source_task_id=item.task_id,
            original_class_id=item.original_class_id,
            last_observed_step=item.added_at_step,
            last_observed_loss=0.0,
            last_observed_uncertainty=0.5,
            last_observed_confidence=0.5,
            last_observed_target_probability=0.5,
            last_observed_correct=False,
            risk_score=risk,
            estimated_forgetting_time_steps=estimated_t,
            next_scheduled_replay_step=item.added_at_step + estimated_t,
            correct_observation_count=0,
            max_loss_so_far=0.0,
            min_target_probability_so_far=0.5,
            max_uncertainty_so_far=0.5,
        )

    def _feature_row(
        self,
        *,
        sample_id: int,
        source_task_id: int,
        anchor_task_id: int,
        loss: float,
        uncertainty: float,
        confidence: float,
        target_probability: float,
        correct: bool,
        previous: ReplayScheduleState | None,
    ) -> dict[str, Any]:
        previous_loss = previous.last_observed_loss if previous is not None else loss
        previous_uncertainty = (
            previous.last_observed_uncertainty
            if previous is not None
            else uncertainty
        )
        previous_target_probability = (
            previous.last_observed_target_probability
            if previous is not None
            else target_probability
        )
        history_eval_count = (
            previous.observation_count + 1 if previous is not None else 1
        )
        correct_observation_count = (
            previous.correct_observation_count + int(correct)
            if previous is not None
            else int(correct)
        )
        loss_delta = loss - previous_loss
        target_probability_delta = target_probability - previous_target_probability
        return {
            "sample_id": int(sample_id),
            "source_task_id": int(source_task_id),
            "anchor_trained_task_id": int(anchor_task_id),
            "anchor_loss": loss,
            "anchor_uncertainty": uncertainty,
            "anchor_confidence": confidence,
            "anchor_target_probability": target_probability,
            "history_eval_count": history_eval_count,
            "has_previous_eval": int(previous is not None),
            "previous_correct": (
                int(previous.last_observed_correct)
                if previous is not None
                else int(correct)
            ),
            "previous_loss": previous_loss,
            "previous_uncertainty": previous_uncertainty,
            "previous_target_probability": previous_target_probability,
            "loss_delta_from_previous": loss_delta,
            "uncertainty_delta_from_previous": uncertainty - previous_uncertainty,
            "target_probability_delta_from_previous": target_probability_delta,
            "loss_increase_from_previous": max(0.0, loss_delta),
            "target_probability_drop_from_previous": max(0.0, -target_probability_delta),
            "max_loss_so_far": (
                max(previous.max_loss_so_far, loss)
                if previous is not None
                else loss
            ),
            "min_target_probability_so_far": (
                min(previous.min_target_probability_so_far, target_probability)
                if previous is not None
                else target_probability
            ),
            "max_uncertainty_so_far": (
                max(previous.max_uncertainty_so_far, uncertainty)
                if previous is not None
                else uncertainty
            ),
            "correct_history_rate": (
                correct_observation_count / history_eval_count
                if history_eval_count
                else 0.0
            ),
            "tasks_since_source": int(anchor_task_id) - int(source_task_id),
            "anchor_task_progress": (
                int(anchor_task_id) / self.config.max_anchor_task_id
                if self.config.max_anchor_task_id > 0
                else 0.0
            ),
        }

    def _score_feature_row(
        self,
        *,
        feature_row: dict[str, Any],
        loss: float,
        uncertainty: float,
        target_probability: float,
        previous_loss: float | None,
    ) -> float:
        if self.config.risk_score_source == "learned":
            return _clamp(float(self._risk_scorer.score(feature_row)), 0.0, 1.0)
        return self._risk_score(
            loss=loss,
            uncertainty=uncertainty,
            target_probability=target_probability,
            previous_loss=previous_loss,
        )

    def _risk_score(
        self,
        *,
        loss: float,
        uncertainty: float,
        target_probability: float,
        previous_loss: float | None,
    ) -> float:
        loss_component = _clamp(loss / self.config.loss_scale, 0.0, 1.0)
        uncertainty_component = _clamp(uncertainty, 0.0, 1.0)
        target_probability_component = _clamp(1.0 - target_probability, 0.0, 1.0)
        if previous_loss is None:
            loss_increase_component = 0.0
        else:
            loss_increase_component = _clamp(
                (loss - previous_loss) / self.config.loss_scale,
                0.0,
                1.0,
            )
        weighted_sum = (
            self.config.loss_weight * loss_component
            + self.config.uncertainty_weight * uncertainty_component
            + self.config.target_probability_weight * target_probability_component
            + self.config.loss_increase_weight * loss_increase_component
        )
        total_weight = (
            self.config.loss_weight
            + self.config.uncertainty_weight
            + self.config.target_probability_weight
            + self.config.loss_increase_weight
        )
        return _clamp(weighted_sum / total_weight, 0.0, 1.0)

    def _estimated_time_from_risk(self, risk_score: float) -> int:
        risk = _clamp(risk_score, 0.0, 1.0)
        span = self.config.max_interval_steps - self.config.min_interval_steps
        return int(round(self.config.max_interval_steps - risk * span))
