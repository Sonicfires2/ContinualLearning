"""Maximally Interfered Retrieval utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.replay.buffer import ReplayItem


MIR_TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MIRCandidateScore:
    """One candidate memory sample and its MIR interference score."""

    global_step: int
    optimizer_step: int
    sample_id: int
    source_task_id: int
    original_class_id: int
    replay_count_before_selection: int
    last_replay_step_before_selection: int | None
    pre_update_loss: float
    post_update_loss: float
    interference_score: float
    candidate_rank: int
    candidate_count: int


@dataclass(frozen=True)
class MIRSelection:
    """One selected memory sample and its interference score."""

    global_step: int
    optimizer_step: int
    sample_id: int
    source_task_id: int
    original_class_id: int
    replay_count_before_selection: int
    last_replay_step_before_selection: int | None
    pre_update_loss: float
    post_update_loss: float
    interference_score: float
    candidate_rank: int
    candidate_count: int
    selection_reason: str = "max_interference"


class MIRTraceLogger:
    """Append-only trace for MIR replay selections."""

    def __init__(self) -> None:
        self.rows: list[MIRSelection] = []
        self.candidate_counts: list[int] = []

    def record(self, selections: Sequence[MIRSelection], *, candidate_count: int) -> None:
        self.rows.extend(selections)
        self.candidate_counts.append(int(candidate_count))

    def summary(self) -> dict[str, object]:
        scores = [row.interference_score for row in self.rows]
        return {
            "schema_version": MIR_TRACE_SCHEMA_VERSION,
            "trace_row_count": len(self.rows),
            "selection_event_count": len(self.candidate_counts),
            "mean_candidate_count": (
                sum(self.candidate_counts) / len(self.candidate_counts)
                if self.candidate_counts
                else None
            ),
            "mean_selected_interference_score": (
                sum(scores) / len(scores) if scores else None
            ),
            "min_selected_interference_score": min(scores) if scores else None,
            "max_selected_interference_score": max(scores) if scores else None,
        }

    def to_json_payload(self) -> dict[str, object]:
        return {
            "schema_version": MIR_TRACE_SCHEMA_VERSION,
            "definition": {
                "policy": "ER-MIR: sample candidates from memory, perform a virtual incoming-batch update, and replay candidates with largest post_loss - pre_loss",
                "score": "cross_entropy_after_virtual_update - cross_entropy_before_virtual_update",
                "future_leakage_guard": "uses only current batch and replay memory available before the actual optimizer step",
            },
            "summary": self.summary(),
            "rows": [asdict(row) for row in self.rows],
        }


def _candidate_tensors(
    candidate_items: Sequence[ReplayItem],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidate_items:
        raise ValueError("candidate_items must not be empty")
    x = torch.stack([item.x for item in candidate_items]).to(device)
    y = torch.tensor([item.target for item in candidate_items], dtype=torch.long).to(device)
    return x, y


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _restore_parameters(
    parameters: Sequence[nn.Parameter],
    backup: Sequence[torch.Tensor],
) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, backup, strict=True):
            parameter.copy_(value)


def score_mir_replay_candidates(
    *,
    model: nn.Module,
    current_x: torch.Tensor,
    current_y: torch.Tensor,
    candidate_items: Sequence[ReplayItem],
    virtual_lr: float,
    global_step: int,
    device: torch.device,
) -> list[MIRCandidateScore]:
    """Score all candidate items using ER-MIR's MI-1 virtual-update score."""

    if virtual_lr <= 0:
        raise ValueError("virtual_lr must be positive")
    if not candidate_items:
        return []

    candidate_x, candidate_y = _candidate_tensors(candidate_items, device=device)
    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("model has no trainable parameters")

    was_training = model.training
    model.train()
    with torch.no_grad():
        pre_losses = F.cross_entropy(model(candidate_x), candidate_y, reduction="none")

    virtual_loss = F.cross_entropy(model(current_x), current_y)
    gradients = torch.autograd.grad(
        virtual_loss,
        parameters,
        allow_unused=True,
    )
    backup = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                parameter.add_(gradient, alpha=-virtual_lr)

    try:
        with torch.no_grad():
            post_losses = F.cross_entropy(model(candidate_x), candidate_y, reduction="none")
    finally:
        _restore_parameters(parameters, backup)
        model.train(was_training)

    scores = post_losses - pre_losses
    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    candidate_scores: list[MIRCandidateScore] = []
    for rank, candidate_index in enumerate(order, start=1):
        item = candidate_items[candidate_index]
        pre_loss = float(pre_losses[candidate_index].detach().cpu().item())
        post_loss = float(post_losses[candidate_index].detach().cpu().item())
        score = float(scores[candidate_index].detach().cpu().item())
        candidate_scores.append(
            MIRCandidateScore(
                global_step=int(global_step),
                optimizer_step=int(global_step + 1),
                sample_id=item.sample_id,
                source_task_id=item.task_id,
                original_class_id=item.original_class_id,
                replay_count_before_selection=item.replay_count,
                last_replay_step_before_selection=item.last_replayed_step,
                pre_update_loss=pre_loss,
                post_update_loss=post_loss,
                interference_score=score,
                candidate_rank=rank,
                candidate_count=len(candidate_items),
            )
        )
    return candidate_scores


def select_mir_replay_items(
    *,
    model: nn.Module,
    current_x: torch.Tensor,
    current_y: torch.Tensor,
    candidate_items: Sequence[ReplayItem],
    replay_batch_size: int,
    virtual_lr: float,
    global_step: int,
    device: torch.device,
) -> list[MIRSelection]:
    """Select the most interfered replay items using ER-MIR's MI-1 score."""

    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")

    candidate_scores = score_mir_replay_candidates(
        model=model,
        current_x=current_x,
        current_y=current_y,
        candidate_items=candidate_items,
        virtual_lr=virtual_lr,
        global_step=global_step,
        device=device,
    )
    selected_count = min(replay_batch_size, len(candidate_scores))
    selections: list[MIRSelection] = []
    for candidate_score in candidate_scores[:selected_count]:
        selections.append(
            MIRSelection(
                global_step=candidate_score.global_step,
                optimizer_step=candidate_score.optimizer_step,
                sample_id=candidate_score.sample_id,
                source_task_id=candidate_score.source_task_id,
                original_class_id=candidate_score.original_class_id,
                replay_count_before_selection=candidate_score.replay_count_before_selection,
                last_replay_step_before_selection=candidate_score.last_replay_step_before_selection,
                pre_update_loss=candidate_score.pre_update_loss,
                post_update_loss=candidate_score.post_update_loss,
                interference_score=candidate_score.interference_score,
                candidate_rank=candidate_score.candidate_rank,
                candidate_count=candidate_score.candidate_count,
            )
        )
    return selections
