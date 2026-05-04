"""Run a speed-first NLP continual-learning pilot with DistilBERT.

The script is intentionally compact and artifact-oriented. It builds a sampled
Split DBpedia14 task stream, runs a small set of replay methods, and writes the
same core evidence the image study reports: accuracy matrix, final accuracy,
average forgetting, replay count, and per-method runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset, load_dataset
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.metrics.continual import (
    AccuracyMatrix,
    average_forgetting,
    final_accuracy,
    forgetting_by_task,
)


@dataclass(frozen=True)
class EncodedTextExample:
    sample_id: int
    split: str
    task_id: int
    original_class_id: int
    label: int
    input_ids: list[int]
    attention_mask: list[int]


@dataclass
class MemoryState:
    risk_score: float
    next_due_step: int
    last_loss: float | None = None
    replay_count: int = 0
    last_replay_step: int | None = None
    representation_reference: torch.Tensor | None = None
    representation_reference_l2: float | None = None
    representation_reference_step: int | None = None
    representation_reference_task_id: int | None = None
    representation_drift_score: float = 0.0
    representation_risk_score: float = 0.0
    representation_next_due_step: int | None = None
    representation_due_step_before_update: int | None = None
    representation_last_scored_step: int | None = None
    representation_score_count: int = 0


@dataclass
class RepresentationReference:
    vector: torch.Tensor
    trained_task_id: int
    global_step: int
    l2: float


@dataclass(frozen=True)
class MethodResult:
    method: str
    run_dir: str
    final_accuracy: float
    average_forgetting: float
    replay_samples: int
    training_time_seconds: float
    accuracy_matrix: AccuracyMatrix


DRIFT_REPLAY_METHODS = {
    "representation_drift_replay": "drift_due_time",
    "drift_ranked_replay": "drift_ranked",
    "drift_due_time_replay": "drift_due_time",
    "drift_hybrid_replay": "drift_hybrid",
}


def _set_repo_local_caches() -> None:
    os.environ.setdefault("HF_HOME", str(Path(".tmp") / "hf-home"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mpl_config = Path(".tmp") / "matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_id_for_label(label: int, label_to_task: dict[int, int]) -> int:
    return label_to_task[int(label)]


def _build_label_to_task(class_order: list[int], classes_per_task: int) -> dict[int, int]:
    return {
        int(label): index // classes_per_task
        for index, label in enumerate(class_order)
    }


def _text_from_row(row: dict[str, Any], text_fields: list[str]) -> str:
    parts = [str(row[field]) for field in text_fields if field in row and row[field]]
    if parts:
        return " ".join(parts)
    for fallback in ("text", "sentence", "content", "title"):
        if fallback in row and row[fallback]:
            return str(row[fallback])
    raise ValueError("Could not find a usable text field in dataset row")


def _sample_indices_by_class(
    dataset: Dataset,
    *,
    class_order: list[int],
    label_column: str,
    examples_per_class: int,
    seed: int,
) -> list[int]:
    labels = np.asarray(dataset[label_column])
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in class_order:
        indices = np.flatnonzero(labels == int(label))
        if len(indices) < examples_per_class:
            raise ValueError(
                f"Class {label} only has {len(indices)} rows, "
                f"needed {examples_per_class}"
            )
        rng.shuffle(indices)
        selected.extend(int(index) for index in indices[:examples_per_class])
    rng.shuffle(selected)
    return selected


def _encode_examples(
    *,
    dataset: Dataset,
    indices: list[int],
    split: str,
    tokenizer: Any,
    text_fields: list[str],
    label_column: str,
    label_to_task: dict[int, int],
    max_length: int,
    sample_id_offset: int,
    batch_size: int = 512,
) -> list[EncodedTextExample]:
    examples: list[EncodedTextExample] = []
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        rows = [dataset[int(index)] for index in batch_indices]
        texts = [_text_from_row(row, text_fields) for row in rows]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        for row, original_index, input_ids, attention_mask in zip(
            rows,
            batch_indices,
            encoded["input_ids"],
            encoded["attention_mask"],
        ):
            label = int(row[label_column])
            examples.append(
                EncodedTextExample(
                    sample_id=sample_id_offset + int(original_index),
                    split=split,
                    task_id=_task_id_for_label(label, label_to_task),
                    original_class_id=label,
                    label=label,
                    input_ids=list(input_ids),
                    attention_mask=list(attention_mask),
                )
            )
    return examples


def _group_by_task(examples: list[EncodedTextExample], task_count: int) -> list[list[EncodedTextExample]]:
    grouped = [[] for _ in range(task_count)]
    for example in examples:
        grouped[example.task_id].append(example)
    return grouped


def _collate_text_examples(batch: list[EncodedTextExample]) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([example.input_ids for example in batch], dtype=torch.long),
        "attention_mask": torch.tensor(
            [example.attention_mask for example in batch],
            dtype=torch.long,
        ),
        "labels": torch.tensor([example.label for example in batch], dtype=torch.long),
        "sample_ids": [example.sample_id for example in batch],
        "task_ids": [example.task_id for example in batch],
        "class_ids": [example.original_class_id for example in batch],
        "examples": batch,
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def _cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first_norm = torch.linalg.vector_norm(first, ord=2)
    second_norm = torch.linalg.vector_norm(second, ord=2)
    first_norm_value = float(first_norm.item())
    second_norm_value = float(second_norm.item())
    if first_norm_value == 0.0 and second_norm_value == 0.0:
        return 1.0
    if first_norm_value == 0.0 or second_norm_value == 0.0:
        return 0.0
    similarity = torch.dot(first, second) / (first_norm * second_norm)
    return max(-1.0, min(1.0, float(similarity.item())))


def _risk_from_observation(
    *,
    loss: float,
    target_probability: float,
    confidence: float,
    previous_loss: float | None,
    loss_scale: float,
) -> float:
    loss_component = min(max(loss / loss_scale, 0.0), 1.0)
    uncertainty = 1.0 - confidence
    low_target_probability = 1.0 - target_probability
    loss_increase = 0.0
    if previous_loss is not None:
        loss_increase = min(max((loss - previous_loss) / loss_scale, 0.0), 1.0)
    risk = (
        loss_component
        + uncertainty
        + low_target_probability
        + loss_increase
    ) / 4.0
    return min(max(float(risk), 0.0), 1.0)


def _estimated_time_from_risk(
    *,
    risk_score: float,
    min_interval_steps: int,
    max_interval_steps: int,
) -> int:
    span = max_interval_steps - min_interval_steps
    return int(round(max_interval_steps - risk_score * span))


def _stable_method_offset(method: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(method)) % 100_000


def _select_replay_examples(
    *,
    method: str,
    memory: list[EncodedTextExample],
    memory_state: dict[int, MemoryState],
    replay_batch_size: int,
    global_step: int,
    rng: random.Random,
) -> list[EncodedTextExample]:
    if method == "fine_tuning" or not memory:
        return []
    if method == "random_replay":
        count = min(replay_batch_size, len(memory))
        return rng.sample(memory, count)
    if method in {"risk_ranked_replay", "learned_risk_replay"}:
        records = [(example, memory_state[example.sample_id]) for example in memory]
        records.sort(
            key=lambda record: (
                -record[1].risk_score,
                record[1].replay_count,
                record[1].next_due_step,
                record[0].sample_id,
            )
        )
        return [record[0] for record in records[: min(replay_batch_size, len(records))]]
    if method == "spaced_replay":
        records = []
        for example in memory:
            state = memory_state[example.sample_id]
            overdue = global_step - state.next_due_step
            records.append((example, state, overdue))
        records.sort(
            key=lambda record: (
                int(record[2] < 0),
                -record[2],
                record[1].next_due_step,
                -record[1].risk_score,
                record[0].sample_id,
            )
        )
        return [record[0] for record in records[: min(replay_batch_size, len(records))]]
    raise ValueError(f"Unsupported method: {method}")


def _clamp_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _drift_variant_for_method(method: str, config: dict[str, Any]) -> str:
    if method == "representation_drift_replay":
        drift_config = config.get("representation_drift_replay", {})
        return str(drift_config.get("drift_variant", "drift_due_time"))
    return DRIFT_REPLAY_METHODS[method]


@torch.no_grad()
def _representations_for_text_examples(
    *,
    model: nn.Module,
    examples: list[EncodedTextExample],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[int, tuple[torch.Tensor, float]]:
    was_training = model.training
    model.eval()
    representations: dict[int, tuple[torch.Tensor, float]] = {}
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = _collate_text_examples(batch_examples)
        device_batch = _move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**device_batch, output_hidden_states=True)
        reps = outputs.hidden_states[-1][:, 0, :].detach().float().cpu()
        rep_l2 = torch.linalg.vector_norm(reps, ord=2, dim=1)
        for example, vector, l2_value in zip(batch_examples, reps, rep_l2):
            representations[example.sample_id] = (
                vector.clone(),
                float(l2_value.item()),
            )
    model.train(was_training)
    return representations


def _ensure_representation_memory_references(
    *,
    model: nn.Module,
    memory: list[EncodedTextExample],
    memory_state: dict[int, MemoryState],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    global_step: int,
    task_id: int,
) -> None:
    missing = [
        example
        for example in memory
        if memory_state[example.sample_id].representation_reference is None
    ]
    if not missing:
        return
    representations = _representations_for_text_examples(
        model=model,
        examples=missing,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    for example in missing:
        vector, l2_value = representations[example.sample_id]
        state = memory_state[example.sample_id]
        state.representation_reference = vector
        state.representation_reference_l2 = l2_value
        state.representation_reference_step = int(global_step)
        state.representation_reference_task_id = int(task_id)
        if state.representation_next_due_step is None:
            state.representation_next_due_step = int(state.next_due_step)
        if state.representation_due_step_before_update is None:
            state.representation_due_step_before_update = int(state.next_due_step)


def _score_representation_drift_candidates(
    *,
    model: nn.Module,
    candidates: list[EncodedTextExample],
    memory_state: dict[int, MemoryState],
    global_step: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    drift_scale: float,
    min_interval_steps: int,
    max_interval_steps: int,
) -> list[tuple[EncodedTextExample, MemoryState]]:
    if drift_scale <= 0:
        raise ValueError("representation_drift_replay.drift_scale must be positive")
    representations = _representations_for_text_examples(
        model=model,
        examples=candidates,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    scored: list[tuple[EncodedTextExample, MemoryState]] = []
    for example in candidates:
        state = memory_state[example.sample_id]
        if state.representation_reference is None:
            raise ValueError("missing representation reference for memory example")
        current_vector, _current_l2 = representations[example.sample_id]
        drift = 1.0 - _cosine_similarity(current_vector, state.representation_reference)
        risk = _clamp_probability(drift / drift_scale)
        interval = _estimated_time_from_risk(
            risk_score=risk,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
        )
        previous_due_step = (
            state.representation_next_due_step
            if state.representation_next_due_step is not None
            else state.next_due_step
        )
        state.representation_drift_score = float(drift)
        state.representation_risk_score = float(risk)
        state.representation_due_step_before_update = int(previous_due_step)
        state.representation_next_due_step = int(global_step + interval)
        state.representation_last_scored_step = int(global_step)
        state.representation_score_count += 1
        scored.append((example, state))
    return scored


def _select_class_balanced_text_examples(
    *,
    memory: list[EncodedTextExample],
    excluded_sample_ids: set[int],
    quota: int,
    rng: random.Random,
) -> list[EncodedTextExample]:
    candidates = [
        example for example in memory if example.sample_id not in excluded_sample_ids
    ]
    if quota <= 0 or not candidates:
        return []
    by_class: dict[int, list[EncodedTextExample]] = {}
    for example in candidates:
        by_class.setdefault(example.original_class_id, []).append(example)
    for examples in by_class.values():
        rng.shuffle(examples)
    class_ids = list(by_class)
    rng.shuffle(class_ids)
    selected: list[EncodedTextExample] = []
    offsets = {class_id: 0 for class_id in class_ids}
    while len(selected) < quota and class_ids:
        for class_id in list(class_ids):
            offset = offsets[class_id]
            examples = by_class[class_id]
            if offset >= len(examples):
                class_ids.remove(class_id)
                continue
            selected.append(examples[offset])
            offsets[class_id] = offset + 1
            if len(selected) >= quota:
                break
    return selected


def _select_random_text_examples(
    *,
    memory: list[EncodedTextExample],
    excluded_sample_ids: set[int],
    quota: int,
    rng: random.Random,
) -> list[EncodedTextExample]:
    candidates = [
        example for example in memory if example.sample_id not in excluded_sample_ids
    ]
    if quota <= 0 or not candidates:
        return []
    return rng.sample(candidates, min(quota, len(candidates)))


def _representation_drift_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    drift_values = []
    risk_values = []
    for row in rows:
        reason = str(row["selection_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        drift_values.append(float(row["drift_score"]))
        risk_values.append(float(row["risk_score"]))
    return {
        "trace_row_count": len(rows),
        "selection_reason_counts": reason_counts,
        "mean_selected_drift_score": (
            sum(drift_values) / len(drift_values) if drift_values else None
        ),
        "mean_selected_risk_score": (
            sum(risk_values) / len(risk_values) if risk_values else None
        ),
        "max_selected_drift_score": max(drift_values) if drift_values else None,
    }


def _select_representation_drift_replay_examples(
    *,
    method: str,
    config: dict[str, Any],
    model: nn.Module,
    memory: list[EncodedTextExample],
    memory_state: dict[int, MemoryState],
    replay_batch_size: int,
    global_step: int,
    current_task_id: int,
    device: torch.device,
    use_amp: bool,
    rng: random.Random,
) -> tuple[list[EncodedTextExample], list[dict[str, Any]]]:
    if not memory or replay_batch_size <= 0:
        return [], []

    drift_config = config.get("representation_drift_replay", {})
    variant = _drift_variant_for_method(method, config)
    candidate_count = int(drift_config.get("candidate_count", 64))
    scoring_batch_size = int(drift_config.get("scoring_batch_size", 64))
    drift_scale = float(drift_config.get("drift_scale", 1.0))
    min_interval_steps = int(
        drift_config.get(
            "min_interval_steps",
            config["spaced_replay"]["min_interval_steps"],
        )
    )
    max_interval_steps = int(
        drift_config.get(
            "max_interval_steps",
            config["spaced_replay"]["max_interval_steps"],
        )
    )
    hybrid_fraction = float(drift_config.get("hybrid_fraction", 0.5))
    hybrid_diversity_mode = str(drift_config.get("hybrid_diversity_mode", "class_balanced"))

    _ensure_representation_memory_references(
        model=model,
        memory=memory,
        memory_state=memory_state,
        batch_size=scoring_batch_size,
        device=device,
        use_amp=use_amp,
        global_step=global_step,
        task_id=current_task_id,
    )
    candidate_total = min(max(candidate_count, replay_batch_size), len(memory))
    candidates = (
        list(memory)
        if candidate_total == len(memory)
        else rng.sample(memory, candidate_total)
    )
    scored = _score_representation_drift_candidates(
        model=model,
        candidates=candidates,
        memory_state=memory_state,
        global_step=global_step,
        batch_size=scoring_batch_size,
        device=device,
        use_amp=use_amp,
        drift_scale=drift_scale,
        min_interval_steps=min_interval_steps,
        max_interval_steps=max_interval_steps,
    )
    scored_ids = {example.sample_id for example, _state in scored}
    target_count = min(replay_batch_size, len(memory))

    if variant == "drift_ranked":
        selected_records = sorted(
            scored,
            key=lambda record: (
                -record[1].representation_risk_score,
                -record[1].representation_drift_score,
                record[1].replay_count,
                record[0].sample_id,
            ),
        )[:target_count]
        selected = [example for example, _state in selected_records]
        selection_reasons = {example.sample_id: "drift_ranked" for example in selected}
    elif variant == "drift_due_time":
        selected_records = sorted(
            scored,
            key=lambda record: (
                int(
                    global_step
                    < (
                        record[1].representation_due_step_before_update
                        if record[1].representation_due_step_before_update is not None
                        else record[1].next_due_step
                    )
                ),
                -(
                    global_step
                    - (
                        record[1].representation_due_step_before_update
                        if record[1].representation_due_step_before_update is not None
                        else record[1].next_due_step
                    )
                ),
                -record[1].representation_risk_score,
                (
                    record[1].representation_due_step_before_update
                    if record[1].representation_due_step_before_update is not None
                    else record[1].next_due_step
                ),
                record[0].sample_id,
            ),
        )[:target_count]
        selected = [example for example, _state in selected_records]
        selection_reasons = {}
        for example, state in selected_records:
            due_step = (
                state.representation_due_step_before_update
                if state.representation_due_step_before_update is not None
                else state.next_due_step
            )
            selection_reasons[example.sample_id] = (
                "due" if global_step >= due_step else "budget_fill_near_due"
            )
    elif variant == "drift_hybrid":
        if not 0.0 <= hybrid_fraction <= 1.0:
            raise ValueError("representation_drift_replay.hybrid_fraction must be in [0, 1]")
        drift_quota = min(target_count, int(round(target_count * hybrid_fraction)))
        diversity_quota = target_count - drift_quota
        selected_records = sorted(
            scored,
            key=lambda record: (
                -record[1].representation_risk_score,
                -record[1].representation_drift_score,
                record[1].replay_count,
                record[0].sample_id,
            ),
        )[:drift_quota]
        selected = [example for example, _state in selected_records]
        selected_ids = {example.sample_id for example in selected}
        if hybrid_diversity_mode == "class_balanced":
            diversity_examples = _select_class_balanced_text_examples(
                memory=memory,
                excluded_sample_ids=selected_ids,
                quota=diversity_quota,
                rng=rng,
            )
        elif hybrid_diversity_mode == "random":
            diversity_examples = _select_random_text_examples(
                memory=memory,
                excluded_sample_ids=selected_ids,
                quota=diversity_quota,
                rng=rng,
            )
        else:
            raise ValueError(
                "representation_drift_replay.hybrid_diversity_mode must be "
                "class_balanced or random"
            )
        selected.extend(diversity_examples)
        selected_ids.update(example.sample_id for example in diversity_examples)
        if len(selected) < target_count:
            selected.extend(
                _select_random_text_examples(
                    memory=memory,
                    excluded_sample_ids=selected_ids,
                    quota=target_count - len(selected),
                    rng=rng,
                )
            )
        selection_reasons = {
            example.sample_id: "drift_ranked"
            for example, _state in selected_records
        }
        selection_reasons.update(
            {
                example.sample_id: f"diversity_{hybrid_diversity_mode}"
                for example in diversity_examples
            }
        )
    else:
        raise ValueError(
            "representation drift variant must be drift_ranked, drift_due_time, "
            "or drift_hybrid"
        )

    unscored_selected = [
        example for example in selected if example.sample_id not in scored_ids
    ]
    if unscored_selected:
        _score_representation_drift_candidates(
            model=model,
            candidates=unscored_selected,
            memory_state=memory_state,
            global_step=global_step,
            batch_size=scoring_batch_size,
            device=device,
            use_amp=use_amp,
            drift_scale=drift_scale,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
        )

    trace_rows = []
    for example in selected:
        state = memory_state[example.sample_id]
        due_step = (
            state.representation_due_step_before_update
            if state.representation_due_step_before_update is not None
            else state.next_due_step
        )
        trace_rows.append(
            {
                "global_step": int(global_step),
                "method": method,
                "variant": variant,
                "sample_id": int(example.sample_id),
                "task_id": int(example.task_id),
                "original_class_id": int(example.original_class_id),
                "drift_score": float(state.representation_drift_score),
                "risk_score": float(state.representation_risk_score),
                "next_due_step": int(
                    state.representation_next_due_step
                    if state.representation_next_due_step is not None
                    else state.next_due_step
                ),
                "due_step_before_update": int(due_step),
                "overdue_steps": int(max(0, global_step - due_step)),
                "selection_reason": selection_reasons.get(
                    example.sample_id,
                    "fallback_random",
                ),
            }
        )
    return selected, trace_rows


@torch.no_grad()
def _candidate_losses(
    *,
    model: nn.Module,
    examples: list[EncodedTextExample],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> list[float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = _collate_text_examples(batch_examples)
        device_batch = _move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**device_batch)
            batch_losses = F.cross_entropy(
                outputs.logits,
                device_batch["labels"],
                reduction="none",
            )
        losses.extend(float(value) for value in batch_losses.detach().float().cpu().tolist())
    model.train(was_training)
    return losses


def _select_mir_replay_examples(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    current_batch: dict[str, Any],
    memory: list[EncodedTextExample],
    replay_batch_size: int,
    candidate_count: int,
    scoring_batch_size: int,
    virtual_learning_rate: float,
    device: torch.device,
    use_amp: bool,
    rng: random.Random,
) -> list[EncodedTextExample]:
    if not memory or replay_batch_size <= 0:
        return []

    candidate_total = min(max(candidate_count, replay_batch_size), len(memory))
    if candidate_total == len(memory):
        candidates = list(memory)
    else:
        candidates = rng.sample(memory, candidate_total)

    before_losses = _candidate_losses(
        model=model,
        examples=candidates,
        batch_size=scoring_batch_size,
        device=device,
        use_amp=use_amp,
    )

    was_training = model.training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_current_batch = _move_batch_to_device(current_batch, device)
    with torch.amp.autocast("cuda", enabled=use_amp):
        outputs = model(**device_current_batch)
        current_loss = F.cross_entropy(outputs.logits, device_current_batch["labels"])
    current_loss.backward()

    saved_parameters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            saved_parameters.append((parameter, parameter.detach().clone()))
            parameter.add_(parameter.grad, alpha=-virtual_learning_rate)
    optimizer.zero_grad(set_to_none=True)

    after_losses = _candidate_losses(
        model=model,
        examples=candidates,
        batch_size=scoring_batch_size,
        device=device,
        use_amp=use_amp,
    )

    with torch.no_grad():
        for parameter, saved_value in saved_parameters:
            parameter.copy_(saved_value)
    optimizer.zero_grad(set_to_none=True)
    model.train(was_training)

    scored_examples = [
        (after_loss - before_loss, after_loss, example)
        for example, before_loss, after_loss in zip(candidates, before_losses, after_losses)
    ]
    scored_examples.sort(
        key=lambda record: (-record[0], -record[1], record[2].sample_id)
    )
    return [
        record[2]
        for record in scored_examples[: min(replay_batch_size, len(scored_examples))]
    ]


def _add_to_memory(
    *,
    memory: list[EncodedTextExample],
    memory_state: dict[int, MemoryState],
    seen_count: int,
    new_examples: list[EncodedTextExample],
    memory_capacity: int,
    global_step: int,
    min_interval_steps: int,
    max_interval_steps: int,
    rng: random.Random,
) -> int:
    for example in new_examples:
        seen_count += 1
        risk = 0.5
        estimated_t = _estimated_time_from_risk(
            risk_score=risk,
            min_interval_steps=min_interval_steps,
            max_interval_steps=max_interval_steps,
        )
        state = MemoryState(
            risk_score=risk,
            next_due_step=global_step + estimated_t,
        )
        if len(memory) < memory_capacity:
            memory.append(example)
            memory_state[example.sample_id] = state
            continue
        replace_index = rng.randrange(seen_count)
        if replace_index < memory_capacity:
            removed = memory[replace_index]
            memory_state.pop(removed.sample_id, None)
            memory[replace_index] = example
            memory_state[example.sample_id] = state
    return seen_count


@torch.no_grad()
def _evaluate_task(
    *,
    model: nn.Module,
    examples: list[EncodedTextExample],
    batch_size: int,
    device: torch.device,
    trained_task_id: int,
    eval_task_id: int,
    global_step: int,
    num_workers: int,
    log_representation_drift: bool = False,
    representation_references: dict[int, RepresentationReference] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if log_representation_drift and representation_references is None:
        raise ValueError("representation_references is required when logging drift")
    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_text_examples,
    )
    model.eval()
    correct_count = 0
    total_count = 0
    signal_rows: list[dict[str, Any]] = []
    for batch in loader:
        device_batch = _move_batch_to_device(batch, device)
        outputs = model(
            **device_batch,
            output_hidden_states=log_representation_drift,
        )
        logits = outputs.logits
        labels = device_batch["labels"]
        losses = F.cross_entropy(logits, labels, reduction="none")
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1).values
        target_probabilities = probabilities.gather(
            1,
            labels.unsqueeze(1),
        ).squeeze(1)
        correct = predictions.eq(labels)
        representations = None
        representation_l2 = None
        if log_representation_drift:
            representations = outputs.hidden_states[-1][:, 0, :].detach().float().cpu()
            representation_l2 = torch.linalg.vector_norm(representations, ord=2, dim=1)
        correct_count += int(correct.sum().item())
        total_count += int(labels.numel())
        for index, sample_id in enumerate(batch["sample_ids"]):
            row = {
                "global_step": int(global_step),
                "trained_task_id": int(trained_task_id),
                "eval_task_id": int(eval_task_id),
                "sample_id": int(sample_id),
                "task_id": int(batch["task_ids"][index]),
                "original_class_id": int(batch["class_ids"][index]),
                "label": int(labels[index].item()),
                "loss": float(losses[index].item()),
                "predicted_label": int(predictions[index].item()),
                "confidence": float(confidence[index].item()),
                "target_probability": float(target_probabilities[index].item()),
                "correct": bool(correct[index].item()),
            }
            if log_representation_drift:
                assert representations is not None
                assert representation_l2 is not None
                assert representation_references is not None
                sample_id_int = int(sample_id)
                current_vector = representations[index].clone()
                current_l2 = float(representation_l2[index].item())
                if sample_id_int not in representation_references:
                    representation_references[sample_id_int] = RepresentationReference(
                        vector=current_vector,
                        trained_task_id=int(trained_task_id),
                        global_step=int(global_step),
                        l2=current_l2,
                    )
                reference = representation_references[sample_id_int]
                similarity = _cosine_similarity(current_vector, reference.vector)
                row.update(
                    {
                        "representation_scope": "distilbert_last_hidden_cls",
                        "representation_l2": current_l2,
                        "reference_representation_l2": reference.l2,
                        "reference_trained_task_id": reference.trained_task_id,
                        "reference_global_step": reference.global_step,
                        "cosine_similarity_to_reference": similarity,
                        "representation_drift": 1.0 - similarity,
                    }
                )
            signal_rows.append(row)
    if total_count == 0:
        raise ValueError("Cannot evaluate an empty task")
    return correct_count / total_count, signal_rows


def _build_task_streams(config: dict[str, Any]) -> tuple[
    list[list[EncodedTextExample]],
    list[list[EncodedTextExample]],
    dict[str, Any],
]:
    dataset_config = config["dataset"]
    model_config = config["model"]
    seed = int(config["trainer"]["seed"])
    class_order = [int(label) for label in dataset_config["class_order"]]
    classes_per_task = int(dataset_config["classes_per_task"])
    task_count = math.ceil(len(class_order) / classes_per_task)
    label_to_task = _build_label_to_task(class_order, classes_per_task)

    raw = load_dataset(dataset_config["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

    train_split = raw[dataset_config["train_split"]]
    eval_split = raw[dataset_config["eval_split"]]
    train_indices = _sample_indices_by_class(
        train_split,
        class_order=class_order,
        label_column=dataset_config["label_column"],
        examples_per_class=int(dataset_config["train_examples_per_class"]),
        seed=seed,
    )
    eval_indices = _sample_indices_by_class(
        eval_split,
        class_order=class_order,
        label_column=dataset_config["label_column"],
        examples_per_class=int(dataset_config["eval_examples_per_class"]),
        seed=seed + 10_000,
    )

    train_examples = _encode_examples(
        dataset=train_split,
        indices=train_indices,
        split="train",
        tokenizer=tokenizer,
        text_fields=list(dataset_config["text_fields"]),
        label_column=dataset_config["label_column"],
        label_to_task=label_to_task,
        max_length=int(model_config["max_length"]),
        sample_id_offset=0,
    )
    eval_examples = _encode_examples(
        dataset=eval_split,
        indices=eval_indices,
        split="eval",
        tokenizer=tokenizer,
        text_fields=list(dataset_config["text_fields"]),
        label_column=dataset_config["label_column"],
        label_to_task=label_to_task,
        max_length=int(model_config["max_length"]),
        sample_id_offset=100_000_000,
    )
    label_names = None
    try:
        label_names = list(train_split.features[dataset_config["label_column"]].names)
    except AttributeError:
        label_names = [str(label) for label in class_order]

    metadata = {
        "task_count": task_count,
        "class_order": class_order,
        "classes_per_task": classes_per_task,
        "label_names": label_names,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
    }
    return (
        _group_by_task(train_examples, task_count),
        _group_by_task(eval_examples, task_count),
        metadata,
    )


def _run_method(
    *,
    method: str,
    config: dict[str, Any],
    train_tasks: list[list[EncodedTextExample]],
    eval_tasks: list[list[EncodedTextExample]],
    task_metadata: dict[str, Any],
    device: torch.device,
    output_root: Path,
) -> MethodResult:
    trainer_config = config["trainer"]
    spaced_config = config["spaced_replay"]
    mir_config = config.get("mir_replay", {})
    seed = int(trainer_config["seed"])
    _seed_everything(seed)
    rng = random.Random(seed + _stable_method_offset(method))
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(config["dataset"]["class_order"]),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]),
    )
    use_amp = bool(trainer_config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    memory: list[EncodedTextExample] = []
    memory_state: dict[int, MemoryState] = {}
    memory_seen_count = 0
    replay_samples = 0
    global_step = 0
    train_losses: list[float] = []
    eval_signal_rows: list[dict[str, Any]] = []
    scheduler_trace_rows: list[dict[str, Any]] = []
    log_representation_drift = bool(trainer_config.get("log_representation_drift", False))
    representation_references: dict[int, RepresentationReference] = {}
    accuracy_matrix: AccuracyMatrix = [
        [None for _ in range(len(train_tasks))]
        for _ in range(len(train_tasks))
    ]
    start_time = time.perf_counter()

    for task_id, task_examples in enumerate(train_tasks):
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed + task_id)
        loader = DataLoader(
            task_examples,
            batch_size=int(trainer_config["batch_size"]),
            shuffle=True,
            num_workers=int(trainer_config.get("num_workers", 0)),
            collate_fn=_collate_text_examples,
            generator=loader_generator,
        )
        for _epoch in range(int(trainer_config["epochs_per_task"])):
            model.train()
            for batch in loader:
                if method == "mir_replay":
                    replay_examples = _select_mir_replay_examples(
                        model=model,
                        optimizer=optimizer,
                        current_batch=batch,
                        memory=memory,
                        replay_batch_size=int(trainer_config["replay_batch_size"]),
                        candidate_count=int(
                            mir_config.get(
                                "candidate_count",
                                int(trainer_config["replay_batch_size"]) * 4,
                            )
                        ),
                        scoring_batch_size=int(
                            mir_config.get(
                                "scoring_batch_size",
                                int(trainer_config["batch_size"]),
                            )
                        ),
                        virtual_learning_rate=float(trainer_config["learning_rate"])
                        * float(mir_config.get("virtual_step_scale", 1.0)),
                        device=device,
                        use_amp=use_amp,
                        rng=rng,
                    )
                elif method in DRIFT_REPLAY_METHODS:
                    replay_examples, drift_trace_rows = (
                        _select_representation_drift_replay_examples(
                            method=method,
                            config=config,
                            model=model,
                            memory=memory,
                            memory_state=memory_state,
                            replay_batch_size=int(trainer_config["replay_batch_size"]),
                            global_step=global_step,
                            current_task_id=task_id,
                            device=device,
                            use_amp=use_amp,
                            rng=rng,
                        )
                    )
                    scheduler_trace_rows.extend(drift_trace_rows)
                else:
                    replay_examples = _select_replay_examples(
                        method=method,
                        memory=memory,
                        memory_state=memory_state,
                        replay_batch_size=int(trainer_config["replay_batch_size"]),
                        global_step=global_step,
                        rng=rng,
                    )
                replay_batch = (
                    _collate_text_examples(replay_examples)
                    if replay_examples
                    else None
                )
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                replay_start_index = input_ids.shape[0]
                if replay_batch is not None:
                    input_ids = torch.cat([input_ids, replay_batch["input_ids"]], dim=0)
                    attention_mask = torch.cat(
                        [attention_mask, replay_batch["attention_mask"]],
                        dim=0,
                    )
                    labels = torch.cat([labels, replay_batch["labels"]], dim=0)
                    replay_samples += len(replay_examples)
                device_batch = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "labels": labels.to(device),
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(**device_batch)
                    logits = outputs.logits
                    per_sample_losses = F.cross_entropy(
                        logits,
                        device_batch["labels"],
                        reduction="none",
                    )
                    loss = per_sample_losses.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu().item()))

                if replay_examples:
                    with torch.no_grad():
                        probabilities = torch.softmax(logits.detach(), dim=-1)
                        confidence = torch.max(probabilities, dim=-1).values
                        target_probabilities = probabilities.gather(
                            1,
                            device_batch["labels"].unsqueeze(1),
                        ).squeeze(1)
                    for offset, example in enumerate(replay_examples):
                        index = replay_start_index + offset
                        state = memory_state[example.sample_id]
                        loss_value = float(per_sample_losses[index].detach().cpu().item())
                        risk = _risk_from_observation(
                            loss=loss_value,
                            target_probability=float(
                                target_probabilities[index].detach().cpu().item()
                            ),
                            confidence=float(confidence[index].detach().cpu().item()),
                            previous_loss=state.last_loss,
                            loss_scale=float(spaced_config["loss_scale"]),
                        )
                        estimated_t = _estimated_time_from_risk(
                            risk_score=risk,
                            min_interval_steps=int(spaced_config["min_interval_steps"]),
                            max_interval_steps=int(spaced_config["max_interval_steps"]),
                        )
                        state.risk_score = risk
                        state.next_due_step = global_step + estimated_t
                        state.last_loss = loss_value
                        state.replay_count += 1
                        state.last_replay_step = global_step
                global_step += 1

        memory_seen_count = _add_to_memory(
            memory=memory,
            memory_state=memory_state,
            seen_count=memory_seen_count,
            new_examples=task_examples,
            memory_capacity=int(trainer_config["memory_capacity"]),
            global_step=global_step,
            min_interval_steps=int(spaced_config["min_interval_steps"]),
            max_interval_steps=int(spaced_config["max_interval_steps"]),
            rng=rng,
        )
        if method in DRIFT_REPLAY_METHODS:
            drift_config = config.get("representation_drift_replay", {})
            _ensure_representation_memory_references(
                model=model,
                memory=memory,
                memory_state=memory_state,
                batch_size=int(drift_config.get("scoring_batch_size", 64)),
                device=device,
                use_amp=use_amp,
                global_step=global_step,
                task_id=task_id,
            )

        for eval_task_id in range(task_id + 1):
            accuracy, signal_rows = _evaluate_task(
                model=model,
                examples=eval_tasks[eval_task_id],
                batch_size=int(trainer_config["batch_size"]),
                device=device,
                trained_task_id=task_id,
                eval_task_id=eval_task_id,
                global_step=global_step,
                num_workers=int(trainer_config.get("num_workers", 0)),
                log_representation_drift=log_representation_drift,
                representation_references=representation_references,
            )
            accuracy_matrix[task_id][eval_task_id] = accuracy
            eval_signal_rows.extend(signal_rows)

    training_time = time.perf_counter() - start_time
    metrics = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "final_accuracy": final_accuracy(accuracy_matrix),
        "average_forgetting": average_forgetting(accuracy_matrix),
        "forgetting_by_task": {
            str(key): value
            for key, value in forgetting_by_task(accuracy_matrix).items()
        },
        "replay_samples": replay_samples,
        "training_time_seconds": training_time,
        "train_loss_count": len(train_losses),
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "memory_size": len(memory),
        "global_steps": global_step,
        "task_metadata": task_metadata,
        "mir_config": mir_config if method == "mir_replay" else None,
        "representation_drift": (
            {
                "enabled": True,
                "scope": "distilbert_last_hidden_cls",
                "reference_rule": "first observed evaluation representation for each sample",
                "reference_count": len(representation_references),
            }
            if log_representation_drift
            else {"enabled": False}
        ),
        "representation_drift_scheduler": (
            {
                "enabled": True,
                "variant": _drift_variant_for_method(method, config),
                "candidate_count": int(
                    config.get("representation_drift_replay", {}).get(
                        "candidate_count",
                        64,
                    )
                ),
                **_representation_drift_selection_summary(scheduler_trace_rows),
            }
            if method in DRIFT_REPLAY_METHODS
            else {"enabled": False}
        ),
    }
    run_dir = output_root / method / f"{method}_seed{seed}_dbpedia14_speed_pilot"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "accuracy_matrix.json", {"accuracy_matrix": accuracy_matrix})
    _write_json(run_dir / "train_losses.json", {"train_losses": train_losses})
    _write_json(run_dir / "eval_signals.json", {"rows": eval_signal_rows})
    if method in DRIFT_REPLAY_METHODS:
        _write_json(
            run_dir / "scheduler_trace.json",
            {
                "schema_version": 1,
                "method": method,
                "variant": _drift_variant_for_method(method, config),
                "rows": scheduler_trace_rows,
            },
        )
    _write_json(
        run_dir / "environment.json",
        {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device": str(device),
        },
    )
    return MethodResult(
        method=method,
        run_dir=str(run_dir),
        final_accuracy=float(metrics["final_accuracy"]),
        average_forgetting=float(metrics["average_forgetting"]),
        replay_samples=int(replay_samples),
        training_time_seconds=float(training_time),
        accuracy_matrix=accuracy_matrix,
    )


def _summary_markdown(results: list[MethodResult]) -> str:
    lines = [
        "| Method | Final accuracy | Avg forgetting | Replay samples | Time, sec |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.method} | "
            f"`{result.final_accuracy}` | "
            f"`{result.average_forgetting}` | "
            f"`{result.replay_samples}` | "
            f"`{result.training_time_seconds:.2f}` |"
        )
    return "\n".join(lines)


def run_pilot(config: dict[str, Any], *, methods: list[str] | None = None) -> list[MethodResult]:
    if methods:
        config = json.loads(json.dumps(config))
        config["trainer"]["methods"] = methods
    seed = int(config["trainer"]["seed"])
    _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use .venv-nlp with CUDA PyTorch.")
    device = torch.device("cuda")
    train_tasks, eval_tasks, task_metadata = _build_task_streams(config)
    output_root = Path(config["output_root"])
    results = []
    for method in config["trainer"]["methods"]:
        print(f"\n=== Running NLP method: {method} ===", flush=True)
        result = _run_method(
            method=method,
            config=config,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            task_metadata=task_metadata,
            device=device,
            output_root=output_root,
        )
        print(
            f"{method}: final_accuracy={result.final_accuracy:.4f}, "
            f"average_forgetting={result.average_forgetting:.4f}, "
            f"replay_samples={result.replay_samples}",
            flush=True,
        )
        results.append(result)
    summary = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "seed": seed,
        "task_metadata": task_metadata,
        "results": [asdict(result) for result in results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        _summary_markdown(results) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/nlp_dbpedia14_speed_pilot.yaml"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method override, e.g. fine_tuning random_replay.",
    )
    parser.add_argument("--train-examples-per-class", type=int, default=None)
    parser.add_argument("--eval-examples-per-class", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional trainer seed override.",
    )
    return parser.parse_args()


def main() -> None:
    _set_repo_local_caches()
    args = parse_args()
    config = _read_config(args.config)
    if args.seed is not None:
        config["trainer"]["seed"] = args.seed
    if args.train_examples_per_class is not None:
        config["dataset"]["train_examples_per_class"] = args.train_examples_per_class
    if args.eval_examples_per_class is not None:
        config["dataset"]["eval_examples_per_class"] = args.eval_examples_per_class
    results = run_pilot(config, methods=args.methods)
    print("\nSummary:")
    print(_summary_markdown(results))


if __name__ == "__main__":
    main()
