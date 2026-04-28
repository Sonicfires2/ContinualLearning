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


@dataclass(frozen=True)
class MethodResult:
    method: str
    run_dir: str
    final_accuracy: float
    average_forgetting: float
    replay_samples: int
    training_time_seconds: float
    accuracy_matrix: AccuracyMatrix


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
) -> tuple[float, list[dict[str, Any]]]:
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
        outputs = model(**device_batch)
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
        correct_count += int(correct.sum().item())
        total_count += int(labels.numel())
        for index, sample_id in enumerate(batch["sample_ids"]):
            signal_rows.append(
                {
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
            )
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
    seed = int(trainer_config["seed"])
    _seed_everything(seed)
    rng = random.Random(seed + hash(method) % 100_000)
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
    }
    run_dir = output_root / method / f"{method}_seed{seed}_dbpedia14_speed_pilot"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "accuracy_matrix.json", {"accuracy_matrix": accuracy_matrix})
    _write_json(run_dir / "train_losses.json", {"train_losses": train_losses})
    _write_json(run_dir / "eval_signals.json", {"rows": eval_signal_rows})
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
    return parser.parse_args()


def main() -> None:
    _set_repo_local_caches()
    args = parse_args()
    config = _read_config(args.config)
    if args.train_examples_per_class is not None:
        config["dataset"]["train_examples_per_class"] = args.train_examples_per_class
    if args.eval_examples_per_class is not None:
        config["dataset"]["eval_examples_per_class"] = args.eval_examples_per_class
    results = run_pilot(config, methods=args.methods)
    print("\nSummary:")
    print(_summary_markdown(results))


if __name__ == "__main__":
    main()
