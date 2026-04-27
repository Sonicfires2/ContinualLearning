"""Inspect raw CIFAR-100 files and save dataset preview artifacts.

This script reads the CIFAR-100 Python pickle files directly, not through the
training transforms. It is intended for research audits: confirm class counts,
task split order, sample-ID conventions, and visual sanity before interpreting
continual-learning results.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from random import Random
import sys
from typing import Any
import warnings

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_cifar100 import DEFAULT_SAMPLE_ID_OFFSETS, build_task_specs


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = pickle.load(handle, encoding="latin1")
    return {str(key): value for key, value in raw.items()}


def _raw_dir(data_root: Path) -> Path:
    return data_root / "cifar-100-python"


def _image_from_flat_row(row) -> Image.Image:
    channels = row.reshape(3, 32, 32)
    image_array = channels.transpose(1, 2, 0)
    return Image.fromarray(image_array, mode="RGB")


def _counts(values: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        result[key] = result.get(key, 0) + 1
    return result


def _split_summary(split: dict[str, Any], *, split_name: str, sample_id_offset: int) -> dict[str, Any]:
    fine_labels = [int(value) for value in split["fine_labels"]]
    coarse_labels = [int(value) for value in split["coarse_labels"]]
    return {
        "split": split_name,
        "example_count": len(fine_labels),
        "image_shape": [32, 32, 3],
        "flat_data_shape": list(split["data"].shape),
        "sample_id_offset": sample_id_offset,
        "sample_id_min": sample_id_offset,
        "sample_id_max": sample_id_offset + len(fine_labels) - 1,
        "fine_class_counts": _counts(fine_labels),
        "coarse_class_counts": _counts(coarse_labels),
    }


def _task_rows(*, fine_label_names: list[str], class_order: tuple[int, ...], task_count: int, classes_per_task: int) -> list[dict[str, Any]]:
    specs = build_task_specs(
        task_count=task_count,
        classes_per_task=classes_per_task,
        class_order=class_order,
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for within_task_label, class_id in enumerate(spec.class_ids):
            rows.append(
                {
                    "task_id": spec.task_id,
                    "within_task_label": within_task_label,
                    "class_id": class_id,
                    "fine_label_name": fine_label_names[class_id],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_preview_grid(
    *,
    train_split: dict[str, Any],
    fine_label_names: list[str],
    output_path: Path,
    class_order: tuple[int, ...],
    classes_to_show: int,
    samples_per_class: int,
) -> None:
    fine_labels = [int(value) for value in train_split["fine_labels"]]
    class_to_indices: dict[int, list[int]] = {}
    for index, class_id in enumerate(fine_labels):
        class_to_indices.setdefault(class_id, []).append(index)

    shown_classes = class_order[:classes_to_show]
    cell_size = 48
    label_width = 122
    header_height = 18
    width = label_width + samples_per_class * cell_size
    height = max(1, len(shown_classes)) * cell_size + header_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), "class", fill=(0, 0, 0))
    for sample_index in range(samples_per_class):
        draw.text((label_width + sample_index * cell_size + 4, 2), f"s{sample_index}", fill=(0, 0, 0))

    for row, class_id in enumerate(shown_classes):
        top = header_height + row * cell_size
        label = f"{class_id}: {fine_label_names[class_id]}"
        draw.text((4, top + 14), label[:18], fill=(0, 0, 0))
        for sample_index, original_index in enumerate(class_to_indices[class_id][:samples_per_class]):
            image = _image_from_flat_row(train_split["data"][original_index]).resize((32, 32))
            left = label_width + sample_index * cell_size + 8
            canvas.paste(image, (left, top + 8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def inspect_cifar100(
    *,
    data_root: Path,
    output_dir: Path,
    task_count: int,
    classes_per_task: int,
    split_seed: int,
    classes_to_show: int,
    samples_per_class: int,
) -> dict[str, Any]:
    raw_dir = _raw_dir(data_root)
    required_files = [raw_dir / "meta", raw_dir / "train", raw_dir / "test"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CIFAR-100 raw files are missing: "
            + ", ".join(missing)
            + ". Run the dataset download step before inspection."
        )

    meta = _load_pickle(raw_dir / "meta")
    train = _load_pickle(raw_dir / "train")
    test = _load_pickle(raw_dir / "test")
    fine_names = [str(name) for name in meta["fine_label_names"]]
    coarse_names = [str(name) for name in meta["coarse_label_names"]]

    class_order = tuple(range(task_count * classes_per_task))
    shuffled = list(class_order)
    Random(split_seed).shuffle(shuffled)
    class_order = tuple(shuffled)

    summary = {
        "dataset": "CIFAR-100",
        "data_root": str(data_root),
        "raw_dir": str(raw_dir),
        "fine_class_count": len(fine_names),
        "coarse_class_count": len(coarse_names),
        "fine_label_names": fine_names,
        "coarse_label_names": coarse_names,
        "split_seed": split_seed,
        "task_count": task_count,
        "classes_per_task": classes_per_task,
        "class_order": list(class_order),
        "sample_id_offsets": DEFAULT_SAMPLE_ID_OFFSETS,
        "splits": {
            "train": _split_summary(
                train,
                split_name="train",
                sample_id_offset=DEFAULT_SAMPLE_ID_OFFSETS["train"],
            ),
            "test": _split_summary(
                test,
                split_name="test",
                sample_id_offset=DEFAULT_SAMPLE_ID_OFFSETS["test"],
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "task_class_order.csv",
        _task_rows(
            fine_label_names=fine_names,
            class_order=class_order,
            task_count=task_count,
            classes_per_task=classes_per_task,
        ),
    )
    _save_preview_grid(
        train_split=train,
        fine_label_names=fine_names,
        output_path=output_dir / "train_preview_grid.png",
        class_order=class_order,
        classes_to_show=classes_to_show,
        samples_per_class=samples_per_class,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw CIFAR-100 dataset files")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/dataset_preview/cifar100"))
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--classes-to-show", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = inspect_cifar100(
        data_root=args.data_root,
        output_dir=args.output_dir,
        task_count=args.task_count,
        classes_per_task=args.classes_per_task,
        split_seed=args.split_seed,
        classes_to_show=args.classes_to_show,
        samples_per_class=args.samples_per_class,
    )
    print(f"Saved CIFAR-100 inspection artifacts to: {args.output_dir}")
    print(f"Train examples: {summary['splits']['train']['example_count']}")
    print(f"Test examples: {summary['splits']['test']['example_count']}")
    print(f"Fine classes: {summary['fine_class_count']}")
    print(f"Coarse classes: {summary['coarse_class_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
