"""Aggregate and plot the three-seed NLP continual-learning results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
from typing import Any


MPL_CONFIG_DIR = Path(".tmp") / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "fine_tuning",
    "random_replay",
    "spaced_replay",
    "risk_ranked_replay",
    "representation_drift_replay",
    "mir_replay",
]

METHOD_LABELS = {
    "fine_tuning": "Fine-tune",
    "random_replay": "Random",
    "spaced_replay": "Spaced",
    "risk_ranked_replay": "Risk-ranked",
    "representation_drift_replay": "Rep drift",
    "mir_replay": "MIR-style",
}

DEFAULT_EXTRA_ROOTS = [
    Path("experiments/runs/nlp_representation_drift_online"),
]


@dataclass(frozen=True)
class MetricRow:
    method: str
    seed: int
    final_accuracy: float
    average_forgetting: float
    replay_samples: int
    training_time_seconds: float
    metrics_path: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _collect_metric_rows(roots: list[Path]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/*/metrics.json")):
            metrics = _read_json(metrics_path)
            if metrics.get("method") not in METHOD_ORDER:
                continue
            rows.append(
                MetricRow(
                    method=str(metrics["method"]),
                    seed=int(metrics["seed"]),
                    final_accuracy=float(metrics["final_accuracy"]),
                    average_forgetting=float(metrics["average_forgetting"]),
                    replay_samples=int(metrics["replay_samples"]),
                    training_time_seconds=float(metrics["training_time_seconds"]),
                    metrics_path=str(metrics_path),
                )
            )
    return rows


def _aggregate(rows: list[MetricRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    summary_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = sorted(grouped.get(method, []), key=lambda row: row.seed)
        if not method_rows:
            continue
        final_accuracy = [row.final_accuracy for row in method_rows]
        average_forgetting = [row.average_forgetting for row in method_rows]
        replay_samples = [row.replay_samples for row in method_rows]
        training_time = [row.training_time_seconds for row in method_rows]
        summary_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "seeds": [row.seed for row in method_rows],
                "n": len(method_rows),
                "final_accuracy_mean": statistics.mean(final_accuracy),
                "final_accuracy_std": _std(final_accuracy),
                "average_forgetting_mean": statistics.mean(average_forgetting),
                "average_forgetting_std": _std(average_forgetting),
                "replay_samples_mean": statistics.mean(replay_samples),
                "training_time_seconds_mean": statistics.mean(training_time),
            }
        )
    return summary_rows


def _markdown(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Seeds | Final accuracy mean | Final accuracy std | Avg forgetting mean | Avg forgetting std | Replay samples mean | Time mean, sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"`{','.join(str(seed) for seed in row['seeds'])}` | "
            f"`{row['final_accuracy_mean']}` | "
            f"`{row['final_accuracy_std']}` | "
            f"`{row['average_forgetting_mean']}` | "
            f"`{row['average_forgetting_std']}` | "
            f"`{row['replay_samples_mean']}` | "
            f"`{row['training_time_seconds_mean']:.2f}` |"
        )
    return "\n".join(lines)


def _annotate_bars(ax, values: list[float]) -> None:
    for bar, value in zip(ax.patches, values):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 4),
            textcoords="offset points",
        )


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["label"] for row in summary_rows]
    final_means = [row["final_accuracy_mean"] for row in summary_rows]
    forgetting_means = [row["average_forgetting_mean"] for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    axes[0].bar(
        labels,
        final_means,
        color="#2f6f73",
    )
    axes[0].set_ylim(0, 1.05)
    _style_axis(axes[0], "Final Accuracy, Higher Is Better", "Accuracy")
    _annotate_bars(axes[0], final_means)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        labels,
        forgetting_means,
        color="#a44a3f",
    )
    axes[1].set_ylim(0, max(0.05, max(forgetting_means) * 1.18))
    _style_axis(axes[1], "Average Forgetting, Lower Is Better", "Forgetting")
    _annotate_bars(axes[1], forgetting_means)
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Three-Seed Sampled Split DBpedia14 NLP Results", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(root: Path, output_path: Path, extra_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    roots = [root, *(extra_roots or [])]
    rows = _collect_metric_rows(roots)
    summary_rows = _aggregate(rows)
    if not summary_rows:
        raise FileNotFoundError(
            "No NLP metrics found under "
            + ", ".join(str(candidate) for candidate in roots)
        )
    _write_json(
        root / "three_seed_summary.json",
        {
            "schema_version": 1,
            "rows": summary_rows,
            "source_roots": [str(candidate) for candidate in roots],
            "source_metrics": [row.__dict__ for row in rows],
        },
    )
    (root / "three_seed_summary.md").write_text(
        _markdown(summary_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _plot(summary_rows, output_path)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/runs/nlp_continual_pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/nlp_three_seed_main_results.png"),
    )
    parser.add_argument(
        "--extra-root",
        action="append",
        type=Path,
        default=None,
        help="Additional root containing method/seed metrics.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = run(
        args.root,
        args.output,
        extra_roots=args.extra_root if args.extra_root is not None else DEFAULT_EXTRA_ROOTS,
    )
    print(_markdown(summary_rows))
    print(f"\nWrote figure to {args.output}")


if __name__ == "__main__":
    main()
