"""Plot signal-family forgetting-prediction comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CONFIG_DIR = PROJECT_ROOT / ".tmp" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "proposal_signal_diagnostic"
    / "proposal_signal_diagnostic"
    / "proposal_signal_diagnostic_split_cifar100_seed0_random_replay"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "figures" / "cifar100_signal_family_comparison.png"


DISPLAY_NAMES = {
    "loss_trajectory": "Loss trajectory",
    "uncertainty": "Uncertainty",
    "gradient_norm": "Gradient norm",
    "representation_drift": "Representation drift",
}


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _annotate_bars(ax, values) -> None:
    for bar, value in zip(ax.patches, values):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_proposal_signal_comparison(*, report_path: Path, output_path: Path) -> None:
    report = _load_report(report_path)
    rows = report["proposal_signal_comparison"]
    labels = [DISPLAY_NAMES[row["signal_family"]] for row in rows]
    values = [float(row["average_precision"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = ["#3d6f77", "#7a8f3a", "#9a5d42", "#6f5f9a"]
    ax.bar(labels, values, color=colors[: len(labels)])
    lower = max(0.0, min(values) - 0.06)
    upper = min(1.0, max(values) + 0.03)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Average precision")
    ax.set_title("Signal Families Predicting Future Forgetting", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    _annotate_bars(ax, values)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot signal-family comparison")
    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_RUN_DIR / "proposal_signal_diagnostic_report.json"),
        help="Path to proposal_signal_diagnostic_report.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output PNG path",
    )
    args = parser.parse_args(argv)
    plot_proposal_signal_comparison(
        report_path=Path(args.report),
        output_path=Path(args.output),
    )
    print(f"Wrote signal-family comparison figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
