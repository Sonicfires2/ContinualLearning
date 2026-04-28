"""Generate compact figures for the Task 25 final synthesis.

The plotted values are the documented summary metrics from the completed
Split CIFAR-100 study. This script is intentionally data-light: it regenerates
report figures without requiring the large experiment artifact directories.
"""

from __future__ import annotations

import os
from pathlib import Path


MPL_CONFIG_DIR = Path(".tmp") / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


FIGURE_DIR = Path("docs") / "figures"


def _annotate_bars(ax, values, fmt="{:.3f}") -> None:
    for bar, value in zip(ax.patches, values):
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_core_methods() -> None:
    methods = [
        "Fine-tune",
        "Random",
        "Fixed k=1",
        "Spaced",
        "MIR",
    ]
    final_accuracy = [
        0.04703333333333334,
        0.10156666666666665,
        0.10156666666666665,
        0.09863333333333334,
        0.11636666666666667,
    ]
    avg_forgetting = [
        0.43103703703703705,
        0.30274074074074075,
        0.30274074074074075,
        0.3131111111111111,
        0.2167037037037037,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(methods, final_accuracy, color="#2f6f73")
    _style_axis(axes[0], "Final Accuracy, Higher Is Better", "Accuracy")
    _annotate_bars(axes[0], final_accuracy)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(methods, avg_forgetting, color="#a44a3f")
    _style_axis(axes[1], "Average Forgetting, Lower Is Better", "Forgetting")
    _annotate_bars(axes[1], avg_forgetting)
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle("Core Three-Seed Split CIFAR-100 Results", fontsize=13)
    fig.savefig(FIGURE_DIR / "task25_core_methods.png", dpi=180)
    plt.close(fig)


def plot_seed0_replay_methods() -> None:
    methods = [
        "Random",
        "Risk gate",
        "Learned gate",
        "Learned fixed",
        "Hybrid 50/50",
        "25% L + CB",
        "Class-balanced",
        "MIR",
    ]
    final_accuracy = [
        0.10129999999999999,
        0.046,
        0.0379,
        0.0759,
        0.0879,
        0.0986,
        0.10500000000000001,
        0.1183,
    ]
    avg_forgetting = [
        0.30433333333333334,
        0.43377777777777776,
        0.40311111111111114,
        0.3587777777777778,
        0.3428888888888889,
        0.3268888888888889,
        0.2962222222222222,
        0.21400000000000002,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(methods, final_accuracy, color="#4267ac")
    _style_axis(axes[0], "Seed-0 Final Accuracy", "Accuracy")
    _annotate_bars(axes[0], final_accuracy)
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(methods, avg_forgetting, color="#b26a2c")
    _style_axis(axes[1], "Seed-0 Average Forgetting", "Forgetting")
    _annotate_bars(axes[1], avg_forgetting)
    axes[1].tick_params(axis="x", rotation=35)

    fig.suptitle("Learned Replay and Rescue-Ablation Results", fontsize=13)
    fig.savefig(FIGURE_DIR / "task25_seed0_replay_methods.png", dpi=180)
    plt.close(fig)


def plot_predictor_diagnostics() -> None:
    groups = ["Cheap all", "Cheap + grad", "Gradient only", "Best heuristic"]
    average_precision = [
        0.9083240127221096,
        0.9080918805327551,
        0.8386703932509996,
        0.8471253916174554,
    ]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(groups, average_precision, color="#557a46")
    ax.set_ylim(0.78, 0.93)
    _style_axis(ax, "Offline Future-Forgetting Prediction", "Average precision")
    _annotate_bars(ax, average_precision)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIGURE_DIR / "task25_predictor_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_mir_diagnostic() -> None:
    labels = [
        "Base rate",
        "Risk AP",
        "Risk/MIR overlap",
        "Random overlap",
    ]
    values = [
        0.25,
        0.21600508010478187,
        0.1792949398443029,
        0.25,
    ]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(labels, values, color=["#7c6a46", "#9a4f64", "#9a4f64", "#7c6a46"])
    ax.set_ylim(0, 0.32)
    _style_axis(ax, "Learned Risk Does Not Align With MIR", "Score")
    _annotate_bars(ax, values)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIGURE_DIR / "task25_mir_interference_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_core_methods()
    plot_seed0_replay_methods()
    plot_predictor_diagnostics()
    plot_mir_diagnostic()
    print(f"Wrote Task 25 figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
