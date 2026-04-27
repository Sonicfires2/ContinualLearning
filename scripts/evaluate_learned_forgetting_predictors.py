"""Evaluate learned forgetting predictors from run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_learned_predictor_report


def _resolve_signal_path(args: argparse.Namespace) -> Path:
    if args.signals is not None:
        return Path(args.signals)
    if args.run_dir is None:
        raise ValueError("provide either --signals or --run-dir")
    return Path(args.run_dir) / "sample_signals.json"


def _resolve_label_path(args: argparse.Namespace) -> Path:
    if args.labels is not None:
        return Path(args.labels)
    if args.run_dir is None:
        raise ValueError("provide either --labels or --run-dir")
    return Path(args.run_dir) / "forgetting_labels.json"


def _resolve_time_path(args: argparse.Namespace) -> Path | None:
    if args.time_targets is not None:
        return Path(args.time_targets)
    if args.run_dir is None:
        return None
    candidate = Path(args.run_dir) / "time_to_forgetting_targets.json"
    return candidate if candidate.exists() else None


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "learned_forgetting_predictor_report.json"
    return Path(args.labels).with_name("learned_forgetting_predictor_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate logistic, linear, and SVM forgetting predictors"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--labels", type=str, help="Path to forgetting_labels.json")
    parser.add_argument(
        "--time-targets",
        type=str,
        help="Optional path to time_to_forgetting_targets.json",
    )
    parser.add_argument("--output", type=str, help="Output report path")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args)
        report = save_learned_predictor_report(
            signal_path=_resolve_signal_path(args),
            label_path=_resolve_label_path(args),
            time_path=_resolve_time_path(args),
            output_path=output_path,
        )
    except Exception as exc:
        print(f"Failed to evaluate learned forgetting predictors: {exc}", file=sys.stderr)
        return 2

    summary = report["comparison_summary"]
    best_heuristic = summary["best_heuristic_by_average_precision"]
    best_model = summary["best_binary_model_by_average_precision"]
    split = report["binary_classification_report"]["temporal_split"]

    print(f"Saved learned-predictor report to: {output_path}")
    print(
        "Temporal split: "
        f"train<=task {split['train_anchor_task_max']}, "
        f"test>=task {split['test_anchor_task_min']}"
    )
    print(
        "Best heuristic AP: "
        f"{best_heuristic['heuristic']}={best_heuristic['average_precision']}"
    )
    print(
        "Best learned binary model AP: "
        f"{best_model['model']}={best_model['average_precision']}"
    )
    print(
        "Best learned model beats best heuristic: "
        f"{summary['best_binary_model_beats_best_heuristic']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
