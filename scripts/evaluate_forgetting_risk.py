"""Evaluate leakage-safe forgetting-risk heuristics from run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_predictor_report


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


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "forgetting_risk_report.json"
    return Path(args.labels).with_name("forgetting_risk_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate forgetting-risk heuristics and a lightweight predictor"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--labels", type=str, help="Path to forgetting_labels.json")
    parser.add_argument("--output", type=str, help="Output predictor report path")
    args = parser.parse_args(argv)

    try:
        report = save_predictor_report(
            signal_path=_resolve_signal_path(args),
            label_path=_resolve_label_path(args),
            output_path=_resolve_output_path(args),
        )
    except Exception as exc:
        print(f"Failed to evaluate forgetting risk: {exc}", file=sys.stderr)
        return 2

    heuristic_report = report["heuristic_report"]
    best_name = None
    best_ap = None
    for name, metrics in heuristic_report["heuristics"].items():
        ap = metrics["average_precision"]
        if ap is not None and (best_ap is None or ap > best_ap):
            best_name = name
            best_ap = ap

    print(f"Saved forgetting-risk report to: {_resolve_output_path(args)}")
    print(
        "Temporal split: "
        f"train<=task {heuristic_report['temporal_split']['train_anchor_task_max']}, "
        f"test>=task {heuristic_report['temporal_split']['test_anchor_task_min']}"
    )
    print(f"Test anchors: {heuristic_report['test']['n']}")
    print(f"Test positives: {heuristic_report['test']['positive_count']}")
    print(f"Best heuristic AP: {best_name}={best_ap}")
    print(f"Logistic status: {report['logistic_report']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
