"""Evaluate leakage-safe time-to-forgetting heuristics from run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_time_to_forgetting_report


def _resolve_signal_path(args: argparse.Namespace) -> Path:
    if args.signals is not None:
        return Path(args.signals)
    if args.run_dir is None:
        raise ValueError("provide either --signals or --run-dir")
    return Path(args.run_dir) / "sample_signals.json"


def _resolve_time_path(args: argparse.Namespace) -> Path:
    if args.targets is not None:
        return Path(args.targets)
    if args.run_dir is None:
        raise ValueError("provide either --targets or --run-dir")
    return Path(args.run_dir) / "time_to_forgetting_targets.json"


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "time_to_forgetting_report.json"
    return Path(args.targets).with_name("time_to_forgetting_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate time-to-forgetting heuristics and due-time proxies"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--targets", type=str, help="Path to time target JSON")
    parser.add_argument("--output", type=str, help="Output report path")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args)
        report = save_time_to_forgetting_report(
            signal_path=_resolve_signal_path(args),
            time_path=_resolve_time_path(args),
            output_path=output_path,
        )
    except Exception as exc:
        print(f"Failed to evaluate time-to-forgetting: {exc}", file=sys.stderr)
        return 2

    evaluation = report["evaluation"]
    best_name = None
    best_mae = None
    for name, metrics in evaluation["estimators"].items():
        mae = metrics["mae_step_delta_on_observed_events"]
        if mae is not None and (best_mae is None or mae < best_mae):
            best_name = name
            best_mae = mae

    print(f"Saved time-to-forgetting report to: {output_path}")
    print(
        "Temporal split: "
        f"train<=task {evaluation['temporal_split']['train_anchor_task_max']}, "
        f"test>=task {evaluation['temporal_split']['test_anchor_task_min']}"
    )
    print(f"Test anchors: {evaluation['test']['n']}")
    print(f"Test observed events: {evaluation['test']['event_observed_count']}")
    print(f"Best step-delta MAE: {best_name}={best_mae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
