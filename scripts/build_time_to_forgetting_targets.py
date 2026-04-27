"""Build time-to-forgetting target artifacts from signal logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.signals import save_time_to_forgetting_artifact


def _resolve_signal_path(args: argparse.Namespace) -> Path:
    if args.signals is not None:
        return Path(args.signals)
    if args.run_dir is None:
        raise ValueError("provide either --signals or --run-dir")
    return Path(args.run_dir) / "sample_signals.json"


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "time_to_forgetting_targets.json"
    signal_path = Path(args.signals)
    return signal_path.with_name("time_to_forgetting_targets.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive time-to-forgetting targets from sample_signals.json"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--output", type=str, help="Output time target JSON path")
    args = parser.parse_args(argv)

    try:
        signal_path = _resolve_signal_path(args)
        output_path = _resolve_output_path(args)
        artifact = save_time_to_forgetting_artifact(
            signal_path=signal_path,
            output_path=output_path,
        )
    except Exception as exc:
        print(f"Failed to build time-to-forgetting targets: {exc}", file=sys.stderr)
        return 2

    summary = artifact["summary"]
    print(f"Saved time-to-forgetting targets to: {output_path}")
    print(f"Anchor rows: {summary['anchor_count']}")
    print(
        "Eligible anchors: "
        f"{summary['eligible_for_time_to_forgetting_count']}"
    )
    print(f"Observed forgetting events: {summary['event_observed_count']}")
    print(f"Right-censored anchors: {summary['right_censored_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
