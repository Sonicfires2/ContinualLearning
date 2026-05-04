"""Evaluate the proposal's four signal families for forgetting prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors import save_proposal_signal_diagnostic_report


def _resolve_path(
    *,
    explicit_path: str | None,
    run_dir: str | None,
    default_name: str,
    flag_name: str,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    if run_dir is None:
        raise ValueError(f"provide either --{flag_name} or --run-dir")
    return Path(run_dir) / default_name


def _resolve_diagnostic_metrics_path(args: argparse.Namespace) -> Path | None:
    if args.diagnostic_metrics is not None:
        return Path(args.diagnostic_metrics)
    if args.run_dir is not None:
        return Path(args.run_dir) / "metrics.json"
    return None


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    if args.run_dir is not None:
        return Path(args.run_dir) / "proposal_signal_diagnostic_report.json"
    return Path(args.representations).with_name("proposal_signal_diagnostic_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare loss, uncertainty, gradient norm, and representation drift"
    )
    parser.add_argument("--run-dir", type=str, help="Run artifact directory")
    parser.add_argument("--signals", type=str, help="Path to sample_signals.json")
    parser.add_argument("--labels", type=str, help="Path to forgetting_labels.json")
    parser.add_argument("--gradients", type=str, help="Path to gradient_signals.json")
    parser.add_argument(
        "--representations",
        type=str,
        help="Path to representation_signals.json",
    )
    parser.add_argument("--reference-metrics", type=str, help="Reference metrics.json path")
    parser.add_argument("--diagnostic-metrics", type=str, help="Diagnostic metrics.json path")
    parser.add_argument("--output", type=str, help="Output report path")
    args = parser.parse_args(argv)

    try:
        output_path = _resolve_output_path(args)
        report = save_proposal_signal_diagnostic_report(
            signal_path=_resolve_path(
                explicit_path=args.signals,
                run_dir=args.run_dir,
                default_name="sample_signals.json",
                flag_name="signals",
            ),
            label_path=_resolve_path(
                explicit_path=args.labels,
                run_dir=args.run_dir,
                default_name="forgetting_labels.json",
                flag_name="labels",
            ),
            gradient_path=_resolve_path(
                explicit_path=args.gradients,
                run_dir=args.run_dir,
                default_name="gradient_signals.json",
                flag_name="gradients",
            ),
            representation_path=_resolve_path(
                explicit_path=args.representations,
                run_dir=args.run_dir,
                default_name="representation_signals.json",
                flag_name="representations",
            ),
            output_path=output_path,
            reference_metrics_path=(
                Path(args.reference_metrics) if args.reference_metrics else None
            ),
            diagnostic_metrics_path=_resolve_diagnostic_metrics_path(args),
        )
    except Exception as exc:
        print(f"Failed to evaluate proposal signal diagnostics: {exc}", file=sys.stderr)
        return 2

    print(f"Saved proposal-signal diagnostic report to: {output_path}")
    for row in report["proposal_signal_comparison"]:
        print(
            f"{row['signal_family']}: "
            f"{row['best_model']} AP={row['average_precision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
