"""Run the Task 13 controlled core comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.core_comparison import load_config, run_core_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run core continual-learning comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        summary = run_core_comparison(config)
    except Exception as exc:
        print(f"Failed to run core comparison: {exc}", file=sys.stderr)
        return 2

    print(f"Saved core comparison summary to: {summary['summary_path']}")
    for method_name, metrics in summary["aggregates"].items():
        print(
            f"{method_name}: "
            f"final_accuracy_mean={metrics['final_accuracy_mean']}, "
            f"average_forgetting_mean={metrics['average_forgetting_mean']}, "
            f"replay_samples_mean={metrics['total_replay_samples_mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
