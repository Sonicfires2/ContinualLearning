"""Run the Task 14 MIR replay comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.mir_replay_comparison import (
    load_config,
    run_mir_replay_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MIR replay comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        summary = run_mir_replay_comparison(config)
    except Exception as exc:
        print(f"Failed to run MIR replay comparison: {exc}", file=sys.stderr)
        return 2

    metrics = summary["aggregates"]["mir_replay"]
    print(f"Saved MIR replay comparison summary to: {summary['summary_path']}")
    print(
        "mir_replay: "
        f"final_accuracy_mean={metrics['final_accuracy_mean']}, "
        f"average_forgetting_mean={metrics['average_forgetting_mean']}, "
        f"replay_samples_mean={metrics['total_replay_samples_mean']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
