"""Entry point for experiments and quick runs."""
import argparse
from pathlib import Path

try:
    from src.research_protocol import load_research_protocol, summarize_research_protocol
except ModuleNotFoundError:  # pragma: no cover - supports `python src/main.py`
    from research_protocol import load_research_protocol, summarize_research_protocol


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Continual Learning project skeleton")
    if args.config:
        cfg_path = Path(args.config)
        protocol = load_research_protocol(cfg_path)
        print(f"Loaded config from: {cfg_path}")
        print(summarize_research_protocol(protocol))


if __name__ == "__main__":
    main()
