"""Entry point for experiments and quick runs."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Continual Learning project skeleton")
    if args.config:
        cfg_path = Path(args.config)
        print(f"Would load config from: {cfg_path}")


if __name__ == "__main__":
    main()
