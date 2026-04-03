from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one or more experiment configs sequentially.")
    parser.add_argument("experiments", nargs="+", help="Experiment YAML paths.")
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_script = root / "scripts" / "train.py"
    for experiment in parse_args().experiments:
        subprocess.run([sys.executable, str(train_script), "--experiment", experiment], check=True)


if __name__ == "__main__":
    main()
