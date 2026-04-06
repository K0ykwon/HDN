from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twr.training.trainer import train
from src.twr.utils.config import load_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TWR-LM or a baseline from an experiment config.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to an experiment YAML, for example configs/experiment/twr_backbone_lra_listops.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment(args.experiment)
    artifacts = train(config)
    summary = artifacts.summary
    print(f"Run directory: {artifacts.run_dir}")
    print(
        f"Best val loss={summary['final_val_loss']:.4f}, "
        f"val acc={summary['final_val_accuracy']:.4f}, "
        f"avg depth={summary['avg_effective_depth']:.3f}"
    )


if __name__ == "__main__":
    main()
