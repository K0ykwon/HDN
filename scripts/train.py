from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.engine import run_experiment
from src.training.engine import run_study
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an HDN experiment from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument(
        "--mode",
        choices=["train", "study"],
        default="train",
        help="Run a single experiment or an ablation study bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "study":
        artifacts = run_study(config)
        print(f"Study directory: {artifacts['run_dir']}")
        print(f"Ablation runs: {len(artifacts['study_results']['ablations'])}")
        return

    artifacts = run_experiment(config)
    final_metrics = artifacts.metrics["final_metrics"]

    print(f"Run directory: {artifacts.run_dir}")
    print(
        "Final metrics: "
        f"train_loss={final_metrics['train_loss']:.4f}, "
        f"train_accuracy={final_metrics['train_accuracy']:.4f}, "
        f"valid_loss={final_metrics['valid_loss']:.4f}, "
        f"valid_accuracy={final_metrics['valid_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
