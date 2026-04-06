from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_results import main as analyze_results_main
from src.twr.training.trainer import train
from src.twr.utils.config import load_experiment


EXPERIMENTS = [
    "configs/experiment/twr_backbone_lra_listops.yaml",
    "configs/experiment/transformer_lra_listops.yaml",
    "configs/experiment/mamba_lra_listops.yaml",
    "configs/experiment/twr_backbone_ruler_needle.yaml",
    "configs/experiment/transformer_ruler_needle.yaml",
    "configs/experiment/mamba_ruler_needle.yaml",
    "configs/experiment/twr_backbone_longbench_trec.yaml",
    "configs/experiment/transformer_longbench_trec.yaml",
    "configs/experiment/mamba_longbench_trec.yaml",
    "configs/experiment/twr_backbone_hyperpartisan.yaml",
    "configs/experiment/transformer_hyperpartisan.yaml",
    "configs/experiment/mamba_hyperpartisan.yaml",
]


def main() -> None:
    for experiment_path in EXPERIMENTS:
        print(f"==> running {experiment_path}")
        config = load_experiment(ROOT / experiment_path)
        artifacts = train(config)
        summary = artifacts.summary
        print(
            f"completed {summary['run_name']} "
            f"val_acc={summary['final_val_accuracy']:.4f} "
            f"val_loss={summary['final_val_loss']:.4f} "
            f"params={summary['parameter_count']}"
        )
    analyze_results_main()


if __name__ == "__main__":
    main()
