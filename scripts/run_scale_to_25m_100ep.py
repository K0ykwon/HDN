from __future__ import annotations

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twr.training.trainer import train
from src.twr.utils.config import load_experiment
from src.twr.utils.profiling import count_parameters
from src.twr.models.factory import build_model


EXPERIMENTS = [
    "configs/experiment/twr_scale2p8m_lra_listops_100ep.yaml",
    "configs/experiment/twr_scale6p3m_lra_listops_100ep.yaml",
    "configs/experiment/twr_scale11p2m_lra_listops_100ep.yaml",
    "configs/experiment/twr_scale17p4m_lra_listops_100ep.yaml",
    "configs/experiment/twr_scale25m_lra_listops_100ep.yaml",
    "configs/experiment/twr_scale2p8m_ruler_needle_100ep.yaml",
    "configs/experiment/twr_scale6p3m_ruler_needle_100ep.yaml",
    "configs/experiment/twr_scale11p2m_ruler_needle_100ep.yaml",
    "configs/experiment/twr_scale17p4m_ruler_needle_100ep.yaml",
    "configs/experiment/twr_scale25m_ruler_needle_100ep.yaml",
]


def main() -> None:
    for experiment_path in EXPERIMENTS:
        config = load_experiment(experiment_path)
        run_name = config["run"].get("name")
        if run_name:
            run_dir = Path("experiments/runs") / run_name
            if run_dir.exists():
                shutil.rmtree(run_dir)
                print(f"removed stale run_dir {run_dir}")
        model = build_model(config["model"], config["data"])
        params = count_parameters(model)
        print(f"==> running {experiment_path} params={params}")
        artifacts = train(config)
        summary = artifacts.summary
        print(
            f"completed {summary['run_name']} "
            f"val_acc={summary['final_val_accuracy']:.4f} "
            f"val_loss={summary['final_val_loss']:.4f} "
            f"params={summary['parameter_count']}"
        )


if __name__ == "__main__":
    main()
