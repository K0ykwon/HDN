from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twr.models.factory import build_model
from src.twr.training.trainer import train
from src.twr.utils.config import load_experiment, load_yaml
from src.twr.utils.profiling import count_parameters


SCALE_DIMS = [192, 320, 576, 960]
VARIANT_LABEL = "gmc8c2"
TRAIN_CONFIG_PATH = ROOT / "configs" / "train" / "benchmark_100ep.yaml"

BENCHMARKS = [
    ("lra_listops", "configs/experiment/twr_backbone_lra_listops.yaml"),
    ("ruler_needle", "configs/experiment/twr_backbone_ruler_needle.yaml"),
    ("longbench_trec", "configs/experiment/twr_backbone_longbench_trec.yaml"),
    ("hyperpartisan", "configs/experiment/twr_backbone_hyperpartisan.yaml"),
]


def build_scaled_config(template_path: str, dim: int) -> dict:
    config = load_experiment(template_path)
    config["train"] = load_yaml(TRAIN_CONFIG_PATH)
    config["model"]["embed_dim"] = dim
    config["model"]["slot_dim"] = dim
    config["model"]["mlp_hidden_dim"] = dim * 2
    config["model"]["use_global_memory"] = True
    config["model"]["memory_slots"] = 8
    config["model"]["carrier_slots"] = 2
    return config


def main() -> None:
    for benchmark_name, template_path in BENCHMARKS:
        for dim in SCALE_DIMS:
            config = build_scaled_config(template_path, dim)
            run_name = f"twr_{VARIANT_LABEL}_d{dim}_{benchmark_name}_100ep"
            config["run"] = {"name": run_name}
            run_dir = Path("experiments/runs") / run_name
            if run_dir.exists():
                shutil.rmtree(run_dir)
                print(f"removed stale run_dir {run_dir}")

            model = build_model(config["model"], config["data"])
            params = count_parameters(model)
            print(
                f"==> running {benchmark_name} dim={dim} "
                f"run={run_name} params={params}"
            )
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
