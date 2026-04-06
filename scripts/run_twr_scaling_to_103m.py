from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "configs/experiment/twr_scale25m_lra_listops.yaml",
    "configs/experiment/twr_scale57m_lra_listops.yaml",
    "configs/experiment/twr_scale84m_lra_listops.yaml",
    "configs/experiment/twr_scale103m_lra_listops.yaml",
    "configs/experiment/twr_scale25m_ruler_needle.yaml",
    "configs/experiment/twr_scale57m_ruler_needle.yaml",
    "configs/experiment/twr_scale84m_ruler_needle.yaml",
    "configs/experiment/twr_scale103m_ruler_needle.yaml",
]


def run_experiment(path: str) -> None:
    command = [sys.executable, "scripts/train.py", "--experiment", path]
    print(f"==> running {path}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    for experiment in EXPERIMENTS:
        run_experiment(experiment)


if __name__ == "__main__":
    main()
