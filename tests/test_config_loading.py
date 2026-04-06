from pathlib import Path

from src.twr.utils.config import load_experiment


def test_load_experiment_merges_sections() -> None:
    config = load_experiment(Path("configs/experiment/twr_backbone_lra_listops.yaml"))
    assert config["model"]["name"] == "twr"
    assert config["data"]["kind"] == "lra_listops"
    assert config["train"]["epochs"] == 15
    assert config["train"]["write_penalty_weight"] == 0.001


def test_load_mamba_placeholder_experiment() -> None:
    config = load_experiment(Path("configs/experiment/mamba_placeholder_debug.yaml"))
    assert config["model"]["name"] == "mamba_placeholder"


def test_load_current_long_benchmark_experiment() -> None:
    config = load_experiment(Path("configs/experiment/twr_backbone_longbench_trec.yaml"))
    assert config["data"]["kind"] == "longbench"
    assert config["data"]["benchmark_name"] == "trec"
    assert config["train"]["epochs"] == 15
