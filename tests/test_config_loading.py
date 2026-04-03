from pathlib import Path

from src.twr.utils.config import load_experiment


def test_load_experiment_merges_sections() -> None:
    config = load_experiment(Path("configs/experiment/twr_debug.yaml"))
    assert config["model"]["name"] == "twr"
    assert config["data"]["task"] == "special_token_parity"
    assert config["train"]["epochs"] == 2
    assert config["train"]["depth_penalty_weight"] == 0.0001


def test_load_mamba_placeholder_experiment() -> None:
    config = load_experiment(Path("configs/experiment/mamba_placeholder_debug.yaml"))
    assert config["model"]["name"] == "mamba_placeholder"


def test_load_count_compare_experiment() -> None:
    config = load_experiment(Path("configs/experiment/twr_count_compare_adaptive.yaml"))
    assert config["data"]["task"] == "count_compare"
    assert config["train"]["epochs"] == 6
