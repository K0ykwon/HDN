from src.utils.config import load_config


def test_load_config_reads_json(tmp_path):
    config_path = tmp_path / "example.json"
    config_path.write_text('{"experiment_name": "smoke"}', encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment_name"] == "smoke"
