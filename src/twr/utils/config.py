from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {resolved}.")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment(path: str | Path) -> dict[str, Any]:
    experiment = load_yaml(path)
    root = Path(path).resolve().parents[2]

    config: dict[str, Any] = {}
    for section in ("model", "data", "train"):
        target = experiment.get(section)
        if target is None:
            raise ValueError(f"Experiment config is missing '{section}'.")
        section_payload = load_yaml(root / "configs" / section / target)
        config[section] = section_payload

    if "overrides" in experiment:
        for key, value in experiment["overrides"].items():
            if key not in config:
                raise ValueError(f"Unknown override section '{key}'.")
            config[key] = deep_merge(config[key], value)

    config["run"] = experiment.get("run", {})
    return config
