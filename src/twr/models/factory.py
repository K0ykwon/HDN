from __future__ import annotations

from typing import Any

from torch import nn

from src.twr.baselines.mamba_ssm_placeholder import MambaPlaceholderConfig, MambaSSMPlaceholder
from src.twr.baselines.perceiver_latent import PerceiverBaselineConfig, PerceiverLatentBaseline
from src.twr.baselines.transformer_encoder import (
    TransformerBaselineConfig,
    TransformerEncoderBaseline,
)
from src.twr.models.twr_lm import TWRConfig, TWRLM


def build_model(model_config: dict[str, Any], data_config: dict[str, Any]) -> nn.Module:
    common = {
        "vocab_size": data_config["vocab_size"],
        "max_seq_len": data_config["seq_len"],
        "num_classes": data_config["num_classes"],
    }
    name = model_config["name"]
    params = {k: v for k, v in model_config.items() if k != "name"}

    if name == "twr":
        return TWRLM(TWRConfig(**common, **params))
    if name == "transformer":
        return TransformerEncoderBaseline(TransformerBaselineConfig(**common, **params))
    if name == "perceiver":
        return PerceiverLatentBaseline(PerceiverBaselineConfig(**common, **params))
    if name == "mamba_placeholder":
        return MambaSSMPlaceholder(MambaPlaceholderConfig(**common, **params))
    raise ValueError(f"Unknown model name: {name}")
