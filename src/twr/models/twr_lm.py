from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from src.twr.modules.latent_backbone import HierarchicalLatentBackbone
from src.twr.modules.latent_encoder import LatentEncoderConfig, PretrainedLatentEncoder


@dataclass
class TWRConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    event_dim: int = 128
    slots: int = 16
    slot_dim: int = 128
    think_steps: int = 2
    mixer_rank: int = 4
    mlp_hidden_dim: int = 256
    dropout: float = 0.1
    local_context_kernel: int = 5
    attention_heads: int = 4
    adaptive_depth: bool = False
    use_slot_gate: bool = False
    write_variant: str = "unused"
    write_topk: int | None = None
    write_temperature: float = 1.0
    write_usage_decay: float = 1.0
    write_usage_penalty: float = 0.0
    write_novelty_scale: float = 0.0
    step_gate_scale: float = 1.0
    step_gate_bias: float = 0.0
    slot_gate_scale: float = 1.0
    slot_gate_bias: float = 0.0
    use_token_residual: bool = False
    window_size: int = 8
    stride: int = 4
    num_readout_queries: int = 4
    pretrained_encoder_path: str | None = None
    freeze_encoder: bool = False


class TWRLM(nn.Module):
    """Hierarchical latent backbone intended to replace token-persistent transformers."""

    def __init__(self, config: TWRConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PretrainedLatentEncoder(
            LatentEncoderConfig(
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                embed_dim=config.embed_dim,
                latent_dim=config.slot_dim,
                window_size=config.window_size,
                stride=config.stride,
                dropout=config.dropout,
            )
        )
        if config.pretrained_encoder_path:
            self.encoder.load_encoder_state(Path(config.pretrained_encoder_path))
        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.backbone = HierarchicalLatentBackbone(
            latent_dim=config.slot_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            kernel_size=config.local_context_kernel,
            depth=config.think_steps,
            num_classes=config.num_classes,
            num_readout_queries=config.num_readout_queries,
            dropout=config.dropout,
        )

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected [batch, seq] tokens, got shape={tuple(tokens.shape)}.")
        latents = self.encoder(tokens)
        logits, readout_weights, level_lengths = self.backbone(latents)
        batch_size = tokens.size(0)
        latent_length = latents.size(1)
        final_length = max(level_lengths[-1], 1)
        depth = torch.full(
            (batch_size,),
            float(len(level_lengths) - 1),
            device=tokens.device,
        )
        step_count = max(len(level_lengths) - 1, 1)
        step_gates = torch.ones(batch_size, step_count, device=tokens.device)
        slot_gates = torch.ones(batch_size, step_count, final_length, device=tokens.device)
        slot_histogram = readout_weights.mean(dim=(0, 1))
        return {
            "logits": logits,
            "effective_depth": depth,
            "step_gates": step_gates,
            "slot_gates": slot_gates,
            "avg_active_slots": torch.tensor([float(latent_length)], device=tokens.device),
            "slot_histogram": slot_histogram,
            "avg_active_think_slots": torch.full((batch_size,), float(final_length), device=tokens.device),
            "think_slot_histogram": slot_histogram,
        }
