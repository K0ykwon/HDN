from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PerceiverBaselineConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    latent_slots: int = 16
    latent_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class CrossAttention(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(input_dim, latent_dim)
        self.value = nn.Linear(input_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: Tensor, inputs: Tensor) -> Tensor:
        q = self.query(latents)
        k = self.key(inputs)
        v = self.value(inputs)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (latents.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return self.out(torch.matmul(weights, v))


class PerceiverLatentBaseline(nn.Module):
    """Simple latent bottleneck baseline with token access after cross-attention."""

    def __init__(self, config: PerceiverBaselineConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.latents = nn.Parameter(torch.randn(1, config.latent_slots, config.latent_dim) * 0.02)
        self.cross = CrossAttention(latent_dim=config.latent_dim, input_dim=config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=4,
            dim_feedforward=config.latent_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.latent_dim, config.num_classes)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        inputs = self.token_embedding(tokens) + self.position_embedding(positions)
        latents = self.latents.expand(batch_size, -1, -1)
        latents = latents + self.cross(latents, inputs)
        encoded = self.encoder(latents)
        logits = self.classifier(encoded.mean(dim=1))
        depth = torch.full((batch_size,), float(len(self.encoder.layers)), device=tokens.device)
        ones = torch.ones(batch_size, len(self.encoder.layers), device=tokens.device)
        return {
            "logits": logits,
            "effective_depth": depth,
            "step_gates": ones,
            "slot_gates": torch.ones(batch_size, len(self.encoder.layers), encoded.size(1), device=tokens.device),
            "avg_active_slots": torch.full((1,), float(encoded.size(1)), device=tokens.device),
            "slot_histogram": torch.ones(encoded.size(1), device=tokens.device),
        }
