from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerBaselineConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_hidden_dim: int = 256
    dropout: float = 0.1


class TransformerEncoderBaseline(nn.Module):
    """Small token-persistent baseline."""

    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        ones = torch.ones(batch_size, 1, device=tokens.device)
        return {
            "logits": logits,
            "effective_depth": ones.squeeze(-1),
            "step_gates": ones,
            "slot_gates": ones.unsqueeze(1),
            "avg_active_slots": ones.mean().unsqueeze(0),
            "slot_histogram": ones.squeeze(-1),
        }
