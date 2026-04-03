from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class EventEncoderConfig:
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    event_dim: int
    dropout: float = 0.0


class EventEncoder(nn.Module):
    """Embed tokens into transient event vectors."""

    def __init__(self, config: EventEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.event_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}."
            )
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.proj(hidden)
