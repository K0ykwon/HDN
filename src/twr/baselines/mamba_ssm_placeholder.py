from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MambaPlaceholderConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    state_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class ResidualStateBlock(nn.Module):
    """Lightweight state-space-inspired placeholder block."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        conv_in = x.transpose(1, 2)
        mixed = self.conv(conv_in).transpose(1, 2)
        gated = torch.sigmoid(self.gate(x)) * mixed
        return residual + self.dropout(self.proj(gated))


class MambaSSMPlaceholder(nn.Module):
    """Shared-interface placeholder until a real Mamba/SSM baseline is added."""

    def __init__(self, config: MambaPlaceholderConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.input_proj = nn.Linear(config.embed_dim, config.state_dim)
        self.blocks = nn.ModuleList(
            [ResidualStateBlock(dim=config.state_dim, dropout=config.dropout) for _ in range(config.num_layers)]
        )
        self.classifier = nn.Linear(config.state_dim, config.num_classes)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.input_proj(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.classifier(hidden.mean(dim=1))
        depth = torch.full((batch_size,), float(len(self.blocks)), device=tokens.device)
        step_gates = torch.ones(batch_size, len(self.blocks), device=tokens.device)
        return {
            "logits": logits,
            "effective_depth": depth,
            "step_gates": step_gates,
            "slot_gates": torch.ones(batch_size, len(self.blocks), seq_len, device=tokens.device),
            "avg_active_slots": torch.full((1,), float(seq_len), device=tokens.device),
            "slot_histogram": torch.ones(seq_len, device=tokens.device),
            "avg_active_think_slots": torch.full((batch_size,), float(seq_len), device=tokens.device),
            "think_slot_histogram": torch.ones(seq_len, device=tokens.device),
        }
