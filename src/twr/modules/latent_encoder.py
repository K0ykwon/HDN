from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LatentEncoderConfig:
    vocab_size: int
    max_seq_len: int
    embed_dim: int
    latent_dim: int
    window_size: int = 8
    stride: int = 4
    dropout: float = 0.1


class PretrainedLatentEncoder(nn.Module):
    """Compress overlapping token windows into a shorter latent sequence."""

    def __init__(self, config: LatentEncoderConfig) -> None:
        super().__init__()
        if config.window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if config.stride < 1:
            raise ValueError("stride must be at least 1.")
        self.config = config
        position_dim = max(min(config.embed_dim // 2, 16), 8)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, position_dim)
        self.position_proj = nn.Linear(position_dim, config.embed_dim, bias=False)
        self.window_offsets = nn.Parameter(torch.randn(config.window_size, config.embed_dim) * 0.02)
        self.window_norm = nn.LayerNorm(config.embed_dim)
        self.token_proj = nn.Linear(config.embed_dim, config.latent_dim)
        self.token_gate = nn.Linear(config.embed_dim, 1)
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def _pad_tokens(self, tokens: Tensor) -> Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}."
            )
        if seq_len < self.config.window_size:
            pad = self.config.window_size - seq_len
        else:
            remainder = (seq_len - self.config.window_size) % self.config.stride
            pad = (self.config.stride - remainder) % self.config.stride
        if pad == 0:
            return tokens
        return F.pad(tokens, (0, pad), value=0)

    def _embed_tokens(self, tokens: Tensor) -> Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        position_hidden = self.position_proj(self.position_embedding(positions.clamp_max(self.config.max_seq_len - 1)))
        return self.token_embedding(tokens) + position_hidden

    def window_count(self, seq_len: int) -> int:
        padded = self._pad_tokens(torch.zeros(1, seq_len, dtype=torch.long))
        return 1 + (padded.size(1) - self.config.window_size) // self.config.stride

    def token_windows(self, tokens: Tensor) -> Tensor:
        padded = self._pad_tokens(tokens)
        windows = padded.unfold(dimension=1, size=self.config.window_size, step=self.config.stride)
        return windows.contiguous()

    def forward(self, tokens: Tensor) -> Tensor:
        padded_tokens = self._pad_tokens(tokens)
        hidden = self._embed_tokens(padded_tokens)
        windows = hidden.unfold(dimension=1, size=self.config.window_size, step=self.config.stride)
        windows = windows.contiguous().permute(0, 1, 3, 2)
        windows = self.window_norm(windows + self.window_offsets.view(1, 1, self.config.window_size, -1))
        gates = self.token_gate(windows).squeeze(-1).softmax(dim=-1)
        projected = self.token_proj(windows)
        pooled = projected.mean(dim=2) + torch.einsum("bnw,bnwd->bnd", gates, projected)
        return self.output_norm(self.dropout(F.gelu(pooled)))

    def load_encoder_state(self, path: str | Path) -> None:
        state_dict = torch.load(Path(path), map_location="cpu")
        self.load_state_dict(state_dict, strict=True)


class LatentWindowAutoencoder(nn.Module):
    """Pretrain the latent encoder by reconstructing overlapping token windows."""

    def __init__(self, config: LatentEncoderConfig) -> None:
        super().__init__()
        self.encoder = PretrainedLatentEncoder(config)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.window_size * config.vocab_size),
        )
        self.vocab_size = config.vocab_size
        self.window_size = config.window_size

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        latents = self.encoder(tokens)
        logits = self.decoder(latents)
        logits = logits.view(tokens.size(0), latents.size(1), self.window_size, self.vocab_size)
        targets = self.encoder.token_windows(tokens)
        return {"latents": latents, "logits": logits, "targets": targets}
