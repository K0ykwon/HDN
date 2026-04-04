from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalLatentRefinement(nn.Module):
    """Refine neighboring latent chunks before hierarchical composition."""

    def __init__(self, latent_dim: int, mlp_hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.sequence_norm = nn.LayerNorm(latent_dim)
        self.sequence_mixer = nn.Sequential(
            nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=latent_dim,
            ),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(latent_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents: Tensor) -> Tensor:
        mixed = self.sequence_norm(latents).transpose(1, 2)
        mixed = self.sequence_mixer(mixed).transpose(1, 2)
        latents = latents + mixed
        latents = latents + self.channel_mlp(self.channel_norm(latents))
        return latents


class PairwiseLatentMerge(nn.Module):
    """Merge adjacent latent chunks with learned keep-vs-compose control."""

    def __init__(self, latent_dim: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(latent_dim * 2)
        self.left_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.right_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.diff_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim * 4),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.size(1) == 1:
            return latents
        if latents.size(1) % 2 == 1:
            latents = torch.cat([latents, latents[:, -1:, :]], dim=1)
        left = latents[:, 0::2, :]
        right = latents[:, 1::2, :]
        pair = torch.cat([left, right], dim=-1)
        pair = self.pair_norm(pair)
        compose_gate, contrast_gate, keep_left, keep_right = self.gate_mlp(pair).chunk(4, dim=-1)
        composed = torch.tanh(self.left_proj(left) + self.right_proj(right))
        contrast = torch.tanh(self.diff_proj(left - right))
        merged = (
            torch.sigmoid(keep_left) * left
            + torch.sigmoid(keep_right) * right
            + torch.sigmoid(compose_gate) * composed
            + torch.sigmoid(contrast_gate) * contrast
        )
        return self.output_norm(merged)


class QueryReadout(nn.Module):
    """Read all pyramid levels with multiple learned queries and level embeddings."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        max_levels: int,
        num_queries: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)
        self.level_embedding = nn.Embedding(max_levels, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.output_norm = nn.LayerNorm(latent_dim * num_queries)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dim * num_queries, num_classes)

    def forward(self, levels: list[Tensor]) -> tuple[Tensor, Tensor]:
        stacked_levels: list[Tensor] = []
        for level_index, latents in enumerate(levels):
            positions = self.level_embedding.weight[level_index].view(1, 1, -1)
            stacked_levels.append(latents + positions)
        latents = torch.cat(stacked_levels, dim=1)
        query = self.query.expand(latents.size(0), -1, -1)
        keys = self.key(latents)
        values = self.value(latents)
        scores = torch.einsum("bqd,bnd->bqn", query, keys) / max(latents.size(-1) ** 0.5, 1.0)
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bqn,bnd->bqd", weights, values).reshape(latents.size(0), -1)
        pooled = self.output_norm(self.dropout(pooled))
        return self.classifier(pooled), weights


class HierarchicalLatentBackbone(nn.Module):
    """Compose overlapping chunk latents into a shorter semantic pyramid."""

    def __init__(
        self,
        latent_dim: int,
        mlp_hidden_dim: int,
        kernel_size: int,
        depth: int,
        num_classes: int,
        num_readout_queries: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.refinement = LocalLatentRefinement(
            latent_dim=latent_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.merge = PairwiseLatentMerge(
            latent_dim=latent_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
        self.readout = QueryReadout(
            latent_dim=latent_dim,
            num_classes=num_classes,
            max_levels=depth + 1,
            num_queries=num_readout_queries,
            dropout=dropout,
        )

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, list[int]]:
        level_lengths = [latents.size(1)]
        levels = [latents]
        for _ in range(self.depth):
            latents = self.refinement(latents)
            if latents.size(1) > 1:
                latents = self.merge(latents)
                level_lengths.append(latents.size(1))
                levels.append(latents)
        latents = self.refinement(latents)
        levels[-1] = latents
        logits, weights = self.readout(levels)
        return logits, weights, level_lengths
