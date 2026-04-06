from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class StridedLatentReducer(nn.Module):
    """Reduce latent length with overlap-aware strided mixing instead of rigid pairwise merge."""

    def __init__(self, latent_dim: int, dropout: float, carrier_slots: int = 0) -> None:
        super().__init__()
        self.carrier_slots = carrier_slots
        self.sequence_norm = nn.LayerNorm(latent_dim)
        self.reducer = nn.Sequential(
            nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=latent_dim,
            ),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
        )
        self.mix_gate = nn.Linear(latent_dim * 2, 1)
        self.carrier_score = nn.Linear(latent_dim, 1)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.mix_gate.bias, 1.0)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        if latents.size(1) == 1:
            gate = latents.new_ones(latents.size(0), 1)
            return latents, gate
        normalized = self.sequence_norm(latents)
        reduced = self.reducer(normalized.transpose(1, 2)).transpose(1, 2)
        pooled = F.avg_pool1d(latents.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
        min_len = min(reduced.size(1), pooled.size(1))
        reduced = reduced[:, :min_len, :]
        pooled = pooled[:, :min_len, :]
        pair = torch.cat([reduced, pooled], dim=-1)
        gate = 0.5 + 0.5 * torch.sigmoid(self.mix_gate(pair))
        merged = pooled + gate * (reduced - pooled)
        if self.carrier_slots > 0 and latents.size(1) > self.carrier_slots:
            carrier_source = self.sequence_norm(latents)
            scores = self.carrier_score(carrier_source).squeeze(-1)
            topk = min(self.carrier_slots, scores.size(1))
            carrier_indices = scores.topk(k=topk, dim=1).indices
            carrier_indices, _ = carrier_indices.sort(dim=1)
            gather_index = carrier_indices.unsqueeze(-1).expand(-1, -1, latents.size(-1))
            carriers = torch.gather(latents, dim=1, index=gather_index)
            merged = torch.cat([carriers, merged], dim=1)
        return self.output_norm(self.dropout(merged)), gate.squeeze(-1)


class SummaryReadout(nn.Module):
    """Read all pyramid levels with simple per-level summaries and salience gates."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        max_levels: int,
        _num_queries: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_levels = max_levels
        self.level_summary = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.level_gate = nn.Linear(latent_dim, 1)
        self.output_norm = nn.LayerNorm(latent_dim * max_levels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dim * max_levels, num_classes)
        nn.init.constant_(self.level_gate.bias, 1.0)

    def forward(self, levels: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = levels[0].size(0)
        latent_dim = levels[0].size(-1)
        pooled_levels = levels[0].new_zeros(batch_size, self.max_levels, latent_dim)
        level_gates = levels[0].new_zeros(batch_size, self.max_levels)
        slot_weights: list[Tensor] = []
        for level_index, latents in enumerate(levels):
            level_mean = latents.mean(dim=1)
            level_max = latents.amax(dim=1)
            level_hidden = self.level_summary(torch.cat([level_mean, level_max], dim=-1))
            gate = 0.5 + 0.5 * torch.sigmoid(self.level_gate(level_hidden))
            pooled_levels[:, level_index, :] = level_hidden * gate
            level_gates[:, level_index] = gate.squeeze(-1)
            weights = gate / max(latents.size(1), 1)
            slot_weights.append(weights.expand(-1, latents.size(1)))
        pooled = self.output_norm(self.dropout(pooled_levels.reshape(batch_size, -1)))
        readout_weights = torch.cat(slot_weights, dim=1).unsqueeze(1)
        return self.classifier(pooled), readout_weights, level_gates


class GlobalMemoryChannel(nn.Module):
    """A fixed-size latent memory bank for cheap global communication without attention."""

    def __init__(self, latent_dim: int, memory_slots: int, dropout: float) -> None:
        super().__init__()
        self.memory_slots = memory_slots
        self.initial_memory = nn.Parameter(torch.randn(1, memory_slots, latent_dim) * 0.02)
        self.summary_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.write_gate = nn.Linear(latent_dim * 2, 1)
        self.write_candidate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.read_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout),
        )
        self.memory_norm = nn.LayerNorm(latent_dim)
        self.latents_norm = nn.LayerNorm(latent_dim)
        nn.init.constant_(self.write_gate.bias, 1.0)

    def initial_state(self, batch_size: int) -> Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).clone()

    def forward(self, latents: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        summary = torch.cat([latents.mean(dim=1), latents.amax(dim=1)], dim=-1)
        summary = self.summary_proj(summary).unsqueeze(1).expand(-1, self.memory_slots, -1)

        normalized_memory = self.memory_norm(memory)
        write_inputs = torch.cat([normalized_memory, summary], dim=-1)
        write_gate = 0.5 + 0.5 * torch.sigmoid(self.write_gate(write_inputs))
        candidate = self.write_candidate(write_inputs)
        memory = memory + write_gate * (candidate - memory)

        read_summary = memory.mean(dim=1, keepdim=True).expand(-1, latents.size(1), -1)
        latents = latents + self.read_proj(torch.cat([self.latents_norm(latents), read_summary], dim=-1))
        return latents, memory


class HierarchicalLatentBackbone(nn.Module):
    """Compose latent chunks into a multi-scale pyramid with strided reducers."""

    def __init__(
        self,
        latent_dim: int,
        mlp_hidden_dim: int,
        kernel_size: int,
        depth: int,
        num_classes: int,
        num_readout_queries: int,
        dropout: float,
        use_global_memory: bool = False,
        memory_slots: int = 0,
        carrier_slots: int = 0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.use_global_memory = use_global_memory and memory_slots > 0
        max_levels = depth + 1
        self.refinement = LocalLatentRefinement(
            latent_dim=latent_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.reducer = StridedLatentReducer(
            latent_dim=latent_dim,
            dropout=dropout,
            carrier_slots=carrier_slots,
        )
        self.readout = SummaryReadout(
            latent_dim=latent_dim,
            num_classes=num_classes,
            max_levels=max_levels,
            _num_queries=num_readout_queries,
            dropout=dropout,
        )
        self.memory_channel = (
            GlobalMemoryChannel(latent_dim=latent_dim, memory_slots=memory_slots, dropout=dropout)
            if self.use_global_memory
            else None
        )

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, list[int], list[Tensor], Tensor]:
        level_lengths = [latents.size(1)]
        levels = [latents]
        reduce_gates: list[Tensor] = []
        memory = None
        if self.memory_channel is not None:
            memory = self.memory_channel.initial_state(latents.size(0))
        for _ in range(self.depth):
            latents = self.refinement(latents)
            if self.memory_channel is not None and memory is not None:
                latents, memory = self.memory_channel(latents, memory)
            if latents.size(1) > 1:
                latents, reduce_gate = self.reducer(latents)
                level_lengths.append(latents.size(1))
                levels.append(latents)
                reduce_gates.append(reduce_gate)
        latents = self.refinement(latents)
        if self.memory_channel is not None and memory is not None:
            latents, memory = self.memory_channel(latents, memory)
        levels[-1] = latents
        logits, weights, level_gates = self.readout(levels)
        return logits, weights, level_lengths, reduce_gates, level_gates
