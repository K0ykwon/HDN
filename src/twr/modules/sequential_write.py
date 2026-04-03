from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class WriteStats:
    attention: Tensor
    avg_active_slots: Tensor
    slot_histogram: Tensor


class SequentialSoftWrite(nn.Module):
    """Write each event into the latent slots with soft attention."""

    def __init__(self, event_dim: int, slot_dim: int, slots: int, variant: str = "soft_write") -> None:
        super().__init__()
        self.slots = slots
        self.slot_dim = slot_dim
        self.variant = variant
        self.query = nn.Linear(event_dim, slot_dim)
        self.value = nn.Linear(event_dim, slot_dim)
        self.slot_bias = nn.Parameter(torch.zeros(slots, slot_dim))
        self.residual_gate = nn.Linear(event_dim, slot_dim)

    def forward(self, memory: Tensor, events: Tensor) -> tuple[Tensor, WriteStats]:
        batch_size, seq_len, _ = events.shape
        attn_steps: list[Tensor] = []
        if self.variant == "pooled_one_shot_write":
            pooled = events.mean(dim=1, keepdim=True)
            write_values = self.value(pooled).expand(-1, seq_len, -1)
        else:
            write_values = self.value(events)

        for step in range(seq_len):
            event = events[:, step]
            query = self.query(event).unsqueeze(1)
            scores = ((memory + self.slot_bias.unsqueeze(0)) * query).sum(dim=-1) / (self.slot_dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            update = attn.unsqueeze(-1) * write_values[:, step].unsqueeze(1)
            if self.variant == "stronger_residual_write":
                update = update + 0.1 * torch.tanh(self.residual_gate(event)).unsqueeze(1)
            memory = memory + update
            attn_steps.append(attn)

        attention = torch.stack(attn_steps, dim=1)
        slot_usage = attention.sum(dim=1)
        normalized_usage = slot_usage / max(float(seq_len), 1.0)
        entropy = -(normalized_usage * (normalized_usage.clamp_min(1e-8).log())).sum(dim=-1)
        avg_active_slots = entropy.exp().mean()
        slot_histogram = slot_usage.mean(dim=0)
        return memory, WriteStats(
            attention=attention,
            avg_active_slots=avg_active_slots,
            slot_histogram=slot_histogram,
        )
