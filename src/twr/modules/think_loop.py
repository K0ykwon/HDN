from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .refine_block import RefineBlock


@dataclass
class ThinkStats:
    step_gates: Tensor
    slot_gates: Tensor
    effective_depth: Tensor
    avg_active_think_slots: Tensor
    slot_gate_histogram: Tensor


class ThinkLoop(nn.Module):
    """Refine memory for a fixed or adaptive number of steps."""

    def __init__(
        self,
        slots: int,
        slot_dim: int,
        think_steps: int,
        mixer_rank: int,
        mlp_hidden_dim: int,
        dropout: float,
        adaptive_depth: bool,
        use_slot_gate: bool,
    ) -> None:
        super().__init__()
        self.think_steps = think_steps
        self.adaptive_depth = adaptive_depth
        self.use_slot_gate = use_slot_gate
        self.blocks = nn.ModuleList(
            [
                RefineBlock(
                    slots=slots,
                    dim=slot_dim,
                    mixer_rank=mixer_rank,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(think_steps)
            ]
        )
        self.step_gate = nn.Linear(slot_dim, 1)
        self.slot_gate = nn.Linear(slot_dim, 1)

    def forward(self, memory: Tensor) -> tuple[Tensor, ThinkStats]:
        step_gates: list[Tensor] = []
        slot_gates: list[Tensor] = []
        effective_depth = torch.zeros(memory.size(0), device=memory.device)

        for block in self.blocks:
            delta = block(memory)
            pooled = memory.mean(dim=1)
            step_gate = torch.sigmoid(self.step_gate(pooled))
            if not self.adaptive_depth:
                step_gate = torch.ones_like(step_gate)

            if self.use_slot_gate:
                slot_gate = torch.sigmoid(self.slot_gate(memory))
            else:
                slot_gate = torch.ones(memory.size(0), memory.size(1), 1, device=memory.device)

            memory = memory + step_gate.unsqueeze(1) * slot_gate * delta
            step_gates.append(step_gate.squeeze(-1))
            slot_gates.append(slot_gate.squeeze(-1))
            effective_depth = effective_depth + step_gate.squeeze(-1)

        step_tensor = torch.stack(step_gates, dim=1)
        slot_tensor = torch.stack(slot_gates, dim=1)
        avg_active_think_slots = (slot_tensor > 0.5).float().sum(dim=-1).mean(dim=1)
        slot_gate_histogram = slot_tensor.mean(dim=(0, 1))

        return memory, ThinkStats(
            step_gates=step_tensor,
            slot_gates=slot_tensor,
            effective_depth=effective_depth,
            avg_active_think_slots=avg_active_think_slots,
            slot_gate_histogram=slot_gate_histogram,
        )
