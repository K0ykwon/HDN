from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentMemory(nn.Module):
    """Create a learnable initial latent memory bank."""

    def __init__(self, slots: int, slot_dim: int) -> None:
        super().__init__()
        self.slots = slots
        self.slot_dim = slot_dim
        self.initial_memory = nn.Parameter(torch.zeros(1, slots, slot_dim))
        nn.init.normal_(self.initial_memory, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.initial_memory.expand(batch_size, -1, -1).clone()
