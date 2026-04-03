from __future__ import annotations

from torch import Tensor, nn


class ReadoutHead(nn.Module):
    """Pool memory slots and produce class logits."""

    def __init__(self, slot_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(slot_dim, num_classes)

    def forward(self, memory: Tensor) -> Tensor:
        pooled = memory.mean(dim=1)
        return self.proj(pooled)
