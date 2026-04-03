from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class GLUFeedForward(nn.Module):
    """Slot-wise GLU MLP."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(self.dropout(F.gelu(gate) * value))


class LowRankSlotMixer(nn.Module):
    """Mix information across slots with a low-rank bottleneck."""

    def __init__(self, slots: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(slots, rank, bias=False)
        self.up = nn.Linear(rank, slots, bias=False)

    def forward(self, memory: Tensor) -> Tensor:
        transposed = memory.transpose(1, 2)
        mixed = self.up(self.down(transposed))
        return mixed.transpose(1, 2)


class RefineBlock(nn.Module):
    """LayerNorm -> low-rank slot mixing -> GLU MLP with residual output."""

    def __init__(self, slots: int, dim: int, mixer_rank: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = LowRankSlotMixer(slots=slots, rank=mixer_rank)
        self.mlp = GLUFeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(self, memory: Tensor) -> Tensor:
        normalized = self.norm(memory)
        mixed = self.mixer(normalized)
        ff = self.mlp(normalized + mixed)
        return mixed + ff
