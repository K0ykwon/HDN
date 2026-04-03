from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def collate_batch(batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "tokens": torch.stack([item["tokens"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "difficulty": torch.stack([item["difficulty"] for item in batch], dim=0),
    }
