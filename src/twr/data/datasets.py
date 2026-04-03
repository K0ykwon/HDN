from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SyntheticSequenceConfig:
    vocab_size: int
    seq_len: int
    num_classes: int
    train_size: int
    val_size: int
    batch_size: int
    task: str = "special_token_parity"
    special_token: int = 1
    compare_token_a: int = 1
    compare_token_b: int = 2


class SyntheticSequenceDataset(Dataset[dict[str, Tensor]]):
    """Deterministic synthetic sequence classification dataset."""

    def __init__(self, size: int, config: SyntheticSequenceConfig, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(size, config.seq_len),
            generator=generator,
        )
        if config.task == "special_token_parity":
            counts = (tokens == config.special_token).sum(dim=1)
            labels = torch.remainder(counts, config.num_classes)
            difficulty = counts.float() / max(config.seq_len, 1)
        elif config.task == "count_compare":
            counts_a = (tokens == config.compare_token_a).sum(dim=1)
            counts_b = (tokens == config.compare_token_b).sum(dim=1)
            labels = (counts_a > counts_b).long()
            margin = (counts_a - counts_b).abs().float()
            difficulty = 1.0 - margin / max(config.seq_len, 1)
        else:
            raise ValueError(f"Unsupported synthetic task: {config.task}")

        self.tokens = tokens
        self.labels = labels.long()
        self.difficulty = difficulty.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }
