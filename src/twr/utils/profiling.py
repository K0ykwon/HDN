from __future__ import annotations

import time
from dataclasses import dataclass

from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def approx_flops_per_token(model_name: str, seq_len: int, hidden_dim: int, slots: int) -> int:
    if model_name == "twr":
        return seq_len * hidden_dim * slots
    if model_name == "transformer":
        return seq_len * seq_len * hidden_dim
    if model_name == "perceiver":
        return seq_len * hidden_dim * slots
    return 0


@dataclass
class ThroughputTracker:
    total_examples: int = 0
    total_time: float = 0.0

    def update(self, batch_size: int, duration: float) -> None:
        self.total_examples += batch_size
        self.total_time += duration

    @property
    def examples_per_second(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.total_examples / self.total_time


def now() -> float:
    return time.perf_counter()
