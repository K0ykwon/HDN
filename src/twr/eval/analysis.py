from __future__ import annotations

from torch import Tensor

from src.twr.eval.metrics import pearson_correlation


def difficulty_depth_correlation(difficulty: Tensor, effective_depth: Tensor) -> float:
    return pearson_correlation(difficulty.float(), effective_depth.float())
