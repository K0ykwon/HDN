from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BatchMetrics:
    loss: float
    accuracy: float
    batch_size: int


def classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean().item()


def binary_f1(logits: Tensor, labels: Tensor) -> float:
    if logits.size(-1) != 2:
        return 0.0
    predictions = logits.argmax(dim=-1)
    true_positive = ((predictions == 1) & (labels == 1)).sum().float()
    precision = true_positive / ((predictions == 1).sum().float() + 1e-8)
    recall = true_positive / ((labels == 1).sum().float() + 1e-8)
    return float((2 * precision * recall / (precision + recall + 1e-8)).item())


def pearson_correlation(x: Tensor, y: Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if torch.isclose(denom, torch.tensor(0.0, device=x.device)):
        return 0.0
    return float(torch.dot(vx, vy).div(denom).item())
