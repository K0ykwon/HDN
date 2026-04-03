from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class LossBreakdown:
    task_loss: Tensor
    depth_penalty: Tensor
    slot_penalty: Tensor
    total_loss: Tensor


def compute_losses(
    logits: Tensor,
    labels: Tensor,
    effective_depth: Tensor,
    slot_gates: Tensor,
    think_steps: int,
    depth_penalty_weight: float,
    slot_penalty_weight: float,
) -> LossBreakdown:
    task_loss = nn.functional.cross_entropy(logits, labels)
    normalized_depth = effective_depth.float().mean() / max(float(think_steps), 1.0)
    avg_slot_gate = slot_gates.float().mean()
    depth_penalty = normalized_depth * depth_penalty_weight
    slot_penalty = avg_slot_gate * slot_penalty_weight
    total_loss = task_loss + depth_penalty + slot_penalty
    return LossBreakdown(
        task_loss=task_loss,
        depth_penalty=depth_penalty,
        slot_penalty=slot_penalty,
        total_loss=total_loss,
    )
