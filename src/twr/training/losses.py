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
    avg_active_slots: Tensor,
    slot_histogram: Tensor,
    think_steps: int,
    depth_penalty_weight: float,
    slot_penalty_weight: float,
    write_penalty_weight: float,
    balance_penalty_weight: float,
) -> LossBreakdown:
    task_loss = nn.functional.cross_entropy(logits, labels)
    normalized_depth = effective_depth.float().mean() / max(float(think_steps), 1.0)
    avg_slot_gate = slot_gates.float().mean()
    total_slots = max(float(slot_histogram.numel()), 1.0)
    normalized_active_slots = avg_active_slots.float().mean() / total_slots
    normalized_usage = slot_histogram.float()
    normalized_usage = normalized_usage / normalized_usage.sum().clamp_min(1e-8)
    uniform_usage = normalized_usage.new_full(normalized_usage.shape, 1.0 / total_slots)
    depth_penalty = normalized_depth * depth_penalty_weight
    slot_penalty = avg_slot_gate * slot_penalty_weight
    write_penalty = normalized_active_slots * write_penalty_weight
    balance_penalty = (normalized_usage - uniform_usage).pow(2).mean() * balance_penalty_weight
    total_loss = task_loss + depth_penalty + slot_penalty + write_penalty + balance_penalty
    return LossBreakdown(
        task_loss=task_loss,
        depth_penalty=depth_penalty,
        slot_penalty=slot_penalty + write_penalty + balance_penalty,
        total_loss=total_loss,
    )
