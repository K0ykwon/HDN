from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .mlp import BaselineMLP, FeedForwardClassifier, _copy_linear_parameters


@dataclass
class StructuralEvent:
    step: int
    event_type: str
    reason: str
    metadata: dict = field(default_factory=dict)


class HDNPrototype(FeedForwardClassifier):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        backbone: dict | None = None,
        structural: dict | None = None,
    ) -> None:
        self.structural = structural or {}
        self.structural_events: list[StructuralEvent] = []
        self.latest_usage_summary: list[list[float]] = []
        super().__init__(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            backbone=backbone,
        )

    def record_event(self, step: int, event_type: str, reason: str, metadata: dict | None = None) -> None:
        self.structural_events.append(
            StructuralEvent(step=step, event_type=event_type, reason=reason, metadata=metadata or {})
        )

    def maybe_adapt(self, step: int, train_loss: float, valid_loss: float, usage_summary: list[list[float]]) -> bool:
        self.latest_usage_summary = usage_summary
        interval = int(self.structural.get("interval", 10))
        if step % interval != 0:
            return False

        mutated = False
        if self.structural.get("birth", {}).get("enabled", True):
            mutated = self._maybe_birth(step, train_loss, usage_summary) or mutated
        if self.structural.get("split", {}).get("enabled", True):
            mutated = self._maybe_split(step, usage_summary) or mutated
        if self.structural.get("deepen", {}).get("enabled", True):
            mutated = self._maybe_deepen(step, train_loss, valid_loss) or mutated
        if self.structural.get("prune", {}).get("enabled", True):
            mutated = self._maybe_prune(step, usage_summary) or mutated
        return mutated

    def _maybe_birth(self, step: int, train_loss: float, usage_summary: list[list[float]]) -> bool:
        config = self.structural.get("birth", {})
        loss_threshold = float(config.get("loss_threshold", 0.25))
        max_width = int(config.get("max_width", 256))
        growth_amount = int(config.get("growth_amount", 4))
        if train_loss < loss_threshold or not usage_summary:
            return False

        stage_idx = _stage_with_highest_load(usage_summary)
        if self.hidden_dims[stage_idx] >= max_width:
            return False

        self._grow_stage(stage_idx, growth_amount)
        self.record_event(
            step=step,
            event_type="birth",
            reason="Loss remained above threshold and stage load was high.",
            metadata={"stage": stage_idx, "growth_amount": growth_amount, "new_width": self.hidden_dims[stage_idx]},
        )
        return True

    def _maybe_split(self, step: int, usage_summary: list[list[float]]) -> bool:
        config = self.structural.get("split", {})
        heterogeneity_threshold = float(config.get("heterogeneity_threshold", 0.15))
        max_width = int(config.get("max_width", 256))
        growth_amount = int(config.get("growth_amount", 2))
        if not usage_summary:
            return False

        stage_scores = [_coefficient_of_variation(stage_usage) for stage_usage in usage_summary if stage_usage]
        if not stage_scores or max(stage_scores) < heterogeneity_threshold:
            return False

        stage_idx = int(torch.tensor(stage_scores).argmax().item())
        if self.hidden_dims[stage_idx] >= max_width:
            return False

        overloaded_neuron = int(torch.tensor(usage_summary[stage_idx]).argmax().item())
        self._grow_stage(stage_idx, growth_amount)
        self.record_event(
            step=step,
            event_type="split",
            reason="Stage utilization heterogeneity exceeded threshold.",
            metadata={
                "stage": stage_idx,
                "source_neuron": overloaded_neuron,
                "growth_amount": growth_amount,
                "new_width": self.hidden_dims[stage_idx],
            },
        )
        return True

    def _maybe_deepen(self, step: int, train_loss: float, valid_loss: float) -> bool:
        config = self.structural.get("deepen", {})
        plateau_gap = float(config.get("plateau_gap", 0.03))
        max_depth = int(config.get("max_depth", 6))
        if len(self.hidden_dims) >= max_depth:
            return False
        if abs(valid_loss - train_loss) > plateau_gap:
            return False

        self._deepen_tail()
        self.record_event(
            step=step,
            event_type="deepen",
            reason="Training and validation losses plateaued within target gap.",
            metadata={"new_depth": len(self.hidden_dims), "tail_width": self.hidden_dims[-1]},
        )
        return True

    def _maybe_prune(self, step: int, usage_summary: list[list[float]]) -> bool:
        config = self.structural.get("prune", {})
        usage_threshold = float(config.get("usage_threshold", 0.02))
        min_width = int(config.get("min_width", 4))
        if not usage_summary:
            return False

        best_stage = None
        prune_indices: list[int] = []
        for stage_idx, stage_usage in enumerate(usage_summary):
            candidates = [idx for idx, value in enumerate(stage_usage) if value < usage_threshold]
            if len(candidates) > len(prune_indices) and self.hidden_dims[stage_idx] - len(candidates) >= min_width:
                best_stage = stage_idx
                prune_indices = candidates

        if best_stage is None or not prune_indices:
            return False

        self._prune_stage(best_stage, prune_indices)
        self.record_event(
            step=step,
            event_type="prune",
            reason="Detected persistently low-usage neurons.",
            metadata={
                "stage": best_stage,
                "pruned_neurons": prune_indices,
                "new_width": self.hidden_dims[best_stage],
            },
        )
        return True

    def _grow_stage(self, stage_idx: int, amount: int) -> None:
        self.hidden_dims[stage_idx] += amount
        self._rebuild_with_shape_change(stage_idx=stage_idx, keep_indices=None)

    def _deepen_tail(self) -> None:
        tail_width = self.hidden_dims[-1] if self.hidden_dims else max(8, self.input_dim // 2)
        self.hidden_dims.append(tail_width)
        old_layers = list(self.layers)
        old_output = self.output_layer
        self._rebuild_layers()

        with torch.no_grad():
            for index in range(len(old_layers)):
                _copy_linear_parameters(old_layers[index], self.layers[index])

            new_tail = self.layers[-1]
            new_tail.weight.zero_()
            diag = min(new_tail.weight.shape[0], new_tail.weight.shape[1])
            new_tail.weight[:diag, :diag] = torch.eye(diag)
            new_tail.bias.zero_()
            _copy_linear_parameters(old_output, self.output_layer)

    def _prune_stage(self, stage_idx: int, prune_indices: list[int]) -> None:
        keep_indices = [idx for idx in range(self.hidden_dims[stage_idx]) if idx not in set(prune_indices)]
        self.hidden_dims[stage_idx] = len(keep_indices)
        self._rebuild_with_shape_change(stage_idx=stage_idx, keep_indices=keep_indices)

    def _rebuild_with_shape_change(self, stage_idx: int, keep_indices: list[int] | None) -> None:
        old_layers = list(self.layers)
        old_output = self.output_layer
        old_hidden_dims = [layer.out_features for layer in old_layers]

        self._rebuild_layers()

        with torch.no_grad():
            for index, new_layer in enumerate(self.layers):
                if index == stage_idx and keep_indices is not None:
                    source = old_layers[index]
                    new_layer.weight.zero_()
                    new_layer.bias.zero_()
                    new_layer.weight[: len(keep_indices), : source.weight.shape[1]] = source.weight[keep_indices]
                    new_layer.bias[: len(keep_indices)] = source.bias[keep_indices]
                    continue

                if index == stage_idx + 1 and keep_indices is not None:
                    source = old_layers[index]
                    new_layer.weight.zero_()
                    cols = min(len(keep_indices), new_layer.weight.shape[1])
                    new_layer.weight[:, :cols] = source.weight[:, keep_indices[:cols]]
                    new_layer.bias.copy_(source.bias)
                    continue

                if index < len(old_layers):
                    _copy_linear_parameters(old_layers[index], new_layer)

            if keep_indices is not None and stage_idx == len(old_hidden_dims) - 1:
                self.output_layer.weight.zero_()
                cols = min(len(keep_indices), self.output_layer.weight.shape[1])
                self.output_layer.weight[:, :cols] = old_output.weight[:, keep_indices[:cols]]
                self.output_layer.bias.copy_(old_output.bias)
            else:
                _copy_linear_parameters(old_output, self.output_layer)


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = tensor.mean().item()
    if mean == 0.0:
        return 0.0
    return (tensor.std(unbiased=False) / mean).item()


def _stage_with_highest_load(usage_summary: list[list[float]]) -> int:
    stage_means = [sum(stage) / max(len(stage), 1) for stage in usage_summary]
    return int(torch.tensor(stage_means).argmax().item())
