from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ForwardResult:
    logits: torch.Tensor
    hidden_activations: list[torch.Tensor]


def _activation_factory(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class FlattenEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        feature_dim = 1
        for dim in input_shape:
            feature_dim *= dim
        self.output_dim = int(feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.view(inputs.size(0), -1)


class SmallConvEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], channels: list[int], projection_dim: int) -> None:
        super().__init__()
        input_channels = input_shape[0]
        layers: list[nn.Module] = []
        current_channels = input_channels

        for channel in channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, channel, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            current_channels = channel

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(current_channels, projection_dim)
        self.output_dim = projection_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.projection(pooled)


class FeedForwardClassifier(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        backbone: dict | None = None,
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.hidden_dims = list(hidden_dims)
        self.output_dim = int(output_dim)
        self.backbone_config = backbone or {"name": "flatten"}
        self.encoder = self._build_encoder(input_shape, self.backbone_config)
        self.input_dim = self.encoder.output_dim
        self.layers = nn.ModuleList()
        self.output_layer = nn.Linear(self.hidden_dims[-1] if self.hidden_dims else self.input_dim, self.output_dim)
        self._rebuild_layers()

    def _build_encoder(self, input_shape: tuple[int, ...], backbone: dict) -> nn.Module:
        name = backbone.get("name", "flatten")
        if name == "flatten":
            return FlattenEncoder(input_shape)
        if name == "small_cnn":
            channels = backbone.get("channels", [32, 64])
            projection_dim = int(backbone.get("projection_dim", 128))
            return SmallConvEncoder(input_shape, channels=channels, projection_dim=projection_dim)
        raise ValueError(f"Unsupported backbone: {name}")

    def _rebuild_layers(self) -> None:
        dims = [self.input_dim] + self.hidden_dims
        new_layers = nn.ModuleList()
        old_layers = list(self.layers) if hasattr(self, "layers") else []

        for index in range(len(self.hidden_dims)):
            linear = nn.Linear(dims[index], dims[index + 1])
            if index < len(old_layers):
                _copy_linear_parameters(old_layers[index], linear)
            new_layers.append(linear)

        old_output = getattr(self, "output_layer", None)
        new_output = nn.Linear(dims[-1], self.output_dim)
        if old_output is not None:
            _copy_linear_parameters(old_output, new_output)

        self.layers = new_layers
        self.output_layer = new_output

    def forward(
        self,
        inputs: torch.Tensor,
        neuron_ablation: dict[int, list[int]] | None = None,
        collect_stats: bool = False,
    ) -> ForwardResult | torch.Tensor:
        features = self.encoder(inputs)
        hidden_activations: list[torch.Tensor] = []
        current = features

        for stage_idx, layer in enumerate(self.layers):
            current = layer(current)
            current = _activation_factory(self.activation_name)(current)
            if neuron_ablation and stage_idx in neuron_ablation:
                current = current.clone()
                current[:, neuron_ablation[stage_idx]] = 0.0
            hidden_activations.append(current)

        logits = self.output_layer(current)
        if collect_stats:
            return ForwardResult(logits=logits, hidden_activations=hidden_activations)
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class BaselineMLP(FeedForwardClassifier):
    pass


def _copy_linear_parameters(source: nn.Linear, target: nn.Linear) -> None:
    with torch.no_grad():
        rows = min(source.weight.shape[0], target.weight.shape[0])
        cols = min(source.weight.shape[1], target.weight.shape[1])
        target.weight.zero_()
        target.weight[:rows, :cols] = source.weight[:rows, :cols]
        target.bias.zero_()
        target.bias[:rows] = source.bias[:rows]
