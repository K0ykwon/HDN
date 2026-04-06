from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from src.twr.modules.latent_backbone import HierarchicalLatentBackbone
from src.twr.modules.latent_encoder import LatentEncoderConfig, PretrainedLatentEncoder


@dataclass
class TWRConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    event_dim: int = 128
    slots: int = 16
    slot_dim: int = 128
    think_steps: int = 2
    mixer_rank: int = 4
    mlp_hidden_dim: int = 256
    dropout: float = 0.1
    local_context_kernel: int = 5
    attention_heads: int = 4
    adaptive_depth: bool = False
    use_slot_gate: bool = False
    write_variant: str = "unused"
    write_topk: int | None = None
    write_temperature: float = 1.0
    write_usage_decay: float = 1.0
    write_usage_penalty: float = 0.0
    write_novelty_scale: float = 0.0
    step_gate_scale: float = 1.0
    step_gate_bias: float = 0.0
    slot_gate_scale: float = 1.0
    slot_gate_bias: float = 0.0
    use_token_residual: bool = False
    window_size: int = 8
    stride: int = 4
    latents_per_window: int = 1
    num_readout_queries: int = 4
    use_global_memory: bool = False
    memory_slots: int = 0
    carrier_slots: int = 0
    pretrained_encoder_path: str | None = None
    freeze_encoder: bool = False


class TWRLM(nn.Module):
    """Hierarchical latent backbone intended to replace token-persistent transformers."""

    def __init__(self, config: TWRConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PretrainedLatentEncoder(
            LatentEncoderConfig(
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                embed_dim=config.embed_dim,
                latent_dim=config.slot_dim,
                window_size=config.window_size,
                stride=config.stride,
                latents_per_window=config.latents_per_window,
                dropout=config.dropout,
            )
        )
        if config.pretrained_encoder_path:
            self.encoder.load_encoder_state(Path(config.pretrained_encoder_path))
        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.backbone = HierarchicalLatentBackbone(
            latent_dim=config.slot_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            kernel_size=config.local_context_kernel,
            depth=config.think_steps,
            num_classes=config.num_classes,
            num_readout_queries=config.num_readout_queries,
            dropout=config.dropout,
            use_global_memory=config.use_global_memory,
            memory_slots=config.memory_slots,
            carrier_slots=config.carrier_slots,
        )

    def _build_gate_stats(
        self,
        merge_gates: list[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        step_count = max(self.config.think_steps, 1)
        if not merge_gates:
            step_gates = torch.ones(batch_size, 1, device=device)
            slot_gates = torch.ones(batch_size, 1, 1, device=device)
            effective_depth = torch.ones(batch_size, device=device)
            return self._pad_gate_tensors(step_gates, slot_gates, step_count, device), slot_gates, effective_depth

        merge_step_gates = [gate.mean(dim=1) for gate in merge_gates]
        step_gates = torch.stack(merge_step_gates, dim=1)
        max_slots = max(gate.size(1) for gate in merge_gates)
        slot_gates = torch.zeros(batch_size, len(merge_gates), max_slots, device=device)
        for index, gate in enumerate(merge_gates):
            slot_gates[:, index, : gate.size(1)] = gate
        effective_depth = 1.0 + step_gates.sum(dim=1)
        step_gates, slot_gates = self._pad_gate_tensors(step_gates, slot_gates, step_count, device)
        return step_gates, slot_gates, effective_depth

    @staticmethod
    def _pad_gate_tensors(
        step_gates: Tensor,
        slot_gates: Tensor,
        step_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        current_steps = step_gates.size(1)
        if current_steps < step_count:
            pad = step_count - current_steps
            step_gates = torch.cat([step_gates, torch.zeros(step_gates.size(0), pad, device=device)], dim=1)
            slot_pad = torch.zeros(slot_gates.size(0), pad, slot_gates.size(2), device=device)
            slot_gates = torch.cat([slot_gates, slot_pad], dim=1)
        elif current_steps > step_count:
            step_gates = step_gates[:, :step_count]
            slot_gates = slot_gates[:, :step_count, :]
        return step_gates, slot_gates

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected [batch, seq] tokens, got shape={tuple(tokens.shape)}.")
        latents = self.encoder(tokens)
        logits, readout_weights, _level_lengths, merge_gates, level_gates = self.backbone(latents)
        batch_size = tokens.size(0)
        latent_length = latents.size(1)
        step_gates, slot_gates, effective_depth = self._build_gate_stats(
            merge_gates=merge_gates,
            batch_size=batch_size,
            device=tokens.device,
        )
        slot_histogram = readout_weights.mean(dim=(0, 1))
        readout_level_gates = level_gates.mean(dim=0)
        return {
            "logits": logits,
            "effective_depth": effective_depth,
            "step_gates": step_gates,
            "slot_gates": slot_gates,
            "avg_active_slots": torch.tensor([float(latent_length)], device=tokens.device),
            "slot_histogram": slot_histogram,
            "avg_active_think_slots": slot_gates.sum(dim=-1).mean(dim=1),
            "think_slot_histogram": readout_level_gates,
        }
