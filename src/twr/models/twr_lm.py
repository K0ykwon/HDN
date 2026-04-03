from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.twr.modules.event_encoder import EventEncoder, EventEncoderConfig
from src.twr.modules.latent_memory import LatentMemory
from src.twr.modules.readout import ReadoutHead
from src.twr.modules.sequential_write import SequentialSoftWrite
from src.twr.modules.think_loop import ThinkLoop


@dataclass
class TWRConfig:
    vocab_size: int
    max_seq_len: int
    num_classes: int
    embed_dim: int = 128
    event_dim: int = 128
    slots: int = 16
    slot_dim: int = 128
    think_steps: int = 4
    mixer_rank: int = 4
    mlp_hidden_dim: int = 256
    dropout: float = 0.1
    adaptive_depth: bool = True
    use_slot_gate: bool = True
    write_variant: str = "soft_write"
    use_token_residual: bool = False


class TWRLM(nn.Module):
    """Tokenless write-think-read latent classifier."""

    def __init__(self, config: TWRConfig) -> None:
        super().__init__()
        self.config = config
        self.event_encoder = EventEncoder(
            EventEncoderConfig(
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                embed_dim=config.embed_dim,
                event_dim=config.event_dim,
                dropout=config.dropout,
            )
        )
        self.memory = LatentMemory(slots=config.slots, slot_dim=config.slot_dim)
        self.writer = SequentialSoftWrite(
            event_dim=config.event_dim,
            slot_dim=config.slot_dim,
            slots=config.slots,
            variant=config.write_variant,
        )
        self.think = ThinkLoop(
            slots=config.slots,
            slot_dim=config.slot_dim,
            think_steps=config.think_steps,
            mixer_rank=config.mixer_rank,
            mlp_hidden_dim=config.mlp_hidden_dim,
            dropout=config.dropout,
            adaptive_depth=config.adaptive_depth,
            use_slot_gate=config.use_slot_gate,
        )
        self.readout = ReadoutHead(slot_dim=config.slot_dim, num_classes=config.num_classes)
        self.token_residual = (
            nn.Linear(config.event_dim, config.num_classes) if config.use_token_residual else None
        )

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected [batch, seq] tokens, got shape={tuple(tokens.shape)}.")
        events = self.event_encoder(tokens)
        memory = self.memory(tokens.size(0))
        memory, write_stats = self.writer(memory, events)
        memory, think_stats = self.think(memory)
        logits = self.readout(memory)
        if self.token_residual is not None:
            logits = logits + self.token_residual(events.mean(dim=1))
        return {
            "logits": logits,
            "effective_depth": think_stats.effective_depth,
            "step_gates": think_stats.step_gates,
            "slot_gates": think_stats.slot_gates,
            "avg_active_slots": write_stats.avg_active_slots.unsqueeze(0),
            "slot_histogram": write_stats.slot_histogram,
        }
