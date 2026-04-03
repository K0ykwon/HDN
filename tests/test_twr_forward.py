import torch

from src.twr.models.factory import build_model


def test_twr_forward_shapes() -> None:
    model = build_model(
        {
            "name": "twr",
            "embed_dim": 32,
            "event_dim": 32,
            "slots": 8,
            "slot_dim": 32,
            "think_steps": 2,
            "mixer_rank": 2,
            "mlp_hidden_dim": 64,
            "dropout": 0.0,
            "adaptive_depth": True,
            "use_slot_gate": True,
        },
        {"vocab_size": 20, "seq_len": 10, "num_classes": 2},
    )
    outputs = model(torch.randint(0, 20, (4, 10)))
    assert outputs["logits"].shape == (4, 2)
    assert outputs["step_gates"].shape == (4, 2)
    assert outputs["slot_gates"].shape == (4, 2, 8)
    assert outputs["slot_histogram"].shape == (8,)
    assert outputs["think_slot_histogram"].shape == (8,)
    assert outputs["avg_active_think_slots"].shape == (4,)
