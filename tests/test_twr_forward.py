import torch

from src.twr.models.factory import build_model


def test_twr_forward_shapes() -> None:
    model = build_model(
        {
            "name": "twr",
            "embed_dim": 32,
            "slot_dim": 32,
            "think_steps": 2,
            "mlp_hidden_dim": 64,
            "dropout": 0.0,
            "local_context_kernel": 3,
            "window_size": 4,
            "stride": 2,
        },
        {"vocab_size": 20, "seq_len": 10, "num_classes": 2},
    )
    outputs = model(torch.randint(0, 20, (4, 10)))
    assert outputs["logits"].shape == (4, 2)
    assert outputs["step_gates"].shape == (4, 2)
    assert outputs["slot_gates"].shape == (4, 2, 2)
    assert outputs["slot_histogram"].shape == (7,)
    assert outputs["think_slot_histogram"].shape == (3,)
    assert outputs["avg_active_think_slots"].shape == (4,)


def test_twr_forward_produces_finite_adaptive_stats() -> None:
    model = build_model(
        {
            "name": "twr",
            "embed_dim": 24,
            "slot_dim": 24,
            "think_steps": 3,
            "mlp_hidden_dim": 48,
            "dropout": 0.0,
            "local_context_kernel": 3,
            "window_size": 4,
            "stride": 2,
        },
        {"vocab_size": 30, "seq_len": 12, "num_classes": 3},
    )
    outputs = model(torch.randint(0, 30, (3, 12)))
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["step_gates"]).all()
    assert torch.isfinite(outputs["slot_gates"]).all()
    assert torch.isfinite(outputs["effective_depth"]).all()
    assert ((outputs["step_gates"] >= 0.0) & (outputs["step_gates"] <= 1.0)).all()
    assert ((outputs["slot_gates"] >= 0.0) & (outputs["slot_gates"] <= 1.0)).all()
    assert ((outputs["effective_depth"] >= 1.0) & (outputs["effective_depth"] <= 4.0)).all()


def test_twr_forward_with_global_memory_channel() -> None:
    model = build_model(
        {
            "name": "twr",
            "embed_dim": 32,
            "slot_dim": 32,
            "think_steps": 2,
            "mlp_hidden_dim": 64,
            "dropout": 0.0,
            "local_context_kernel": 3,
            "window_size": 4,
            "stride": 2,
            "use_global_memory": True,
            "memory_slots": 4,
        },
        {"vocab_size": 20, "seq_len": 10, "num_classes": 2},
    )
    outputs = model(torch.randint(0, 20, (4, 10)))
    assert outputs["logits"].shape == (4, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.isfinite(outputs["effective_depth"]).all()


def test_twr_forward_with_global_memory_and_carriers() -> None:
    model = build_model(
        {
            "name": "twr",
            "embed_dim": 32,
            "slot_dim": 32,
            "think_steps": 2,
            "mlp_hidden_dim": 64,
            "dropout": 0.0,
            "local_context_kernel": 3,
            "window_size": 4,
            "stride": 2,
            "use_global_memory": True,
            "memory_slots": 4,
            "carrier_slots": 2,
        },
        {"vocab_size": 20, "seq_len": 10, "num_classes": 2},
    )
    outputs = model(torch.randint(0, 20, (4, 10)))
    assert outputs["logits"].shape == (4, 2)
    assert torch.isfinite(outputs["logits"]).all()
    assert outputs["slot_histogram"].numel() >= 7
