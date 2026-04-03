import torch

from src.twr.models.factory import build_model


def test_mamba_placeholder_forward_shapes() -> None:
    model = build_model(
        {
            "name": "mamba_placeholder",
            "embed_dim": 32,
            "state_dim": 32,
            "num_layers": 2,
            "dropout": 0.0,
        },
        {"vocab_size": 20, "seq_len": 10, "num_classes": 2},
    )
    outputs = model(torch.randint(0, 20, (4, 10)))
    assert outputs["logits"].shape == (4, 2)
    assert outputs["step_gates"].shape == (4, 2)
