import torch

from src.twr.modules.latent_encoder import LatentEncoderConfig, LatentWindowAutoencoder, PretrainedLatentEncoder


def test_latent_encoder_emits_overlapping_latents() -> None:
    encoder = PretrainedLatentEncoder(
        LatentEncoderConfig(
            vocab_size=32,
            max_seq_len=16,
            embed_dim=12,
            latent_dim=20,
            window_size=4,
            stride=2,
            dropout=0.0,
        )
    )
    tokens = torch.randint(0, 32, (3, 10))
    latents = encoder(tokens)
    assert latents.shape == (3, 4, 20)
    assert torch.isfinite(latents).all()


def test_latent_autoencoder_reconstructs_windows() -> None:
    model = LatentWindowAutoencoder(
        LatentEncoderConfig(
            vocab_size=24,
            max_seq_len=12,
            embed_dim=8,
            latent_dim=16,
            window_size=4,
            stride=2,
            dropout=0.0,
        )
    )
    outputs = model(torch.randint(0, 24, (2, 9)))
    assert outputs["logits"].shape == (2, 4, 4, 24)
    assert outputs["targets"].shape == (2, 4, 4)
