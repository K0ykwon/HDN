from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twr.modules.latent_encoder import LatentEncoderConfig, LatentWindowAutoencoder
from src.twr.training.trainer import build_dataloaders, move_batch
from src.twr.utils.logging import JsonlLogger, ensure_dir, write_json
from src.twr.utils.seed import set_seed
from src.twr.utils.config import load_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the overlapping latent encoder.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to an experiment YAML, for example configs/experiment/latent_pretrain_lra_listops.yaml",
    )
    return parser.parse_args()


def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    vocab_size = logits.size(-1)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, accuracy


def run_epoch(
    model: LatentWindowAutoencoder,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    acc_sum = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["tokens"])
        loss, accuracy = reconstruction_loss(outputs["logits"], outputs["targets"])
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item())
        acc_sum += accuracy
        batch_count += 1

    normalizer = max(batch_count, 1)
    return {
        "loss": loss_sum / normalizer,
        "token_accuracy": acc_sum / normalizer,
    }


def main() -> None:
    args = parse_args()
    config = load_experiment(args.experiment)
    train_config = config["train"]
    model_config = config["model"]
    data_config = config["data"]
    run_name = config["run"].get("name", "latent_pretrain")

    seed = int(train_config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(data_config, seed=seed)
    model = LatentWindowAutoencoder(
        LatentEncoderConfig(
            vocab_size=data_config["vocab_size"],
            max_seq_len=data_config["seq_len"],
            embed_dim=int(model_config["embed_dim"]),
            latent_dim=int(model_config["slot_dim"]),
            window_size=int(model_config.get("window_size", 8)),
            stride=int(model_config.get("stride", 4)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )

    run_dir = ensure_dir(ROOT / "experiments" / "pretrained" / run_name)
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    write_json(run_dir / "config_snapshot.json", config)

    best_val_loss = float("inf")
    best_summary: dict[str, float | int | str] = {}
    for epoch in range(1, int(train_config["epochs"]) + 1):
        train_metrics = run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        with torch.no_grad():
            val_metrics = run_epoch(model=model, loader=val_loader, optimizer=None, device=device)
        logger.log({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.encoder.state_dict(), run_dir / "encoder.pt")
            best_summary = {
                "run_name": run_name,
                "seed": seed,
                "final_train_loss": train_metrics["loss"],
                "final_val_loss": val_metrics["loss"],
                "final_val_token_accuracy": val_metrics["token_accuracy"],
                "latent_dim": int(model_config["slot_dim"]),
                "window_size": int(model_config.get("window_size", 8)),
                "stride": int(model_config.get("stride", 4)),
                "encoder_path": str(run_dir / "encoder.pt"),
            }
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_token_acc={val_metrics['token_accuracy']:.4f}"
        )

    write_json(run_dir / "summary.json", best_summary)
    print(f"Saved encoder to {run_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()
