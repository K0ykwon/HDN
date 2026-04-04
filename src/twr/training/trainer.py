from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.twr.data.collators import collate_batch
from src.twr.data.datasets import (
    HuggingFaceTextConfig,
    HuggingFaceTextDataset,
    ListOpsConfig,
    ListOpsDataset,
    LongBenchConfig,
    LongBenchDataset,
    RulerNeedleConfig,
    RulerNeedleDataset,
    SyntheticSequenceConfig,
    SyntheticSequenceDataset,
)
from src.twr.eval.analysis import difficulty_depth_correlation
from src.twr.eval.metrics import binary_f1, classification_accuracy
from src.twr.models.factory import build_model
from src.twr.training.losses import compute_losses
from src.twr.utils.logging import JsonlLogger, ensure_dir, write_json
from src.twr.utils.profiling import ThroughputTracker, approx_flops_per_token, count_parameters, now
from src.twr.utils.seed import set_seed


@dataclass
class TrainingArtifacts:
    run_dir: Path
    summary: dict[str, Any]


def build_dataloaders(data_config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    data_kind = data_config.get("kind", "synthetic")
    if data_kind == "synthetic":
        config = SyntheticSequenceConfig(**data_config)
        train_dataset = SyntheticSequenceDataset(size=config.train_size, config=config, seed=seed)
        val_dataset = SyntheticSequenceDataset(size=config.val_size, config=config, seed=seed + 1)
        batch_size = config.batch_size
    elif data_kind == "lra_listops":
        config = ListOpsConfig(**{k: v for k, v in data_config.items() if k != "kind"})
        train_dataset = ListOpsDataset(size=config.train_size, config=config, seed=seed)
        val_dataset = ListOpsDataset(size=config.val_size, config=config, seed=seed + 1)
        batch_size = config.batch_size
    elif data_kind == "ruler_needle":
        config = RulerNeedleConfig(**{k: v for k, v in data_config.items() if k != "kind"})
        train_dataset = RulerNeedleDataset(size=config.train_size, config=config, seed=seed)
        val_dataset = RulerNeedleDataset(size=config.val_size, config=config, seed=seed + 1)
        batch_size = config.batch_size
    elif data_kind == "hf_text":
        config = HuggingFaceTextConfig(**{k: v for k, v in data_config.items() if k != "kind"})
        train_dataset = HuggingFaceTextDataset(split="train", config=config)
        val_dataset = HuggingFaceTextDataset(split="val", config=config)
        batch_size = config.batch_size
    elif data_kind == "longbench":
        config = LongBenchConfig(**{k: v for k, v in data_config.items() if k != "kind"})
        train_dataset = LongBenchDataset(split="train", config=config)
        val_dataset = LongBenchDataset(split="val", config=config)
        batch_size = config.batch_size
    else:
        raise ValueError(f"Unsupported data kind: {data_kind}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_warmup_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_fraction: float,
) -> LambdaLR:
    warmup_steps = max(int(total_steps * warmup_fraction), 1)

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if current_step <= warmup_steps:
            return current_step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def aggregate_epoch(records: list[dict[str, float]]) -> dict[str, float]:
    weights = [record["batch_size"] for record in records]
    total = max(sum(weights), 1)
    aggregated: dict[str, float] = {}
    for key in records[0]:
        if key == "batch_size":
            continue
        aggregated[key] = sum(record[key] * record["batch_size"] for record in records) / total
    return aggregated


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    device: torch.device,
    train_config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    throughput = ThroughputTracker()
    batch_records: list[dict[str, float]] = []
    slot_histograms: list[torch.Tensor] = []
    think_slot_histograms: list[torch.Tensor] = []
    depth_values: list[torch.Tensor] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        start = now()
        batch = move_batch(batch, device)
        outputs = model(batch["tokens"])
        think_steps = int(outputs["step_gates"].size(1))
        loss_breakdown = compute_losses(
            logits=outputs["logits"],
            labels=batch["labels"],
            effective_depth=outputs["effective_depth"],
            slot_gates=outputs["slot_gates"],
            avg_active_slots=outputs["avg_active_slots"],
            slot_histogram=outputs["slot_histogram"],
            think_steps=think_steps,
            depth_penalty_weight=float(train_config.get("depth_penalty_weight", 0.0)),
            slot_penalty_weight=float(train_config.get("slot_penalty_weight", 0.0)),
            write_penalty_weight=float(train_config.get("write_penalty_weight", 0.0)),
            balance_penalty_weight=float(train_config.get("balance_penalty_weight", 0.0)),
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss_breakdown.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(train_config.get("grad_clip_norm", 1.0)),
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        duration = now() - start
        throughput.update(batch_size=batch["tokens"].size(0), duration=duration)

        accuracy = classification_accuracy(outputs["logits"], batch["labels"])
        step_gates = outputs["step_gates"].float()
        slot_gates = outputs["slot_gates"].float()
        slot_histograms.append(outputs["slot_histogram"].float().detach().cpu())
        think_slot_histograms.append(outputs["think_slot_histogram"].float().detach().cpu())
        depth_values.append(outputs["effective_depth"].float().detach().cpu())
        record = {
            "loss": float(loss_breakdown.total_loss.item()),
            "task_loss": float(loss_breakdown.task_loss.item()),
            "depth_penalty": float(loss_breakdown.depth_penalty.item()),
            "slot_penalty": float(loss_breakdown.slot_penalty.item()),
            "accuracy": accuracy,
            "f1": binary_f1(outputs["logits"], batch["labels"]),
            "effective_depth": float(outputs["effective_depth"].float().mean().item()),
            "avg_step_gate": float(step_gates.mean().item()),
            "avg_active_slots": float(outputs["avg_active_slots"].float().mean().item()),
            "avg_active_think_slots": float(outputs["avg_active_think_slots"].float().mean().item()),
            "avg_slot_gate": float(slot_gates.mean().item()),
            "depth_difficulty_corr": difficulty_depth_correlation(
                batch["difficulty"], outputs["effective_depth"]
            ),
            "batch_size": float(batch["tokens"].size(0)),
        }
        batch_records.append(record)

    epoch_metrics = aggregate_epoch(batch_records)
    epoch_metrics["throughput"] = throughput.examples_per_second
    epoch_metrics["slot_histogram"] = torch.stack(slot_histograms).mean(dim=0).tolist()
    epoch_metrics["think_slot_histogram"] = torch.stack(think_slot_histograms).mean(dim=0).tolist()
    epoch_metrics["depth_distribution"] = torch.cat(depth_values).tolist()
    if torch.cuda.is_available():
        epoch_metrics["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        epoch_metrics["peak_gpu_memory_mb"] = 0.0
    return epoch_metrics, batch_records


def train(config: dict[str, Any]) -> TrainingArtifacts:
    train_config = config["train"]
    data_config = config["data"]
    model_config = config["model"]
    run_config = config["run"]

    seed = int(train_config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(data_config, seed=seed)
    model = build_model(model_config, data_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])
    total_steps = max(len(train_loader) * int(train_config["epochs"]), 1)
    scheduler = build_warmup_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_fraction=float(train_config.get("warmup_fraction", 0.0)),
    )

    dataset_name = data_config.get("task", data_config.get("dataset_name", "dataset"))
    run_name = run_config.get("name") or f"{model_config['name']}_{dataset_name}_seed{seed}"
    run_dir = ensure_dir(Path("experiments/runs") / run_name)
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    write_json(run_dir / "config_snapshot.json", config)

    parameter_count = count_parameters(model)
    approx_flops = approx_flops_per_token(
        model_name=model_config["name"],
        seq_len=data_config["seq_len"],
        hidden_dim=model_config.get("slot_dim", model_config.get("embed_dim", 128)),
        slots=model_config.get("slots", model_config.get("latent_slots", data_config["seq_len"])),
    )

    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    best_summary: dict[str, Any] = {}
    best_analysis: dict[str, Any] = {}
    best_loss_summary: dict[str, Any] = {}

    for epoch in range(1, int(train_config["epochs"]) + 1):
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_config=train_config,
        )
        with torch.no_grad():
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scheduler=None,
                device=device,
                train_config=train_config,
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "parameter_count": parameter_count,
            "approx_flops_per_example": approx_flops,
        }
        logger.log(record)

        if (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["loss"] < best_summary.get("final_val_loss", float("inf"))
            )
        ):
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
            best_summary = {
                "run_name": run_name,
                "model_name": model_config["name"],
                "dataset_name": dataset_name,
                "seed": seed,
                "epoch": epoch,
                "final_train_loss": train_metrics["loss"],
                "final_val_loss": val_metrics["loss"],
                "final_val_accuracy": val_metrics["accuracy"],
                "final_val_f1": val_metrics["f1"],
                "parameter_count": parameter_count,
                "approx_flops_per_example": approx_flops,
                "avg_effective_depth": val_metrics["effective_depth"],
                "avg_active_slots": val_metrics["avg_active_slots"],
                "avg_active_think_slots": val_metrics["avg_active_think_slots"],
                "avg_step_gate": val_metrics["avg_step_gate"],
                "avg_slot_gate": val_metrics["avg_slot_gate"],
                "throughput": val_metrics["throughput"],
                "peak_gpu_memory_mb": val_metrics["peak_gpu_memory_mb"],
                "depth_difficulty_corr": val_metrics["depth_difficulty_corr"],
            }
            best_analysis = {
                "run_name": run_name,
                "sample_wise_depth_distribution": val_metrics["depth_distribution"],
                "slot_usage_histogram": val_metrics["slot_histogram"],
                "think_slot_usage_histogram": val_metrics["think_slot_histogram"],
                "avg_active_think_slots": val_metrics["avg_active_think_slots"],
                "avg_step_gate": val_metrics["avg_step_gate"],
                "avg_slot_gate": val_metrics["avg_slot_gate"],
                "depth_difficulty_corr": val_metrics["depth_difficulty_corr"],
            }
            write_json(run_dir / "best_accuracy_summary.json", best_summary)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_loss_summary = {
                "run_name": run_name,
                "model_name": model_config["name"],
                "dataset_name": dataset_name,
                "seed": seed,
                "epoch": epoch,
                "final_train_loss": train_metrics["loss"],
                "final_val_loss": val_metrics["loss"],
                "final_val_accuracy": val_metrics["accuracy"],
                "final_val_f1": val_metrics["f1"],
                "parameter_count": parameter_count,
                "approx_flops_per_example": approx_flops,
                "avg_effective_depth": val_metrics["effective_depth"],
                "avg_active_slots": val_metrics["avg_active_slots"],
                "avg_active_think_slots": val_metrics["avg_active_think_slots"],
                "avg_step_gate": val_metrics["avg_step_gate"],
                "avg_slot_gate": val_metrics["avg_slot_gate"],
                "throughput": val_metrics["throughput"],
                "peak_gpu_memory_mb": val_metrics["peak_gpu_memory_mb"],
                "depth_difficulty_corr": val_metrics["depth_difficulty_corr"],
            }

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"depth={val_metrics['effective_depth']:.3f} "
            f"slots={val_metrics['avg_active_slots']:.3f}"
        )

    write_json(run_dir / "summary.json", best_summary)
    write_json(run_dir / "analysis.json", best_analysis)
    if best_loss_summary:
        write_json(run_dir / "best_loss_summary.json", best_loss_summary)
    return TrainingArtifacts(run_dir=run_dir, summary=best_summary)
