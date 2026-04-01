from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.data.datasets import DataBundle, build_dataloaders
from src.metrics.structure import (
    average_used_depth,
    busiest_node_load_share,
    coefficient_of_variation,
    dead_node_ratio,
    flatten_usage,
    gini,
    summarize_usage,
)
from src.models.hdn import HDNPrototype
from src.models.mlp import BaselineMLP, ForwardResult
from src.utils.experiments import build_ablation_variants
from src.utils.logging import make_run_dir, write_json


@dataclass
class TrainingArtifacts:
    run_dir: str
    metrics: dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config: dict) -> TrainingArtifacts:
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_bundle = build_dataloaders(
        dataset_config=config["dataset"],
        batch_size=int(config["training"]["batch_size"]),
        seed=seed,
    )

    model_name = config.get("model_name", "baseline_mlp")
    model = _build_model(config, data_bundle)
    optimizer = _build_optimizer(config, model)
    run_dir = make_run_dir(config["logging"]["output_dir"], config["experiment_name"])

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "balance_cv": [],
        "balance_gini": [],
        "dead_node_ratio": [],
        "busiest_node_load_share": [],
        "average_used_depth": [],
        "parameter_count": [],
    }

    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, data_bundle.train_loader, optimizer)
        valid_metrics = _evaluate(model, data_bundle.valid_loader)
        usage_summary = summarize_usage(train_metrics["usage_sums"], train_metrics["usage_counts"])
        flat_usage = flatten_usage(usage_summary)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        history["balance_cv"].append(coefficient_of_variation(flat_usage))
        history["balance_gini"].append(gini(flat_usage))
        history["dead_node_ratio"].append(dead_node_ratio(flat_usage))
        history["busiest_node_load_share"].append(busiest_node_load_share(flat_usage))
        history["average_used_depth"].append(average_used_depth(usage_summary))
        history["parameter_count"].append(sum(parameter.numel() for parameter in model.parameters()))

        if isinstance(model, HDNPrototype):
            mutated = model.maybe_adapt(
                step=epoch,
                train_loss=train_metrics["loss"],
                valid_loss=valid_metrics["loss"],
                usage_summary=usage_summary,
            )
            if mutated:
                optimizer = _build_optimizer(config, model)

    results = _build_results(config, model_name, seed, history, model, data_bundle)
    evaluation = _run_requested_evaluations(config, model, data_bundle)
    if evaluation:
        results["evaluation"] = evaluation

    write_json(run_dir / "metrics.json", results)
    write_json(run_dir / "config.json", config)
    return TrainingArtifacts(run_dir=str(run_dir), metrics=results)


def run_study(config: dict) -> dict:
    study_results = {"main": None, "ablations": []}
    main_artifacts = run_experiment(config)
    study_results["main"] = main_artifacts.metrics

    for variant_config in build_ablation_variants(config):
        artifacts = run_experiment(variant_config)
        study_results["ablations"].append(artifacts.metrics)

    output_dir = config.get("logging", {}).get("output_dir", "logs")
    run_dir = make_run_dir(output_dir, f"{config['experiment_name']}_study")
    write_json(run_dir / "study_results.json", study_results)
    return {"run_dir": str(run_dir), "study_results": study_results}


def _build_model(config: dict, data_bundle: DataBundle):
    model_config = copy.deepcopy(config["model"])
    model_config.setdefault("input_shape", data_bundle.input_shape)
    model_config.setdefault("output_dim", data_bundle.num_classes)
    model_name = config.get("model_name", "baseline_mlp")

    if model_name == "hdn_prototype":
        return HDNPrototype(**model_config)
    return BaselineMLP(**model_config)


def _build_optimizer(config: dict, model):
    training_config = config["training"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )


def _run_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    usage_sums: list[torch.Tensor] | None = None
    usage_counts: list[int] | None = None

    for features, labels in dataloader:
        optimizer.zero_grad()
        forward_result = model(features, collect_stats=True)
        logits = forward_result.logits
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_usage = [activation.detach().abs().sum(dim=0) for activation in forward_result.hidden_activations]
        if usage_sums is None:
            usage_sums = [stage_sum.clone() for stage_sum in batch_usage]
            usage_counts = [features.size(0) for _ in batch_usage]
        else:
            for index, stage_sum in enumerate(batch_usage):
                usage_sums[index] += stage_sum
                usage_counts[index] += features.size(0)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    if usage_sums is None or usage_counts is None:
        usage_sums = []
        usage_counts = []

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "usage_sums": usage_sums,
        "usage_counts": usage_counts,
    }


def _evaluate(model, dataloader, perturbation: dict | None = None, neuron_ablation: dict[int, list[int]] | None = None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            if perturbation:
                features = _apply_input_perturbation(features, perturbation)
            forward_result = model(features, collect_stats=True, neuron_ablation=neuron_ablation)
            logits = forward_result.logits
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return {"loss": total_loss / max(total_examples, 1), "accuracy": total_correct / max(total_examples, 1)}


def _apply_input_perturbation(features: torch.Tensor, perturbation: dict) -> torch.Tensor:
    kind = perturbation.get("type", "gaussian_noise")
    if kind == "gaussian_noise":
        std = float(perturbation.get("std", 0.1))
        return features + torch.randn_like(features) * std
    if kind == "feature_dropout":
        probability = float(perturbation.get("probability", 0.1))
        mask = (torch.rand_like(features) > probability).float()
        return features * mask
    raise ValueError(f"Unsupported perturbation type: {kind}")


def _build_results(config: dict, model_name: str, seed: int, history: dict, model, data_bundle: DataBundle) -> dict:
    final_metrics = {key: values[-1] for key, values in history.items() if values}
    results = {
        "experiment_name": config["experiment_name"],
        "model_name": model_name,
        "seed": seed,
        "dataset": data_bundle.dataset_name,
        "history": history,
        "final_metrics": final_metrics,
        "structural_events": [
            {
                "step": event.step,
                "event_type": event.event_type,
                "reason": event.reason,
                "metadata": event.metadata,
            }
            for event in getattr(model, "structural_events", [])
        ],
    }
    return results


def _run_requested_evaluations(config: dict, model, data_bundle: DataBundle) -> dict:
    evaluation_config = config.get("evaluation", {})
    if not evaluation_config:
        return {}

    results: dict = {}
    baseline_metrics = _evaluate(model, data_bundle.valid_loader)
    usage_summary = getattr(model, "latest_usage_summary", [])
    busiest_ablation = _busiest_neuron_ablation(usage_summary)
    random_ablation = _random_neuron_ablation(usage_summary, seed=int(config.get("seed", 0)))

    if evaluation_config.get("corruptions"):
        corruption_results = []
        for perturbation in evaluation_config["corruptions"]:
            metrics = _evaluate(model, data_bundle.valid_loader, perturbation=perturbation)
            corruption_results.append({"perturbation": perturbation, "metrics": metrics})
        results["corruptions"] = corruption_results

    if evaluation_config.get("busiest_node_ablation", False) and busiest_ablation:
        metrics = _evaluate(model, data_bundle.valid_loader, neuron_ablation=busiest_ablation)
        results["busiest_node_ablation"] = {
            "target": busiest_ablation,
            "metrics": metrics,
            "accuracy_drop": baseline_metrics["accuracy"] - metrics["accuracy"],
        }

    if evaluation_config.get("random_node_ablation", False) and random_ablation:
        metrics = _evaluate(model, data_bundle.valid_loader, neuron_ablation=random_ablation)
        results["random_node_ablation"] = {
            "target": random_ablation,
            "metrics": metrics,
            "accuracy_drop": baseline_metrics["accuracy"] - metrics["accuracy"],
        }

    return results


def _busiest_neuron_ablation(usage_summary: list[list[float]]) -> dict[int, list[int]]:
    best_stage = None
    best_index = None
    best_value = -1.0
    for stage_idx, stage_usage in enumerate(usage_summary):
        for neuron_idx, value in enumerate(stage_usage):
            if value > best_value:
                best_value = value
                best_stage = stage_idx
                best_index = neuron_idx
    if best_stage is None or best_index is None:
        return {}
    return {best_stage: [best_index]}


def _random_neuron_ablation(usage_summary: list[list[float]], seed: int) -> dict[int, list[int]]:
    candidates = [(stage_idx, neuron_idx) for stage_idx, stage in enumerate(usage_summary) for neuron_idx in range(len(stage))]
    if not candidates:
        return {}
    rng = random.Random(seed)
    stage_idx, neuron_idx = rng.choice(candidates)
    return {stage_idx: [neuron_idx]}
