from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class DataBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    input_shape: tuple[int, ...]
    num_classes: int
    dataset_name: str


def build_dataloaders(dataset_config: dict, batch_size: int, seed: int) -> DataBundle:
    dataset_name = dataset_config["name"]
    dataset_type = dataset_config.get("type", "toy")

    if dataset_type == "toy":
        return _build_toy_bundle(dataset_config, batch_size, seed)
    if dataset_type == "torchvision":
        return _build_torchvision_bundle(dataset_config, batch_size, seed)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _build_toy_bundle(dataset_config: dict, batch_size: int, seed: int) -> DataBundle:
    dataset_name = dataset_config["name"]
    n_samples = int(dataset_config.get("n_samples", 512))
    noise = float(dataset_config.get("noise", 0.05))
    train_split = float(dataset_config.get("train_split", 0.8))

    if dataset_name == "xor":
        features, labels = _make_xor(n_samples, noise, seed)
    elif dataset_name == "two_moons":
        features, labels = _make_two_moons(n_samples, noise, seed)
    elif dataset_name == "concentric_circles":
        features, labels = _make_concentric_circles(n_samples, noise, seed)
    else:
        raise ValueError(f"Unsupported toy dataset: {dataset_name}")

    dataset = TensorDataset(features, labels)
    train_dataset, valid_dataset = _split_dataset(dataset, train_split, seed)
    train_loader, valid_loader = _make_loaders(train_dataset, valid_dataset, batch_size, seed)
    num_classes = int(labels.max().item()) + 1

    return DataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_shape=tuple(features.shape[1:]),
        num_classes=num_classes,
        dataset_name=dataset_name,
    )


def _build_torchvision_bundle(dataset_config: dict, batch_size: int, seed: int) -> DataBundle:
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for MNIST/CIFAR experiments. Install project dependencies first."
        ) from exc

    dataset_name = dataset_config["name"].lower()
    root = dataset_config.get("root", "./data")
    train_split = float(dataset_config.get("train_split", 0.9))

    normalize = dataset_config.get("normalize", {})
    transform_steps = [transforms.ToTensor()]
    if normalize:
        mean = normalize.get("mean")
        std = normalize.get("std")
        if mean is not None and std is not None:
            transform_steps.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_steps)

    dataset_cls_map = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    if dataset_name not in dataset_cls_map:
        raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")

    dataset_cls = dataset_cls_map[dataset_name]
    train_dataset_full = dataset_cls(root=root, train=True, download=True, transform=transform)
    valid_dataset_source = dataset_cls(root=root, train=False, download=True, transform=transform)

    train_dataset, train_holdout = _split_dataset(train_dataset_full, train_split, seed)
    if dataset_config.get("use_test_as_valid", True):
        valid_dataset = valid_dataset_source
    else:
        valid_dataset = train_holdout

    train_loader, valid_loader = _make_loaders(train_dataset, valid_dataset, batch_size, seed)
    sample_features, sample_labels = next(iter(valid_loader))
    num_classes = int(sample_labels.max().item()) + 1

    return DataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_shape=tuple(sample_features.shape[1:]),
        num_classes=num_classes,
        dataset_name=dataset_name,
    )


def _split_dataset(dataset, train_split: float, seed: int):
    train_size = int(len(dataset) * train_split)
    valid_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, valid_size], generator=generator)


def _make_loaders(train_dataset, valid_dataset, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def _make_xor(n_samples: int, noise: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 2, (n_samples, 2), generator=generator).float()
    jitter = torch.randn((n_samples, 2), generator=generator) * noise
    features = (base * 2.0 - 1.0) + jitter
    labels = ((base[:, 0] + base[:, 1]) % 2).long()
    return features, labels


def _make_two_moons(n_samples: int, noise: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    first_count = n_samples // 2
    second_count = n_samples - first_count

    theta_first = torch.rand(first_count, generator=generator) * math.pi
    theta_second = torch.rand(second_count, generator=generator) * math.pi

    first = torch.stack([torch.cos(theta_first), torch.sin(theta_first)], dim=1)
    second = torch.stack([1.0 - torch.cos(theta_second), -torch.sin(theta_second) - 0.5], dim=1)

    features = torch.cat([first, second], dim=0)
    labels = torch.cat(
        [torch.zeros(first_count, dtype=torch.long), torch.ones(second_count, dtype=torch.long)],
        dim=0,
    )

    permutation = torch.randperm(n_samples, generator=generator)
    features = features[permutation]
    labels = labels[permutation]
    features = features + torch.randn(features.shape, generator=generator) * noise
    return features, labels


def _make_concentric_circles(n_samples: int, noise: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    inner_count = n_samples // 2
    outer_count = n_samples - inner_count

    theta_inner = torch.rand(inner_count, generator=generator) * (2.0 * math.pi)
    theta_outer = torch.rand(outer_count, generator=generator) * (2.0 * math.pi)

    inner_radius = 0.5 + torch.randn(inner_count, generator=generator) * noise
    outer_radius = 1.0 + torch.randn(outer_count, generator=generator) * noise

    inner = torch.stack([inner_radius * torch.cos(theta_inner), inner_radius * torch.sin(theta_inner)], dim=1)
    outer = torch.stack([outer_radius * torch.cos(theta_outer), outer_radius * torch.sin(theta_outer)], dim=1)

    features = torch.cat([inner, outer], dim=0)
    labels = torch.cat(
        [torch.zeros(inner_count, dtype=torch.long), torch.ones(outer_count, dtype=torch.long)],
        dim=0,
    )

    permutation = torch.randperm(n_samples, generator=generator)
    return features[permutation], labels[permutation]
