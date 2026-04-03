from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None


@dataclass
class SyntheticSequenceConfig:
    vocab_size: int
    seq_len: int
    num_classes: int
    train_size: int
    val_size: int
    batch_size: int
    task: str = "special_token_parity"
    special_token: int = 1
    compare_token_a: int = 1
    compare_token_b: int = 2


@dataclass
class HuggingFaceTextConfig:
    dataset_name: str
    text_field: str
    label_field: str
    seq_len: int
    vocab_size: int
    batch_size: int
    train_size: int
    val_size: int
    num_classes: int
    train_split: str = "train"
    val_split: str = "test"
    lowercase: bool = True


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = Path(os.environ.get("TWR_HF_CACHE_DIR", REPO_ROOT / "experiments" / "cache" / "huggingface"))
IMDB_ARROW_CACHE_DIR = Path(
    os.environ.get(
        "TWR_IMDB_ARROW_CACHE_DIR",
        REPO_ROOT / "experiments" / "cache" / "huggingface" / "imdb_arrow",
    )
)


def hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big")
    return (value % max(vocab_size - 1, 1)) + 1


def tokenize_to_tensor(text: str, seq_len: int, vocab_size: int, lowercase: bool) -> Tensor:
    normalized = text.lower() if lowercase else text
    pieces = TOKEN_PATTERN.findall(normalized)
    token_ids = [hash_token(piece, vocab_size=vocab_size) for piece in pieces[:seq_len]]
    if len(token_ids) < seq_len:
        token_ids.extend([0] * (seq_len - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long)


class SyntheticSequenceDataset(Dataset[dict[str, Tensor]]):
    """Deterministic synthetic sequence classification dataset."""

    def __init__(self, size: int, config: SyntheticSequenceConfig, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(size, config.seq_len),
            generator=generator,
        )
        if config.task == "special_token_parity":
            counts = (tokens == config.special_token).sum(dim=1)
            labels = torch.remainder(counts, config.num_classes)
            difficulty = counts.float() / max(config.seq_len, 1)
        elif config.task == "count_compare":
            counts_a = (tokens == config.compare_token_a).sum(dim=1)
            counts_b = (tokens == config.compare_token_b).sum(dim=1)
            labels = (counts_a > counts_b).long()
            margin = (counts_a - counts_b).abs().float()
            difficulty = 1.0 - margin / max(config.seq_len, 1)
        else:
            raise ValueError(f"Unsupported synthetic task: {config.task}")

        self.tokens = tokens
        self.labels = labels.long()
        self.difficulty = difficulty.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }


class HuggingFaceTextDataset(Dataset[dict[str, Tensor]]):
    """Fixed-length hashed-token dataset built from a Hugging Face text classification corpus."""

    def __init__(self, split: str, config: HuggingFaceTextConfig) -> None:
        super().__init__()
        if load_dataset is None:
            raise ImportError("datasets is required for HuggingFaceTextDataset.")
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target_rows = config.train_size if split == "train" else config.val_size
        split_name = config.train_split if split == "train" else config.val_split
        if config.dataset_name == "imdb" and IMDB_ARROW_CACHE_DIR.exists():
            dataset = load_dataset(
                "arrow",
                data_files={
                    "train": str(IMDB_ARROW_CACHE_DIR / "imdb-train.arrow"),
                    "test": str(IMDB_ARROW_CACHE_DIR / "imdb-test.arrow"),
                },
                split=f"{split_name}[:{target_rows}]",
                cache_dir=str(HF_CACHE_DIR),
            )
        else:
            dataset = load_dataset(
                config.dataset_name,
                split=f"{split_name}[:{target_rows}]",
                cache_dir=str(HF_CACHE_DIR),
            )

        tokens: list[Tensor] = []
        labels: list[int] = []
        difficulties: list[float] = []
        for row in dataset:
            text = str(row[config.text_field])
            token_tensor = tokenize_to_tensor(
                text=text,
                seq_len=config.seq_len,
                vocab_size=config.vocab_size,
                lowercase=config.lowercase,
            )
            labels.append(int(row[config.label_field]))
            effective_len = min(len(TOKEN_PATTERN.findall(text.lower() if config.lowercase else text)), config.seq_len)
            difficulties.append(effective_len / max(config.seq_len, 1))
            tokens.append(token_tensor)

        self.tokens = torch.stack(tokens, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.difficulty = torch.tensor(difficulties, dtype=torch.float32).clamp(0.0, 1.0)

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }
