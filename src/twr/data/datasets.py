from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import random
import re
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from datasets import DownloadConfig, load_dataset, load_from_disk
except ImportError:  # pragma: no cover
    load_dataset = None
    load_from_disk = None
    DownloadConfig = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None


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
    streaming: bool = False
    local_files_only: bool = False
    local_dataset_path: str | None = None


@dataclass
class LongBenchConfig:
    benchmark_name: str
    seq_len: int
    vocab_size: int
    batch_size: int
    train_size: int
    val_size: int
    num_classes: int
    split_seed: int = 7
    lowercase: bool = True
    use_e_variant: bool = False
    local_files_only: bool = False
    local_archive_path: str | None = None


@dataclass
class ListOpsConfig:
    vocab_size: int
    seq_len: int
    num_classes: int
    train_size: int
    val_size: int
    batch_size: int
    max_depth: int = 4
    max_args: int = 4


@dataclass
class RulerNeedleConfig:
    vocab_size: int
    seq_len: int
    num_classes: int
    train_size: int
    val_size: int
    batch_size: int
    num_pairs: int = 24
    num_keys: int = 32
    filler_span: int = 32


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = Path(os.environ.get("TWR_HF_CACHE_DIR", REPO_ROOT / "experiments" / "cache" / "huggingface"))
IMDB_ARROW_CACHE_DIR = Path(
    os.environ.get(
        "TWR_IMDB_ARROW_CACHE_DIR",
        REPO_ROOT / "experiments" / "cache" / "huggingface" / "imdb_arrow",
    )
)
LONGBENCH_CACHE_DIR = Path(os.environ.get("TWR_LONGBENCH_CACHE_DIR", HF_CACHE_DIR / "longbench"))

_NORMALIZE_WHITESPACE_RE = re.compile(r"\s+")


def _dataset_snapshot_dir() -> Path:
    path = REPO_ROOT / "experiments" / "cache" / "dataset_snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snapshot_key(parts: list[str]) -> str:
    joined = "||".join(parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=10).hexdigest()


def _load_snapshot(path: Path) -> dict[str, Tensor] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    return payload


def _save_snapshot(path: Path, payload: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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


def normalize_label(value: str) -> str:
    return _NORMALIZE_WHITESPACE_RE.sub(" ", value.strip()).lower()


def split_indices(size: int, train_size: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[:train_size], indices[train_size : train_size + val_size]


def build_difficulty_from_text(text: str, seq_len: int, lowercase: bool) -> float:
    effective_len = min(len(TOKEN_PATTERN.findall(text.lower() if lowercase else text)), seq_len)
    return float(effective_len / max(seq_len, 1))


def load_hf_rows(config: HuggingFaceTextConfig, split: str) -> list[dict[str, object]]:
    if load_dataset is None:
        raise ImportError("datasets is required for HuggingFaceTextDataset.")
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_rows = config.train_size if split == "train" else config.val_size
    split_name = config.train_split if split == "train" else config.val_split
    download_config = None
    if config.local_files_only:
        if DownloadConfig is None:
            raise ImportError("datasets DownloadConfig is required for local-files-only loading.")
        download_config = DownloadConfig(local_files_only=True)
    if config.local_dataset_path:
        if load_from_disk is None:
            raise ImportError("datasets load_from_disk is required for local dataset loading.")
        dataset = load_from_disk(config.local_dataset_path)
        if split_name not in dataset:
            raise ValueError(
                f"Split '{split_name}' not found in local dataset at {config.local_dataset_path}."
            )
        split_dataset = dataset[split_name]
        return list(split_dataset.select(range(min(target_rows, len(split_dataset)))))
    if config.dataset_name == "imdb" and IMDB_ARROW_CACHE_DIR.exists():
        dataset = load_dataset(
            "arrow",
            data_files={
                "train": str(IMDB_ARROW_CACHE_DIR / "imdb-train.arrow"),
                "test": str(IMDB_ARROW_CACHE_DIR / "imdb-test.arrow"),
            },
            split=f"{split_name}[:{target_rows}]",
            cache_dir=str(HF_CACHE_DIR),
            streaming=config.streaming,
            download_config=download_config,
        )
    else:
        dataset = load_dataset(
            config.dataset_name,
            split=split_name,
            cache_dir=str(HF_CACHE_DIR),
            streaming=config.streaming,
            download_config=download_config,
        )

    if config.streaming:
        return list(dataset.take(target_rows))
    return list(dataset.select(range(min(target_rows, len(dataset)))))


def read_longbench_rows(config: LongBenchConfig) -> list[dict[str, object]]:
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required for LongBenchDataset.")
    LONGBENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if config.local_archive_path:
        archive_path = str(Path(config.local_archive_path).expanduser().resolve())
    else:
        archive_path = hf_hub_download(
            repo_id="THUDM/LongBench",
            repo_type="dataset",
            filename="data.zip",
            local_dir=str(LONGBENCH_CACHE_DIR),
            local_files_only=config.local_files_only,
        )
    suffix = "_e" if config.use_e_variant else ""
    member_name = f"data/{config.benchmark_name}{suffix}.jsonl"
    with ZipFile(archive_path) as archive:
        raw_lines = archive.read(member_name).decode("utf-8").splitlines()
    return [json.loads(line) for line in raw_lines]


def build_longbench_splits(config: LongBenchConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_longbench_rows(config)
    train_indices, val_indices = split_indices(
        size=len(rows),
        train_size=min(config.train_size, len(rows)),
        val_size=min(config.val_size, max(len(rows) - config.train_size, 0)),
        seed=config.split_seed,
    )
    return [rows[index] for index in train_indices], [rows[index] for index in val_indices]


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
        snapshot_path = _dataset_snapshot_dir() / (
            f"hf_text_{_snapshot_key([
                config.dataset_name,
                split,
                config.text_field,
                config.label_field,
                str(config.seq_len),
                str(config.vocab_size),
                str(config.train_size),
                str(config.val_size),
                str(config.lowercase),
            ])}.pt"
        )
        snapshot = _load_snapshot(snapshot_path)
        if snapshot is not None:
            self.tokens = snapshot["tokens"]
            self.labels = snapshot["labels"]
            self.difficulty = snapshot["difficulty"]
            return
        dataset = load_hf_rows(config=config, split=split)

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
            labels.append(int(bool(row[config.label_field])) if isinstance(row[config.label_field], bool) else int(row[config.label_field]))
            difficulties.append(build_difficulty_from_text(text=text, seq_len=config.seq_len, lowercase=config.lowercase))
            tokens.append(token_tensor)

        self.tokens = torch.stack(tokens, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.difficulty = torch.tensor(difficulties, dtype=torch.float32).clamp(0.0, 1.0)
        _save_snapshot(
            snapshot_path,
            {
                "tokens": self.tokens,
                "labels": self.labels,
                "difficulty": self.difficulty,
            },
        )

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }


class LongBenchDataset(Dataset[dict[str, Tensor]]):
    """Hashed-token classification view over a LongBench task."""

    def __init__(self, split: str, config: LongBenchConfig) -> None:
        super().__init__()
        snapshot_path = _dataset_snapshot_dir() / (
            f"longbench_{_snapshot_key([
                config.benchmark_name,
                split,
                str(config.seq_len),
                str(config.vocab_size),
                str(config.train_size),
                str(config.val_size),
                str(config.split_seed),
                str(config.lowercase),
                str(config.use_e_variant),
            ])}.pt"
        )
        snapshot = _load_snapshot(snapshot_path)
        if snapshot is not None:
            self.tokens = snapshot["tokens"]
            self.labels = snapshot["labels"]
            self.difficulty = snapshot["difficulty"]
            return
        train_rows, val_rows = build_longbench_splits(config)
        rows = train_rows if split == "train" else val_rows
        label_space = train_rows[0]["all_classes"] or []
        if len(label_space) != config.num_classes:
            raise ValueError(
                f"Expected {config.num_classes} classes for LongBench task {config.benchmark_name}, "
                f"but found {len(label_space)}."
            )
        label_to_index = {normalize_label(str(label)): idx for idx, label in enumerate(label_space)}

        tokens: list[Tensor] = []
        labels: list[int] = []
        difficulties: list[float] = []
        for row in rows:
            prompt = (
                f"Task: {config.benchmark_name}\n"
                f"Question: {row['input']}\n"
                f"Context:\n{row['context']}"
            )
            answer = normalize_label(str(row["answers"][0]))
            if answer not in label_to_index:
                continue
            tokens.append(
                tokenize_to_tensor(
                    text=prompt,
                    seq_len=config.seq_len,
                    vocab_size=config.vocab_size,
                    lowercase=config.lowercase,
                )
            )
            labels.append(label_to_index[answer])
            difficulties.append(build_difficulty_from_text(text=prompt, seq_len=config.seq_len, lowercase=config.lowercase))

        self.tokens = torch.stack(tokens, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.difficulty = torch.tensor(difficulties, dtype=torch.float32).clamp(0.0, 1.0)
        _save_snapshot(
            snapshot_path,
            {
                "tokens": self.tokens,
                "labels": self.labels,
                "difficulty": self.difficulty,
            },
        )

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }


class ListOpsDataset(Dataset[dict[str, Tensor]]):
    """Synthetic ListOps-style task for long-range classification."""

    PAD = 0
    OPEN = 1
    CLOSE = 2
    OP_MIN = 3
    OP_MAX = 4
    OP_SUM = 5
    OP_MED = 6
    DIGIT_START = 7

    def __init__(self, size: int, config: ListOpsConfig, seed: int) -> None:
        super().__init__()
        generator = random.Random(seed)
        token_rows: list[Tensor] = []
        labels: list[int] = []
        difficulties: list[float] = []
        for _ in range(size):
            token_ids, value, depth = self._build_expression(generator, config, depth=0)
            clipped = token_ids[: config.seq_len]
            padded = clipped + [self.PAD] * max(config.seq_len - len(clipped), 0)
            token_rows.append(torch.tensor(padded, dtype=torch.long))
            labels.append(value)
            depth_ratio = depth / max(config.max_depth, 1)
            length_ratio = min(len(clipped) / max(config.seq_len, 1), 1.0)
            difficulties.append(float((depth_ratio + length_ratio) / 2.0))

        self.tokens = torch.stack(token_rows, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.difficulty = torch.tensor(difficulties, dtype=torch.float32).clamp(0.0, 1.0)

    def _build_expression(
        self,
        generator: random.Random,
        config: ListOpsConfig,
        depth: int,
    ) -> tuple[list[int], int, int]:
        stop = depth >= config.max_depth or (depth > 0 and generator.random() < 0.35)
        if stop:
            value = generator.randrange(config.num_classes)
            return [self.DIGIT_START + value], value, depth

        op_token = generator.choice([self.OP_MIN, self.OP_MAX, self.OP_SUM, self.OP_MED])
        arg_count = generator.randint(2, config.max_args)
        values: list[int] = []
        tokens = [self.OPEN, op_token]
        max_depth = depth
        for _ in range(arg_count):
            child_tokens, child_value, child_depth = self._build_expression(generator, config, depth + 1)
            tokens.extend(child_tokens)
            values.append(child_value)
            max_depth = max(max_depth, child_depth)
        tokens.append(self.CLOSE)
        if op_token == self.OP_MIN:
            result = min(values)
        elif op_token == self.OP_MAX:
            result = max(values)
        elif op_token == self.OP_SUM:
            result = sum(values) % config.num_classes
        else:
            result = sorted(values)[len(values) // 2]
        return tokens, result, max_depth

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "tokens": self.tokens[index],
            "labels": self.labels[index],
            "difficulty": self.difficulty[index],
        }


class RulerNeedleDataset(Dataset[dict[str, Tensor]]):
    """Synthetic RULER-style key-value retrieval benchmark."""

    PAD = 0
    QUERY = 1
    KEY_MARKER = 2
    VALUE_MARKER = 3

    def __init__(self, size: int, config: RulerNeedleConfig, seed: int) -> None:
        super().__init__()
        generator = random.Random(seed)
        token_rows: list[Tensor] = []
        labels: list[int] = []
        difficulties: list[float] = []
        filler_start = 4 + config.num_keys + config.num_classes
        for _ in range(size):
            mapping = {key: generator.randrange(config.num_classes) for key in range(config.num_keys)}
            selected_keys = generator.sample(range(config.num_keys), k=min(config.num_pairs, config.num_keys))
            query_key = generator.choice(selected_keys)
            sequence: list[int] = []
            for key in selected_keys:
                filler_len = generator.randint(1, config.filler_span)
                sequence.extend(generator.randint(filler_start, max(config.vocab_size - 1, filler_start)) for _ in range(filler_len))
                sequence.extend([self.KEY_MARKER, 4 + key, self.VALUE_MARKER, 4 + config.num_keys + mapping[key]])
            query_segment = [self.QUERY, 4 + query_key]
            budget = max(config.seq_len - len(query_segment), 0)
            clipped = (sequence[:budget] + query_segment)[: config.seq_len]
            padded = clipped + [self.PAD] * max(config.seq_len - len(clipped), 0)
            token_rows.append(torch.tensor(padded, dtype=torch.long))
            labels.append(mapping[query_key])
            difficulties.append(float(min(len(clipped) / max(config.seq_len, 1), 1.0)))

        self.tokens = torch.stack(token_rows, dim=0)
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
