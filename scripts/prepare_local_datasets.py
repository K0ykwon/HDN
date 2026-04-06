from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twr.data.datasets import HF_CACHE_DIR, LONGBENCH_CACHE_DIR

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None


def prepare_hyperpartisan() -> None:
    if load_dataset is None:
        raise ImportError("datasets is required to prepare local Hugging Face datasets.")
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        dataset = load_dataset(
            "pietrolesci/hyperpartisan_news_detection",
            split=split,
            cache_dir=str(HF_CACHE_DIR),
            streaming=False,
        )
        print(f"cached hyperpartisan split={split} rows={len(dataset)} cache_dir={HF_CACHE_DIR}")


def prepare_longbench() -> None:
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required to prepare LongBench locally.")
    LONGBENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        repo_type="dataset",
        filename="data.zip",
        local_dir=str(LONGBENCH_CACHE_DIR),
        local_files_only=False,
    )
    print(f"cached longbench archive={archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into local cache.")
    parser.add_argument(
        "--dataset",
        choices=("all", "hyperpartisan", "longbench"),
        default="all",
        help="Which dataset cache to prepare locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"all", "hyperpartisan"}:
        prepare_hyperpartisan()
    if args.dataset in {"all", "longbench"}:
        prepare_longbench()


if __name__ == "__main__":
    main()
