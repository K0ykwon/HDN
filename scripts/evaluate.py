from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a stored experiment summary.")
    parser.add_argument("--run-dir", required=True, help="Path to an experiments/runs/<name> directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.run_dir) / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
