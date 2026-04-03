from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    "configs/experiment/twr_imdb_long_debug.yaml",
    "configs/experiment/transformer_imdb_long_debug.yaml",
]


def main() -> None:
    out_dir = ROOT / "experiments" / "validation" / "imdb_long_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        completed = subprocess.run(
            [sys.executable, "scripts/train.py", "--experiment", experiment],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        run_name = Path(experiment).stem
        entry: dict[str, object] = {
            "experiment": experiment,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode == 0:
            run_name = "twr_imdb_long_debug" if "twr_" in run_name else "transformer_imdb_long_debug"
            summary = json.loads((ROOT / "experiments" / "runs" / run_name / "summary.json").read_text())
            analysis_path = ROOT / "experiments" / "runs" / run_name / "analysis.json"
            entry["summary"] = summary
            if analysis_path.exists():
                entry["analysis"] = json.loads(analysis_path.read_text())
        runs.append(entry)

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "runs": runs,
        "all_passed": all(run["returncode"] == 0 for run in runs),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "report.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"IMDB long debug timestamp: {report['timestamp_utc']}\n")
        handle.write(f"All passed: {report['all_passed']}\n\n")
        for run in runs:
            handle.write(f"[{run['experiment']}]\n")
            handle.write(f"returncode: {run['returncode']}\n")
            summary = run.get("summary")
            if isinstance(summary, dict):
                handle.write(
                    "summary: "
                    f"val_acc={summary.get('final_val_accuracy')}, "
                    f"avg_depth={summary.get('avg_effective_depth')}, "
                    f"depth_corr={summary.get('depth_difficulty_corr')}\n"
                )
            handle.write("\n")
    print(f"Saved IMDB report to {out_dir}")
    print(f"All passed: {report['all_passed']}")


if __name__ == "__main__":
    main()
