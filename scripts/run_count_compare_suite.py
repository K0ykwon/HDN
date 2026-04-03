from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    "configs/experiment/twr_count_compare_adaptive.yaml",
    "configs/experiment/twr_count_compare_fixed.yaml",
    "configs/experiment/twr_count_compare_no_slot_gate.yaml",
    "configs/experiment/twr_count_compare_deeper.yaml",
    "configs/experiment/transformer_count_compare.yaml",
    "configs/experiment/mamba_placeholder_count_compare.yaml",
]


def load_summary(run_name: str) -> dict[str, object]:
    summary_path = ROOT / "experiments" / "runs" / run_name / "summary.json"
    analysis_path = ROOT / "experiments" / "runs" / run_name / "analysis.json"
    summary = json.loads(summary_path.read_text())
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        summary["analysis"] = {
            "depth_difficulty_corr": analysis.get("depth_difficulty_corr"),
            "avg_active_think_slots": analysis.get("avg_active_think_slots"),
        }
    return summary


def main() -> None:
    suite_dir = ROOT / "experiments" / "validation" / "count_compare_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        completed = subprocess.run(
            [sys.executable, "scripts/train.py", "--experiment", experiment],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        experiment_payload = yaml.safe_load((ROOT / experiment).read_text(encoding="utf-8"))
        run_name = experiment_payload.get("run", {}).get("name", Path(experiment).stem)
        entry: dict[str, object] = {
            "experiment": experiment,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode == 0:
            entry["summary"] = load_summary(run_name)
        runs.append(entry)

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "runs": runs,
        "all_passed": all(run["returncode"] == 0 for run in runs),
    }

    (suite_dir / "suite_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (suite_dir / "suite_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Count compare suite timestamp: {report['timestamp_utc']}\n")
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
                    f"avg_active_think_slots={summary.get('avg_active_think_slots')}, "
                    f"depth_corr={summary.get('depth_difficulty_corr')}\n"
                )
            handle.write("\n")

    print(f"Saved suite report to {suite_dir}")
    print(f"All passed: {report['all_passed']}")


if __name__ == "__main__":
    main()
