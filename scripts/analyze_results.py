from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
ANALYSIS_DIR = ROOT / "experiments" / "analysis"


def infer_group(run_name: str) -> str:
    if "imdb" in run_name:
        return "imdb_debug"
    if "long_" in run_name:
        return "count_compare_long"
    if "count_compare" in run_name:
        return "count_compare"
    if run_name in {
        "twr_debug",
        "transformer_debug",
        "perceiver_debug",
        "mamba_placeholder_debug",
    }:
        return "parity_baselines"
    return "parity_ablations"


def load_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(RUNS_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        run_name = run_dir.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        analysis_path = run_dir / "analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
        rows.append(
            {
                "run_name": run_name,
                "group": infer_group(run_name),
                "model_name": summary.get("model_name"),
                "dataset_name": summary.get("dataset_name"),
                "final_val_accuracy": float(summary.get("final_val_accuracy", 0.0)),
                "final_val_f1": float(summary.get("final_val_f1", 0.0)),
                "final_val_loss": float(summary.get("final_val_loss", 0.0)),
                "avg_effective_depth": float(summary.get("avg_effective_depth", 0.0)),
                "avg_active_slots": float(summary.get("avg_active_slots", 0.0)),
                "avg_active_think_slots": float(summary.get("avg_active_think_slots", 0.0)),
                "avg_step_gate": float(summary.get("avg_step_gate", 0.0)),
                "avg_slot_gate": float(summary.get("avg_slot_gate", 0.0)),
                "throughput": float(summary.get("throughput", 0.0)),
                "peak_gpu_memory_mb": float(summary.get("peak_gpu_memory_mb", 0.0)),
                "parameter_count": int(summary.get("parameter_count", 0)),
                "approx_flops_per_example": int(summary.get("approx_flops_per_example", 0)),
                "depth_difficulty_corr": float(summary.get("depth_difficulty_corr", 0.0)),
                "slot_usage_histogram": analysis.get("slot_usage_histogram", []),
                "think_slot_usage_histogram": analysis.get("think_slot_usage_histogram", []),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "run_summary.csv"
    fieldnames = [
        "run_name",
        "group",
        "model_name",
        "dataset_name",
        "final_val_accuracy",
        "final_val_f1",
        "final_val_loss",
        "avg_effective_depth",
        "avg_active_slots",
        "avg_active_think_slots",
        "avg_step_gate",
        "avg_slot_gate",
        "throughput",
        "peak_gpu_memory_mb",
        "parameter_count",
        "approx_flops_per_example",
        "depth_difficulty_corr",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_json(rows: list[dict[str, Any]]) -> None:
    payload = {
        "runs": rows,
        "best_by_group": {
            group: max(group_rows, key=lambda row: row["final_val_accuracy"])["run_name"]
            for group in sorted({row["group"] for row in rows})
            for group_rows in [[row for row in rows if row["group"] == group]]
        },
    }
    (ANALYSIS_DIR / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = ["# Experiment Summary", ""]
    for group in sorted({row["group"] for row in rows}):
        group_rows = sorted(
            [row for row in rows if row["group"] == group],
            key=lambda row: (-row["final_val_accuracy"], row["final_val_loss"]),
        )
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| run | acc | loss | depth | throughput | depth corr |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in group_rows:
            lines.append(
                f"| {row['run_name']} | {row['final_val_accuracy']:.4f} | "
                f"{row['final_val_loss']:.4f} | {row['avg_effective_depth']:.3f} | "
                f"{row['throughput']:.1f} | {row['depth_difficulty_corr']:.4f} |"
            )
        lines.append("")
    (ANALYSIS_DIR / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_group(rows: list[dict[str, Any]], group: str, filename: str, title: str) -> None:
    group_rows = sorted(
        [row for row in rows if row["group"] == group],
        key=lambda row: row["final_val_accuracy"],
    )
    if not group_rows:
        return

    labels = [row["run_name"] for row in group_rows]
    accuracy = [row["final_val_accuracy"] for row in group_rows]
    depth = [row["avg_effective_depth"] for row in group_rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(labels, accuracy, color="#2C7FB8")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title(title)
    ax1.tick_params(axis="x", rotation=35, labelsize=9)

    ax2 = ax1.twinx()
    ax2.plot(labels, depth, color="#D95F0E", marker="o", linewidth=2)
    ax2.set_ylabel("Average Effective Depth")

    for bar, value in zip(bars, accuracy, strict=True):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / filename, dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in rows
        if row["group"] in {"count_compare", "count_compare_long", "imdb_debug"}
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "count_compare": "#2C7FB8",
        "count_compare_long": "#41AB5D",
        "imdb_debug": "#D95F0E",
    }
    for row in selected:
        ax.scatter(
            row["throughput"],
            row["final_val_accuracy"],
            s=90,
            color=colors[row["group"]],
        )
        ax.annotate(row["run_name"], (row["throughput"], row["final_val_accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Throughput (examples/sec)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy vs Throughput")
    ax.set_ylim(0.6, 1.02)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "accuracy_vs_throughput.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_runs()
    write_csv(rows)
    write_json(rows)
    write_markdown(rows)
    plot_group(rows, "parity_baselines", "parity_baselines.png", "Parity Debug Baselines")
    plot_group(rows, "parity_ablations", "parity_ablations.png", "Parity Ablations")
    plot_group(rows, "count_compare", "count_compare.png", "Count Compare Results")
    plot_group(rows, "count_compare_long", "count_compare_long.png", "Long Count Compare Results")
    plot_group(rows, "imdb_debug", "imdb_debug.png", "IMDB Long Debug Results")
    plot_efficiency(rows)
    print(f"Wrote analysis artifacts to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
