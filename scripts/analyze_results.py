from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
ANALYSIS_DIR = ROOT / "experiments" / "analysis"


def infer_group(run_name: str) -> str:
    if run_name.startswith("smoke_"):
        return "smoke"
    if "hyperpartisan" in run_name:
        return "hyperpartisan"
    if "longbench" in run_name:
        return "longbench"
    if "ruler" in run_name:
        return "ruler"
    if "lra" in run_name:
        return "lra"
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


def plot_training_curves() -> None:
    for metrics_path in sorted(RUNS_DIR.glob("*/metrics.jsonl")):
        run_dir = metrics_path.parent
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            continue
        epochs = [record["epoch"] for record in records]
        train_loss = [record["train"]["loss"] for record in records]
        val_loss = [record["val"]["loss"] for record in records]
        train_acc = [record["train"]["accuracy"] for record in records]
        val_acc = [record["val"]["accuracy"] for record in records]
        val_depth = [record["val"]["effective_depth"] for record in records]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, marker="o", label="train")
        axes[0].plot(epochs, val_loss, marker="o", label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, marker="o", label="train acc")
        axes[1].plot(epochs, val_acc, marker="o", label="val acc")
        axes[1].plot(epochs, val_depth, marker="o", label="val depth")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Accuracy / Depth")
        axes[1].legend()

        fig.suptitle(run_dir.name)
        fig.tight_layout()
        fig.savefig(run_dir / "training_curves.png", dpi=180)
        plt.close(fig)


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
        if row["group"] in {
            "count_compare",
            "count_compare_long",
            "imdb_debug",
            "lra",
            "ruler",
            "longbench",
            "hyperpartisan",
        }
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "count_compare": "#2C7FB8",
        "count_compare_long": "#41AB5D",
        "imdb_debug": "#D95F0E",
        "lra": "#6A3D9A",
        "ruler": "#E31A1C",
        "longbench": "#1B9E77",
        "hyperpartisan": "#7570B3",
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


def plot_benchmark_summary(rows: list[dict[str, Any]]) -> None:
    benchmark_rows = [
        row for row in rows if row["group"] in {"lra", "ruler", "longbench", "hyperpartisan"}
    ]
    if not benchmark_rows:
        return

    benchmark_rows = sorted(benchmark_rows, key=lambda row: (row["group"], row["model_name"]))
    labels = [row["run_name"] for row in benchmark_rows]
    scores = [row["final_val_accuracy"] for row in benchmark_rows]
    params_m = [row["parameter_count"] / 1_000_000 for row in benchmark_rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - width / 2, scores, width=width, color="#2C7FB8", label="val acc")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, params_m, width=width, color="#D95F0E", label="params (M)")
    ax2.set_ylabel("Parameters (millions)")

    ax1.set_title("Benchmark Accuracy and Parameter Count")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "benchmark_accuracy_and_params.png", dpi=180)
    plt.close(fig)


def render_results_table(rows: list[dict[str, Any]]) -> None:
    benchmark_rows = [
        row for row in rows if row["group"] in {"lra", "ruler", "longbench", "hyperpartisan"}
    ]
    if not benchmark_rows:
        return

    benchmark_rows = sorted(benchmark_rows, key=lambda row: (row["group"], row["run_name"]))
    table_rows = [
        [
            row["run_name"],
            row["group"],
            f"{row['final_val_accuracy']:.3f}",
            f"{row['final_val_loss']:.3f}",
            f"{row['avg_effective_depth']:.2f}",
            f"{row['parameter_count']:,}",
        ]
        for row in benchmark_rows
    ]
    fig, ax = plt.subplots(figsize=(13, max(3.5, 0.45 * len(table_rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["run", "benchmark", "acc", "loss", "depth", "params"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title("Benchmark Results")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "benchmark_results_table.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = load_runs()
    write_csv(rows)
    write_json(rows)
    write_markdown(rows)
    plot_training_curves()
    plot_group(rows, "parity_baselines", "parity_baselines.png", "Parity Debug Baselines")
    plot_group(rows, "parity_ablations", "parity_ablations.png", "Parity Ablations")
    plot_group(rows, "count_compare", "count_compare.png", "Count Compare Results")
    plot_group(rows, "count_compare_long", "count_compare_long.png", "Long Count Compare Results")
    plot_group(rows, "imdb_debug", "imdb_debug.png", "IMDB Long Debug Results")
    plot_group(rows, "lra", "lra.png", "LRA ListOps Results")
    plot_group(rows, "ruler", "ruler.png", "RULER Needle Results")
    plot_group(rows, "longbench", "longbench.png", "LongBench TREC Results")
    plot_group(rows, "hyperpartisan", "hyperpartisan.png", "Hyperpartisan Results")
    plot_efficiency(rows)
    plot_benchmark_summary(rows)
    render_results_table(rows)
    print(f"Wrote analysis artifacts to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
