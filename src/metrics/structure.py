from __future__ import annotations

import math


def summarize_usage(usage_sums: list, usage_counts: list[int]) -> list[list[float]]:
    summary: list[list[float]] = []
    for stage_idx, stage_sum in enumerate(usage_sums):
        count = max(usage_counts[stage_idx], 1)
        stage_values = (stage_sum / count).detach().cpu().tolist()
        summary.append([float(value) for value in stage_values])
    return summary


def flatten_usage(usage_summary: list[list[float]]) -> list[float]:
    return [value for stage in usage_summary for value in stage]


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(max(value, 0.0) for value in values)
    total = sum(sorted_values)
    if total == 0.0:
        return 0.0
    weighted_sum = 0.0
    for index, value in enumerate(sorted_values, start=1):
        weighted_sum += index * value
    n = len(sorted_values)
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def dead_node_ratio(values: list[float], threshold: float = 1e-6) -> float:
    if not values:
        return 0.0
    dead = sum(1 for value in values if value <= threshold)
    return dead / len(values)


def busiest_node_load_share(values: list[float]) -> float:
    total = sum(values)
    if total == 0.0:
        return 0.0
    return max(values) / total


def average_used_depth(usage_summary: list[list[float]], threshold: float = 1e-3) -> float:
    if not usage_summary:
        return 0.0
    active = sum(1 for stage in usage_summary if stage and max(stage) > threshold)
    return active / len(usage_summary)
