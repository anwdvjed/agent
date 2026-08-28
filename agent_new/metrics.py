"""Evaluation helpers that respect trajectory-level dependence."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return sum(items) / len(items) if items else None


def binary_auc(targets: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Compute ROC AUC in O(n log n), including average ranks for ties."""

    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    positives = sum(int(value) == 1 for value in targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: float(item[1]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and float(ordered[end][1]) == float(ordered[index][1]):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, target in zip(ranks, targets) if int(target) == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    targets: Sequence[int], scores: Sequence[float], bins: int = 10
) -> Optional[float]:
    if not targets:
        return None
    if len(targets) != len(scores) or bins <= 0:
        raise ValueError("invalid ECE inputs")
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, score in enumerate(scores)
            if lower <= float(score) < upper
            or (bin_index == bins - 1 and float(score) == 1.0)
        ]
        if not members:
            continue
        confidence = sum(float(scores[index]) for index in members) / len(members)
        frequency = sum(int(targets[index]) for index in members) / len(members)
        error += len(members) / len(targets) * abs(confidence - frequency)
    return error


def trajectory_outcomes(
    records: Sequence[Dict[str, Any]], threshold: float
) -> Dict[str, Optional[float]]:
    """Measure early warnings and false alerts using whole trajectories."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["trajectory_id"])].append(record)
    unsafe = safe = warned = false_alerts = 0
    leads: List[float] = []
    for trajectory in grouped.values():
        trajectory.sort(key=lambda item: int(item["step_index"]))
        violations = [int(item["step_index"]) for item in trajectory if item.get("current_violation")]
        alerts = [
            int(item["step_index"])
            for item in trajectory
            if float(item["risk"]) > threshold
        ]
        if violations:
            unsafe += 1
            first = min(violations)
            earlier = [step for step in alerts if step < first]
            if earlier:
                warned += 1
                leads.append(float(first - min(earlier)))
        else:
            safe += 1
            false_alerts += int(bool(alerts))
    return {
        "unsafe_trajectories": float(unsafe),
        "safe_trajectories": float(safe),
        "early_warning_recall": warned / unsafe if unsafe else None,
        "safe_trajectory_fpr": false_alerts / safe if safe else None,
        "mean_lead_steps": mean(leads),
    }


def cluster_bootstrap_interval(
    records: Sequence[Dict[str, Any]],
    metric,
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Bootstrap complete trajectory/scenario clusters, never individual prefixes."""

    if not records or samples <= 0 or not 0.0 < confidence < 1.0:
        return None
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("scenario_id") or record.get("trajectory_id"))
        grouped[key].append(record)
    keys = sorted(grouped)
    rng = random.Random(seed)
    values: List[float] = []
    for _ in range(samples):
        sampled: List[Dict[str, Any]] = []
        for draw_index, _ in enumerate(keys):
            selected_key = rng.choice(keys)
            for record in grouped[selected_key]:
                cloned = dict(record)
                bootstrap_id = f"bootstrap-{draw_index}:{selected_key}"
                cloned["trajectory_id"] = bootstrap_id
                if "scenario_id" in cloned:
                    cloned["scenario_id"] = bootstrap_id
                sampled.append(cloned)
        value = metric(sampled)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    values.sort()
    tail = (1.0 - confidence) / 2.0
    low = values[int(tail * (len(values) - 1))]
    high = values[int((1.0 - tail) * (len(values) - 1))]
    return low, high


__all__ = [
    "binary_auc",
    "cluster_bootstrap_interval",
    "expected_calibration_error",
    "mean",
    "trajectory_outcomes",
]
