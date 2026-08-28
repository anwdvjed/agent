"""Epoch-visible ensemble prediction collection for calibration and testing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .metrics import binary_auc, expected_calibration_error, trajectory_outcomes
from .calibration import GroupConformalCalibration
from .model import AgentNewModel
from .runtime import move_nested


def collect_ensemble_epoch(
    models: Sequence[AgentNewModel],
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
    ensemble_z: float,
    threshold: float,
    calibration: Optional[GroupConformalCalibration] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not models:
        raise ValueError("ensemble must contain at least one model")
    for model in models:
        model.eval()
    targets: List[int] = []
    scores: List[float] = []
    group_ids: List[str] = []
    records: List[Dict[str, Any]] = []
    intent_targets: List[int] = []
    intent_scores: List[float] = []
    utility_targets: List[int] = []
    utility_scores: List[float] = []
    type_correct = type_total = 0
    time_errors: List[float] = []
    source_counts: Counter = Counter()
    progress = tqdm(loader, desc=f"{phase} Epoch {epoch:03d}/{total_epochs:03d}", unit="batch")
    with torch.inference_mode():
        for raw_batch in progress:
            tracking = raw_batch["tracking"]
            batch = move_nested(raw_batch, device)
            outputs = [model(batch) for model in models]
            risks = torch.stack([output["actual_risk"] for output in outputs])
            score = (risks.mean(dim=0) + ensemble_z * risks.std(dim=0, unbiased=False)).clamp(0, 1)
            if calibration is not None:
                score = torch.tensor(
                    calibration.risk_upper_bounds(score.detach().cpu().tolist()),
                    device=device,
                    dtype=risks.dtype,
                )
            cause_risk = torch.stack(
                [output["actual"]["selected"]["cause_risk"] for output in outputs]
            ).mean(dim=0)
            incidence = torch.stack(
                [output["actual"]["selected"]["incidence_mass"] for output in outputs]
            ).mean(dim=0).sum(dim=-1)
            intent = torch.stack(
                [torch.sigmoid(output["actual"]["attack_intent_logit"]) for output in outputs]
            ).mean(dim=0)
            utility = torch.stack(
                [torch.sigmoid(output["actual"]["utility_logit"]) for output in outputs]
            ).mean(dim=0)
            batch_size = int(score.shape[0])
            for index in range(batch_size):
                source_counts[str(tracking[index].get("source_family", "unknown"))] += 1
                supervised = float(batch["supervision_survival"][index].item()) > 0
                observed = bool(batch["event_observed"][index].item())
                if supervised:
                    targets.append(int(observed))
                    scores.append(float(score[index].item()))
                    group_ids.append(str(tracking[index]["group_id"]))
                if supervised and observed:
                    type_total += 1
                    type_correct += int(
                        int(cause_risk[index].argmax().item())
                        == int(batch["event_type"][index].item())
                    )
                    mass = incidence[index]
                    expected_time = float(
                        (mass * torch.arange(1, mass.shape[0] + 1, device=mass.device)).sum().item()
                        / max(float(mass.sum().item()), 1e-8)
                    )
                    time_errors.append(
                        abs(expected_time - int(batch["event_time"][index].item()))
                    )
                if float(batch["supervision_attack_intent"][index].item()) > 0:
                    intent_targets.append(int(batch["attack_intent_target"][index].item()))
                    intent_scores.append(float(intent[index].item()))
                if float(batch["supervision_utility"][index].item()) > 0:
                    utility_targets.append(int(batch["utility_target"][index].item()))
                    utility_scores.append(float(utility[index].item()))
                records.append(
                    {
                        "trajectory_id": tracking[index]["trajectory_id"],
                        "step_index": int(tracking[index]["step_index"]),
                        "current_violation": bool(
                            observed and int(batch["event_time"][index].item()) == 1
                        ),
                        "risk": float(score[index].item()),
                    }
                )
            progress.set_postfix(prefixes=len(scores))
    predictions = [int(score > threshold) for score in scores]
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "samples": len(records),
        "supervised_survival_samples": len(targets),
        "horizon_auc": binary_auc(targets, scores),
        "horizon_brier": sum((s - t) ** 2 for s, t in zip(scores, targets)) / len(targets)
        if targets
        else None,
        "horizon_ece": expected_calibration_error(targets, scores),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "event_type_accuracy": type_correct / type_total if type_total else None,
        "event_time_mae": sum(time_errors) / len(time_errors) if time_errors else None,
        "attack_intent_auc": binary_auc(intent_targets, intent_scores),
        "utility_auc": binary_auc(utility_targets, utility_scores),
        "trajectory": trajectory_outcomes(records, threshold),
        "source_prefix_counts": dict(sorted(source_counts.items())),
    }
    return metrics, {
        "targets": targets,
        "scores": scores,
        "group_ids": group_ids,
        "records": records,
    }


__all__ = ["collect_ensemble_epoch"]
