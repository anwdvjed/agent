"""Shared epoch-visible evaluation for validation and frozen tests."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .losses import LossWeights, agent_new_loss
from .metrics import binary_auc, expected_calibration_error, trajectory_outcomes
from .model import AgentNewModel
from .runtime import move_nested


def choose_device(value: str) -> torch.device:
    requested = str(value).casefold()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _binary_metrics(targets: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    predictions = [int(score > threshold) for score in scores]
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auc": binary_auc(targets, scores),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "brier": (
            sum((score - target) ** 2 for score, target in zip(scores, targets))
            / len(targets)
            if targets
            else None
        ),
        "ece": expected_calibration_error(targets, scores),
    }


def evaluate_epoch(
    model: AgentNewModel,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    event_types: Sequence[str],
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
    threshold: float = 0.5,
    loss_weights: Optional[LossWeights] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    losses = 0.0
    samples = 0
    targets: List[int] = []
    scores: List[float] = []
    group_ids: List[str] = []
    intent_targets: List[int] = []
    intent_scores: List[float] = []
    utility_targets: List[int] = []
    utility_scores: List[float] = []
    type_correct = type_total = 0
    time_errors: List[float] = []
    records: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    progress = tqdm(loader, desc=f"{phase} Epoch {epoch:03d}/{total_epochs:03d}", unit="batch")
    with torch.inference_mode():
        for raw_batch in progress:
            tracking = raw_batch["tracking"]
            batch = move_nested(raw_batch, device)
            output = model(batch)
            loss, _ = agent_new_loss(output, batch, loss_weights)
            batch_size = int(batch["event_observed"].shape[0])
            losses += float(loss.item()) * batch_size
            samples += batch_size
            risks = output["actual_risk"].detach().cpu().tolist()
            observed = batch["event_observed"].long().cpu().tolist()
            survival_mask = (batch["supervision_survival"] > 0).cpu().tolist()
            cause_risk = output["actual"]["selected"]["cause_risk"].cpu()
            incidence = output["actual"]["selected"]["incidence_mass"].cpu().sum(dim=-1)
            for index in range(batch_size):
                source_counts[str(tracking[index].get("source_family", "unknown"))] += 1
                if survival_mask[index]:
                    targets.append(int(observed[index]))
                    scores.append(float(risks[index]))
                    group_ids.append(str(tracking[index]["group_id"]))
                if survival_mask[index] and observed[index]:
                    predicted_type = int(cause_risk[index].argmax().item())
                    type_correct += int(predicted_type == int(batch["event_type"][index].item()))
                    type_total += 1
                    mass = incidence[index]
                    expected_time = float(
                        (mass * torch.arange(1, mass.shape[0] + 1)).sum().item()
                        / max(float(mass.sum().item()), 1e-8)
                    )
                    time_errors.append(
                        abs(expected_time - int(batch["event_time"][index].item()))
                    )
                if float(batch["supervision_attack_intent"][index].item()) > 0:
                    intent_targets.append(int(batch["attack_intent_target"][index].item()))
                    intent_scores.append(
                        float(torch.sigmoid(output["actual"]["attack_intent_logit"][index]).item())
                    )
                if float(batch["supervision_utility"][index].item()) > 0:
                    utility_targets.append(int(batch["utility_target"][index].item()))
                    utility_scores.append(
                        float(torch.sigmoid(output["actual"]["utility_logit"][index]).item())
                    )
                records.append(
                    {
                        "trajectory_id": tracking[index]["trajectory_id"],
                        "step_index": int(tracking[index]["step_index"]),
                        "current_violation": bool(
                            observed[index] and int(batch["event_time"][index].item()) == 1
                        ),
                        "risk": float(risks[index]),
                    }
                )
            progress.set_postfix(loss=f"{losses / max(samples, 1):.4f}")
    if samples == 0:
        raise RuntimeError(f"{phase} loader produced no samples")
    metrics: Dict[str, Any] = {
        "loss": losses / samples,
        "samples": samples,
        "supervised_survival_samples": len(targets),
        "horizon": _binary_metrics(targets, scores, threshold),
        "event_type_accuracy": type_correct / type_total if type_total else None,
        "event_time_mae": sum(time_errors) / len(time_errors) if time_errors else None,
        "attack_intent": _binary_metrics(intent_targets, intent_scores, 0.5)
        if intent_targets
        else None,
        "utility": _binary_metrics(utility_targets, utility_scores, 0.5)
        if utility_targets
        else None,
        "trajectory": trajectory_outcomes(records, threshold),
        "source_prefix_counts": dict(sorted(source_counts.items())),
    }
    raw = {
        "targets": targets,
        "scores": scores,
        "group_ids": group_ids,
        "records": records,
    }
    return metrics, raw


__all__ = ["choose_device", "evaluate_epoch"]
