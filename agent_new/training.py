"""Small, explicit training loop for Agent New batches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from .losses import LossWeights, agent_new_loss
from .model import AgentNewModel
from .runtime import move_nested


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    patience: int = 5

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("epochs and patience must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.gradient_clip)
            or self.learning_rate <= 0.0
            or self.gradient_clip <= 0.0
        ):
            raise ValueError("learning_rate and gradient_clip must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_epoch(
    model: AgentNewModel,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    loss_weights: Optional[LossWeights] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    samples = 0
    for raw_batch in batches:
        batch = move_nested(raw_batch, device)
        actual_name = "none" if "none" in batch["branches"] else next(iter(batch["branches"]))
        batch_size = int(batch["branches"][actual_name]["event_features"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = agent_new_loss(outputs, batch, loss_weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite Agent New loss")
        if training:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite Agent New gradient norm")
            optimizer.step()
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size
        samples += batch_size
    if samples == 0:
        raise RuntimeError("training/evaluation loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def fit(
    model: AgentNewModel,
    train_batches: Iterable[Mapping[str, Any]],
    validation_batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    config: Optional[TrainingConfig] = None,
    loss_weights: Optional[LossWeights] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Fit with early stopping and return history plus the best CPU state dict.

    Callers should pass re-iterable loaders, not one-shot generators.
    """

    config = config or TrainingConfig()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_batches,
            device,
            loss_weights,
            optimizer,
            config.gradient_clip,
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model,
                validation_batches,
                device,
                loss_weights,
            )
        value = float(validation_metrics["total"])
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        if value < best_loss - 1.0e-9:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    return history, best_state


__all__ = ["TrainingConfig", "fit", "run_epoch"]
