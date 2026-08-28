"""End-to-end engineering smoke test; it does not produce research evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .data import collate_prepared, prepare_request
from .losses import agent_new_loss
from .model import AgentNewConfig, AgentNewModel
from .registry import ToolRegistry


def run_smoke(root: Optional[Path] = None) -> Dict[str, Any]:
    asset_root = (
        Path(__file__).resolve().parent / "assets"
        if root is None
        else Path(root)
    )
    registry_path = (
        asset_root / "tool_registry.json"
        if root is None
        else asset_root / "configs" / "tool_registry.json"
    )
    request_path = (
        asset_root / "unsafe_request.json"
        if root is None
        else asset_root / "examples" / "unsafe_request.json"
    )
    model_path = (
        asset_root / "model.json"
        if root is None
        else asset_root / "configs" / "model.json"
    )
    registry = ToolRegistry.from_json(registry_path)
    request = json.loads(request_path.read_text())
    config = AgentNewConfig.from_dict(
        json.loads(model_path.read_text())
    )
    torch.manual_seed(20260824)
    model = AgentNewModel(config)
    example = prepare_request(request, registry)
    batch = collate_prepared([example])
    # Engineering-only labels exercise the likelihood and gradient paths.
    batch.update(
        {
            "event_observed": torch.tensor([True]),
            "event_time": torch.tensor([1]),
            "event_type": torch.tensor([0]),
            "censor_time": torch.tensor([config.horizon]),
            "sample_weight": torch.tensor([1.0]),
            "supervision_survival": torch.tensor([1.0]),
        }
    )
    outputs = model(batch)
    batch["intervention_event_targets"] = torch.ones_like(
        outputs["intervention_risk_matrix"]
    )
    batch["intervention_target_names"] = list(outputs["intervention_names"])
    loss, parts = agent_new_loss(outputs, batch)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "status": "ok" if bool(torch.isfinite(loss)) and finite_gradients else "failed",
        "warning": "untrained random-weight engineering smoke; risks are not safety predictions",
        "branch_names": list(outputs["branch_names"]),
        "risk_matrix_shape": list(outputs["risk_matrix"].shape),
        "actual_initial_risk": float(outputs["actual_risk"][0].item()),
        "loss": float(loss.item()),
        "loss_parts": {name: float(value.item()) for name, value in parts.items()},
        "finite_gradients": finite_gradients,
        "node_count": int(batch["branches"]["none"]["node_features"].shape[0]),
        "edge_count": int(batch["branches"]["none"]["edge_types"].shape[0]),
        "fail_closed_reasons": batch["metadata"]["branches"]["none"][0][
            "fail_reasons"
        ],
    }


__all__ = ["run_smoke"]
