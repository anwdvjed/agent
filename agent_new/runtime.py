"""Pre-execution runtime that combines hard constraints with model forecasts."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .calibration import GroupConformalCalibration
from .checkpoint import CheckpointMetadata, canonical_digest, load_checkpoint
from .constants import (
    DEFAULT_EVENT_TYPES,
    EDGE_STATE_DIM,
    EVENT_FEATURE_DIM,
    NODE_FEATURE_DIM,
)
from .data import collate_prepared, prepare_request
from .model import AgentNewConfig, AgentNewModel
from .policy import InterventionPolicy, PolicyDecision
from .registry import ToolRegistry


PathLike = Union[str, Path]


def move_nested(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested(item, device) for item in value)
    return value


def _load_calibration(path: Optional[PathLike]) -> Optional[GroupConformalCalibration]:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return GroupConformalCalibration.from_dict(value)


@dataclass
class LoadedEnsemble:
    models: List[AgentNewModel]
    metadata: List[CheckpointMetadata]
    config: AgentNewConfig
    event_types: Tuple[str, ...]
    registry_digest: str
    calibration_digest: Optional[str]
    state_fingerprints: Tuple[str, ...]
    independent_runs_verified: bool


def _state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor values independent of torch archive metadata/filenames."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_ensemble(
    checkpoint_paths: Sequence[PathLike],
    registry: ToolRegistry,
    device: torch.device,
    calibration: Optional[GroupConformalCalibration] = None,
) -> LoadedEnsemble:
    if not checkpoint_paths:
        raise ValueError("at least one trained checkpoint is required")
    registry_digest = canonical_digest(registry.to_dict())
    calibration_digest = calibration.digest() if calibration is not None else None
    models: List[AgentNewModel] = []
    metadata_items: List[CheckpointMetadata] = []
    config: Optional[AgentNewConfig] = None
    event_types: Optional[Tuple[str, ...]] = None
    seen_state_digests = set()
    state_fingerprints: List[str] = []
    for path in checkpoint_paths:
        state, metadata = load_checkpoint(
            path,
            map_location=device,
            expected_model_config=config.to_dict() if config is not None else None,
            expected_event_types=event_types,
            expected_registry_digest=registry_digest,
            expected_calibration_digest=calibration_digest,
        )
        current_config = AgentNewConfig.from_dict(metadata.model_config)
        if current_config.num_event_types != len(metadata.event_types):
            raise ValueError("checkpoint event ontology does not match model output")
        if current_config.actual_branch != "none":
            raise ValueError("runtime checkpoints must use 'none' as the actual branch")
        if (
            current_config.event_dim != EVENT_FEATURE_DIM
            or current_config.node_dim != NODE_FEATURE_DIM
            or current_config.edge_state_dim != EDGE_STATE_DIM
        ):
            raise ValueError("checkpoint feature dimensions do not match this runtime schema")
        state_fingerprint = _state_dict_fingerprint(state)
        if state_fingerprint in seen_state_digests:
            raise ValueError(
                "ensemble checkpoints must contain distinct independently trained states"
            )
        seen_state_digests.add(state_fingerprint)
        state_fingerprints.append(state_fingerprint)
        model = AgentNewModel(current_config)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        if config is None:
            config = current_config
            event_types = tuple(metadata.event_types)
        models.append(model)
        metadata_items.append(metadata)
    assert config is not None and event_types is not None
    run_ids = [str(item.extra.get("training_run_id", "")) for item in metadata_items]
    seeds = [item.extra.get("seed") for item in metadata_items]
    independent_runs_verified = bool(
        len(models) >= 2
        and all(run_ids)
        and len(set(run_ids)) == len(run_ids)
        and all(seed is not None for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    checkpoint_calibration_digests = {
        item.calibration_digest for item in metadata_items
    }
    if len(checkpoint_calibration_digests) != 1:
        raise ValueError("ensemble checkpoints use different calibration artifacts")
    checkpoint_calibration_digest = next(iter(checkpoint_calibration_digests))
    return LoadedEnsemble(
        models,
        metadata_items,
        config,
        event_types,
        registry_digest,
        checkpoint_calibration_digest,
        tuple(state_fingerprints),
        independent_runs_verified,
    )


class SafetyRuntime:
    """Assess a candidate action before any real side effect is committed.

    Registry and graph prerequisites form a hard fail-closed layer.  Learned
    risk is consulted only for branches that satisfy those prerequisites.
    Uncalibrated or single-model forecasts may be inspected, but they cannot
    produce ``ALLOW``/repair decisions through this runtime.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        ensemble: LoadedEnsemble,
        *,
        calibration: Optional[GroupConformalCalibration] = None,
        policy: Optional[InterventionPolicy] = None,
        ensemble_z: float = 1.96,
        device: Optional[torch.device] = None,
        trusted_context_verifier: Optional[
            Callable[[Mapping[str, Any]], bool]
        ] = None,
    ) -> None:
        if ensemble_z < 0.0 or not math.isfinite(ensemble_z):
            raise ValueError("ensemble_z must be finite and non-negative")
        self.registry = registry
        self.ensemble = ensemble
        self.calibration = calibration
        self.policy = policy or InterventionPolicy()
        self.ensemble_z = float(ensemble_z)
        self.device = device or next(ensemble.models[0].parameters()).device
        self.trusted_context_verifier = trusted_context_verifier
        if canonical_digest(registry.to_dict()) != ensemble.registry_digest:
            raise ValueError("runtime registry does not match the verified ensemble registry")
        supplied_calibration_digest = (
            calibration.digest() if calibration is not None else None
        )
        if (
            supplied_calibration_digest is not None
            and supplied_calibration_digest != ensemble.calibration_digest
        ):
            raise ValueError("runtime calibration does not match checkpoint metadata")
        if calibration is not None and not math.isclose(
            self.ensemble_z, calibration.ensemble_z, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("runtime ensemble_z does not match the calibrated score protocol")

    def _forecast(self, batch: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
        tensor_batch = move_nested({"branches": batch["branches"]}, self.device)
        outputs: List[Mapping[str, Any]] = []
        with torch.inference_mode():
            for model in self.ensemble.models:
                outputs.append(model(tensor_batch))
        return outputs, tensor_batch

    def assess(self, request: Mapping[str, Any], evidence_top_k: int = 5) -> Dict[str, Any]:
        if canonical_digest(self.registry.to_dict()) != self.ensemble.registry_digest:
            raise RuntimeError(
                "trusted tool registry changed after checkpoint verification; reload the ensemble"
            )
        trusted_context_verified = bool(
            self.trusted_context_verifier is not None
            and self.trusted_context_verifier(request)
        )
        prepared = prepare_request(request, self.registry)
        batch = collate_prepared([prepared])
        forecasts, _ = self._forecast(batch)
        branch_names = tuple(forecasts[0]["branch_names"])
        for output in forecasts[1:]:
            if tuple(output["branch_names"]) != branch_names:
                raise ValueError("ensemble checkpoints disagree on intervention branches")

        mean_risks: Dict[str, float] = {}
        standard_deviations: Dict[str, float] = {}
        upper_bounds: Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}
        hard_status: Dict[str, Dict[str, Any]] = {}
        calibrated_ensemble = bool(
            self.calibration is not None
            and len(forecasts) >= 2
            and self.ensemble.independent_runs_verified
        )
        for name in branch_names:
            values = torch.tensor(
                [float(output["branches"][name]["risk"][0].item()) for output in forecasts]
            )
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
            metadata = batch["metadata"]["branches"][name][0]
            fail_closed = bool(metadata["fail_closed"])
            applicable = bool(
                name == "none" or metadata.get("intervention_effective", False)
            )
            point_upper = min(1.0, mean + self.ensemble_z * std)
            # The current calibration artifact is fitted to the actual
            # no-intervention score.  Repair branches remain informational
            # until branch-conditional simultaneous calibration is supplied.
            upper = (
                self.calibration.risk_upper_bound(point_upper)
                if self.calibration is not None and name == "none"
                else (point_upper if name == "none" else 1.0)
            )
            if fail_closed or not applicable:
                upper = 1.0
            mean_risks[name] = mean
            standard_deviations[name] = std
            upper_bounds[name] = upper
            uncertainties[name] = std if calibrated_ensemble else 1.0
            hard_status[name] = {
                "fail_closed": fail_closed,
                "applicable": applicable,
                "reasons": list(metadata["fail_reasons"]),
            }

        if all(
            value["fail_closed"] or not value["applicable"]
            for value in hard_status.values()
        ):
            decision = PolicyDecision(
                decision="BLOCK",
                intervention=None,
                reason=(
                    "every assessed branch violates a trusted capability, permission, "
                    "approval, or destination prerequisite"
                ),
                risk_upper_bound=None,
                actual_risk=mean_risks.get("none"),
                epistemic_uncertainty=None,
                intervention_cost=None,
            )
        elif not trusted_context_verified:
            decision = PolicyDecision(
                decision="HOLD",
                intervention=None,
                reason="trusted runtime context attestation is missing or invalid",
                risk_upper_bound=upper_bounds.get("none"),
                actual_risk=mean_risks.get("none"),
                epistemic_uncertainty=1.0,
                intervention_cost=None,
            )
        elif not calibrated_ensemble:
            decision = PolicyDecision(
                decision="HOLD",
                intervention=None,
                reason=(
                    "a calibrated ensemble with independently identified training runs "
                    "is required before model-based admission"
                ),
                risk_upper_bound=upper_bounds.get("none"),
                actual_risk=mean_risks.get("none"),
                epistemic_uncertainty=1.0,
                intervention_cost=None,
            )
        elif hard_status["none"]["fail_closed"]:
            decision = PolicyDecision(
                decision="BLOCK",
                intervention=None,
                reason="the actual candidate violates a trusted hard prerequisite",
                risk_upper_bound=upper_bounds["none"],
                actual_risk=mean_risks["none"],
                epistemic_uncertainty=uncertainties["none"],
                intervention_cost=None,
            )
        else:
            decision = self.policy.select_from_values(
                {"none": upper_bounds["none"]},
                {"none": mean_risks["none"]},
                {"none": uncertainties["none"]},
            )

        actual_outputs = [output["branches"]["none"] for output in forecasts]
        mean_cause_risk = torch.stack(
            [value["selected"]["cause_risk"][0].cpu() for value in actual_outputs]
        ).mean(dim=0)
        predicted_type_index = int(mean_cause_risk.argmax().item())
        mean_incidence = torch.stack(
            [value["selected"]["incidence_mass"][0].cpu() for value in actual_outputs]
        ).mean(dim=0).sum(dim=-1)
        time_values = torch.arange(1, mean_incidence.shape[0] + 1, dtype=mean_incidence.dtype)
        expected_step = float(
            (mean_incidence * time_values).sum().item()
            / max(float(mean_incidence.sum().item()), 1.0e-8)
        )
        evidence = torch.stack(
            [value["evidence_weights"].cpu() for value in actual_outputs]
        ).mean(dim=0)
        actual_metadata = batch["metadata"]["branches"]["none"][0]
        edge_metadata = actual_metadata["edge_metadata"]
        node_lookup = {
            int(item["index"]): item for item in actual_metadata["node_metadata"]
        }
        ranked = sorted(
            zip(evidence.tolist(), edge_metadata),
            key=lambda item: item[0],
            reverse=True,
        )[: max(0, int(evidence_top_k))]
        evidence_edges = [
            {
                "score": float(score),
                **dict(metadata),
                "source_node": {
                    key: node_lookup.get(int(metadata["source"]), {}).get(key)
                    for key in ("index", "type", "entity_id", "flags")
                },
                "target_node": {
                    key: node_lookup.get(int(metadata["target"]), {}).get(key)
                    for key in ("index", "type", "entity_id", "flags")
                },
            }
            for score, metadata in ranked
        ]
        event_types = self.ensemble.event_types or tuple(DEFAULT_EVENT_TYPES)
        commit = None
        if decision.decision == "ALLOW":
            actual_branch = prepared.branches["none"]
            commit = {
                "sha256": actual_branch.prepared_action_digest,
                "event_id": prepared.metadata["request_id"],
                "tool": actual_metadata["candidate_tool"],
                "require_digest_match_before_execution": True,
            }
        public_request_metadata = {
            key: prepared.metadata.get(key)
            for key in (
                "request_id",
                "policy_id",
                "registry_version",
                "history_length",
                "unknown_tool_events",
                "fail_closed",
            )
        }
        return {
            "decision": decision.to_dict(),
            "calibrated_ensemble": calibrated_ensemble,
            "trusted_context_verified": trusted_context_verified,
            "ensemble_size": len(forecasts),
            "branch_risk_mean": mean_risks,
            "branch_risk_std": standard_deviations,
            "branch_risk_upper_bound": upper_bounds,
            "hard_constraints": hard_status,
            "predicted_first_event_type": event_types[predicted_type_index],
            "expected_first_event_step": expected_step,
            "evidence_edges": evidence_edges,
            "evidence_path_verified": False,
            "commit": commit,
            "reassessment_required": decision.decision
            in {"HOLD", "REPLAN", "CONFIRM"},
            "repair_branch_certification": (
                "disabled: branch-conditional simultaneous calibration is required"
            ),
            "request_metadata": public_request_metadata,
        }

    def verify_commit_digest(self, request: Mapping[str, Any], digest: str) -> bool:
        """Rebuild the actual branch and verify an opaque commit authorization."""

        if canonical_digest(self.registry.to_dict()) != self.ensemble.registry_digest:
            return False
        prepared = prepare_request(request, self.registry)
        return prepared.branches["none"].prepared_action_digest == str(digest)


__all__ = ["LoadedEnsemble", "SafetyRuntime", "load_ensemble", "move_nested"]
