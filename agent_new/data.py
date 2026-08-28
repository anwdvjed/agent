"""Strict request preparation and multi-branch tensor collation for Agent New."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .constants import INTERVENTIONS
from .features import (
    MAX_TEXT_CHARS,
    bounded_text,
    event_feature,
    security_progress,
    side_effect_flags,
)
from .graph import (
    DATA_ARGUMENT_ROLES,
    DESTINATION_ARGUMENT_ROLES,
    GraphEvent,
    GraphTensors,
    build_persistent_graph,
)
from .registry import MAX_REQUIRED_SCOPES, ToolRegistry, ToolSpec


MAX_HISTORY_EVENTS = 32
MAX_POLICY_CLAUSES = 32
MAX_ARGUMENTS = 32
MAX_COLLECTION_ITEMS = 128
MAX_NESTING_DEPTH = 6
MAX_EVENT_ID_CHARS = 200
MAX_POLICY_ID_CHARS = 200
MAX_REQUEST_TEXT_CHARS = 32768


class RequestValidationError(ValueError):
    """Raised when a request cannot be represented safely and unambiguously."""


def _strict_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise RequestValidationError(f"{name} must be a JSON boolean")
    return value


def _strict_text(value: Any, name: str, limit: int) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if len(text) > limit:
        raise RequestValidationError(f"{name} exceeds the limit of {limit} characters")
    return text


@dataclass
class PreparedBranch:
    name: str
    event_features: torch.Tensor
    wall_deltas: torch.Tensor
    step_positions: torch.Tensor
    security_progress: torch.Tensor
    graph: GraphTensors
    metadata: Dict[str, Any]
    prepared_action: Dict[str, Any]
    commit_payload: Dict[str, Any]
    prepared_action_digest: str


@dataclass
class PreparedExample:
    branches: Dict[str, PreparedBranch]
    metadata: Dict[str, Any]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RequestValidationError(f"{name} must be an object")
    return value


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_NESTING_DEPTH:
        raise RequestValidationError(
            f"request value exceeds nesting depth {MAX_NESTING_DEPTH}"
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RequestValidationError("request contains a non-finite number")
        return value
    if isinstance(value, str):
        return _strict_text(value, "request string value", MAX_TEXT_CHARS)
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise RequestValidationError(
                f"request object exceeds {MAX_COLLECTION_ITEMS} fields"
            )
        return {
            bounded_text(key, 160): _bounded_value(child, depth=depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise RequestValidationError(
                f"request list exceeds {MAX_COLLECTION_ITEMS} items"
            )
        return [_bounded_value(child, depth=depth + 1) for child in value]
    return bounded_text(value, MAX_TEXT_CHARS)


def _bounded_string_list(value: Any, name: str, limit: int) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RequestValidationError(f"{name} must be a list")
    if len(value) > limit:
        raise RequestValidationError(f"{name} exceeds the limit of {limit}")
    result: List[str] = []
    seen = set()
    for raw in value:
        item = _strict_text(raw, name, 512).strip()
        if not item:
            raise RequestValidationError(f"{name} contains an empty value")
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _normalize_policy(value: Any) -> Tuple[str, List[Dict[str, str]]]:
    policy = _require_mapping(value, "policy")
    policy_id = _strict_text(
        policy.get("id", ""), "policy.id", MAX_POLICY_ID_CHARS
    ).strip()
    if not policy_id:
        raise RequestValidationError("policy.id is required")
    raw_clauses = policy.get("clauses")
    if isinstance(raw_clauses, str) or not isinstance(raw_clauses, Sequence):
        raise RequestValidationError("policy.clauses must be a list")
    if not raw_clauses:
        raise RequestValidationError("policy.clauses must not be empty")
    if len(raw_clauses) > MAX_POLICY_CLAUSES:
        raise RequestValidationError(
            f"policy.clauses exceeds the limit of {MAX_POLICY_CLAUSES}"
        )
    clauses: List[Dict[str, str]] = []
    seen = set()
    for index, raw_clause in enumerate(raw_clauses):
        if isinstance(raw_clause, Mapping):
            clause_id = _strict_text(
                raw_clause.get("id", f"clause-{index + 1:03d}"),
                f"policy.clauses[{index}].id",
                160,
            ).strip()
            text = _strict_text(
                raw_clause.get("text", ""),
                f"policy.clauses[{index}].text",
                MAX_TEXT_CHARS,
            ).strip()
        else:
            clause_id = f"clause-{index + 1:03d}"
            text = _strict_text(
                raw_clause, f"policy.clauses[{index}]", MAX_TEXT_CHARS
            ).strip()
        if not clause_id or not text:
            raise RequestValidationError("policy clauses require non-empty id and text")
        if clause_id in seen:
            raise RequestValidationError(f"duplicate policy clause id: {clause_id!r}")
        clauses.append({"id": clause_id, "text": text})
        seen.add(clause_id)
    return policy_id, clauses


def _normalize_event(value: Any, *, candidate: bool) -> Dict[str, Any]:
    event = _require_mapping(value, "candidate" if candidate else "history event")
    required = {
        "event_id",
        "timestamp",
        "observation",
        "action",
        "permission_receipt",
        "runtime",
    }
    missing = required.difference(event)
    if missing:
        raise RequestValidationError(
            f"event is missing required fields: {sorted(missing)}"
        )
    event_id = _strict_text(
        event.get("event_id"), "event.event_id", MAX_EVENT_ID_CHARS
    ).strip()
    if not event_id:
        raise RequestValidationError("event_id must not be empty")
    try:
        timestamp = float(event.get("timestamp"))
    except (TypeError, ValueError) as exc:
        raise RequestValidationError("event timestamp must be numeric") from exc
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise RequestValidationError("event timestamp must be finite and non-negative")

    observation_raw = _require_mapping(event.get("observation"), "event.observation")
    observation = {
        "source_id": _strict_text(
            observation_raw.get("source_id", "unknown-source"),
            "event.observation.source_id",
            512,
        ),
        "source_type": _strict_text(
            observation_raw.get("source_type", "unknown"),
            "event.observation.source_type",
            256,
        ),
        "trust_level": _strict_text(
            observation_raw.get("trust_level", "unknown"),
            "event.observation.trust_level",
            80,
        ).casefold(),
        "content": _strict_text(
            observation_raw.get("content", ""),
            "event.observation.content",
            MAX_TEXT_CHARS,
        ),
        "data_ids": list(
            _bounded_string_list(
                observation_raw.get("data_ids", ()),
                "observation.data_ids",
                MAX_COLLECTION_ITEMS,
            )
        ),
        "instruction_like": _strict_bool(
            observation_raw.get("instruction_like"),
            "event.observation.instruction_like",
        ),
        "contains_sensitive": _strict_bool(
            observation_raw.get("contains_sensitive"),
            "event.observation.contains_sensitive",
        ),
    }

    action_raw = _require_mapping(event.get("action"), "event.action")
    tool_name = _strict_text(action_raw.get("tool", ""), "event.action.tool", 160).strip()
    if not tool_name:
        raise RequestValidationError("event.action.tool is required")
    arguments_raw = action_raw.get("arguments", {})
    arguments = _require_mapping(arguments_raw, "event.action.arguments")
    if len(arguments) > MAX_ARGUMENTS:
        raise RequestValidationError(
            f"event.action.arguments exceeds the limit of {MAX_ARGUMENTS}"
        )
    # Intentionally do not copy operation/scopes/approval/side effects from the
    # action.  Those fields are owned exclusively by ToolRegistry.
    action = {
        "tool": tool_name,
        "arguments": {
            _strict_text(name, "event.action argument name", 160): _bounded_value(argument)
            for name, argument in arguments.items()
        },
    }

    receipt_raw = _require_mapping(
        event.get("permission_receipt"), "event.permission_receipt"
    )
    receipt = {
        "granted_scopes": list(
            _bounded_string_list(
                receipt_raw.get("granted_scopes", ()),
                "permission_receipt.granted_scopes",
                MAX_REQUIRED_SCOPES,
            )
        ),
        "approval_verified": _strict_bool(
            receipt_raw.get("approval_verified"),
            "event.permission_receipt.approval_verified",
        ),
        "receipt_verified": _strict_bool(
            receipt_raw.get("receipt_verified"),
            "event.permission_receipt.receipt_verified",
        ),
        "receipt_id": bounded_text(receipt_raw.get("receipt_id", ""), 256),
    }

    runtime_raw = _require_mapping(event.get("runtime"), "event.runtime")
    policy_results_raw = _require_mapping(
        runtime_raw.get("policy_clause_results", {}),
        "event.runtime.policy_clause_results",
    )
    if len(policy_results_raw) > MAX_POLICY_CLAUSES:
        raise RequestValidationError("policy_clause_results exceeds the policy clause limit")
    intervention_evaluations_raw = _require_mapping(
        runtime_raw.get("intervention_policy_evaluations", {}),
        "event.runtime.intervention_policy_evaluations",
    )
    if len(intervention_evaluations_raw) > len(INTERVENTIONS):
        raise RequestValidationError("too many intervention policy evaluations")
    intervention_evaluations: Dict[str, Any] = {}
    for intervention_name, raw_evaluation in intervention_evaluations_raw.items():
        name = bounded_text(intervention_name, 160)
        if name not in INTERVENTIONS or name == "none":
            raise RequestValidationError(
                f"invalid intervention policy evaluation: {name!r}"
            )
        evaluation = _require_mapping(
            raw_evaluation,
            f"event.runtime.intervention_policy_evaluations.{name}",
        )
        clause_results = _require_mapping(
            evaluation.get("policy_clause_results", {}),
            f"event.runtime.intervention_policy_evaluations.{name}.policy_clause_results",
        )
        if len(clause_results) > MAX_POLICY_CLAUSES:
            raise RequestValidationError("intervention clause results exceed the limit")
        intervention_evaluations[name] = {
            "verified": _strict_bool(
                evaluation.get("verified"),
                f"event.runtime.intervention_policy_evaluations.{name}.verified",
            ),
            "policy_clause_results": {
                bounded_text(key, 160): _strict_bool(
                    result,
                    (
                        "event.runtime.intervention_policy_evaluations."
                        f"{name}.policy_clause_results.{key}"
                    ),
                )
                for key, result in clause_results.items()
            },
        }
    runtime = {
        "actor_id": bounded_text(runtime_raw.get("actor_id", "agent"), 160),
        "actor_verified": _strict_bool(
            runtime_raw.get("actor_verified"), "event.runtime.actor_verified"
        ),
        "sandboxed": _strict_bool(
            runtime_raw.get("sandboxed"), "event.runtime.sandboxed"
        ),
        "read_only": _strict_bool(
            runtime_raw.get("read_only"), "event.runtime.read_only"
        ),
        "user_confirmed": _strict_bool(
            runtime_raw.get("user_confirmed"), "event.runtime.user_confirmed"
        ),
        "sensitive_data_ids": list(
            _bounded_string_list(
                runtime_raw.get("sensitive_data_ids", ()),
                "runtime.sensitive_data_ids",
                MAX_COLLECTION_ITEMS,
            )
        ),
        "known_destinations": list(
            _bounded_string_list(
                runtime_raw.get("known_destinations", ()),
                "runtime.known_destinations",
                MAX_COLLECTION_ITEMS,
            )
        ),
        "policy_evaluation_verified": _strict_bool(
            runtime_raw.get("policy_evaluation_verified"),
            "event.runtime.policy_evaluation_verified",
        ),
        "policy_clause_results": {
            bounded_text(key, 160): _strict_bool(
                result, f"event.runtime.policy_clause_results.{key}"
            )
            for key, result in policy_results_raw.items()
        },
        "intervention_policy_evaluations": intervention_evaluations,
    }
    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "observation": observation,
        "action": action,
        "permission_receipt": receipt,
        "runtime": runtime,
        "candidate": candidate,
    }


def _matches_sensitive(value: Any, sensitive_ids: Sequence[str]) -> bool:
    text = bounded_text(value, MAX_TEXT_CHARS).casefold()
    return any(identifier.casefold() in text for identifier in sensitive_ids)


def _destination_identity(value: Any) -> Tuple[str, Optional[str]]:
    exact = bounded_text(value, 1024).strip().casefold()
    domain: Optional[str] = None
    if "@" in exact:
        domain = exact.rsplit("@", 1)[-1]
    else:
        parsed = urlparse(exact if "://" in exact else "//" + exact)
        if parsed.hostname:
            domain = parsed.hostname.casefold()
    return exact, domain


def _destination_is_known(value: Any, known_destinations: Sequence[str]) -> bool:
    exact, domain = _destination_identity(value)
    for raw_known in known_destinations:
        known = bounded_text(raw_known, 1024).strip().casefold()
        if known.startswith("exact:"):
            if exact == known[len("exact:") :]:
                return True
        elif known.startswith("domain:"):
            allowed_domain = known[len("domain:") :].lstrip(".")
            if domain and (
                domain == allowed_domain or domain.endswith("." + allowed_domain)
            ):
                return True
        elif exact == known:
            # Bare entries are exact destinations.  Domain-wide authority must
            # be explicit with the ``domain:`` prefix.
            return True
    return False


def _apply_candidate_intervention(
    event: Mapping[str, Any], spec: ToolSpec, intervention: str
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(event))
    arguments: MutableMapping[str, Any] = result["action"]["arguments"]
    runtime = result["runtime"]
    receipt = result["permission_receipt"]
    changed_fields: List[str] = []
    if intervention == "verified_approval":
        if spec.approval_required and not receipt.get("approval_verified", False):
            receipt["approval_verified"] = True
            changed_fields.append("permission_receipt.approval_verified")
    elif intervention == "redact_sensitive":
        for argument_name in list(arguments):
            role = spec.role_for(str(argument_name))
            if (
                role in DATA_ARGUMENT_ROLES
                and arguments[argument_name] != "[REDACTED_BY_POLICY]"
            ):
                arguments[argument_name] = "[REDACTED_BY_POLICY]"
                changed_fields.append(f"action.arguments.{argument_name}")
    elif intervention == "constrain_destination":
        known = runtime.get("known_destinations", [])
        replacement = next(
            (
                value[len("exact:") :] if value.casefold().startswith("exact:") else value
                for value in known
                if not value.casefold().startswith("domain:")
            ),
            None,
        )
        if replacement is not None:
            for argument_name in list(arguments):
                if spec.role_for(str(argument_name)) in DESTINATION_ARGUMENT_ROLES:
                    if arguments[argument_name] != replacement:
                        arguments[argument_name] = replacement
                        changed_fields.append(f"action.arguments.{argument_name}")
    elif intervention == "sandbox":
        if spec.operation == "execute" and not runtime.get("sandboxed", False):
            runtime["sandboxed"] = True
            changed_fields.append("runtime.sandboxed")
    elif intervention == "restrict_capability":
        if not runtime.get("capability_restricted", False):
            runtime["capability_restricted"] = True
            changed_fields.append("runtime.capability_restricted")
    elif intervention != "none":
        raise RequestValidationError(f"unsupported intervention: {intervention!r}")
    effective = intervention == "none" or bool(changed_fields)
    runtime["intervention_effective"] = effective
    runtime["intervention_changed_fields"] = changed_fields
    if intervention != "none" and effective:
        evaluation = runtime.get("intervention_policy_evaluations", {}).get(
            intervention
        )
        if evaluation is None:
            runtime["policy_evaluation_verified"] = False
            runtime["policy_clause_results"] = {}
        else:
            runtime["policy_evaluation_verified"] = evaluation["verified"]
            runtime["policy_clause_results"] = dict(
                evaluation["policy_clause_results"]
            )
    return result


def _event_state(
    event: Mapping[str, Any],
    spec: ToolSpec,
    clauses: Sequence[Mapping[str, Any]],
    intervention: str,
) -> Dict[str, Any]:
    observation = event["observation"]
    action = event["action"]
    receipt = event["permission_receipt"]
    runtime = event["runtime"]
    trust = str(observation.get("trust_level", "unknown")).casefold()
    source_verified = trust == "trusted"
    source_untrusted = trust != "trusted"
    receipt_verified = bool(receipt.get("receipt_verified"))
    granted = tuple(receipt.get("granted_scopes", ())) if receipt_verified else ()
    granted_set = set(granted)
    missing = tuple(scope for scope in spec.required_scopes if scope not in granted_set)
    approval_verified = bool(
        receipt_verified and receipt.get("approval_verified", False)
    ) or intervention == "verified_approval"

    arguments = action.get("arguments", {})
    unknown_arguments = tuple(
        sorted(str(name) for name in arguments if spec.role_for(str(name)) is None)
    )
    sensitive_ids = tuple(runtime.get("sensitive_data_ids", ()))
    observation_data = tuple(observation.get("data_ids", ()))
    sensitive = bool(observation.get("contains_sensitive")) or any(
        item in sensitive_ids for item in observation_data
    )
    for argument_name, argument_value in arguments.items():
        role = spec.role_for(str(argument_name))
        if role in DATA_ARGUMENT_ROLES and _matches_sensitive(
            argument_value, sensitive_ids
        ):
            sensitive = True
    original_sensitive = sensitive

    destination_values = [
        argument_value
        for argument_name, argument_value in arguments.items()
        if spec.role_for(str(argument_name)) in DESTINATION_ARGUMENT_ROLES
    ]
    effects = dict(side_effect_flags(spec))
    known_destinations = tuple(runtime.get("known_destinations", ()))
    destinations_known = bool(destination_values) and all(
        _destination_is_known(value, known_destinations)
        for value in destination_values
    )
    destination_bearing = bool(
        spec.operation == "write_external"
        or effects.get("network_effect")
        or destination_values
    )
    if destination_bearing:
        external = True
        destination_allowed = destinations_known
    else:
        external = False
        destination_allowed = True

    redacted = intervention == "redact_sensitive"
    if redacted and not unknown_arguments:
        sensitive = False
    destination_constrained = intervention == "constrain_destination"
    capability_restricted = bool(runtime.get("capability_restricted", False))
    all_scopes_granted = bool(
        spec.known and not missing and receipt_verified and not capability_restricted
    )

    clause_results = runtime.get("policy_clause_results", {})
    if not isinstance(clause_results, Mapping):
        clause_results = {}
    policy_verified = bool(runtime.get("policy_evaluation_verified", False))
    policy_satisfied = bool(
        policy_verified
        and all(clause_results.get(str(clause["id"])) is True for clause in clauses)
    )
    state: Dict[str, Any] = {
        "candidate": bool(event.get("candidate")),
        "actor_id": runtime.get("actor_id", "agent"),
        "actor_verified": bool(runtime.get("actor_verified", False)),
        "source_id": observation.get("source_id", "unknown-source"),
        "source_origin": observation.get("source_type", "unknown"),
        "source_verified": source_verified,
        "source_untrusted": source_untrusted,
        "instruction_like": bool(observation.get("instruction_like", False)),
        "contains_sensitive": sensitive,
        "original_sensitive": original_sensitive,
        "known_tool": spec.known,
        "unknown_capability": not spec.known or bool(unknown_arguments),
        "unknown_arguments": unknown_arguments,
        "required_scopes": tuple(spec.required_scopes),
        "granted_scopes": granted,
        "missing_scopes": missing,
        "receipt_verified": receipt_verified,
        "all_scopes_granted": all_scopes_granted,
        "approval_required": spec.approval_required,
        "approval_verified": approval_verified,
        "external_destination": external,
        "destination_allowed": destination_allowed,
        "policy_clause_results": dict(clause_results),
        "policy_evaluation_verified": policy_verified,
        "policy_satisfied": policy_satisfied,
        "sandboxed": bool(runtime.get("sandboxed", False)),
        "read_only": bool(runtime.get("read_only", False)) or spec.operation == "read",
        "intervention": intervention,
        "intervention_effective": bool(
            runtime.get("intervention_effective", intervention == "none")
        ),
        "intervention_changed_fields": tuple(
            runtime.get("intervention_changed_fields", ())
        ),
        "redacted": redacted,
        "destination_constrained": destination_constrained,
        "capability_restricted": capability_restricted,
        **effects,
    }
    return state


def _branch_from_events(
    *,
    trusted_goal: str,
    policy_id: str,
    clauses: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    specs: Sequence[ToolSpec],
    intervention: str,
) -> PreparedBranch:
    graph_events: List[GraphEvent] = []
    features: List[np.ndarray] = []
    progress: List[np.ndarray] = []
    states: List[Dict[str, Any]] = []
    branch_events: List[Dict[str, Any]] = []
    previous_timestamp: Optional[float] = None
    wall_deltas: List[float] = []
    for event, spec in zip(events, specs):
        event_intervention = intervention if event.get("candidate") else "none"
        branch_event = (
            _apply_candidate_intervention(event, spec, event_intervention)
            if event.get("candidate")
            else copy.deepcopy(dict(event))
        )
        effective_intervention = (
            event_intervention
            if branch_event["runtime"].get(
                "intervention_effective", event_intervention == "none"
            )
            else "none"
        )
        state = _event_state(branch_event, spec, clauses, effective_intervention)
        timestamp = float(branch_event["timestamp"])
        wall_deltas.append(
            0.0 if previous_timestamp is None else timestamp - previous_timestamp
        )
        previous_timestamp = timestamp
        features.append(
            event_feature(
                trusted_goal=trusted_goal,
                policy_clauses=clauses,
                event=branch_event,
                tool_spec=spec,
                state=state,
                intervention=effective_intervention,
            )
        )
        progress.append(security_progress(state))
        graph_events.append(
            GraphEvent(
                event_id=str(branch_event["event_id"]),
                timestamp=timestamp,
                observation=branch_event["observation"],
                action=branch_event["action"],
                tool_spec=spec,
                state=state,
                candidate=bool(branch_event.get("candidate")),
            )
        )
        branch_events.append(branch_event)
        states.append(state)
    graph = build_persistent_graph(
        graph_events,
        policy_id=policy_id,
        policy_clauses=clauses,
        intervention=str(states[-1]["intervention"]),
    )
    candidate_state = states[-1]
    commit_payload = {
        "event_id": branch_events[-1]["event_id"],
        "timestamp": branch_events[-1]["timestamp"],
        "trusted_goal_sha256": hashlib.sha256(
            trusted_goal.encode("utf-8")
        ).hexdigest(),
        "action": branch_events[-1]["action"],
        "permission_receipt": branch_events[-1]["permission_receipt"],
        "actor_id": branch_events[-1]["runtime"].get("actor_id"),
        "actor_verified": branch_events[-1]["runtime"].get("actor_verified"),
        "policy_id": policy_id,
        "policy_clauses": list(clauses),
        "policy_clause_results": branch_events[-1]["runtime"].get(
            "policy_clause_results", {}
        ),
        "effective_intervention": candidate_state["intervention"],
        "registry_tool": specs[-1].to_dict(),
    }
    serialized_commit_payload = json.dumps(
        commit_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    prepared_action_digest = hashlib.sha256(
        serialized_commit_payload.encode("utf-8")
    ).hexdigest()
    fail_reasons: List[str] = []
    if not candidate_state["known_tool"]:
        fail_reasons.append("unknown_tool")
    if candidate_state["unknown_arguments"]:
        fail_reasons.append("unknown_arguments")
    if not candidate_state["receipt_verified"]:
        fail_reasons.append("permission_receipt_unverified")
    if candidate_state["missing_scopes"]:
        fail_reasons.append("missing_scopes")
    if not candidate_state["actor_verified"]:
        fail_reasons.append("actor_unverified")
    if candidate_state["approval_required"] and not candidate_state["approval_verified"]:
        fail_reasons.append("missing_verified_approval")
    if not candidate_state["destination_allowed"]:
        fail_reasons.append("destination_not_allowed")
    if not candidate_state["policy_evaluation_verified"]:
        fail_reasons.append("policy_evaluation_unverified")
    elif not candidate_state["policy_satisfied"]:
        fail_reasons.append("policy_violation")
    if candidate_state["capability_restricted"]:
        fail_reasons.append("capability_restricted")
    if any(not spec.known for spec in specs[:-1]):
        fail_reasons.append("unknown_history_tool")
    return PreparedBranch(
        name=intervention,
        event_features=torch.tensor(np.stack(features), dtype=torch.float32),
        wall_deltas=torch.tensor(wall_deltas, dtype=torch.float32),
        step_positions=torch.arange(len(features), dtype=torch.float32),
        security_progress=torch.tensor(np.stack(progress), dtype=torch.float32),
        graph=graph,
        prepared_action=copy.deepcopy(branch_events[-1]["action"]),
        commit_payload=commit_payload,
        prepared_action_digest=prepared_action_digest,
        metadata={
            "intervention": intervention,
            "effective_intervention": candidate_state["intervention"],
            "intervention_effective": candidate_state["intervention_effective"],
            "intervention_changed_fields": list(
                candidate_state["intervention_changed_fields"]
            ),
            "candidate_event_id": branch_events[-1]["event_id"],
            "candidate_tool": specs[-1].name,
            "candidate_requested_tool": branch_events[-1]["action"]["tool"],
            "candidate_operation": specs[-1].operation,
            "candidate_registry_known": specs[-1].known,
            "prepared_action_digest": prepared_action_digest,
            "candidate_state": candidate_state,
            "fail_closed": bool(fail_reasons),
            "fail_reasons": fail_reasons,
        },
    )


def prepare_request(
    request: Mapping[str, Any],
    registry: ToolRegistry,
    interventions: Sequence[str] = INTERVENTIONS,
) -> PreparedExample:
    """Validate and tensorize one pre-execution request under interventions.

    Unknown tools do not inherit any agent-declared semantics.  They are
    represented by an explicit ``operation='unknown'`` fail-closed ToolSpec,
    which leaves the request inspectable while preserving a mandatory-denial
    signal in every branch.
    """

    request = _require_mapping(request, "request")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    required = {"trusted_goal", "policy", "history", "candidate"}
    missing = required.difference(request)
    if missing:
        raise RequestValidationError(
            f"request is missing required fields: {sorted(missing)}"
        )
    trusted_goal = _strict_text(
        request.get("trusted_goal", ""), "trusted_goal", MAX_TEXT_CHARS
    ).strip()
    if not trusted_goal:
        raise RequestValidationError("trusted_goal must not be empty")
    policy_id, clauses = _normalize_policy(request.get("policy"))
    history_raw = request.get("history")
    if isinstance(history_raw, (str, bytes)) or not isinstance(history_raw, Sequence):
        raise RequestValidationError("history must be a list")
    if len(history_raw) > MAX_HISTORY_EVENTS:
        raise RequestValidationError(
            f"history exceeds the limit of {MAX_HISTORY_EVENTS} events"
        )
    events = [
        _normalize_event(event, candidate=False) for event in history_raw
    ]
    events.append(_normalize_event(request.get("candidate"), candidate=True))
    request_text_budget = len(trusted_goal) + sum(
        len(clause["text"]) for clause in clauses
    )
    request_text_budget += sum(
        len(event["observation"]["content"])
        + len(
            json.dumps(
                event["action"]["arguments"],
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        for event in events
    )
    if request_text_budget > MAX_REQUEST_TEXT_CHARS:
        raise RequestValidationError(
            f"request text budget exceeds {MAX_REQUEST_TEXT_CHARS} characters"
        )
    ids = [event["event_id"] for event in events]
    if len(ids) != len(set(ids)):
        raise RequestValidationError("event_id values must be unique")
    timestamps = [float(event["timestamp"]) for event in events]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise RequestValidationError("event timestamps must be non-decreasing")

    names = tuple(str(name) for name in interventions)
    if not names:
        raise RequestValidationError("at least one intervention is required")
    if len(names) != len(set(names)):
        raise RequestValidationError("interventions must not contain duplicates")
    invalid = [name for name in names if name not in INTERVENTIONS]
    if invalid:
        raise RequestValidationError(f"unknown interventions: {invalid}")
    specs = [
        registry.resolve_or_fail_closed(event["action"]["tool"]) for event in events
    ]
    branches = {
        name: _branch_from_events(
            trusted_goal=trusted_goal,
            policy_id=policy_id,
            clauses=clauses,
            events=events,
            specs=specs,
            intervention=name,
        )
        for name in names
    }
    unknown_events = [
        event["event_id"] for event, spec in zip(events, specs) if not spec.known
    ]
    return PreparedExample(
        branches=branches,
        metadata={
            "request_id": bounded_text(request.get("request_id", ids[-1]), 240),
            "trusted_goal": trusted_goal,
            "policy_id": policy_id,
            "policy_clauses": clauses,
            "registry_version": registry.version,
            "event_ids": ids,
            "history_length": len(events) - 1,
            "interventions": list(names),
            "unknown_tool_events": unknown_events,
            "fail_closed": bool(unknown_events),
            "limits": {
                "history_events": MAX_HISTORY_EVENTS,
                "policy_clauses": MAX_POLICY_CLAUSES,
                "required_scopes": MAX_REQUIRED_SCOPES,
                "text_chars": MAX_TEXT_CHARS,
            },
        },
    )


def _collate_branch(branches: Sequence[PreparedBranch]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor(
        [branch.event_features.shape[0] for branch in branches], dtype=torch.long
    )
    event_features = pad_sequence(
        [branch.event_features for branch in branches], batch_first=True
    )
    wall_deltas = pad_sequence(
        [branch.wall_deltas for branch in branches], batch_first=True
    )
    step_positions = pad_sequence(
        [branch.step_positions for branch in branches], batch_first=True
    )
    progress = pad_sequence(
        [branch.security_progress for branch in branches], batch_first=True
    )
    sequence_mask = (
        torch.arange(event_features.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    )

    node_features: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_types: List[torch.Tensor] = []
    edge_states: List[torch.Tensor] = []
    node_batch: List[torch.Tensor] = []
    edge_batch: List[torch.Tensor] = []
    candidate_indices: List[int] = []
    offset = 0
    for batch_index, branch in enumerate(branches):
        graph = branch.graph
        node_count = int(graph.node_features.shape[0])
        edge_count = int(graph.edge_types.shape[0])
        node_features.append(graph.node_features)
        edge_indices.append(graph.edge_index + offset)
        edge_types.append(graph.edge_types)
        edge_states.append(graph.edge_state)
        node_batch.append(torch.full((node_count,), batch_index, dtype=torch.long))
        edge_batch.append(torch.full((edge_count,), batch_index, dtype=torch.long))
        candidate_indices.append(offset + graph.candidate_node_index)
        offset += node_count
    return {
        "event_features": event_features,
        "wall_deltas": wall_deltas,
        "step_positions": step_positions,
        "security_progress": progress,
        "sequence_mask": sequence_mask,
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_types": torch.cat(edge_types, dim=0),
        "edge_state": torch.cat(edge_states, dim=0),
        "node_batch": torch.cat(node_batch, dim=0),
        "candidate_node_indices": torch.tensor(candidate_indices, dtype=torch.long),
        "edge_batch": torch.cat(edge_batch, dim=0),
    }


def collate_prepared(examples: Sequence[PreparedExample]) -> Dict[str, Any]:
    """Collate examples while retaining readable graph metadata out of tensors."""

    if not examples:
        raise ValueError("cannot collate an empty sequence")
    expected_names = tuple(examples[0].branches)
    if any(tuple(example.branches) != expected_names for example in examples[1:]):
        raise ValueError("all examples in a batch must have identical intervention branches")
    branch_batches = {
        name: _collate_branch([example.branches[name] for example in examples])
        for name in expected_names
    }
    branch_metadata: Dict[str, List[Dict[str, Any]]] = {}
    for name in expected_names:
        values = []
        for example in examples:
            branch = example.branches[name]
            values.append(
                {
                    **branch.metadata,
                    "node_metadata": branch.graph.node_metadata,
                    "edge_metadata": branch.graph.edge_metadata,
                }
            )
        branch_metadata[name] = values
    return {
        "branches": branch_batches,
        "metadata": {
            "examples": [example.metadata for example in examples],
            "branches": branch_metadata,
        },
    }


__all__ = [
    "MAX_ARGUMENTS",
    "MAX_HISTORY_EVENTS",
    "MAX_POLICY_CLAUSES",
    "PreparedBranch",
    "PreparedExample",
    "RequestValidationError",
    "collate_prepared",
    "prepare_request",
]
