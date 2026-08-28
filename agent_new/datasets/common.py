"""Shared normalized trajectory schema and weak-label conversion."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from agent_new.constants import CONCEPTS, DEFAULT_EVENT_TYPES
from agent_new.registry import ToolRegistry, ToolSpec


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:@-]+")


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bounded_text(value: Any, limit: int = 4096) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " …<dataset-truncated>… " + text[-limit // 2 :]


def canonical_tool_name(value: Any) -> str:
    original = str(value or "dataset_action").strip()
    safe = re.sub(r"[^A-Za-z0-9_.:/@-]", "_", original)[:120]
    if not safe:
        safe = "dataset_action"
    if safe != original:
        safe += "_" + stable_digest(original)[:10]
    return safe


def canonical_argument_name(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:/@-]", "_", str(value))[:120]
    return safe or "value"


def parse_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {canonical_argument_name(key): item for key, item in value.items()}
    text = str(value or "").strip()
    if not text:
        return {}
    candidates = [text]
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.insert(0, text[left : right + 1])
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, Mapping):
                    return {
                        canonical_argument_name(key): item
                        for key, item in parsed.items()
                    }
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return {"raw": bounded_text(text, 2000)}


def infer_operation(tool: str, arguments: Mapping[str, Any]) -> str:
    text = tool.casefold()
    names = {name.casefold() for name in arguments}
    if any(word in text for word in ("delete", "remove", "purge", "erase", "drop")):
        return "delete"
    if any(word in text for word in ("bash", "shell", "terminal", "exec", "run_code", "python")):
        return "execute"
    if any(word in text for word in ("grant", "revoke", "permission", "chmod", "chown")):
        return "permission_change"
    if any(word in text for word in ("send", "email", "post", "upload", "publish", "webhook")):
        if names.intersection({"recipient", "destination", "url", "endpoint", "to"}):
            return "write_external"
        return "write_internal"
    if any(word in text for word in ("write", "edit", "save", "update", "create")):
        return "write_internal"
    if any(word in text for word in ("read", "view", "get", "list", "search", "query", "open", "find")):
        return "read"
    if any(word in text for word in ("finish", "done", "complete")):
        return "finish"
    return "other"


def argument_role(name: str, operation: str) -> str:
    lowered = name.casefold()
    if lowered in {"recipient", "destination", "url", "endpoint", "to"}:
        return "destination"
    if lowered in {"command", "cmd", "script", "code"}:
        return "command"
    if lowered in {
        "content",
        "body",
        "payload",
        "attachment",
        "file",
        "path",
        "record",
        "query",
        "text",
    }:
        return "data"
    return "identifier"


def _tool_spec(name: str, argument_names: Iterable[str]) -> ToolSpec:
    arguments = tuple(sorted(set(argument_names)))
    provisional = {arg: None for arg in arguments}
    operation = infer_operation(name, provisional)
    roles = {arg: argument_role(arg, operation) for arg in arguments}
    if operation == "write_external" and not any(
        role == "destination" for role in roles.values()
    ):
        operation = "write_internal"
    scopes = {
        "read": ("data:read",),
        "write_internal": ("data:write",),
        "write_external": ("data:read", "external:send"),
        "execute": ("code:execute",),
        "delete": ("data:delete",),
        "permission_change": ("permission:admin",),
    }.get(operation, ())
    approval = operation in {
        "write_external",
        "execute",
        "delete",
        "permission_change",
    }
    effects = {
        "write_internal": ("internal_write",),
        "write_external": ("network", "external_transfer"),
        "execute": ("code_execution", "system_change"),
        "delete": ("data_deletion", "irreversible"),
        "permission_change": ("permission_change", "irreversible"),
    }.get(operation, ())
    return ToolSpec(
        name=name,
        operation=operation,
        required_scopes=scopes,
        approval_required=approval,
        side_effects=effects,
        argument_roles=roles,
    )


@dataclass
class NormalizedTrajectory:
    source_dataset: str
    source_family: str
    record_id: str
    group_id: str
    trusted_goal: str
    events: List[Dict[str, Any]]
    unsafe: bool
    event_type: str
    risk_text: str = ""
    label_quality: str = "trajectory_label"
    concept_supervision: float = 0.0
    attack_intent: Optional[int] = None
    utility: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_tool_registry(
    trajectories: Sequence[NormalizedTrajectory], version: str
) -> ToolRegistry:
    arguments_by_tool: Dict[str, set] = {}
    for trajectory in trajectories:
        for event in trajectory.events:
            action = event["action"]
            tool = canonical_tool_name(action.get("tool"))
            arguments_by_tool.setdefault(tool, set()).update(
                map(str, action.get("arguments", {}))
            )
    specs = [_tool_spec(name, names) for name, names in sorted(arguments_by_tool.items())]
    return ToolRegistry(specs, version=version)


def risk_type_from_text(value: str) -> str:
    text = str(value).casefold()
    mapping = (
        (("privacy", "exfiltrat", "credential", "secret"), "privacy_leakage"),
        (("security", "cyber", "malware", "prompt injection", "unauthorized access"), "computer_security"),
        (("financial", "money", "payment", "fraud"), "financial_loss"),
        (("data loss", "delete", "destruction", "tamper"), "data_loss"),
    )
    for needles, event_type in mapping:
        if any(needle in text for needle in needles):
            return event_type
    return "unauthorized_external_action"


def heuristic_concepts(
    event: Mapping[str, Any], unsafe: bool, terminal: bool, risk_text: str
) -> Dict[str, int]:
    action = event["action"]
    operation = infer_operation(str(action.get("tool", "")), action.get("arguments", {}))
    observation = event.get("observation", {})
    text = str(risk_text).casefold()
    values = {
        "goal_deviation": int(unsafe and terminal),
        "untrusted_provenance": int(observation.get("trust_level") == "untrusted"),
        "privilege_excess": int(unsafe and terminal and "unauthorized" in text),
        "sensitive_data_access": int(unsafe and operation == "read" and "data" in text),
        "sensitive_data_flow": int(unsafe and operation == "write_external"),
        "anomalous_destination": int(unsafe and operation == "write_external"),
        "missing_approval": int(unsafe and terminal and "approval" in text),
        "irreversibility": int(unsafe and operation in {"delete", "execute", "permission_change"}),
        "unknown_capability": 0,
    }
    return {name: int(values.get(name, 0)) for name in CONCEPTS}


def _request_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "timestamp": float(event["timestamp"]),
        "observation": event["observation"],
        "action": event["action"],
        "permission_receipt": {
            "granted_scopes": [],
            "approval_verified": False,
            "receipt_verified": False,
            "receipt_id": "dataset-metadata-unavailable",
        },
        "runtime": {
            "actor_verified": False,
            "sandboxed": False,
            "read_only": False,
            "user_confirmed": False,
            "sensitive_data_ids": [],
            "known_destinations": [],
            "policy_evaluation_verified": False,
            "policy_clause_results": {},
        },
    }


def prefix_rows(
    trajectory: NormalizedTrajectory,
    split: str,
    *,
    horizon: int = 5,
    history_limit: int = 32,
) -> Iterator[Dict[str, Any]]:
    if not trajectory.events:
        return
    terminal = len(trajectory.events) - 1
    violation_step = terminal if trajectory.unsafe else None
    policy = {
        "id": f"dataset-policy-{trajectory.source_family}",
        "clauses": [
            {
                "id": "generic-safe-completion",
                "text": "Complete the trusted task without unsafe tool effects or policy violations.",
            }
        ],
    }
    for step, event in enumerate(trajectory.events):
        distance = violation_step - step + 1 if violation_step is not None else None
        observed = bool(distance is not None and 1 <= distance <= horizon)
        remaining_observed = len(trajectory.events) - step
        censor_time = horizon if trajectory.unsafe and not observed else min(
            horizon, remaining_observed
        )
        concepts = heuristic_concepts(
            event, trajectory.unsafe, step == terminal, trajectory.risk_text
        )
        request = {
            "request_id": f"{trajectory.source_dataset}:{trajectory.record_id}:{step}",
            "trusted_goal": bounded_text(trajectory.trusted_goal, 4096),
            "policy": policy,
            "history": [
                _request_event(item)
                for item in trajectory.events[max(0, step - history_limit) : step]
            ],
            "candidate": _request_event(event),
        }
        labels = {
            "event_observed": observed,
            "event_time": int(distance) if observed else 0,
            "event_type": trajectory.event_type if observed else None,
            "censor_time": int(censor_time),
            "concept_targets": concepts,
            "concept_mask": {
                name: float(trajectory.concept_supervision) for name in CONCEPTS
            },
            "attack_intent_target": trajectory.attack_intent,
            "supervision_attack_intent": float(
                trajectory.attack_intent is not None
            ),
            "utility_target": trajectory.utility,
            "supervision_utility": float(trajectory.utility is not None),
            "supervision_survival": 1.0,
            "weak_terminal_proxy": bool(trajectory.unsafe),
        }
        yield {
            "request": request,
            "labels": labels,
            "tracking": {
                "source_dataset": trajectory.source_dataset,
                "source_family": trajectory.source_family,
                "record_id": trajectory.record_id,
                "trajectory_id": f"{trajectory.source_dataset}:{trajectory.record_id}",
                "group_id": trajectory.group_id,
                "step_index": step,
                "split": split,
                "label_quality": trajectory.label_quality,
                **trajectory.metadata,
            },
        }


def normalized_conversation_group(messages: Any) -> str:
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).casefold()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return stable_digest(text)


def assign_split(group_id: str, seed: int) -> str:
    bucket = int(stable_digest([seed, group_id])[:12], 16) % 10000
    if bucket < 8000:
        return "train"
    if bucket < 9000:
        return "validation"
    return "calibration"


__all__ = [
    "NormalizedTrajectory",
    "assign_split",
    "bounded_text",
    "build_tool_registry",
    "canonical_tool_name",
    "normalized_conversation_group",
    "parse_arguments",
    "prefix_rows",
    "risk_type_from_text",
    "stable_digest",
]
