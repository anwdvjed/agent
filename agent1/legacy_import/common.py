"""Shared normalized trajectory schema and weak-label conversion."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from agent1.constants import CONCEPTS, DEFAULT_EVENT_TYPES
from agent1.registry import ToolRegistry, ToolSpec


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


