"""Bounded Unicode-aware feature extraction for Agent New.

Text is represented with stable signed hashes over role-separated Unicode
character n-grams and UTF-8 byte n-grams.  This keeps Chinese and other
non-ASCII content observable without a mutable vocabulary while remaining
deterministic across Python processes.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .constants import (
    EVENT_FEATURE_DIM,
    EVENT_FLAG_DIM,
    EVENT_HASH_DIM,
    NODE_FEATURE_DIM,
    NODE_FLAG_DIM,
    NODE_HASH_DIM,
    NODE_TYPES,
    OPERATIONS,
)
from .registry import ToolSpec


MAX_TEXT_CHARS = 4096
MAX_HASH_PART_CHARS = 2048
MAX_HASH_PARTS = 256
MAX_NGRAMS_PER_PART = 1024


def bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    """Return valid, bounded Unicode while retaining both ends of long text."""

    if limit <= 0:
        return ""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if len(text) <= limit:
        return text
    marker = " …<truncated>… "
    remaining = max(0, limit - len(marker))
    left = remaining // 2
    right = remaining - left
    return text[:left] + marker + (text[-right:] if right else "")


def canonical_json(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return bounded_text(text, limit)


def _signed_bucket(payload: bytes, dimension: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(payload, digest_size=16, person=b"agent-new-hash").digest()
    bucket = int.from_bytes(digest[:8], "little") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


def _add_feature(vector: np.ndarray, role: str, family: str, value: bytes) -> None:
    prefix = (role + "\x1f" + family + "\x1f").encode("utf-8")
    bucket, sign = _signed_bucket(prefix + value, vector.shape[0])
    vector[bucket] += sign


def stable_unicode_hash(
    parts: Iterable[Tuple[str, Any]], dimension: int
) -> np.ndarray:
    """Hash role-separated Unicode character and UTF-8 byte n-grams.

    A field marker and whole-field digest preserve roles and coarse ordering;
    local n-grams provide useful overlap for unseen text.  Work is explicitly
    bounded to prevent oversized observations from causing unbounded latency.
    """

    if dimension <= 0:
        raise ValueError("hash dimension must be positive")
    vector = np.zeros(dimension, dtype=np.float32)
    for part_index, (raw_role, raw_value) in enumerate(parts):
        if part_index >= MAX_HASH_PARTS:
            break
        role = bounded_text(raw_role, 80).casefold() or "field"
        text = bounded_text(raw_value, MAX_HASH_PART_CHARS).casefold()
        encoded = text.encode("utf-8")
        _add_feature(vector, role, "field", encoded)
        produced = 0
        for width in (1, 2, 3):
            if len(text) < width:
                continue
            for index in range(len(text) - width + 1):
                _add_feature(
                    vector,
                    role,
                    "char" + str(width),
                    text[index : index + width].encode("utf-8"),
                )
                produced += 1
                if produced >= MAX_NGRAMS_PER_PART:
                    break
            if produced >= MAX_NGRAMS_PER_PART:
                break
        for width in (2, 3, 4):
            if produced >= MAX_NGRAMS_PER_PART or len(encoded) < width:
                break
            for index in range(len(encoded) - width + 1):
                _add_feature(
                    vector,
                    role,
                    "byte" + str(width),
                    encoded[index : index + width],
                )
                produced += 1
                if produced >= MAX_NGRAMS_PER_PART:
                    break
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _effect_present(spec: ToolSpec, *needles: str) -> bool:
    values = " ".join(spec.side_effects).casefold()
    return any(needle in values for needle in needles)


def side_effect_flags(spec: ToolSpec) -> Mapping[str, bool]:
    operation = spec.operation
    return {
        "network_effect": _effect_present(
            spec, "network", "external", "upload", "send", "http"
        )
        or operation == "write_external",
        "filesystem_effect": _effect_present(
            spec, "filesystem", "file_write", "storage", "delete"
        )
        or operation in {"write_internal", "delete"},
        "code_execution": _effect_present(
            spec, "execute", "code", "shell", "process"
        )
        or operation == "execute",
        "irreversible": _effect_present(
            spec, "irreversible", "delete", "destructive", "permission_change"
        )
        or operation in {"delete", "permission_change"},
    }


def _policy_parts(policy_clauses: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Any]]:
    result: List[Tuple[str, Any]] = []
    for index, clause in enumerate(policy_clauses):
        result.append(("policy_clause_id", clause.get("id", str(index))))
        result.append(("policy_clause_text", clause.get("text", "")))
    return result


def event_feature(
    *,
    trusted_goal: str,
    policy_clauses: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    tool_spec: ToolSpec,
    state: Mapping[str, Any],
    intervention: str,
) -> np.ndarray:
    """Return one fixed-size event vector using registry-owned semantics."""

    observation = event.get("observation", {})
    action = event.get("action", {})
    arguments = action.get("arguments", {})
    if not isinstance(arguments, Mapping):
        arguments = {}
    parts: List[Tuple[str, Any]] = [
        ("trusted_goal", trusted_goal),
        ("observation_content", observation.get("content", "")),
        ("observation_source_id", state.get("source_id", "unknown-source")),
        ("observation_origin", state.get("source_origin", "unknown")),
        ("registry_tool", tool_spec.name),
        ("registry_operation", tool_spec.operation),
        ("registry_side_effects", canonical_json(tool_spec.side_effects)),
        ("intervention", intervention),
    ]
    parts.extend(_policy_parts(policy_clauses))
    for name in sorted(arguments):
        role = tool_spec.role_for(str(name)) or "untyped"
        parts.append(("argument_name", name))
        parts.append(("argument:" + role, canonical_json(arguments[name])))
    semantic = stable_unicode_hash(parts, EVENT_HASH_DIM)
    operation = np.zeros(len(OPERATIONS), dtype=np.float32)
    operation[OPERATIONS.index(tool_spec.operation)] = 1.0
    flags = np.array(
        [
            float(bool(state.get("candidate"))),
            float(bool(state.get("source_untrusted"))),
            float(bool(state.get("source_verified"))),
            float(bool(state.get("contains_sensitive"))),
            float(bool(state.get("external_destination"))),
            float(bool(state.get("destination_allowed"))),
            float(bool(state.get("irreversible"))),
            float(bool(state.get("network_effect"))),
            float(bool(state.get("filesystem_effect"))),
            float(bool(state.get("code_execution"))),
            float(bool(state.get("all_scopes_granted"))),
            float(bool(state.get("missing_scopes"))),
            float(bool(state.get("approval_required"))),
            float(bool(state.get("approval_verified"))),
            float(bool(state.get("policy_satisfied"))),
            float(bool(state.get("sandboxed"))),
            float(bool(state.get("read_only"))),
            float(bool(state.get("instruction_like"))),
            float(not tool_spec.known),
            float(intervention != "none"),
            float(bool(state.get("redacted"))),
            float(bool(state.get("destination_constrained"))),
            float(bool(state.get("capability_restricted"))),
            1.0,
        ],
        dtype=np.float32,
    )
    if flags.shape != (EVENT_FLAG_DIM,):
        raise AssertionError("event flag ontology does not match EVENT_FLAG_DIM")
    vector = np.concatenate([semantic, operation, flags])
    if vector.shape != (EVENT_FEATURE_DIM,):
        raise AssertionError(
            f"event feature has shape {vector.shape}, expected {(EVENT_FEATURE_DIM,)}"
        )
    return vector


def node_feature(
    node_type: str,
    entity_id: str,
    value: Any,
    flags: Mapping[str, Any],
) -> np.ndarray:
    """Return one role-separated heterogeneous-node feature vector."""

    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown node type: {node_type!r}")
    semantic = stable_unicode_hash(
        [
            ("node_type", node_type),
            ("entity_id", entity_id),
            ("semantic_value", canonical_json(value)),
        ],
        NODE_HASH_DIM,
    )
    type_vector = np.zeros(len(NODE_TYPES), dtype=np.float32)
    type_vector[NODE_TYPES.index(node_type)] = 1.0
    flag_vector = np.array(
        [
            float(bool(flags.get("candidate"))),
            float(bool(flags.get("untrusted"))),
            float(bool(flags.get("verified"))),
            float(bool(flags.get("sensitive"))),
            float(bool(flags.get("external"))),
            float(bool(flags.get("permission_granted"))),
            float(bool(flags.get("approval_required"))),
            float(bool(flags.get("approval_verified"))),
            float(bool(flags.get("irreversible"))),
            float(bool(flags.get("sandboxed"))),
            float(bool(flags.get("policy_clause"))),
            float(bool(flags.get("intervention"))),
            float(bool(flags.get("unknown_capability"))),
            float(bool(flags.get("redacted"))),
            float(bool(flags.get("restricted"))),
            1.0,
        ],
        dtype=np.float32,
    )
    if flag_vector.shape != (NODE_FLAG_DIM,):
        raise AssertionError("node flag ontology does not match NODE_FLAG_DIM")
    vector = np.concatenate([semantic, type_vector, flag_vector])
    if vector.shape != (NODE_FEATURE_DIM,):
        raise AssertionError(
            f"node feature has shape {vector.shape}, expected {(NODE_FEATURE_DIM,)}"
        )
    return vector


def security_progress(state: Mapping[str, Any]) -> np.ndarray:
    """Five explicit security-progress coordinates for one event."""

    missing = state.get("missing_scopes", ())
    required = state.get("required_scopes", ())
    privilege_gap = len(missing) / max(1, len(required))
    values = np.array(
        [
            float(bool(state.get("source_untrusted")) or not bool(state.get("source_verified"))),
            float(bool(state.get("contains_sensitive"))),
            float(min(1.0, privilege_gap)),
            float(bool(state.get("external_destination"))),
            float(
                bool(state.get("irreversible"))
                or bool(state.get("code_execution"))
                or bool(state.get("unknown_capability"))
            ),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("security progress contains non-finite values")
    return values


__all__ = [
    "MAX_HASH_PART_CHARS",
    "MAX_HASH_PARTS",
    "MAX_TEXT_CHARS",
    "bounded_text",
    "canonical_json",
    "event_feature",
    "node_feature",
    "security_progress",
    "side_effect_flags",
    "stable_unicode_hash",
]
