"""Persistent heterogeneous provenance, permission and intervention graphs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import EDGE_STATE_DIM, EDGE_TYPES, NODE_TYPES
from .features import bounded_text, canonical_json, node_feature
from .registry import ToolSpec


DATA_ARGUMENT_ROLES = {
    "data",
    "payload",
    "content",
    "attachment",
    "resource",
    "record",
    "query",
    "command",
    "identifier",
}
DESTINATION_ARGUMENT_ROLES = {
    "destination",
    "recipient",
    "url",
    "endpoint",
    "path_destination",
    "external_destination",
}


@dataclass(frozen=True)
class GraphEvent:
    event_id: str
    timestamp: float
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    tool_spec: ToolSpec
    state: Mapping[str, Any]
    candidate: bool = False


@dataclass
class GraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    edge_state: torch.Tensor
    candidate_node_index: int
    node_metadata: List[Dict[str, Any]]
    edge_metadata: List[Dict[str, Any]]


@dataclass
class _NodeRecord:
    node_type: str
    entity_id: str
    value: Any
    flags: Dict[str, bool]
    metadata: Dict[str, Any]


def _entity_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:20]


def _safe_preview(value: Any, *, sensitive: bool = False) -> str:
    if sensitive:
        return "[sensitive-value-hidden]"
    return bounded_text(canonical_json(value, 512), 512)


def _edge_state(
    state: Mapping[str, Any],
    *,
    edge_age: float = 0.0,
    permission_granted: Optional[bool] = None,
    active: Optional[bool] = None,
    intervened: Optional[bool] = None,
    policy_result: Optional[bool] = None,
) -> np.ndarray:
    approval_ok = not bool(state.get("approval_required")) or bool(
        state.get("approval_verified")
    )
    values = np.array(
        [
            float(
                (not bool(state.get("capability_restricted")) and bool(state.get("known_tool", True)))
                if active is None
                else active
            ),
            float(
                bool(state.get("all_scopes_granted"))
                if permission_granted is None
                else permission_granted
            ),
            float(approval_ok),
            float(bool(state.get("source_verified")) and not bool(state.get("source_untrusted"))),
            float(bool(state.get("contains_sensitive"))),
            float(bool(state.get("external_destination"))),
            float(bool(state.get("irreversible")) or bool(state.get("code_execution"))),
            float(bool(state.get("intervention", "none") != "none") if intervened is None else intervened),
            math.tanh(math.log1p(max(0.0, float(edge_age))) / 10.0),
            1.0 if policy_result is True else -1.0 if policy_result is False else 0.0,
        ],
        dtype=np.float32,
    )
    if values.shape != (EDGE_STATE_DIM,):
        raise AssertionError("edge-state ontology does not match EDGE_STATE_DIM")
    return values


class _PersistentGraphBuilder:
    def __init__(self) -> None:
        self.nodes: List[_NodeRecord] = []
        self.node_lookup: Dict[Tuple[str, str], int] = {}
        self.edge_pairs: List[Tuple[int, int]] = []
        self.edge_types: List[int] = []
        self.edge_states: List[np.ndarray] = []
        self.edge_metadata: List[Dict[str, Any]] = []

    def node(
        self,
        node_type: str,
        entity_id: str,
        value: Any,
        *,
        flags: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        if node_type not in NODE_TYPES:
            raise ValueError(f"unknown graph node type: {node_type!r}")
        key = (node_type, str(entity_id))
        existing = self.node_lookup.get(key)
        if existing is not None:
            record = self.nodes[existing]
            for name, enabled in (flags or {}).items():
                if record.node_type == "data" and name == "sensitive":
                    # Sensitivity is an object-level taint and remains sticky.
                    record.flags[name] = bool(
                        record.flags.get(name, False) or enabled
                    )
                else:
                    # Trust, permission, approval and destination flags are
                    # time-varying.  The persistent entity stores the latest
                    # state; historical state remains on event-specific edges.
                    record.flags[name] = bool(enabled)
            return existing
        index = len(self.nodes)
        record = _NodeRecord(
            node_type=node_type,
            entity_id=bounded_text(entity_id, 240),
            value=value,
            flags={name: bool(enabled) for name, enabled in (flags or {}).items()},
            metadata={str(key): value for key, value in (metadata or {}).items()},
        )
        self.nodes.append(record)
        self.node_lookup[key] = index
        return index

    def edge(
        self,
        source: int,
        target: int,
        relation: str,
        state: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        if relation not in EDGE_TYPES:
            raise ValueError(f"unknown graph edge type: {relation!r}")
        if tuple(state.shape) != (EDGE_STATE_DIM,):
            raise ValueError("edge state has the wrong dimension")
        self.edge_pairs.append((source, target))
        self.edge_types.append(EDGE_TYPES.index(relation))
        self.edge_states.append(state.astype(np.float32, copy=False))
        self.edge_metadata.append(
            {
                "relation": relation,
                **{str(key): value for key, value in metadata.items()},
            }
        )

    def tensors(self, candidate_node_index: int) -> GraphTensors:
        if candidate_node_index < 0:
            raise ValueError("candidate action node was not constructed")
        features = np.stack(
            [
                node_feature(record.node_type, record.entity_id, record.value, record.flags)
                for record in self.nodes
            ]
        )
        if self.edge_pairs:
            edge_index = torch.tensor(self.edge_pairs, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(self.edge_types, dtype=torch.long)
            edge_state = torch.tensor(np.stack(self.edge_states), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_types = torch.empty((0,), dtype=torch.long)
            edge_state = torch.empty((0, EDGE_STATE_DIM), dtype=torch.float32)
        node_metadata = []
        for index, record in enumerate(self.nodes):
            node_metadata.append(
                {
                    "index": index,
                    "type": record.node_type,
                    "entity_id": record.entity_id,
                    "flags": dict(sorted(record.flags.items())),
                    **record.metadata,
                }
            )
        edge_metadata = [
            {
                "index": index,
                "source": pair[0],
                "target": pair[1],
                **metadata,
            }
            for index, (pair, metadata) in enumerate(
                zip(self.edge_pairs, self.edge_metadata)
            )
        ]
        return GraphTensors(
            node_features=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            edge_types=edge_types,
            edge_state=edge_state,
            candidate_node_index=candidate_node_index,
            node_metadata=node_metadata,
            edge_metadata=edge_metadata,
        )


def _argument_entities(
    builder: _PersistentGraphBuilder,
    event: GraphEvent,
    action_node: int,
    edge_age: float,
) -> None:
    arguments = event.action.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return
    state = event.state
    base_state = _edge_state(state, edge_age=edge_age)
    for argument_name in sorted(arguments):
        role = event.tool_spec.role_for(str(argument_name))
        if role is None:
            continue
        value = arguments[argument_name]
        if role in DESTINATION_ARGUMENT_ROLES:
            destination_id = "destination:" + _entity_digest([role, value])
            destination = builder.node(
                "destination",
                destination_id,
                {"role": role, "value": value},
                flags={
                    "external": bool(state.get("external_destination")),
                    "verified": bool(state.get("destination_allowed")),
                },
                metadata={
                    "role": role,
                    "label": bounded_text(argument_name, 120),
                    "value_preview": _safe_preview(value),
                },
            )
            builder.edge(
                action_node,
                destination,
                "sends_to",
                base_state,
                {
                    "event_id": event.event_id,
                    "argument": argument_name,
                    "role": role,
                },
            )
        elif role in DATA_ARGUMENT_ROLES:
            sensitive = bool(state.get("contains_sensitive")) and role != "command"
            original_sensitive = bool(state.get("original_sensitive")) and role != "command"
            data_id = "data:" + _entity_digest([role, value])
            data = builder.node(
                "data",
                data_id,
                {"role": role, "value": "[redacted]" if state.get("redacted") else value},
                flags={
                    "sensitive": sensitive,
                    "redacted": bool(state.get("redacted")),
                },
                metadata={
                    "role": role,
                    "label": bounded_text(argument_name, 120),
                    "value_preview": _safe_preview(
                        value, sensitive=original_sensitive or sensitive
                    ),
                },
            )
            builder.edge(
                data,
                action_node,
                "uses_data",
                base_state,
                {
                    "event_id": event.event_id,
                    "argument": argument_name,
                    "role": role,
                },
            )


def build_persistent_graph(
    events: Sequence[GraphEvent],
    *,
    policy_id: str,
    policy_clauses: Sequence[Mapping[str, Any]],
    intervention: str,
) -> GraphTensors:
    """Build one accumulated graph with entity reuse and time-local edge state."""

    if not events or not events[-1].candidate:
        raise ValueError("graph events must end with the candidate action")
    builder = _PersistentGraphBuilder()
    policy_nodes: List[Tuple[Mapping[str, Any], int]] = []
    for index, clause in enumerate(policy_clauses):
        clause_id = bounded_text(clause.get("id", str(index)), 160)
        text = bounded_text(clause.get("text", ""), 2048)
        node = builder.node(
            "policy",
            f"policy:{policy_id}:clause:{clause_id}",
            {"policy_id": policy_id, "clause_id": clause_id, "text": text},
            flags={"policy_clause": True, "verified": True},
            metadata={
                "policy_id": policy_id,
                "clause_id": clause_id,
                "text": text,
            },
        )
        policy_nodes.append((clause, node))

    previous_action: Optional[int] = None
    candidate_node = -1
    candidate_timestamp = float(events[-1].timestamp)
    for event in events:
        state = event.state
        edge_age = max(0.0, candidate_timestamp - float(event.timestamp))
        actor_id = bounded_text(state.get("actor_id", "agent"), 160)
        actor = builder.node(
            "actor",
            "actor:" + actor_id,
            actor_id,
            flags={"verified": bool(state.get("actor_verified"))},
            metadata={"actor_id": actor_id},
        )
        source_id = bounded_text(state.get("source_id", "unknown-source"), 200)
        source = builder.node(
            "source",
            "source:" + source_id,
            {
                "source_id": source_id,
                "origin": state.get("source_origin", "unknown"),
            },
            flags={
                "untrusted": bool(state.get("source_untrusted")),
                "verified": bool(state.get("source_verified")),
            },
            metadata={
                "source_id": source_id,
                "origin": bounded_text(state.get("source_origin", "unknown"), 240),
            },
        )
        observation_content = event.observation.get("content", "")
        observation_data_id = "observation:" + _entity_digest(
            [source_id, observation_content]
        )
        observation_data = builder.node(
            "data",
            observation_data_id,
            {
                "source_id": source_id,
                "content": "[redacted]" if state.get("redacted") else observation_content,
            },
            flags={
                "sensitive": bool(state.get("contains_sensitive")),
                "redacted": bool(state.get("redacted")),
            },
            metadata={
                "label": "observation",
                "value_preview": _safe_preview(
                    observation_content,
                    sensitive=bool(state.get("original_sensitive"))
                    or bool(state.get("contains_sensitive")),
                ),
            },
        )
        action_entity_id = "action:" + bounded_text(event.event_id, 200)
        action_node = builder.node(
            "action",
            action_entity_id,
            {
                "event_id": event.event_id,
                "tool": event.tool_spec.name,
                "operation": event.tool_spec.operation,
                "side_effects": event.tool_spec.side_effects,
            },
            flags={
                "candidate": event.candidate,
                "approval_required": bool(state.get("approval_required")),
                "approval_verified": bool(state.get("approval_verified")),
                "irreversible": bool(state.get("irreversible")),
                "sandboxed": bool(state.get("sandboxed")),
                "unknown_capability": not event.tool_spec.known,
                "restricted": bool(state.get("capability_restricted")),
            },
            metadata={
                "event_id": event.event_id,
                "tool": event.tool_spec.name,
                "requested_tool": bounded_text(event.action.get("tool", ""), 160),
                "operation": event.tool_spec.operation,
                "candidate": event.candidate,
                "registry_known": event.tool_spec.known,
            },
        )
        if event.candidate:
            candidate_node = action_node
        common = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "candidate": event.candidate,
        }
        state_vector = _edge_state(state, edge_age=edge_age)
        builder.edge(actor, action_node, "proposes", state_vector, common)
        builder.edge(source, observation_data, "derived_from", state_vector, common)
        builder.edge(source, action_node, "influences", state_vector, common)
        builder.edge(observation_data, action_node, "uses_data", state_vector, common)

        _argument_entities(builder, event, action_node, edge_age)

        granted_scopes = set(state.get("granted_scopes", ()))
        for scope in event.tool_spec.required_scopes:
            granted = scope in granted_scopes and bool(state.get("receipt_verified"))
            permission = builder.node(
                "permission",
                f"permission:{actor_id}:scope:{scope}",
                {"actor_id": actor_id, "scope": scope},
                flags={
                    "permission_granted": granted,
                    "verified": bool(state.get("receipt_verified")),
                },
                metadata={"actor_id": actor_id, "scope": scope},
            )
            builder.edge(
                permission,
                action_node,
                "authorizes",
                _edge_state(
                    state,
                    edge_age=edge_age,
                    permission_granted=granted,
                ),
                {**common, "scope": scope, "granted": granted},
            )
        if event.tool_spec.approval_required:
            approval = builder.node(
                "permission",
                f"permission:{actor_id}:verified-approval",
                {"actor_id": actor_id, "kind": "verified_approval"},
                flags={
                    "permission_granted": bool(state.get("approval_verified")),
                    "approval_required": True,
                    "approval_verified": bool(state.get("approval_verified")),
                    "verified": bool(state.get("approval_verified")),
                },
                metadata={"actor_id": actor_id, "kind": "verified_approval"},
            )
            builder.edge(
                approval,
                action_node,
                "authorizes",
                _edge_state(
                    state,
                    edge_age=edge_age,
                    permission_granted=bool(state.get("approval_verified")),
                ),
                {
                    **common,
                    "scope": "verified_approval",
                    "granted": bool(state.get("approval_verified")),
                },
            )

        clause_results = state.get("policy_clause_results", {})
        for clause, policy_node in policy_nodes:
            clause_id = str(clause.get("id", ""))
            result = clause_results.get(clause_id) if isinstance(clause_results, Mapping) else None
            builder.edge(
                policy_node,
                action_node,
                "governed_by",
                _edge_state(
                    state,
                    edge_age=edge_age,
                    policy_result=result,
                ),
                {**common, "clause_id": clause_id, "satisfied": result},
            )
        if previous_action is not None:
            builder.edge(previous_action, action_node, "follows", state_vector, common)
        previous_action = action_node

        if event.candidate:
            intervention_node = builder.node(
                "intervention",
                "intervention:" + intervention,
                {"name": intervention},
                flags={
                    "intervention": True,
                    "verified": intervention != "none",
                    "restricted": intervention == "restrict_capability",
                    "redacted": intervention == "redact_sensitive",
                },
                metadata={"name": intervention},
            )
            builder.edge(
                intervention_node,
                action_node,
                "applies",
                _edge_state(
                    state,
                    edge_age=edge_age,
                    active=True,
                    intervened=intervention != "none",
                ),
                {**common, "intervention": intervention},
            )

    return builder.tensors(candidate_node)


__all__ = [
    "DATA_ARGUMENT_ROLES",
    "DESTINATION_ARGUMENT_ROLES",
    "GraphEvent",
    "GraphTensors",
    "build_persistent_graph",
]
