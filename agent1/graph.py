"""Persistent heterogeneous provenance, permission and intervention graphs."""

from __future__ import annotations

import hashlib
import json
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import EDGE_STATE_DIM, EDGE_TYPES, NODE_TYPES
from .features import bounded_text, canonical_json, node_feature
from .registry import ToolSpec
from .cache import GRAPH_PREFIXES


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
    node_steps: Optional[torch.Tensor] = None
    node_times: Optional[torch.Tensor] = None
    node_first_steps: Optional[torch.Tensor] = None
    node_change_steps: Optional[torch.Tensor] = None
    node_persistent: Optional[torch.Tensor] = None
    node_types: Optional[torch.Tensor] = None
    node_latest: Optional[torch.Tensor] = None
    node_entity: Optional[torch.Tensor] = None


@dataclass
class _NodeRecord:
    node_type: str
    entity_id: str
    value: Any
    flags: Dict[str, bool]
    metadata: Dict[str, Any]


def _entity_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


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
    def __init__(self, versioned=False) -> None:
        self.versioned = versioned
        self.step = 0
        self.timestamp = 0.
        self.latest = {}
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
        entity_key = (node_type, str(entity_id))
        key = (*entity_key, self.step) if self.versioned else entity_key
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
            if record.flags.get("sensitive"):
                record.metadata["value_preview"] = "[sensitive-value-hidden]"
            return existing
        previous = self.latest.get(entity_key)
        if self.versioned and previous is not None and node_type == "data" and self.nodes[previous].flags.get("sensitive"):
            flags = {**(flags or {}), "sensitive": True}
        index = len(self.nodes)
        record = _NodeRecord(
            node_type=node_type,
            entity_id=bounded_text(entity_id, 240),
            value=value,
            flags={name: bool(enabled) for name, enabled in (flags or {}).items()},
            metadata={str(key): value for key, value in (metadata or {}).items()},
        )
        old = self.nodes[previous] if previous is not None else None
        record.metadata.update(available_step=self.step, observed_at=self.timestamp,
                               first_step=old.metadata["first_step"] if old else self.step,
                               change_step=old.metadata["change_step"] if old and old.flags == record.flags and old.value == value else self.step,
                               stable_node_id=_entity_digest([node_type,entity_id,self.step]))
        if record.flags.get("sensitive"):
            record.metadata["value_preview"] = "[sensitive-value-hidden]"
        self.nodes.append(record)
        self.node_lookup[key] = index
        self.latest[entity_key] = index
        if self.versioned and previous is not None:
            self.edge(previous, index, "state_update", np.zeros(EDGE_STATE_DIM, dtype=np.float32),
                      {"event_id": f"state:{self.step}", "available_step": self.step})
        if len(self.nodes) > 8192:
            raise ValueError("graph node budget exceeded")
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
        if len(self.edge_pairs) >= 16384:
            raise ValueError("graph edge budget exceeded")
        metadata = {**metadata, "available_step": self.step,
                    "edge_id": _entity_digest(["agent-astra-graph-v1", self.nodes[source].metadata["stable_node_id"],
                        self.nodes[target].metadata["stable_node_id"], relation,
                        {k:metadata.get(k) for k in ("event_id","argument","role","scope","clause_id","intervention")}])}
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
        latest_indices = set(self.latest.values())
        entity_ids = {}
        for r in self.nodes:
            entity_ids.setdefault((r.node_type,r.entity_id),len(entity_ids))
        node_metadata = []
        tainted = {r.entity_id for r in self.nodes if r.flags.get("sensitive")}
        for r in self.nodes:
            if r.entity_id in tainted: r.metadata["value_preview"] = "[sensitive-value-hidden]"
        for index, record in enumerate(self.nodes):
            node_metadata.append(
                {
                    "index": index,
                    "type": record.node_type,
                    "entity_id": record.entity_id,
                    "flags": dict(sorted(record.flags.items())),
                    **record.metadata,
                    **({"value_preview":"[sensitive-value-hidden]"} if record.flags.get("sensitive") else {}),
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
            node_entity=torch.tensor([entity_ids[(r.node_type,r.entity_id)] for r in self.nodes],dtype=torch.long),
            node_latest=torch.tensor([i in latest_indices for i in range(len(self.nodes))], dtype=torch.bool),
            node_steps=torch.tensor([r.metadata["available_step"] for r in self.nodes], dtype=torch.long),
            node_times=torch.tensor([r.metadata["observed_at"] for r in self.nodes], dtype=torch.float64),
            node_first_steps=torch.tensor([r.metadata["first_step"] for r in self.nodes], dtype=torch.long),
            node_change_steps=torch.tensor([r.metadata["change_step"] for r in self.nodes], dtype=torch.long),
            node_types=torch.tensor([NODE_TYPES.index(r.node_type) for r in self.nodes], dtype=torch.long),
            node_persistent=torch.tensor([r.node_type in {"policy","permission"} or r.flags.get("sensitive",False) or r.flags.get("untrusted",False) for r in self.nodes], dtype=torch.bool),
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
    use_cache: bool = True,
) -> GraphTensors:
    """Build one accumulated graph with entity reuse and time-local edge state."""

    if not events or not events[-1].candidate:
        raise ValueError("graph events must end with the candidate action")
    builder = _PersistentGraphBuilder(versioned=True)
    builder.timestamp = float(events[0].timestamp)
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

    previous_action = None
    count = 0
    keys = [_entity_digest(["agent-astra-graph-prefix-v1",policy_id,policy_clauses,events[0].timestamp])]
    for event in events[:-1]:
        if event.candidate: raise ValueError("history cannot contain a pending candidate")
        keys.append(_entity_digest([keys[-1],event.event_id,event.timestamp,event.observation,event.action,event.state,event.tool_spec.to_dict()]))
    if use_cache:
        for size in range(len(events)-1,0,-1):
            snapshot=GRAPH_PREFIXES.get(keys[size])
            if snapshot is not None:
                builder,policy_nodes,previous_action=snapshot
                count=size
                break
    for step_index in range(count,len(events)-1):
        previous_action,_=_append_event(builder,events[step_index],step_index,policy_nodes,previous_action,"none")
    if use_cache and len(events)>1:
        GRAPH_PREFIXES.put(keys[-1],(builder,policy_nodes,previous_action),len(builder.nodes)+len(builder.edge_pairs))
    _,candidate_node=_append_event(builder,events[-1],len(events)-1,policy_nodes,previous_action,intervention)
    return builder.tensors(candidate_node)


def _append_event(builder,event,step_index,policy_nodes,previous_action,intervention):
    candidate_node=-1
    builder.step = step_index
    builder.timestamp = float(event.timestamp)
    state = event.state
    edge_age = 0.0  # relative age is evaluated per query by the temporal attention
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
            "untrusted": bool(state.get("source_untrusted")),
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
    for condition, truth in state.get("risk_conditions", {}).items():
        condition_node = builder.node(
            "risk_condition", f"condition:{event.event_id}:{condition}",
            {"condition": condition, "value": truth},
            flags={"verified": truth is not None and state.get("risk_conditions_verified", False)},
            metadata={"condition": condition, "condition_value": truth,
                      "condition_status": "unknown" if truth is None else "true" if truth else "false"},
        )
        builder.edge(condition_node, action_node, "governed_by", state_vector, common)
    builder.edge(source, observation_data, "derived_from", state_vector, common)
    # Temporal availability is not evidence of causal influence.
    builder.edge(source, action_node, "observed_before", state_vector, common)
    builder.edge(observation_data, action_node, "uses_data", state_vector, common)

    # Link returned data identifiers to the same persistent resource nodes
    # used by tool arguments. A resource may connect distant tool results.
    for resource_id in event.observation.get("data_ids", []):
        resource = builder.node(
            "data", "data:" + _entity_digest(["data", resource_id]),
            {"role": "data", "value": resource_id},
            flags={"sensitive": bool(state.get("contains_sensitive"))},
            metadata={"label": "resource", "resource_digest": _entity_digest(resource_id)},
        )
        builder.edge(resource, observation_data, "derived_from", state_vector, common)

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

    return action_node,candidate_node


__all__ = [
    "DATA_ARGUMENT_ROLES",
    "DESTINATION_ARGUMENT_ROLES",
    "GraphEvent",
    "GraphTensors",
    "build_persistent_graph",
]
