"""Core neural architecture for intervention-conditioned agent safety.

The model predicts the first unsafe event before a candidate tool action is
executed.  It combines a causally *ordered* (not causal-inference) step/time
encoder with a deterministic, relation-aware provenance graph encoder.  Each
intervention branch is evaluated by the same parameters and without dropout,
so identical branches produce identical outputs even while the model is in
training mode.

Evidence is part of the computation rather than a post-hoc explanation.  A
differentiable edge mask produces selected and complementary subgraphs; the
selected subgraph is the primary risk path, while full-graph and complement
predictions support sufficiency and necessity objectives.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    CONCEPTS,
    CRITICAL_CONCEPT_INDICES,
    DEFAULT_EVENT_TYPES,
    EDGE_STATE_DIM,
    EDGE_TYPES,
    EVENT_FEATURE_DIM,
    INTERVENTIONS,
    NODE_FEATURE_DIM,
)


@dataclass(frozen=True)
class AgentNewConfig:
    """Configuration for :class:`AgentNewModel`.

    ``dropout`` is intentionally absent from the computation.  Counterfactual
    branches must not differ because independent stochastic masks were drawn.
    Epistemic uncertainty should be estimated with independently trained model
    ensembles rather than Monte-Carlo dropout inside a paired comparison.
    """

    event_dim: int = EVENT_FEATURE_DIM
    node_dim: int = NODE_FEATURE_DIM
    edge_state_dim: int = EDGE_STATE_DIM
    security_progress_dim: int = 5
    hidden_dim: int = 128
    edge_dim: int = 32
    graph_layers: int = 2
    horizon: int = 5
    num_event_types: int = len(DEFAULT_EVENT_TYPES)
    num_concepts: int = len(CONCEPTS)
    base_event_rate: float = 0.01
    evidence_temperature: float = 1.0
    actual_branch: str = "none"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.num_event_types < 1:
            raise ValueError("num_event_types must be positive")
        if self.graph_layers < 1:
            raise ValueError("graph_layers must be positive")
        if not 0.0 < self.base_event_rate < 1.0:
            raise ValueError("base_event_rate must lie strictly between zero and one")
        if self.evidence_temperature <= 0.0:
            raise ValueError("evidence_temperature must be positive")
        if self.num_concepts <= max(CRITICAL_CONCEPT_INDICES):
            raise ValueError("num_concepts does not include every critical concept")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentNewConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StepTimeSafetyEncoder(nn.Module):
    """Encode an ordered action prefix with step and elapsed-time information.

    The recurrent scan is causal in the sequence-processing sense: state at
    step ``t`` only reads steps ``<= t``.  It does not claim to discover causal
    effects.  Wall-clock decay, ordinal step positions, and a structured
    security-progress vector remain separate input channels.
    """

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int,
        security_progress_dim: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.event_projection = nn.Linear(event_dim, hidden_dim)
        self.timing_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.progress_projection = nn.Sequential(
            nn.Linear(security_progress_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        # softplus(-2) gives gentle decay at initialization instead of erasing
        # history after the first non-zero wall-clock interval.
        self.wall_decay_raw = nn.Parameter(torch.full((hidden_dim,), -2.0))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        event_features: torch.Tensor,
        wall_deltas: torch.Tensor,
        step_positions: torch.Tensor,
        security_progress: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one prefix representation per sample.

        Args:
            event_features: ``[B, T, F]`` structured/text event features.
            wall_deltas: ``[B, T]`` non-negative elapsed times.
            step_positions: ``[B, T]`` monotonically increasing step indices.
            security_progress: ``[B, T, P]`` structured state-change features.
            sequence_mask: ``[B, T]`` truthy for observed prefix positions.
        """

        if event_features.ndim != 3:
            raise ValueError("event_features must have shape [B, T, F]")
        batch_size, steps, _ = event_features.shape
        expected_bt = (batch_size, steps)
        if wall_deltas.shape != expected_bt or step_positions.shape != expected_bt:
            raise ValueError("wall_deltas and step_positions must have shape [B, T]")
        if sequence_mask.shape != expected_bt:
            raise ValueError("sequence_mask must have shape [B, T]")
        if security_progress.shape[:2] != expected_bt:
            raise ValueError("security_progress must have shape [B, T, P]")

        state = event_features.new_zeros((batch_size, self.hidden_dim))
        if steps == 0:
            return self.output_norm(state)

        wall = wall_deltas.to(event_features.dtype).clamp(min=0.0, max=1.0e6)
        positions = step_positions.to(event_features.dtype).clamp(min=0.0)
        previous_positions = torch.cat(
            [torch.zeros_like(positions[:, :1]), positions[:, :-1]], dim=1
        )
        step_deltas = (positions - previous_positions).clamp(min=0.0)
        timing = torch.stack(
            [torch.log1p(wall), torch.log1p(positions), torch.log1p(step_deltas)],
            dim=-1,
        )
        inputs = self.input_norm(
            self.event_projection(event_features)
            + self.timing_projection(timing)
            + self.progress_projection(security_progress.to(event_features.dtype))
        )
        inputs = F.silu(inputs)
        decay_rate = F.softplus(self.wall_decay_raw).unsqueeze(0)

        for step in range(steps):
            active = sequence_mask[:, step].to(torch.bool).unsqueeze(-1)
            decayed = state * torch.exp(-decay_rate * wall[:, step].unsqueeze(-1))
            proposal = self.cell(inputs[:, step], decayed)
            state = torch.where(active, proposal, state)
        return self.output_norm(state)


class RelationAwareMessageLayer(nn.Module):
    """One deterministic message-passing layer with relation-specific gates."""

    def __init__(self, hidden_dim: int, edge_dim: int, num_relations: int) -> None:
        super().__init__()
        self.relation_scale = nn.Embedding(num_relations, hidden_dim)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.relation_scale.weight)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_context: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(nodes)
            return self.output_norm(self.update(aggregate, nodes))

        source, target = edge_index
        relation_gate = 2.0 * torch.sigmoid(self.relation_scale(edge_types))
        messages = self.source_projection(nodes[source]) * relation_gate
        messages = F.silu(self.message_norm(messages + self.edge_projection(edge_context)))
        weights = edge_weights.to(nodes.dtype).clamp(min=0.0, max=1.0).unsqueeze(-1)
        messages = messages * weights

        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, target, messages)
        denominator = nodes.new_zeros((nodes.shape[0], 1))
        # Normalize by the structural in-degree, not by the sum of mask
        # weights.  Dividing by the weight sum would cancel uniform masks and
        # make a single incoming edge invariant to its evidence score.
        denominator.index_add_(0, target, torch.ones_like(weights))
        aggregate = aggregate / denominator.clamp(min=1.0)
        return self.output_norm(self.update(aggregate, nodes))


class DeterministicRelationGraphEncoder(nn.Module):
    """Encode provenance graphs under arbitrary differentiable edge masks.

    No stochastic layer is used.  All intervention branches share this module,
    making their differences attributable to branch inputs and edge masks.
    """

    def __init__(self, config: AgentNewConfig) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(config.node_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.edge_embedding = nn.Embedding(len(EDGE_TYPES), config.edge_dim)
        self.edge_state_projection = nn.Linear(config.edge_state_dim, config.edge_dim)
        self.edge_norm = nn.LayerNorm(config.edge_dim)
        self.layers = nn.ModuleList(
            [
                RelationAwareMessageLayer(
                    config.hidden_dim, config.edge_dim, len(EDGE_TYPES)
                )
                for _ in range(config.graph_layers)
            ]
        )

    def prepare(
        self,
        node_features: torch.Tensor,
        edge_types: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = self.node_projection(node_features)
        edge_context = self.edge_norm(
            self.edge_embedding(edge_types)
            + self.edge_state_projection(edge_state.to(node_features.dtype))
        )
        return nodes, F.silu(edge_context)

    def propagate(
        self,
        initial_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_context: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_count = edge_index.shape[1]
        if edge_weights is None:
            edge_weights = initial_nodes.new_ones((edge_count,))
        if edge_weights.shape != (edge_count,):
            raise ValueError("edge_weights must have shape [E]")
        nodes = initial_nodes
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edge_types, edge_context, edge_weights)
        return nodes

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_state: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        initial_nodes, edge_context = self.prepare(
            node_features, edge_types, edge_state
        )
        return self.propagate(
            initial_nodes, edge_index, edge_types, edge_context, edge_weights
        )


class EvidenceMasker(nn.Module):
    """Score provenance edges before masked message passing."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        initial_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_context: torch.Tensor,
        temporal_state: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return initial_nodes.new_empty((0,))
        source, target = edge_index
        values = torch.cat(
            [
                initial_nodes[source],
                initial_nodes[target],
                temporal_state[edge_batch],
                edge_context,
            ],
            dim=-1,
        )
        return self.network(values).squeeze(-1)


class TemporalGraphFusion(nn.Module):
    """Fuse prefix state, graph context, and the current candidate node."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def _attention_pool(
        self,
        temporal_state: torch.Tensor,
        nodes: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query(temporal_state)
        keys = self.key(nodes)
        pooled = []
        for batch_index in range(temporal_state.shape[0]):
            mask = node_batch == batch_index
            if not bool(mask.any()):
                pooled.append(nodes.new_zeros((nodes.shape[-1],)))
                continue
            scores = (keys[mask] * queries[batch_index]).sum(dim=-1) * self.scale
            weights = torch.softmax(scores, dim=0)
            pooled.append(torch.sum(weights.unsqueeze(-1) * nodes[mask], dim=0))
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        temporal_state: torch.Tensor,
        nodes: torch.Tensor,
        node_batch: torch.Tensor,
        candidate_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_node_indices.shape != (temporal_state.shape[0],):
            raise ValueError("candidate_node_indices must have shape [B]")
        pooled = self._attention_pool(temporal_state, nodes, node_batch)
        candidate = nodes[candidate_node_indices]
        return self.output(torch.cat([temporal_state, pooled, candidate], dim=-1))


class MonotoneFactorizedCompetingRiskHead(nn.Module):
    """Factorize event occurrence from event cause.

    For fixed fused state and fixed non-critical concepts, every critical
    concept adds a non-negative value to the any-event logit in each horizon
    bin.  Thus it cannot reduce the conditional any-event probability.  Cause
    probabilities are separately normalized and do not alter total event risk.
    """

    def __init__(self, config: AgentNewConfig) -> None:
        super().__init__()
        critical = tuple(int(index) for index in CRITICAL_CONCEPT_INDICES)
        noncritical = tuple(
            index for index in range(config.num_concepts) if index not in critical
        )
        self.register_buffer(
            "critical_indices", torch.tensor(critical, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "noncritical_indices",
            torch.tensor(noncritical, dtype=torch.long),
            persistent=False,
        )
        event_input_dim = config.hidden_dim + len(noncritical)
        cause_input_dim = config.hidden_dim + config.num_concepts
        self.event_hidden = nn.Sequential(
            nn.Linear(event_input_dim, config.hidden_dim), nn.SiLU()
        )
        self.event_output = nn.Linear(config.hidden_dim, config.horizon)
        self.cause_network = nn.Sequential(
            nn.Linear(cause_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim, config.horizon * config.num_event_types
            ),
        )
        self.monotone_weights_raw = nn.Parameter(
            torch.full((config.horizon, len(critical)), -5.0)
        )
        self.horizon = config.horizon
        self.num_event_types = config.num_event_types

        base_logit = math.log(config.base_event_rate / (1.0 - config.base_event_rate))
        nn.init.zeros_(self.event_output.weight)
        nn.init.constant_(self.event_output.bias, base_logit)

    def forward(
        self, fused_state: torch.Tensor, concept_probabilities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noncritical = concept_probabilities.index_select(1, self.noncritical_indices)
        event_logits = self.event_output(
            self.event_hidden(torch.cat([fused_state, noncritical], dim=-1))
        )
        critical = concept_probabilities.index_select(1, self.critical_indices)
        nonnegative_weights = F.softplus(self.monotone_weights_raw)
        event_logits = event_logits + torch.einsum(
            "bc,hc->bh", critical, nonnegative_weights
        )
        cause_logits = self.cause_network(
            torch.cat([fused_state, concept_probabilities], dim=-1)
        ).reshape(fused_state.shape[0], self.horizon, self.num_event_types)
        return event_logits, cause_logits


def factorized_competing_risk_distribution(
    event_logits: torch.Tensor, cause_logits: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Convert factorized logits into stable survival/incidence quantities.

    ``event_logits[:, t]`` parameterizes ``P(event at t | survived before t)``.
    ``cause_logits[:, t]`` parameterizes the cause conditional on an event at
    that bin.  The returned cumulative incidence therefore sums to horizon
    risk (up to floating-point error).
    """

    if event_logits.ndim != 2:
        raise ValueError("event_logits must have shape [B, H]")
    if cause_logits.ndim != 3 or cause_logits.shape[:2] != event_logits.shape:
        raise ValueError("cause_logits must have shape [B, H, K]")
    log_no_event = F.logsigmoid(-event_logits)
    log_event = F.logsigmoid(event_logits)
    log_survival_before = torch.cat(
        [
            torch.zeros_like(log_no_event[:, :1]),
            torch.cumsum(log_no_event[:, :-1], dim=1),
        ],
        dim=1,
    )
    cause_probabilities = torch.softmax(cause_logits, dim=-1)
    incidence_mass = torch.exp(log_survival_before + log_event).unsqueeze(-1)
    incidence_mass = incidence_mass * cause_probabilities
    log_survival = torch.cumsum(log_no_event, dim=1)
    survival = torch.exp(log_survival)
    cumulative_incidence = torch.cumsum(incidence_mass, dim=1)
    horizon_log_survival = log_survival[:, -1]
    risk = -torch.expm1(horizon_log_survival)
    return {
        "event_probabilities": torch.sigmoid(event_logits),
        "cause_probabilities": cause_probabilities,
        "conditional_cause_hazards": torch.sigmoid(event_logits).unsqueeze(-1)
        * cause_probabilities,
        "incidence_mass": incidence_mass,
        "cumulative_incidence": cumulative_incidence,
        "survival": survival,
        "log_survival": log_survival,
        "risk": risk,
        "cause_risk": cumulative_incidence[:, -1],
        "cumulative_hazard": -horizon_log_survival,
    }


class AgentNewModel(nn.Module):
    """Intervention-conditioned provenance-aware competing-risk model."""

    def __init__(self, config: Optional[AgentNewConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentNewConfig()
        config = self.config
        self.temporal_encoder = StepTimeSafetyEncoder(
            config.event_dim, config.hidden_dim, config.security_progress_dim
        )
        self.graph_encoder = DeterministicRelationGraphEncoder(config)
        self.evidence_masker = EvidenceMasker(config.hidden_dim, config.edge_dim)
        self.fusion = TemporalGraphFusion(config.hidden_dim)
        self.concept_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_concepts),
        )
        self.attack_intent_head = nn.Linear(config.hidden_dim, 1)
        self.utility_head = nn.Linear(config.hidden_dim, 1)
        self.risk_head = MonotoneFactorizedCompetingRiskHead(config)

    @staticmethod
    def _ordered_branch_names(branches: Mapping[str, Any]) -> Tuple[str, ...]:
        known = [name for name in INTERVENTIONS if name in branches]
        extras = sorted(name for name in branches if name not in known)
        return tuple(known + extras)

    def _prediction_from_nodes(
        self,
        temporal_state: torch.Tensor,
        nodes: torch.Tensor,
        node_batch: torch.Tensor,
        candidate_node_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused_state = self.fusion(
            temporal_state, nodes, node_batch, candidate_node_indices
        )
        concept_logits = self.concept_head(fused_state)
        concept_probabilities = torch.sigmoid(concept_logits)
        event_logits, cause_logits = self.risk_head(
            fused_state, concept_probabilities
        )
        prediction = factorized_competing_risk_distribution(
            event_logits, cause_logits
        )
        prediction.update(
            {
                "fused_state": fused_state,
                "concept_logits": concept_logits,
                "concept_probabilities": concept_probabilities,
                "event_logits": event_logits,
                "cause_logits": cause_logits,
                "attack_intent_logit": self.attack_intent_head(fused_state).squeeze(-1),
                "utility_logit": self.utility_head(fused_state).squeeze(-1),
            }
        )
        return prediction

    def _forward_branch(
        self, branch: Mapping[str, torch.Tensor]
    ) -> Dict[str, Any]:
        temporal_state = self.temporal_encoder(
            branch["event_features"],
            branch["wall_deltas"],
            branch["step_positions"],
            branch["security_progress"],
            branch["sequence_mask"],
        )
        edge_index = branch["edge_index"].long()
        edge_types = branch["edge_types"].long()
        edge_batch = branch["edge_batch"].long()
        initial_nodes, edge_context = self.graph_encoder.prepare(
            branch["node_features"], edge_types, branch["edge_state"]
        )
        evidence_logits = self.evidence_masker(
            initial_nodes,
            edge_index,
            edge_context,
            temporal_state,
            edge_batch,
        )
        evidence_weights = torch.sigmoid(
            evidence_logits / self.config.evidence_temperature
        )
        full_nodes = self.graph_encoder.propagate(
            initial_nodes, edge_index, edge_types, edge_context
        )
        selected_nodes = self.graph_encoder.propagate(
            initial_nodes,
            edge_index,
            edge_types,
            edge_context,
            evidence_weights,
        )
        complement_nodes = self.graph_encoder.propagate(
            initial_nodes,
            edge_index,
            edge_types,
            edge_context,
            1.0 - evidence_weights,
        )
        node_batch = branch["node_batch"].long()
        candidate_indices = branch["candidate_node_indices"].long()
        full = self._prediction_from_nodes(
            temporal_state, full_nodes, node_batch, candidate_indices
        )
        selected = self._prediction_from_nodes(
            temporal_state, selected_nodes, node_batch, candidate_indices
        )
        complement = self._prediction_from_nodes(
            temporal_state, complement_nodes, node_batch, candidate_indices
        )
        # Frequently consumed fields are exposed directly; complete
        # distributions remain available in the three named prediction paths.
        return {
            "temporal_state": temporal_state,
            "evidence_logits": evidence_logits,
            "evidence_weights": evidence_weights,
            "edge_batch": edge_batch,
            "selected": selected,
            "full": full,
            "complement": complement,
            "risk": selected["risk"],
            "full_risk": full["risk"],
            "complement_risk": complement["risk"],
            "event_logits": selected["event_logits"],
            "cause_logits": selected["cause_logits"],
            "concept_logits": selected["concept_logits"],
            "concept_probabilities": selected["concept_probabilities"],
            "cumulative_hazard": selected["cumulative_hazard"],
            "attack_intent_logit": selected["attack_intent_logit"],
            "utility_logit": selected["utility_logit"],
        }

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate the actual state and every supplied intervention branch.

        ``batch['branches']`` maps an intervention name to a graph/sequence
        batch.  Output deltas are ``actual risk - intervention risk``; positive
        values therefore mean that an intervention is predicted to reduce risk.
        """

        branches = batch.get("branches")
        if not isinstance(branches, Mapping) or not branches:
            raise ValueError("batch['branches'] must be a non-empty mapping")
        branch_names = self._ordered_branch_names(branches)
        branch_outputs = {
            name: self._forward_branch(branches[name]) for name in branch_names
        }
        if self.config.actual_branch in branch_outputs:
            actual_name = self.config.actual_branch
        elif "actual" in branch_outputs:
            actual_name = "actual"
        else:
            actual_name = branch_names[0]
        actual = branch_outputs[actual_name]
        risk_matrix = torch.stack(
            [branch_outputs[name]["risk"] for name in branch_names], dim=1
        )
        risk_deltas = actual["risk"].unsqueeze(1) - risk_matrix
        intervention_names = tuple(
            name for name in branch_names if name != actual_name
        )
        if intervention_names:
            intervention_risk_matrix = torch.stack(
                [branch_outputs[name]["risk"] for name in intervention_names], dim=1
            )
            intervention_risk_deltas = (
                actual["risk"].unsqueeze(1) - intervention_risk_matrix
            )
        else:
            intervention_risk_matrix = actual["risk"].new_empty(
                (actual["risk"].shape[0], 0)
            )
            intervention_risk_deltas = intervention_risk_matrix.clone()
        return {
            "branches": branch_outputs,
            "branch_names": branch_names,
            "actual_name": actual_name,
            "actual": actual,
            "actual_risk": actual["risk"],
            "risk_matrix": risk_matrix,
            "risk_deltas": risk_deltas,
            "intervention_names": intervention_names,
            "intervention_risk_matrix": intervention_risk_matrix,
            "intervention_risk_deltas": intervention_risk_deltas,
        }


# Concise aliases for callers and checkpoints that prefer the project name.
AgentNew = AgentNewModel
P2THNv2 = AgentNewModel
StepTimeCausalEncoder = StepTimeSafetyEncoder


__all__ = [
    "AgentNew",
    "AgentNewConfig",
    "AgentNewModel",
    "DeterministicRelationGraphEncoder",
    "EvidenceMasker",
    "MonotoneFactorizedCompetingRiskHead",
    "P2THNv2",
    "RelationAwareMessageLayer",
    "StepTimeCausalEncoder",
    "StepTimeSafetyEncoder",
    "TemporalGraphFusion",
    "factorized_competing_risk_distribution",
]
