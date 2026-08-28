"""Shared ontologies and fixed feature dimensions for Agent New."""

SCHEMA_VERSION = "agent-new-input-v1"
CHECKPOINT_VERSION = "agent-new-checkpoint-v1"

OPERATIONS = (
    "unknown",
    "context",
    "read",
    "write_internal",
    "write_external",
    "execute",
    "delete",
    "permission_change",
    "finish",
    "other",
)

NODE_TYPES = (
    "actor",
    "source",
    "data",
    "action",
    "destination",
    "permission",
    "policy",
    "intervention",
)

EDGE_TYPES = (
    "proposes",
    "influences",
    "derived_from",
    "uses_data",
    "sends_to",
    "authorizes",
    "follows",
    "governed_by",
    "applies",
)

CONCEPTS = (
    "goal_deviation",
    "untrusted_provenance",
    "privilege_excess",
    "sensitive_data_access",
    "sensitive_data_flow",
    "anomalous_destination",
    "missing_approval",
    "irreversibility",
    "unknown_capability",
)

CRITICAL_CONCEPT_INDICES = (2, 4, 5, 6, 7, 8)
NONCRITICAL_CONCEPT_INDICES = tuple(
    index for index in range(len(CONCEPTS)) if index not in CRITICAL_CONCEPT_INDICES
)

DEFAULT_EVENT_TYPES = (
    "privacy_leakage",
    "computer_security",
    "financial_loss",
    "data_loss",
    "unauthorized_external_action",
)

INTERVENTIONS = (
    "none",
    "verified_approval",
    "redact_sensitive",
    "constrain_destination",
    "sandbox",
    "restrict_capability",
)

INTERVENTION_COSTS = {
    "none": 0.0,
    "verified_approval": 1.0,
    "redact_sensitive": 1.5,
    "constrain_destination": 1.5,
    "sandbox": 2.0,
    "restrict_capability": 4.0,
}

DECISIONS = ("ALLOW", "HOLD", "REPLAN", "CONFIRM", "BLOCK")

EVENT_HASH_DIM = 64
EVENT_FLAG_DIM = 24
EVENT_FEATURE_DIM = EVENT_HASH_DIM + len(OPERATIONS) + EVENT_FLAG_DIM

NODE_HASH_DIM = 48
NODE_FLAG_DIM = 16
NODE_FEATURE_DIM = NODE_HASH_DIM + len(NODE_TYPES) + NODE_FLAG_DIM

EDGE_STATE_DIM = 10
