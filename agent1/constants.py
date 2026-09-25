"""Shared ontologies and fixed feature dimensions for Agent1."""

SCHEMA_VERSION = "agent-astra-input-v1"
CHECKPOINT_VERSION = "agent-astra-checkpoint-v1"

FORMATION_VERSION = "privacy-readiness-v1"
RISK_CONDITIONS = (
    "sensitive_data_available",
    "sensitive_payload_prepared",
    "recipient_selected",
    "disclosure_not_authorized",
)

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
    "risk_condition",
)

EDGE_TYPES = (
    "proposes",
    "observed_before",
    "derived_from",
    "uses_data",
    "sends_to",
    "authorizes",
    "follows",
    "governed_by",
    "applies",
    "state_update",
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
