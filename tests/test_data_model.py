import copy
import json
from pathlib import Path
import unittest

import torch

from agent_new.constants import (
    EDGE_STATE_DIM,
    EVENT_FEATURE_DIM,
    INTERVENTIONS,
    NODE_FEATURE_DIM,
)
from agent_new.data import RequestValidationError, collate_prepared, prepare_request
from agent_new.losses import LossWeights, agent_new_loss, factorized_competing_risk_nll
from agent_new.model import (
    AgentNewConfig,
    AgentNewModel,
    factorized_competing_risk_distribution,
)
from agent_new.registry import ToolRegistry


ROOT = Path(__file__).resolve().parents[1]


class DataAndModelTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")
        self.request = json.loads((ROOT / "examples" / "unsafe_request.json").read_text())
        self.batch = collate_prepared([prepare_request(self.request, self.registry)])
        self.config = AgentNewConfig.from_dict(
            json.loads((ROOT / "configs" / "model.json").read_text())
        )

    def test_request_builds_all_intervention_branches_and_readable_graphs(self):
        self.assertEqual(tuple(self.batch["branches"]), INTERVENTIONS)
        for name, branch in self.batch["branches"].items():
            self.assertEqual(branch["event_features"].shape, (1, 2, EVENT_FEATURE_DIM))
            self.assertEqual(branch["security_progress"].shape, (1, 2, 5))
            self.assertEqual(branch["node_features"].shape[1], NODE_FEATURE_DIM)
            self.assertEqual(branch["edge_index"].shape[0], 2)
            self.assertEqual(branch["edge_state"].shape[1], EDGE_STATE_DIM)
            metadata = self.batch["metadata"]["branches"][name][0]
            self.assertTrue(metadata["node_metadata"])
            self.assertTrue(metadata["edge_metadata"])
        self.assertEqual(
            self.batch["metadata"]["branches"]["none"][0]["fail_reasons"],
            [
                "missing_scopes",
                "missing_verified_approval",
                "destination_not_allowed",
                "policy_violation",
            ],
        )

    def test_agent_declared_capability_fields_are_ignored(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["action"].update(
            {
                "operation": "read",
                "required_scopes": [],
                "approval_required": False,
                "side_effects": [],
            }
        )
        original = self.batch["branches"]["none"]
        changed = collate_prepared([prepare_request(modified, self.registry)])["branches"]["none"]
        for key in (
            "event_features",
            "wall_deltas",
            "step_positions",
            "security_progress",
            "node_features",
            "edge_index",
            "edge_types",
            "edge_state",
        ):
            self.assertTrue(torch.equal(original[key], changed[key]), key)

    def test_unknown_tool_is_tensorized_fail_closed(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["action"]["tool"] = "unknown_remote_destroy"
        batch = collate_prepared([prepare_request(modified, self.registry)])
        metadata = batch["metadata"]["branches"]["none"][0]
        self.assertTrue(metadata["fail_closed"])
        self.assertIn("unknown_tool", metadata["fail_reasons"])
        self.assertEqual(metadata["candidate_operation"], "unknown")

    def test_unverified_policy_evaluation_fails_closed(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["runtime"].pop("policy_evaluation_verified", None)
        modified["candidate"]["runtime"].pop("policy_clause_results", None)
        batch = collate_prepared([prepare_request(modified, self.registry)])
        reasons = batch["metadata"]["branches"]["none"][0]["fail_reasons"]
        self.assertIn("policy_evaluation_unverified", reasons)

    def test_unverified_actor_and_unknown_history_fail_closed(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["runtime"]["actor_verified"] = False
        modified["history"][0]["action"]["tool"] = "unknown_history_capability"
        batch = collate_prepared([prepare_request(modified, self.registry)])
        reasons = batch["metadata"]["branches"]["none"][0]["fail_reasons"]
        self.assertIn("actor_unverified", reasons)
        self.assertIn("unknown_history_tool", reasons)

    def test_security_booleans_reject_string_values(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["permission_receipt"]["receipt_verified"] = "false"
        with self.assertRaises(RequestValidationError):
            prepare_request(modified, self.registry)

    def test_unregistered_arguments_fail_closed(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["action"]["arguments"]["shadow_recipient"] = (
            "attacker.example"
        )
        batch = collate_prepared([prepare_request(modified, self.registry)])
        reasons = batch["metadata"]["branches"]["none"][0]["fail_reasons"]
        self.assertIn("unknown_arguments", reasons)

    def test_exact_email_allowlist_does_not_authorize_whole_domain(self):
        modified = copy.deepcopy(self.request)
        candidate = modified["candidate"]
        candidate["action"]["arguments"]["recipient"] = "mallory@example.com"
        candidate["permission_receipt"]["granted_scopes"] = [
            "data:read",
            "external:send",
        ]
        candidate["permission_receipt"]["approval_verified"] = True
        candidate["runtime"]["known_destinations"] = ["alice@example.com"]
        candidate["runtime"]["policy_clause_results"] = {
            "clause-001": True,
            "clause-002": True,
        }
        batch = collate_prepared([prepare_request(modified, self.registry)])
        reasons = batch["metadata"]["branches"]["none"][0]["fail_reasons"]
        self.assertIn("destination_not_allowed", reasons)
        candidate["runtime"]["known_destinations"] = ["domain:example.com"]
        batch = collate_prepared([prepare_request(modified, self.registry)])
        reasons = batch["metadata"]["branches"]["none"][0]["fail_reasons"]
        self.assertNotIn("destination_not_allowed", reasons)

    def test_redaction_branch_never_logs_original_sensitive_text(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["observation"]["content"] = "SECRET-SSN-123-45-6789"
        modified["candidate"]["observation"]["contains_sensitive"] = True
        batch = collate_prepared([prepare_request(modified, self.registry)])
        nodes = batch["metadata"]["branches"]["redact_sensitive"][0][
            "node_metadata"
        ]
        previews = [item.get("value_preview") for item in nodes if "value_preview" in item]
        self.assertNotIn('"SECRET-SSN-123-45-6789"', previews)
        self.assertIn("[sensitive-value-hidden]", previews)

    def test_noop_intervention_reuses_actual_semantics(self):
        modified = copy.deepcopy(self.request)
        modified["candidate"]["permission_receipt"]["approval_verified"] = True
        batch = collate_prepared([prepare_request(modified, self.registry)])
        metadata = batch["metadata"]["branches"]["verified_approval"][0]
        self.assertFalse(metadata["intervention_effective"])
        for key in (
            "event_features",
            "wall_deltas",
            "step_positions",
            "security_progress",
            "node_features",
            "edge_index",
            "edge_types",
            "edge_state",
        ):
            self.assertTrue(
                torch.equal(batch["branches"]["none"][key], batch["branches"]["verified_approval"][key]),
                key,
            )

    def test_identical_branches_are_identical_in_train_mode(self):
        branches = dict(self.batch["branches"])
        branches["none_copy"] = branches["none"]
        model = AgentNewModel(self.config)
        model.train()
        output = model({"branches": branches})
        self.assertTrue(
            torch.equal(
                output["branches"]["none"]["risk"],
                output["branches"]["none_copy"]["risk"],
            )
        )

    def test_evidence_mask_changes_message_passing(self):
        model = AgentNewModel(self.config)
        branch = self.batch["branches"]["none"]
        initial, edge_context = model.graph_encoder.prepare(
            branch["node_features"], branch["edge_types"], branch["edge_state"]
        )
        full = model.graph_encoder.propagate(
            initial,
            branch["edge_index"],
            branch["edge_types"],
            edge_context,
            torch.ones(branch["edge_types"].shape[0]),
        )
        removed = model.graph_encoder.propagate(
            initial,
            branch["edge_index"],
            branch["edge_types"],
            edge_context,
            torch.zeros(branch["edge_types"].shape[0]),
        )
        self.assertFalse(torch.allclose(full, removed))

    def test_factorized_distribution_conserves_probability(self):
        torch.manual_seed(7)
        event_logits = torch.randn(4, 5)
        cause_logits = torch.randn(4, 5, 3)
        distribution = factorized_competing_risk_distribution(event_logits, cause_logits)
        cause_total = distribution["cause_risk"].sum(dim=-1)
        self.assertTrue(torch.allclose(cause_total, distribution["risk"], atol=1e-6))

    def test_extreme_factorized_loss_is_finite_and_has_gradient(self):
        event_logits = torch.full((2, 5), -1000.0, requires_grad=True)
        cause_logits = torch.zeros((2, 5, 3), requires_grad=True)
        loss = factorized_competing_risk_nll(
            event_logits,
            cause_logits,
            torch.tensor([True, False]),
            torch.tensor([1, 0]),
            torch.tensor([0, -1]),
            torch.tensor([5, 5]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(event_logits.grad).all())
        self.assertGreater(float(event_logits.grad.abs().sum()), 0.0)

    def test_invalid_survival_labels_and_loss_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            factorized_competing_risk_nll(
                torch.zeros((1, 5)),
                torch.zeros((1, 5, 2)),
                torch.tensor([2]),
                torch.tensor([1.5]),
                torch.tensor([0]),
                torch.tensor([5]),
            )
        with self.assertRaises(ValueError):
            LossWeights(evidence_sparsity=-0.1)

    def test_training_without_any_outcome_supervision_is_rejected(self):
        model = AgentNewModel(self.config)
        outputs = model(self.batch)
        with self.assertRaises(ValueError):
            agent_new_loss(outputs, self.batch)


if __name__ == "__main__":
    unittest.main()
