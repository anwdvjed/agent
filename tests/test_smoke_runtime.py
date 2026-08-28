import json
import tempfile
from pathlib import Path
import unittest

import torch

from agent_new.calibration import fit_group_conformal
from agent_new.checkpoint import canonical_digest, save_checkpoint
from agent_new.model import AgentNewConfig, AgentNewModel
from agent_new.registry import ToolRegistry, ToolSpec
from agent_new.runtime import SafetyRuntime, load_ensemble
from agent_new.smoke import run_smoke
from agent_new.training import TrainingConfig, fit
from agent_new.data import collate_prepared, prepare_request
from agent_new.trust import HMACRequestVerifier, sign_request


ROOT = Path(__file__).resolve().parents[1]


class SmokeAndRuntimeTests(unittest.TestCase):
    def test_end_to_end_smoke_has_finite_gradients(self):
        result = run_smoke(ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["risk_matrix_shape"], [1, 6])
        self.assertTrue(result["finite_gradients"])
        self.assertLess(result["actual_initial_risk"], 0.20)

    def test_runtime_hard_constraints_override_random_models(self):
        registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")
        request = json.loads((ROOT / "examples" / "unsafe_request.json").read_text())
        config = AgentNewConfig.from_dict(
            json.loads((ROOT / "configs" / "model.json").read_text())
        )
        calibration = fit_group_conformal(
            [0.05, 0.10, 0.8, 0.9], [0, 0, 1, 1], ["safe", "safe", "unsafe", "unsafe"], alpha=0.5
        )
        registry_digest = canonical_digest(registry.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (1, 2):
                torch.manual_seed(seed)
                model = AgentNewModel(config)
                path = Path(directory) / ("model-%d" % seed)
                save_checkpoint(
                    path,
                    model.state_dict(),
                    model_config=config.to_dict(),
                    event_types=[
                        "privacy_leakage",
                        "computer_security",
                        "financial_loss",
                        "data_loss",
                        "unauthorized_external_action",
                    ],
                    registry_digest=registry_digest,
                    calibration_digest=calibration.digest(),
                )
                paths.append(path)
            ensemble = load_ensemble(paths, registry, torch.device("cpu"), calibration)
            runtime = SafetyRuntime(
                registry, ensemble, calibration=calibration, device=torch.device("cpu")
            )
            result = runtime.assess(request)
            self.assertEqual(result["decision"]["decision"], "BLOCK")
            self.assertTrue(all(item["fail_closed"] for item in result["hard_constraints"].values()))
            self.assertTrue(result["evidence_edges"])
            self.assertFalse(result["evidence_path_verified"])
            registry.register(
                ToolSpec(
                    name="late_unverified_tool",
                    operation="read",
                    required_scopes=("data:read",),
                    approval_required=False,
                )
            )
            with self.assertRaises(RuntimeError):
                runtime.assess(request)

    def test_duplicate_checkpoints_cannot_fake_an_ensemble(self):
        registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")
        config = AgentNewConfig(hidden_dim=32, edge_dim=16, graph_layers=1)
        calibration = fit_group_conformal(
            [0.1, 0.8], [0, 1], ["safe", "unsafe"], alpha=0.5
        )
        registry_digest = canonical_digest(registry.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            model = AgentNewModel(config)
            paths = []
            for name in ("copy-a", "copy-b"):
                path = Path(directory) / name
                save_checkpoint(
                    path,
                    model.state_dict(),
                    model_config=config.to_dict(),
                    event_types=[
                        "privacy_leakage",
                        "computer_security",
                        "financial_loss",
                        "data_loss",
                        "unauthorized_external_action",
                    ],
                    registry_digest=registry_digest,
                    calibration_digest=calibration.digest(),
                )
                paths.append(path)
            with self.assertRaises(ValueError):
                load_ensemble(paths, registry, torch.device("cpu"), calibration)

    def test_signed_calibrated_independent_ensemble_can_allow_safe_action(self):
        registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")
        config = AgentNewConfig(hidden_dim=32, edge_dim=16, graph_layers=1)
        calibration = fit_group_conformal(
            [0.05, 0.99], [0, 1], ["safe", "unsafe"], alpha=0.5
        )
        request = {
            "trusted_goal": "Read the public document.",
            "policy": {
                "id": "read-policy",
                "clauses": [{"id": "read-only", "text": "Only read the document."}],
            },
            "history": [],
            "candidate": {
                "event_id": "safe-read",
                "timestamp": 1.0,
                "observation": {
                    "source_id": "trusted-user",
                    "source_type": "user",
                    "trust_level": "trusted",
                    "content": "Read the document.",
                    "data_ids": [],
                },
                "action": {
                    "tool": "read_document",
                    "arguments": {"path": "/public/readme.txt"},
                },
                "permission_receipt": {
                    "granted_scopes": ["data:read"],
                    "approval_verified": False,
                    "receipt_verified": True,
                },
                "runtime": {
                    "actor_verified": True,
                    "sandboxed": True,
                    "sensitive_data_ids": [],
                    "known_destinations": [],
                    "policy_evaluation_verified": True,
                    "policy_clause_results": {"read-only": True},
                },
            },
        }
        signed = sign_request(
            request,
            b"runtime-secret",
            issuer="policy-engine",
            subject="agent-1",
            nonce="safe-read-once",
            lifetime_seconds=60,
        )
        registry_digest = canonical_digest(registry.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (101, 202):
                torch.manual_seed(seed)
                model = AgentNewModel(config)
                path = Path(directory) / ("independent-%d" % seed)
                save_checkpoint(
                    path,
                    model.state_dict(),
                    model_config=config.to_dict(),
                    event_types=[
                        "privacy_leakage",
                        "computer_security",
                        "financial_loss",
                        "data_loss",
                        "unauthorized_external_action",
                    ],
                    registry_digest=registry_digest,
                    calibration_digest=calibration.digest(),
                    extra_metadata={
                        "training_run_id": "run-%d" % seed,
                        "seed": seed,
                    },
                )
                paths.append(path)
            ensemble = load_ensemble(paths, registry, torch.device("cpu"), calibration)
            runtime = SafetyRuntime(
                registry,
                ensemble,
                calibration=calibration,
                device=torch.device("cpu"),
                trusted_context_verifier=HMACRequestVerifier(
                    b"runtime-secret", ["policy-engine"]
                ),
            )
            result = runtime.assess(signed)
            self.assertEqual(result["decision"]["decision"], "ALLOW")
            self.assertTrue(result["trusted_context_verified"])
            self.assertIsNotNone(result["commit"])
            self.assertNotIn("prepared_action", result["commit"])
            self.assertTrue(runtime.verify_commit_digest(signed, result["commit"]["sha256"]))

    def test_compact_training_loop_runs_and_restores_best_state(self):
        registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")
        request = json.loads((ROOT / "examples" / "unsafe_request.json").read_text())
        batch = collate_prepared([prepare_request(request, registry)])
        batch.update(
            {
                "event_observed": torch.tensor([True]),
                "event_time": torch.tensor([1]),
                "event_type": torch.tensor([0]),
                "censor_time": torch.tensor([5]),
                "sample_weight": torch.tensor([1.0]),
                "supervision_survival": torch.tensor([1.0]),
                "intervention_event_targets": torch.ones((1, 5)),
                "intervention_target_names": [
                    "verified_approval",
                    "redact_sensitive",
                    "constrain_destination",
                    "sandbox",
                    "restrict_capability",
                ],
            }
        )
        model = AgentNewModel(
            AgentNewConfig(hidden_dim=32, edge_dim=16, graph_layers=1)
        )
        history, state = fit(
            model,
            [batch],
            [batch],
            torch.device("cpu"),
            TrainingConfig(epochs=2, patience=2),
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(set(state), set(model.state_dict()))
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))


if __name__ == "__main__":
    unittest.main()
