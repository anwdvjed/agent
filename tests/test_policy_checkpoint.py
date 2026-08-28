import tempfile
from pathlib import Path
import unittest

import torch

from agent_new.calibration import (
    fit_group_conformal,
    select_threshold_at_safe_trajectory_fpr,
)
from agent_new.checkpoint import canonical_digest, load_checkpoint, save_checkpoint
from agent_new.policy import InterventionPolicy


class PolicyCalibrationCheckpointTests(unittest.TestCase):
    def test_calibration_and_fpr_use_trajectory_groups(self):
        predictions = [0.05, 0.10, 0.70, 0.90, 0.15, 0.20]
        outcomes = [0, 0, 1, 1, 0, 0]
        groups = ["safe-a", "safe-a", "unsafe", "unsafe", "safe-b", "safe-b"]
        calibration = fit_group_conformal(predictions, outcomes, groups, alpha=0.5)
        self.assertEqual(calibration.group_count, 3)
        self.assertGreaterEqual(calibration.risk_upper_bound(0.1), 0.1)
        selected = select_threshold_at_safe_trajectory_fpr(
            predictions, outcomes, groups, target_fpr=0.0
        )
        self.assertEqual(selected.safe_trajectory_count, 2)
        self.assertEqual(selected.unsafe_trajectory_count, 1)
        self.assertEqual(selected.safe_trajectory_fpr, 0.0)
        with self.assertRaises(ValueError):
            fit_group_conformal([0.1, 0.2], [0, 0], ["safe-a", "safe-b"], alpha=0.5)

    def test_policy_selects_allow_confirm_replan_and_block(self):
        policy = InterventionPolicy(safe_risk_limit=0.2, epistemic_uncertainty_limit=0.25)
        allow = policy.select_from_values(
            {"none": 0.10}, {"none": 0.08}, {"none": 0.05}
        )
        self.assertEqual(allow.decision, "ALLOW")
        confirm = policy.select_from_values(
            {"none": 0.9, "verified_approval": 0.1},
            {"none": 0.8, "verified_approval": 0.1},
            {"none": 0.1, "verified_approval": 0.1},
        )
        self.assertEqual(confirm.decision, "CONFIRM")
        replan = policy.select_from_values(
            {"none": 0.9, "constrain_destination": 0.1},
            {"none": 0.8, "constrain_destination": 0.1},
            {"none": 0.1, "constrain_destination": 0.1},
        )
        self.assertEqual(replan.decision, "REPLAN")
        restricted = policy.select_from_values(
            {"none": 0.9, "restrict_capability": 0.0},
            {"none": 0.8, "restrict_capability": 0.0},
            {"none": 0.1, "restrict_capability": 0.0},
        )
        self.assertEqual(restricted.decision, "BLOCK")
        self.assertEqual(restricted.intervention, "restrict_capability")
        block = policy.select_from_values(
            {"none": 0.9, "sandbox": 0.8},
            {"none": 0.8, "sandbox": 0.7},
            {"none": 0.1, "sandbox": 0.1},
        )
        self.assertEqual(block.decision, "BLOCK")

    def test_checkpoint_integrity_round_trip(self):
        model = torch.nn.Linear(3, 2)
        registry_digest = canonical_digest({"registry": "test"})
        calibration_digest = canonical_digest({"calibration": "test"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            metadata = save_checkpoint(
                path,
                model.state_dict(),
                model_config={"input": 3, "output": 2},
                event_types=["unsafe"],
                registry_digest=registry_digest,
                calibration_digest=calibration_digest,
            )
            state, loaded = load_checkpoint(
                path,
                expected_model_config={"input": 3, "output": 2},
                expected_event_types=["unsafe"],
                expected_registry_digest=registry_digest,
                expected_calibration_digest=calibration_digest,
            )
            self.assertEqual(metadata.digest(), loaded.digest())
            self.assertEqual(set(state), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
