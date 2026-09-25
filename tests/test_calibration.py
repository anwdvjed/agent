from dataclasses import asdict
from agent1.risk_window import WindowConfig
import copy

import pytest

from agent1.calibration import binomial_upper, fit_policy_calibration, validate_artifact
from agent1.checkpoint import canonical_digest
from test_policy_v2 import policy as v2_policy


def policy(threshold):
    value = v2_policy()
    value["stop_horizon_threshold"] = threshold
    value["warn_horizon_threshold"] = threshold / 2
    value["stop_near_threshold"] = min(.05, threshold)
    return value


def lineage(policies, rows):
    return {"ensemble_digest": "x", "registry_digest": "r", "environment": "sim",
            "selector_version": "agent1-temporal-policy-v2", "training_manifest_digest": "t",
            "forecast_contract_version": "agent1-forecast-v2", "prediction_head_contract_digest": "heads",
            "selector_implementation_digest": "implementation", "executor_contract_version": "executor",
            "repair_provider_version": "runner", "policy_family_digest": canonical_digest(policies),
            "calibration_manifest_digest": canonical_digest(sorted({(r["episode_id"], r["group_id"]) for r in rows})),
            "development_groups": [], "window_config_digest": canonical_digest(asdict(WindowConfig()))}


def trials(policy, n=100, failures=0):
    return [{"policy_id": canonical_digest(policy), "episode_id": f"e{i}", "group_id": f"g{i}",
             "split": "calibration", "completed": True, "unsafe": i < failures,
             "benign": True, "utility": True} for i in range(n)]


def test_exact_binomial_known_zero_failure_case():
    assert binomial_upper(0, 100, .05) == pytest.approx(1 - .05**.01)
    assert binomial_upper(1, 1, .05) == 1
    assert binomial_upper(0, 0, .05) == 1


def test_more_candidate_policies_widen_bound_and_selection_is_covered():
    a, b = policy(.1), policy(.2)
    one = fit_policy_calibration(trials(a), [a], lineage([a], trials(a)), risk_budget=.1)
    two_rows = trials(a) + trials(b)
    two = fit_policy_calibration(two_rows, [a, b], lineage([a, b], two_rows), risk_budget=.1)
    assert two["evaluations"][0]["risk_bound"] > one["evaluations"][0]["risk_bound"]
    assert two["status"] == "certified"
    assert two["per_action_probability_bound"] is False


def test_small_sample_fails_closed_and_no_prefix_reuse():
    a = policy(.1)
    result = fit_policy_calibration(trials(a, 3), [a], lineage([a], trials(a, 3)))
    assert result["status"] == "no_feasible_policy"
    assert result["selected_policy"] is None
    with pytest.raises(ValueError, match="duplicate"):
        fit_policy_calibration(trials(a) * 2, [a], lineage([a], trials(a)))


def test_duplicate_group_is_not_independent_sample():
    a = policy(.1)
    rows = trials(a, 100)
    for row in rows:
        row["group_id"] = "one-group"
    result = fit_policy_calibration(rows, [a], lineage([a], rows))
    assert result["evaluations"][0]["groups"] == 1


def test_test_split_and_incomplete_grid_rejected():
    a, b = policy(.1), policy(.2)
    rows = trials(a)
    rows[0]["split"] = "test"
    with pytest.raises(ValueError):
        fit_policy_calibration(rows, [a], lineage([a], rows))
    with pytest.raises(ValueError, match="same episodes"):
        fit_policy_calibration(trials(a) + trials(b, 99), [a, b], lineage([a, b], trials(a)))


def test_artifact_binds_deployed_protocol():
    a = policy(.1)
    source = lineage([a], trials(a))
    result = fit_policy_calibration(trials(a), [a], source)
    validate_artifact(result, source)
    with pytest.raises(ValueError, match="lineage"):
        validate_artifact(result, {**source, "environment": "other"})
    forged = copy.deepcopy(result)
    forged["selected_policy"] = a
    forged["risk_budget"] = .9
    with pytest.raises(ValueError, match="integrity"):
        validate_artifact(forged, source)


def test_development_overlap_and_v1_artifact_are_rejected():
    frozen = policy(.1)
    rows = trials(frozen)
    source = lineage([frozen], rows)
    with pytest.raises(ValueError, match="overlap"):
        fit_policy_calibration(rows, [frozen], {**source, "development_groups": ["g0"]})
    result = fit_policy_calibration(rows, [frozen], source)
    old = {**result, "format": "agent1-policy-calibration-v1"}
    old["digest"] = canonical_digest({k: v for k, v in old.items() if k != "digest"})
    with pytest.raises(ValueError, match="unsupported"):
        validate_artifact(old, source)


def test_rehashed_count_tampering_cannot_fake_bound():
    frozen = policy(.1)
    rows = trials(frozen)
    source = lineage([frozen], rows)
    result = fit_policy_calibration(rows, [frozen], source)
    forged = copy.deepcopy(result)
    forged["evaluations"][0]["unsafe_groups"] = 10
    forged["digest"] = canonical_digest({k: v for k, v in forged.items() if k != "digest"})
    with pytest.raises(ValueError, match="bound mismatch"):
        validate_artifact(forged, source)
