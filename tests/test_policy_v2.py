import math

import pytest

from agent1.policy import build_decision_curve, select_action_v2, validate_policy_v2


def policy():
    return {"format": "agent1-temporal-policy-v2", "horizon_bins": 5,
            "near_bins": 1, "score_method": "cdf_member_mean_std_running_max_v1",
            "disagreement_penalty": 0, "stop_horizon_threshold": .8,
            "stop_near_threshold": .1, "warn_horizon_threshold": .5,
            "min_utility": .1, "cost_weight": 0, "utility_mode": "learned",
            "readiness_warning": {"mode": "off", "lookahead_bins": 0, "threshold": None}}


def forecast(q):
    survival, mass, cdf = 1., [], []
    for probability in q:
        amount = survival * probability
        mass.append(amount)
        survival -= amount
        cdf.append(1 - survival)
    return {"schema_version": "agent1-forecast-v2", "branch_id": "none",
            "candidate_event_id": "candidate", "horizon_bins": 5,
            "time_unit": "action_bin", "bin_one_semantics": "current_candidate",
            "forecast_scope": "first_unsafe_event_under_learned_continuation_distribution",
            "risk_mean": cdf[-1], "ensemble_disagreement": 0.,
            "event_mass_mean": mass, "survival_mean": [1-x for x in cdf],
            "event_cdf_mean": cdf, "event_cdf_std": [0.]*5,
            "cause_names": ["privacy_leakage"],
            "cause_cumulative_incidence_mean": None,
            "expected_event_bin_conditional": sum((i+1)*v for i,v in enumerate(mass))/cdf[-1],
            "utility_mean": .9, "head_availability": {"event_occurrence": True, "utility": True},
            "hard_failure": False, "hard_reasons": [], "prepared_digest": "a"*64}


def test_same_window_risk_different_near_term_decision():
    late = forecast([.02, .05, .1, .25, .4])
    early = forecast([.4, .25, .1, .05, .02])
    assert late["risk_mean"] == pytest.approx(early["risk_mean"])
    allowed = select_action_v2({"none": late}, policy())
    blocked = select_action_v2({"none": early}, policy())
    assert allowed["decision"] == "ALLOW"
    assert allowed["alert"]["severity"] == "WARN"
    assert blocked["decision"] == "BLOCK"
    assert "NEAR_RISK_EXCEEDED" in blocked["branch_checks"]["none"]["reason_codes"]


@pytest.mark.parametrize("field,value", [("risk_mean", -1), ("risk_mean", math.nan),
                                          ("risk_mean", math.inf), ("ensemble_disagreement", -.1),
                                          ("utility_mean", math.nan)])
def test_bad_numbers_fail_before_clipping(field, value):
    value_forecast = forecast([.02, .05, .1, .25, .4])
    value_forecast[field] = value
    with pytest.raises(ValueError):
        select_action_v2({"none": value_forecast}, policy())


def test_running_max_is_decision_score_not_probability():
    curve = build_decision_curve([.4, .8], [.4, 0], 2)
    assert curve["raw_score"] == pytest.approx([1., .8])
    assert curve["decision_score"] == pytest.approx([1., 1.])


def test_policy_rejects_unknown_and_invalid_near_window():
    with pytest.raises(ValueError):
        validate_policy_v2({**policy(), "threshold": .4})
    with pytest.raises(ValueError):
        validate_policy_v2({**policy(), "near_bins": 6})


def test_real_repair_can_win_while_baseline_rejected():
    baseline = forecast([.4, .25, .1, .05, .02])
    repair = forecast([.01] * 5)
    repair["branch_id"] = "redact_sensitive"
    repair["prepared_digest"] = "b" * 64
    result = select_action_v2({"none": baseline, "redact_sensitive": repair}, policy())
    assert result["decision"] == "ALLOW"
    assert result["selected_branch"] == "redact_sensitive"
    assert "NEAR_RISK_EXCEEDED" in result["branch_checks"]["none"]["reason_codes"]
