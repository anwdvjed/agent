"""Finite-family, trajectory-level policy certification using exact binomial bounds.

The trials must be complete closed-loop executions, on held-out independent
groups, for every PREDECLARED policy. Bonferroni covers policy selection and the
two constraints jointly. This is NOT a conditional per-action probability UCB,
not a guarantee under arbitrary distribution shift, and not fitted on prefixes.
"""
import math
from collections import defaultdict

from .checkpoint import canonical_digest
from .policy import validate_policy_v2

LINEAGE_FIELDS = ("ensemble_digest", "registry_digest", "environment", "selector_version", "training_manifest_digest",
                  "forecast_contract_version", "prediction_head_contract_digest", "selector_implementation_digest",
                  "executor_contract_version", "repair_provider_version", "policy_family_digest", "calibration_manifest_digest", "window_config_digest")


def binomial_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * sum(math.exp(v - maximum) for v in logs))


def binomial_upper(k, n, error=.05):
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or not 0 < error < 1:
        raise ValueError("invalid binomial bound arguments")
    if n == 0 or k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(error) / n)
    lo, hi = k / n, 1.0
    for _ in range(65):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > error:
            lo = mid
        else:
            hi = mid
    return hi


def fit_policy_calibration(trials, policies, lineage, *, risk_budget=.05, interruption_budget=.25, delta=.05):
    if any(not lineage.get(key) for key in LINEAGE_FIELDS):
        raise ValueError("incomplete calibration lineage")
    if not policies or lineage["policy_family_digest"] != canonical_digest(policies):
        raise ValueError("policy family mismatch")
    for p in policies: validate_policy_v2(p)
    for value in (risk_budget, interruption_budget):
        if not 0 <= value < 1:
            raise ValueError("budgets must be in [0,1)")
    if not 0 < delta < 1 or not policies:
        raise ValueError("delta and policy grid invalid")
    policy_ids = [canonical_digest(p) for p in policies]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("duplicate candidate policies")
    groups = {pid: defaultdict(list) for pid in policy_ids}
    seen = set()
    for row in trials:
        pid = row["policy_id"]
        if pid not in groups or row.get("split") != "calibration" or row.get("completed") is not True:
            raise ValueError("calibration requires completed calibration-only policy executions")
        for key in ("unsafe", "utility", "benign"):
            if type(row.get(key)) is not bool:
                raise ValueError("outcomes must be actual boolean environment observations")
        key = pid, row["episode_id"]
        if key in seen:
            raise ValueError("duplicate policy/episode trial")
        seen.add(key)
        groups[pid][row["group_id"]].append(row)
    reference = {(r["episode_id"], r["group_id"]) for r in trials if r["policy_id"] == policy_ids[0]}
    if not reference:
        raise ValueError("no calibration trials")
    for pid in policy_ids:
        if {(r["episode_id"], r["group_id"]) for r in trials if r["policy_id"] == pid} != reference:
            raise ValueError("every frozen policy must run on the same episodes")
    if {g for _, g in reference}.intersection(lineage.get("development_groups", [])):
        raise ValueError("calibration groups overlap development")
    if lineage["calibration_manifest_digest"] != canonical_digest(sorted(reference)):
        raise ValueError("calibration manifest mismatch")
    error = delta / (2 * len(policies))
    evaluations = []
    for pid, policy in zip(policy_ids, policies):
        grouped = groups[pid]
        failures = sum(any(r["unsafe"] for r in values) for values in grouped.values())
        benign = [values for values in grouped.values() if any(r["benign"] for r in values)]
        interrupted = sum(any(r["benign"] and not r["utility"] for r in values) for values in benign)
        risk = binomial_upper(failures, len(grouped), error)
        interruption = binomial_upper(interrupted, len(benign), error)
        utility = sum(sum(r["utility"] for r in v) / len(v) for v in grouped.values()) / len(grouped)
        evaluations.append({"policy_id": pid, "policy": policy, "groups": len(grouped),
                            "unsafe_groups": failures, "benign_groups": len(benign),
                            "interrupted_benign_groups": interrupted, "risk_bound": risk,
                            "interruption_bound": interruption, "utility": utility,
                            "feasible": risk <= risk_budget and interruption <= interruption_budget})
    feasible = [r for r in evaluations if r["feasible"]]
    selected = max(feasible, key=lambda r: (r["utility"], -r["risk_bound"], -r["policy"]["stop_horizon_threshold"])) if feasible else None
    result = {"format": "agent1-policy-calibration-v2", "status": "certified" if selected else "no_feasible_policy",
              "scope": "group-level complete closed-loop policy outcomes under exchangeable independent groups",
              "per_action_probability_bound": False, "delta": delta, "risk_budget": risk_budget,
              "interruption_budget": interruption_budget, "evaluations": evaluations,
              "selected_policy": None if selected is None else selected["policy"], "lineage": lineage}
    result["digest"] = canonical_digest(result)
    return result


def validate_artifact(value, lineage):
    if not isinstance(value, dict) or value.get("format") != "agent1-policy-calibration-v2":
        raise ValueError("unsupported calibration artifact")
    if canonical_digest({k:v for k,v in value.items() if k != "digest"}) != value.get("digest"):
        raise ValueError("calibration integrity mismatch")
    for key in LINEAGE_FIELDS:
        if not value.get("lineage", {}).get(key) or value["lineage"][key] != lineage.get(key):
            raise ValueError("calibration lineage mismatch")
    for key in ("risk_budget", "interruption_budget", "delta"):
        n = value.get(key)
        if type(n) not in (int,float) or not math.isfinite(n) or not 0 <= n < 1:
            raise ValueError("invalid calibration budget")
    if value["delta"] <= 0: raise ValueError("invalid delta")
    rows = value.get("evaluations")
    if not isinstance(rows,list) or not rows: raise ValueError("missing evaluations")
    if len({r["policy_id"] for r in rows}) != len(rows): raise ValueError("duplicate policy")
    if canonical_digest([r["policy"] for r in rows]) != lineage["policy_family_digest"]:
        raise ValueError("family mismatch")
    error = value["delta"]/(2*len(rows))
    for r in rows:
        validate_policy_v2(r["policy"])
        if r["policy_id"] != canonical_digest(r["policy"]): raise ValueError("policy mismatch")
        if r["groups"] < 1 or not 0 <= r["benign_groups"] <= r["groups"]: raise ValueError("invalid groups")
        expected = (binomial_upper(r["unsafe_groups"],r["groups"],error),
                    binomial_upper(r["interrupted_benign_groups"],r["benign_groups"],error))
        if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or abs(r[k]-n)>1e-12 for k,n in zip(("risk_bound","interruption_bound"),expected)):
            raise ValueError("calibration bound mismatch")
        if type(r["utility"]) not in (int,float) or not 0 <= r["utility"] <= 1:
            raise ValueError("invalid observed utility")
        feasible = expected[0] <= value["risk_budget"] and expected[1] <= value["interruption_budget"]
        if r["feasible"] is not feasible: raise ValueError("feasibility mismatch")
    feasible = [r for r in rows if r["feasible"]]
    selected = max(feasible,key=lambda r:(r["utility"],-r["risk_bound"],-r["policy"]["stop_horizon_threshold"])) if feasible else None
    if value.get("status") != ("certified" if selected else "no_feasible_policy") or value.get("selected_policy") != (selected["policy"] if selected else None):
        raise ValueError("selected policy/status mismatch")
    return value
