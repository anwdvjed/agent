"""Closed-loop policy replay: every committed repair changes the next state."""
import time

from .checkpoint import canonical_digest
from .environments import environment_factory
from .io import read_jsonl
from .runtime import candidate_requests
from .policy import select_action_v2
from .risk_window import RiskWindow, WindowConfig
from dataclasses import asdict


def run_episode(scenario, ensemble, policy, adapter="builtin", max_steps=256, window_config=None):
    started = time.perf_counter()
    env = environment_factory(adapter)(scenario, ensemble.registry)
    decisions, repairs, steps = [], 0, 0
    window = RiskWindow(window_config)
    while not env.done:
        if steps >= max_steps:
            env.stop("step_budget_exhausted")
            break
        request = env.request()
        proposals = candidate_requests(request, env.repairs(request))
        predictions = ensemble.predict(proposals)
        selection = select_action_v2(predictions, policy) if policy is not None else None
        selected = selection["selected_branch"] if selection else None
        window.assess(request["candidate"]["event_id"], selection["branch_checks"]["none"]["near_score"] if selection else predictions["none"]["event_cdf_mean"][0],
                      decision=selection["decision"] if selection else "HOLD",
                      reason_codes=selection["branch_checks"]["none"]["reason_codes"] if selection else ())
        decisions.append({"step": steps, "selected": selected,
                          "risk_mean": predictions["none"]["risk_mean"],
                          "ensemble_disagreement": predictions["none"]["ensemble_disagreement"]})
        if selected is None:
            env.stop("policy_block")
        else:
            env.commit(proposals[selected])
            window.commit(request["candidate"]["event_id"], verified=True)
            repairs += int(selected != "none")
        steps += 1
    outcome = env.outcome()
    if outcome.get("completed") is not True:
        raise ValueError("environment did not supply a terminal outcome")
    return {**outcome, "policy_id": canonical_digest(policy), "episode_id": scenario["episode_id"],
            "group_id": scenario["group_id"], "split": scenario["split"], "environment": env.environment_id,
            "latency_scope": "warm model; includes environment initialization and complete rollout, excludes checkpoint loading",
            "repairs": repairs, "decisions": decisions, "risk_window": window.snapshot(), "window_config_digest": canonical_digest(asdict(window.config)), "latency_ms": (time.perf_counter() - started) * 1000}


def replay_split(scenarios_path, ensemble, policies, split, adapter="builtin", window_config=None):
    if split not in {"calibration", "test"}:
        raise ValueError("policy replay requires calibration or test split")
    all_scenarios = list(read_jsonl(scenarios_path))
    owners, ids = {}, set()
    for s in all_scenarios:
        if s["episode_id"] in ids or s["group_id"] in owners and owners[s["group_id"]] != s["split"]:
            raise ValueError("duplicate episode or group leakage in replay scenarios")
        owners[s["group_id"]] = s["split"]
        ids.add(s["episode_id"])
    scenarios = [s for s in all_scenarios if s["split"] == split]
    if not scenarios:
        raise ValueError(f"no {split} scenarios; no fallback allowed")
    if any(s["group_id"] in ensemble.development_groups for s in scenarios):
        raise ValueError("replay groups overlap training/validation groups")
    from tqdm import tqdm
    results = []
    for policy in policies:
        for scenario in tqdm(scenarios, desc=f"{split.capitalize()} closed-loop", unit="episode"):
            results.append(run_episode(scenario, ensemble, policy, adapter, window_config=window_config))
    return results
