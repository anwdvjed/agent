"""Budgeted graph-guided source GROUP interventions and matched controls.

These are explanation experiments, never replacement execution histories.
Every variant is re-tensorized, rebuilding both temporal and graph features.
Operational sensitivity is explicitly separate from action/outcome causality.
"""
import copy
import hashlib
import itertools
import random
from collections import defaultdict, deque

from .data import prepare_request


def rank_sources(request, registry):
    graph = prepare_request(request, registry, interventions=("none",)).branches["none"].graph
    reverse = defaultdict(list)
    for source, target in graph.edge_index.t().tolist():
        reverse[target].append(source)
    distance, queue = {graph.candidate_node_index: 0}, deque([graph.candidate_node_index])
    while queue:
        target = queue.popleft()
        for source in reverse[target]:
            if source not in distance:
                distance[source] = distance[target] + 1
                queue.append(source)
    scores = {}
    for node in graph.node_metadata:
        if node["type"] == "source" and node["flags"].get("untrusted"):
            name=node["source_id"]
            scores[name]=max(scores.get(name,0.),1/(1+distance.get(node["index"],10000)))
    return sorted(scores.items(),key=lambda item:(-item[1],item[0]))


def mask_sources(request, source_ids):
    result = copy.deepcopy(request)
    result.pop("attestation", None)
    selected, affected = set(source_ids), 0
    for event in result["history"] + [result["candidate"]]:
        obs = event.get("observation", {})
        if obs.get("source_id") in selected:
            if obs.get("trust_level") == "trusted":
                raise ValueError("trusted observations cannot be masked")
            obs.update(content="[SOURCE_REMOVED]", data_ids=[], contains_sensitive=False, instruction_like=False)
            affected += 1
    if not affected:
        raise ValueError("source group is absent from observations")
    return result


def explain_groups(request, ensemble, *, max_sources=6, max_group_size=2, budget=12, seed=101, replay_evaluator=None):
    if not 1 <= max_sources <= 16 or not 1 <= max_group_size <= 3 or not 1 <= budget <= 128:
        raise ValueError("source explanation budget is out of bounds")
    ranked = rank_sources(request, ensemble.registry)
    sources = [s for s, _ in ranked[:max_sources]]
    baseline_forecast = ensemble.predict({"none": request})["none"]
    base = baseline_forecast["risk_mean"]
    groups = list(itertools.chain.from_iterable(itertools.combinations(sources, size)
                  for size in range(1, min(max_group_size, len(sources)) + 1)))
    scores = dict(ranked)
    # Interleave group sizes so a small budget still exercises joint removals.
    ordered = []
    layers = [[g for g in groups if len(g) == size] for size in range(1, max_group_size + 1)]
    for layer in layers:
        layer.sort(key=lambda g: (-sum(scores[s] for s in g), g))
    while any(layers):
        for layer in layers:
            if layer:
                ordered.append(layer.pop(0))
    rng, outcomes = random.Random(seed), []
    all_sources = [s for s, _ in ranked]
    for group in ordered[:budget]:
        variant = mask_sources(request, group)
        risk = ensemble.predict({"none": variant})["none"]["risk_mean"]
        alternatives = [s for s in all_sources if s not in group]
        control = tuple(sorted(rng.sample(alternatives, len(group)))) if len(alternatives) >= len(group) else None
        control_risk = ensemble.predict({"none": mask_sources(request, control)})["none"]["risk_mean"] if control else None
        observed_outcome = replay_evaluator(request, variant, group) if replay_evaluator else None
        if observed_outcome is not None:
            observed_outcome={key:observed_outcome[key] for key in ("unsafe","utility","completed","candidate_action_changed","synthetic") if type(observed_outcome.get(key)) is bool}
        outcomes.append({"sources": list(group), "risk_after": risk, "risk_reduction": base - risk,
                         "random_control_sources": list(control) if control else None,
                         "random_control_risk_reduction": base - control_risk if control_risk is not None else None,
                         "replay_outcome": observed_outcome})
    singles = {r["sources"][0]: r["risk_reduction"] for r in outcomes if len(r["sources"]) == 1}
    for row in outcomes:
        row["interaction_excess"] = (row["risk_reduction"] - sum(singles[s] for s in row["sources"])) if len(row["sources"]) > 1 and all(s in singles for s in row["sources"]) else None
    public = lambda value: hashlib.sha256(str(value).encode()).hexdigest()[:16]
    for row in outcomes:
        row["sources"] = [public(x) for x in row["sources"]]
        row["random_control_sources"] = [public(x) for x in row["random_control_sources"]] if row["random_control_sources"] else None
    ranked = [(public(name), score) for name, score in ranked]
    graph=prepare_request(request,ensemble.registry,interventions=("none",)).branches["none"].graph
    structural=[{"edge_id":e["edge_id"],"relation":e["relation"],"available_step":e["available_step"],
                 "semantics":"temporal_association" if e["relation"] in ("follows","observed_before","state_update") else "structural_relation"}
                for e in graph.edge_metadata[:budget]]
    return {"model_forecast":baseline_forecast,"structural_evidence":structural,
            "format":"agent-astra-explanation-v1", "verified_replay":None,"baseline_risk_mean": base, "groups": outcomes, "ranking": ranked,
            "method": "graph-ancestor-group-removal-v1", "seed": seed,
            "evidence_path_verified": False,
            "scope": "model sensitivity; causal outcomes require trusted replay_evaluator",
            "argument_lineage_present": False}
