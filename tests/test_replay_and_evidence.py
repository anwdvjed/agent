import copy

import pytest
import torch

from agent1.data import prepare_request
from agent1.environments import DocumentEnvironment, source_replay_evaluator
from agent1.evidence import mask_sources, rank_sources
from agent1.metrics import probability_metrics, temporal_metrics
from agent1.replay import run_episode
from agent1.runtime import select_action
from test_policy_v2 import forecast, policy


def at_send(registry):
    env = DocumentEnvironment({"attack": True}, registry)
    env.commit(env.request())
    env.commit(env.request())
    return env


def test_repair_changes_actual_tool_state_and_retains_utility(registry):
    env = at_send(registry)
    unsafe, repaired = env.fork(), env.fork()
    unsafe.commit(unsafe.request())
    repaired.commit(repaired.repairs(repaired.request())["redact_sensitive"])
    while not unsafe.done:
        unsafe.commit(unsafe.request())
    while not repaired.done:
        repaired.commit(repaired.request())
    assert unsafe.outcome()["unsafe"]
    assert not repaired.outcome()["unsafe"]
    assert repaired.outcome()["utility"]
    assert env.outbox == []
    assert unsafe.outbox != repaired.outbox


def test_commit_rejects_stale_request_and_fake_hard_receipt(registry):
    env = at_send(registry)
    request = env.request()
    env.commit(request)
    with pytest.raises(ValueError):
        env.commit(request)
    fake = copy.deepcopy(env.request())
    fake["candidate"]["permission_receipt"]["receipt_verified"] = False
    with pytest.raises(ValueError):
        env.commit(fake)


def test_synchronized_source_removal_changes_graph_and_temporal_input(registry):
    env = at_send(registry)
    req = env.request()
    source = req["candidate"]["observation"]["source_id"]
    changed = mask_sources(req, [source])
    old = prepare_request(req, registry, interventions=("none",)).branches["none"]
    new = prepare_request(changed, registry, interventions=("none",)).branches["none"]
    assert not torch.equal(old.event_features, new.event_features)
    assert old.graph.node_features.shape != new.graph.node_features.shape or not torch.equal(old.graph.node_features, new.graph.node_features)
    assert changed["trusted_goal"] == req["trusted_goal"]
    assert any(s == source for s, _ in rank_sources(req, registry))
    assert req["candidate"]["observation"]["content"] != "[SOURCE_REMOVED]"


def test_history_beyond_32_retains_initial_sources(registry):
    env = DocumentEnvironment({"attack": True, "delay": 35}, registry)
    for _ in range(37):
        env.commit(env.request())
    request = env.request()
    prepared = prepare_request(request, registry, interventions=("none",))
    assert prepared.metadata["history_length"] == 37
    assert any(s == "tool-result-0" for s, _ in rank_sources(request, registry))


def test_probability_metrics_do_not_use_conservative_decision_score():
    assert probability_metrics([0, 1], [.1, .9])["brier"] == pytest.approx(.01)
    records = [dict(episode_id="a", step=0, kind="exact", event_time=1, event_observed=True, decision_score=.9)]
    metrics = temporal_metrics(records)
    assert metrics["pre_commit_interception_recall"] == 1
    assert metrics["at_least_one_step_early_recall"] == 0


def test_selector_never_relaxes_hard_failures():
    p = {"none": {"risk_mean": 0., "ensemble_disagreement": 0., "utility_mean": 1., "hard_failure": True}}
    assert select_action(p, {"threshold": 1.}) is None


def test_runner_continues_from_repaired_state(registry):
    class OracleForTest:
        def __init__(self):
            self.registry = registry
        def predict(self, requests):
            result = {}
            for name, request in requests.items():
                bad = "SECRET:" in request["candidate"]["action"]["arguments"].get("body", "")
                branch = prepare_request(request, registry, interventions=("none",)).branches["none"]
                value = forecast([.9 if bad else .01] + [.01] * 4)
                value.update(branch_id=name, candidate_event_id=request["candidate"]["event_id"],
                             prepared_digest=branch.prepared_action_digest, hard_failure=branch.metadata["fail_closed"],
                             hard_reasons=branch.metadata["fail_reasons"])
                result[name] = value
            return result
    outcome = run_episode({"episode_id": "e", "group_id": "g", "split": "test", "attack": True}, OracleForTest(), policy())
    assert outcome["repairs"] == 1
    assert outcome["executed_actions"] == 4
    assert outcome["utility"] and not outcome["unsafe"]
    assert outcome["completed"]


def test_joint_source_removal_has_real_regenerated_action_effect(registry):
    env = at_send(registry)
    request = env.request()
    class Predictor:
        def __init__(self):
            self.registry = registry
        def predict(self, requests):
            return {"none": {"risk_mean": float("SECRET:" in requests["none"]["candidate"]["action"]["arguments"].get("body", ""))}}
    evaluate = source_replay_evaluator({"attack": True}, Predictor())
    single = evaluate(request, mask_sources(request, ["tool-result-0"]), ["tool-result-0"])
    joint = evaluate(request, mask_sources(request, ["tool-result-0", "tool-result-1"]), ["tool-result-0", "tool-result-1"])
    assert single["unsafe"] and not single["candidate_action_changed"]
    assert not joint["unsafe"] and joint["candidate_action_changed"] and joint["utility"]
