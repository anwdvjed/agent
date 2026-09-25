import copy
import json
import math

import pytest
import torch

from agent1.checkpoint import canonical_digest
from agent1.data import RequestValidationError, exact_action_digest, prepare_request
from agent1.dataset import PrefixDataset, collate_rows, prepare_dataset
from agent1.environments import DocumentEnvironment, generate_demo
from agent1.formation import RiskFormationHead, formation_losses, joint_ready
from agent1.graph import _PersistentGraphBuilder, _entity_digest
from agent1.io import read_json, read_jsonl, write_jsonl
from agent1.model import Agent1Config, Agent1Model
from agent1.objective import research_loss
from agent1.policy import select_action_v2
from test_policy_v2 import forecast, policy


def test_exact_action_unicode_and_json_duplicate_rejected(registry, tmp_path):
    request = DocumentEnvironment({"attack": False}, registry).request()
    original = exact_action_digest(request)
    changed = copy.deepcopy(request)
    changed["candidate"]["action"]["arguments"]["path"] += "Ｘ"
    assert exact_action_digest(changed) != original
    changed["candidate"]["action"]["tool"] = "ｒｅａｄ"
    with pytest.raises(RequestValidationError):
        prepare_request(changed, registry, interventions=("none",))
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)


def test_graph_identity_uses_full_value_and_final_taint_redacts_preview():
    left = "prefix" + "a" * 700 + "suffix"
    right = "prefix" + "b" * 700 + "suffix"
    assert _entity_digest(left) != _entity_digest(right)
    builder = _PersistentGraphBuilder()
    idx = builder.node("data", "same", left, flags={"sensitive": False}, metadata={"value_preview": "secret"})
    assert builder.node("data", "same", left, flags={"sensitive": True}) == idx
    assert builder.nodes[idx].metadata["value_preview"] == "[sensitive-value-hidden]"


def test_stable_edge_ids_survive_rebuild(registry):
    request = DocumentEnvironment({"attack": False}, registry).request()
    first = prepare_request(request, registry, interventions=("none",)).branches["none"].graph
    second = prepare_request(copy.deepcopy(request), registry, interventions=("none",)).branches["none"].graph
    assert [e["edge_id"] for e in first.edge_metadata] == [e["edge_id"] for e in second.edge_metadata]
    assert all(len(e["edge_id"]) == 64 for e in first.edge_metadata)


def test_historical_formation_does_not_cause_permanent_warning():
    value = forecast([.01] * 5)
    value["formation"] = {"formed_before_probability": .99, "p_ready_now": .01, "p_ready_after": [.01] * 5}
    result = select_action_v2({"none": value}, policy())
    assert result["decision"] == "ALLOW"
    assert result["alert"]["severity"] == "NONE"
    enabled = policy()
    enabled["readiness_warning"] = {"mode": "joint_occupancy_v2", "lookahead_bins": 2, "threshold": .5}
    value["head_availability"].update(formation_current_ready=True, formation_future_ready=True)
    value["formation"]["p_ready_after"] = [.1, .8, .1, .1, .1]
    assert "READINESS_WARNING" in select_action_v2({"none": value}, enabled)["alert"]["reason_codes"]


def test_zero_length_hazard_supervision_does_not_train_and_joint_is_reversible():
    assert joint_ready([True, True, True, True]) is True
    assert joint_ready([True, False, True, True]) is False
    assert joint_ready([True, None, True, True]) is None
    head = RiskFormationHead(6, 3)
    out = head(torch.zeros(1, 6))
    target = {"current": [True] * 4, "future": [[None] * 4] * 3,
              "formed_before": False, "time_supervision": True,
              "kind": "right_censored", "censor_time": 0, "lower": 0, "upper": 0}
    parts = formation_losses(out, [{"supervision": 1., "formation": target}])
    assert parts["formation_time_labels"] == 0
    assert parts["ready_now_labels"] == 1
    assert out["future_ready_logits"].shape == (1, 3)


def test_verified_explanation_labels_flow_through_dataset_and_training(registry, tmp_path):
    raw = tmp_path / "raw"
    generate_demo(raw, registry, groups=16)
    episodes = list(read_jsonl(raw / "episodes.jsonl"))
    episode = next(e for e in episodes if e["split"] == "train")
    request = {"trusted_goal": episode["trusted_goal"], "policy": episode["policy"],
               "history": [], "candidate": episode["events"][0]}
    graph = prepare_request(request, registry, interventions=("none",)).branches["none"].graph
    edge_id = graph.edge_metadata[0]["edge_id"]
    episode["annotation"]["explanation_by_step"] = {"0": {
        "version": "agent1-explanation-labels-v1", "verified": True,
        "provenance": "test-human-label", "request_digest": canonical_digest(request),
        "concepts": {"goal_deviation": True}, "edges": {edge_id: True}}}
    write_jsonl(raw / "episodes.jsonl", episodes)
    data = tmp_path / "data"
    prepare_dataset(raw / "episodes.jsonl", raw / "registry.json", data)
    dataset = PrefixDataset(data, "train")
    row = next(r for r in dataset.rows if r["tracking"]["episode_id"] == episode["episode_id"] and r["tracking"]["step"] == 0)
    batch = collate_rows([row], registry)
    labels = batch["targets"]["none"][0]
    assert labels["concept_labels"][0] is True
    assert sum(v is True for v in labels["edge_labels"]) == 1
    model = Agent1Model(Agent1Config(hidden_dim=12, edge_dim=4, graph_layers=1))
    output = model({"branches": batch["branches"]})
    loss, parts = research_loss(output, batch["targets"], concept_weight=.25, edge_weight=.25)
    loss.backward()
    assert math.isfinite(float(loss.detach())) and parts["concept_labels"] == 1 and parts["edge_labels"] == 1
    assert model.concept_head[-1].weight.grad is not None


def test_untrained_utility_requires_hold_even_with_zero_minimum():
    value = forecast([.01] * 5)
    value["utility_mean"] = None
    value["head_availability"]["utility"] = False
    chosen = select_action_v2({"none": value}, policy())
    assert chosen["decision"] == "HOLD"
    assert chosen["selected_branch"] is None
