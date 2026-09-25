import copy
import math

import pytest
import torch

from agent1.constants import RISK_CONDITIONS, FORMATION_VERSION
from agent1.formation import (RiskFormationHead, condition_values, joint_ready,
                              validate_formation, formation_target, formation_losses)
from agent1.formation_env import FormationDocumentEnvironment, collect_formation_episode
from agent1.dataset import prepare_dataset, PrefixDataset, collate_rows
from agent1.data import prepare_request
from agent1.io import write_json, write_jsonl
from agent1.model import Agent1Model, Agent1Config
from agent1.objective import research_loss


def state(value):
    return dict.fromkeys(RISK_CONDITIONS, value)


def trace(after, initial=None, complete=False):
    return {"version": FORMATION_VERSION, "verified": True, "provenance": "test-measurement",
            "initial": state(False) if initial is None else initial, "after_actions": after, "complete": complete}


def test_unknown_is_not_false_and_conditions_can_revert():
    assert joint_ready(condition_values(state(None))) is None
    assert joint_ready(condition_values({**state(None), RISK_CONDITIONS[0]: False})) is False
    t = trace([state(False), state(True), state(False)], complete=True)
    first = formation_target(t, 0, 5)
    assert first["kind"] == "exact" and first["lower"] == 2
    after = formation_target(t, 3, 5)
    assert after["formed_before"] is True
    assert not after["time_supervision"]
    assert after["current"] == [False] * 4


def test_unknown_gap_creates_interval_not_imputed_exact_time():
    target = formation_target(trace([state(False), state(None), state(True)]), 0, 5)
    assert target["kind"] == "interval" and target["lower"] == 2 and target["upper"] == 3
    unknown_past = formation_target(trace([state(None), state(False), state(True)]), 2, 5)
    assert unknown_past["formed_before"] is None
    assert unknown_past["time_supervision"] is False


def test_partial_censoring_and_terminal_completion():
    short = formation_target(trace([state(False)]), 0, 5)
    assert short["censor_time"] == 1 and short["within_horizon"] is None
    complete = formation_target(trace([state(False)], complete=True), 0, 5)
    assert complete["censor_time"] == 5 and complete["within_horizon"] == 0


def test_distribution_includes_past_mass_and_normalizes():
    head = RiskFormationHead(8, 5)
    output = head(torch.randn(4, 8))
    total = output["formed_before_probability"] + output["first_formation_mass"].sum(1) + output["beyond_horizon_probability"]
    assert torch.allclose(total, torch.ones(4), atol=1e-6)
    assert (output["formation_cdf"][:, 1:] >= output["formation_cdf"][:, :-1]).all()


def test_exact_interval_nll_and_finite_gradients():
    head = RiskFormationHead(8, 5)
    for p in head.parameters():
        torch.nn.init.zeros_(p)
    out = head(torch.ones(1, 8))
    out["hazard_logits"].retain_grad()
    t = formation_target(trace([state(False), state(None), state(True)]), 0, 5)
    losses = formation_losses(out, [{"supervision": 1., "formation": t}])
    assert losses["formation_time"].item() == pytest.approx(-math.log(.25 + .125))
    losses["formation_time"].backward()
    assert out["hazard_logits"].grad.abs().sum() > 0
    assert torch.isfinite(out["hazard_logits"].grad).all()


def test_missing_formation_has_zero_auxiliary_supervision():
    output = RiskFormationHead(8, 5)(torch.randn(1, 8))
    losses = formation_losses(output, [{"supervision": 1.}])
    assert losses["known_condition_labels"] == losses["formation_time_labels"] == 0
    assert float(losses["formation_time"].detach()) == 0


@pytest.mark.parametrize("bad", ["false", 0, 1, [], {}])
def test_condition_types_strict(bad):
    with pytest.raises(ValueError):
        condition_values({RISK_CONDITIONS[0]: bad})


def test_privacy_event_cannot_precede_known_readiness():
    with pytest.raises(ValueError, match="contradicts"):
        validate_formation(trace([state(False), state(False)]), 2,
                           {"kind": "exact", "event_type": "privacy_leakage", "event_step": 0})


def test_verified_current_observation_cannot_contradict_gold():
    event = {"runtime": {"risk_conditions_verified": True, "risk_conditions": state(True)}}
    with pytest.raises(ValueError, match="current observation"):
        validate_formation(trace([state(True)]), 1, events=[event])


def test_unknown_past_but_observed_future_ready_has_positive_horizon_label():
    target = formation_target(trace([state(True)], initial=state(None)), 0, 5)
    assert target["formed_before"] is None and not target["time_supervision"]
    assert target["within_horizon"] == 1


def test_pair_increment_is_missing_without_comparison_provenance(tmp_path, registry):
    episode = collect_formation_episode({"episode_id": "a", "group_id": "a", "split": "train", "attack": True}, registry)
    for pairs in episode["pairs"].values():
        for pair in pairs.values():
            pair.pop("comparison_verified")
    write_json(tmp_path / "registry.json", registry.to_dict())
    write_jsonl(tmp_path / "episodes.jsonl", [episode])
    prepare_dataset(tmp_path / "episodes.jsonl", tmp_path / "registry.json", tmp_path / "data")
    data = PrefixDataset(tmp_path / "data", "train")
    assert all(p["target"]["risk_increment_target"] is None for row in data.rows for p in row["pairs"].values())


def test_current_condition_graph_and_unverified_observations(registry):
    env = FormationDocumentEnvironment({"attack": True}, registry)
    request = env.request()
    nodes = prepare_request(request, registry, interventions=("none",)).branches["none"].graph.node_metadata
    condition_nodes = [n for n in nodes if n["type"] == "risk_condition"]
    assert len(condition_nodes) == 4
    assert {n["condition_status"] for n in condition_nodes} == {"true", "false"}
    request["candidate"]["runtime"]["risk_conditions_verified"] = False
    nodes = prepare_request(request, registry, interventions=("none",)).branches["none"].graph.node_metadata
    assert all(n["condition_status"] == "unknown" for n in nodes if n["type"] == "risk_condition")


def test_environment_readiness_precedes_harm_and_can_disappear(registry):
    scenario = {"episode_id": "a", "group_id": "a", "split": "train", "attack": True}
    unsafe = collect_formation_episode(scenario, registry)
    ready = [i for i, s in enumerate(unsafe["annotation"]["formation"]["after_actions"]) if joint_ready(condition_values(s))]
    assert min(ready) < unsafe["annotation"]["event_step"]
    recovered = collect_formation_episode({**scenario, "sanitize_after_prepare": True}, registry)
    states = [joint_ready(condition_values(s)) for s in recovered["annotation"]["formation"]["after_actions"]]
    assert True in states and states[-1] is False
    assert recovered["annotation"]["kind"] == "right_censored" and recovered["annotation"]["utility"]


def test_targets_do_not_leak_future_conditions_and_pairs_train(tmp_path, registry):
    episode = collect_formation_episode({"episode_id": "a", "group_id": "a", "split": "train", "attack": True}, registry)
    write_json(tmp_path / "registry.json", registry.to_dict())
    write_jsonl(tmp_path / "episodes.jsonl", [episode])
    prepare_dataset(tmp_path / "episodes.jsonl", tmp_path / "registry.json", tmp_path / "data")
    data = PrefixDataset(tmp_path / "data", "train")
    row = next(r for r in data.rows if r["pairs"])
    assert row["pairs"]["redact_sensitive"]["target"]["risk_increment_target"] == 1
    batch = collate_rows([row], registry)
    changed = copy.deepcopy(row)
    changed["target"]["formation"]["future"] = [[None] * 4] * 5
    changed_batch = collate_rows([changed], registry)
    assert torch.equal(batch["branches"]["none"]["event_features"], changed_batch["branches"]["none"]["event_features"])
    assert torch.equal(batch["branches"]["none"]["node_features"], changed_batch["branches"]["none"]["node_features"])
    model = Agent1Model(Agent1Config(hidden_dim=16, edge_dim=8, graph_layers=1))
    out = model(batch)
    value, metrics = research_loss(out, batch["targets"])
    value.backward()
    assert model.formation_head.hazard.weight.grad.abs().sum() > 0
    assert model.formation_head.future.weight.grad.abs().sum() > 0
    assert metrics["risk_increment_pairs"] == 1 and metrics["known_condition_labels"] > 0


def test_event_only_ablation_and_missing_pairs(registry):
    env = FormationDocumentEnvironment({"attack": True}, registry)
    model = Agent1Model(Agent1Config(hidden_dim=12, edge_dim=4, graph_layers=1, formation_enabled=False))
    from agent1.data import collate_prepared
    batch = collate_prepared([prepare_request(env.request(), registry, interventions=("none",))])
    assert model(batch)["branches"]["none"]["formation"] is None
