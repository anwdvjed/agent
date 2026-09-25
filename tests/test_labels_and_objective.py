import math

import pytest
import torch

from agent1.labels import last_prefix, target_for, validate_annotation
from agent1.model import Agent1Config, Agent1Model
from agent1.objective import branch_losses, research_loss
from agent1.dataset import collate_rows
from agent1.environments import DocumentEnvironment


def exact(step=2):
    return {"kind": "exact", "event_step": step, "event_type": "privacy_leakage", "verified": True, "provenance": "annotator:test"}


def test_no_terminal_imputation_and_no_post_event_risk_set():
    assert last_prefix(exact(), 9) == 2
    assert target_for(exact(), 1, 5)["event_time"] == 2
    with pytest.raises(ValueError, match="post-first"):
        target_for(exact(), 3, 5)
    assert last_prefix({"kind": "weak"}, 9) == -1


def test_interval_likelihood_matches_sum_not_endpoint():
    annotation = {"kind": "interval", "lower_step": 1, "upper_step": 2,
                  "event_type": "privacy_leakage", "verified": True, "provenance": "two-annotators"}
    assert last_prefix(annotation, 5) == 1
    target = target_for(annotation, 0, 5)
    logits = torch.zeros(1, 5, requires_grad=True)
    loss = branch_losses({"event_logits": logits, "cause_logits": torch.zeros(1, 5, 1)}, [target])
    assert loss.item() == pytest.approx(-math.log(.25 + .125))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_censoring_is_not_automatically_negative_horizon_label():
    incomplete = {"kind": "right_censored", "observed_steps": 2, "verified": True, "provenance": "log"}
    assert target_for(incomplete, 0, 5)["horizon_target"] is None
    complete = {**incomplete, "complete": True}
    assert target_for(complete, 0, 5)["horizon_target"] == 0
    interval = {"kind": "interval", "lower_step": 2, "upper_step": 7, "event_type": "privacy_leakage"}
    target = target_for(interval, 0, 5)
    assert target["censor_time"] == 2
    assert target["horizon_target"] is None


@pytest.mark.parametrize("annotation", [
    {"kind": "exact", "event_step": True, "event_type": "privacy_leakage", "verified": True, "provenance": "x"},
    {"kind": "exact", "event_step": 0, "event_type": "privacy_leakage", "verified": "true", "provenance": "x"},
    {"kind": "interval", "lower_step": 3, "upper_step": 1, "event_type": "privacy_leakage", "verified": True, "provenance": "x"},
])
def test_bad_gold_rejected(annotation):
    with pytest.raises(ValueError):
        validate_annotation(annotation, 5)


def test_paired_targets_train_both_risk_and_utility(registry):
    env = DocumentEnvironment({"attack": True}, registry)
    env.commit(env.request())
    env.commit(env.request())
    request = env.request()
    target = target_for({**exact(0), "utility": True}, 0, 5)
    repair_target = target_for({"kind": "right_censored", "observed_steps": 2, "complete": True, "utility": True}, 0, 5)
    row = {"request": request, "target": target, "tracking": {},
           "pairs": {"redact_sensitive": {"request": env.repairs(request)["redact_sensitive"], "target": repair_target}}}
    batch = collate_rows([row], registry)
    model = Agent1Model(Agent1Config(hidden_dim=16, edge_dim=8, graph_layers=1))
    output = model(batch)
    output["branches"]["redact_sensitive"]["event_logits"].retain_grad()
    output["branches"]["redact_sensitive"]["utility_logit"].retain_grad()
    loss, _ = research_loss(output, batch["targets"])
    loss.backward()
    assert output["branches"]["redact_sensitive"]["event_logits"].grad.abs().sum() > 0
    assert output["branches"]["redact_sensitive"]["utility_logit"].grad.abs().sum() > 0


def test_extreme_interval_logits_have_finite_nonzero_gradients():
    annotation = {"kind": "interval", "lower_step": 0, "upper_step": 1, "event_type": "privacy_leakage"}
    logits = torch.full((1, 5), -100., requires_grad=True)
    value = branch_losses({"event_logits": logits, "cause_logits": torch.zeros(1, 5, 1)}, [target_for(annotation, 0, 5)])
    value.sum().backward()
    assert torch.isfinite(value).all()
    assert logits.grad.abs().sum() > 0
