"""Privacy-risk formation: reversible conditions and first-readiness time.

Gold traces are separate from model requests. Input observations may contain
only current, independently verified conditions. 'Unknown' never means false.
Readiness is jointly predicted; we do not multiply condition marginals.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .constants import FORMATION_VERSION, RISK_CONDITIONS


def condition_values(mapping):
    if not isinstance(mapping, dict) or set(mapping) - set(RISK_CONDITIONS):
        raise ValueError("unknown condition name or invalid condition mapping")
    values = [mapping.get(name) for name in RISK_CONDITIONS]
    if any(v is not None and type(v) is not bool for v in values):
        raise ValueError("conditions require true, false or null")
    return values


def joint_ready(values):
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None


def validate_formation(trace, length, event_annotation=None, events=None):
    if trace is None:
        return
    if not isinstance(trace, dict):
        raise ValueError("formation trace must be an object")
    if trace.get("version") != FORMATION_VERSION or trace.get("verified") is not True or not trace.get("provenance"):
        raise ValueError("formation trace needs supported version and verified provenance")
    if type(trace.get("complete", False)) is not bool:
        raise ValueError("formation complete must be boolean")
    condition_values(trace["initial"])
    after = trace["after_actions"]
    if not isinstance(after, list) or len(after) != length:
        raise ValueError("formation trace must align exactly with executed actions")
    for values in after:
        condition_values(values)
    if events is not None:
        for event, current in zip(events, [trace["initial"], *after[:-1]]):
            observed = event.get("runtime", {})
            if observed.get("risk_conditions_verified") is True:
                values = condition_values(observed.get("risk_conditions", {}))
                for prediction_input, truth in zip(values, condition_values(current)):
                    if prediction_input is not None and truth is not None and prediction_input != truth:
                        raise ValueError("formation gold contradicts a verified current observation")
    if event_annotation and event_annotation.get("kind") == "exact" and event_annotation.get("event_type") == "privacy_leakage":
        states = [trace["initial"]] + after[:event_annotation["event_step"] + 1]
        if all(joint_ready(condition_values(s)) is False for s in states):
            raise ValueError("privacy event contradicts known never-ready conditions")


def formation_target(trace, step, horizon):
    if trace is None:
        return None
    initial = condition_values(trace["initial"])
    states = [condition_values(s) for s in trace["after_actions"]]
    past = [initial] + states[:step]
    readiness = [joint_ready(s) for s in past]
    formed = True if any(r is True for r in readiness) else False if all(r is False for r in readiness) else None
    future = states[step:step + horizon]
    if trace.get("complete") and len(future) < horizon:
        future += [states[-1]] * (horizon - len(future))
    future += [[None] * len(RISK_CONDITIONS)] * (horizon - len(future))
    target = {"current": past[-1], "future": future, "formed_before": formed,
              "kind": "right_censored", "lower": 0, "upper": 0, "censor_time": 0,
              "time_supervision": formed is False,
              "within_horizon": 1 if formed is True or any(joint_ready(s) is True for s in future) else None}
    if formed is not False:
        return target
    ready = [joint_ready(s) for s in future]
    first_true = next((i for i, r in enumerate(ready) if r is True), None)
    first_unknown = next((i for i, r in enumerate(ready) if r is None), None)
    if first_true is not None:
        lo = min(first_true, first_unknown) if first_unknown is not None else first_true
        target.update(kind="exact" if lo == first_true else "interval", lower=lo + 1,
                      upper=first_true + 1, within_horizon=1)
    else:
        target["censor_time"] = first_unknown if first_unknown is not None else horizon
        if target["censor_time"] == horizon:
            target["within_horizon"] = 0
    return target


class RiskFormationHead(nn.Module):
    def __init__(self, hidden_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.current = nn.Linear(hidden_dim, len(RISK_CONDITIONS))
        self.future = nn.Linear(hidden_dim, horizon * len(RISK_CONDITIONS))
        self.formed = nn.Linear(hidden_dim, 1)
        self.hazard = nn.Linear(hidden_dim, horizon)
        self.current_ready = nn.Linear(hidden_dim, 1)
        self.future_ready = nn.Linear(hidden_dim, horizon)

    def forward(self, state):
        current = self.current(state)
        future = self.future(state).reshape(-1, self.horizon, len(RISK_CONDITIONS))
        formed = self.formed(state).squeeze(-1)
        hazard = self.hazard(state)
        no = F.logsigmoid(-hazard)
        log_before = torch.cat([torch.zeros_like(no[:, :1]), no[:, :-1].cumsum(1)], 1)
        conditional_mass = (log_before + F.logsigmoid(hazard)).exp()
        prior = torch.sigmoid(formed)
        mass = (1 - prior[:, None]) * conditional_mass
        beyond = (1 - prior) * no.sum(1).exp()
        return {"current_logits": current, "future_logits": future,
                "current_ready_logit": self.current_ready(state).squeeze(-1), "future_ready_logits": self.future_ready(state),
                "formed_before_logit": formed, "hazard_logits": hazard,
                "formed_before_probability": prior, "first_formation_mass": mass,
                "beyond_horizon_probability": beyond,
                "formation_cdf": prior[:, None] + mass.cumsum(1)}


def formation_losses(prediction, labels):
    zero = prediction["hazard_logits"].sum()*0
    terms = {k:[] for k in ("conditions","formed_before","formation_time","ready_now","ready_future")}
    counts = {k:0 for k in ("known_condition_labels","current_condition_labels","future_condition_labels",
                           "formed_before_labels","formation_time_labels","ready_now_labels","ready_future_labels")}
    for i,label in enumerate(labels):
        if label.get("supervision",0) <= 0 or label.get("formation") is None: continue
        t = label["formation"]
        for key, logits, values in (("current",prediction["current_logits"][i], t["current"]),
                                    ("future",prediction["future_logits"][i].flatten(),[v for row in t["future"] for v in row])):
            ids=[j for j,v in enumerate(values) if v is not None]
            if ids:
                terms["conditions"].append(F.binary_cross_entropy_with_logits(logits[ids], logits.new_tensor([float(values[j]) for j in ids])))
                counts["known_condition_labels"] += len(ids)
                counts[key+"_condition_labels"] += len(ids)
        if t["formed_before"] is not None:
            terms["formed_before"].append(F.binary_cross_entropy_with_logits(prediction["formed_before_logit"][i], zero.new_tensor(float(t["formed_before"]))))
            counts["formed_before_labels"]+=1
        now = joint_ready(t["current"])
        if now is not None:
            terms["ready_now"].append(F.binary_cross_entropy_with_logits(prediction["current_ready_logit"][i],zero.new_tensor(float(now))))
            counts["ready_now_labels"]+=1
        future=[joint_ready(row) for row in t["future"]]
        ids=[j for j,v in enumerate(future) if v is not None]
        if ids:
            terms["ready_future"].append(F.binary_cross_entropy_with_logits(prediction["future_ready_logits"][i,ids],zero.new_tensor([float(future[j]) for j in ids])))
            counts["ready_future_labels"]+=len(ids)
        if t["time_supervision"]:
            logits=prediction["hazard_logits"][i]; no=F.logsigmoid(-logits)
            if t["kind"] in ("exact","interval"):
                before=torch.cat([no.new_zeros(1),no[:-1].cumsum(0)])
                terms["formation_time"].append(-torch.logsumexp((before+F.logsigmoid(logits))[t["lower"]-1:t["upper"]],0))
                counts["formation_time_labels"]+=1
            elif t["censor_time"]>0:
                terms["formation_time"].append(-no[:t["censor_time"]].sum())
                counts["formation_time_labels"]+=1
    return {**{k:torch.stack(v).mean() if v else zero for k,v in terms.items()},**counts}


class FormationMetrics:
    def __init__(self):
        self.condition_truth, self.condition_scores = [], []
        self.joint_now_truth, self.joint_now_scores, self.joint_future_truth, self.joint_future_scores = [], [], [], []
        self.future_truth, self.future_scores = [], []
        self.ready_truth, self.ready_scores, self.before_truth, self.before_scores = [], [], [], []
        self.time_errors, self.changes_truth, self.changes_pred = [], [], []

    def add(self, predictions, targets):
        joint_now = torch.stack([p["current_ready_logit"].sigmoid().detach().cpu() for p in predictions]).mean(0)
        joint_future = torch.stack([p["future_ready_logits"].sigmoid().detach().cpu() for p in predictions]).mean(0)
        current = torch.stack([p["current_logits"].sigmoid().detach().cpu() for p in predictions]).mean(0)
        future = torch.stack([p["future_logits"].sigmoid().detach().cpu() for p in predictions]).mean(0)
        cdf = torch.stack([p["formation_cdf"].detach().cpu() for p in predictions]).mean(0)
        before = torch.stack([p["formed_before_probability"].detach().cpu() for p in predictions]).mean(0)
        mass = torch.stack([p["first_formation_mass"].detach().cpu() for p in predictions]).mean(0)
        for i, label in enumerate(targets):
            target = label.get("formation")
            if target is None or label.get("supervision", 0) <= 0:
                continue
            now = joint_ready(target["current"])
            if now is not None:
                self.joint_now_truth.append(int(now)); self.joint_now_scores.append(float(joint_now[i]))
            for d, row in enumerate(target["future"]):
                truth=joint_ready(row)
                if truth is not None:
                    self.joint_future_truth.append(int(truth)); self.joint_future_scores.append(float(joint_future[i,d]))
            for j, truth in enumerate(target["current"]):
                if truth is not None:
                    self.condition_truth.append(int(truth)); self.condition_scores.append(float(current[i, j]))
            previous_truth, previous_prediction = target["current"], current[i]
            for t, row in enumerate(target["future"]):
                for j, truth in enumerate(row):
                    if truth is not None:
                        self.future_truth.append(int(truth)); self.future_scores.append(float(future[i, t, j]))
                    if truth is not None and previous_truth[j] is not None:
                        self.changes_truth.append(truth != previous_truth[j])
                        self.changes_pred.append(bool((future[i, t, j] >= .5) != (previous_prediction[j] >= .5)))
                previous_truth, previous_prediction = row, future[i, t]
            if target["within_horizon"] is not None:
                self.ready_truth.append(target["within_horizon"]); self.ready_scores.append(float(cdf[i, -1]))
            if target["formed_before"] is not None:
                self.before_truth.append(int(target["formed_before"])); self.before_scores.append(float(before[i]))
            if target["time_supervision"] and target["kind"] == "exact":
                expected = float((mass[i] * torch.arange(1, len(mass[i]) + 1)).sum() / mass[i].sum().clamp(min=1e-8))
                self.time_errors.append(abs(expected - target["lower"]))

    def report(self):
        from .metrics import probability_metrics
        tp = sum(a and b for a, b in zip(self.changes_truth, self.changes_pred))
        positives, predicted = sum(self.changes_truth), sum(self.changes_pred)
        precision = tp / predicted if predicted else 0.
        recall = tp / positives if positives else None
        return {"joint_ready_now": probability_metrics(self.joint_now_truth,self.joint_now_scores),
                "joint_ready_future": probability_metrics(self.joint_future_truth,self.joint_future_scores),
                "current_conditions": probability_metrics(self.condition_truth, self.condition_scores),
                "future_conditions": probability_metrics(self.future_truth, self.future_scores),
                "formed_before": probability_metrics(self.before_truth, self.before_scores),
                "formation_within_horizon": probability_metrics(self.ready_truth, self.ready_scores),
                "exact_future_formation_time_mae": sum(self.time_errors) / len(self.time_errors) if self.time_errors else None,
                "exact_future_formation_time_count": len(self.time_errors),
                "condition_change_accuracy": sum(a == b for a, b in zip(self.changes_truth, self.changes_pred)) / len(self.changes_truth) if self.changes_truth else None,
                "condition_change_positive_count": positives,
                "condition_change_precision": precision if self.changes_truth else None,
                "condition_change_recall": recall,
                "condition_change_f1": 2 * tp / (positives + predicted) if positives + predicted else None}
