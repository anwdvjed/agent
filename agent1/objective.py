"""Exact/interval/censored likelihood and observed paired-utility supervision."""
import torch
from torch.nn import functional as F

from .losses import factorized_competing_risk_losses
from .formation import formation_losses


def branch_losses(output, targets):
    device = output["event_logits"].device
    def tensor(key, dtype):
        return torch.tensor([t[key] for t in targets], device=device, dtype=dtype)
    observed = tensor("event_observed", torch.bool)
    times = tensor("event_time", torch.long)
    causes = tensor("event_type", torch.long)
    censor = tensor("censor_time", torch.long)
    losses = factorized_competing_risk_losses(output["event_logits"], output["cause_logits"], observed, times, causes, censor)
    # Sum event likelihood in log space; no probability clamp / dead BCE gradients.
    event = output["event_logits"]
    log_no = F.logsigmoid(-event)
    log_before = torch.cat([torch.zeros_like(log_no[:, :1]), log_no[:, :-1].cumsum(1)], 1)
    log_mass = log_before + F.logsigmoid(event)
    log_causes = F.log_softmax(output["cause_logits"], -1)
    replacements = []
    for i, target in enumerate(targets):
        if target["kind"] == "interval" and target["event_observed"]:
            start, end = target["lower"] - 1, target["upper"]
            replacements.append(-torch.logsumexp(log_mass[i, start:end] + log_causes[i, start:end, target["event_type"]], 0))
        else:
            replacements.append(losses[i])
    return torch.stack(replacements)


LOSS_DEFAULTS = {"utility_weight":.25,"intent_weight":.25,"sparsity_weight":.001,
                 "condition_weight":.25,"formation_weight":.5,"increment_weight":.1,"ordering_weight":.1,
                 "full_weight":1.,"sufficiency_weight":.1,"necessity_weight":0.,
                 "ready_now_weight":.25,"ready_future_weight":.25,"concept_weight":.25,"edge_weight":.25}


def research_loss(outputs, targets, *, cause_names=None, **config):
    import math
    if set(config)-set(LOSS_DEFAULTS): raise ValueError("unknown loss parameter")
    weights={**LOSS_DEFAULTS,**config}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError("loss weights must be finite and nonnegative")
    if weights["necessity_weight"]: raise ValueError("necessity requires verified removable evidence")
    zero=outputs["actual_risk"].sum()*0
    terms={k:[] for k in ("survival_nll","full_event_nll","utility_bce","intent_bce","sparsity",
                          "condition_bce","formation_nll","ready_now_bce","ready_future_bce","risk_increment_mse",
                          "formation_event_ordering","evidence_sufficiency","concept_bce","edge_bce")}
    counts={k:0 for k in ("supervised_branch_rows","known_condition_labels","formation_time_labels","risk_increment_pairs","concept_labels","edge_labels")}
    for name,labels in targets.items():
        b=outputs["branches"][name]
        mask=zero.new_tensor([t["supervision"] if t["event_observed"] or t["censor_time"]>0 else 0 for t in labels])
        if bool((mask>0).any()):
            counts["supervised_branch_rows"]+=int((mask>0).sum())
            terms["survival_nll"].append((branch_losses(b,labels)*mask).sum()/mask.sum())
            terms["full_event_nll"].append((branch_losses(b["full"],labels)*mask).sum()/mask.sum())
            suff=F.smooth_l1_loss(b["selected"]["cumulative_incidence"],b["full"]["cumulative_incidence"].detach(),reduction="none").mean((1,2))
            terms["evidence_sufficiency"].append((suff*mask).sum()/mask.sum())
            per=[]
            for i in range(len(labels)):
                edges=b["edge_batch"]==i
                if mask[i]>0 and bool(edges.any()): per.append(b["evidence_weights"][edges].mean())
            if per: terms["sparsity"].append(torch.stack(per).mean())
        for field,logit,key in (("utility","utility_logit","utility_bce"),("attack_intent","attack_intent_logit","intent_bce")):
            ids=[i for i,t in enumerate(labels) if t.get(field) is not None and t["supervision"]>0]
            if ids:
                terms[key].append(F.binary_cross_entropy_with_logits(b[logit][ids],zero.new_tensor([float(labels[i][field]) for i in ids])))
        for i,t in enumerate(labels):
            if t["supervision"]<=0: continue
            values=t.get("concept_labels") or []
            ids=[j for j,v in enumerate(values) if v is not None]
            if ids:
                terms["concept_bce"].append(F.binary_cross_entropy_with_logits(b["concept_logits"][i,ids],zero.new_tensor([float(values[j]) for j in ids])))
                counts["concept_labels"]+=len(ids)
            values=t.get("edge_labels")
            if values is not None:
                indices=(b["edge_batch"]==i).nonzero(as_tuple=True)[0]
                if len(values)!=len(indices): raise ValueError("edge labels misaligned")
                ids=[j for j,v in enumerate(values) if v is not None]
                if ids:
                    terms["edge_bce"].append(F.binary_cross_entropy_with_logits(b["evidence_logits"][indices[ids]],zero.new_tensor([float(values[j]) for j in ids])))
                    counts["edge_labels"]+=len(ids)
        if b.get("formation") is not None:
            f=formation_losses(b["formation"],labels)
            for src,dst,count in (("conditions","condition_bce","known_condition_labels"),("formed_before","formation_nll","formed_before_labels"),
                                  ("formation_time","formation_nll","formation_time_labels"),("ready_now","ready_now_bce","ready_now_labels"),
                                  ("ready_future","ready_future_bce","ready_future_labels")):
                if f[count]: terms[dst].append(f[src])
            counts["known_condition_labels"]+=f["known_condition_labels"]
            counts["formation_time_labels"]+=f["formation_time_labels"]
            if cause_names and "privacy_leakage" in cause_names:
                ids=[i for i,t in enumerate(labels) if t.get("formation") is not None and t["supervision"]>0]
                if ids:
                    terms["formation_event_ordering"].append(F.relu(b["selected"]["cumulative_incidence"][ids,:,cause_names.index("privacy_leakage")]-b["formation"]["formation_cdf"][ids]).square().mean())
        if name!="none":
            ids=[i for i,t in enumerate(labels) if t.get("risk_increment_target") is not None and t["supervision"]>0]
            if ids:
                delta=outputs["branches"]["none"]["risk"][ids]-b["risk"][ids]
                terms["risk_increment_mse"].append(F.mse_loss(delta,zero.new_tensor([labels[i]["risk_increment_target"] for i in ids])))
                counts["risk_increment_pairs"]+=len(ids)
    if not terms["survival_nll"]: raise ValueError("no verified outcome supervision in batch")
    means={k:torch.stack(v).mean() if v else zero for k,v in terms.items()}
    mapping={"full_event_nll":"full_weight","utility_bce":"utility_weight","intent_bce":"intent_weight","sparsity":"sparsity_weight",
             "condition_bce":"condition_weight","formation_nll":"formation_weight","ready_now_bce":"ready_now_weight",
             "ready_future_bce":"ready_future_weight","risk_increment_mse":"increment_weight","formation_event_ordering":"ordering_weight",
             "evidence_sufficiency":"sufficiency_weight","concept_bce":"concept_weight","edge_bce":"edge_weight"}
    total=means["survival_nll"]+sum(weights[mapping[k]]*v for k,v in means.items() if k!="survival_nll")
    return total,{"loss":float(total.detach()),**{k:float(v.detach()) for k,v in means.items()},**counts}
