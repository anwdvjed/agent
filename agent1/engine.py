"""Reproducible training, validation and frozen probability/temporal evaluation."""
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import canonical_digest, save_checkpoint
from .dataset import PrefixDataset, make_loader
from .io import write_json, write_jsonl
from .metrics import probability_metrics, temporal_metrics
from .model import Agent1Config, Agent1Model
from .objective import research_loss, LOSS_DEFAULTS
from .supervision import HeadTracker
from .policy import build_decision_curve
from .runtime import move_nested, implementation_digest
from .formation import FormationMetrics
from .labels import risk_at_bin

DEFAULT_TRAIN = {"epochs":30,"batch_size":16,"learning_rate":.0003,"weight_decay":.0001,
                 "patience":5,"gradient_clip":1.,"threads":1,**LOSS_DEFAULTS}


def validate_train_config(config):
    import math
    if set(config)-set(DEFAULT_TRAIN): raise ValueError("unknown training setting")
    config={**DEFAULT_TRAIN,**config}
    for key,v in config.items():
        if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise ValueError("invalid training setting")
    for key in ("epochs","batch_size","patience","threads"):
        if type(config[key]) is not int or config[key]<1: raise ValueError("invalid integer training setting")
    if config["learning_rate"]<=0 or config["gradient_clip"]<=0: raise ValueError("invalid optimizer setting")
    if config["necessity_weight"]: raise ValueError("necessity supervision unavailable")
    return config


def epoch_pass(models, loader, device, *, optimizer=None, config=None, description="Validation", disagreement_penalty=1.0, tracker=None, cause_names=None):
    config = validate_train_config(config or {})
    for model in models:
        model.train(optimizer is not None)
    totals, samples, records, branch_scores = {}, 0, [], {}
    formation_metrics, delta_truth, delta_predictions = {}, [], []
    curve_metrics = {}
    count_keys = {"supervised_branch_rows", "known_condition_labels", "formation_time_labels", "risk_increment_pairs", "concept_labels", "edge_labels"}
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for raw in tqdm(loader, desc=description, unit="batch"):
            batch = move_nested({"branches": raw["branches"]}, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = [m(batch) for m in models]
            loss, parts = research_loss(predictions[0], raw["targets"], cause_names=cause_names,
                                        **{key:config[key] for key in LOSS_DEFAULTS})
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(models[0].parameters(), config["gradient_clip"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradients")
                optimizer.step()
                if tracker is not None: tracker.record(raw["targets"])
            count = len(raw["tracking"])
            samples += count
            for name, value in parts.items():
                totals[name] = totals.get(name, 0) + value * (1 if name in count_keys else count)
            for name, targets in raw["targets"].items():
                formation = [p["branches"][name].get("formation") for p in predictions]
                if all(p is not None for p in formation):
                    formation_metrics.setdefault(name, FormationMetrics()).add(formation, targets)
                values = torch.stack([p["branches"][name]["risk"].detach().cpu() for p in predictions])
                mean, std = values.mean(0), values.std(0, unbiased=False)
                member_cdf = torch.stack([p["branches"][name]["selected"]["cumulative_incidence"].detach().cpu().sum(-1) for p in predictions])
                cdf_mean, cdf_std = member_cdf.mean(0), member_cdf.std(0,unbiased=False)
                utility = torch.stack([torch.sigmoid(p["branches"][name]["utility_logit"]).detach().cpu() for p in predictions]).mean(0)
                incidence = torch.stack([p["branches"][name]["selected"]["incidence_mass"].detach().cpu() for p in predictions]).mean(0).sum(-1)
                bucket = branch_scores.setdefault(name, {"truth": [], "probability": [], "utility_truth": [], "utility_probability": [], "time_errors": []})
                for i, target in enumerate(targets):
                    if not target["supervision"]:
                        continue
                    for d in sorted({1,min(3,cdf_mean.shape[1]),cdf_mean.shape[1]}):
                        truth=risk_at_bin(target,d)
                        entry=curve_metrics.setdefault(name,{}).setdefault(str(d),{"truth":[],"predictions":[]})
                        if truth is not None:
                            entry["truth"].append(truth); entry["predictions"].append(float(cdf_mean[i,d-1]))
                    if name != "none" and target.get("risk_increment_target") is not None:
                        baseline_mean = sum(float(p["branches"]["none"]["risk"][i].detach()) for p in predictions) / len(predictions)
                        delta_truth.append(target["risk_increment_target"])
                        delta_predictions.append(baseline_mean - float(mean[i]))
                    if target["horizon_target"] is not None:
                        bucket["truth"].append(target["horizon_target"])
                        bucket["probability"].append(float(mean[i]))
                    if target["utility"] is not None:
                        bucket["utility_truth"].append(int(target["utility"]))
                        bucket["utility_probability"].append(float(utility[i]))
                    if target["kind"] == "exact" and target["event_observed"]:
                        mass = incidence[i]
                        expected = float((mass * torch.arange(1, len(mass) + 1)).sum() / mass.sum().clamp(min=1e-8))
                        bucket["time_errors"].append(abs(expected - target["event_time"]))
                    if name == "none":
                        curve=build_decision_curve(cdf_mean[i].tolist(),cdf_std[i].tolist(),disagreement_penalty)
                        records.append({**raw["tracking"][i], "kind": target["kind"], "event_time": target["event_time"],
                                        "event_observed": target["event_observed"], "risk_mean": float(mean[i]),
                                        "ensemble_disagreement": float(std[i]), "decision_score": curve["decision_score"][-1], "event_cdf_mean":cdf_mean[i].tolist()})
    metrics = {name: value if name in count_keys else value / samples for name, value in totals.items()}
    metrics["probability_by_branch"] = {name: {**probability_metrics(v["truth"], v["probability"]),
                                               "utility": probability_metrics(v["utility_truth"], v["utility_probability"]),
                                               "exact_time_mae": sum(v["time_errors"]) / len(v["time_errors"]) if v["time_errors"] else None}
                                         for name, v in branch_scores.items()}
    metrics["temporal"] = temporal_metrics(records)
    metrics["risk_by_bin"] = {name:{d:probability_metrics(v["truth"],v["predictions"]) for d,v in bins.items()} for name,bins in curve_metrics.items()}
    metrics["formation_by_branch"] = {name: value.report() for name, value in formation_metrics.items()}
    metrics["risk_increment"] = {"pairs": len(delta_truth),
                                 "mae": sum(abs(a - b) for a, b in zip(delta_truth, delta_predictions)) / len(delta_truth) if delta_truth else None,
                                 "scope": "matched observed horizon outcomes; not an identified population causal effect"}
    metrics["loss_scope"] = "first ensemble member NLL; probability metrics use ensemble mean"
    return metrics, records


def train(data_dir, output, *, seed=101, config=None, model_config=None, device="cpu"):
    config = validate_train_config(config or {})
    if config["epochs"] < 1 or config["patience"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("invalid training parameters")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("training output must be empty; checkpoints are never silently overwritten")
    torch.set_num_threads(config["threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training, validation = PrefixDataset(data_dir, "train"), PrefixDataset(data_dir, "validation")
    values = dict(model_config or {})
    values["horizon"] = training.manifest["horizon"]
    cfg = Agent1Config.from_dict(values)
    model = Agent1Model(cfg).to(device)
    initial_fingerprint = canonical_digest({"seed": seed, "config": cfg.to_dict()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    tracker = HeadTracker(model, optimizer, config)
    best_heads = None
    train_loader = make_loader(training, config["batch_size"], training=True, seed=seed)
    validation_loader = make_loader(validation, config["batch_size"])
    best, best_state, history, stale, best_epoch = float("inf"), None, [], 0, 0
    for epoch in range(1, config["epochs"] + 1):
        tr, _ = epoch_pass([model], train_loader, device, optimizer=optimizer, config=config, description=f"Train {epoch}/{config['epochs']}", tracker=tracker, cause_names=training.manifest["event_types"])
        va, _ = epoch_pass([model], validation_loader, device, config=config, description=f"Validation {epoch}/{config['epochs']}", cause_names=training.manifest["event_types"])
        history.append({"epoch": epoch, "train": tr, "validation": va})
        print(f"epoch={epoch} train_loss={tr['loss']:.5f} validation_loss={va['loss']:.5f}", flush=True)
        if va["loss"] < best - 1e-9:
            best, stale, best_epoch = va["loss"], 0, epoch
            best_heads = tracker.snapshot()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    output.mkdir(parents=True, exist_ok=True)
    training_groups = sorted({r["tracking"]["group_id"] for r in training.rows})
    development_groups = sorted(set(training_groups) | {r["tracking"]["group_id"] for r in validation.rows})
    trained_heads = {name:value["trained"] for name,value in best_heads["heads"].items()}
    supervised = {name:value["effective_labels"] for name,value in best_heads["heads"].items()}
    save_checkpoint(output / "best", best_state, model_config=cfg.to_dict(), event_types=training.manifest["event_types"],
                    registry_digest=training.manifest["registry_digest"],
                    calibration_digest=canonical_digest({"status": "not_policy_calibrated"}),
                    extra_metadata={"task": "agent-astra-risk-v1", "seed": seed, "training_run_id": f"seed-{seed}-{initial_fingerprint[:12]}",
                                    "dataset_manifest_digest": training.manifest["manifest_digest"], "training_groups": training_groups,
                                    "development_groups": development_groups, "best_epoch": best_epoch,
                                    "implementation_digest": implementation_digest(), "prediction_head_contract": best_heads, "training_config": config,
                                    "trained_prediction_heads": trained_heads, "prediction_supervision_counts": supervised,
                                    "synthetic": training.manifest["synthetic"]})
    report = {"seed": seed, "epochs_completed": len(history), "best_epoch": best_epoch, "best_validation_loss": best,
              "checkpoint": str(output / "best"), "config": config, "model_config": cfg.to_dict(),
              "synthetic": training.manifest["synthetic"], "torch_version": torch.__version__, "history": history}
    report["prediction_head_contract"] = best_heads
    report["trained_prediction_heads"] = trained_heads
    report["prediction_supervision_counts"] = supervised
    write_json(output / "training.json", report)
    return report


def evaluate(ensemble, data_dir, split, output, batch_size=16):
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    dataset = PrefixDataset(data_dir, split)
    if dataset.manifest["registry_digest"] != ensemble.registry_digest or dataset.manifest["horizon"] != ensemble.config.horizon:
        raise ValueError("evaluation registry or horizon mismatch")
    if split == "test" and any(r["tracking"]["group_id"] in ensemble.development_groups for r in dataset.rows):
        raise ValueError("test overlaps model development groups")
    if split == "validation" and dataset.manifest["manifest_digest"] != ensemble.training_manifest_digest:
        raise ValueError("validation dataset differs from training lineage")
    metrics, records = epoch_pass(ensemble.models, make_loader(dataset, batch_size), ensemble.device, description=split.capitalize(), cause_names=dataset.manifest["event_types"])
    report = {"split": split, "metrics": metrics, "synthetic": dataset.manifest["synthetic"],
              "ensemble_digest": ensemble.digest, "dataset_manifest_digest": dataset.manifest["manifest_digest"],
              "head_availability":ensemble.trained_heads,"model_config":ensemble.config.to_dict()}
    write_json(output, report)
    write_jsonl(Path(output).with_suffix(".predictions.jsonl"), records)
    return report
