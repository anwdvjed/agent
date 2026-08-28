#!/usr/bin/env python3
"""Fit group calibration for an ensemble with visible validation epochs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.calibration import (
    fit_group_conformal,
    select_threshold_at_safe_trajectory_fpr,
)
from agent_new.checkpoint import canonical_digest, load_checkpoint, save_checkpoint
from agent_new.datasets.loader import ProcessedPrefixDataset, make_dataloader
from agent_new.ensemble_evaluation import collect_ensemble_epoch
from agent_new.evaluation import choose_device
from agent_new.model import AgentNewConfig, AgentNewModel
from agent_new.runtime import _state_dict_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "validation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("calibrated admission requires at least two independent checkpoints")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    epochs = int(config["epochs"] if args.epochs is None else args.epochs)
    if epochs <= 0:
        raise ValueError("validation epochs must be positive")
    device = choose_device(args.device or config.get("device", "auto"))
    dataset = ProcessedPrefixDataset(args.data_dir, "calibration")
    loader = make_dataloader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    registry_digest = canonical_digest(dataset.registry.to_dict())
    models = []
    payloads = []
    states = []
    fingerprints = set()
    model_config = None
    for checkpoint in args.checkpoint:
        state, metadata = load_checkpoint(
            checkpoint,
            map_location=device,
            expected_event_types=dataset.event_types,
            expected_registry_digest=registry_digest,
        )
        current_config = AgentNewConfig.from_dict(metadata.model_config)
        if model_config is not None and current_config != model_config:
            raise ValueError("ensemble checkpoints use different model configurations")
        fingerprint = _state_dict_fingerprint(state)
        if fingerprint in fingerprints:
            raise ValueError("duplicate model state cannot be used as an ensemble")
        fingerprints.add(fingerprint)
        model = AgentNewModel(current_config)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        payloads.append(metadata)
        states.append(state)
        model_config = current_config
    run_ids = [str(metadata.extra.get("training_run_id", "")) for metadata in payloads]
    seeds = [metadata.extra.get("seed") for metadata in payloads]
    if not all(run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("validation requires distinct training_run_id metadata")
    if not all(seed is not None for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("validation requires distinct training seeds")

    epoch_metrics = []
    raw = None
    for epoch in range(1, epochs + 1):
        metrics, raw = collect_ensemble_epoch(
            models,
            loader,
            device,
            epoch=epoch,
            total_epochs=epochs,
            phase="Validation",
            ensemble_z=float(config["ensemble_z"]),
            threshold=0.5,
        )
        epoch_metrics.append(metrics)
        print(
            f"Validation Epoch {epoch:03d}/{epochs:03d} | "
            f"AUC={metrics['horizon_auc']} | ECE={metrics['horizon_ece']} | "
            f"trajectory_FPR={metrics['trajectory']['safe_trajectory_fpr']}",
            flush=True,
        )
    assert raw is not None
    calibration = fit_group_conformal(
        raw["scores"],
        raw["targets"],
        raw["group_ids"],
        alpha=float(config["alpha"]),
        ensemble_z=float(config["ensemble_z"]),
    )
    threshold = select_threshold_at_safe_trajectory_fpr(
        raw["scores"],
        raw["targets"],
        raw["group_ids"],
        target_fpr=float(config["target_safe_trajectory_fpr"]),
        calibration=calibration,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "operating_threshold.json").write_text(
        json.dumps(threshold.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    calibration_digest = calibration.digest()
    threshold_digest = canonical_digest(threshold.to_dict())
    calibrated_paths = []
    checkpoint_dir = args.output_dir / "checkpoints"
    for state, metadata, run_id, seed in zip(states, payloads, run_ids, seeds):
        destination = checkpoint_dir / run_id
        extra = dict(metadata.extra)
        extra.update(
            {
                "seed": seed,
                "training_run_id": run_id,
                "calibration_status": "calibrated_actual_branch_only",
                "calibration_digest": calibration_digest,
                "operating_threshold_digest": threshold_digest,
            }
        )
        save_checkpoint(
            destination,
            state,
            model_config=metadata.model_config,
            event_types=metadata.event_types,
            registry_digest=metadata.registry_digest,
            calibration_digest=calibration_digest,
            extra_metadata=extra,
        )
        calibrated_paths.append(str(destination))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "epochs": epoch_metrics,
        "calibration": calibration.to_dict(),
        "operating_threshold": threshold.to_dict(),
        "calibrated_checkpoints": calibrated_paths,
        "repair_branch_certification": "disabled",
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
