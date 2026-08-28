#!/usr/bin/env python3
"""Evaluate calibrated checkpoints on frozen test sets with visible epochs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.calibration import GroupConformalCalibration, SafeTrajectoryThreshold
from agent_new.checkpoint import load_checkpoint
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
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--operating-threshold", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluation.json")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "test_report.json")
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("frozen testing requires the calibrated multi-seed ensemble")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    epochs = int(config["epochs"] if args.epochs is None else args.epochs)
    if epochs <= 0:
        raise ValueError("test epochs must be positive")
    calibration = GroupConformalCalibration.from_dict(
        json.loads(args.calibration.read_text(encoding="utf-8"))
    )
    threshold = SafeTrajectoryThreshold(
        **json.loads(args.operating_threshold.read_text(encoding="utf-8"))
    )
    ensemble_z = float(config["ensemble_z"])
    if abs(ensemble_z - calibration.ensemble_z) > 1e-12:
        raise ValueError("test ensemble_z differs from calibration protocol")
    device = choose_device(args.device or config.get("device", "auto"))
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = args.split or sorted(
        split for split in manifest.get("split_files", {}) if split.startswith("test_")
    )
    if not splits:
        raise ValueError("no frozen test split is present; validation fallback is forbidden")

    states = []
    metadata_items = []
    fingerprints = set()
    model_config = None
    models = []
    for checkpoint in args.checkpoint:
        state, metadata = load_checkpoint(
            checkpoint,
            map_location=device,
            expected_calibration_digest=calibration.digest(),
        )
        current_config = AgentNewConfig.from_dict(metadata.model_config)
        if model_config is not None and current_config != model_config:
            raise ValueError("test ensemble model configurations differ")
        fingerprint = _state_dict_fingerprint(state)
        if fingerprint in fingerprints:
            raise ValueError("duplicate checkpoint state in test ensemble")
        fingerprints.add(fingerprint)
        model = AgentNewModel(current_config)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        states.append(state)
        metadata_items.append(metadata)
        model_config = current_config
    run_ids = [str(item.extra.get("training_run_id", "")) for item in metadata_items]
    seeds = [item.extra.get("seed") for item in metadata_items]
    if not all(run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("test ensemble requires distinct training_run_id metadata")
    if not all(seed is not None for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("test ensemble requires distinct seeds")

    results = {}
    for split in splits:
        dataset = ProcessedPrefixDataset(args.data_dir, split)
        if tuple(dataset.event_types) != tuple(metadata_items[0].event_types):
            raise ValueError(f"event ontology mismatch for split {split}")
        loader = make_dataloader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        epoch_results = []
        for epoch in range(1, epochs + 1):
            metrics, _ = collect_ensemble_epoch(
                models,
                loader,
                device,
                epoch=epoch,
                total_epochs=epochs,
                phase=f"Test[{split}]",
                ensemble_z=ensemble_z,
                threshold=threshold.threshold,
                calibration=calibration,
            )
            epoch_results.append(metrics)
            print(
                f"Test Epoch {epoch:03d}/{epochs:03d} [{split}] | "
                f"AUC={metrics['horizon_auc']} | F1={metrics['f1']} | "
                f"safe_trajectory_FPR={metrics['trajectory']['safe_trajectory_fpr']} | "
                f"early_warning_recall={metrics['trajectory']['early_warning_recall']}",
                flush=True,
            )
        results[split] = {
            "epochs": epoch_results,
            "registry": manifest["split_registries"][split],
            "frozen_external_test": True,
        }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "checkpoints": [str(path) for path in args.checkpoint],
        "calibration": str(args.calibration),
        "operating_threshold": threshold.to_dict(),
        "splits": results,
        "test_data_used_for_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
