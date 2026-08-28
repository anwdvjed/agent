#!/usr/bin/env python3
"""Train one Agent New seed with visible train/validation epochs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.checkpoint import canonical_digest, save_checkpoint
from agent_new.datasets.loader import ProcessedPrefixDataset, make_dataloader
from agent_new.evaluation import choose_device, evaluate_epoch
from agent_new.losses import LossWeights
from agent_new.model import AgentNewConfig, AgentNewModel
from agent_new.training import run_epoch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model.json")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "training.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "runs" / "default")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.train_config.read_text(encoding="utf-8"))
    seed = int(config["seed"] if args.seed is None else args.seed)
    run_id = args.run_id or f"seed-{seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(args.device)

    train_dataset = ProcessedPrefixDataset(args.data_dir, "train")
    validation_dataset = ProcessedPrefixDataset(args.data_dir, "validation")
    if train_dataset.event_types != validation_dataset.event_types:
        raise ValueError("train and validation event ontologies differ")
    train_loader = make_dataloader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        source_shares=config.get("source_batch_share"),
        seed=seed,
        num_workers=int(config.get("num_workers", 0)),
    )
    validation_loader = make_dataloader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    manifest = train_dataset.manifest
    model_values = json.loads(args.model_config.read_text(encoding="utf-8"))
    model_values.update(
        {
            "horizon": int(manifest["horizon"]),
            "num_event_types": len(train_dataset.event_types),
        }
    )
    model_config = AgentNewConfig.from_dict(model_values)
    model = AgentNewModel(model_config).to(device)
    loss_weights = LossWeights.from_dict(config.get("loss", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    stale = 0
    epochs = int(config["epochs"])
    pending_calibration_digest = canonical_digest(
        {"status": "pending_group_calibration", "run_id": run_id}
    )
    registry_digest = canonical_digest(train_dataset.registry.to_dict())
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch:03d}/{epochs:03d} ===", flush=True)
        train_metrics = run_epoch(
            model,
            tqdm(
                train_loader,
                desc=f"Train Epoch {epoch:03d}/{epochs:03d}",
                unit="batch",
            ),
            device,
            loss_weights,
            optimizer,
            float(config["gradient_clip"]),
        )
        validation_metrics, _ = evaluate_epoch(
            model,
            validation_loader,
            device,
            train_dataset.event_types,
            epoch=epoch,
            total_epochs=epochs,
            phase="Validation",
            threshold=0.5,
            loss_weights=loss_weights,
        )
        validation_loss = float(validation_metrics["loss"])
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['total']:.5f} | "
            f"validation_loss={validation_loss:.5f} | "
            f"validation_auc={validation_metrics['horizon']['auc']} | "
            f"intent_auc={None if validation_metrics['attack_intent'] is None else validation_metrics['attack_intent']['auc']} | "
            f"utility_auc={None if validation_metrics['utility'] is None else validation_metrics['utility']['auc']}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            stale = 0
            save_checkpoint(
                args.output_dir / "best",
                model.state_dict(),
                model_config=model_config.to_dict(),
                event_types=train_dataset.event_types,
                registry_digest=registry_digest,
                calibration_digest=pending_calibration_digest,
                extra_metadata={
                    "training_run_id": run_id,
                    "seed": seed,
                    "epoch": epoch,
                    "dataset_manifest_digest": manifest["manifest_digest"],
                    "calibration_status": "pending",
                },
            )
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if stale >= int(config["patience"]):
            print(f"Early stopping after epoch {epoch:03d}; best validation loss={best_loss:.5f}")
            break
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "seed": seed,
        "device": str(device),
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "checkpoint": str(args.output_dir / "best"),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
