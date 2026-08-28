#!/usr/bin/env python3
"""Inspect processed batches and optionally run an untrained forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.datasets.loader import ProcessedPrefixDataset, make_dataloader
from agent_new.model import AgentNewConfig, AgentNewModel


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    dataset = ProcessedPrefixDataset(args.data_dir, args.split)
    loader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    model = None
    if args.forward:
        model = AgentNewModel(
            AgentNewConfig(
                horizon=int(dataset.manifest["horizon"]),
                num_event_types=len(dataset.event_types),
            )
        )
        model.eval()
    for batch_index, batch in enumerate(loader, 1):
        branch = batch["branches"]["none"]
        report = {
            "batch": batch_index,
            "event_features": list(branch["event_features"].shape),
            "node_features": list(branch["node_features"].shape),
            "edge_index": list(branch["edge_index"].shape),
            "event_observed": batch["event_observed"].long().tolist(),
            "event_time": batch["event_time"].tolist(),
            "sources": [item["source_family"] for item in batch["tracking"]],
        }
        if model is not None:
            output = model(batch)
            report["risk"] = output["actual_risk"].detach().tolist()
            report["attack_intent_probability"] = (
                output["actual"]["attack_intent_logit"].sigmoid().detach().tolist()
            )
            report["utility_probability"] = (
                output["actual"]["utility_logit"].sigmoid().detach().tolist()
            )
        print(json.dumps(report, indent=2))
        if batch_index >= args.max_batches:
            break


if __name__ == "__main__":
    main()
