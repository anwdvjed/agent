"""Processed JSONL loader and source-balanced Agent New model batches."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from agent_new.constants import CONCEPTS
from agent_new.data import collate_prepared, prepare_request
from agent_new.registry import ToolRegistry


FORBIDDEN_MODEL_KEYS = {
    "label",
    "risk_type",
    "failure_mode",
    "side_task",
    "side_task_success",
    "has_side_task",
    "policy_name",
    "source_dataset",
    "split",
    "attack_intent_target",
    "utility_target",
}


def _nested_keys(value: Any) -> set:
    if isinstance(value, Mapping):
        result = set(map(str, value))
        for child in value.values():
            result.update(_nested_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_nested_keys(child))
        return result
    return set()


def read_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            leaked = _nested_keys(value.get("request", {})) & FORBIDDEN_MODEL_KEYS
            if leaked:
                raise ValueError(
                    f"label/lineage leakage at {path}:{line_number}: {sorted(leaked)}"
                )
            yield value


class ProcessedPrefixDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        *,
        interventions: Sequence[str] = ("none",),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = json.loads(
            (self.data_dir / "manifest.json").read_text(encoding="utf-8")
        )
        files = self.manifest.get("split_files", {})
        if split not in files:
            raise KeyError(f"split is not present in processed manifest: {split}")
        entry = files[split]
        filename = entry["path"] if isinstance(entry, Mapping) else entry
        registry_name = self.manifest["split_registries"][split]
        self.registry = ToolRegistry.from_json(self.data_dir / registry_name)
        self.split = split
        self.event_types = tuple(self.manifest["event_types"])
        self.interventions = tuple(interventions)
        self.rows = list(read_jsonl_gz(self.data_dir / filename))
        if not self.rows:
            raise ValueError(f"processed split contains no records: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]

    @property
    def source_counts(self) -> Counter:
        return Counter(
            str(row["tracking"].get("source_family", "unknown")) for row in self.rows
        )


def collate_labeled_rows(
    rows: Sequence[Dict[str, Any]], registry: ToolRegistry, event_types: Sequence[str]
) -> Dict[str, Any]:
    prepared = [
        prepare_request(row["request"], registry, interventions=("none",))
        for row in rows
    ]
    batch = collate_prepared(prepared)
    event_type_indices: List[int] = []
    concept_targets: List[List[float]] = []
    concept_masks: List[List[float]] = []
    attack_targets: List[float] = []
    attack_masks: List[float] = []
    utility_targets: List[float] = []
    utility_masks: List[float] = []
    for row in rows:
        labels = row["labels"]
        event_name = labels.get("event_type")
        event_type_indices.append(
            list(event_types).index(event_name) if labels["event_observed"] else -1
        )
        targets = labels.get("concept_targets", {})
        masks = labels.get("concept_mask", {})
        concept_targets.append([float(targets.get(name, 0.0)) for name in CONCEPTS])
        concept_masks.append([float(masks.get(name, 0.0)) for name in CONCEPTS])
        attack_target = labels.get("attack_intent_target")
        attack_targets.append(float(attack_target or 0.0))
        attack_masks.append(float(labels.get("supervision_attack_intent", 0.0)))
        utility_target = labels.get("utility_target")
        utility_targets.append(float(utility_target or 0.0))
        utility_masks.append(float(labels.get("supervision_utility", 0.0)))
    batch.update(
        {
            "event_observed": torch.tensor(
                [bool(row["labels"]["event_observed"]) for row in rows],
                dtype=torch.bool,
            ),
            "event_time": torch.tensor(
                [int(row["labels"]["event_time"]) for row in rows], dtype=torch.long
            ),
            "event_type": torch.tensor(event_type_indices, dtype=torch.long),
            "censor_time": torch.tensor(
                [int(row["labels"]["censor_time"]) for row in rows], dtype=torch.long
            ),
            "concept_targets": torch.tensor(concept_targets, dtype=torch.float32),
            "concept_mask": torch.tensor(concept_masks, dtype=torch.float32),
            "attack_intent_target": torch.tensor(attack_targets, dtype=torch.float32),
            "supervision_attack_intent": torch.tensor(attack_masks, dtype=torch.float32),
            "utility_target": torch.tensor(utility_targets, dtype=torch.float32),
            "supervision_utility": torch.tensor(utility_masks, dtype=torch.float32),
            "supervision_survival": torch.tensor(
                [float(row["labels"].get("supervision_survival", 1.0)) for row in rows],
                dtype=torch.float32,
            ),
            "sample_weight": torch.ones(len(rows), dtype=torch.float32),
            "tracking": [dict(row["tracking"]) for row in rows],
        }
    )
    return batch


def make_dataloader(
    dataset: ProcessedPrefixDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    source_shares: Optional[Mapping[str, float]] = None,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers != 0:
        raise ValueError(
            "num_workers must remain 0 because the trusted registry is intentionally not pickled"
        )
    sampler = None
    if source_shares:
        counts = dataset.source_counts
        available = {
            source: float(share)
            for source, share in source_shares.items()
            if counts.get(source, 0) > 0 and float(share) > 0
        }
        total_share = sum(available.values())
        if total_share <= 0:
            raise ValueError("source_shares has no source present in the dataset")
        weights = [
            (available.get(str(row["tracking"].get("source_family")), 0.0) / total_share)
            / max(1, counts[str(row["tracking"].get("source_family"))])
            for row in dataset.rows
        ]
        generator = torch.Generator().manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda rows: collate_labeled_rows(
            rows, dataset.registry, dataset.event_types
        ),
    )


__all__ = [
    "FORBIDDEN_MODEL_KEYS",
    "ProcessedPrefixDataset",
    "collate_labeled_rows",
    "make_dataloader",
    "read_jsonl_gz",
]
