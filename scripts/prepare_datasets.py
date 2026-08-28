#!/usr/bin/env python3
"""Normalize pinned datasets into grouped Agent New prefix records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm import tqdm
import pyarrow.parquet as parquet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_new.constants import CONCEPTS, DEFAULT_EVENT_TYPES
from agent_new.datasets import (
    NormalizedTrajectory,
    build_tool_registry,
    iter_assebench,
    iter_atbench,
    iter_linuxarena,
    iter_openagentsafety,
    prefix_rows,
)
from agent_new.datasets.common import assign_split, stable_digest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_record_counts(raw: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    counts: Dict[str, Any] = {}
    for name, source in config["sources"].items():
        path = raw / source["relative_path"]
        if not path.exists() or source["format"] == "git_repository":
            continue
        if source["format"] == "parquet":
            actual = parquet.ParquetFile(path).metadata.num_rows
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("data", value.get("records", value.get("test", [])))
            actual = len(value) if isinstance(value, list) else None
        counts[name] = {
            "actual": actual,
            "expected": source.get("expected_records"),
            "matches": source.get("expected_records") in (None, actual),
        }
    return counts


class GzipJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.temporary.open("wb")
        zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        import io

        self.raw = raw
        self.text = io.TextIOWrapper(zipped, encoding="utf-8")
        self.rows = 0

    def write(self, value: Dict[str, Any]) -> None:
        self.text.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        )
        self.rows += 1

    def close(self) -> None:
        self.text.close()
        self.raw.close()
        os.replace(self.temporary, self.path)


def _train_trajectories(raw: Path) -> List[NormalizedTrajectory]:
    safety = raw / "assebench" / "AgentJudge-safety.json"
    security = raw / "assebench" / "AgentJudge-security.json"
    linux = raw / "linuxarena" / "train-00000-of-00001.parquet"
    for path in (safety, security, linux):
        if not path.exists():
            raise FileNotFoundError(f"required raw dataset is missing: {path}")
    result = list(iter_assebench(safety, "safety"))
    result.extend(iter_assebench(security, "security"))
    result.extend(iter_linuxarena(linux))
    return result


def _eval_sets(raw: Path) -> Dict[str, List[NormalizedTrajectory]]:
    result: Dict[str, List[NormalizedTrajectory]] = {}
    atbench = raw / "atbench" / "test.json"
    if atbench.exists():
        result["test_atbench"] = list(iter_atbench(atbench))
    oas = raw / "openagentsafety" / "repository"
    if oas.exists():
        result["test_openagentsafety"] = list(iter_openagentsafety(oas))
    if not result:
        raise FileNotFoundError(
            "no reserved evaluation dataset found; run the downloader with --profile eval --execute"
        )
    return result


def _write_registry(path: Path, trajectories: Sequence[NormalizedTrajectory], version: str) -> None:
    registry = build_tool_registry(trajectories, version)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.json")
    parser.add_argument("--profile", choices=("train", "eval", "full"), default="train")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--history-limit", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-record-counts", action="store_true")
    args = parser.parse_args()
    if args.horizon <= 0 or not 0 <= args.history_limit <= 32:
        raise ValueError("horizon must be positive and history-limit must be in [0, 32]")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_record_counts = _raw_record_counts(args.raw, config)
    mismatches = {
        name: value for name, value in raw_record_counts.items() if not value["matches"]
    }
    if mismatches and args.strict_record_counts:
        raise ValueError(f"raw dataset record counts differ from the pinned plan: {mismatches}")
    forbidden = set(config.get("forbidden_training_files", []))
    discovered_forbidden = [
        str(path) for path in args.raw.rglob("*.json") if path.name in forbidden
    ]
    # Vendored files may exist in an evaluation repository; they are never read.
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"processed manifest already exists: {manifest_path}; pass --overwrite explicitly"
        )

    split_files: Dict[str, str] = {}
    split_counts: Dict[str, int] = {}
    trajectory_counts: Dict[str, int] = {}
    split_sources: Dict[str, Dict[str, int]] = {}
    split_registries: Dict[str, str] = {}
    group_owner: Dict[str, str] = {}

    if args.profile in {"train", "full"}:
        trajectories = _train_trajectories(args.raw)
        _write_registry(
            args.output / "train_tool_registry.json",
            trajectories,
            "derived-train-registry-weak-semantics-v1",
        )
        writers = {
            split: GzipJsonlWriter(args.output / f"{split}.jsonl.gz")
            for split in ("train", "validation", "calibration")
        }
        source_counts: Dict[str, Counter] = defaultdict(Counter)
        trajectory_split_counts: Counter = Counter()
        try:
            for trajectory in tqdm(trajectories, desc="prepare training trajectories"):
                split = assign_split(trajectory.group_id, args.seed)
                previous = group_owner.setdefault(trajectory.group_id, split)
                if previous != split:
                    raise ValueError("trajectory/task group crosses dataset splits")
                trajectory_split_counts[split] += 1
                for row in prefix_rows(
                    trajectory,
                    split,
                    horizon=args.horizon,
                    history_limit=args.history_limit,
                ):
                    writers[split].write(row)
                    source_counts[split][trajectory.source_family] += 1
        finally:
            for writer in writers.values():
                writer.close()
        for split, writer in writers.items():
            split_files[split] = writer.path.name
            split_counts[split] = writer.rows
            trajectory_counts[split] = trajectory_split_counts[split]
            split_sources[split] = dict(sorted(source_counts[split].items()))
            split_registries[split] = "train_tool_registry.json"

    if args.profile in {"eval", "full"}:
        for split, trajectories in _eval_sets(args.raw).items():
            registry_name = f"{split}_tool_registry.json"
            _write_registry(
                args.output / registry_name,
                trajectories,
                f"derived-{split}-registry-weak-semantics-v1",
            )
            writer = GzipJsonlWriter(args.output / f"{split}.jsonl.gz")
            try:
                for trajectory in tqdm(trajectories, desc=f"prepare {split}"):
                    for row in prefix_rows(
                        trajectory,
                        split,
                        horizon=args.horizon,
                        history_limit=args.history_limit,
                    ):
                        writer.write(row)
            finally:
                writer.close()
            split_files[split] = writer.path.name
            split_counts[split] = writer.rows
            trajectory_counts[split] = len(trajectories)
            split_sources[split] = {
                split.removeprefix("test_"): writer.rows
            }
            split_registries[split] = registry_name

    files = {
        split: {
            "path": filename,
            "sha256": _sha256(args.output / filename),
            "rows": split_counts[split],
        }
        for split, filename in split_files.items()
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "seed": args.seed,
        "horizon": args.horizon,
        "history_limit": args.history_limit,
        "event_types": list(DEFAULT_EVENT_TYPES),
        "concepts": list(CONCEPTS),
        "split_policy": "deterministic 80/10/10 hash over complete trajectory/task groups",
        "training_mix_target": config.get("training_mix", {}),
        "split_files": files,
        "split_registries": split_registries,
        "trajectory_counts": trajectory_counts,
        "source_prefix_counts": split_sources,
        "raw_record_counts": raw_record_counts,
        "raw_record_count_warnings": mismatches,
        "forbidden_files_discovered_but_not_read": discovered_forbidden,
        "notes": [
            "ASSEBench unsafe event times are explicitly weak terminal proxies.",
            "LinuxArena side_task/policy/side_task_success remain labels or lineage only.",
            "LinuxArena action_reasoning is excluded from model input to avoid deployment mismatch.",
            "Derived tool registries use deterministic weak capability semantics and require review before deployment.",
            "OpenAgentSafety and ATBench are evaluation-only and never enter train/validation/calibration.",
        ],
        "manifest_digest": stable_digest(
            [args.profile, args.seed, args.horizon, files, split_registries]
        ),
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
