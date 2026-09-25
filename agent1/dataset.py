"""Group-disjoint gold data, paired interventions and label-free model inputs."""
import copy
import hashlib
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .checkpoint import canonical_digest, file_sha256
from .constants import DEFAULT_EVENT_TYPES, INTERVENTIONS
from .data import PreparedExample, collate_prepared, prepare_request, MAX_HISTORY_EVENTS
from .io import read_jsonl, write_json, write_jsonl, verify_manifest
from .labels import last_prefix, target_for, validate_annotation
from .registry import ToolRegistry
from .formation import validate_formation, formation_target
from .constants import FORMATION_VERSION, CONCEPTS

SPLITS = ("train", "validation", "calibration", "test")


def assigned_split(group, seed):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < .6 else "validation" if value < .75 else "calibration" if value < .9 else "test"


def explanation_target(value, request, branch):
    if value is None: return {"concept_labels":None,"edge_labels":None}
    if value.get("version") != "agent1-explanation-labels-v1" or value.get("verified") is not True or not value.get("provenance") or value.get("request_digest") != canonical_digest(request):
        raise ValueError("explanation labels require bound verified provenance")
    concepts, edges = value.get("concepts",{}),value.get("edges",{})
    if set(concepts)-set(CONCEPTS) or any(v is not None and type(v) is not bool for v in [*concepts.values(),*edges.values()]):
        raise ValueError("invalid explanation labels")
    ids=[e["edge_id"] for e in branch.graph.edge_metadata]
    if set(edges)-set(ids): raise ValueError("explanation edge absent")
    return {"concept_labels":[concepts.get(k) for k in CONCEPTS],"edge_labels":[edges.get(k) for k in ids]}


def prepare_dataset(episodes_path, registry_path, output, *, annotations_path=None, horizon=5, history_limit=128, seed=101):
    if type(horizon) is not int or horizon < 1 or not 1 <= history_limit <= MAX_HISTORY_EVENTS:
        raise ValueError("invalid horizon or history limit")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; use a new dataset version")
    registry = ToolRegistry.from_json(registry_path)
    overrides = {}
    if annotations_path:
        for item in read_jsonl(annotations_path):
            key = item["episode_id"]
            if key in overrides:
                raise ValueError("duplicate annotation")
            overrides[key] = item["annotation"]
    rows = {name: [] for name in SPLITS}
    groups, seen, excluded = {}, set(), Counter()
    for episode in read_jsonl(episodes_path):
        identifier, group = episode["episode_id"], episode["group_id"]
        if not isinstance(identifier, str) or not identifier or not isinstance(group, str) or not group:
            raise ValueError("non-empty string episode_id and group_id required")
        if identifier in seen:
            raise ValueError("duplicate episode_id")
        seen.add(identifier)
        split = episode.get("split", assigned_split(group, seed))
        if split not in SPLITS or group in groups and groups[group] != split:
            raise ValueError("invalid split or group leakage")
        groups[group] = split
        annotation = overrides.get(identifier, episode.get("annotation"))
        events = episode["events"]
        if not events:
            raise ValueError("empty trajectory")
        end = last_prefix(annotation, len(events))
        formation = annotation.get("formation")
        validate_formation(formation, len(events), annotation, events)
        if end < 0:
            excluded["weak_no_action_time"] += 1
            continue
        for step in range(end + 1):
            if step > history_limit:
                raise ValueError("observed history exceeds declared data budget; use a verified state-summary adapter")
            request = {"trusted_goal": episode["trusted_goal"], "policy": episode["policy"],
                       "history": events[max(0, step - history_limit):step], "candidate": events[step]}
            baseline = prepare_request(request, registry, interventions=("none",))
            baseline_target = target_for(annotation, step, horizon)
            baseline_target["formation"] = formation_target(formation, step, horizon)
            baseline_target.update(explanation_target(annotation.get("explanation_by_step",{}).get(str(step)),request,baseline.branches["none"]))
            pairs = {}
            for name, pair in episode.get("pairs", {}).get(str(step), {}).items():
                if name not in INTERVENTIONS or name == "none":
                    raise ValueError("unknown repair branch")
                if pair.get("verified") is not True or not pair.get("provenance"):
                    raise ValueError("paired labels require verified replay provenance")
                if pair.get("baseline_digest") != canonical_digest(request):
                    raise ValueError("paired outcome is bound to a different baseline request")
                if pair.get("request_digest") != canonical_digest(pair["request"]):
                    raise ValueError("paired request digest mismatch")
                actual = pair["request"]
                for key in ("trusted_goal", "policy", "history"):
                    if actual[key] != request[key]:
                        raise ValueError("repair cannot rewrite trusted goal, policy or executed history")
                if actual["candidate"]["event_id"] != request["candidate"]["event_id"]:
                    raise ValueError("repair must target the same pending event")
                validate_annotation(pair["annotation"], pair["observed_steps"])
                if pair["annotation"]["kind"] == "weak":
                    continue
                prepare_request(actual, registry, interventions=("none",))
                pair_formation = pair["annotation"].get("formation")
                validate_formation(pair_formation, pair["observed_steps"], pair["annotation"], [actual["candidate"]])
                pair_target = target_for(pair["annotation"], 0, horizon)
                pair_target["formation"] = formation_target(pair_formation, 0, horizon)
                pair_target.update(explanation_target(pair["annotation"].get("explanation"),actual,prepare_request(actual, registry, interventions=("none",)).branches["none"]))
                pair_target["risk_increment_target"] = None
                if pair.get("comparison_verified") is True:
                    provenance = pair["provenance"]
                    if not isinstance(provenance, dict) or not provenance.get("snapshot_digest") or not provenance.get("continuation_policy_id"):
                        raise ValueError("risk increment supervision requires snapshot and continuation policy lineage")
                    if baseline_target["horizon_target"] is not None and pair_target["horizon_target"] is not None:
                        pair_target["risk_increment_target"] = baseline_target["horizon_target"] - pair_target["horizon_target"]
                pairs[name] = {"request": actual, "target": pair_target,
                               "provenance": pair["provenance"]}
            rows[split].append({"request": request, "target": baseline_target, "pairs": pairs,
                                "tracking": {"episode_id": identifier, "group_id": group, "step": step,
                                             "split": split, "annotation_kind": annotation["kind"],
                                             "source": episode.get("source", "custom"),
                                             "synthetic": episode.get("synthetic", False),
                                             "hard_failure": baseline.branches["none"].metadata["fail_closed"]}})
    if set(overrides) - seen:
        raise ValueError("annotations refer to unknown episodes")
    if not any(rows.values()):
        raise ValueError("no action-time supervision; annotate weak episodes before preparing this dataset")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "registry.json", registry.to_dict())
    files = {"registry": {"path": "registry.json", "sha256": file_sha256(output / "registry.json")}}
    for split in SPLITS:
        write_jsonl(output / f"{split}.jsonl", rows[split])
        files[split] = {"path": f"{split}.jsonl", "sha256": file_sha256(output / f"{split}.jsonl"),
                        "rows": len(rows[split]), "groups": len({r["tracking"]["group_id"] for r in rows[split]})}
    manifest = {"format": "agent-astra-dataset-v1", "horizon": horizon, "history_limit": history_limit,
                "formation_version": FORMATION_VERSION,
                "formation_rows": {s: sum(r["target"].get("formation") is not None for r in rs) for s, rs in rows.items()},
                "event_types": list(DEFAULT_EVENT_TYPES), "files": files, "seed": seed,
                "registry_digest": canonical_digest(registry.to_dict()),
                "source_sha256": file_sha256(episodes_path), "excluded": dict(excluded),
                "group_splits": groups, "synthetic": any(r["tracking"]["synthetic"] for rs in rows.values() for r in rs)}
    manifest["manifest_digest"] = canonical_digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


class PrefixDataset(Dataset):
    def __init__(self, root, split):
        if split not in SPLITS:
            raise ValueError("unknown split")
        self.manifest = verify_manifest(root)
        if self.manifest.get("format") != "agent-astra-dataset-v1":
            raise ValueError("regenerate data for the Astra contract")
        self.registry = ToolRegistry.from_json(Path(root) / self.manifest["files"]["registry"]["path"])
        self.rows = list(read_jsonl(Path(root) / self.manifest["files"][split]["path"]))
        if not self.rows:
            raise ValueError(f"empty {split} split; no fallback permitted")
        for row in self.rows:
            t = row["tracking"]
            if t["split"] != split or self.manifest["group_splits"][t["group_id"]] != split:
                raise ValueError("group/split mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_rows(rows, registry):
    names = [n for n in INTERVENTIONS if n == "none" or any(n in r["pairs"] for r in rows)]
    examples, targets = [], {name: [] for name in names}
    for row in rows:
        branches = {}
        for name in names:
            pair = row["pairs"].get(name)
            req = pair["request"] if pair else row["request"]
            branches[name] = prepare_request(req, registry, interventions=("none",)).branches["none"]
            target = copy.deepcopy(pair["target"] if pair else row["target"])
            if name != "none" and pair is None:
                target["supervision"] = 0.0
                target["utility"] = target["attack_intent"] = None
                target["formation"] = None
                target["risk_increment_target"] = None
                target["concept_labels"] = target["edge_labels"] = None
            targets[name].append(target)
        examples.append(PreparedExample(branches=branches, metadata={}))
    batch = collate_prepared(examples)
    batch["targets"] = targets
    batch["tracking"] = [r["tracking"] for r in rows]
    return batch


def make_loader(dataset, batch_size=16, training=False, seed=0):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training:
        # Equal group mass: long trajectories cannot dominate the objective.
        counts = Counter(r["tracking"]["group_id"] for r in dataset.rows)
        sampler = WeightedRandomSampler([1 / counts[r["tracking"]["group_id"]] for r in dataset.rows],
                                       len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                      collate_fn=partial(collate_rows, registry=dataset.registry))
