import copy

import pytest

from agent1.dataset import PrefixDataset, prepare_dataset
from agent1.environments import collect_episode
from agent1.io import write_json, write_jsonl, read_json


def create(tmp_path, registry, rows):
    write_json(tmp_path / "registry.json", registry.to_dict())
    write_jsonl(tmp_path / "episodes.jsonl", rows)
    return prepare_dataset(tmp_path / "episodes.jsonl", tmp_path / "registry.json", tmp_path / "data")


def episode(registry, identifier="one", split="train"):
    return collect_episode({"episode_id": identifier, "group_id": identifier, "split": split, "attack": True}, registry)


def test_post_harm_summary_not_imputed_as_first_event(tmp_path, registry):
    row = episode(registry)
    assert row["annotation"]["event_step"] < len(row["events"]) - 1
    create(tmp_path, registry, [row])
    dataset = PrefixDataset(tmp_path / "data", "train")
    assert len(dataset) == row["annotation"]["event_step"] + 1
    assert dataset.rows[-1]["target"]["event_time"] == 1
    assert "redact_sensitive" in dataset.rows[-1]["pairs"]


def test_cross_split_group_leakage_rejected(tmp_path, registry):
    a, b = episode(registry), episode(registry, "two", "test")
    b["group_id"] = a["group_id"]
    with pytest.raises(ValueError, match="group leakage"):
        create(tmp_path, registry, [a, b])


def test_pair_request_and_history_bound(tmp_path, registry):
    row = episode(registry)
    pair = row["pairs"]["2"]["redact_sensitive"]
    pair["request"]["candidate"]["action"]["arguments"]["body"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        create(tmp_path, registry, [row])


def test_file_tampering_detected(tmp_path, registry):
    create(tmp_path, registry, [episode(registry)])
    with (tmp_path / "data/train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        PrefixDataset(tmp_path / "data", "train")


def test_weak_only_requires_annotation_and_test_has_no_fallback(tmp_path, registry):
    row = episode(registry)
    row["annotation"] = {"kind": "weak"}
    with pytest.raises(ValueError, match="no action-time"):
        create(tmp_path, registry, [row])
    create(tmp_path, registry, [episode(registry)])
    with pytest.raises(ValueError, match="empty test"):
        PrefixDataset(tmp_path / "data", "test")
