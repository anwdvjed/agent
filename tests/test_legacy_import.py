import json

from agent1.importers import import_legacy
from agent1.io import read_jsonl


def test_legacy_import_does_not_fabricate_first_event_or_authority(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"label": 1, "contents": [
        {"role": "user", "content": "Read the report."},
        {"role": "agent", "action": {"tool": "read_document", "arguments": {"path": "/report"}}}
    ]}]))
    report = import_legacy("assebench-safety", source, tmp_path / "imported")
    assert report["episodes"] == 1
    row = next(read_jsonl(tmp_path / "imported/episodes.jsonl"))
    assert row["annotation"]["kind"] == "weak"
    assert "event_step" not in row["annotation"]
    assert "assebench" not in row["events"][0]["event_id"]
    assert not row["events"][0]["permission_receipt"]["receipt_verified"]
