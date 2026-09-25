"""Small, deterministic, integrity-checked JSON artifacts."""
import json
import os
from pathlib import Path

from .checkpoint import canonical_digest, file_sha256


def _strict_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result: raise ValueError("duplicate JSON key")
        result[k] = v
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON number")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_bad_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line, object_pairs_hook=_strict_pairs, parse_constant=_bad_constant)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                yield value


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def checked_path(root, relative, digest):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != digest:
        raise ValueError("artifact path escape or SHA-256 mismatch")
    return path


def verify_manifest(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if canonical_digest(payload) != manifest.get("manifest_digest"):
        raise ValueError("manifest digest mismatch")
    for item in manifest["files"].values():
        checked_path(root, item["path"], item["sha256"])
    return manifest
