#!/usr/bin/env python3
"""Pinned dataset downloader. It is dry-run only unless --execute is supplied."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def selected_sources(config: Dict[str, Any], profile: str, names: Iterable[str]) -> List[str]:
    explicit = list(names)
    if explicit:
        unknown = set(explicit).difference(config["sources"])
        if unknown:
            raise ValueError(f"unknown dataset sources: {sorted(unknown)}")
        return explicit
    roles = {
        "train": {"train"},
        "eval": {"frozen_test", "ood_test"},
        "full": {"train", "frozen_test", "ood_test"},
    }[profile]
    return [
        name for name, source in config["sources"].items() if source["role"] in roles
    ]


def download_http(source: Dict[str, Any], destination: Path, overwrite: bool) -> Dict[str, Any]:
    expected_hash = source.get("sha256")
    expected_size = source.get("size_bytes")
    if destination.exists() and not overwrite:
        actual_hash = sha256_file(destination)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"existing file checksum mismatch: {destination}")
        return {
            "status": "existing_verified" if expected_hash else "existing_revision_only",
            "sha256": actual_hash,
            "size_bytes": destination.stat().st_size,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with requests.get(source["url"], stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", expected_size or 0))
        with temporary.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        handle_size = temporary.stat().st_size
    actual_hash = sha256_file(temporary)
    if expected_size is not None and handle_size != int(expected_size):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded size mismatch for {destination}")
    if expected_hash and actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded checksum mismatch for {destination}")
    os.replace(temporary, destination)
    return {
        "status": "downloaded_verified" if expected_hash else "downloaded_revision_only",
        "sha256": actual_hash,
        "size_bytes": handle_size,
    }


def download_git(source: Dict[str, Any], destination: Path, overwrite: bool) -> Dict[str, Any]:
    revision = str(source["revision"])
    if destination.exists() and not overwrite:
        head = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != revision:
            raise ValueError(f"existing repository revision mismatch: {destination}")
        return {"status": "existing_verified", "revision": head}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", source["url"], str(temporary)],
            check=True,
        )
        subprocess.run(["git", "-C", str(temporary), "checkout", revision], check=True)
        head = subprocess.check_output(
            ["git", "-C", str(temporary), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != revision:
            raise ValueError("checked-out git revision does not match manifest")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return {"status": "downloaded_verified", "revision": head}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--profile", choices=("train", "eval", "full"), default="train")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform network writes; without this flag the command is a dry run",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    names = selected_sources(config, args.profile, args.source)
    plan = {
        name: {
            "role": config["sources"][name]["role"],
            "format": config["sources"][name]["format"],
            "revision": config["sources"][name]["revision"],
            "url": config["sources"][name]["url"],
            "destination": str(args.output / config["sources"][name]["relative_path"]),
            "sha256": config["sources"][name].get("sha256"),
        }
        for name in names
    }
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "network_writes": False, "sources": plan}, indent=2))
        return

    results: Dict[str, Any] = {}
    for name in names:
        source = config["sources"][name]
        destination = args.output / source["relative_path"]
        if source["format"] == "git_repository":
            result = download_git(source, destination, args.overwrite)
        else:
            result = download_http(source, destination, args.overwrite)
        results[name] = {**plan[name], **result}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "sources": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "download_manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output / "download_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

