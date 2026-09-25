"""Safe, integrity-checked checkpoint metadata and tensor state loading.

Checkpoint metadata is JSON.  Tensor state is stored separately and loaded only
with ``torch.load(weights_only=True)``.  A digest-named immutable tensor file is
written before an atomic metadata-pointer replacement, making the JSON metadata
the transaction commit point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from .constants import CHECKPOINT_VERSION


PathLike = Union[str, Path]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible lineage metadata."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> str:
    result = str(value).lower()
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError("%s must be a lowercase SHA-256 hex digest" % name)
    return result


def _validate_event_types(values: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError("event_types must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError("event_types must not contain duplicates")
    return result


def _validate_model_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("model_config must be a non-empty mapping")
    result = dict(value)
    # This both verifies JSON serializability and rejects NaN/Infinity.
    _canonical_json(result)
    return result


@dataclass(frozen=True)
class CheckpointMetadata:
    format: str
    created_at: str
    model_config: Dict[str, Any]
    event_types: Tuple[str, ...]
    registry_digest: str
    calibration_digest: str
    state_dict_file: str
    state_dict_sha256: str
    extra: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.format != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint format: %s" % self.format)
        if not self.created_at:
            raise ValueError("created_at must not be empty")
        _validate_model_config(self.model_config)
        _validate_event_types(self.event_types)
        if (
            "num_event_types" in self.model_config
            and int(self.model_config["num_event_types"]) != len(self.event_types)
        ):
            raise ValueError("model_config num_event_types does not match event_types")
        _validate_digest(self.registry_digest, "registry_digest")
        _validate_digest(self.calibration_digest, "calibration_digest")
        _validate_digest(self.state_dict_sha256, "state_dict_sha256")
        if not self.state_dict_file or Path(self.state_dict_file).name != self.state_dict_file:
            raise ValueError("state_dict_file must be a safe relative file name")
        if not isinstance(self.extra, dict):
            raise ValueError("extra metadata must be a mapping")
        _canonical_json(self.extra)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["event_types"] = list(self.event_types)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        required = {
            "format",
            "created_at",
            "model_config",
            "event_types",
            "registry_digest",
            "calibration_digest",
            "state_dict_file",
            "state_dict_sha256",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError("checkpoint metadata is missing keys: %s" % sorted(missing))
        return cls(
            format=str(value["format"]),
            created_at=str(value["created_at"]),
            model_config=_validate_model_config(value["model_config"]),
            event_types=_validate_event_types(value["event_types"]),
            registry_digest=_validate_digest(value["registry_digest"], "registry_digest"),
            calibration_digest=_validate_digest(
                value["calibration_digest"], "calibration_digest"
            ),
            state_dict_file=str(value["state_dict_file"]),
            state_dict_sha256=_validate_digest(
                value["state_dict_sha256"], "state_dict_sha256"
            ),
            extra=dict(value.get("extra", {})),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_dict())


def _checkpoint_layout(path: PathLike) -> Tuple[Path, Path, str]:
    value = Path(path)
    if value.suffix.lower() in {".pt", ".pth"}:
        directory = value.parent
        metadata_path = value.with_suffix(".json")
        state_prefix = value.stem
    else:
        directory = value
        metadata_path = directory / "metadata.json"
        state_prefix = "state_dict"
    return directory, metadata_path, state_prefix


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    result: Dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError("state_dict keys must be non-empty strings")
        if not torch.is_tensor(value):
            raise ValueError("state_dict values must be tensors")
        result[name] = value.detach().cpu()
    return result


def save_checkpoint(
    path: PathLike,
    state_dict: Mapping[str, torch.Tensor],
    *,
    model_config: Mapping[str, Any],
    event_types: Sequence[str],
    registry_digest: str,
    calibration_digest: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> CheckpointMetadata:
    """Atomically commit JSON metadata pointing to an immutable tensor file."""

    directory, metadata_path, state_prefix = _checkpoint_layout(path)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = _validate_state_dict(state_dict)
    config = _validate_model_config(model_config)
    types = _validate_event_types(event_types)
    registry = _validate_digest(registry_digest, "registry_digest")
    calibration = _validate_digest(calibration_digest, "calibration_digest")
    extra = dict(extra_metadata or {})
    _canonical_json(extra)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=state_prefix + ".", suffix=".pt.part", dir=str(directory)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(tensors, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        state_digest = file_sha256(temporary)
        state_name = "%s.state-%s.pt" % (state_prefix, state_digest[:16])
        state_path = directory / state_name
        if state_path.exists():
            if file_sha256(state_path) != state_digest:
                raise ValueError("digest-named state file has unexpected contents")
            temporary.unlink()
        else:
            os.replace(str(temporary), str(state_path))
            _fsync_directory(directory)

        metadata = CheckpointMetadata(
            format=CHECKPOINT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_config=config,
            event_types=types,
            registry_digest=registry,
            calibration_digest=calibration,
            state_dict_file=state_name,
            state_dict_sha256=state_digest,
            extra=extra,
        )
        # The metadata pointer is the commit point.  Until this replace, an old
        # checkpoint continues to reference its old immutable tensor file.
        _atomic_write_json(metadata_path, metadata.to_dict())
        return metadata
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_metadata(path: Path) -> CheckpointMetadata:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint metadata JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return CheckpointMetadata.from_dict(value)


def load_checkpoint(
    path: PathLike,
    *,
    map_location: Union[str, torch.device] = "cpu",
    expected_model_config: Optional[Mapping[str, Any]] = None,
    expected_event_types: Optional[Sequence[str]] = None,
    expected_registry_digest: Optional[str] = None,
    expected_calibration_digest: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], CheckpointMetadata]:
    """Load an integrity-checked state dict using ``weights_only=True`` only."""

    directory, metadata_path, _ = _checkpoint_layout(path)
    metadata = _read_metadata(metadata_path)
    if expected_model_config is not None and _canonical_json(metadata.model_config) != _canonical_json(
        _validate_model_config(expected_model_config)
    ):
        raise ValueError("checkpoint model configuration does not match")
    if expected_event_types is not None and metadata.event_types != _validate_event_types(
        expected_event_types
    ):
        raise ValueError("checkpoint event types do not match")
    if expected_registry_digest is not None and metadata.registry_digest != _validate_digest(
        expected_registry_digest, "expected_registry_digest"
    ):
        raise ValueError("checkpoint registry digest does not match")
    if (
        expected_calibration_digest is not None
        and metadata.calibration_digest
        != _validate_digest(expected_calibration_digest, "expected_calibration_digest")
    ):
        raise ValueError("checkpoint calibration digest does not match")

    state_path = directory / metadata.state_dict_file
    if not state_path.is_file():
        raise FileNotFoundError("checkpoint state file not found: %s" % state_path)
    actual_digest = file_sha256(state_path)
    if actual_digest != metadata.state_dict_sha256:
        raise ValueError("checkpoint state_dict SHA-256 mismatch")
    try:
        value = torch.load(
            state_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "installed PyTorch does not support safe weights_only checkpoint loading"
        ) from exc
    tensors = _validate_state_dict(value)
    return tensors, metadata


__all__ = [
    "CheckpointMetadata",
    "canonical_digest",
    "file_sha256",
    "load_checkpoint",
    "save_checkpoint",
]
