"""Trusted, fail-closed tool capability registry.

The registry is the authority for capability semantics.  Fields supplied by an
agent such as ``operation``, ``required_scopes`` or ``approval_required`` are
never consulted when a request is prepared; only the registered ``ToolSpec``
is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .constants import OPERATIONS


MAX_REGISTERED_TOOLS = 4096
MAX_TOOL_ALIASES = 32
MAX_REQUIRED_SCOPES = 32
MAX_ARGUMENT_ROLES = 64
MAX_IDENTIFIER_LENGTH = 160

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:/@-]+$")

ALLOWED_ARGUMENT_ROLES = {
    "data",
    "payload",
    "content",
    "attachment",
    "resource",
    "record",
    "query",
    "command",
    "identifier",
    "destination",
    "recipient",
    "url",
    "endpoint",
    "path_destination",
    "external_destination",
}
ALLOWED_SIDE_EFFECTS = {
    "network",
    "external_transfer",
    "internal_write",
    "filesystem",
    "code_execution",
    "system_change",
    "data_deletion",
    "irreversible",
    "permission_change",
    "unknown_effect",
}


class RegistryError(ValueError):
    """Base class for trusted-registry validation errors."""


class UnknownToolError(RegistryError):
    """Raised when an action names no registered tool or alias."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"tool is not present in the trusted registry: {tool_name!r}")
        self.tool_name = tool_name


def _identifier(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or len(text) > MAX_IDENTIFIER_LENGTH or not IDENTIFIER_RE.fullmatch(text):
        raise RegistryError(f"invalid {field_name}: {text!r}")
    return text


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    limit: int,
    identifiers: bool = True,
) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RegistryError(f"{field_name} must be a sequence of strings")
    if len(value) > limit:
        raise RegistryError(f"{field_name} exceeds the limit of {limit}")
    result = []
    seen = set()
    for raw in value:
        item = (
            _identifier(raw, field_name=field_name)
            if identifiers
            else str(raw).strip()
        )
        if not item:
            raise RegistryError(f"{field_name} contains an empty value")
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


@dataclass(frozen=True)
class ToolSpec:
    """Trusted semantics for one canonical tool capability.

    ``argument_roles`` maps argument names to semantic roles.  The graph
    builder recognizes roles such as ``data``, ``destination``, ``resource``,
    ``command`` and ``identifier``; unknown role strings remain readable
    metadata but do not silently acquire authority.
    """

    name: str
    operation: str
    required_scopes: Tuple[str, ...] = ()
    approval_required: bool = False
    side_effects: Tuple[str, ...] = ()
    argument_roles: Mapping[str, str] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    known: bool = True

    def __post_init__(self) -> None:
        name = _identifier(self.name, field_name="tool name")
        operation = str(self.operation).strip().lower()
        if operation not in OPERATIONS:
            raise RegistryError(f"unknown operation in tool registry: {operation!r}")
        if self.known and operation == "unknown":
            raise RegistryError("registered tools may not use the fail-closed 'unknown' operation")
        scopes = _string_tuple(
            self.required_scopes,
            field_name="required_scopes",
            limit=MAX_REQUIRED_SCOPES,
        )
        effects = _string_tuple(
            self.side_effects,
            field_name="side_effects",
            limit=MAX_ARGUMENT_ROLES,
        )
        aliases = _string_tuple(
            self.aliases,
            field_name="aliases",
            limit=MAX_TOOL_ALIASES,
        )
        if not isinstance(self.argument_roles, Mapping):
            raise RegistryError("argument_roles must be an object mapping names to roles")
        if len(self.argument_roles) > MAX_ARGUMENT_ROLES:
            raise RegistryError(f"argument_roles exceeds the limit of {MAX_ARGUMENT_ROLES}")
        roles: Dict[str, str] = {}
        for raw_name, raw_role in self.argument_roles.items():
            argument_name = _identifier(raw_name, field_name="argument name")
            role = _identifier(raw_role, field_name="argument role").lower()
            if role not in ALLOWED_ARGUMENT_ROLES:
                raise RegistryError(f"unsupported argument role: {role!r}")
            roles[argument_name] = role
        invalid_effects = set(effects).difference(ALLOWED_SIDE_EFFECTS)
        if invalid_effects:
            raise RegistryError(f"unsupported side effects: {sorted(invalid_effects)}")
        if operation == "write_external":
            if "external:send" not in scopes or not self.approval_required:
                raise RegistryError(
                    "write_external tools require external:send and verified approval"
                )
            if not any(role in {"destination", "recipient", "url", "endpoint", "external_destination"} for role in roles.values()):
                raise RegistryError("write_external tools require a destination argument")
        if operation == "execute" and (
            "code:execute" not in scopes or not self.approval_required
        ):
            raise RegistryError("execute tools require code:execute and verified approval")
        if operation == "delete" and (
            "data:delete" not in scopes or not self.approval_required
        ):
            raise RegistryError("delete tools require data:delete and verified approval")
        if operation == "permission_change" and (
            "permission:admin" not in scopes or not self.approval_required
        ):
            raise RegistryError(
                "permission_change tools require permission:admin and verified approval"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "required_scopes", scopes)
        object.__setattr__(self, "side_effects", effects)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "argument_roles", MappingProxyType(dict(sorted(roles.items()))))
        if type(self.approval_required) is not bool:
            raise RegistryError("approval_required must be a JSON boolean")
        if type(self.known) is not bool:
            raise RegistryError("known must be a boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolSpec":
        allowed = {
            "name",
            "operation",
            "required_scopes",
            "approval_required",
            "side_effects",
            "argument_roles",
            "aliases",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise RegistryError(f"unknown ToolSpec fields: {sorted(unknown)}")
        if "name" not in value or "operation" not in value:
            raise RegistryError("ToolSpec requires name and operation")
        return cls(
            name=value["name"],
            operation=value["operation"],
            required_scopes=tuple(value.get("required_scopes", ())),
            approval_required=value.get("approval_required", False),
            side_effects=tuple(value.get("side_effects", ())),
            argument_roles=value.get("argument_roles", {}),
            aliases=tuple(value.get("aliases", ())),
        )

    @classmethod
    def fail_closed(cls, requested_name: Any) -> "ToolSpec":
        """Create a non-registrable sentinel used only for model preparation."""

        raw = str(requested_name or "missing-tool").strip()
        safe = re.sub(r"[^A-Za-z0-9_.:/@-]", "_", raw)[:MAX_IDENTIFIER_LENGTH]
        if not safe:
            safe = "missing-tool"
        return cls(
            name=safe,
            operation="unknown",
            required_scopes=("capability:registered",),
            approval_required=True,
            side_effects=("unknown_effect",),
            argument_roles={},
            aliases=(),
            known=False,
        )

    def role_for(self, argument_name: str) -> Optional[str]:
        return self.argument_roles.get(argument_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "required_scopes": list(self.required_scopes),
            "approval_required": self.approval_required,
            "side_effects": list(self.side_effects),
            "argument_roles": dict(self.argument_roles),
            "aliases": list(self.aliases),
        }


class ToolRegistry:
    """Immutable-by-resolution registry with explicit unknown-tool failure."""

    def __init__(self, specs: Iterable[ToolSpec] = (), *, version: str = "unversioned") -> None:
        self.version = str(version).strip() or "unversioned"
        self._specs: Dict[str, ToolSpec] = {}
        self._lookup: Dict[str, str] = {}
        for spec in specs:
            self.register(spec)

    @staticmethod
    def _lookup_key(name: Any) -> str:
        return str(name or "").strip().casefold()

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise RegistryError("registry entries must be ToolSpec instances")
        if not spec.known:
            raise RegistryError("fail-closed sentinel specs cannot be registered")
        if len(self._specs) >= MAX_REGISTERED_TOOLS:
            raise RegistryError(f"registry exceeds the limit of {MAX_REGISTERED_TOOLS} tools")
        canonical_key = self._lookup_key(spec.name)
        names = (spec.name,) + spec.aliases
        for name in names:
            key = self._lookup_key(name)
            owner = self._lookup.get(key)
            if owner is not None and owner != canonical_key:
                raise RegistryError(f"duplicate tool name or alias: {name!r}")
        if canonical_key in self._specs:
            raise RegistryError(f"duplicate canonical tool: {spec.name!r}")
        self._specs[canonical_key] = spec
        for name in names:
            self._lookup[self._lookup_key(name)] = canonical_key

    def resolve(self, tool_name: Any) -> ToolSpec:
        key = self._lookup_key(tool_name)
        canonical = self._lookup.get(key)
        if canonical is None:
            raise UnknownToolError(str(tool_name or ""))
        return self._specs[canonical]

    def resolve_or_fail_closed(self, tool_name: Any) -> ToolSpec:
        try:
            return self.resolve(tool_name)
        except UnknownToolError:
            return ToolSpec.fail_closed(tool_name)

    def __contains__(self, tool_name: object) -> bool:
        return self._lookup_key(tool_name) in self._lookup

    def __len__(self) -> int:
        return len(self._specs)

    def specs(self) -> Tuple[ToolSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))

    @classmethod
    def from_dict(cls, value: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]) -> "ToolRegistry":
        if isinstance(value, Mapping):
            unknown = set(value).difference({"version", "tools"})
            if unknown:
                raise RegistryError(f"unknown ToolRegistry fields: {sorted(unknown)}")
            raw_specs = value.get("tools", ())
            version = str(value.get("version", "unversioned"))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_specs = value
            version = "unversioned"
        else:
            raise RegistryError("registry JSON must be an object or a list of tools")
        if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
            raise RegistryError("registry tools must be a list")
        return cls((ToolSpec.from_dict(item) for item in raw_specs), version=version)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ToolRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(value)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "tools": [spec.to_dict() for spec in self.specs()]}


__all__ = [
    "ALLOWED_ARGUMENT_ROLES",
    "ALLOWED_SIDE_EFFECTS",
    "MAX_ARGUMENT_ROLES",
    "MAX_REGISTERED_TOOLS",
    "MAX_REQUIRED_SCOPES",
    "RegistryError",
    "ToolRegistry",
    "ToolSpec",
    "UnknownToolError",
]
