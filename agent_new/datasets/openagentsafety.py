"""Conservative best-effort OpenAgentSafety trajectory adapter.

The legacy repository has multiple result layouts. Only records with an
explicit binary label and a message-like trajectory are accepted; everything
else is skipped and reported by the preparation manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

from .atbench import _actions, _content
from .common import (
    NormalizedTrajectory,
    bounded_text,
    normalized_conversation_group,
    risk_type_from_text,
    stable_digest,
)


def _records(path: Path) -> Iterator[Mapping[str, Any]]:
    for file in sorted(path.rglob("*.json")):
        if ".git" in file.parts:
            continue
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item
        elif isinstance(value, Mapping):
            candidates = value.get("records", value.get("data"))
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, Mapping):
                        yield item
            else:
                yield value


def iter_openagentsafety(path: Path) -> Iterator[NormalizedTrajectory]:
    for index, record in enumerate(_records(path)):
        if "is_safe" in record:
            unsafe = not bool(record["is_safe"])
        elif record.get("label") in (0, 1, False, True):
            unsafe = bool(int(record["label"]))
        else:
            continue
        messages = _content(record)
        if not messages:
            continue
        events: List[Dict[str, Any]] = []
        goals: List[str] = []
        observation = {
            "source_id": "oas-initial",
            "source_type": "initial_request",
            "trust_level": "trusted",
            "content": "",
            "data_ids": [],
            "instruction_like": False,
            "contains_sensitive": False,
        }
        for position, message in enumerate(messages):
            role = str(message.get("role", message.get("type", ""))).casefold()
            content = bounded_text(message.get("content", message.get("text", "")), 800)
            if role in {"user", "human"}:
                goals.append(content)
                observation = {
                    "source_id": f"oas-user-{position}",
                    "source_type": "user_request",
                    "trust_level": "trusted",
                    "content": content,
                    "data_ids": [],
                    "instruction_like": True,
                    "contains_sensitive": False,
                }
            elif role in {"environment", "tool", "function", "tool_result"}:
                observation = {
                    "source_id": f"oas-environment-{position}",
                    "source_type": "tool_result",
                    "trust_level": "untrusted",
                    "content": content,
                    "data_ids": [],
                    "instruction_like": False,
                    "contains_sensitive": False,
                }
            for tool, arguments in _actions(message):
                events.append(
                    {
                        "event_id": f"oas-{index}-{len(events)}",
                        "timestamp": float(len(events) + 1),
                        "observation": dict(observation),
                        "action": {"tool": tool, "arguments": arguments},
                    }
                )
        if not events:
            continue
        risk_text = " ".join(
            str(record.get(key, ""))
            for key in ("risk_type", "risk_source", "failure_mode", "harm_type", "reason")
        )
        record_id = str(
            record.get("trajectory_id")
            or record.get("conv_id")
            or record.get("id")
            or stable_digest(record)
        )
        yield NormalizedTrajectory(
            source_dataset="openagentsafety",
            source_family="openagentsafety",
            record_id=record_id,
            group_id="openagentsafety:" + record_id,
            trusted_goal="\n".join(goals[-3:]) or "Complete the task safely.",
            events=events,
            unsafe=unsafe,
            event_type=risk_type_from_text(risk_text),
            risk_text=risk_text,
            label_quality="external_trajectory_label_with_weak_terminal_proxy",
            concept_supervision=0.0,
            metadata={"conversation_fingerprint": normalized_conversation_group(messages)},
        )


__all__ = ["iter_openagentsafety"]

