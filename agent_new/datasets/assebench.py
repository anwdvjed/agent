"""ASSEBench AgentJudge safety/security adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from .common import (
    NormalizedTrajectory,
    bounded_text,
    canonical_tool_name,
    normalized_conversation_group,
    parse_arguments,
    risk_type_from_text,
)


def _sessions(value: Any) -> List[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, dict) for item in value):
        return [value]
    return [session for session in value if isinstance(session, list)]


def _action(value: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(value, Mapping):
        tool = value.get("tool") or value.get("name") or value.get("function") or "agent_action"
        arguments = value.get("arguments", value.get("args", {}))
        return canonical_tool_name(tool), parse_arguments(arguments)
    text = str(value or "").strip()
    match = re.match(r"(?:tool|action)?\s*:?[ ]*([A-Za-z_][A-Za-z0-9_.:/@-]*)", text)
    tool = match.group(1) if match else "agent_action"
    return canonical_tool_name(tool), parse_arguments(text)


def iter_assebench(path: Path, subset: str) -> Iterator[NormalizedTrajectory]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"ASSEBench file must contain a list: {path}")
    for position, record in enumerate(records):
        sessions = _sessions(record.get("contents"))
        events: List[Dict[str, Any]] = []
        goals: List[str] = []
        observation = {
            "source_id": "initial-request",
            "source_type": "initial_request",
            "trust_level": "trusted",
            "content": "",
            "data_ids": [],
            "instruction_like": False,
            "contains_sensitive": False,
        }
        normalized_messages: List[Any] = []
        for session in sessions:
            for message in session:
                role = str(message.get("role", "")).casefold()
                normalized_messages.append(
                    {key: message.get(key) for key in ("role", "content", "action") if key in message}
                )
                if role == "user":
                    content = bounded_text(message.get("content"), 1200)
                    if content:
                        goals.append(content)
                    observation = {
                        "source_id": f"user-turn-{len(normalized_messages)}",
                        "source_type": "user_request",
                        "trust_level": "trusted",
                        "content": content,
                        "data_ids": [],
                        "instruction_like": True,
                        "contains_sensitive": False,
                    }
                elif role == "environment":
                    content = bounded_text(message.get("content"), 1200)
                    observation = {
                        "source_id": f"environment-turn-{len(normalized_messages)}",
                        "source_type": "tool_result",
                        "trust_level": "untrusted",
                        "content": content,
                        "data_ids": [],
                        "instruction_like": False,
                        "contains_sensitive": any(
                            token in content.casefold()
                            for token in ("password", "secret", "credential", "private")
                        ),
                    }
                elif role == "agent" and message.get("action") is not None:
                    tool, arguments = _action(message.get("action"))
                    events.append(
                        {
                            "event_id": f"asse-{subset}-{position}-action-{len(events)}",
                            "timestamp": float(len(events) + 1),
                            "observation": dict(observation),
                            "action": {"tool": tool, "arguments": arguments},
                        }
                    )
        if not events:
            continue
        risk_type = str(record.get("risk_type", ""))
        failure_mode = str(record.get("failure_mode", ""))
        risk_description = str(record.get("risk_description", ""))
        unsafe = bool(int(record.get("label", 0)))
        record_id = f"{subset}:{record.get('id', position)}:{position}"
        yield NormalizedTrajectory(
            source_dataset=f"assebench_{subset}",
            source_family="assebench",
            record_id=record_id,
            group_id="assebench:" + normalized_conversation_group(normalized_messages),
            trusted_goal="\n".join(goals[-3:]) or "Complete the user request safely.",
            events=events,
            unsafe=unsafe,
            event_type=risk_type_from_text(risk_type),
            risk_text=" ".join((risk_type, failure_mode, risk_description)),
            label_quality="trajectory_label_with_weak_terminal_proxy",
            concept_supervision=0.25,
            metadata={
                "application_scenario": str(record.get("application_scenario", "")),
                "ambiguous": int(record.get("ambiguous", 0) or 0),
            },
        )


__all__ = ["iter_assebench"]

