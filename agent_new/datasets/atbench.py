"""ATBench frozen-test adapter for trajectory-level diagnostic evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

from .common import (
    NormalizedTrajectory,
    bounded_text,
    canonical_tool_name,
    normalized_conversation_group,
    parse_arguments,
    risk_type_from_text,
)


def _actions(message: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    result: List[Tuple[str, Dict[str, Any]]] = []
    if message.get("action") is not None:
        action = message["action"]
        if isinstance(action, Mapping):
            tool = action.get("tool") or action.get("name") or action.get("function")
            result.append((canonical_tool_name(tool), parse_arguments(action.get("arguments", action.get("args", {})))))
        else:
            result.append(("agent_action", parse_arguments(action)))
    for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function", call)
        if isinstance(function, Mapping):
            result.append(
                (
                    canonical_tool_name(function.get("name") or "tool_call"),
                    parse_arguments(function.get("arguments", {})),
                )
            )
    return result


def _content(record: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    for key in ("content", "conversation", "messages", "trajectory"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def iter_atbench(path: Path) -> Iterator[NormalizedTrajectory]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, Mapping):
        records = records.get("data", records.get("records", records.get("test", [])))
    if not isinstance(records, list):
        raise ValueError("ATBench test file must contain a list of trajectories")
    for index, record in enumerate(records):
        messages = _content(record)
        events: List[Dict[str, Any]] = []
        goals: List[str] = []
        observation = {
            "source_id": "atbench-initial",
            "source_type": "initial_request",
            "trust_level": "trusted",
            "content": "",
            "data_ids": [],
            "instruction_like": False,
            "contains_sensitive": False,
        }
        for position, message in enumerate(messages):
            role = str(message.get("role", message.get("type", ""))).casefold()
            content = bounded_text(message.get("content", message.get("text", "")), 1000)
            if role in {"user", "human"}:
                goals.append(content)
                observation = {
                    "source_id": f"atbench-user-{position}",
                    "source_type": "user_request",
                    "trust_level": "trusted",
                    "content": content,
                    "data_ids": [],
                    "instruction_like": True,
                    "contains_sensitive": False,
                }
            elif role in {"environment", "tool", "function", "tool_result"}:
                observation = {
                    "source_id": f"atbench-environment-{position}",
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
                        "event_id": f"atbench-{index}-{len(events)}",
                        "timestamp": float(len(events) + 1),
                        "observation": dict(observation),
                        "action": {"tool": tool, "arguments": arguments},
                    }
                )
        if not events:
            continue
        label = record.get("label")
        if label is None and "is_safe" in record:
            label = 0 if bool(record["is_safe"]) else 1
        if label not in (0, 1, False, True):
            raise ValueError(f"ATBench row {index} has no binary safety label")
        unsafe = bool(int(label))
        risk_source = str(record.get("risk_source", ""))
        failure_mode = str(record.get("failure_mode", ""))
        harm_type = str(record.get("harm_type", record.get("real_world_harm", "")))
        conv_id = str(record.get("conv_id", record.get("id", index)))
        yield NormalizedTrajectory(
            source_dataset="atbench",
            source_family="atbench",
            record_id=conv_id,
            group_id="atbench:" + conv_id,
            trusted_goal="\n".join(goals[-3:]) or "Complete the benchmark task safely.",
            events=events,
            unsafe=unsafe,
            event_type=risk_type_from_text(" ".join((risk_source, failure_mode, harm_type))),
            risk_text=" ".join((risk_source, failure_mode, harm_type)),
            label_quality="frozen_trajectory_label_with_weak_terminal_proxy",
            concept_supervision=0.0,
            metadata={
                "risk_source": risk_source,
                "failure_mode": failure_mode,
                "harm_type": harm_type,
                "conversation_fingerprint": normalized_conversation_group(messages),
            },
        )


__all__ = ["iter_atbench"]

