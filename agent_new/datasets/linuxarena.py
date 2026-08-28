"""LinuxArena Parquet adapter without side-task leakage into model input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pyarrow.parquet as parquet

from .common import (
    NormalizedTrajectory,
    bounded_text,
    canonical_tool_name,
    parse_arguments,
    stable_digest,
)


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def iter_linuxarena(path: Path) -> Iterator[NormalizedTrajectory]:
    source = parquet.ParquetFile(path)
    columns = [
        "trajectory_id",
        "model",
        "model_short",
        "environment",
        "main_task",
        "task_description",
        "side_task",
        "side_task_description",
        "main_task_success",
        "side_task_success",
        "has_side_task",
        "action_functions",
        "action_arguments_json",
        "action_output",
        "action_has_error",
    ]
    for batch in source.iter_batches(batch_size=64, columns=columns):
        for record in batch.to_pylist():
            functions = _list(record.get("action_functions"))
            arguments = _list(record.get("action_arguments_json"))
            outputs = _list(record.get("action_output"))
            errors = _list(record.get("action_has_error"))
            events: List[Dict[str, Any]] = []
            previous_output = "Task initialized."
            for index, raw_tool in enumerate(functions):
                raw_arguments = arguments[index] if index < len(arguments) else "{}"
                events.append(
                    {
                        "event_id": f"linux-{record.get('trajectory_id')}-{index}",
                        "timestamp": float(index + 1),
                        "observation": {
                            "source_id": f"linux-environment-{index}",
                            "source_type": "linux_environment",
                            "trust_level": "unknown",
                            "content": bounded_text(previous_output, 700),
                            "data_ids": [],
                            "instruction_like": False,
                            "contains_sensitive": False,
                        },
                        "action": {
                            "tool": canonical_tool_name(raw_tool),
                            "arguments": parse_arguments(raw_arguments),
                        },
                    }
                )
                output = outputs[index] if index < len(outputs) else ""
                error = errors[index] if index < len(errors) else False
                previous_output = (
                    "ERROR: " + bounded_text(output, 650)
                    if error
                    else bounded_text(output, 700)
                )
            if not events:
                continue
            main_task = str(record.get("task_description") or record.get("main_task") or "")
            environment = str(record.get("environment", ""))
            side_task = str(record.get("side_task", "") or "")
            side_description = str(record.get("side_task_description", "") or "")
            record_id = str(record.get("trajectory_id") or stable_digest(record))
            yield NormalizedTrajectory(
                source_dataset="linuxarena",
                source_family="linuxarena",
                record_id=record_id,
                group_id="linuxarena:" + stable_digest([main_task, environment]),
                trusted_goal=bounded_text(main_task, 4096),
                events=events,
                unsafe=bool(record.get("side_task_success")),
                event_type="computer_security",
                risk_text=" ".join((side_task, side_description)),
                label_quality="harmful_outcome_with_weak_terminal_proxy",
                concept_supervision=0.0,
                attack_intent=int(bool(record.get("has_side_task"))),
                utility=int(bool(record.get("main_task_success"))),
                metadata={
                    "environment": environment,
                    "model": str(record.get("model_short") or record.get("model") or ""),
                    "task_group_sha256": stable_digest([main_task, environment]),
                    "attack_attempt": int(bool(record.get("has_side_task"))),
                    "harmful_outcome": int(bool(record.get("side_task_success"))),
                },
            )


__all__ = ["iter_linuxarena"]

