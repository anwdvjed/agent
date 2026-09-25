"""Legacy corpus import with weak labels explicitly quarantined for annotation."""
import copy
from pathlib import Path

from .checkpoint import canonical_digest
from .dataset import assigned_split
from .io import write_json, write_jsonl
from .legacy_import.common import build_tool_registry, _request_event


def import_legacy(source, input_path, output, seed=101):
    if source.startswith("assebench"):
        from .legacy_import.assebench import iter_assebench
        trajectories = list(iter_assebench(Path(input_path), source.rsplit("-", 1)[-1]))
    elif source == "linuxarena":
        from .legacy_import.linuxarena import iter_linuxarena
        trajectories = list(iter_linuxarena(Path(input_path)))
    elif source == "atbench":
        from .legacy_import.atbench import iter_atbench
        trajectories = list(iter_atbench(Path(input_path)))
    elif source == "openagentsafety":
        from .legacy_import.openagentsafety import iter_openagentsafety
        trajectories = list(iter_openagentsafety(Path(input_path)))
    else:
        raise ValueError("unknown legacy source")
    if not trajectories:
        raise ValueError("source contains no usable trajectories")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("import output must be empty")
    registry = build_tool_registry(trajectories, version=f"agent1-inferred-{source}-v1")
    rows, templates = [], []
    for trajectory in trajectories:
        identifier = canonical_digest([source, trajectory.record_id])[:24]
        events = []
        for step, event in enumerate(trajectory.events):
            event = copy.deepcopy(_request_event(event))
            event["event_id"] = canonical_digest([identifier, step])[:24]
            event["observation"]["source_id"] = canonical_digest(event["observation"].get("source_id", "unknown"))[:24]
            events.append(event)
        group = canonical_digest(trajectory.group_id)
        rows.append({"episode_id": identifier, "group_id": group,
                     "split": "test" if source in {"atbench", "openagentsafety"} else assigned_split(group, seed),
                     "trusted_goal": trajectory.trusted_goal,
                     "policy": {"id": "dataset-policy", "clauses": [{"id": "task", "text": "Complete the trusted task without unsafe tool effects."}]},
                     "events": events, "annotation": {"kind": "weak", "trajectory_unsafe": trajectory.unsafe},
                     "source": source, "synthetic": False})
        templates.append({"episode_id": identifier, "annotation": {"kind": "weak", "verified": False,
                                                                   "provenance": "", "trajectory_unsafe": trajectory.unsafe}})
    write_jsonl(output / "episodes.jsonl", rows)
    write_jsonl(output / "annotation_template.jsonl", templates)
    write_json(output / "registry.json", registry.to_dict())
    report = {"episodes": len(rows), "source": source, "label_status": "weak_requires_action_time_annotation",
              "registry_status": "inferred_dataset_semantics_not_production_authorization"}
    write_json(output / "import_report.json", report)
    return report
