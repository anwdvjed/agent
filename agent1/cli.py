"""Unified reproducible command line; no implicit dataset download or tool execution."""
import argparse
import json
from dataclasses import asdict
from .risk_window import WindowConfig
from pathlib import Path

import torch

from .calibration import fit_policy_calibration, validate_artifact
from .checkpoint import canonical_digest, file_sha256
from .dataset import prepare_dataset
from .engine import train, evaluate
from .environments import generate_demo, environment_factory, environment_identity, source_replay_evaluator
from .evidence import explain_groups
from .io import read_json, read_jsonl, write_json, write_jsonl
from .metrics import outcome_report
from .registry import ToolRegistry
from .replay import replay_split
from .runtime import Ensemble, candidate_requests


def add_models(parser):
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--device", default="cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("describe", help="show the local model architecture and parameter count")
    p.add_argument("--model-config", type=Path)
    p = sub.add_parser("demo", help="generate clearly marked synthetic tool-state episodes and paired outcomes")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=48)
    p.add_argument("--formation", action="store_true", help="generate readiness-before-harm and readiness-then-sanitization trajectories")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--registry", type=Path, default=Path(__file__).parent / "assets" / "tool_registry.json")
    p = sub.add_parser("import-legacy", help="import original corpora as weak labels; human timing annotations still required")
    p.add_argument("--source", choices=["assebench-safety", "assebench-security", "linuxarena", "atbench", "openagentsafety"], required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=101)
    p = sub.add_parser("prepare")
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history-limit", type=int, default=128)
    p.add_argument("--seed", type=int, default=101)
    p = sub.add_parser("train")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--model-config", type=Path)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", default="cpu")
    for command in ("validate", "test"):
        p = sub.add_parser(command, help="offline probability/temporal evaluation; never tunes test thresholds")
        add_models(p)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--batch-size", type=int, default=16)
    for command in ("calibrate", "replay-test"):
        p = sub.add_parser(command, help="complete state-changing environment episodes")
        add_models(p)
        p.add_argument("--scenarios", type=Path, required=True)
        p.add_argument("--window-config", type=Path, help="frozen observational window settings")
        p.add_argument("--adapter", default="builtin", help="builtin or trusted_module:factory")
        p.add_argument("--output", type=Path, required=True)
        if command == "calibrate":
            p.add_argument("--policies", type=Path, required=True)
        else:
            p.add_argument("--calibration", type=Path, required=True)
    for command in ("predict", "explain"):
        p = sub.add_parser(command)
        add_models(p)
        p.add_argument("--request", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if command == "predict":
            p.add_argument("--references", type=Path, help="optional named same-state alternative requests for risk-difference forecasting")
        if command == "explain":
            p.add_argument("--budget", type=int, default=12)
            p.add_argument("--scenario", type=Path, help="optional matching builtin scenario for regenerated-action outcome verification")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "describe":
        from .model import Agent1Model, Agent1Config
        cfg=Agent1Config.from_dict(read_json(args.model_config)) if args.model_config else Agent1Config()
        model=Agent1Model(cfg)
        result={"project":"agent-astra","version":"0.4.0","model_config":cfg.to_dict(),
                "parameters":sum(p.numel() for p in model.parameters()),
                "scope":"local PyTorch research architecture; no OpenAI proprietary weights or internal architecture"}
    elif args.command == "demo":
        result = generate_demo(args.output, ToolRegistry.from_json(args.registry), args.groups, args.seed, formation=args.formation)
    elif args.command == "import-legacy":
        from .importers import import_legacy
        result = import_legacy(args.source, args.input, args.output, args.seed)
    elif args.command == "prepare":
        result = prepare_dataset(args.episodes, args.registry, args.output, annotations_path=args.annotations,
                                 horizon=args.horizon, history_limit=args.history_limit, seed=args.seed)
        result = {"manifest_digest": result["manifest_digest"], "files": result["files"], "excluded": result["excluded"]}
    elif args.command == "train":
        result = train(args.data, args.output, seed=args.seed, config=read_json(args.config) if args.config else None,
                       model_config=read_json(args.model_config) if args.model_config else None, device=args.device)
        result = {k: v for k, v in result.items() if k != "history"}
    else:
        ensemble = Ensemble(args.checkpoint, ToolRegistry.from_json(args.registry), args.device)
        if args.command in {"validate", "test"}:
            result = evaluate(ensemble, args.data, "validation" if args.command == "validate" else "test", args.output, args.batch_size)
        elif args.command in {"calibrate", "replay-test"}:
            if not ensemble.independent:
                raise ValueError("policy calibration/test requires at least two independently identified model states")
            environment = environment_identity(args.adapter)
            lineage = ensemble.lineage(environment)
            window_config = WindowConfig.from_dict(read_json(args.window_config)) if args.window_config else WindowConfig()
            lineage["window_config_digest"] = canonical_digest(asdict(window_config))
            if args.command == "calibrate":
                settings = read_json(args.policies)
                rows = replay_split(args.scenarios, ensemble, settings["policies"], "calibration", args.adapter, window_config=window_config)
                lineage.update(calibration_groups=sorted({r["group_id"] for r in rows}),
                               scenarios_sha256=file_sha256(args.scenarios), policies_digest=canonical_digest(settings),
                               synthetic=any(r.get("synthetic", False) for r in rows),
                               development_groups=sorted(ensemble.development_groups),
                               policy_family_digest=canonical_digest(settings["policies"]),
                               calibration_manifest_digest=canonical_digest(sorted({(r["episode_id"],r["group_id"]) for r in rows})),
                               repair_provider_version=environment,executor_contract_version=environment)
                result = fit_policy_calibration(rows, settings["policies"], lineage,
                                                risk_budget=settings["risk_budget"], interruption_budget=settings["interruption_budget"], delta=settings["delta"])
            else:
                source = read_json(args.calibration)
                lineage.update(repair_provider_version=environment, executor_contract_version=environment,
                               policy_family_digest=canonical_digest([r["policy"] for r in source["evaluations"]]),
                               calibration_manifest_digest=source["lineage"]["calibration_manifest_digest"])
                artifact = validate_artifact(source, lineage)
                scenarios = [s for s in read_jsonl(args.scenarios) if s["split"] == "test"]
                if any(s["group_id"] in artifact["lineage"]["calibration_groups"] for s in scenarios):
                    raise ValueError("test group overlaps calibration")
                rows = replay_split(args.scenarios, ensemble, [artifact["selected_policy"]], "test", args.adapter, window_config=window_config)
                result = {"policy_status": artifact["status"], "calibration_digest": artifact["digest"],
                          "metrics": outcome_report(rows), "fallback": "HOLD without execution" if artifact["selected_policy"] is None else None}
            write_json(args.output, result)
            write_jsonl(args.output.with_suffix(".trials.jsonl"), rows)
        elif args.command == "predict":
            candidates = candidate_requests(read_json(args.request), read_json(args.references) if args.references else None)
            result = {"predictions": ensemble.predict(candidates),
                      "scope": "research forecast only; not execution authorization"}
            write_json(args.output, result)
        else:
            verifier = source_replay_evaluator(read_json(args.scenario), ensemble) if args.scenario else None
            result = explain_groups(read_json(args.request), ensemble, budget=args.budget, replay_evaluator=verifier)
            write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
