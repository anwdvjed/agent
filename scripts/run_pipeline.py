#!/usr/bin/env python3
"""One-command engineering pipeline. Outputs are synthetic, not benchmark results."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" / "demo_run")
    p.add_argument("--groups", type=int, default=48)
    p.add_argument("--formation", action="store_true", help="exercise the new dual-target forecasting module")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("choose a new output directory")
    def run(*arguments):
        subprocess.run([sys.executable, "-m", "agent_astra", *map(str, arguments)], cwd=ROOT, check=True)
    run("demo", "--output", out / "raw", "--groups", args.groups, *(["--formation"] if args.formation else []))
    run("prepare", "--episodes", out / "raw/episodes.jsonl", "--registry", out / "raw/registry.json", "--output", out / "data")
    for seed in (101, 202):
        run("train", "--data", out / "data", "--output", out / f"seed-{seed}", "--seed", seed,
            "--config", ROOT / "configs/smoke_train.json", "--model-config", ROOT / "configs/smoke_model.json")
    model_args = ["--checkpoint", out / "seed-101/best", "--checkpoint", out / "seed-202/best", "--registry", out / "data/registry.json"]
    run("validate", *model_args, "--data", out / "data", "--output", out / "validation.json")
    adapter = "formation" if args.formation else "builtin"
    run("calibrate", *model_args, "--adapter", adapter, "--scenarios", out / "raw/scenarios.jsonl", "--policies", ROOT / "configs/policies.json", "--output", out / "calibration.json")
    run("test", *model_args, "--data", out / "data", "--output", out / "test.json")
    run("replay-test", *model_args, "--adapter", adapter, "--scenarios", out / "raw/scenarios.jsonl", "--calibration", out / "calibration.json", "--output", out / "replay_test.json")
    print(f"Completed synthetic engineering pipeline: {out}")


if __name__ == "__main__":
    main()
