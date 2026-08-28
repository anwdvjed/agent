"""Command-line pre-execution assessment using trained Agent New checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .calibration import GroupConformalCalibration
from .policy import InterventionPolicy
from .registry import ToolRegistry
from .runtime import SafetyRuntime, load_ensemble
from .trust import HMACRequestVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--safe-risk-limit", type=float, default=0.20)
    parser.add_argument("--uncertainty-limit", type=float, default=0.25)
    parser.add_argument("--ensemble-z", type=float, default=1.96)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument(
        "--hmac-key-env",
        default=None,
        help="environment variable containing the trusted-context HMAC key",
    )
    parser.add_argument("--trusted-issuer", action="append", default=[])
    args = parser.parse_args()

    registry = ToolRegistry.from_json(args.registry)
    calibration = None
    if args.calibration is not None:
        calibration = GroupConformalCalibration.from_dict(
            json.loads(args.calibration.read_text(encoding="utf-8"))
        )
    device = torch.device(args.device)
    ensemble = load_ensemble(args.checkpoint, registry, device, calibration)
    verifier = None
    if args.hmac_key_env is not None:
        secret = os.environ.get(args.hmac_key_env)
        if not secret:
            raise ValueError("trusted-context HMAC environment variable is empty")
        if not args.trusted_issuer:
            raise ValueError("--trusted-issuer is required with --hmac-key-env")
        verifier = HMACRequestVerifier(
            secret=secret.encode("utf-8"),
            allowed_issuers=args.trusted_issuer,
        )
    runtime = SafetyRuntime(
        registry,
        ensemble,
        calibration=calibration,
        policy=InterventionPolicy(
            safe_risk_limit=args.safe_risk_limit,
            epistemic_uncertainty_limit=args.uncertainty_limit,
        ),
        ensemble_z=args.ensemble_z,
        device=device,
        trusted_context_verifier=verifier,
    )
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = runtime.assess(request, evidence_top_k=args.evidence_top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    exit_codes = {
        "ALLOW": 0,
        "CONFIRM": 20,
        "REPLAN": 21,
        "HOLD": 22,
        "BLOCK": 23,
    }
    raise SystemExit(exit_codes[result["decision"]["decision"]])


if __name__ == "__main__":
    main()
