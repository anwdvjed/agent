"""Experiment runner: python -m ids.evaluation.run --dataset cicids2017 --config configs/eval.json

Fits the anomaly baseline on benign flows from the training files, evaluates
every baseline on the test files, and writes results/*.csv and
paper/tables/results_main.tex so the paper regenerates from code.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import List

from ..detectors import AnomalyDetector, SignatureDetector
from .datasets import LabelledFlow, load_cicids2017_csv
from .metrics import evaluate
from .tables import write_csv, write_latex


def _load(patterns: List[str], root: str, limit) -> List[LabelledFlow]:
    out: List[LabelledFlow] = []
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            out.extend(load_cicids2017_csv(path, limit=limit))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--config", default="configs/eval.json")
    ap.add_argument("--limit", type=int, default=None, help="rows per file (smoke runs)")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = json.load(fh)
    here = os.path.dirname(os.path.abspath(args.config))
    root = os.path.join(here, "..")
    ds = cfg["datasets"][args.dataset]

    t0 = time.time()
    train = [lf for lf in _load(ds["train"], root, args.limit) if not lf.is_attack]
    test = _load(ds["test"], root, args.limit)
    if not train or not test:
        raise SystemExit(f"no data found under {os.path.join(root, 'data')}; see scripts/download_datasets.sh")
    print(f"loaded train={len(train)} benign flows, test={len(test)} flows in {time.time()-t0:.1f}s")

    sig = SignatureDetector.from_yaml(os.path.join(root, cfg["rules"]))
    systems = {}
    for theta in cfg["theta_sweep"]:
        anom = AnomalyDetector(threshold=theta).fit(lf.flow for lf in train)
        systems[f"anomaly-only (theta={theta})"] = evaluate([anom], test)
        systems[f"fused (theta={theta})"] = evaluate([sig, anom], test)
    systems = {"signature-only": evaluate([sig], test), **systems}

    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    os.makedirs(os.path.join(root, "paper", "tables"), exist_ok=True)
    write_csv(os.path.join(root, "results", f"{args.dataset}.csv"), systems)
    write_latex(
        os.path.join(root, "paper", "tables", "results_main.tex"), systems,
        caption=f"Flow-level results on {args.dataset}.", label="tab:main",
    )
    for name, rep in systems.items():
        r = rep.row()
        print(f"{name:28s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} FPR={r['fpr']:.4f} attr={r['attribution_acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
