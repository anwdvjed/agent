"""Command-line entry point: `ids run --source synthetic --config configs/default.json`."""

from __future__ import annotations

import argparse
import json
import os

from .alerting import AlertManager, JsonlSink, StdoutSink
from .capture import SyntheticSource
from .detectors import AnomalyDetector, SignatureDetector
from .features import FlowAssembler


def build_source(args):
    if args.source == "synthetic":
        return SyntheticSource()
    if args.source == "pcap":
        from .capture.pcap import PcapSource

        if not args.path:
            raise SystemExit("--path is required for --source pcap")
        return PcapSource(args.path)
    raise SystemExit(f"unknown source {args.source}")


def run(args) -> int:
    with open(args.config) as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(args.config))

    flows = list(FlowAssembler(timeout=cfg["flow_timeout"]).feed(build_source(args)))

    rules_path = os.path.join(base, "..", cfg["rules"])
    sig = SignatureDetector.from_yaml(rules_path)
    anomaly = AnomalyDetector(threshold=cfg["anomaly"]["threshold"]).fit(flows)

    sinks = [StdoutSink()]
    if cfg["alerts"].get("jsonl"):
        sinks.append(JsonlSink(cfg["alerts"]["jsonl"]))
    manager = AlertManager(sinks, min_severity=cfg["alerts"]["min_severity"])

    manager.handle(sig.inspect_many(flows))
    manager.handle(anomaly.inspect_many(flows))
    print(f"flows={len(flows)} alerts={len(manager.emitted)} suppressed={manager.suppressed}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ids")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run detectors over a packet source")
    p.add_argument("--source", choices=["synthetic", "pcap"], default="synthetic")
    p.add_argument("--path", help="pcap file path (for --source pcap)")
    p.add_argument("--config", default="configs/default.json")
    p.set_defaults(func=run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
