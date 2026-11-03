"""Loaders that turn public IDS datasets into labelled Flow objects.

CIC-IDS2017 ships pre-computed flow CSVs (CICFlowMeter). We map the handful of
columns we need onto our Flow model so the same detectors run unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterator, Optional

from ..features.flows import Flow

# CICFlowMeter column names (with their inconsistent leading spaces stripped).
_COLS = {
    "src_ip": "Source IP",
    "dst_ip": "Destination IP",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "proto": "Protocol",
    "duration": "Flow Duration",  # microseconds
    "fwd_pkts": "Total Fwd Packets",
    "bwd_pkts": "Total Backward Packets",
    "fwd_bytes": "Total Length of Fwd Packets",
    "bwd_bytes": "Total Length of Bwd Packets",
    "syn": "SYN Flag Count",
    "label": "Label",
}
_PROTO = {"6": "tcp", "17": "udp", "1": "icmp"}

# Which of our features *should* explain each attack family (used for RQ2).
EXPECTED_FEATURE = {
    "PortScan": "syn_ratio",
    "DoS": "pkts_per_sec",
    "DDoS": "pkts_per_sec",
    "Bot": "duration",
    "Infiltration": "bytes",
    "Brute Force": "packets",
}


@dataclass
class LabelledFlow:
    flow: Flow
    label: str  # "BENIGN" or an attack name

    @property
    def is_attack(self) -> bool:
        return self.label.upper() != "BENIGN"

    @property
    def family(self) -> str:
        for key in EXPECTED_FEATURE:
            if key.lower() in self.label.lower():
                return key
        return "Other" if self.is_attack else "BENIGN"


def load_cicids2017_csv(path: str, limit: Optional[int] = None) -> Iterator[LabelledFlow]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [f.strip() for f in reader.fieldnames or []]
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            try:
                yield _row_to_flow(row)
            except (KeyError, ValueError):
                continue  # malformed row; CIC CSVs contain a few


def _row_to_flow(row: dict) -> LabelledFlow:
    g = lambda k: row[_COLS[k]].strip()  # noqa: E731
    pkts = int(float(g("fwd_pkts"))) + int(float(g("bwd_pkts")))
    if pkts <= 0:
        raise ValueError("empty flow")
    duration = max(float(g("duration")) / 1e6, 1e-6)
    flow = Flow(
        key=(g("src_ip"), g("dst_ip"), int(float(g("src_port"))), int(float(g("dst_port"))), _PROTO.get(g("proto"), "other")),
        first_ts=0.0,
        last_ts=duration,
        packets=pkts,
        bytes=int(float(g("fwd_bytes"))) + int(float(g("bwd_bytes"))),
        syn_count=min(int(float(g("syn"))), pkts),
        lengths=[],
    )
    return LabelledFlow(flow=flow, label=g("label"))
