"""Rule-based signature detector. Rules are loaded from YAML (see rules/)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..alerting.alert import Alert
from ..features.flows import Flow
from .base import Detector


@dataclass
class Rule:
    id: str
    description: str
    severity: str = "medium"
    dst_port: Optional[int] = None
    proto: Optional[str] = None
    src_ip: Optional[str] = None
    min_syn_ratio: Optional[float] = None
    min_pkts_per_sec: Optional[float] = None

    def matches(self, flow: Flow) -> bool:
        src_ip, _dst_ip, _sport, dport, proto = flow.key
        f = flow.features()
        if self.dst_port is not None and dport != self.dst_port:
            return False
        if self.proto is not None and proto != self.proto:
            return False
        if self.src_ip is not None and src_ip != self.src_ip:
            return False
        if self.min_syn_ratio is not None and f["syn_ratio"] < self.min_syn_ratio:
            return False
        if self.min_pkts_per_sec is not None and f["pkts_per_sec"] < self.min_pkts_per_sec:
            return False
        return True


class SignatureDetector(Detector):
    name = "signature"

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: str) -> "SignatureDetector":
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls([Rule(**r) for r in raw.get("rules", [])])

    def inspect(self, flow: Flow) -> List[Alert]:
        return [
            Alert(
                detector=self.name,
                rule_id=rule.id,
                severity=rule.severity,
                message=rule.description,
                flow_key=flow.key,
                ts=flow.last_ts,
            )
            for rule in self.rules
            if rule.matches(flow)
        ]
