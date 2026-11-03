from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Alert:
    detector: str
    rule_id: str
    severity: str
    message: str
    flow_key: tuple
    ts: float
    score: Optional[float] = None

    @property
    def dedupe_key(self) -> tuple:
        src_ip, dst_ip, _sport, _dport, proto = self.flow_key
        return (self.detector, self.rule_id, src_ip, dst_ip, proto)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flow_key"] = list(self.flow_key)
        return d
