"""Assemble packets into bidirectional flows and compute per-flow features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List

from ..capture.base import Packet

FEATURE_NAMES = [
    "duration",
    "packets",
    "bytes",
    "mean_len",
    "syn_ratio",
    "pkts_per_sec",
]


@dataclass
class Flow:
    key: tuple
    first_ts: float
    last_ts: float
    packets: int = 0
    bytes: int = 0
    syn_count: int = 0
    lengths: List[int] = field(default_factory=list)

    def add(self, pkt: Packet) -> None:
        self.last_ts = pkt.ts
        self.packets += 1
        self.bytes += pkt.length
        self.lengths.append(pkt.length)
        if "S" in pkt.flags and "A" not in pkt.flags:
            self.syn_count += 1

    @property
    def duration(self) -> float:
        return max(self.last_ts - self.first_ts, 1e-6)

    def features(self) -> Dict[str, float]:
        return {
            "duration": self.duration,
            "packets": float(self.packets),
            "bytes": float(self.bytes),
            "mean_len": self.bytes / self.packets,
            "syn_ratio": self.syn_count / self.packets,
            "pkts_per_sec": self.packets / self.duration,
        }

    def vector(self) -> List[float]:
        f = self.features()
        return [f[name] for name in FEATURE_NAMES]


class FlowAssembler:
    """Groups packets by 5-tuple and expires flows after `timeout` seconds of idleness."""

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._active: Dict[tuple, Flow] = {}

    def feed(self, packets: Iterable[Packet]) -> Iterator[Flow]:
        for pkt in packets:
            yield from self._expire(pkt.ts)
            flow = self._active.get(pkt.flow_key)
            if flow is None:
                flow = Flow(key=pkt.flow_key, first_ts=pkt.ts, last_ts=pkt.ts)
                self._active[pkt.flow_key] = flow
            flow.add(pkt)
        yield from self.flush()

    def _expire(self, now: float) -> Iterator[Flow]:
        stale = [k for k, f in self._active.items() if now - f.last_ts > self.timeout]
        for k in stale:
            yield self._active.pop(k)

    def flush(self) -> Iterator[Flow]:
        flows = list(self._active.values())
        self._active.clear()
        yield from flows
