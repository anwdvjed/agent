"""Collects alerts from detectors, de-duplicates them and forwards to sinks."""

from __future__ import annotations

import json
import sys
from typing import Dict, Iterable, List, Optional, TextIO

from .alert import SEVERITY_ORDER, Alert


class Sink:
    def emit(self, alert: Alert) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class StdoutSink(Sink):
    def __init__(self, stream: Optional[TextIO] = None):
        self.stream = stream

    def emit(self, alert: Alert) -> None:
        src, dst, _, dport, proto = alert.flow_key
        (self.stream or sys.stdout).write(
            f"[{alert.severity.upper():8}] {alert.detector}/{alert.rule_id} "
            f"{src} -> {dst}:{dport}/{proto}  {alert.message}\n"
        )


class JsonlSink(Sink):
    def __init__(self, path: str):
        self.path = path

    def emit(self, alert: Alert) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(alert.to_dict()) + "\n")


class AlertManager:
    def __init__(self, sinks: List[Sink], min_severity: str = "low", dedupe_window: float = 60.0):
        self.sinks = sinks
        self.min_severity = SEVERITY_ORDER[min_severity]
        self.dedupe_window = dedupe_window
        self._last_seen: Dict[tuple, float] = {}
        self.emitted: List[Alert] = []
        self.suppressed = 0

    def handle(self, alerts: Iterable[Alert]) -> None:
        for alert in alerts:
            if SEVERITY_ORDER[alert.severity] < self.min_severity:
                self.suppressed += 1
                continue
            last = self._last_seen.get(alert.dedupe_key)
            if last is not None and alert.ts - last < self.dedupe_window:
                self.suppressed += 1
                continue
            self._last_seen[alert.dedupe_key] = alert.ts
            self.emitted.append(alert)
            for sink in self.sinks:
                sink.emit(alert)
