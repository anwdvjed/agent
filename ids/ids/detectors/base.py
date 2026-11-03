from __future__ import annotations

from typing import Iterable, List

from ..alerting.alert import Alert
from ..features.flows import Flow


class Detector:
    name = "base"

    def inspect(self, flow: Flow) -> List[Alert]:  # pragma: no cover - interface
        raise NotImplementedError

    def inspect_many(self, flows: Iterable[Flow]) -> List[Alert]:
        out: List[Alert] = []
        for flow in flows:
            out.extend(self.inspect(flow))
        return out
