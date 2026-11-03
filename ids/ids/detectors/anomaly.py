"""Baseline anomaly detector: per-feature z-score against a fitted benign baseline.

Dependency-light on purpose. A scikit-learn Isolation Forest variant can be
swapped in behind the same `fit` / `inspect` interface (see docs/DESIGN.md).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ..alerting.alert import Alert
from ..features.flows import FEATURE_NAMES, Flow
from .base import Detector


class AnomalyDetector(Detector):
    name = "anomaly"

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, flows: Iterable[Flow]) -> "AnomalyDetector":
        x = np.log1p(np.array([f.vector() for f in flows], dtype=float))
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        return self

    def score(self, flow: Flow) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("AnomalyDetector must be fitted before use")
        x = np.log1p(np.array(flow.vector(), dtype=float))
        return np.abs((x - self.mean) / self.std)

    def inspect(self, flow: Flow) -> List[Alert]:
        z = self.score(flow)
        worst = int(z.argmax())
        if z[worst] < self.threshold:
            return []
        return [
            Alert(
                detector=self.name,
                rule_id=f"zscore:{FEATURE_NAMES[worst]}",
                severity="high" if z[worst] > 2 * self.threshold else "medium",
                message=f"{FEATURE_NAMES[worst]} deviates {z[worst]:.1f} sigma from baseline",
                flow_key=flow.key,
                ts=flow.last_ts,
                score=float(z[worst]),
            )
        ]
