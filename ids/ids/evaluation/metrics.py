"""Flow-level and attribution metrics for the paper's result tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from ..alerting.alert import Alert
from ..detectors.base import Detector
from .datasets import EXPECTED_FEATURE, LabelledFlow


@dataclass
class ClassificationReport:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    attribution_hits: int = 0
    attribution_total: int = 0
    per_family: Dict[str, "ClassificationReport"] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def attribution_accuracy(self) -> float:
        return self.attribution_hits / self.attribution_total if self.attribution_total else 0.0

    def row(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "attribution_acc": self.attribution_accuracy,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def _attributed_feature(alerts: Sequence[Alert]) -> str:
    for a in alerts:
        if a.rule_id.startswith("zscore:"):
            return a.rule_id.split(":", 1)[1]
    return ""


def evaluate(detectors: Iterable[Detector], flows: Iterable[LabelledFlow]) -> ClassificationReport:
    """Flow-level evaluation: a flow is 'flagged' if any detector raises an alert on it."""
    detectors = list(detectors)
    report = ClassificationReport()
    fam: Dict[str, ClassificationReport] = defaultdict(ClassificationReport)

    for lf in flows:
        alerts: List[Alert] = []
        for d in detectors:
            alerts.extend(d.inspect(lf.flow))
        flagged = bool(alerts)
        targets = [report] if not lf.is_attack else [report, fam[lf.family]]
        for r in targets:
            if lf.is_attack and flagged:
                r.tp += 1
                expected = EXPECTED_FEATURE.get(lf.family)
                attributed = _attributed_feature(alerts)
                if expected and attributed:
                    r.attribution_total += 1
                    r.attribution_hits += int(attributed == expected)
            elif lf.is_attack:
                r.fn += 1
            elif flagged:
                r.fp += 1
            else:
                r.tn += 1
    report.per_family = dict(fam)
    return report
