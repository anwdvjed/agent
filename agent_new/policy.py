"""Cost-aware intervention policy driven by calibrated risk upper bounds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .constants import DECISIONS, INTERVENTION_COSTS, INTERVENTIONS


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("%s must be finite and in [0, 1]" % name)
    return result


def _cost(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return result


@dataclass(frozen=True)
class InterventionAssessment:
    """Risk evidence for one candidate intervention."""

    intervention: str
    risk_upper_bound: float
    actual_risk: float
    epistemic_uncertainty: float
    cost: Optional[float] = None

    def __post_init__(self) -> None:
        if self.intervention not in INTERVENTIONS:
            raise ValueError("unknown intervention: %s" % self.intervention)
        _probability(self.risk_upper_bound, "risk_upper_bound")
        _probability(self.actual_risk, "actual_risk")
        _probability(self.epistemic_uncertainty, "epistemic_uncertainty")
        if self.cost is not None:
            _cost(self.cost, "cost")

    @property
    def resolved_cost(self) -> float:
        value = INTERVENTION_COSTS[self.intervention] if self.cost is None else self.cost
        return _cost(value, "cost")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["cost"] = self.resolved_cost
        return value


@dataclass(frozen=True)
class PolicyDecision:
    decision: str
    intervention: Optional[str]
    reason: str
    risk_upper_bound: Optional[float]
    actual_risk: Optional[float]
    epistemic_uncertainty: Optional[float]
    intervention_cost: Optional[float]

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError("unknown decision: %s" % self.decision)
        if self.intervention is not None and self.intervention not in INTERVENTIONS:
            raise ValueError("unknown intervention: %s" % self.intervention)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AssessmentInput = Union[
    InterventionAssessment,
    Mapping[str, Any],
]


class InterventionPolicy:
    """Select the least-cost intervention with a trustworthy safe UCB.

    Decision semantics are intentionally strict:

    * ``ALLOW`` only when the ``none`` intervention is reliably safe.
    * ``CONFIRM`` when ``verified_approval`` is the cheapest reliable repair.
    * ``REPLAN`` for every other reliable, non-empty repair.
    * ``BLOCK`` when capability restriction is the selected deny intervention.
    * ``HOLD`` when uncertainty prevents a reliable safety conclusion.
    * ``BLOCK`` when no intervention is safe and uncertainty is not the cause.
    """

    def __init__(
        self,
        safe_risk_limit: float = 0.20,
        epistemic_uncertainty_limit: float = 0.25,
    ) -> None:
        self.safe_risk_limit = _probability(safe_risk_limit, "safe_risk_limit")
        if self.safe_risk_limit >= 1.0:
            raise ValueError("safe_risk_limit must be strictly below 1")
        self.epistemic_uncertainty_limit = _probability(
            epistemic_uncertainty_limit, "epistemic_uncertainty_limit"
        )

    def _coerce(
        self,
        assessments: Union[
            Iterable[InterventionAssessment],
            Mapping[str, AssessmentInput],
        ],
    ) -> List[InterventionAssessment]:
        result: List[InterventionAssessment] = []
        if isinstance(assessments, Mapping):
            for intervention, value in assessments.items():
                if isinstance(value, InterventionAssessment):
                    if value.intervention != intervention:
                        raise ValueError("assessment mapping key does not match intervention")
                    result.append(value)
                elif isinstance(value, Mapping):
                    fields = dict(value)
                    fields.pop("intervention", None)
                    result.append(
                        InterventionAssessment(intervention=intervention, **fields)
                    )
                else:
                    raise TypeError("assessment values must be mappings or InterventionAssessment")
        else:
            for value in assessments:
                if not isinstance(value, InterventionAssessment):
                    raise TypeError("assessments must contain InterventionAssessment values")
                result.append(value)
        if not result:
            raise ValueError("at least one intervention assessment is required")
        names = [item.intervention for item in result]
        if len(names) != len(set(names)):
            raise ValueError("each intervention may be assessed only once")
        if "none" not in names:
            raise ValueError("the baseline 'none' intervention assessment is required")
        return result

    def _reliably_safe(self, value: InterventionAssessment) -> bool:
        return (
            value.risk_upper_bound <= self.safe_risk_limit
            and value.actual_risk <= self.safe_risk_limit
            and value.epistemic_uncertainty <= self.epistemic_uncertainty_limit
        )

    @staticmethod
    def _result(
        decision: str,
        intervention: Optional[InterventionAssessment],
        reason: str,
    ) -> PolicyDecision:
        if intervention is None:
            return PolicyDecision(decision, None, reason, None, None, None, None)
        return PolicyDecision(
            decision=decision,
            intervention=intervention.intervention,
            reason=reason,
            risk_upper_bound=intervention.risk_upper_bound,
            actual_risk=intervention.actual_risk,
            epistemic_uncertainty=intervention.epistemic_uncertainty,
            intervention_cost=intervention.resolved_cost,
        )

    def select(
        self,
        assessments: Union[
            Iterable[InterventionAssessment],
            Mapping[str, AssessmentInput],
        ],
    ) -> PolicyDecision:
        values = self._coerce(assessments)
        baseline = next(item for item in values if item.intervention == "none")

        # ALLOW can only be emitted from a calibrated, low-UCB no-intervention
        # branch.  A low point estimate alone is never sufficient.
        if self._reliably_safe(baseline):
            return self._result(
                "ALLOW", baseline, "no intervention has a reliably safe risk upper bound"
            )

        order = {name: index for index, name in enumerate(INTERVENTIONS)}
        repairs = [
            item
            for item in values
            if item.intervention != "none" and self._reliably_safe(item)
        ]
        repairs.sort(key=lambda item: (item.resolved_cost, order[item.intervention]))
        if repairs:
            selected = repairs[0]
            if selected.intervention == "verified_approval":
                return self._result(
                    "CONFIRM",
                    selected,
                    "verified approval is the least-cost reliably safe intervention",
                )
            if selected.intervention == "restrict_capability":
                return self._result(
                    "BLOCK",
                    selected,
                    "capability restriction denies the proposed action",
                )
            return self._result(
                "REPLAN",
                selected,
                "a repair intervention is required to reach the safe risk bound",
            )

        # If a high-uncertainty assessment is the reason no branch can be
        # certified safe, pause rather than claiming that every repair failed.
        uncertain = [
            item
            for item in values
            if item.epistemic_uncertainty > self.epistemic_uncertainty_limit
        ]
        if uncertain:
            uncertain.sort(key=lambda item: (item.resolved_cost, order[item.intervention]))
            return self._result(
                "HOLD",
                uncertain[0],
                "epistemic uncertainty prevents a reliable safety decision",
            )

        return self._result(
            "BLOCK", None, "no assessed intervention has a safe risk upper bound"
        )

    # A concise alias for call sites that model the policy as a decision rule.
    decide = select

    def select_from_values(
        self,
        risk_upper_bounds: Mapping[str, float],
        actual_risks: Mapping[str, float],
        epistemic_uncertainties: Mapping[str, float],
        intervention_costs: Optional[Mapping[str, float]] = None,
    ) -> PolicyDecision:
        names = set(risk_upper_bounds)
        if names != set(actual_risks) or names != set(epistemic_uncertainties):
            raise ValueError("all intervention value mappings must have identical keys")
        costs = intervention_costs or {}
        values = [
            InterventionAssessment(
                intervention=name,
                risk_upper_bound=risk_upper_bounds[name],
                actual_risk=actual_risks[name],
                epistemic_uncertainty=epistemic_uncertainties[name],
                cost=costs.get(name),
            )
            for name in names
        ]
        return self.select(values)


def choose_intervention(
    assessments: Union[
        Iterable[InterventionAssessment],
        Mapping[str, AssessmentInput],
    ],
    safe_risk_limit: float = 0.20,
    epistemic_uncertainty_limit: float = 0.25,
) -> PolicyDecision:
    """Functional wrapper around :class:`InterventionPolicy`."""

    return InterventionPolicy(
        safe_risk_limit=safe_risk_limit,
        epistemic_uncertainty_limit=epistemic_uncertainty_limit,
    ).select(assessments)


__all__ = [
    "InterventionAssessment",
    "InterventionPolicy",
    "PolicyDecision",
    "choose_intervention",
]
