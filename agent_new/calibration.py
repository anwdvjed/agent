"""Group-aware conformal calibration and trajectory-level threshold selection.

The public APIs in this module deliberately aggregate prefixes before fitting or
measuring false-positive rates.  A trajectory is unsafe when any labelled prefix
in its group is unsafe, and its risk score is the maximum prefix risk.  This
prevents correlated prefixes from being treated as independent calibration
examples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple


CALIBRATION_FORMAT = "agent-new-group-conformal-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("%s must be finite and in [0, 1]" % name)
    return result


def _validate_binary(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    result = float(value)
    if not math.isfinite(result) or result not in (0.0, 1.0):
        raise ValueError("%s must contain only binary outcomes" % name)
    return int(result)


def _materialize_inputs(
    predictions: Iterable[float],
    outcomes: Iterable[int],
    group_ids: Iterable[Hashable],
) -> Tuple[List[float], List[int], List[Hashable]]:
    prediction_values = [
        _validate_probability(value, "predictions") for value in predictions
    ]
    outcome_values = [_validate_binary(value, "outcomes") for value in outcomes]
    group_values = list(group_ids)
    if not prediction_values:
        raise ValueError("calibration inputs must not be empty")
    if not (
        len(prediction_values) == len(outcome_values) == len(group_values)
    ):
        raise ValueError("predictions, outcomes, and group_ids must have equal length")
    for index, group_id in enumerate(group_values):
        if group_id is None:
            raise ValueError("group_ids[%d] must not be None" % index)
        try:
            hash(group_id)
        except TypeError as exc:
            raise ValueError("group IDs must be hashable") from exc
    return prediction_values, outcome_values, group_values


def aggregate_trajectory_risk(
    predictions: Iterable[float],
    outcomes: Iterable[int],
    group_ids: Iterable[Hashable],
) -> Dict[Hashable, Tuple[float, int]]:
    """Return ``group_id -> (max_prefix_risk, any_unsafe)``.

    Maximum aggregation matches the operational question "did any prefix in
    this trajectory cross the risk boundary?" and is intentionally shared by
    conformal fitting and safe-trajectory FPR selection.
    """

    prediction_values, outcome_values, group_values = _materialize_inputs(
        predictions, outcomes, group_ids
    )
    aggregated: Dict[Hashable, Tuple[float, int]] = {}
    for prediction, outcome, group_id in zip(
        prediction_values, outcome_values, group_values
    ):
        previous = aggregated.get(group_id)
        if previous is None:
            aggregated[group_id] = (prediction, outcome)
        else:
            aggregated[group_id] = (
                max(previous[0], prediction),
                max(previous[1], outcome),
            )
    return aggregated


def _finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal ``ceil((n + 1) * (1 - alpha))`` order statistic."""

    if not scores:
        raise ValueError("at least one group score is required")
    ordered = sorted(float(score) for score in scores)
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    if rank > len(ordered):
        # The finite-sample conformal order statistic is +infinity.  Risks are
        # probabilities, so 1.0 is the conservative representable correction.
        return 1.0
    index = max(rank, 1) - 1
    return ordered[index]


@dataclass(frozen=True)
class GroupConformalCalibration:
    """Serializable one-sided correction fitted on independent groups."""

    alpha: float
    correction: float
    group_count: int
    unsafe_group_count: int
    aggregation: str = "max"
    branch: str = "none"
    ensemble_z: float = 1.96
    score_protocol: str = "ensemble_mean_plus_z_std_v1"
    format: str = CALIBRATION_FORMAT

    def __post_init__(self) -> None:
        _validate_probability(self.alpha, "alpha")
        if self.alpha in (0.0, 1.0):
            raise ValueError("alpha must lie strictly between 0 and 1")
        _validate_probability(self.correction, "correction")
        if self.group_count <= 0:
            raise ValueError("group_count must be positive")
        if not 0 < self.unsafe_group_count < self.group_count:
            raise ValueError(
                "calibration requires at least one safe and one unsafe independent group"
            )
        minimum_unsafe = max(1, int(math.ceil(1.0 / self.alpha)) - 1)
        if self.unsafe_group_count < minimum_unsafe:
            raise ValueError(
                "insufficient unsafe calibration groups for the requested alpha; "
                f"need at least {minimum_unsafe}"
            )
        if self.aggregation != "max":
            raise ValueError("only max trajectory aggregation is supported")
        if self.branch != "none":
            raise ValueError(
                "this calibration format currently certifies only the actual 'none' branch"
            )
        if not math.isfinite(self.ensemble_z) or self.ensemble_z < 0.0:
            raise ValueError("ensemble_z must be finite and non-negative")
        if self.score_protocol != "ensemble_mean_plus_z_std_v1":
            raise ValueError("unsupported calibration score protocol")
        if self.format != CALIBRATION_FORMAT:
            raise ValueError("unsupported calibration format")

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @property
    def calibrated_quantity(self) -> float:
        """The fitted additive one-sided conformal correction."""

        return self.correction

    def risk_upper_bound(self, prediction: float) -> float:
        value = _validate_probability(prediction, "prediction")
        return min(1.0, value + self.correction)

    def risk_upper_bounds(self, predictions: Iterable[float]) -> List[float]:
        return [self.risk_upper_bound(value) for value in predictions]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroupConformalCalibration":
        required = {
            "alpha",
            "correction",
            "group_count",
            "unsafe_group_count",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError("calibration is missing keys: %s" % sorted(missing))
        return cls(
            alpha=float(value["alpha"]),
            correction=float(value["correction"]),
            group_count=int(value["group_count"]),
            unsafe_group_count=int(value["unsafe_group_count"]),
            aggregation=str(value.get("aggregation", "max")),
            branch=str(value.get("branch", "none")),
            ensemble_z=float(value.get("ensemble_z", 1.96)),
            score_protocol=str(
                value.get("score_protocol", "ensemble_mean_plus_z_std_v1")
            ),
            format=str(value.get("format", CALIBRATION_FORMAT)),
        )

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


class GroupConformalRiskCalibrator:
    """Fit a one-sided risk correction using trajectories as exchangeable units.

    For each trajectory, the nonconformity score is
    ``max(0, any_unsafe - max_prefix_prediction)``.  Clipping at zero makes the
    result a conservative upper bound that never reduces the model prediction.
    """

    def __init__(self, alpha: float = 0.05, ensemble_z: float = 1.96) -> None:
        value = _validate_probability(alpha, "alpha")
        if value in (0.0, 1.0):
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.alpha = value
        if not math.isfinite(float(ensemble_z)) or float(ensemble_z) < 0.0:
            raise ValueError("ensemble_z must be finite and non-negative")
        self.ensemble_z = float(ensemble_z)
        self.calibration_: Optional[GroupConformalCalibration] = None

    def fit(
        self,
        predictions: Iterable[float],
        outcomes: Iterable[int],
        group_ids: Iterable[Hashable],
    ) -> "GroupConformalRiskCalibrator":
        grouped = aggregate_trajectory_risk(predictions, outcomes, group_ids)
        # Safety uses a Mondrian/class-conditional correction on unsafe
        # trajectories.  Mixing many safe zero residuals would hide a rare but
        # completely missed harmful trajectory while preserving only marginal
        # coverage.
        scores = [
            max(0.0, 1.0 - prediction)
            for prediction, outcome in grouped.values()
            if outcome == 1
        ]
        correction = _finite_sample_quantile(scores, self.alpha)
        unsafe = sum(outcome for _, outcome in grouped.values())
        self.calibration_ = GroupConformalCalibration(
            alpha=self.alpha,
            correction=correction,
            group_count=len(grouped),
            unsafe_group_count=unsafe,
            ensemble_z=self.ensemble_z,
        )
        return self

    @property
    def calibration(self) -> GroupConformalCalibration:
        if self.calibration_ is None:
            raise RuntimeError("calibrator has not been fitted")
        return self.calibration_

    def risk_upper_bound(self, prediction: float) -> float:
        return self.calibration.risk_upper_bound(prediction)

    def risk_upper_bounds(self, predictions: Iterable[float]) -> List[float]:
        return self.calibration.risk_upper_bounds(predictions)

    def calibrated_quantity(self) -> float:
        return self.calibration.calibrated_quantity


def fit_group_conformal(
    predictions: Iterable[float],
    outcomes: Iterable[int],
    group_ids: Iterable[Hashable],
    alpha: float = 0.05,
    ensemble_z: float = 1.96,
) -> GroupConformalCalibration:
    """Functional wrapper returning a serializable calibration object."""

    return GroupConformalRiskCalibrator(alpha=alpha, ensemble_z=ensemble_z).fit(
        predictions, outcomes, group_ids
    ).calibration


@dataclass(frozen=True)
class SafeTrajectoryThreshold:
    """Threshold selected from trajectory aggregates, never independent prefixes.

    Alerts use the strict comparison ``trajectory_score > threshold``.  A
    threshold of 1.0 therefore represents an explicit no-alert operating point
    without inventing a probability larger than one.
    """

    threshold: float
    target_safe_trajectory_fpr: float
    safe_trajectory_fpr: float
    unsafe_trajectory_recall: float
    safe_trajectory_count: int
    unsafe_trajectory_count: int
    safe_false_positives: int
    unsafe_true_positives: int
    comparison: str = "greater_than"

    def __post_init__(self) -> None:
        _validate_probability(self.threshold, "threshold")
        _validate_probability(
            self.target_safe_trajectory_fpr, "target_safe_trajectory_fpr"
        )
        _validate_probability(self.safe_trajectory_fpr, "safe_trajectory_fpr")
        _validate_probability(self.unsafe_trajectory_recall, "unsafe_trajectory_recall")
        if self.safe_trajectory_count <= 0:
            raise ValueError("safe_trajectory_count must be positive")
        if self.unsafe_trajectory_count < 0:
            raise ValueError("unsafe_trajectory_count must be non-negative")
        if self.comparison != "greater_than":
            raise ValueError("unsupported threshold comparison")

    def is_alert(self, trajectory_score: float) -> bool:
        return _validate_probability(trajectory_score, "trajectory_score") > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_threshold_at_safe_trajectory_fpr(
    predictions: Iterable[float],
    outcomes: Iterable[int],
    group_ids: Iterable[Hashable],
    target_fpr: float = 0.05,
    calibration: Optional[GroupConformalCalibration] = None,
) -> SafeTrajectoryThreshold:
    """Maximize unsafe-trajectory recall under an empirical safe-trajectory FPR.

    Prefixes are first aggregated with ``max`` within each group.  If a
    conformal calibration is supplied, it is applied to every prefix before
    aggregation.  The returned FPR denominator is the number of safe
    trajectories, not the number of negative prefixes.
    """

    target = _validate_probability(target_fpr, "target_fpr")
    prediction_values, outcome_values, group_values = _materialize_inputs(
        predictions, outcomes, group_ids
    )
    if calibration is not None:
        prediction_values = calibration.risk_upper_bounds(prediction_values)
    grouped = aggregate_trajectory_risk(
        prediction_values, outcome_values, group_values
    )
    safe_scores = [score for score, outcome in grouped.values() if outcome == 0]
    unsafe_scores = [score for score, outcome in grouped.values() if outcome == 1]
    if not safe_scores:
        raise ValueError("safe-trajectory FPR requires at least one safe trajectory")

    # Strict comparison permits 1.0 to be a valid, explicit no-alert point.  A
    # predecessor threshold includes all ties at each positive observed score.
    candidates = {1.0}
    for score, _ in grouped.values():
        if score > 0.0:
            candidates.add(max(0.0, math.nextafter(score, 0.0)))
        else:
            candidates.add(0.0)

    best: Optional[Tuple[float, float, float, int, int]] = None
    for threshold in sorted(candidates, reverse=True):
        false_positives = sum(score > threshold for score in safe_scores)
        true_positives = sum(score > threshold for score in unsafe_scores)
        safe_fpr = false_positives / len(safe_scores)
        recall = true_positives / len(unsafe_scores) if unsafe_scores else 0.0
        if safe_fpr > target:
            continue
        candidate = (recall, -safe_fpr, threshold, false_positives, true_positives)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:  # Defensive: threshold 1.0 is always feasible.
        raise RuntimeError("no feasible trajectory threshold was found")
    recall, negative_fpr, threshold, false_positives, true_positives = best
    return SafeTrajectoryThreshold(
        threshold=threshold,
        target_safe_trajectory_fpr=target,
        safe_trajectory_fpr=-negative_fpr,
        unsafe_trajectory_recall=recall,
        safe_trajectory_count=len(safe_scores),
        unsafe_trajectory_count=len(unsafe_scores),
        safe_false_positives=false_positives,
        unsafe_true_positives=true_positives,
    )


# Readable compatibility alias for callers migrating from older metric code.
choose_threshold_at_safe_trajectory_fpr = select_threshold_at_safe_trajectory_fpr


__all__ = [
    "CALIBRATION_FORMAT",
    "GroupConformalCalibration",
    "GroupConformalRiskCalibrator",
    "SafeTrajectoryThreshold",
    "aggregate_trajectory_risk",
    "choose_threshold_at_safe_trajectory_fpr",
    "fit_group_conformal",
    "select_threshold_at_safe_trajectory_fpr",
]
