"""Stable objectives for the Agent New factorized competing-risk model.

The occurrence head and cause head are trained directly in logit space.  In
particular, intervention branches are never supervised with ``clamp`` followed
by binary cross entropy on a saturated horizon-risk probability.  Branches
with event time/cause labels use the full right-censored likelihood; branches
with only horizon labels use a stable survival/event likelihood.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    """Weights and margins for the composite training objective."""

    survival: float = 1.0
    intervention_survival: float = 0.75
    concepts: float = 0.30
    evidence_supervision: float = 0.20
    evidence_sufficiency: float = 0.20
    evidence_necessity: float = 0.20
    evidence_sparsity: float = 0.01
    intervention_ranking: float = 0.10
    attack_intent: float = 0.25
    utility: float = 0.25
    necessity_margin: float = 0.05
    intervention_margin: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LossWeights":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _reduce_losses(
    values: torch.Tensor,
    reduction: str,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if reduction == "none":
        if weights is None:
            return values
        return values * weights.to(values.dtype)
    if reduction == "sum":
        if weights is None:
            return values.sum()
        return (values * weights.to(values.dtype)).sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if weights is None:
        return values.mean()
    cast_weights = weights.to(values.dtype)
    if not bool(torch.isfinite(cast_weights).all()) or bool((cast_weights < 0).any()):
        raise ValueError("loss weights must be finite and non-negative")
    denominator = cast_weights.sum()
    numerator = (values * cast_weights).sum()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        values.sum() * 0.0,
    )


def factorized_competing_risk_losses(
    event_logits: torch.Tensor,
    cause_logits: torch.Tensor,
    event_observed: torch.Tensor,
    event_time: torch.Tensor,
    event_type: torch.Tensor,
    censor_time: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample first-event negative log likelihoods.

    Args:
        event_logits: Any-event logits with shape ``[B, H]``.
        cause_logits: Conditional cause logits with shape ``[B, H, K]``.
        event_observed: Boolean/0-1 tensor with shape ``[B]``.
        event_time: One-indexed event bin for observed events.  Values for
            censored examples are ignored.
        event_type: Zero-indexed cause for observed events.  Values for
            censored examples are ignored.
        censor_time: Number of fully observed no-event bins for censored
            examples.  Zero is allowed; values beyond ``H`` are administratively
            censored at ``H``.

    For an event of cause ``k`` at bin ``t``, the likelihood is survival in
    bins before ``t``, followed by the event probability and conditional cause
    probability at ``t``.  For a right-censored example it is survival through
    every observed bin.  ``softplus`` and ``log_softmax`` keep the calculation
    stable even for extreme logits.
    """

    if event_logits.ndim != 2:
        raise ValueError("event_logits must have shape [B, H]")
    if cause_logits.ndim != 3 or cause_logits.shape[:2] != event_logits.shape:
        raise ValueError("cause_logits must have shape [B, H, K]")
    batch_size, horizon = event_logits.shape
    expected = (batch_size,)
    for name, value in (
        ("event_observed", event_observed),
        ("event_time", event_time),
        ("event_type", event_type),
        ("censor_time", censor_time),
    ):
        if value.shape != expected:
            raise ValueError("{} must have shape [B]".format(name))

    if event_observed.dtype != torch.bool:
        observed_values = event_observed.to(torch.float32)
        if not bool(torch.isfinite(observed_values).all()) or bool(
            ((observed_values != 0) & (observed_values != 1)).any()
        ):
            raise ValueError("event_observed must contain only 0/1 values")
    for name, value in (
        ("event_time", event_time),
        ("event_type", event_type),
        ("censor_time", censor_time),
    ):
        numeric = value.to(torch.float64)
        if not bool(torch.isfinite(numeric).all()) or bool(
            (numeric != torch.round(numeric)).any()
        ):
            raise ValueError(f"{name} must contain finite integers")
    observed = event_observed.to(torch.bool)
    times = event_time.long()
    causes = event_type.long()
    censors = censor_time.long()
    if bool((observed & ((times < 1) | (times > horizon))).any()):
        raise ValueError("an observed event_time lies outside [1, H]")
    num_causes = cause_logits.shape[-1]
    if bool((observed & ((causes < 0) | (causes >= num_causes))).any()):
        raise ValueError("an observed event_type lies outside [0, K)")
    if bool((~observed & (censors < 0)).any()):
        raise ValueError("censor_time cannot be negative")

    time_index = (times - 1).clamp(min=0, max=horizon - 1)
    safe_causes = causes.clamp(min=0, max=num_causes - 1)
    bins = torch.arange(horizon, device=event_logits.device).unsqueeze(0)

    # -log P(no event) = softplus(event_logit)
    no_event_nll = F.softplus(event_logits)
    survived_before = bins < time_index.unsqueeze(1)
    observed_survival = (no_event_nll * survived_before).sum(dim=1)
    selected_event_logits = event_logits.gather(1, time_index.unsqueeze(1)).squeeze(1)
    event_nll = F.softplus(-selected_event_logits)
    selected_cause_logits = cause_logits[
        torch.arange(batch_size, device=cause_logits.device), time_index
    ]
    cause_nll = -F.log_softmax(selected_cause_logits, dim=-1).gather(
        1, safe_causes.unsqueeze(1)
    ).squeeze(1)
    observed_loss = observed_survival + event_nll + cause_nll

    observed_censor_bins = censors.clamp(min=0, max=horizon)
    censor_mask = bins < observed_censor_bins.unsqueeze(1)
    censored_loss = (no_event_nll * censor_mask).sum(dim=1)
    return torch.where(observed, observed_loss, censored_loss)


def factorized_competing_risk_nll(
    event_logits: torch.Tensor,
    cause_logits: torch.Tensor,
    event_observed: torch.Tensor,
    event_time: torch.Tensor,
    event_type: torch.Tensor,
    censor_time: torch.Tensor,
    reduction: str = "mean",
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce the stable right-censored factorized likelihood."""

    values = factorized_competing_risk_losses(
        event_logits,
        cause_logits,
        event_observed,
        event_time,
        event_type,
        censor_time,
    )
    return _reduce_losses(values, reduction, sample_weight)


def horizon_event_losses(
    event_logits: torch.Tensor, event_within_horizon: torch.Tensor
) -> torch.Tensor:
    """Stable per-sample NLL when only a horizon event/no-event label exists.

    This is a weaker target than a first-event label and should only be used for
    intervention branches lacking event time/cause annotation.  It is computed
    from log survival, not from a clamped horizon-risk probability.
    """

    if event_logits.ndim != 2:
        raise ValueError("event_logits must have shape [B, H]")
    if event_within_horizon.shape != event_logits.shape[:1]:
        raise ValueError("event_within_horizon must have shape [B]")
    if event_within_horizon.dtype != torch.bool:
        numeric_target = event_within_horizon.to(torch.float32)
        if not bool(torch.isfinite(numeric_target).all()) or bool(
            ((numeric_target != 0) & (numeric_target != 1)).any()
        ):
            raise ValueError("event_within_horizon must contain only 0/1 values")
    log_no_event = F.logsigmoid(-event_logits)
    log_survival_before = torch.cat(
        [
            torch.zeros_like(log_no_event[:, :1]),
            torch.cumsum(log_no_event[:, :-1], dim=1),
        ],
        dim=1,
    )
    # Summing mutually exclusive first-event masses avoids computing
    # log(1 - S_H), whose subtraction loses all precision when every event
    # probability is extremely small.
    log_first_event_mass = log_survival_before + F.logsigmoid(event_logits)
    log_horizon_event = torch.logsumexp(log_first_event_mass, dim=1)
    log_survival = log_no_event.sum(dim=1)
    event_target = event_within_horizon.to(torch.bool)
    return torch.where(event_target, -log_horizon_event, -log_survival)


def _weighted_mean(
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0:
        return reference.sum() * 0.0
    if weights is None:
        return values.mean()
    cast_weights = weights.to(dtype=values.dtype)
    if not bool(torch.isfinite(cast_weights).all()) or bool((cast_weights < 0).any()):
        raise ValueError("supervision/sample weights must be finite and non-negative")
    denominator = cast_weights.sum()
    return torch.where(
        denominator > 0,
        (values * cast_weights).sum()
        / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        reference.sum() * 0.0,
    )


_TARGET_KEYS = (
    "event_observed",
    "event_time",
    "event_type",
    "censor_time",
    "event_within_horizon",
    "concept_targets",
    "concept_mask",
    "supervision_concepts",
    "evidence_labels",
    "evidence_mask",
    "supervision_evidence",
    "evidence_available",
    "sample_weight",
    "supervision_survival",
    "supervision_intervention",
    "attack_intent_target",
    "supervision_attack_intent",
    "utility_target",
    "supervision_utility",
)


def _masked_binary_auxiliary_loss(
    logits: torch.Tensor,
    targets: Mapping[str, Any],
    target_key: str,
    supervision_key: str,
) -> Optional[torch.Tensor]:
    if target_key not in targets:
        return None
    labels = targets[target_key].to(logits.dtype)
    if labels.shape != logits.shape:
        raise ValueError(f"{target_key} must match auxiliary logits")
    if not bool(torch.isfinite(labels).all()) or bool(
        ((labels != 0) & (labels != 1)).any()
    ):
        raise ValueError(f"{target_key} must contain only finite 0/1 values")
    supervision = targets.get(
        supervision_key, torch.ones_like(labels)
    ).to(logits.dtype)
    sample_weight = targets.get(
        "sample_weight", torch.ones_like(labels)
    ).to(logits.dtype)
    values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return _weighted_mean(values, supervision * sample_weight, logits)


def _branch_targets(
    batch: Mapping[str, Any], name: str, actual_name: str
) -> Dict[str, Any]:
    """Resolve labels while allowing either co-located or separate targets."""

    resolved: Dict[str, Any] = {}
    branch_inputs = batch.get("branches", {})
    if isinstance(branch_inputs, Mapping):
        branch = branch_inputs.get(name)
        if isinstance(branch, Mapping):
            resolved.update({key: branch[key] for key in _TARGET_KEYS if key in branch})
    for container_name in ("branch_targets", "intervention_targets"):
        container = batch.get(container_name)
        if isinstance(container, Mapping):
            values = container.get(name)
            if isinstance(values, Mapping):
                resolved.update(values)
    if name == actual_name:
        resolved.update({key: batch[key] for key in _TARGET_KEYS if key in batch})
    return resolved


def _has_survival_targets(targets: Mapping[str, Any]) -> bool:
    return all(
        key in targets
        for key in (
            "event_observed",
            "event_time",
            "event_type",
            "censor_time",
        )
    )


def _survival_loss_for_branch(
    branch_output: Mapping[str, torch.Tensor], targets: Mapping[str, Any]
) -> Optional[torch.Tensor]:
    if _has_survival_targets(targets):
        values = factorized_competing_risk_losses(
            branch_output["event_logits"],
            branch_output["cause_logits"],
            targets["event_observed"],
            targets["event_time"],
            targets["event_type"],
            targets["censor_time"],
        )
    elif "event_within_horizon" in targets:
        values = horizon_event_losses(
            branch_output["event_logits"], targets["event_within_horizon"]
        )
    else:
        return None
    sample_weight = targets.get("sample_weight", torch.ones_like(values))
    supervision = targets.get(
        "supervision_survival",
        targets.get("supervision_intervention", torch.ones_like(values)),
    )
    effective_weight = sample_weight.to(values.dtype) * supervision.to(values.dtype)
    if not bool((effective_weight > 0).any()):
        return None
    return _weighted_mean(
        values,
        effective_weight,
        branch_output["event_logits"],
    )


def multi_intervention_label_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any]
) -> torch.Tensor:
    """Train every labeled non-actual intervention branch.

    Preferred labels are per-branch first-event targets under
    ``batch['branch_targets'][name]`` (or ``intervention_targets``).  When only
    binary horizon outcomes are available, either place
    ``event_within_horizon`` in each branch target or provide an aligned
    ``batch['intervention_event_targets']`` matrix of shape ``[B, I]``.
    """

    actual_name = str(outputs["actual_name"])
    intervention_names = outputs["intervention_names"]
    matrix_labels = batch.get("intervention_event_targets")
    matrix_mask = batch.get("intervention_label_mask")
    if matrix_labels is not None:
        target_names = batch.get("intervention_target_names")
        if target_names is None:
            raise ValueError(
                "intervention_target_names is required with matrix intervention labels"
            )
        target_names = tuple(str(name) for name in target_names)
        if len(target_names) != matrix_labels.shape[1]:
            raise ValueError(
                "intervention_target_names must align with intervention label columns"
            )
        if set(target_names) != set(intervention_names):
            raise ValueError("intervention target names do not match model branches")
        reorder = torch.tensor(
            [target_names.index(name) for name in intervention_names],
            device=matrix_labels.device,
            dtype=torch.long,
        )
        matrix_labels = matrix_labels.index_select(1, reorder)
        if matrix_mask is None:
            matrix_mask = torch.ones_like(matrix_labels)
        elif matrix_mask.shape != matrix_labels.shape:
            raise ValueError("intervention_label_mask must match intervention targets")
        else:
            matrix_mask = matrix_mask.index_select(1, reorder)

    losses: List[torch.Tensor] = []
    for column, name in enumerate(intervention_names):
        branch_output = outputs["branches"][name]
        targets = _branch_targets(batch, name, actual_name)
        survival_loss = _survival_loss_for_branch(branch_output, targets)
        if survival_loss is not None:
            losses.append(survival_loss)
            continue
        if matrix_labels is not None:
            values = horizon_event_losses(
                branch_output["event_logits"], matrix_labels[:, column]
            )
            sample_weight = batch.get("sample_weight", torch.ones_like(values))
            supervision = matrix_mask[:, column]
            losses.append(
                _weighted_mean(
                    values,
                    sample_weight.to(values.dtype) * supervision.to(values.dtype),
                    branch_output["event_logits"],
                )
            )

    if not losses:
        return outputs["actual_risk"].sum() * 0.0
    return torch.stack(losses).mean()


def concept_supervision_loss(
    branch_output: Mapping[str, torch.Tensor], targets: Mapping[str, Any]
) -> Optional[torch.Tensor]:
    """Masked binary supervision for interpretable safety concepts."""

    if "concept_targets" not in targets:
        return None
    logits = branch_output["concept_logits"]
    labels = targets["concept_targets"].to(logits.dtype)
    if labels.shape != logits.shape:
        raise ValueError("concept_targets must match concept_logits")
    mask = targets.get(
        "concept_mask",
        targets.get("supervision_concepts", torch.ones_like(labels)),
    ).to(logits.dtype)
    if mask.shape == logits.shape[:1]:
        mask = mask.unsqueeze(1)
    if mask.shape != logits.shape:
        try:
            mask = torch.broadcast_to(mask, logits.shape)
        except RuntimeError as error:
            raise ValueError("concept supervision mask is not broadcastable") from error
    sample_weight = targets.get(
        "sample_weight", logits.new_ones((logits.shape[0],))
    ).to(logits.dtype)
    if sample_weight.ndim == 0:
        sample_weight = sample_weight.expand(logits.shape[0])
    if sample_weight.shape != logits.shape[:1]:
        raise ValueError("sample_weight must have shape [B]")
    weights = mask * sample_weight.unsqueeze(1)
    values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return _weighted_mean(values.reshape(-1), weights.reshape(-1), logits)


def evidence_supervision_loss(
    branch_output: Mapping[str, torch.Tensor], targets: Mapping[str, Any]
) -> Optional[torch.Tensor]:
    """Masked edge-level supervision for the differentiable evidence mask."""

    if "evidence_labels" not in targets:
        return None
    logits = branch_output["evidence_logits"]
    labels = targets["evidence_labels"].to(logits.dtype)
    if labels.shape != logits.shape:
        raise ValueError("evidence_labels must match evidence_logits")
    raw_mask = targets.get(
        "evidence_mask",
        targets.get("supervision_evidence", torch.ones_like(labels)),
    ).to(logits.dtype)
    edge_batch = branch_output["edge_batch"]
    if raw_mask.ndim == 0:
        mask = raw_mask.expand_as(logits)
    elif raw_mask.shape == logits.shape:
        mask = raw_mask
    elif raw_mask.shape == branch_output["risk"].shape:
        mask = raw_mask[edge_batch]
    else:
        raise ValueError(
            "evidence mask must be scalar, sample-level [B], or edge-level [E]"
        )
    if "evidence_available" in targets:
        available = targets["evidence_available"]
        if available.ndim == 0:
            mask = mask * available.to(logits.dtype)
        elif available.shape == logits.shape:
            mask = mask * available.to(logits.dtype)
        elif available.shape == branch_output["risk"].shape:
            mask = mask * available[edge_batch].to(logits.dtype)
        else:
            raise ValueError(
                "evidence_available must be scalar, sample-level [B], or edge-level [E]"
            )
    sample_weight = targets.get(
        "sample_weight",
        logits.new_ones((branch_output["risk"].shape[0],)),
    ).to(logits.dtype)
    if sample_weight.ndim == 0:
        sample_weight = sample_weight.expand(branch_output["risk"].shape[0])
    if sample_weight.shape != branch_output["risk"].shape:
        raise ValueError("sample_weight must have shape [B]")
    weights = mask * sample_weight[edge_batch]
    values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return _weighted_mean(values, weights, logits)


def _edge_mean_by_sample(
    edge_values: torch.Tensor, edge_batch: torch.Tensor, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = edge_values.new_zeros((batch_size,))
    counts = edge_values.new_zeros((batch_size,))
    if edge_values.numel() > 0:
        sums.index_add_(0, edge_batch, edge_values)
        counts.index_add_(0, edge_batch, torch.ones_like(edge_values))
    return sums / counts.clamp_min(1.0), counts > 0


def evidence_regularization_losses(
    branch_output: Mapping[str, torch.Tensor],
    targets: Mapping[str, Any],
    necessity_margin: float,
) -> Dict[str, torch.Tensor]:
    """Return differentiable evidence sufficiency, necessity, and sparsity.

    Sufficiency asks the selected subgraph to preserve full-graph risk.
    Necessity asks removal of selected evidence to reduce risk on labeled unsafe
    samples.  Sparsity penalizes the mean selected edge mass per non-empty
    graph, preventing the trivial all-edge explanation.
    """

    selected_risk = branch_output["risk"]
    full_risk = branch_output["full_risk"]
    complement_risk = branch_output["complement_risk"]
    sample_weight = targets.get(
        "sample_weight", torch.ones_like(selected_risk)
    ).to(selected_risk.dtype)
    if sample_weight.ndim == 0:
        sample_weight = sample_weight.expand_as(selected_risk)
    if sample_weight.shape != selected_risk.shape:
        raise ValueError("sample_weight must have shape [B]")
    sufficiency_values = F.smooth_l1_loss(
        selected_risk, full_risk, reduction="none"
    )
    sufficiency = _weighted_mean(
        sufficiency_values, sample_weight, selected_risk
    )

    unsafe_mask = None
    if "event_observed" in targets:
        unsafe_mask = targets["event_observed"].to(torch.bool)
    elif "event_within_horizon" in targets:
        unsafe_mask = targets["event_within_horizon"].to(torch.bool)
    elif "evidence_labels" in targets:
        labels = targets["evidence_labels"].to(selected_risk.dtype)
        edge_batch = branch_output["edge_batch"]
        positive_sum = selected_risk.new_zeros(selected_risk.shape)
        if labels.numel() > 0:
            positive_sum.index_add_(0, edge_batch, labels)
        unsafe_mask = positive_sum > 0
    if unsafe_mask is None:
        unsafe_mask = torch.zeros_like(selected_risk, dtype=torch.bool)
    if "supervision_survival" in targets:
        unsafe_mask = unsafe_mask & (
            targets["supervision_survival"].to(selected_risk.device) > 0
        )
    if "evidence_available" in targets:
        available = targets["evidence_available"].to(selected_risk.device)
        if available.shape == selected_risk.shape:
            unsafe_mask = unsafe_mask & available.to(torch.bool)
    if unsafe_mask.shape != selected_risk.shape:
        raise ValueError("event/evidence availability labels must resolve to shape [B]")
    necessity_values = F.relu(
        necessity_margin - full_risk + complement_risk
    )
    necessity = _weighted_mean(
        necessity_values,
        sample_weight * unsafe_mask.to(sample_weight.dtype),
        selected_risk,
    )

    sparse_values, has_edges = _edge_mean_by_sample(
        branch_output["evidence_weights"],
        branch_output["edge_batch"],
        selected_risk.shape[0],
    )
    sparsity = _weighted_mean(
        sparse_values,
        sample_weight * has_edges.to(sample_weight.dtype),
        branch_output["evidence_weights"],
    )
    return {
        "sufficiency": sufficiency,
        "necessity": necessity,
        "sparsity": sparsity,
    }


def intervention_ranking_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], margin: float
) -> torch.Tensor:
    """Optional cumulative-hazard ranking for labeled intervention effects.

    ``batch['intervention_effect_targets']`` is aligned to
    ``outputs['intervention_names']``.  ``+1`` means the intervention should be
    safer than actual, ``-1`` means riskier, and ``0`` means unsupervised.  The
    loss compares cumulative hazards, avoiding probability saturation.
    """

    if "intervention_effect_targets" not in batch:
        return outputs["actual_risk"].sum() * 0.0
    targets = batch["intervention_effect_targets"].to(
        outputs["actual"]["cumulative_hazard"].dtype
    )
    expected_shape = outputs["intervention_risk_matrix"].shape
    if targets.shape != expected_shape:
        raise ValueError("intervention_effect_targets must have shape [B, I]")
    actual_hazard = outputs["actual"]["cumulative_hazard"].unsqueeze(1)
    intervention_hazards = torch.stack(
        [
            outputs["branches"][name]["cumulative_hazard"]
            for name in outputs["intervention_names"]
        ],
        dim=1,
    ) if outputs["intervention_names"] else actual_hazard.new_empty(
        (actual_hazard.shape[0], 0)
    )
    signed_target = torch.sign(targets)
    mask = targets != 0
    if "intervention_effect_mask" in batch:
        mask = mask & batch["intervention_effect_mask"].to(torch.bool)
    hazard_improvement = actual_hazard - intervention_hazards
    values = F.softplus(margin - signed_target * hazard_improvement)
    return _weighted_mean(
        values.reshape(-1),
        mask.to(values.dtype).reshape(-1),
        outputs["actual_risk"],
    )


def agent_new_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: Optional[LossWeights] = None,
    *,
    allow_auxiliary_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the complete Agent New training objective.

    Labels may live beside each branch input or under ``branch_targets`` /
    ``intervention_targets``.  Missing auxiliary labels are handled by exact
    differentiable zeros, allowing phased training without silently inventing
    targets.
    """

    weights = weights or LossWeights()
    actual_name = str(outputs["actual_name"])
    actual_targets = _branch_targets(batch, actual_name, actual_name)
    outcome_supervision = (
        _survival_loss_for_branch(outputs["actual"], actual_targets) is not None
    )
    if not outcome_supervision:
        for name in outputs["intervention_names"]:
            targets = _branch_targets(batch, name, actual_name)
            outcome_supervision = outcome_supervision or (
                _survival_loss_for_branch(outputs["branches"][name], targets)
                is not None
            )
    if not outcome_supervision and "intervention_event_targets" in batch:
        mask = batch.get(
            "intervention_label_mask",
            torch.ones_like(batch["intervention_event_targets"]),
        )
        outcome_supervision = bool((mask > 0).any())
    if not outcome_supervision and not allow_auxiliary_only:
        raise ValueError(
            "Agent New training requires at least one supervised outcome; "
            "set allow_auxiliary_only=True only for an explicit auxiliary pretraining phase"
        )
    actual_survival = _survival_loss_for_branch(outputs["actual"], actual_targets)
    if actual_survival is None:
        actual_survival = outputs["actual_risk"].sum() * 0.0
    intervention_survival = multi_intervention_label_loss(outputs, batch)

    concept_terms: List[torch.Tensor] = []
    evidence_terms: List[torch.Tensor] = []
    sufficiency_terms: List[torch.Tensor] = []
    necessity_terms: List[torch.Tensor] = []
    sparsity_terms: List[torch.Tensor] = []
    for name in outputs["branch_names"]:
        branch_output = outputs["branches"][name]
        targets = _branch_targets(batch, name, actual_name)
        concept_term = concept_supervision_loss(branch_output, targets)
        if concept_term is not None:
            concept_terms.append(concept_term)
        evidence_term = evidence_supervision_loss(branch_output, targets)
        if evidence_term is not None:
            evidence_terms.append(evidence_term)
        regularizers = evidence_regularization_losses(
            branch_output, targets, weights.necessity_margin
        )
        sufficiency_terms.append(regularizers["sufficiency"])
        necessity_terms.append(regularizers["necessity"])
        sparsity_terms.append(regularizers["sparsity"])

    zero = outputs["actual_risk"].sum() * 0.0
    concepts = torch.stack(concept_terms).mean() if concept_terms else zero
    evidence = torch.stack(evidence_terms).mean() if evidence_terms else zero
    sufficiency = torch.stack(sufficiency_terms).mean()
    necessity = torch.stack(necessity_terms).mean()
    sparsity = torch.stack(sparsity_terms).mean()
    ranking = intervention_ranking_loss(
        outputs, batch, weights.intervention_margin
    )
    attack_intent = _masked_binary_auxiliary_loss(
        outputs["actual"]["attack_intent_logit"],
        actual_targets,
        "attack_intent_target",
        "supervision_attack_intent",
    )
    utility = _masked_binary_auxiliary_loss(
        outputs["actual"]["utility_logit"],
        actual_targets,
        "utility_target",
        "supervision_utility",
    )
    attack_intent = zero if attack_intent is None else attack_intent
    utility = zero if utility is None else utility
    total = (
        weights.survival * actual_survival
        + weights.intervention_survival * intervention_survival
        + weights.concepts * concepts
        + weights.evidence_supervision * evidence
        + weights.evidence_sufficiency * sufficiency
        + weights.evidence_necessity * necessity
        + weights.evidence_sparsity * sparsity
        + weights.intervention_ranking * ranking
        + weights.attack_intent * attack_intent
        + weights.utility * utility
    )
    parts = {
        "total": total.detach(),
        "survival": actual_survival.detach(),
        "intervention_survival": intervention_survival.detach(),
        "concepts": concepts.detach(),
        "evidence_supervision": evidence.detach(),
        "evidence_sufficiency": sufficiency.detach(),
        "evidence_necessity": necessity.detach(),
        "evidence_sparsity": sparsity.detach(),
        "intervention_ranking": ranking.detach(),
        "attack_intent": attack_intent.detach(),
        "utility": utility.detach(),
    }
    return total, parts


__all__ = [
    "LossWeights",
    "agent_new_loss",
    "concept_supervision_loss",
    "evidence_regularization_losses",
    "evidence_supervision_loss",
    "factorized_competing_risk_losses",
    "factorized_competing_risk_nll",
    "horizon_event_losses",
    "intervention_ranking_loss",
    "multi_intervention_label_loss",
]
