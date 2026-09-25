"""Explicit action-time labels: zero-based episode steps, one-based model bins.

No terminal-step imputation. Unknown and weak labels never become exact events.
Interval prefixes stop at the earliest possible first event; already-at-risk
status must not be assumed after that boundary.
"""
from .constants import DEFAULT_EVENT_TYPES


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_annotation(annotation, length):
    if not isinstance(annotation, dict):
        raise ValueError("annotation must be an object")
    kind = annotation.get("kind")
    if kind not in {"exact", "interval", "right_censored", "weak"}:
        raise ValueError("annotation.kind must be exact, interval, right_censored or weak")
    if kind == "weak":
        return
    if annotation.get("verified") is not True or not annotation.get("provenance"):
        raise ValueError("action-time labels need verified=true and provenance")
    if kind in {"exact", "interval"}:
        if annotation.get("event_type") not in DEFAULT_EVENT_TYPES:
            raise ValueError("first event needs a known cause")
        lo = integer(annotation.get("event_step") if kind == "exact" else annotation.get("lower_step"), "event step")
        hi = lo if kind == "exact" else integer(annotation.get("upper_step"), "upper step")
        if not lo <= hi < length:
            raise ValueError("event interval must lie inside the observed trajectory")
    else:
        end = integer(annotation.get("observed_steps"), "observed_steps", 1)
        if end > length:
            raise ValueError("censoring extends beyond observations")
    for key in ("utility", "attack_intent", "complete"):
        if annotation.get(key) is not None and type(annotation[key]) is not bool:
            raise ValueError(f"{key} must be a boolean or null")


def last_prefix(annotation, length):
    validate_annotation(annotation, length)
    kind = annotation["kind"]
    if kind == "weak":
        return -1
    if kind == "exact":
        return annotation["event_step"]
    if kind == "interval":
        return annotation["lower_step"]
    return annotation["observed_steps"] - 1


def target_for(annotation, step, horizon):
    """Return one prefix target. Interval crossing H is censored at lower-1."""
    kind = annotation["kind"]
    target = {"kind": kind, "event_observed": False, "event_time": 0,
              "event_type": -1, "censor_time": 0, "lower": 0, "upper": 0,
              "supervision": 1.0, "horizon_target": None,
              "utility": annotation.get("utility"),
              "attack_intent": annotation.get("attack_intent")}
    if kind == "right_censored":
        # A verified terminal outcome has an absorbing no-further-action state.
        target["censor_time"] = horizon if annotation.get("complete") is True else min(horizon, annotation["observed_steps"] - step)
        if target["censor_time"] == horizon:
            target["horizon_target"] = 0
        return target
    lo = (annotation["event_step"] if kind == "exact" else annotation["lower_step"]) - step + 1
    hi = (annotation["event_step"] if kind == "exact" else annotation["upper_step"]) - step + 1
    if lo < 1:
        raise ValueError("post-first-event prefixes cannot enter a first-event risk set")
    if hi > horizon:
        target["kind"] = "right_censored"
        target["censor_time"] = min(horizon, lo - 1)
        if lo > horizon:
            target["horizon_target"] = 0
    else:
        target.update(event_observed=True, event_time=lo, lower=lo, upper=hi,
                      event_type=DEFAULT_EVENT_TYPES.index(annotation["event_type"]), horizon_target=1)
    return target


def risk_at_bin(target, bin_index):
    """Known binary target at a bin; interval/censor ambiguity remains unknown."""
    if target.get("supervision",0)<=0: return None
    if target["event_observed"]:
        if target["upper"]<=bin_index: return 1
        if target["lower"]>bin_index: return 0
        return None
    return 0 if target["censor_time"]>=bin_index else None
