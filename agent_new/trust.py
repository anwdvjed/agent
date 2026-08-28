"""Signed request attestations for trusted pre-execution context."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableSet, Optional, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def request_payload(request: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(request))
    value.pop("attestation", None)
    return value


def request_payload_digest(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(request_payload(request))).hexdigest()


def _message(attestation: Mapping[str, Any]) -> bytes:
    fields = {
        key: attestation[key]
        for key in (
            "issuer",
            "subject",
            "issued_at",
            "expires_at",
            "nonce",
            "payload_sha256",
        )
    }
    return _canonical(fields)


def sign_request(
    request: Mapping[str, Any],
    secret: bytes,
    *,
    issuer: str,
    subject: str,
    nonce: str,
    lifetime_seconds: float = 60.0,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    if not secret:
        raise ValueError("attestation secret must not be empty")
    issued_at = float(time.time() if now is None else now)
    if not math.isfinite(issued_at) or lifetime_seconds <= 0.0:
        raise ValueError("invalid attestation time")
    value = copy.deepcopy(dict(request))
    attestation = {
        "issuer": str(issuer),
        "subject": str(subject),
        "issued_at": issued_at,
        "expires_at": issued_at + float(lifetime_seconds),
        "nonce": str(nonce),
        "payload_sha256": request_payload_digest(value),
    }
    attestation["signature"] = hmac.new(
        secret, _message(attestation), hashlib.sha256
    ).hexdigest()
    value["attestation"] = attestation
    return value


@dataclass
class HMACRequestVerifier:
    secret: bytes
    allowed_issuers: Sequence[str]
    clock_skew_seconds: float = 5.0
    used_nonces: MutableSet[str] = field(default_factory=set)

    def __call__(self, request: Mapping[str, Any]) -> bool:
        if not self.secret:
            return False
        attestation = request.get("attestation")
        if not isinstance(attestation, Mapping):
            return False
        required = {
            "issuer",
            "subject",
            "issued_at",
            "expires_at",
            "nonce",
            "payload_sha256",
            "signature",
        }
        if required.difference(attestation):
            return False
        issuer = str(attestation["issuer"])
        nonce = str(attestation["nonce"])
        if issuer not in set(map(str, self.allowed_issuers)) or not nonce:
            return False
        if nonce in self.used_nonces:
            return False
        try:
            issued_at = float(attestation["issued_at"])
            expires_at = float(attestation["expires_at"])
        except (TypeError, ValueError):
            return False
        now = time.time()
        if (
            not math.isfinite(issued_at)
            or not math.isfinite(expires_at)
            or issued_at > now + self.clock_skew_seconds
            or expires_at < now - self.clock_skew_seconds
            or expires_at <= issued_at
        ):
            return False
        if str(attestation["payload_sha256"]) != request_payload_digest(request):
            return False
        expected = hmac.new(self.secret, _message(attestation), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(attestation["signature"])):
            return False
        self.used_nonces.add(nonce)
        return True


__all__ = [
    "HMACRequestVerifier",
    "request_payload",
    "request_payload_digest",
    "sign_request",
]

