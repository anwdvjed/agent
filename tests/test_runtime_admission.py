from dataclasses import asdict
from agent1.risk_window import WindowConfig
"""Test-only ensemble and attestation fixtures; never production validators."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

from agent1.calibration import fit_policy_calibration
from agent1.checkpoint import canonical_digest
from agent1.data import prepare_request
from agent1.environments import DocumentEnvironment
from agent1.runtime import SafetyRuntime, SELECTOR_VERSION
from agent1.trust import sign_request, HMACRequestVerifier
from test_policy_v2 import forecast, policy


class FixedEnsembleForTest:
    independent = True

    def __init__(self, registry):
        self.registry = registry
        self.registry_digest = canonical_digest(registry.to_dict())

    def lineage(self, environment):
        return {"ensemble_digest": "unit-test", "registry_digest": self.registry_digest,
                "environment": environment, "selector_version": SELECTOR_VERSION,
                "training_manifest_digest": "test-training", "forecast_contract_version": "agent1-forecast-v2",
                "prediction_head_contract_digest": "test-heads", "selector_implementation_digest": "test-code",
                "executor_contract_version": "test-gateway", "window_config_digest": canonical_digest(asdict(WindowConfig()))}

    def predict(self, requests):
        result = {}
        for name, request in requests.items():
            branch = prepare_request(request, self.registry, interventions=("none",)).branches["none"]
            value = forecast([.01] * 5)
            value.update(branch_id=name, candidate_event_id=request["candidate"]["event_id"],
                         prepared_digest=branch.prepared_action_digest, hard_failure=branch.metadata["fail_closed"],
                         hard_reasons=branch.metadata["fail_reasons"])
            result[name] = value
        return result


def make_runtime(registry, *, current_context=None, artifact_verifier=None):
    ensemble = FixedEnsembleForTest(registry)
    frozen = policy()
    rows = [{"policy_id": canonical_digest(frozen), "episode_id": str(i), "group_id": str(i),
             "split": "calibration", "completed": True, "unsafe": False, "utility": True, "benign": True}
            for i in range(100)]
    extra = {"repair_provider_version": "test-runner", "policy_family_digest": canonical_digest([frozen]),
             "calibration_manifest_digest": canonical_digest(sorted({(r["episode_id"], r["group_id"]) for r in rows})),
             "development_groups": []}
    calibration = fit_policy_calibration(rows, [frozen], {**ensemble.lineage("test-env"), **extra})
    assert calibration["status"] == "certified"
    verifier = HMACRequestVerifier(b"test", ["test-issuer"])
    context = current_context or (lambda request: {"valid": True, "revision": "r1", "expires_at": time.time() + 60})
    return SafetyRuntime(ensemble, calibration, "test-env", verifier,
                         artifact_verifier=artifact_verifier or (lambda artifact: True),
                         current_context_verifier=context, deployment_lineage=extra)


def signed(request, nonce="one"):
    return sign_request(request, b"test", issuer="test-issuer", subject="test-agent", nonce=nonce)


def test_signed_allow_single_use_permit_and_modified_action_rejection(registry):
    runtime = make_runtime(registry)
    request = signed(DocumentEnvironment({"attack": False}, registry).request())
    response = runtime.assess(request)
    assert response["decision"] == "ALLOW"
    permit_id = response["commit"]["permit_id"]
    changed = copy.deepcopy(request)
    changed["candidate"]["action"]["arguments"]["path"] = "/other"
    assert runtime.authorize_and_consume(changed, permit_id) is None
    # A changed request revokes the old permit; evaluate the original again with fresh attestation.
    again = runtime.assess(signed(DocumentEnvironment({"attack": False}, registry).request(), "two"))
    assert again["decision"] == "ALLOW"
    action = runtime.authorize_and_consume(signed(DocumentEnvironment({"attack": False}, registry).request(), "two"), again["commit"]["permit_id"])
    assert action is not None and action.selected_branch == "none"
    assert runtime.authorize_and_consume(request, again["commit"]["permit_id"]) is None


def test_missing_attestation_and_unapproved_tool_do_not_allow(registry):
    runtime = make_runtime(registry)
    request = DocumentEnvironment({"attack": False}, registry).request()
    assert runtime.assess(request)["decision"] == "HOLD"
    request["candidate"]["action"]["tool"] = "unknown"
    assert runtime.assess(signed(request))["decision"] == "BLOCK"


def test_untrusted_artifact_keeps_hold(registry):
    runtime = make_runtime(registry, artifact_verifier=lambda artifact: False)
    request = signed(DocumentEnvironment({"attack": False}, registry).request())
    assert runtime.assess(request)["reason_codes"] == ["NO_CERTIFIED_POLICY"]


def test_reassessment_revokes_old_permit(registry):
    runtime = make_runtime(registry)
    original = DocumentEnvironment({"attack": False}, registry).request()
    first = signed(original, "first")
    permit = runtime.assess(first)["commit"]["permit_id"]
    second = signed(original, "second")
    assert runtime.assess(second)["decision"] == "ALLOW"
    assert runtime.authorize_and_consume(first, permit) is None


def test_context_revision_revokes_and_concurrent_consume_is_single_use(registry):
    revision = ["r1"]
    runtime = make_runtime(registry, current_context=lambda request: {"valid": True, "revision": revision[0], "expires_at": time.time() + 60})
    original = DocumentEnvironment({"attack": False}, registry).request()
    request = signed(original)
    permit = runtime.assess(request)["commit"]["permit_id"]
    revision[0] = "r2"
    assert runtime.authorize_and_consume(request, permit) is None
    response = runtime.assess(signed(original, "next"))
    second = signed(original, "next")
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: runtime.authorize_and_consume(second, response["commit"]["permit_id"]), range(2)))
    assert sum(value is not None for value in outcomes) == 1


def test_nonfinite_attestation_expiry_and_no_live_verifier_hold(registry):
    runtime = make_runtime(registry)
    request = signed(DocumentEnvironment({"attack": False}, registry).request())
    request["attestation"]["expires_at"] = float("nan")
    assert runtime.assess(request)["decision"] == "HOLD"
    missing = make_runtime(registry)
    missing.current_context_verifier = None
    assert missing.assess(signed(DocumentEnvironment({"attack": False}, registry).request(), "fresh"))["decision"] == "HOLD"
