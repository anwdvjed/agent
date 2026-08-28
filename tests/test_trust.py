import copy
import time
import unittest

from agent_new.trust import HMACRequestVerifier, sign_request


class TrustedContextTests(unittest.TestCase):
    def test_signed_request_verifies_once_and_detects_tampering(self):
        secret = b"unit-test-secret"
        request = {"trusted_goal": "safe", "candidate": {"action": "read"}}
        signed = sign_request(
            request,
            secret,
            issuer="runtime-policy-engine",
            subject="agent-1",
            nonce="nonce-1",
            lifetime_seconds=60,
        )
        verifier = HMACRequestVerifier(secret, ["runtime-policy-engine"])
        self.assertTrue(verifier(signed))
        self.assertFalse(verifier(signed), "nonce replay must fail")
        tampered = copy.deepcopy(signed)
        tampered["candidate"]["action"] = "delete"
        fresh_verifier = HMACRequestVerifier(secret, ["runtime-policy-engine"])
        self.assertFalse(fresh_verifier(tampered))

    def test_expired_or_wrong_issuer_attestation_fails(self):
        secret = b"unit-test-secret"
        signed = sign_request(
            {"value": 1},
            secret,
            issuer="wrong",
            subject="agent",
            nonce="nonce-2",
            lifetime_seconds=1,
            now=time.time() - 100,
        )
        verifier = HMACRequestVerifier(secret, ["trusted"])
        self.assertFalse(verifier(signed))


if __name__ == "__main__":
    unittest.main()

