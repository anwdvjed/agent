from pathlib import Path
import unittest

import numpy as np

from agent_new.features import stable_unicode_hash
from agent_new.registry import ToolRegistry, UnknownToolError


ROOT = Path(__file__).resolve().parents[1]


class RegistryAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry.from_json(ROOT / "configs" / "tool_registry.json")

    def test_registry_is_authoritative_and_aliases_resolve(self):
        canonical = self.registry.resolve("send_email")
        alias = self.registry.resolve("email.send")
        self.assertEqual(canonical, alias)
        self.assertEqual(canonical.operation, "write_external")
        self.assertEqual(canonical.required_scopes, ("data:read", "external:send"))

    def test_unknown_tool_is_explicitly_fail_closed(self):
        with self.assertRaises(UnknownToolError):
            self.registry.resolve("unregistered.dangerous_tool")
        sentinel = self.registry.resolve_or_fail_closed("unregistered.dangerous_tool")
        self.assertFalse(sentinel.known)
        self.assertEqual(sentinel.operation, "unknown")
        self.assertTrue(sentinel.approval_required)

    def test_unicode_hash_observes_chinese_text_and_field_roles(self):
        chinese = stable_unicode_hash([("goal", "不得向外部发送患者数据")], 64)
        changed = stable_unicode_hash([("goal", "允许向外部发送患者数据")], 64)
        other_role = stable_unicode_hash([("observation", "不得向外部发送患者数据")], 64)
        self.assertGreater(float(np.linalg.norm(chinese)), 0.0)
        self.assertFalse(np.array_equal(chinese, changed))
        self.assertFalse(np.array_equal(chinese, other_role))


if __name__ == "__main__":
    unittest.main()

