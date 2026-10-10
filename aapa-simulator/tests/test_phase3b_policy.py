import unittest
from src.event_aware.campaigns import generate
from src.event_aware.policy import EventAwarePolicy, PUBLIC_KEYS


class EventPolicyTests(unittest.TestCase):
    def setUp(self):
        rows, _ = generate('accurate', seed=42)
        self.public = [{k: row[k] for k in PUBLIC_KEYS} for row in rows]

    def decision(self, policy, minute, rps=6):
        return policy(rps, 1, [rps] * 6, {'time': minute * 60})

    def test_announcements_are_not_early(self):
        policy = EventAwarePolicy(self.public)
        self.decision(policy, 64)
        self.assertEqual(policy.last_known_event, 0)
        self.assertEqual(policy.last_event_extra_rps, 0)

    def test_prescales_before_start(self):
        policy = EventAwarePolicy(self.public)
        normal = self.decision(policy, 76)
        prescaled = self.decision(policy, 78)
        self.assertGreater(prescaled, normal)
        self.assertEqual(policy.last_reason, 'event+guardrail')

    def test_guardrail_prevents_underprovisioning(self):
        with_guardrail = EventAwarePolicy(self.public)
        without = EventAwarePolicy(self.public, use_guardrail=False)
        # Observed load exceeds announced intensity.
        strong = self.decision(with_guardrail, 81, rps=75)
        weak = self.decision(without, 81, rps=75)
        self.assertGreaterEqual(strong, weak)

    def test_no_campaign_equivalent_reactive(self):
        rows, _ = generate('no_campaign', seed=42)
        p = EventAwarePolicy([{k: r[k] for k in PUBLIC_KEYS} for r in rows])
        self.assertEqual(self.decision(p, 80, 12), 2)

    def test_rejects_oracle_data(self):
        with self.assertRaises(ValueError):
            EventAwarePolicy([{'minute': 80, 'true_event_extra_rpm': 999}])

    def test_false_positive_event_is_seen_as_plan(self):
        rows, _ = generate('false_positive', seed=42)
        p = EventAwarePolicy([{k: r[k] for k in PUBLIC_KEYS} for r in rows])
        self.assertGreater(self.decision(p, 78), self.decision(p, 70))


if __name__ == '__main__':
    unittest.main()
