"""Unit checks for event-input factorial ablation; no simulation required."""
import unittest
from src.event_aware.campaigns import generate
from src.event_aware.policy import EventAwarePolicy
from examples.phase3da_ablation import VARIANTS, public_only, masked_public

class AblationTests(unittest.TestCase):
    def test_four_unique_combinations(self):
        self.assertEqual(set(VARIANTS.values()), {(True,True),(True,False),(False,True),(False,False)})

    def test_mask_removes_metadata_not_timestamps(self):
        rows, _ = generate('accurate', seed=42)
        hidden = masked_public(rows)
        self.assertTrue(all(r['event_known'] == 0 and r['planned_intensity'] == 0 for r in hidden))
        self.assertEqual([r['minute'] for r in hidden], [r['minute'] for r in rows])

    def test_event_off_equals_masked_plan_at_preparation(self):
        rows, _ = generate('accurate', seed=42)
        full = public_only(rows)
        masked = masked_public(rows)
        for guardrail in (True, False):
            off = EventAwarePolicy(full, use_event=False, use_guardrail=guardrail)
            masked_on = EventAwarePolicy(masked, use_event=True, use_guardrail=guardrail)
            for minute in range(75, 83):
                inputs = (5.8, 1, [5.7, 5.8, 5.9], {'time': minute * 60})
                self.assertEqual(off(*inputs), masked_on(*inputs))

    def test_preprescaling_affects_target_before_start(self):
        rows, _ = generate('accurate', seed=42)
        on = EventAwarePolicy(public_only(rows), use_event=True, use_guardrail=True)
        off = EventAwarePolicy(public_only(rows), use_event=False, use_guardrail=True)
        args = (6.0, 1, [6.0]*5, {'time': 78*60})
        self.assertGreater(on(*args), off(*args))

if __name__ == '__main__':
    unittest.main()
