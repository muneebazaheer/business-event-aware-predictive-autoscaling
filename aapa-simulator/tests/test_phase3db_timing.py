"""Causal announcement visibility regression tests (no SimPy runs required)."""
import unittest
from src.event_aware.campaigns import generate
from examples.phase3db_timing import public_schedule, LEADS, CAMPAIGN_START

class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = generate('accurate', seed=42)

    def test_15min_announcement_visible_early(self):
        pub = public_schedule(self.rows, LEADS['15min'])
        self.assertEqual(pub[64]['event_known'],0)
        self.assertEqual(pub[65]['event_known'],1)

    def test_two_minute_announcement(self):
        pub = public_schedule(self.rows, LEADS['2min'])
        self.assertEqual(pub[77]['event_known'],0)
        self.assertEqual(pub[78]['event_known'],1)

    def test_30sec_notice_not_visible_at_prior_tick(self):
        pub = public_schedule(self.rows, LEADS['30sec'])
        self.assertEqual(pub[79]['event_known'],0)
        self.assertEqual(pub[80]['event_known'],1)

    def test_notice_at_onset(self):
        pub = public_schedule(self.rows, LEADS['onset'])
        self.assertEqual(pub[79]['event_known'],0)
        self.assertEqual(pub[80]['event_known'],1)

    def test_after_start(self):
        pub = public_schedule(self.rows, LEADS['1min_after'])
        self.assertEqual(pub[80]['event_known'],0)
        self.assertEqual(pub[81]['event_known'],1)

    def test_no_notice_oracle_safe(self):
        pub = public_schedule(self.rows, LEADS['no_notice'])
        self.assertTrue(all(p['event_known']==0 for p in pub))
        self.assertTrue(all('true_event_extra_rpm' not in p and 'demand_rpm' not in p for p in pub))

    def test_demand_truth_unchanged_across_notice(self):
        a=public_schedule(self.rows,LEADS['15min'])
        b=public_schedule(self.rows,LEADS['1min_after'])
        self.assertEqual(len(a),len(b))
        self.assertEqual(CAMPAIGN_START,80)
        self.assertTrue(all('demand_rpm' not in p and 'true_event_extra_rpm' not in p for p in a+b))
        self.assertEqual([p['minute'] for p in a], [p['minute'] for p in b])

if __name__=='__main__': unittest.main()
