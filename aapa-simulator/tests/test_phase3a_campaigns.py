import unittest
from src.event_aware.campaigns import SCENARIOS, generate, public_metadata, campaign_for


class CampaignTests(unittest.TestCase):
    def test_determinism(self):
        for s in SCENARIOS:
            self.assertEqual(generate(s, 42), generate(s, 42))

    def test_not_visible_before_announcement(self):
        for s in SCENARIOS:
            c = campaign_for(s)
            if c and c.announced_minute is not None:
                for m in range(c.announced_minute):
                    x = public_metadata(c, m)
                    self.assertEqual(x['event_known'], 0)
                    self.assertEqual(x['planned_intensity'], 0)
                    self.assertIsNone(x['minutes_to_event'])

    def test_unannounced_event_not_revealed(self):
        rows, _ = generate('unannounced')
        self.assertTrue(all(x['event_known'] == 0 for x in rows))
        self.assertTrue(any(x['true_event_extra_rpm'] > 0 for x in rows))

    def test_false_positive_event(self):
        rows, _ = generate('false_positive')
        self.assertTrue(any(x['event_known'] == 1 for x in rows))
        self.assertTrue(all(x['true_event_extra_rpm'] == 0 for x in rows))

    def test_baselines_same_across_scenarios(self):
        a, _ = generate('accurate', 43)
        b, _ = generate('underestimated', 43)
        self.assertEqual([r['baseline_rpm'] for r in a], [r['baseline_rpm'] for r in b])

    def test_event_fields_do_not_depend_on_truth(self):
        a, _ = generate('accurate')
        b, _ = generate('underestimated')
        cols = ('event_known', 'event_active', 'minutes_to_event',
                'event_type', 'planned_intensity', 'planned_duration', 'planned_start_minute')
        self.assertEqual([tuple(r[k] for k in cols) for r in a],
                         [tuple(r[k] for k in cols) for r in b])

    def test_phases_and_plan(self):
        rows, _ = generate('accurate')
        self.assertEqual(len(rows), 110)
        self.assertEqual(rows[64]['event_known'], 0)
        self.assertEqual(rows[65]['minutes_to_event'], 15)
        self.assertEqual(rows[80]['event_active'], 1)
        self.assertEqual(rows[90]['event_active'], 0)


if __name__ == '__main__':
    unittest.main()
