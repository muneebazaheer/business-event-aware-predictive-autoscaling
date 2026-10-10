import unittest
from src.event_aware.campaigns import generate
from src.event_aware.policy import PUBLIC_KEYS, EventAwarePolicy
from src.event_aware.verified_policy import VerifiedEventAwarePolicy


def make(scenario):
    rows,_=generate(scenario,seed=42)
    return [{k:r[k] for k in PUBLIC_KEYS} for r in rows]


def decision(p, minute, rps=6):
    return p(rps,1,[6.0]*5,{'time':minute*60})


class VerifiedTests(unittest.TestCase):
    def test_pre_event_identical(self):
        pub=make('false_positive')
        original=EventAwarePolicy(pub)
        verified=VerifiedEventAwarePolicy(pub)
        self.assertEqual(decision(original,78),decision(verified,78))

    def test_false_positive_cancelled_after_grace(self):
        p=VerifiedEventAwarePolicy(make('false_positive'))
        decision(p,78)
        decision(p,80)
        decision(p,81)
        decision(p,82)
        self.assertEqual(p.last_event_cancelled,0)
        decision(p,83)
        self.assertEqual(p.last_event_cancelled,1)
        self.assertEqual(p.last_reason,'event_headroom_cancelled_reactive')

    def test_accurate_campaign_preserves_headroom(self):
        p=VerifiedEventAwarePolicy(make('accurate'))
        decision(p,78)
        for m in (80,81,82,83,84):
            decision(p,m,26 if m>=81 else 6)
        self.assertEqual(p.last_event_cancelled,0)

    def test_underestimated_still_has_reactive_floor(self):
        p=VerifiedEventAwarePolicy(make('underestimated'))
        decision(p,78)
        for m in (80,81,82,83):
            result=decision(p,m,45)
        self.assertGreaterEqual(result,p.last_observed_floor)

    def test_delayed_surged_recovers_after_cancellation(self):
        p=VerifiedEventAwarePolicy(make('false_positive'))
        decision(p,78)
        for m in (80,81,82,83):
            decision(p,m,6)
        self.assertEqual(p.last_event_cancelled,1)
        decision(p,84,26)
        self.assertEqual(p.last_event_cancelled,0)
        self.assertEqual(p.last_verification,'confirmed')

    def test_no_announcement_equivalent(self):
        p=VerifiedEventAwarePolicy(make('no_campaign'))
        q=EventAwarePolicy(make('no_campaign'))
        for minute in (60,78,81,89,95):
            self.assertEqual(decision(p,minute),decision(q,minute))

    def test_rejects_oracle_keys(self):
        with self.assertRaises(ValueError):
            VerifiedEventAwarePolicy([{'minute':80,'true_event_extra_rpm':1200}])

    def test_invalid_configuration(self):
        for opts in ({'verification_delay_minutes':-1},{'required_low_ticks':0},
                     {'min_realized_fraction':0},{'min_realized_fraction':1.1}):
            with self.assertRaises(ValueError):
                VerifiedEventAwarePolicy(make('accurate'), **opts)


if __name__=='__main__':
    unittest.main()
