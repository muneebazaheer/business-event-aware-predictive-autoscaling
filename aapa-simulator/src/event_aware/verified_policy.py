"""Phase 3D-C: causal verification of planned campaign demand.

Adds a post-start verification gate around the UNCHANGED Phase 3B EventAwarePolicy.
This is an experimental heuristic, not the AAPA paper's uncertainty model.
"""
import math
from src.event_aware.policy import EventAwarePolicy


class VerifiedEventAwarePolicy(EventAwarePolicy):
    """Cancel event-specific headroom after consecutive low-uplift observations.

    Observations are the rps that the simulator supplies at decision time. A
    robust pre-event baseline is captured at the first actionable pre-event tick;
    the verifier compares observed demand against that baseline and the PUBLISHED
    extra_rpm. It never reads true_event_extra_rpm.

    After cancelling event headroom, normal reactive capacity remains mandatory.
    If traffic subsequently lifts, event capacity can resume automatically.
    """
    def __init__(self, public_rows, *, verification_delay_minutes=2,
                 required_low_ticks=2, min_realized_fraction=0.25, **kwargs):
        super().__init__(public_rows, **kwargs)
        if verification_delay_minutes < 0 or required_low_ticks < 1 or not (0 < min_realized_fraction <= 1):
            raise ValueError('Invalid verifier parameters')
        self.delay = float(verification_delay_minutes)
        self.low_ticks_needed = int(required_low_ticks)
        self.min_fraction = float(min_realized_fraction)
        self._plan_key = None
        self._baseline = None
        self._low_ticks = 0
        self._cancelled = False
        self.last_verification = 'not_applicable'
        self.last_event_cancelled = 0

    def __call__(self, rps, replicas, history, context):
        requested = super().__call__(rps, replicas, history, context)
        now_minutes = float(context['time']) / 60.0
        minute = int(float(context['time']) // 60)
        row = self.public.get(minute)
        self.last_verification = 'not_applicable'
        self.last_event_cancelled = 0
        if not (self.use_event and row and int(row.get('event_known') or 0)):
            return requested
        raw_start = row.get('planned_start_minute')
        if raw_start in (None, ''):
            return requested
        start = float(raw_start)
        duration = float(row.get('planned_duration') or 0)
        published_extra = max(0.0, float(row.get('planned_intensity') or 0)) / 60.0
        if duration <= 0 or published_extra <= 0 or not (start-self.lead <= now_minutes < start+duration):
            return requested
        key = (start, duration, published_extra)
        if key != self._plan_key:
            self._plan_key = key
            self._baseline = None
            self._low_ticks = 0
            self._cancelled = False
        if now_minutes < start:
            # Store baseline before onset, without consulting future demand.
            recent = [max(0.0, float(x)) for x in history[-5:]]
            self._baseline = min(recent) if recent else max(0.0, float(rps))
            self.last_verification = 'pre_event'
            return requested
        if self._baseline is None:
            # Late announcement: historical observations are not a trustworthy
            # pre-event baseline. Conservatively decline cancellation.
            self.last_verification = 'no_pre_event_baseline'
            return requested
        observed = max(0.0, float(rps))
        uplift = observed - self._baseline
        sufficient = uplift >= published_extra * self.min_fraction
        if sufficient:
            self._low_ticks = 0
            self._cancelled = False
            self.last_verification = 'confirmed'
        elif now_minutes < start + self.delay:
            self.last_verification = 'grace_period'
        else:
            self._low_ticks += 1
            if self._low_ticks >= self.low_ticks_needed:
                self._cancelled = True
            self.last_verification = 'cancelled' if self._cancelled else 'checking'
        if not self._cancelled:
            return requested
        self.last_event_cancelled = 1
        # Preserve observed-demand floor and safety, even with a false event.
        # This is exactly the reactive expression in EventAwarePolicy.
        reactive = max(1, math.ceil(observed * self.service / self.target - 1e-12))
        self.last_reason = 'event_headroom_cancelled_reactive'
        self.last_event_extra_rps = 0.0
        self.last_event_target = 1
        return reactive
