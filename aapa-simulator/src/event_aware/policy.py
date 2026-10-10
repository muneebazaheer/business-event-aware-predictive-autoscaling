"""Phase 3B reconstructed event-aware autoscaling policy.

Uses ONLY published event columns and observed demand. Not original AAPA.
Each call's `history` may include the current observed rate; never future demand.
"""
import math

PUBLIC_KEYS = ('minute', 'event_known', 'event_active', 'minutes_to_event',
               'event_type', 'planned_intensity', 'planned_duration', 'planned_start_minute')


class EventAwarePolicy:
    def __init__(self, public_rows, *, mean_service_seconds=0.1,
                 target_utilization=0.7, startup_seconds=30,
                 decision_interval_seconds=60, lead_minutes=2,
                 max_event_extra_rpm=10000, use_event=True, use_guardrail=True):
        self.public = {}
        for row in public_rows:
            if set(row) - set(PUBLIC_KEYS):
                raise ValueError('Event policy input must contain public metadata only; unexpected keys: '
                                 + ', '.join(sorted(set(row) - set(PUBLIC_KEYS))))
            t = int(row['minute'])
            if t in self.public:
                raise ValueError('Duplicate minute in published context')
            self.public[t] = dict(row)
        if mean_service_seconds <= 0 or not 0 < target_utilization <= 1:
            raise ValueError('Invalid capacity assumptions')
        if startup_seconds < 0 or decision_interval_seconds <= 0 or lead_minutes < 0:
            raise ValueError('Invalid schedule assumptions')
        self.service = float(mean_service_seconds)
        self.target = float(target_utilization)
        self.startup = float(startup_seconds)
        self.interval = float(decision_interval_seconds)
        self.lead = float(lead_minutes)
        self.max_extra = float(max_event_extra_rpm)
        self.use_event = bool(use_event)
        self.use_guardrail = bool(use_guardrail)
        self.last_reason = 'initial'
        self.last_known_event = 0
        self.last_event_extra_rps = 0.0
        self.last_observed_floor = 1
        self.last_event_target = 1

    def __call__(self, rps, replicas, history, context):
        now = float(context['time'])
        minute = int(now // 60)
        observed = max(0.0, float(rps))
        # All policies share the same nominal service capacity.
        reactive = max(1, math.ceil(observed * self.service / self.target - 1e-12))
        self.last_observed_floor = reactive
        event_target = 1
        extra = 0.0
        published = self.public.get(minute)
        known = bool(published and int(published['event_known']))
        self.last_known_event = int(known)
        if self.use_event and known:
            start_raw = published.get('planned_start_minute')
            duration = int(float(published.get('planned_duration') or 0))
            if start_raw not in (None, '') and duration > 0:
                start = int(float(start_raw))
                end = start + duration
                # Advance provision only within actionable lead window; ensure
                # a newly allocated pod can become ready by scheduled start.
                readiness_lead = self.startup / 60.0 + self.interval / 60.0
                effective_lead = max(self.lead, readiness_lead)
                if start - effective_lead <= minute < end:
                    extra = min(self.max_extra, max(0.0, float(published.get('planned_intensity') or 0))) / 60.0
                    # Today's background estimate is purely observed; use bounded
                    # observed history, not future truth or actual campaign lift.
                    recent = [max(0.0, float(x)) for x in history[-5:]]
                    baseline = min(recent) if recent else observed
                    expected = baseline + extra
                    event_target = max(1, math.ceil(expected * self.service / self.target - 1e-12))
        self.last_event_extra_rps = extra
        self.last_event_target = event_target
        if self.use_guardrail:
            desired = max(reactive, event_target)
        else:
            # Ablation uses event plan if actionable, else observed-load policy.
            # With an event, it DOES NOT override a low forecast using reactive load.
            desired = event_target if extra > 0 else reactive
        self.last_reason = ('event+guardrail' if extra > 0 and self.use_guardrail else
                            'event_only' if extra > 0 else 'reactive')
        return desired


class DecisionLoggingPolicy:
    """Optional wrapper recording the decisions requested by a policy."""
    def __init__(self, base):
        self.base = base
        self.decisions = []

    def __call__(self, rps, replicas, history, context):
        target = self.base(rps, replicas, history, context)
        self.decisions.append(dict(time_s=float(context['time']), observed_rps=float(rps),
                                   current_replicas=int(replicas), target_replicas=int(target),
                                   reason=self.base.last_reason,
                                   event_known=self.base.last_known_event,
                                   planned_extra_rps=self.base.last_event_extra_rps,
                                   reactive_floor=self.base.last_observed_floor,
                                   event_target=self.base.last_event_target))
        return target
