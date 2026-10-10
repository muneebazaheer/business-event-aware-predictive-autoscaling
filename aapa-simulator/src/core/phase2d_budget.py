"""Phase 2D resource-matching utilities for reconstructed AAPA studies.

This is an experiment-layer policy intensity adjustment, NOT original AAPA.
"""
import math
from src.core.simulator_phase2b import AuditedSimulator

class IntensityPolicy:
    """Scale the *additional* requested capacity above the one-pod floor.

    A value of 1 leaves the underlying policy unchanged. We use the same
    transformation for every policy. This is NOT a simulated monetary cap.
    """
    def __init__(self, base_policy, factor):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('factor must be positive and finite')
        self.base = base_policy
        self.factor = float(factor)

    def features_at(self, now, arrivals):
        if hasattr(self.base, 'features_at'):
            return self.base.features_at(now, arrivals)
        return None

    def __call__(self, rps, replicas, history, context):
        desired = max(1, int(self.base(rps, replicas, history, context)))
        return max(1, int(math.ceil(1 + (desired - 1) * self.factor - 1e-10)))

    @property
    def last_archetype(self):
        return getattr(self.base, 'last_archetype', 'N/A')

    @property
    def last_confidence(self):
        return getattr(self.base, 'last_confidence', 0.0)


class ResourceAuditSimulator(AuditedSimulator):
    """Accurately integrate allocated pod-minutes within the evaluation interval."""
    def __init__(self, *args, eval_start_seconds=3600, eval_end_seconds=6600, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_start_seconds = float(eval_start_seconds)
        self.eval_end_seconds = float(eval_end_seconds)
        self.eval_pod_seconds = 0.0

    def _integrate(self):
        left = max(self._last_integral_time, self.eval_start_seconds)
        right = min(float(self.env.now), self.eval_end_seconds)
        if right > left:
            self.eval_pod_seconds += len(self.pods) * (right - left)
        super()._integrate()
