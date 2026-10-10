"""Capacity-aligned reconstructed autoscaling policies; NOT original AAPA paper policies."""
import math
import numpy as np
from src.core.simulator_repaired import NormalizedHPA, NormalizedPredictive
from src.core.simulator_phase2b import RepoAAPAAdapter

class ArchetypePolicy:
    """Uses the Phase 1 classifier; shared service capacity with normalized baselines.

    Ablations: mode='full', 'no_confidence', 'fixed_spike', 'fixed_stationary'.
    These use engineered response rules and are NOT faithful repo-AAPA or uncertainty UQ.
    """
    def __init__(self, model, encoder, mode='full', target_util=0.7,
                 mean_service=0.1, confidence_threshold=0.7):
        self.adapter = RepoAAPAAdapter(model, encoder)
        self.mode = mode
        self.target = target_util
        self.service = mean_service
        self.threshold = confidence_threshold
        self.last_archetype = 'WARMUP'
        self.last_confidence = 0.0
        self.predictions_by_archetype = {}

    def features_at(self, now, arrivals):
        return self.adapter.features_at(now, arrivals)

    def __call__(self, rps, replicas, history, context):
        features = self.adapter._latest_features
        if features is None:
            self.last_archetype = 'WARMUP'
            self.last_confidence = 0.0
            return max(1, math.ceil(rps * self.service / self.target))
        predicted, confidence = self.adapter.aapa.classify_workload(features)
        if self.mode == 'fixed_spike':
            archetype = 'SPIKE'
        elif self.mode == 'fixed_stationary':
            archetype = 'STATIONARY_NOISY'
        elif self.mode == 'full' and confidence < self.threshold:
            archetype = 'STATIONARY_NOISY'
        else:
            archetype = predicted
        self.last_archetype = archetype
        self.last_confidence = float(confidence)
        self.predictions_by_archetype[archetype] = self.predictions_by_archetype.get(archetype, 0) + 1
        recent = np.asarray(history[-6:], dtype=float)
        forecast = float(rps)
        if len(recent) >= 3 and archetype in ('RAMP', 'PERIODIC'):
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            # Bounded positive trend, based ONLY on historical observations.
            forecast = max(rps, rps + max(0, slope) * 2)
        # Every strategy uses the same nominal per-replica service capacity.
        base = max(1, math.ceil(forecast * self.service / self.target))
        headroom = {'SPIKE': 3, 'PERIODIC': 1, 'RAMP': 1, 'STATIONARY_NOISY': 0}
        return base + headroom[archetype]

__all__ = ['ArchetypePolicy', 'NormalizedHPA', 'NormalizedPredictive']
