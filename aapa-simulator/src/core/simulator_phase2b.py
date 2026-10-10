"""Phase 2B auditable reconstruction. Not the paper's original simulator."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import simpy
from src.core.simulator_repaired import RepairedSimulator


class AuditedSimulator(RepairedSimulator):
    """Adds precise readiness and request-deadline accounting to Phase 2A."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_records = {}
        self.all_arrivals = []
        self.decision_log = []
        self._last_requested = None
        self._retired_busy = 0

    def _producer(self, events):
        for ident, (arrival, service) in enumerate(events):
            yield self.env.timeout(max(0, arrival - self.env.now))
            t = float(self.env.now)
            self.request_records[ident] = {'arrival': t, 'service': float(service),
                                           'completion': None, 'status': 'queued'}
            self.metrics.generated += 1
            self.arrivals.append(t)
            self.all_arrivals.append(t)
            yield self.queue.put((ident, t, float(service)))

    def _worker(self, pod):
        pod.ready = False
        try:
            yield self.env.timeout(self.startup_seconds)
            if not pod.live:
                return
            pod.ready = True
            while pod.live:
                pod.busy = False
                ident, arrival, service = yield self.queue.get()
                if not pod.live:
                    yield self.queue.put((ident, arrival, service))
                    return
                pod.busy = True
                self.request_records[ident]['status'] = 'running'
                yield self.env.timeout(service)
                self.request_records[ident]['completion'] = float(self.env.now)
                self.request_records[ident]['status'] = 'completed'
                elapsed = self.env.now - arrival
                self.metrics.completed += 1
                self.metrics.response_times.append(elapsed)
                if elapsed > self.slo:
                    self.metrics.missed_deadline += 1
        except simpy.Interrupt:
            # Phase 2A only interrupts workers that are not busy.
            if pod.busy:
                raise RuntimeError('A busy worker was interrupted')
        finally:
            pod.ready = False
            pod.busy = False

    def _controller(self, duration):
        while self.env.now < duration:
            now = self.env.now
            while self.arrivals and self.arrivals[0] <= now - 60:
                self.arrivals.popleft()
            rate = len(self.arrivals) / 60.0
            self.history.append(rate)
            context = {'time': now}
            if hasattr(self.policy, 'features_at'):
                context['features'] = self.policy.features_at(now, self.all_arrivals)
            current = len(self.pods)
            target = int(self.policy(rate, current, list(self.history), context))
            self._change_replicas(target)
            allocated = len(self.pods)
            ready = sum(bool(getattr(p, 'ready', False)) for p in self.pods)
            busy = sum(p.busy for p in self.pods)
            row = {'time_s': now, 'observed_rps': rate, 'requested_replicas': target,
                   'allocated_replicas': allocated, 'ready_replicas': ready,
                   'busy_replicas': busy, 'queue_length': len(self.queue.items)}
            if hasattr(self.policy, 'last_archetype'):
                row['archetype'] = self.policy.last_archetype
                row['confidence'] = self.policy.last_confidence
            self.metrics.timeline.append(row)
            yield self.env.timeout(self.scale_interval)

    def run(self, events, duration_seconds):
        if any(t < 0 or t >= duration_seconds or s < 0 for t, s in events):
            raise ValueError('Events must fall inside measurement window; service times nonnegative')
        super().run(events, duration_seconds)
        total = self.metrics.generated
        completed = self.metrics.completed
        outstanding = total - completed
        overdue = sum(1 for r in self.request_records.values()
                      if r['completion'] is None and duration_seconds - r['arrival'] > self.slo)
        pending_within_slo = outstanding - overdue
        observed_violations = self.metrics.missed_deadline + overdue
        result = {
            'generated': total,
            'completed': completed,
            'unfinished_at_end': outstanding,
            'unfinished_already_over_slo': overdue,
            'unfinished_within_slo': pending_within_slo,
            'completed_missed_slo': self.metrics.missed_deadline,
            'confirmed_slo_violations': observed_violations,
            'confirmed_slo_violation_rate': observed_violations / total if total else 0.0,
            'worst_case_slo_violation_rate': (observed_violations + pending_within_slo) / total if total else 0.0,
            'mean_response_seconds_completed': float(np.mean(self.metrics.response_times)) if completed else None,
            'p95_response_seconds_completed': float(np.percentile(self.metrics.response_times, 95)) if completed else None,
            'p99_response_seconds_completed': float(np.percentile(self.metrics.response_times, 99)) if completed else None,
            'pod_minutes': self.metrics.pod_seconds / 60.0,
            'scaling_actions': self.metrics.scaling_actions,
            'max_allocated_replicas': max((r['allocated_replicas'] for r in self.metrics.timeline), default=0),
            'max_ready_replicas': max((r['ready_replicas'] for r in self.metrics.timeline), default=0),
        }
        assert completed + outstanding == total
        assert observed_violations + pending_within_slo <= total
        return result


class RepoAAPAAdapter:
    """Repo AAPA strategy with causal 60-minute features; *not* capacity normalized.

    Uses original classifier/strategy formulas. The trial remains exploratory since
    baseline normalized models and AAPA use different resource-load assumptions.
    """
    def __init__(self, model_path, encoder_path, warmup_minutes=60):
        from src.core.simulator import AAPAAutoscaler
        from src.data.features import TimeSeriesFeatureExtractor
        self.aapa = AAPAAutoscaler(str(model_path), str(encoder_path))
        self.extractor = TimeSeriesFeatureExtractor()
        # LightGBM stores placeholder names Column_0..Column_36 for this model.
        # Recover the original training column order from Phase 1 metadata.
        import yaml
        from pathlib import Path
        metadata_path = Path(model_path).resolve().parents[1] / 'phase1_100' / 'model_metadata.yaml'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Missing Phase 1 training feature-order metadata: {metadata_path}')
        metadata = yaml.load(metadata_path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
        self.feature_names = metadata['feature_names']
        if self.aapa.model.num_feature() != len(self.feature_names):
            raise ValueError('Feature count does not match the saved LightGBM model')
        for label, idx in metadata['label_maps']['lightgbm_label_map'].items():
            actual = str(self.aapa.label_encoder.inverse_transform([int(idx)])[0])
            if actual != label:
                raise ValueError(f'Class {idx} maps to {actual}, expected {label}')
        if len(self.feature_names) != 37:
            raise ValueError('Expected 37 LightGBM features; got ' + str(len(self.feature_names)))
        self.warmup_minutes = warmup_minutes
        self.last_archetype = 'WARMUP'
        self.last_confidence = 0.0
        self._latest_features = None

    def features_at(self, now, arrivals):
        # Use only fully completed minutes [now-60*60, now), never the future.
        minute_now = int(now // 60)
        if minute_now < self.warmup_minutes:
            self._latest_features = None
            return None
        edge = minute_now * 60
        ts = np.zeros(60, dtype=float)
        for t in arrivals:
            if edge - 3600 <= t < edge:
                idx = int((t - (edge - 3600)) // 60)
                if 0 <= idx < 60:
                    ts[idx] += 1
        d = self.extractor.extract_features(ts)
        missing = [name for name in self.feature_names if name not in d]
        if missing:
            raise ValueError('Missing model features: ' + ', '.join(missing))
        vector = np.asarray([d[name] for name in self.feature_names], dtype=float)
        self._latest_features = np.nan_to_num(vector, nan=0, posinf=0, neginf=0)
        return self._latest_features

    def __call__(self, rps, replicas, history, context):
        if self._latest_features is None:
            # Warm-up: no usable 60-min observation exists; use simple fallback.
            # This is a documented experimental choice, not the original AAPA design.
            self.last_archetype = 'WARMUP'
            self.last_confidence = 0.0
            return max(1, int(np.ceil(rps * 0.1 / 0.7)))
        context = dict(context, features=self._latest_features)
        desired = self.aapa.get_desired_replicas(rps, replicas, history, context)
        self.last_archetype = self.aapa.current_archetype
        self.last_confidence = float(self.aapa.confidence)
        return desired
