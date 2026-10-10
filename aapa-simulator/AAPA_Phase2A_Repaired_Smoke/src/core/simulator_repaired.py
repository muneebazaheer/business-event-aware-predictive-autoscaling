"""Experimental reconstruction of an autoscaling simulation engine.

NOT the original AAPA paper simulator. Designed for auditable, comparable trials.
Input workload units: arrivals per minute; decisions: observed arrivals per second.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional
import math
import numpy as np
import simpy


@dataclass
class TrialMetrics:
    generated: int = 0
    completed: int = 0
    missed_deadline: int = 0
    unfinished_at_end: int = 0
    pod_seconds: float = 0.0
    scaling_actions: int = 0
    response_times: list = field(default_factory=list)
    timeline: list = field(default_factory=list)

    def as_dict(self):
        return {
            'generated': self.generated,
            'completed': self.completed,
            'unfinished_at_end': self.unfinished_at_end,
            'slo_violations_including_unfinished': self.missed_deadline + self.unfinished_at_end,
            'slo_violation_rate_including_unfinished':
                (self.missed_deadline + self.unfinished_at_end) / self.generated if self.generated else 0.0,
            'mean_response_seconds_completed': float(np.mean(self.response_times)) if self.response_times else None,
            'p99_response_seconds_completed': float(np.percentile(self.response_times, 99)) if self.response_times else None,
            'pod_minutes': self.pod_seconds / 60.0,
            'scaling_actions': self.scaling_actions,
        }


@dataclass
class _Pod:
    number: int
    live: bool = True
    busy: bool = False
    process: object = None


class RepairedSimulator:
    """One SimPy worker per replica. Changes to replica count change concurrency.

    Policy callable: (requests_per_second, replica_count, list_of_prior_rates,
                      {'time': seconds}) -> desired replica count.
    """
    def __init__(self, env, policy: Callable, startup_seconds=30.0,
                 scale_interval_seconds=60.0, slo_seconds=0.5,
                 min_replicas=1, max_replicas=100):
        self.env, self.policy = env, policy
        self.startup_seconds = float(startup_seconds)
        self.scale_interval = float(scale_interval_seconds)
        self.slo = float(slo_seconds)
        self.min_replicas, self.max_replicas = min_replicas, max_replicas
        self.queue = simpy.Store(env)
        self.pods = []
        self.metrics = TrialMetrics()
        self.arrivals = deque()
        self.history = []
        self._next_pod_id = 0
        self._last_integral_time = 0.0

    def _integrate(self):
        now = self.env.now
        self.metrics.pod_seconds += len(self.pods) * (now - self._last_integral_time)
        self._last_integral_time = now

    def _worker(self, pod):
        try:
            yield self.env.timeout(self.startup_seconds)
            while pod.live:
                pod.busy = False
                req = yield self.queue.get()
                if not pod.live:
                    # Return a request taken at the instant the worker was retired.
                    yield self.queue.put(req)
                    break
                pod.busy = True
                arrival, service = req
                yield self.env.timeout(service)
                response = self.env.now - arrival
                self.metrics.completed += 1
                self.metrics.response_times.append(response)
                if response > self.slo:
                    self.metrics.missed_deadline += 1
        except simpy.Interrupt:
            # Only idle or warming pods are interrupted, never busy workers.
            pass

    def _change_replicas(self, target):
        target = max(self.min_replicas, min(self.max_replicas, int(target)))
        current = len(self.pods)
        if target == current:
            return
        self._integrate()
        if target > current:
            for _ in range(target - current):
                p = _Pod(number=self._next_pod_id)
                self._next_pod_id += 1
                p.process = self.env.process(self._worker(p))
                self.pods.append(p)
        else:
            # Retire idle workers first; busy workers complete their service.
            # The retiring busy workers still count as allocated until finished,
            # so avoid downscale while insufficient workers are idle.
            idle = [p for p in self.pods if not p.busy]
            remove = min(current - target, len(idle))
            for p in idle[:remove]:
                p.live = False
                if p.process.is_alive:
                    p.process.interrupt()
                self.pods.remove(p)
        self.metrics.scaling_actions += 1

    def _controller(self, duration):
        while self.env.now < duration:
            now = self.env.now
            while self.arrivals and self.arrivals[0] <= now - 60.0:
                self.arrivals.popleft()
            observed_rps = len(self.arrivals) / 60.0
            self.history.append(observed_rps)
            desired = self.policy(observed_rps, len(self.pods),
                                  list(self.history), {'time': now})
            self._change_replicas(desired)
            self.metrics.timeline.append({
                'time_s': now, 'observed_rps': observed_rps,
                'desired_replicas': int(desired), 'active_replicas': len(self.pods),
                'queue_length': len(self.queue.items),
            })
            yield self.env.timeout(self.scale_interval)

    def _producer(self, events):
        for arrival, service in events:
            yield self.env.timeout(max(0.0, arrival - self.env.now))
            self.metrics.generated += 1
            self.arrivals.append(self.env.now)
            yield self.queue.put((self.env.now, service))

    def run(self, events, duration_seconds):
        self.env.process(self._producer(events))
        self.env.process(self._controller(duration_seconds))
        self.env.run(until=duration_seconds)
        self._integrate()
        self.metrics.unfinished_at_end = self.metrics.generated - self.metrics.completed
        assert self.metrics.unfinished_at_end >= 0
        return self.metrics


def generate_replay(workload_per_minute: Sequence[float], seed=42,
                    mean_service_seconds=0.1):
    """Deterministic replay: rounded per-minute counts, uniform intra-minute times.

    Synthetic generation method; not the repository's original Poisson arrivals.
    """
    rng = np.random.default_rng(seed)
    events = []
    for minute, count in enumerate(workload_per_minute):
        if not np.isfinite(count) or count < 0:
            raise ValueError('Workload counts must be finite and nonnegative')
        n = int(round(float(count)))
        arrival_times = np.sort(rng.uniform(minute * 60, (minute + 1) * 60, n))
        durations = rng.exponential(mean_service_seconds, n)
        events.extend(zip(arrival_times.tolist(), durations.tolist()))
    return events


class NormalizedHPA:
    """Capacity-normalized reference HPA (NOT authors' exact equations)."""
    def __init__(self, target_utilization=0.7, mean_service_seconds=0.1,
                 down_cooldown_seconds=300):
        self.target = target_utilization
        self.service = mean_service_seconds
        self.cooldown = down_cooldown_seconds
        self.last_down = -float('inf')

    def __call__(self, rps, replicas, history, context):
        desired = max(1, math.ceil(rps * self.service / self.target))
        if desired < replicas:
            if context['time'] - self.last_down < self.cooldown:
                return replicas
            self.last_down = context['time']
        return desired


class NormalizedPredictive:
    """Rolling linear trend forecasting; same per-replica capacity as HPA."""
    def __init__(self, target_utilization=0.7, mean_service_seconds=0.1,
                 lookahead_steps=3, history_steps=12):
        self.target = target_utilization
        self.service = mean_service_seconds
        self.lookahead = lookahead_steps
        self.history_steps = history_steps

    def __call__(self, rps, replicas, history, context):
        recent = np.asarray(history[-self.history_steps:], dtype=float)
        if len(recent) < 3:
            estimate = rps
        else:
            x = np.arange(len(recent), dtype=float)
            slope, intercept = np.polyfit(x, recent, 1)
            estimate = max(0.0, slope * (len(recent) - 1 + self.lookahead) + intercept)
        return max(1, math.ceil(estimate * self.service / self.target))
