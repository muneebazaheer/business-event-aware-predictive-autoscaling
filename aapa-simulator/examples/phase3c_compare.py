"""Phase 3C: paired multi-seed synthetic campaign comparison (not paper reproduction)."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.event_aware.campaigns import generate, SCENARIOS
from src.event_aware.policy import EventAwarePolicy, DecisionLoggingPolicy, PUBLIC_KEYS
from src.core.simulator_repaired import generate_replay, NormalizedHPA, NormalizedPredictive
from src.core.simulator_phase2b import AuditedSimulator
from src.core.phase2c_policies import ArchetypePolicy

WARMUP = 60
EVAL = 50


class EvaluationSimulator(AuditedSimulator):
    """Integrate allocated pod consumption only inside evaluation period."""
    def __init__(self, *args, eval_begin=3600.0, eval_end=6600.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_begin = float(eval_begin)
        self.eval_end = float(eval_end)
        self.eval_pod_seconds = 0.0

    def _integrate(self):
        left = max(self._last_integral_time, self.eval_begin)
        right = min(float(self.env.now), self.eval_end)
        if right > left:
            self.eval_pod_seconds += len(self.pods) * (right - left)
        super()._integrate()


def evaluation_metrics(sim, begin, end):
    requests = [r for r in sim.request_records.values() if begin <= r['arrival'] < end]
    completed = [r for r in requests if r['completion'] is not None and r['completion'] <= end]
    unresolved = [r for r in requests if r['completion'] is None or r['completion'] > end]
    completed_missed = sum(r['completion'] - r['arrival'] > sim.slo for r in completed)
    outstanding_overdue = sum(end - r['arrival'] > sim.slo for r in unresolved)
    pending_within_deadline = len(unresolved) - outstanding_overdue
    confirmed = completed_missed + outstanding_overdue
    latencies = np.asarray([r['completion'] - r['arrival'] for r in completed], dtype=float)
    def percentile(q):
        return float(np.percentile(latencies, q)) if len(latencies) else None
    return {
        'eval_generated': len(requests),
        'eval_completed_by_end': len(completed),
        'eval_outstanding_at_end': len(unresolved),
        'eval_outstanding_overdue': outstanding_overdue,
        'eval_outstanding_within_deadline': pending_within_deadline,
        'eval_confirmed_slo_violations': confirmed,
        'eval_confirmed_slo_rate': confirmed / len(requests) if requests else None,
        'eval_worst_case_slo_rate': (confirmed + pending_within_deadline) / len(requests) if requests else None,
        'eval_mean_latency_completed_s': float(np.mean(latencies)) if len(latencies) else None,
        'eval_p95_latency_completed_s': percentile(95),
        'eval_p99_latency_completed_s': percentile(99),
        'eval_pod_minutes': sim.eval_pod_seconds / 60,
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_policies(public_rows, model, encoder, include_aapa):
    factories = {
        'hpa': lambda: NormalizedHPA(),
        'history_predictive': lambda: NormalizedPredictive(),
        'event_guardrail': lambda: DecisionLoggingPolicy(EventAwarePolicy(public_rows)),
        'event_no_guardrail': lambda: DecisionLoggingPolicy(EventAwarePolicy(public_rows, use_guardrail=False)),
        # Exact negative control: same policy, parameters, and state initialization as HPA.
        # EventAwarePolicy(use_event=False) is NOT equivalent: it lacks HPA's scale-down cooldown.
        'event_disabled': lambda: NormalizedHPA(),
    }
    if include_aapa:
        factories['archetype_reconstructed'] = lambda: ArchetypePolicy(model, encoder, 'full')
    return factories


def run(args):
    if args.seeds < 1:
        raise ValueError('--seeds must be at least 1')
    scenarios = SCENARIOS if args.scenario == 'all' else (args.scenario,)
    output = ROOT / 'results' / 'phase3c'
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'models/phase2/lightgbm_model.txt'
    encoder = ROOT / 'models/phase2/label_encoder.pkl'
    if not args.no_aapa and (not model.exists() or not encoder.exists()):
        raise FileNotFoundError('For reconstructed AAPA, export model and encoder to models/phase2, or use --no-aapa')
    trials = []
    for scenario in scenarios:
        for seed in range(args.seed, args.seed + args.seeds):
            rows, manifest = generate(scenario, seed=seed)
            public = [{k: row[k] for k in PUBLIC_KEYS} for row in rows]
            arrivals = generate_replay([row['demand_rpm'] for row in rows], seed=seed, mean_service_seconds=0.1)
            duration = len(rows) * 60
            factories = make_policies(public, model, encoder, not args.no_aapa)
            for policy_name, factory in factories.items():
                policy = factory()
                simulator = EvaluationSimulator(
                    simpy.Environment(), policy,
                    startup_seconds=30, scale_interval_seconds=60,
                    slo_seconds=0.5, min_replicas=1, max_replicas=32,
                    eval_begin=WARMUP * 60, eval_end=duration,
                )
                overall = simulator.run(arrivals, duration)
                evaluation = evaluation_metrics(simulator, WARMUP * 60, duration)
                trial = dict(scenario=scenario, seed=seed, policy=policy_name, **evaluation,
                             total_pod_minutes=overall['pod_minutes'],
                             scaling_actions_whole_run=overall['scaling_actions'],
                             max_ready_replicas_whole_run=overall['max_ready_replicas'])
                if hasattr(policy, 'base') and hasattr(policy.base, 'last_reason'):
                    trial['event_decisions_logged'] = len(policy.decisions)
                    if args.timelines:
                        write_csv(output / f'{scenario}_{seed}_{policy_name}_decisions.csv', policy.decisions)
                if hasattr(policy, 'predictions_by_archetype'):
                    trial['archetype_counts'] = json.dumps(policy.predictions_by_archetype, sort_keys=True)
                if args.timelines:
                    write_csv(output / f'{scenario}_{seed}_{policy_name}_timeline.csv', simulator.metrics.timeline)
                trials.append(trial)
                print(f'{scenario} seed={seed} {policy_name}: '
                      f'eval_SLO={evaluation["eval_confirmed_slo_rate"]:.3%}, '
                      f'eval_pod_min={evaluation["eval_pod_minutes"]:.1f}', flush=True)
    # Negative-control equivalence is a hard requirement for a valid experiment.
    # Compare metrics derived from the same replay, not just mean SLO rates.
    keyed = {(r['scenario'], r['seed'], r['policy']): r for r in trials}
    comparable = (
        'eval_generated', 'eval_completed_by_end', 'eval_outstanding_at_end',
        'eval_confirmed_slo_violations', 'eval_confirmed_slo_rate',
        'eval_pod_minutes', 'total_pod_minutes',
        'scaling_actions_whole_run', 'max_ready_replicas_whole_run',
    )
    for scenario in scenarios:
        for seed in range(args.seed, args.seed + args.seeds):
            hpa = keyed[(scenario, seed, 'hpa')]
            disabled = keyed[(scenario, seed, 'event_disabled')]
            mismatches = {field: (hpa[field], disabled[field]) for field in comparable
                          if hpa[field] != disabled[field]}
            if mismatches:
                raise AssertionError(
                    f'Negative control mismatch for {scenario} seed={seed}: {mismatches}'
                )
    print('PASS: event_disabled exactly matches HPA on all tested scenario/seed pairs.')
    write_csv(output / 'trials.csv', trials)
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial['scenario'], trial['policy'])].append(trial)
    summaries = []
    for (scenario, policy), group in sorted(grouped.items()):
        rates = np.asarray([r['eval_confirmed_slo_rate'] for r in group], dtype=float)
        costs = np.asarray([r['eval_pod_minutes'] for r in group], dtype=float)
        summaries.append(dict(scenario=scenario, policy=policy, n=len(group),
                              mean_eval_slo_rate=float(rates.mean()),
                              sd_eval_slo_rate=float(rates.std(ddof=1)) if len(rates)>1 else '',
                              mean_eval_pod_minutes=float(costs.mean()),
                              sd_eval_pod_minutes=float(costs.std(ddof=1)) if len(costs)>1 else '',
                              min_eval_slo_rate=float(rates.min()),
                              max_eval_slo_rate=float(rates.max())))
    write_csv(output / 'aggregate.csv', summaries)
    # Paired comparison uses the same replay seed, but is NOT resource-matched.
    paired = []
    by_pair = {(r['scenario'], r['seed'], r['policy']): r for r in trials}
    for scenario in scenarios:
        for seed in range(args.seed, args.seed + args.seeds):
            hpa = by_pair[(scenario, seed, 'hpa')]
            event = by_pair[(scenario, seed, 'event_guardrail')]
            paired.append(dict(scenario=scenario, seed=seed,
                               event_minus_hpa_slo_rate=event['eval_confirmed_slo_rate'] - hpa['eval_confirmed_slo_rate'],
                               event_minus_hpa_eval_pod_minutes=event['eval_pod_minutes'] - hpa['eval_pod_minutes'],
                               event_over_hpa_cost_ratio=(event['eval_pod_minutes'] / hpa['eval_pod_minutes'])
                               if hpa['eval_pod_minutes'] else None))
    write_csv(output / 'paired_event_vs_hpa.csv', paired)
    protocol = {
        'phase': '3C', 'study': 'Synthetic controlled campaign comparison',
        'paper_reproduction': False, 'exact_AAPA_paper_policy': False,
        'pretrained_classifier_used_only_by': ['archetype_reconstructed'] if not args.no_aapa else [],
        'event_aware_policy': 'transparent heuristic, not a trained forecasting model',
        'scenarios': list(scenarios), 'first_seed': args.seed, 'number_of_seeds': args.seeds,
        'warmup_minutes': WARMUP, 'evaluation_minutes': EVAL,
        'max_replicas': 32, 'startup_seconds': 30, 'scale_interval_seconds': 60,
        'SLO_seconds': 0.5, 'mean_service_seconds': 0.1,
        'evaluation_requests': 'arrivals within [minute 60, minute 110); completed response times censored at end',
        'cost': 'allocated replica-minutes in evaluation period; total cost also reported',
        'common_replay_per_scenario_seed': True,
        'limitations': [
            'Equal per-pod capacity and maximum replicas do not guarantee equal actual pod-minute usage.',
            'SLO rates include confirmed overdue unfinished requests at evaluation end; in-deadline unfinished are censored.',
            'Reported latency percentiles use only requests completed by evaluation end (censoring bias).',
            'AAPA is a reconstructed archetype-aware policy, not the authors exact implementation.',
            'Synthetic workloads are deterministic profiles with random arrivals and execution times.',
            'No causal claim that event metadata alone drives a gain until cost-controlled comparisons and ablations.',
            'event_disabled is an exact HPA negative control, not EventAwarePolicy with event inputs masked.',
            'EventAwarePolicy reactive fallback lacks HPA scale-down cooldown, so event_guardrail vs HPA is not a pure metadata-only ablation.',
            'Notice-time sensitivity and robustness are reserved for Phase 3D.',
        ],
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('Saved outputs in', output)
    print('Exploratory synthetic comparisons only; NOT paper reproduction or resource-matched proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['all', *SCENARIOS], default='accurate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--no-aapa', action='store_true', help='Skip reconstructed archetype baseline')
    parser.add_argument('--timelines', action='store_true', help='Write per-decision timelines (many files)')
    run(parser.parse_args())
