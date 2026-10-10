"""Phase 2C controlled experiments (synthetic, reconstructed, non-paper-fidelity)."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import simpy
from src.core.simulator_repaired import generate_replay, NormalizedHPA, NormalizedPredictive
from src.core.simulator_phase2b import AuditedSimulator
from src.core.phase2c_policies import ArchetypePolicy

WARMUP = 60
EVAL = 50


def scenario(name):
    t = np.arange(EVAL, dtype=float)
    if name == 'spike':
        v = np.full(EVAL, 120.0)
        v[10:13] = 1800
        v[23:26] = 2400
        v[36:38] = 2160
    elif name == 'periodic':
        v = 600 + 480 * (1 + np.sin(2 * np.pi * t / 12)) / 2
    elif name == 'ramp':
        v = 120 + 42 * t
    elif name == 'stationary':
        v = 540 + 60 * np.sin(t * 1.73)
    else:
        raise ValueError(name)
    return np.concatenate([np.full(WARMUP, 120.0), v])


def policies(model, encoder):
    return {
        'hpa': lambda: NormalizedHPA(),
        'predictive': lambda: NormalizedPredictive(),
        'archetype_full': lambda: ArchetypePolicy(model, encoder, 'full'),
        'archetype_no_confidence': lambda: ArchetypePolicy(model, encoder, 'no_confidence'),
        'archetype_fixed_spike': lambda: ArchetypePolicy(model, encoder, 'fixed_spike'),
        'archetype_fixed_stationary': lambda: ArchetypePolicy(model, encoder, 'fixed_stationary'),
    }


def evaluation_metrics(sim, begin_seconds, end_seconds):
    records = [r for r in sim.request_records.values() if begin_seconds <= r['arrival'] < end_seconds]
    completed = [r for r in records if r['completion'] is not None and r['completion'] <= end_seconds]
    violations = sum((r['completion'] - r['arrival'] > sim.slo) if r['completion'] is not None and r['completion'] <= end_seconds
                     else (end_seconds - r['arrival'] > sim.slo) for r in records)
    unresolved = sum(r['completion'] is None or r['completion'] > end_seconds for r in records)
    latency = [r['completion'] - r['arrival'] for r in completed]
    # total pod minutes includes warmup, reported explicitly for fair cross-policy cost.
    return {'eval_generated': len(records), 'eval_completed_by_end': len(completed),
            'eval_unfinished_at_end': unresolved, 'eval_confirmed_slo_violations': violations,
            'eval_confirmed_slo_violation_rate': violations / len(records) if records else 0.0,
            'eval_mean_latency_completed_s': float(np.mean(latency)) if latency else None,
            'eval_p95_latency_completed_s': float(np.percentile(latency, 95)) if latency else None}


def save_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    model = ROOT / 'models/phase2/lightgbm_model.txt'
    encoder = ROOT / 'models/phase2/label_encoder.pkl'
    if not model.is_file() or not encoder.is_file():
        raise FileNotFoundError('Export the model and label encoder to models/phase2/ first')
    if args.seeds < 1:
        raise ValueError('--seeds must be at least 1')
    out = ROOT / 'results/phase2c'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = ['spike', 'periodic', 'ramp', 'stationary'] if args.scenario == 'all' else [args.scenario]
    for workload_name in selected:
        trace = scenario(workload_name)
        duration = len(trace) * 60
        for seed in range(args.seed, args.seed + args.seeds):
            # Generate events ONCE per scenario and seed. Every policy sees the same events.
            events = generate_replay(trace, seed=seed, mean_service_seconds=0.1)
            for policy_name, factory in policies(model, encoder).items():
                policy = factory()
                sim = AuditedSimulator(simpy.Environment(), policy,
                                       startup_seconds=30, scale_interval_seconds=60,
                                       slo_seconds=0.5, min_replicas=1, max_replicas=32)
                metrics = sim.run(events, duration)
                metrics.update(evaluation_metrics(sim, WARMUP * 60, duration))
                metrics.update({'scenario': workload_name, 'seed': seed, 'policy': policy_name,
                                'evaluation_minutes': EVAL, 'warmup_minutes': WARMUP,
                                'replica_limit': 32, 'startup_seconds': 30,
                                'mean_service_seconds': 0.1, 'slo_seconds': 0.5})
                if hasattr(policy, 'predictions_by_archetype'):
                    metrics['archetype_counts'] = json.dumps(policy.predictions_by_archetype, sort_keys=True)
                rows.append(metrics)
                print(f"{workload_name} seed={seed} {policy_name}: "
                      f"SLO={metrics['eval_confirmed_slo_violation_rate']:.3%}, "
                      f"pod-min={metrics['pod_minutes']:.1f}", flush=True)
                if args.timelines:
                    save_rows(out / f'{workload_name}_{seed}_{policy_name}_timeline.csv', sim.metrics.timeline)
    save_rows(out / 'trials.csv', rows)
    grouped = {}
    for r in rows:
        grouped.setdefault((r['scenario'], r['policy']), []).append(r)
    aggregate = []
    for (scenario_name, policy_name), trials in grouped.items():
        slo = np.asarray([t['eval_confirmed_slo_violation_rate'] for t in trials])
        pods = np.asarray([t['pod_minutes'] for t in trials])
        aggregate.append({'scenario': scenario_name, 'policy': policy_name, 'n': len(trials),
                          'slo_rate_mean': float(slo.mean()), 'pod_minutes_mean': float(pods.mean()),
                          'slo_rate_min': float(slo.min()), 'slo_rate_max': float(slo.max()),
                          'slo_rate_sd': float(slo.std(ddof=1)) if len(slo)>1 else '',
                          'pod_minutes_sd': float(pods.std(ddof=1)) if len(pods)>1 else ''})
    save_rows(out / 'aggregate.csv', aggregate)
    (out / 'protocol.json').write_text(json.dumps({'paper_reproduction': False,
        'description': 'Synthetic, capacity-aligned reconstructed archetype experiment',
        'scenario_names': selected, 'seed_start': args.seed, 'seeds': args.seeds,
        'warmup_minutes': WARMUP, 'evaluation_minutes': EVAL,
        'service_mean_seconds': 0.1, 'startup_seconds': 30, 'SLO_seconds': 0.5,
        'maximum_replicas': 32, 'fixed_headroom_pods': {'SPIKE': 3, 'PERIODIC': 1, 'RAMP': 1, 'STATIONARY_NOISY': 0},
        'caution': 'Policy formulas are investigator-authored, not the paper original. '
                   'Reported pod-minutes include warm-up; evaluation SLO only uses arrivals after warm-up.'}, indent=2), encoding='utf-8')
    print('\nSaved:', out)
    print('NOT paper reproduction; no claims of uncertainty calibration or causal class benefit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['all','spike','periodic','ramp','stationary'], default='spike')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timelines', action='store_true')
    run(parser.parse_args())
