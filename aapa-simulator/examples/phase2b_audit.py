"""Phase 2B simulator audit and OPTIONAL exploratory AAPA comparison.

This is not a faithful reproduction of AAPA paper experiments.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import simpy
from src.core.simulator_repaired import NormalizedHPA, NormalizedPredictive, generate_replay
from src.core.simulator_phase2b import AuditedSimulator, RepoAAPAAdapter


def workload():
    # 50-minute stress case. Warmup requirement is 60min, so prepend 60-min
    # constant-rate context for AAPA; all policies receive the same trace.
    return np.concatenate([
        np.full(60, 10.0),
        np.full(10, 10.0), np.full(3, 150.0),
        np.full(10, 10.0), np.full(3, 200.0),
        np.full(10, 10.0), np.full(2, 180.0),
        np.full(12, 10.0),
    ]) * 12


def run(args):
    trace = workload()
    events = generate_replay(trace, seed=args.seed, mean_service_seconds=0.1)
    out = ROOT / 'results' / 'phase2b_audit'
    out.mkdir(parents=True, exist_ok=True)
    policies = [
        ('normalized_hpa', NormalizedHPA()),
        ('normalized_predictive', NormalizedPredictive()),
    ]
    if args.with_aapa:
        model = ROOT / 'models' / 'phase2' / 'lightgbm_model.txt'
        encoder = ROOT / 'models' / 'phase2' / 'label_encoder.pkl'
        for p in (model, encoder):
            if not p.exists():
                raise FileNotFoundError(f'{p} is missing. Export the Phase 1 Booster and encoder first.')
        policies.append(('repo_aapa_exploratory', RepoAAPAAdapter(model, encoder)))

    results = {}
    duration = len(trace) * 60
    for name, policy in policies:
        sim = AuditedSimulator(simpy.Environment(), policy,
            startup_seconds=30, scale_interval_seconds=60,
            slo_seconds=0.5, min_replicas=1, max_replicas=100)
        summary = sim.run(events, duration)
        results[name] = summary
        with (out / f'{name}_timeline.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(sim.metrics.timeline[0]))
            writer.writeheader()
            writer.writerows(sim.metrics.timeline)
        print(f'\n{name}:')
        for k, v in summary.items():
            print(f'  {k}: {v}')
    (out / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nWARNING: synthetic workloads; AAPA and normalized baselines are NOT capacity-aligned.')
    print('Output:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-aapa', action='store_true', help='Exploratory only; AAPA unit assumptions differ')
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
