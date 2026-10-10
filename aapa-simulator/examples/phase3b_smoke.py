"""Phase 3B smoke comparison: reactive HPA versus causal event-aware policy."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.event_aware.campaigns import generate, save, SCENARIOS
from src.event_aware.policy import EventAwarePolicy, DecisionLoggingPolicy, PUBLIC_KEYS
from src.core.simulator_repaired import generate_replay, NormalizedHPA
from src.core.simulator_phase2b import AuditedSimulator


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_trial(scenario, seed, output):
    rows, manifest = generate(scenario, seed=seed)
    public = [{k: r[k] for k in PUBLIC_KEYS} for r in rows]
    arrivals = generate_replay([r['demand_rpm'] for r in rows], seed=seed,
                               mean_service_seconds=0.1)
    duration = len(rows) * 60
    policies = {
        'hpa': lambda: NormalizedHPA(),
        'event_guardrail': lambda: DecisionLoggingPolicy(EventAwarePolicy(public)),
        'event_no_guardrail': lambda: DecisionLoggingPolicy(EventAwarePolicy(public, use_guardrail=False)),
        'event_disabled': lambda: DecisionLoggingPolicy(EventAwarePolicy(public, use_event=False)),
    }
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for label, make_policy in policies.items():
        policy = make_policy()
        sim = AuditedSimulator(simpy.Environment(), policy, startup_seconds=30,
                               scale_interval_seconds=60, slo_seconds=0.5,
                               min_replicas=1, max_replicas=32)
        summary = sim.run(arrivals, duration)
        # This first smoke test measures the entire 110-minute run. Evaluation-
        # only metrics and matched budgets will be added during Phase 3C.
        item = {'scenario': scenario, 'seed': seed, 'policy': label, **summary}
        results.append(item)
        print(f'{scenario} seed={seed} {label}: '
              f'SLO={summary["confirmed_slo_violation_rate"]:.3%}, '
              f'pod-min={summary["pod_minutes"]:.1f}')
        if hasattr(policy, 'decisions'):
            write_csv(output / f'{scenario}_{label}_seed{seed}_decisions.csv', policy.decisions)
        write_csv(output / f'{scenario}_{label}_seed{seed}_timeline.csv', sim.metrics.timeline)
    write_csv(output / f'{scenario}_seed{seed}_summary.csv', results)
    (output / f'{scenario}_seed{seed}_protocol.json').write_text(json.dumps({
        'phase': '3B', 'scenario': scenario, 'seed': seed,
        'duration_minutes': len(rows), 'warmup_minutes': 60,
        'metrics_scope': 'whole-run, including warmup; not evaluation-only',
        'matched_resource_use': False, 'paper_reproduction': False,
        'event_announcement_visibility': 'published information only',
        'guardrail': 'current observed-load capacity floor',
        'limitations': [
            'Synthetic workloads and reconstructed policies.',
            'No budget-matched comparison or forecast training yet.',
            'Decision logs are used to verify prescaling before event start.',
            'Event-only ablation still uses observed load when event is not actionable.',
        ],
    }, indent=2), encoding='utf-8')
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', choices=['all', *SCENARIOS], default='accurate')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    output = ROOT / 'results/phase3b'
    selected = SCENARIOS if args.scenario == 'all' else [args.scenario]
    for scenario in selected:
        run_trial(scenario, args.seed, output)
    print('Outputs:', output)
    print('Smoke test only: NOT event-aware improvement evidence yet.')


if __name__ == '__main__':
    main()
