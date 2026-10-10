"""Phase 3D-C: verified event provisioning versus unverified and masked controls.

Run from repository root. Reuses the Phase 3C simulated service model and
Phase 3A published event metadata. Synthetic evidence only.
"""
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
from src.event_aware.verified_policy import VerifiedEventAwarePolicy
from src.core.simulator_repaired import generate_replay
from examples.phase3c_compare import EvaluationSimulator, evaluation_metrics

WARMUP = 60
VARIANTS = ('original_event', 'verified_event', 'event_masked')


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.seeds <= 0:
        raise ValueError('--seeds must be positive')
    selected = SCENARIOS if args.scenario == 'all' else (args.scenario,)
    out = ROOT / 'results' / 'phase3dc'
    out.mkdir(parents=True, exist_ok=True)
    trials, paired = [], []
    for scenario in selected:
        for seed in range(args.seed, args.seed + args.seeds):
            rows, _ = generate(scenario, seed=seed)
            public = [{k: row[k] for k in PUBLIC_KEYS} for row in rows]
            replay = generate_replay([r['demand_rpm'] for r in rows], seed=seed,
                                     mean_service_seconds=0.1)
            duration = len(rows)*60
            outcomes = {}
            for variant in VARIANTS:
                if variant == 'original_event':
                    base = EventAwarePolicy(public, use_guardrail=True)
                elif variant == 'verified_event':
                    base = VerifiedEventAwarePolicy(public, use_guardrail=True,
                        verification_delay_minutes=args.grace,
                        required_low_ticks=args.low_ticks,
                        min_realized_fraction=args.min_fraction)
                else:
                    base = EventAwarePolicy(public, use_event=False, use_guardrail=True)
                policy = DecisionLoggingPolicy(base)
                sim = EvaluationSimulator(simpy.Environment(), policy,
                    startup_seconds=30, scale_interval_seconds=60, slo_seconds=0.5,
                    min_replicas=1, max_replicas=32,
                    eval_begin=WARMUP*60, eval_end=duration)
                total = sim.run(replay, duration)
                metrics = evaluation_metrics(sim, WARMUP*60, duration)
                decisions = policy.decisions
                record = dict(scenario=scenario, seed=seed, variant=variant, **metrics,
                    total_pod_minutes=total['pod_minutes'],
                    scaling_actions=total['scaling_actions'],
                    event_headroom_cancelled_decisions=sum(d['reason'] ==
                        'event_headroom_cancelled_reactive' for d in decisions))
                trials.append(record)
                outcomes[variant] = record
                if args.timelines:
                    write_csv(out/f'{scenario}_{seed}_{variant}_timeline.csv', sim.metrics.timeline)
                    write_csv(out/f'{scenario}_{seed}_{variant}_decisions.csv', decisions)
                print(f'{scenario} seed={seed} {variant}: '
                      f'SLO={metrics["eval_confirmed_slo_rate"]:.3%} '
                      f'pods={metrics["eval_pod_minutes"]:.1f} '
                      f'cancellations={record["event_headroom_cancelled_decisions"]}', flush=True)
            old, new, masked = (outcomes[n] for n in VARIANTS)
            if scenario in ('accurate', 'no_campaign', 'unannounced'):
                # No cancellation expected for an accurate, promptly visible
                # event; absent metadata cannot produce event-specific changes.
                if new['event_headroom_cancelled_decisions'] != 0:
                    raise AssertionError(f'Unexpected event cancellation on {scenario} seed {seed}')
            if scenario in ('no_campaign', 'unannounced'):
                for field in ('eval_confirmed_slo_violations','eval_pod_minutes'):
                    if old[field] != new[field] or new[field] != masked[field]:
                        raise AssertionError(f'No-event control mismatch: {scenario} {field}')
            paired.append(dict(scenario=scenario, seed=seed,
                verified_minus_original_slo_pp=100*(new['eval_confirmed_slo_rate'] - old['eval_confirmed_slo_rate']),
                verified_minus_original_pod_minutes=new['eval_pod_minutes']-old['eval_pod_minutes'],
                verified_minus_masked_slo_pp=100*(new['eval_confirmed_slo_rate']-masked['eval_confirmed_slo_rate']),
                verified_cancelled_decisions=new['event_headroom_cancelled_decisions']))
    write_csv(out/'trials.csv', trials)
    write_csv(out/'paired.csv', paired)
    aggregate=[]
    groups=defaultdict(list)
    for r in trials:
        groups[(r['scenario'],r['variant'])].append(r)
    for (scenario,variant), items in sorted(groups.items()):
        rates=np.array([x['eval_confirmed_slo_rate'] for x in items])
        pods=np.array([x['eval_pod_minutes'] for x in items])
        aggregate.append(dict(scenario=scenario,variant=variant,n=len(items),
            mean_slo_rate=float(rates.mean()),
            sd_slo_rate=float(rates.std(ddof=1)) if len(items)>1 else '',
            mean_eval_pod_minutes=float(pods.mean()),
            mean_cancelled_decisions=float(np.mean([x['event_headroom_cancelled_decisions'] for x in items]))))
    write_csv(out/'aggregate.csv',aggregate)
    protocol=dict(phase='3D-C', paper_reproduction=False, scenarios=list(selected),
        seeds=list(range(args.seed,args.seed+args.seeds)),
        verification_grace_minutes=args.grace,
        low_observation_ticks=args.low_ticks,
        min_realized_fraction=args.min_fraction,
        decision_interval_seconds=60,pod_startup_seconds=30,
        evaluation_minutes=50,warmup_minutes=60,
        limitations=['Synthetic workload; causal observed-arrival checks, no oracle ground truth in policy',
            'No resource cap enforced; budget-matching requires a separate experiment',
            'Only baseline campaign scenarios tested; cancellation parameters chosen a priori',
            'Evaluation SLO denominator counts arrival requests in evaluation interval',
            'No statistical significance claim with three seeds',
            'Delayed demand that starts after verification may still suffer when event headroom is withdrawn'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2), encoding='utf-8')
    print('Saved:',out)
    print('Synthetic reconstructed simulator; not paper reproduction or production Kubernetes evidence.')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['all',*SCENARIOS],default='false_positive')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--grace',type=float,default=2.0)
    p.add_argument('--low-ticks',type=int,default=2)
    p.add_argument('--min-fraction',type=float,default=0.25)
    p.add_argument('--timelines',action='store_true')
    run(p.parse_args())
