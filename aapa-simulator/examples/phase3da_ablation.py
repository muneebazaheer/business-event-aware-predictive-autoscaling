"""Phase 3D-A: within-policy event information and guardrail ablations.

All four variants instantiate the SAME EventAwarePolicy class. Only use_event
and use_guardrail differ. This is a reconstructed synthetic study, not a paper
reproduction and not a resource-matched causal estimate.
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
from src.core.simulator_repaired import generate_replay
from examples.phase3c_compare import EvaluationSimulator, evaluation_metrics

VARIANTS = {
    'event_on_guardrail_on': (True, True),
    'event_off_guardrail_on': (False, True),
    'event_on_guardrail_off': (True, False),
    'event_off_guardrail_off': (False, False),
}
WARMUP = 60


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def public_only(rows):
    return [{key: row[key] for key in PUBLIC_KEYS} for row in rows]


def masked_public(rows):
    """Keep timestamps but hide published event data, without modifying truth."""
    return [dict(minute=row['minute'], event_known=0, event_active=0,
                 minutes_to_event=None, event_type='none', planned_intensity=0.0,
                 planned_duration=0, planned_start_minute=None) for row in rows]


def simulate(rows, replay, use_event, use_guardrail, *, mask_metadata=False,
             save_timeline=False):
    context = masked_public(rows) if mask_metadata else public_only(rows)
    policy = DecisionLoggingPolicy(EventAwarePolicy(
        context, use_event=use_event, use_guardrail=use_guardrail,
        startup_seconds=30, decision_interval_seconds=60,
    ))
    duration = len(rows) * 60
    sim = EvaluationSimulator(simpy.Environment(), policy,
        startup_seconds=30, scale_interval_seconds=60, slo_seconds=0.5,
        min_replicas=1, max_replicas=32,
        eval_begin=WARMUP * 60, eval_end=duration)
    overall = sim.run(replay, duration)
    result = evaluation_metrics(sim, WARMUP * 60, duration)
    result.update(total_pod_minutes=overall['pod_minutes'],
                  scaling_actions_whole_run=overall['scaling_actions'],
                  max_ready_replicas_whole_run=overall['max_ready_replicas'],
                  event_decisions_logged=len(policy.decisions))
    return result, (sim.metrics.timeline if save_timeline else None), (policy.decisions if save_timeline else None)


def equal_controls(a, b):
    fields = ['eval_generated','eval_completed_by_end','eval_outstanding_at_end',
              'eval_confirmed_slo_violations','eval_confirmed_slo_rate',
              'eval_pod_minutes','total_pod_minutes','scaling_actions_whole_run',
              'max_ready_replicas_whole_run']
    differences = {k: (a[k], b[k]) for k in fields if a[k] != b[k]}
    if differences:
        raise AssertionError('Masked-input negative control mismatch: ' + repr(differences))


def run(args):
    if args.seeds < 1:
        raise ValueError('--seeds must be >= 1')
    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    output = ROOT / 'results' / 'phase3da'
    output.mkdir(parents=True, exist_ok=True)
    trials, pairs = [], []
    for scenario in scenarios:
        for seed in range(args.seed, args.seed + args.seeds):
            rows, _manifest = generate(scenario, seed=seed)
            replay = generate_replay([row['demand_rpm'] for row in rows], seed=seed,
                                     mean_service_seconds=0.1)
            per_variant = {}
            for name, (use_event, use_guardrail) in VARIANTS.items():
                result, timeline, decisions = simulate(
                    rows, replay, use_event, use_guardrail, save_timeline=args.timelines)
                record = dict(scenario=scenario, seed=seed, policy=name,
                              event_enabled=use_event, guardrail_enabled=use_guardrail,
                              **result)
                trials.append(record)
                per_variant[name] = record
                if args.timelines:
                    write_csv(output / f'{scenario}_seed{seed}_{name}_timeline.csv', timeline)
                    write_csv(output / f'{scenario}_seed{seed}_{name}_decisions.csv', decisions)
                print(f'{scenario} seed={seed} {name}: '
                      f'SLO={result["eval_confirmed_slo_rate"]:.3%}, '
                      f'pods={result["eval_pod_minutes"]:.1f}', flush=True)

            # For both guardrail settings, disabling event processing must be
            # exactly equivalent to passing metadata-free context to same policy.
            for guardrail in (True, False):
                name = f'event_off_guardrail_{"on" if guardrail else "off"}'
                masked_result, _, _ = simulate(rows, replay, True, guardrail, mask_metadata=True)
                equal_controls(per_variant[name], masked_result)

            on = per_variant['event_on_guardrail_on']
            off = per_variant['event_off_guardrail_on']
            on_noguard = per_variant['event_on_guardrail_off']
            pairs.append(dict(scenario=scenario, seed=seed,
                event_information_slo_delta=on['eval_confirmed_slo_rate']-off['eval_confirmed_slo_rate'],
                event_information_pod_min_delta=on['eval_pod_minutes']-off['eval_pod_minutes'],
                guardrail_slo_delta=on['eval_confirmed_slo_rate']-on_noguard['eval_confirmed_slo_rate'],
                guardrail_pod_min_delta=on['eval_pod_minutes']-on_noguard['eval_pod_minutes']))
    grouped = defaultdict(list)
    for row in trials:
        grouped[(row['scenario'], row['policy'])].append(row)
    summary = []
    for (scenario, policy), group in sorted(grouped.items()):
        slo = np.array([r['eval_confirmed_slo_rate'] for r in group], dtype=float)
        cost = np.array([r['eval_pod_minutes'] for r in group], dtype=float)
        summary.append(dict(scenario=scenario, policy=policy, n=len(group),
                            mean_eval_slo_rate=float(slo.mean()),
                            sd_eval_slo_rate=float(slo.std(ddof=1)) if len(slo)>1 else '',
                            mean_eval_pod_minutes=float(cost.mean()),
                            sd_eval_pod_minutes=float(cost.std(ddof=1)) if len(cost)>1 else ''))
    write_csv(output / 'trials.csv', trials)
    write_csv(output / 'aggregate.csv', summary)
    write_csv(output / 'paired_ablations.csv', pairs)
    protocol = dict(phase='3D-A', paper_reproduction=False,
        purpose='Factorial 2x2 event metadata and reactive guardrail ablation',
        variants={k: dict(event_enabled=v[0], guardrail_enabled=v[1]) for k,v in VARIANTS.items()},
        seeds=list(range(args.seed, args.seed+args.seeds)), scenarios=scenarios,
        warmup_minutes=WARMUP, eval_minutes=50,
        startup_seconds=30, scale_interval_seconds=60, slo_seconds=0.5,
        max_replicas=32, mean_service_seconds=0.1,
        replay='Identical generated request arrivals and service times per scenario and seed',
        negative_control='use_event=False equals use_event=True with metadata masked; both guardrail states checked',
        limitations=['Policies share code and capacity constraints but spend different pod-minutes.',
                     'Not a resource-matched effect estimate.',
                     'Generated synthetic campaign profiles; not external operational traces.',
                     'SLO includes completed deadline misses and overdue outstanding requests.',
                     'Latency percentiles on completed requests are subject to censoring.',
                     'A fixed two-minute pre-scale lead is used; timing robustness follows in Phase 3D-B.'])
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('PASS: event-off policies equal masked-metadata policies for all tested pairs.')
    print('Saved:', output)
    print('These are within-controller ablations; NOT a resource-matched or paper-faithful study.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['all', *SCENARIOS], default='accurate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--timelines', action='store_true')
    run(parser.parse_args())
