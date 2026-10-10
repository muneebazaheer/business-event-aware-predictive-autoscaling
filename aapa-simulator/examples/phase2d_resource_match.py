"""Resource-matched capacity response curves and held-out evaluation.

Seed 42 calibrates each policy's provisioned pod-minutes separately per scenario.
Seeds 43+ evaluate fixed calibrated settings. These policies/simulation are reconstructed,
not a paper-faithful AAPA reproduction. Cannot infer a causal classifier effect.
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
from examples.phase2c_experiment import scenario, policies, evaluation_metrics, WARMUP, EVAL
from src.core.simulator_repaired import generate_replay
from src.core.phase2d_budget import IntensityPolicy, ResourceAuditSimulator

POLICY_NAMES = [
    'hpa', 'predictive', 'archetype_full', 'archetype_no_confidence',
    'archetype_fixed_spike', 'archetype_fixed_stationary'
]
DEFAULT_FACTORS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8]


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def evaluate(name, policy_name, factor, seed, event_cache, factories, timelines=False):
    trace = scenario(name)
    duration = len(trace) * 60
    key = (name, seed)
    if key not in event_cache:
        event_cache[key] = generate_replay(trace, seed=seed, mean_service_seconds=0.1)
    policy = IntensityPolicy(factories[policy_name](), factor)
    sim = ResourceAuditSimulator(simpy.Environment(), policy,
        startup_seconds=30, scale_interval_seconds=60, slo_seconds=0.5,
        min_replicas=1, max_replicas=32,
        eval_start_seconds=WARMUP * 60, eval_end_seconds=duration)
    metrics = sim.run(event_cache[key], duration)
    metrics.update(evaluation_metrics(sim, WARMUP * 60, duration))
    metrics.update(scenario=name, policy=policy_name, seed=seed, intensity_factor=factor,
                   eval_pod_minutes=sim.eval_pod_seconds / 60,
                   total_pod_minutes=metrics['pod_minutes'])
    if timelines:
        return metrics, sim.metrics.timeline
    return metrics, None


def run(args):
    if args.validation_seeds < 1:
        raise ValueError('--validation-seeds must be positive')
    if args.calibration_seed in range(args.validation_seed_start, args.validation_seed_start + args.validation_seeds):
        raise ValueError('Calibration seed cannot be included in held-out validation seeds')
    if args.tolerance <= 0 or args.tolerance > 1:
        raise ValueError('--tolerance must be in (0, 1]')
    factors = [float(v) for v in args.factors.split(',')]
    if len(set(factors)) != len(factors) or any(not np.isfinite(f) or f <= 0 for f in factors):
        raise ValueError('Factors must be distinct, positive and finite')
    budgets = [float(x) for x in args.budgets.split(',')]
    if any(not np.isfinite(b) or b < EVAL for b in budgets):
        raise ValueError(f'Budgets must be at least {EVAL} evaluation pod-minutes (one-pod floor)')
    model = ROOT / 'models/phase2/lightgbm_model.txt'
    encoder = ROOT / 'models/phase2/label_encoder.pkl'
    if not model.is_file() or not encoder.is_file():
        raise FileNotFoundError('Missing models/phase2/lightgbm_model.txt or label_encoder.pkl')
    selected = ['spike','periodic','ramp','stationary'] if args.scenario == 'all' else [args.scenario]
    output = ROOT / 'results/phase2d'
    output.mkdir(parents=True, exist_ok=True)
    factories = policies(model, encoder)
    cache = {}
    calibration = []
    heldout = []
    selection = []
    for sc in selected:
        for policy in POLICY_NAMES:
            print(f'Calibrating {sc} / {policy}', flush=True)
            grid = []
            for factor in factors:
                r, _ = evaluate(sc, policy, factor, args.calibration_seed, cache, factories)
                calibration.append(r)
                grid.append(r)
            for budget in budgets:
                # Deterministic nearest match on calibration seed only.
                winner = min(grid, key=lambda x: (abs(x['eval_pod_minutes']-budget), x['intensity_factor']))
                selected_factor = winner['intensity_factor']
                cal_usage = winner['eval_pod_minutes']
                cal_deviation = abs(cal_usage - budget) / budget
                selection.append(dict(scenario=sc, policy=policy, budget=budget,
                    calibrated_factor=selected_factor, calibration_pod_minutes=cal_usage,
                    calibration_slo_rate=winner['eval_confirmed_slo_violation_rate'],
                    calibration_relative_cost_gap=cal_deviation,
                    calibration_match=cal_deviation <= args.tolerance))
                print(f'  target {budget:.0f} pod-min -> factor {selected_factor:g}, '
                      f'calibrated {cal_usage:.1f} pod-min', flush=True)
                for seed in range(args.validation_seed_start, args.validation_seed_start + args.validation_seeds):
                    r, timeline = evaluate(sc, policy, selected_factor, seed, cache, factories,
                                          timelines=args.timelines)
                    r['budget'] = budget
                    r['calibration_relative_cost_gap'] = cal_deviation
                    r['calibration_match'] = cal_deviation <= args.tolerance
                    r['heldout_relative_cost_gap'] = abs(r['eval_pod_minutes'] - budget) / budget
                    r['heldout_match'] = r['calibration_match'] and r['heldout_relative_cost_gap'] <= args.tolerance
                    heldout.append(r)
                    if args.timelines:
                        write_csv(output / f'{sc}_{policy}_b{budget:g}_seed{seed}_timeline.csv', timeline)
                    print(f'    held-out seed={seed}: SLO={r["eval_confirmed_slo_violation_rate"]:.3%}, '
                          f'eval pods={r["eval_pod_minutes"]:.1f}, matched={r["heldout_match"]}', flush=True)
    write_csv(output / 'calibration_grid.csv', calibration)
    write_csv(output / 'selected_factors.csv', selection)
    write_csv(output / 'heldout_trials.csv', heldout)
    grouped = defaultdict(list)
    for r in heldout:
        grouped[(r['scenario'],r['budget'],r['policy'])].append(r)
    summaries = []
    for (sc, budget, policy), group in grouped.items():
        vals = np.array([r['eval_confirmed_slo_violation_rate'] for r in group])
        resources = np.array([r['eval_pod_minutes'] for r in group])
        all_matched = all(r['heldout_match'] for r in group)
        summaries.append(dict(scenario=sc, target_eval_pod_minutes=budget, policy=policy,
            n=len(group), intensity_factor=group[0]['intensity_factor'],
            mean_slo_rate=float(vals.mean()), sd_slo_rate=float(vals.std(ddof=1)) if len(vals)>1 else '',
            mean_eval_pod_minutes=float(resources.mean()),
            mean_total_pod_minutes=float(np.mean([r['total_pod_minutes'] for r in group])),
            max_cost_deviation_fraction=float(max(r['heldout_relative_cost_gap'] for r in group)),
            resource_matched=all_matched))
    write_csv(output / 'matched_summary.csv', summaries)
    protocol = dict(paper_reproduction=False, original_policy_unchanged=False,
        study='Resource-response calibration using common policy intensity adjustment',
        calibration_seed=args.calibration_seed,
        validation_seed_start=args.validation_seed_start, validation_seeds=args.validation_seeds,
        scenarios=selected, policies=POLICY_NAMES, intensity_factors=factors,
        target_eval_pod_minutes=budgets, relative_tolerance=args.tolerance,
        evaluation_minutes=EVAL, warmup_minutes=WARMUP,
        common_max_replicas=32, startup_seconds=30, decision_interval_seconds=60,
        slo_seconds=0.5, mean_service_seconds=0.1,
        limitations=[
          'Resource matching is post-calibration nearest-neighbor; no hard cap is enforced.',
          'Budget matching must be verified on HELD-OUT seeds; unmatched cells are not fair comparisons.',
          'Intensity factors modify every policy; matched policies differ from original unmodified algorithms.',
          'Classifier and scaling policies are reconstructed; not faithful reproduction of published AAPA.',
          'Warm-up pod-minutes are tracked separately from evaluation pod-minutes.',
          'One synthetic workload realization per seed; only seed-to-seed uncertainty is measured.',
          'SLO rates exclude unresolved within-deadline cases from confirmed violations.',
        ])
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2),encoding='utf-8')
    print('\nSaved files under', output)
    print('Use matched_summary.csv ONLY where resource_matched=True; compare SLO rates within same scenario and budget.')
    print('NOT a faithful AAPA paper reproduction.')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['spike','periodic','ramp','stationary','all'],default='spike')
    p.add_argument('--calibration-seed',type=int,default=42)
    p.add_argument('--validation-seed-start',type=int,default=43)
    p.add_argument('--validation-seeds',type=int,default=3)
    p.add_argument('--budgets',default='100,160,240')
    p.add_argument('--factors',default=','.join(map(str,DEFAULT_FACTORS)))
    p.add_argument('--tolerance',type=float,default=0.10)
    p.add_argument('--timelines',action='store_true')
    run(p.parse_args())
