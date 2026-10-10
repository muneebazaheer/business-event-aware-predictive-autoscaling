"""Phase 3D-B: announcement timing sensitivity (synthetic, causal, paired).

Campaign onset: minute 80, unchanged truth for all notice schedules.
Announcements are visible only at controller decision ticks (60-second ticks).
Do not use this as a faithful reproduction of AAPA or resource-matched evidence.
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
from src.event_aware.campaigns import generate
from src.event_aware.policy import EventAwarePolicy, DecisionLoggingPolicy, PUBLIC_KEYS
from src.core.simulator_repaired import generate_replay
from examples.phase3c_compare import EvaluationSimulator, evaluation_metrics

WARMUP = 60
CAMPAIGN_START = 80
CAMPAIGN_DURATION = 10
PLANNED_EXTRA_RPM = 1200.0
# Seconds between publication and planned campaign onset. Negative = published late.
LEADS = {'15min': 900, '2min': 120, '30sec': 30, 'onset': 0,
         '1min_after': -60, 'no_notice': None}
VARIANTS = {'event_guardrail': True, 'event_masked': False}


def public_schedule(rows, announcement_lead_seconds):
    """Minute-resolution context, gated at exact publication second.

    The policy is queried at minute boundaries. A mid-minute publication is
    not visible at the previous tick; it becomes visible at the next tick.
    """
    announcement_at = (CAMPAIGN_START * 60 - announcement_lead_seconds
                       if announcement_lead_seconds is not None else None)
    result = []
    for row in rows:
        minute = int(row['minute'])
        tick_s = minute * 60
        visible = announcement_at is not None and tick_s >= announcement_at
        if visible:
            pub = dict(minute=minute, event_known=1,
                       event_active=int(CAMPAIGN_START <= minute < CAMPAIGN_START + CAMPAIGN_DURATION),
                       minutes_to_event=max(0, CAMPAIGN_START-minute) if minute <= CAMPAIGN_START else None,
                       event_type='digital_marketing',
                       planned_intensity=PLANNED_EXTRA_RPM,
                       planned_duration=CAMPAIGN_DURATION,
                       planned_start_minute=CAMPAIGN_START)
        else:
            pub = dict(minute=minute, event_known=0, event_active=0,
                       minutes_to_event=None, event_type='none', planned_intensity=0.0,
                       planned_duration=0, planned_start_minute=None)
        assert set(pub) == set(PUBLIC_KEYS)
        result.append(pub)
    return result


def save_csv(path, records):
    if not records:
        return
    columns = list(dict.fromkeys(key for r in records for key in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def simulate(rows, replay, public, use_event, timelines=False):
    policy = DecisionLoggingPolicy(EventAwarePolicy(public, use_event=use_event,
        use_guardrail=True, startup_seconds=30, decision_interval_seconds=60,
        lead_minutes=2))
    duration = len(rows) * 60
    sim = EvaluationSimulator(simpy.Environment(), policy,
        startup_seconds=30, scale_interval_seconds=60, slo_seconds=0.5,
        min_replicas=1, max_replicas=32,
        eval_begin=WARMUP * 60, eval_end=duration)
    total = sim.run(replay, duration)
    metrics = evaluation_metrics(sim, WARMUP*60, duration)
    decisions = policy.decisions
    timeline = sim.metrics.timeline
    # 'pre-scale' means a decision requested more than its observed reactive floor.
    event_decisions = [d for d in decisions if d['time_s'] < CAMPAIGN_START*60
                       and d['target_replicas'] > d['reactive_floor']]
    first_pre = min((d['time_s'] for d in event_decisions), default=None)
    # readiness is sampled at decision ticks, not a precise pod-ready event log.
    at_onset = next((t for t in timeline if abs(t['time_s']-CAMPAIGN_START*60) < 1e-7), None)
    pre_window = [t for t in timeline if (CAMPAIGN_START-15)*60 <= t['time_s'] < CAMPAIGN_START*60]
    # Approximate pre-event pod-minutes from 60s snapshots, provided to examine waste.
    pre_pod_snapshot_sum = sum(t['allocated_replicas'] for t in pre_window)
    metrics.update(total_pod_minutes=total['pod_minutes'],
        pre_campaign_first_extra_decision_s=first_pre,
        pre_campaign_first_extra_lead_seconds=(CAMPAIGN_START*60-first_pre if first_pre is not None else None),
        allocated_at_onset=at_onset['allocated_replicas'] if at_onset else None,
        ready_at_onset=at_onset['ready_replicas'] if at_onset else None,
        pre_event_15min_allocated_pod_minutes_snapshot=pre_pod_snapshot_sum,
        scaling_actions_whole_run=total['scaling_actions'])
    return metrics, timeline if timelines else None, decisions if timelines else None


def run(args):
    if args.seeds < 1:
        raise ValueError('--seeds must be >= 1')
    selected = list(LEADS) if args.leads == 'all' else [v.strip() for v in args.leads.split(',')]
    unknown = set(selected)-set(LEADS)
    if unknown:
        raise ValueError(f'Unknown notice conditions: {sorted(unknown)}')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate notice conditions')
    output = ROOT/'results'/'phase3db'
    output.mkdir(parents=True, exist_ok=True)
    trials, paired = [], []
    for seed in range(args.seed, args.seed + args.seeds):
        rows, _ = generate('accurate', seed=seed)
        replay = generate_replay([r['demand_rpm'] for r in rows], seed=seed,
                                 mean_service_seconds=0.1)
        for label in selected:
            pub = public_schedule(rows, LEADS[label])
            first_visible = next((p['minute']*60 for p in pub if p['event_known']), None)
            outcomes = {}
            for variant, enabled in VARIANTS.items():
                metrics, timeline, decisions = simulate(rows, replay, pub, enabled, args.timelines)
                record = dict(notice=label, announcement_lead_s=LEADS[label], seed=seed,
                              variant=variant, first_visible_decision_s=first_visible, **metrics)
                trials.append(record)
                outcomes[variant] = record
                if args.timelines:
                    save_csv(output/f'{label}_seed{seed}_{variant}_timeline.csv', timeline)
                    save_csv(output/f'{label}_seed{seed}_{variant}_decisions.csv', decisions)
                print(f'{label} seed={seed} {variant}: SLO={metrics["eval_confirmed_slo_rate"]:.3%} '
                      f'eval_pods={metrics["eval_pod_minutes"]:.1f} '
                      f'ready_at_start={metrics["ready_at_onset"]}', flush=True)
            # Strong negative control: no notice = same output as event-masked.
            on, off = outcomes['event_guardrail'], outcomes['event_masked']
            if label == 'no_notice':
                keys = ('eval_generated','eval_confirmed_slo_violations','eval_pod_minutes',
                        'scaling_actions_whole_run','ready_at_onset')
                assert all(on[k] == off[k] for k in keys), 'No-notice control mismatch'
            paired.append(dict(notice=label, seed=seed,
                event_slo_rate_delta=on['eval_confirmed_slo_rate']-off['eval_confirmed_slo_rate'],
                event_pod_min_delta=on['eval_pod_minutes']-off['eval_pod_minutes'],
                event_ready_at_onset_delta=on['ready_at_onset']-off['ready_at_onset']))
    grouped = defaultdict(list)
    for row in trials:
        grouped[(row['notice'], row['variant'])].append(row)
    aggregate=[]
    for (notice,variant), group in sorted(grouped.items()):
        def mean(k): return float(np.mean([r[k] for r in group]))
        aggregate.append(dict(notice=notice, variant=variant, n=len(group),
            mean_slo_rate=mean('eval_confirmed_slo_rate'),
            sd_slo_rate=float(np.std([r['eval_confirmed_slo_rate'] for r in group],ddof=1)) if len(group)>1 else '',
            mean_eval_pod_minutes=mean('eval_pod_minutes'),
            mean_ready_at_onset=mean('ready_at_onset')))
    save_csv(output/'trials.csv', trials)
    save_csv(output/'aggregate.csv', aggregate)
    save_csv(output/'paired_timing.csv', paired)
    protocol=dict(phase='3D-B', study='announcement lead-time sensitivity',
        paper_reproduction=False, scenario='accurate', event_true_start_minute=CAMPAIGN_START,
        event_true_duration_minutes=CAMPAIGN_DURATION, event_planned_extra_rpm=PLANNED_EXTRA_RPM,
        notice_leads_seconds={k:v for k,v in LEADS.items()}, selected_conditions=selected,
        seeds=list(range(args.seed,args.seed+args.seeds)), evaluation_minutes=50,
        warmup_minutes=60, decision_interval_seconds=60, startup_seconds=30,
        limitations=['Public event context is sampled at one-minute decision ticks; publication is enforced in seconds.',
                     '15-minute and 2-minute notices may be equivalent because the policy only pre-scales two minutes early.',
                     '30-second and onset notices may be equivalent due to the decision interval.',
                     'Ready replicas measured from minute snapshots, not exact ready-transition timestamps.',
                     'Pre-event allocated pod-minute metric is snapshot-based, not an exact time integral.',
                     'Single fixed marketing campaign schedule; synthetic workloads; no statistical significance claim.',
                     'Policies share capacity constraints but do not necessarily spend equal pod-minutes.'])
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Saved:',output)
    print('Exploratory causal information-timing comparisons; not resource-matched or paper reproduction.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--leads',default='all',help='all or comma-separated: '+','.join(LEADS))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--timelines',action='store_true')
    run(p.parse_args())
