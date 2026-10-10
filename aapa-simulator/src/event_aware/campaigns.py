"""Causal, reproducible synthetic marketing campaign workload generation.

All times are minutes since simulation start. Demand truth and published plans
are deliberately separate. This module generates intensity traces, not request
arrivals; downstream experiments should replay identical arrivals per policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Optional
import numpy as np

WARMUP_MINUTES = 60
EVAL_MINUTES = 50
DURATION_MINUTES = WARMUP_MINUTES + EVAL_MINUTES
SCENARIOS = ('accurate', 'underestimated', 'overestimated', 'late_notice',
             'false_positive', 'unannounced', 'no_campaign')


@dataclass(frozen=True)
class Campaign:
    event_id: str
    event_type: str
    announced_minute: Optional[int]
    planned_start_minute: Optional[int]
    planned_duration_minutes: int
    planned_extra_rpm: float
    actual_start_minute: Optional[int]
    actual_duration_minutes: int
    actual_extra_rpm: float


def campaign_for(scenario: str) -> Optional[Campaign]:
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario {scenario!r}. Choose from {SCENARIOS}')
    if scenario == 'no_campaign':
        return None
    defaults = dict(event_id='campaign-001', event_type='digital_marketing',
                    announced_minute=65, planned_start_minute=80,
                    planned_duration_minutes=10, planned_extra_rpm=1200.0,
                    actual_start_minute=80, actual_duration_minutes=10,
                    actual_extra_rpm=1200.0)
    overrides = {
        'accurate': {},
        'underestimated': {'actual_extra_rpm': 2200.0},
        'overestimated': {'actual_extra_rpm': 500.0},
        'late_notice': {'announced_minute': 78},
        'false_positive': {'actual_start_minute': None, 'actual_duration_minutes': 0,
                           'actual_extra_rpm': 0.0},
        'unannounced': {'announced_minute': None, 'planned_start_minute': None,
                        'planned_duration_minutes': 0, 'planned_extra_rpm': 0.0},
    }
    defaults.update(overrides[scenario])
    return Campaign(**defaults)


def public_metadata(campaign: Optional[Campaign], minute: int) -> dict:
    """Event plan visible at decision time, never actual outcomes.

    event_known means a plan has been published by this minute. event_active
    refers to *scheduled* activity, not the true unseen arrival surge.
    minutes_to_event is null before publication or after scheduled start.
    """
    invisible = dict(event_known=0, event_active=0, minutes_to_event=None,
                     event_type='none', planned_intensity=0.0,
                     planned_duration=0, planned_start_minute=None)
    if (campaign is None or campaign.announced_minute is None or
            minute < campaign.announced_minute or
            campaign.planned_start_minute is None):
        return invisible
    start = campaign.planned_start_minute
    end = start + campaign.planned_duration_minutes
    return dict(event_known=1, event_active=int(start <= minute < end),
                minutes_to_event=max(0, start-minute) if minute <= start else None,
                event_type=campaign.event_type,
                planned_intensity=campaign.planned_extra_rpm,
                planned_duration=campaign.planned_duration_minutes,
                planned_start_minute=start)


def generate(scenario: str, seed: int = 42, warmup: int = WARMUP_MINUTES,
             evaluation: int = EVAL_MINUTES) -> tuple[list[dict], dict]:
    if warmup != WARMUP_MINUTES or evaluation != EVAL_MINUTES:
        raise ValueError('Phase 3A fixed protocol uses 60 warm-up and 50 evaluation minutes')
    campaign = campaign_for(scenario)
    rng = np.random.default_rng(seed)
    # Low-amplitude background variation is identical across scenarios for a
    # given seed, making counterfactual event comparisons more interpretable.
    baseline = np.clip(360.0 + rng.normal(0.0, 12.0, DURATION_MINUTES), 1, None)
    rows = []
    for minute in range(DURATION_MINUTES):
        extra = 0.0
        if (campaign is not None and campaign.actual_start_minute is not None and
                campaign.actual_start_minute <= minute <
                campaign.actual_start_minute + campaign.actual_duration_minutes):
            extra = campaign.actual_extra_rpm
        public = public_metadata(campaign, minute)
        rows.append(dict(minute=minute,
                         phase='warmup' if minute < warmup else 'evaluation',
                         baseline_rpm=round(float(baseline[minute]), 6),
                         true_event_extra_rpm=extra,
                         demand_rpm=round(float(baseline[minute] + extra), 6),
                         **public))
    manifest = dict(schema_version='phase3a-v1', scenario=scenario, seed=seed,
                    workload_units='requests_per_minute', time_units='minutes_since_start',
                    warmup_minutes=warmup, evaluation_minutes=evaluation,
                    policy_observes=['minute', 'demand_rpm from past completed minutes',
                                     'event_known', 'event_active', 'minutes_to_event',
                                     'event_type', 'planned_intensity', 'planned_duration',
                                     'planned_start_minute'],
                    oracle_only=['true_event_extra_rpm', 'actual_start_minute',
                                 'actual_duration_minutes', 'actual_extra_rpm'],
                    campaign=asdict(campaign) if campaign else None,
                    notes=['The public event columns must not reveal any information before announced_minute.',
                           'True-event fields are ground truth for auditing and must never enter scaler inputs.',
                           'The intended scaling decision for minute t uses history strictly before t.',
                           'Workload trace expresses expected requests/minute; generate and replay arrivals separately.',
                           'Synthetic design is not an AAPA paper dataset or evidence of improved scaling.'])
    return rows, manifest


def save(rows: list[dict], manifest: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'workload.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Only public columns (never true demand at or after decision time) are
    # allowed in this event schedule. Current demand comes from prior arrivals.
    public_cols = ['minute', 'event_known', 'event_active', 'minutes_to_event',
                   'event_type', 'planned_intensity', 'planned_duration',
                   'planned_start_minute']
    with (output / 'published_event_context.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=public_cols)
        writer.writeheader()
        writer.writerows({k: row[k] for k in public_cols} for row in rows)
