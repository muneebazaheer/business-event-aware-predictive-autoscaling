"""Run a paired synthetic smoke test on the repaired simulation engine.
Do not report these results as reproduced AAPA paper measurements.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import simpy
from src.core.simulator_repaired import (
    RepairedSimulator, NormalizedHPA, NormalizedPredictive, generate_replay
)


def main():
    workload = np.concatenate([
        np.full(10, 10.0), np.full(3, 150.0),
        np.full(10, 10.0), np.full(3, 200.0),
        np.full(10, 10.0), np.full(2, 180.0),
        np.full(12, 10.0),
    ])
    # Higher load only for concurrency stress-testing; not paper evaluation.
    workload *= 12
    events = generate_replay(workload, seed=42)
    output = ROOT / 'results' / 'phase2_repaired_smoke'
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, policy in [
        ('normalized_hpa', NormalizedHPA()),
        ('normalized_predictive', NormalizedPredictive()),
    ]:
        sim = RepairedSimulator(
            env=simpy.Environment(), policy=policy,
            startup_seconds=30, scale_interval_seconds=60,
            slo_seconds=0.5, max_replicas=100,
        )
        metrics = sim.run(events, len(workload) * 60)
        summary = metrics.as_dict()
        results[name] = summary
        assert summary['generated'] == len(events)
        assert summary['completed'] + summary['unfinished_at_end'] == summary['generated']
        with (output / f'{name}_timeline.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.timeline[0]))
            writer.writeheader()
            writer.writerows(metrics.timeline)
        print(f'\n{name}:')
        for key, value in summary.items():
            print(f'  {key}: {value}')
    with (output / 'summary.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {output}')
    print('WARNING: reconstructed simulator + synthetic stress test; not paper reproduction.')


if __name__ == '__main__':
    main()
