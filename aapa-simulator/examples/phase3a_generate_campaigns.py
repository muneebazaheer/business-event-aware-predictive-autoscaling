"""Create reproducible Phase 3A marketing campaign traces."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.event_aware.campaigns import SCENARIOS, generate, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=(*SCENARIOS, 'all'), default='all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='results/phase3a')
    args = p.parse_args()
    selected = SCENARIOS if args.scenario == 'all' else (args.scenario,)
    for name in selected:
        rows, manifest = generate(name, args.seed)
        dest = ROOT / args.output_dir / f'{name}_seed{args.seed}'
        save(rows, manifest, dest)
        print(f'{name}: {len(rows)} minutes | eval peak={max(x["demand_rpm"] for x in rows[60:]):.1f} rpm | {dest}')
    print('Generated traces only; no autoscaling performance is claimed.')


if __name__ == '__main__':
    main()
