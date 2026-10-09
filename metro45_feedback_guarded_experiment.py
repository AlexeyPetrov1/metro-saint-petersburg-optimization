"""Preserved replay with prior-forecast provenance captured before startup."""

import argparse
import hashlib
import json
from pathlib import Path

from metro45_feedback_experiment import ROOT, run as run_feedback


def watched_paths(guard):
    files = [Path(__file__).resolve(), ROOT / 'metro45_feedback_experiment.py']
    if guard:
        cache = ROOT / 'data/processed/metro_daily_experiments_20261009/forecasts'
        files.extend(cache / f'ML_{hour:02}.json' for hour in (7, 8))
    return files


def run(output, *, arm='MPC_ML', guard=False, quality='strict', steps=20):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = watched_paths(guard)
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    try:
        run_feedback(output, arm=arm, guard=guard, quality=quality, steps=steps)
    finally:
        after = hashes()
        receipt = dict(startup_hashes_captured_before_replay=True,
                       before=before, after=after, sources_preserved=before == after)
        if output.is_dir():
            with (output / 'startup_history_guard.json').open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, indent=2)
    if before != after:
        raise ValueError('prior forecast history changed during replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['MPC_ML', 'MPC_BASELINE_FORECAST'], default='MPC_ML')
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--quality', choices=['strict', 'pareto'], default='strict')
    parser.add_argument('--steps', type=int, choices=range(1, 21), default=20)
    run(**vars(parser.parse_args()))
