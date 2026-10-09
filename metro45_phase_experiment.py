"""Run a declared future-phase arm through the preserved paired replay."""

import argparse
from pathlib import Path

import metro45_experiment as trial
from metro45_phase import rephase_bridge


def run(output, *, shift=-20, scenario='S1'):
    files = [Path(__file__).resolve(), Path(__file__).with_name('metro45_phase.py')]
    before = {str(path): trial.sha(path) for path in files}
    trial.rebuild_bridge_timetable = lambda *args, **kwargs: rephase_bridge(
        *args, shift_seconds=shift, **kwargs)
    result = trial.run(output, scenario=scenario, policy='160')
    after = {str(path): trial.sha(path) for path in files}
    receipt = dict(extension='future phase and free future internal departures',
        shift_seconds=shift, sources_before=before, sources_after=after,
        sources_preserved=before == after, result_status=result['status'])
    trial.write(Path(output) / 'phase_extension.json', receipt)
    assert before == after
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shift', type=int, default=-20)
    parser.add_argument('--scenario', choices=['S1', 'S2', 'S3'], default='S1')
    args = parser.parse_args()
    run(args.output, shift=args.shift, scenario=args.scenario)
