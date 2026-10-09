"""Run paired, separately labelled within-minute arrival diagnostics."""

import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT/'data/processed/metro44_bridge_execution_v5'
sys.path.insert(0, str(SNAPSHOT))
from goal43_replay import save_run
from metro_ml_compare import _metrics
from passenger_groups import GroupedPassengerSimulation
from simulation import digest


def local(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    output = ROOT/'data/processed/metro44_arrival_sensitivity_20261009'
    output.mkdir(exist_ok=False)
    closed = ROOT/'data/processed/metro44_continuation_6h_closed_20261009'
    robust = ROOT/'data/processed/metro44_profile_6h_robustness_20261009'
    bridge = ROOT/'data/processed/metro44_bridge_off160_actual_6h_20261009_v2'
    physical = ROOT/'data/processed/metro44_bridge_off160_6h_20261009'
    bases = {'S1': (closed, 'keep_result.json'), 'S2': (robust, 'keep_s2_result.json'),
             'S3': (robust, 'keep_s3_result.json')}
    jitter = local('metro44_arrivals').jitter_arrivals
    journey = local('metro44_journey').cohort_journey
    bounds = local('metro44_window').paired_window_bounds
    manifest = read(SNAPSHOT/'execution_manifest.json')
    inputs = [physical/'candidate_plan.json.gz', physical/'candidate_motion.json.gz']
    sources = [SNAPSHOT/name for name in manifest['source_hashes']]
    sources += [ROOT/f'{name}.py' for name in ('metro44_sensitivity','metro44_arrivals','metro44_journey','metro44_window')]
    for folder, name in bases.values():
        inputs.append(folder/name)
        run = folder/read(folder/name)['run']['folder']
        inputs += [run/name for name in ('movement_input.json','motion.json','groups.json.gz',
                                         'profiles.json','passenger_checkpoint.json.gz')]
    for od in bases:
        path = bridge/f'{od.lower()}_result.json'; inputs.append(path)
        run = bridge/read(path)['run']['folder']
        inputs += [run/name for name in ('passenger_checkpoint.json.gz','motion.json')]
    hashes = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs+sources}
    before = hashes(); write(output/'before.json', before)
    candidate = read(physical/'candidate_plan.json.gz'); candidate_motion = read(physical/'candidate_motion.json.gz')
    summary = []
    originals = []
    for od, (folder, name) in bases.items():
        base_result = read(folder/name); run = folder/base_result['run']['folder']
        data = read(run/'movement_input.json'); groups = read(run/'groups.json.gz'); profiles = read(run/'profiles.json')
        baseline = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles)
        alternate = GroupedPassengerSimulation(candidate, candidate['plans'][0]['plan_id'], groups, profiles)
        result = read(bridge/f'{od.lower()}_result.json'); other_run = bridge/result['run']['folder']
        state = read(run/'passenger_checkpoint.json.gz'); other_state = read(other_run/'passenger_checkpoint.json.gz')
        window = bounds(data, read(run/'motion.json'), candidate, candidate_motion, 3600, 21600)
        originals.append(dict(od=od, window_bounds=window,
            journeys={arm: journey(runner, checkpoint, 3600, 21000) for arm,runner,checkpoint in
                (('KEEP',baseline,state),('BRIDGE160',alternate,other_state))}))
        roster = [dict(train_id=t['train_id'],type=t['train_type']) for t in data['trains']]
        census = lambda rows: Counter((g['group_id'],g['count'],g['turnstile_second']//60) for g in rows)
        for seed in (7, 11):
            changed = jitter(groups, seed); assert census(groups) == census(changed)
            old = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], changed, profiles)
            new = GroupedPassengerSimulation(candidate, candidate['plans'][0]['plan_id'], changed, profiles)
            began = time.monotonic(); warmup = old.run(end_second=6300)['checkpoint']
            branched = new.branch_checkpoint(warmup, old)
            arms = {}
            for arm, runner, checkpoint, movement in (('KEEP',old,warmup,data),('BRIDGE160',new,branched,candidate)):
                state = runner.run(start_second=6300, checkpoint=checkpoint, deadline=time.monotonic()+470)['checkpoint']
                metrics = _metrics(movement, runner, state, roster, data['headway_policy'], 3600, 21600)
                saved = save_run(output,f'JITTER{seed}_{arm}_{od}_7',movement,runner,state,roster,
                    data['headway_policy'],'within_minute_sensitivity',arm,od,7,full_day=False,
                    cohort_start=3600,cohort_end=21600,service_end_second=22200)
                arms[arm] = dict(metrics=metrics,journey=journey(runner,state,3600,21000),run=saved)
            row = dict(od=od,seed=seed,groups_digest=digest(changed),same_groups_across_arms=True,
                observed_counts_preserved=True,assumed_within_minute_times=True,arms=arms,
                mean_wait_delta_seconds=arms['BRIDGE160']['metrics']['mean_wait_seconds']-arms['KEEP']['metrics']['mean_wait_seconds'],
                p95_wait_delta_seconds=arms['BRIDGE160']['metrics']['p95_wait_seconds']-arms['KEEP']['metrics']['p95_wait_seconds'],
                elapsed_seconds=time.monotonic()-began)
            write(output/f'{od.lower()}_seed{seed}.json',row); summary.append(row)
            print(json.dumps({key:row[key] for key in ('od','seed','mean_wait_delta_seconds','p95_wait_delta_seconds','elapsed_seconds')}) ,flush=True)
            assert before == hashes()
    write(output/'original_window_and_journey.json', originals)
    write(output/'summary.json',dict(scope='paired sensitivity, separate from immutable observed-input experiment',
        seeds=[7,11],rows=[{k:v for k,v in row.items() if k!='arms'} for row in summary],
        input_and_source_hashes_preserved=True,cash_saving_rub=None,release_authorized=False))
    write(output/'after.json',hashes())


if __name__ == '__main__':
    main()
