"""Read saved journals independently; never rerun passengers or modify inputs."""

import gzip
import hashlib
import json
import sys
from pathlib import Path


def prefix_equal(first, second, end):
    return ([r for r in first if r['second'] < end]
            == [r for r in second if r['second'] < end])


def scoped_windows(windows, start, end):
    return [r for r in windows if start <= r['start'] and r['end'] <= end]


def failed_windows(windows):
    return [r for r in windows if r['status'] == 'fail']


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    snapshot = root / 'data/processed/metro44_bridge_execution_v5'
    source = root / 'data/processed/metro44_bridge_off160_actual_6h_20261009_v2'
    output = root / 'data/processed/metro44_bridge_off160_independent_audit_20261009_v2'
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(snapshot))
    from goal43_verify import verify
    from metro44_planner_noidle import terminal_total_audit
    from rolling import operational_exposure, operational_headways
    from simulation import digest

    def write(name, value):
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

    manifest = read(snapshot / 'execution_manifest.json')
    frozen = {name: sha(snapshot / name) for name in manifest['source_hashes']}
    assert frozen == manifest['source_hashes']
    protected = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    prior = read(source / 'before.json')
    assert prior == read(source / 'after.json') and prior['sources'] == frozen
    assert sha(source / 'passenger_replay.py') == prior['script']
    for name, value in prior['inputs'].items():
        assert sha(Path(name)) == value, name
    contract = read(source / 'contract.json')
    records = [read(source / f'{od.lower()}_result.json')['run'] for od in ('S1', 'S2', 'S3')]
    records = [dict(r, folder=str(source / r['folder'])) for r in records]
    write('runs.json', records)
    write('sourcebaseline_manifest.json', dict(workspace_root=str(snapshot), files=frozen))
    verified = verify(output)
    assert verified['status'] == 'passed' and verified['run_count'] == 3
    closed = root / 'data/processed/metro44_continuation_6h_closed_20261009'
    robust = root / 'data/processed/metro44_profile_6h_robustness_20261009'
    physical = root / 'data/processed/metro44_bridge_off160_6h_20261009'
    physical_plan = read(physical / 'candidate_plan.json.gz')
    physical_motion = read(physical / 'candidate_motion.json.gz')
    bases = {'S1': (closed, 'keep_result.json'), 'S2': (robust, 'keep_s2_result.json'),
             'S3': (robust, 'keep_s3_result.json')}
    rows, common_entries = [], None
    for od in ('S1', 'S2', 'S3'):
        result = read(source / f'{od.lower()}_result.json')
        folder = source / result['run']['folder']
        base_parent, base_file = bases[od]
        base_result = read(base_parent / base_file)
        base_folder = base_parent / base_result['run']['folder']
        data, motion = read(folder / 'movement_input.json'), read(folder / 'motion.json')
        groups, profiles = read(folder / 'groups.json.gz'), read(folder / 'profiles.json')
        state = read(folder / 'passenger_checkpoint.json.gz')
        base_data = read(base_folder / 'movement_input.json')
        base_motion = read(base_folder / 'motion.json')
        base_state = read(base_folder / 'passenger_checkpoint.json.gz')
        assert data == physical_plan and motion == physical_motion
        assert digest(data) == contract['physical_plan_digest'] == result['movement_digest']
        assert groups == read(base_folder / 'groups.json.gz') and digest(groups) == result['groups_digest']
        assert profiles == read(base_folder / 'profiles.json')
        assert all(data[k] == base_data[k] for k in ('trains', 'tracks', 'headway_policy'))
        entries = [{k: v for k, v in g.items() if k != 'destinations'} for g in groups]
        if common_entries is None:
            common_entries = entries
        assert entries == common_entries
        change = contract['as_of_second']
        prefixes = {k: prefix_equal(base_state[k], state[k], change)
                    for k in ('load_events', 'queue_events', 'board_events')}
        prefixes['motion_events'] = prefix_equal(base_motion['events'], motion['events'], change)
        assert all(prefixes.values()), (od, prefixes)
        policy = data['headway_policy']
        whole = operational_exposure(state, profiles, policy, 0, motion['duration_seconds'])
        control = operational_exposure(state, profiles, policy, 3600, 21600)
        assert whole == result['complete_episode_exposure']
        assert control == result['metrics']['policy_exposure']
        historical = failed_windows(whole['windows'])
        baseline_whole = operational_exposure(base_state, profiles, policy, 0, base_motion['duration_seconds'])
        old_bad = failed_windows(baseline_whole['windows'])
        assert historical == old_bad and all(r['end'] <= 3600 for r in historical)
        terminal = terminal_total_audit(data, motion)
        assert terminal['passed']
        headways = operational_headways(data, motion, 3600, 22200)
        assert headways['passed'] and result['planning_headways_passed']
        assert motion['checks']['peak_trains_on_line'] <= 53 and motion['obligations_closed']
        assert state['entered'] == state['completed'] and not (state['queued'] or state['approaching'] or state['onboard_count'])
        reconstructed = next(r for r in verified['runs'] if r['run_id'] == result['run']['run_id'])
        recorded_bad = [r for r in reconstructed['violations'] if r['kind'] == 'exceptional_time']
        assert len(recorded_bad) == len(historical)
        m = result['metrics']
        rows.append(dict(od=od, exact_groups_and_profiles=True, same_immutable_station_entry_fields=True,
            same_physical_plan_across_od=True, immutable_prefix_checks=prefixes, terminal_resources=terminal,
            controlled_window=[3600,21600], controlled_load_admissible=control['admissible'],
            controlled_bad_quarters=len(failed_windows(control['windows'])),
            whole_episode_admissible=whole['admissible'], unchanged_historical_failed_windows=historical,
            empty_tail_unknown_windows=[r for r in whole['windows'] if r['status'] == 'unknown'],
            all_passengers_completed=True, movement_obligations_closed=True,
            planning_headways_passed=True, train_km=m['train_km'], mean_wait_seconds=m['mean_wait_seconds'],
            p95_wait_seconds=m['p95_wait_seconds'], controlled_max_car_load=m['max_car_load'],
            controlled_max_queue=m['max_queue'], entered=state['entered'], completed=state['completed']))
    assert {name: sha(snapshot / name) for name in frozen} == frozen
    assert {str(p): sha(p) for p in source.rglob('*') if p.is_file()} == protected
    write('independent_audit.json', dict(status='passed', run_count=3, source_files_checked=len(frozen),
        immutable_actual_files_checked=len(protected), independent_verifier=str(snapshot/'goal43_verify.py'),
        independent_verifier_sha256=sha(snapshot/'goal43_verify.py'),
        terminal_auditor_sha256=sha(Path(sys.modules[terminal_total_audit.__module__].__file__)),
        sources_and_inputs_preserved=True, rows=rows, cash_saving_rub=None, release_authorized=False,
        scope_note='All three controlled windows pass; S3 whole episode retains two shared historical failed quarters.'))
    print(json.dumps(dict(status='passed',output=str(output),run_count=3,rows=rows),ensure_ascii=False))


if __name__ == '__main__':
    main()
