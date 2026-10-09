"""Resume saved operational decisions without replaying their archived alternatives."""

import copy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from economics import write_json
from goal43_actions import attach_operational_depots
from goal43_controller import run_controller
from goal43_economics import compare_train_economics
from goal43_notifications import analytics
from goal43_replay import ROOT, actual_groups, fixture_history, mileage_context, save_run, _series
from passenger_groups import GroupedPassengerSimulation
from simulation import digest


def _read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8'))


def _ready(data, roster, policy):
    stripped = copy.deepcopy(data)
    depots = {row['station_id'] for row in data['stations'] if row['kind'] == 'depot'}
    stripped['stations'] = [row for row in data['stations'] if row['station_id'] not in depots]
    stripped['tracks'] = [row for row in data['tracks'] if not depots & {row['from_station'],row['to_station']}]
    _, ready = attach_operational_depots(stripped, roster, policy)
    return ready


def _entry_digest(groups):
    rows = copy.deepcopy(groups)
    for row in rows:
        for destination in row['destinations']:
            destination.pop('direction', None)
    return digest(sorted(rows, key=lambda row:(row['turnstile_second'],row['group_id'])))


def refresh_saved_series(output):
    """Refresh graph headways from primary OD journals; repeat after late producers finish."""
    from types import SimpleNamespace
    from goal43_replay import _series as current_series
    output = Path(output)
    counts = dict(comparisons_updated=0, scenarios_updated=0, existing_series_files_updated=0)
    def headways(folder, bundle, assumptions):
        summary = _read(folder/'summary.json')
        scope = summary['cohort_scope']
        service_end = summary.get('service_scope', {}).get('end_second', assumptions.get('service_end_second'))
        runner = SimpleNamespace(motion=_read(folder/'motion.json'), by_id={})
        values = current_series(runner, dict(load_events=[],queue_samples=[]),
            scope['start_second'], scope['end_second'], bundle, _read(folder/'movement_input.json'),
            service_end_second=service_end)
        return {key:values[key] for key in ('headway','headway_label')}
    def write(path, value):
        temporary = path.with_name(path.name+'.series.tmp')
        write_json(temporary, value)
        temporary.replace(path)
    for path in sorted((output/'scenarios').glob('*/summary.json')):
        public = _read(path)
        if not public.get('comparisons'):
            continue
        bundle = _read(path.parent/'forecast.json'); changed = False
        assumptions_path = path.parent/'assumptions.json'
        assumptions = _read(assumptions_path) if assumptions_path.exists() else {}
        for comparison in public.get('comparisons', []):
            suffix = comparison['candidate_id'].replace(':','_').replace('+','-')
            primary = list((output/'runs').glob(f'{path.parent.name}_{suffix}_S1_*/summary.json'))
            if len(primary) != 1:
                continue
            comparison.setdefault('series', {}).update(headways(primary[0].parent, bundle, assumptions))
            counts['comparisons_updated'] += 1; changed = True
        if changed:
            write(path, public); counts['scenarios_updated'] += 1
    for path in (output/'runs').glob('*/series.json'):
        summary = _read(path.parent/'summary.json')
        scenario = output/'scenarios'/summary['scenario_id']
        bundle = _read(scenario/'forecast.json')
        assumptions = _read(scenario/'assumptions.json') if (scenario/'assumptions.json').exists() else {}
        series = _read(path); series.update(headways(path.parent, bundle, assumptions)); write(path, series)
        counts['existing_series_files_updated'] += 1
    return counts


def extend_fixture(output, scenario_id='fixture_offpeak', *, steps=2, minimum_remaining=36,
                   count_per_minute=2):
    public = _read(Path(output)/'scenarios'/scenario_id/'summary.json')
    if public.get('evaluation_mode') != 'fixture' or public.get('incident') is not None:
        raise ValueError('extension supports the saved offpeak fixture only')
    return _extend_saved(output, scenario_id, public, steps=steps, minimum_remaining=minimum_remaining,
                         count_per_minute=count_per_minute)


def resume_stop(output, scenario_id='station_stop_8', *, steps=4, minimum_remaining=None, frame=None):
    """Resume a known-duration incident from saved alternatives and the first live decision."""
    folder = Path(output)/'scenarios'/scenario_id
    decision, meta = _read(folder/'decision.json'), _read(folder/'assumptions.json')
    estimates = decision.get('incident_estimates', [])
    if len(estimates) != 1 or type(estimates[0].get('expected_end_second')) is not int:
        raise ValueError('stop resume requires one known-duration first incident')
    incident = copy.deepcopy(estimates[0])
    if incident['detected_second'] != decision['decision_second'] or incident['expected_end_second'] <= incident['detected_second']:
        raise ValueError('saved incident must begin at the first decision and have a later recovery')
    if frame is None:
        import pandas as pd
        refreshed = ROOT/'data/processed/goal4_3_source_refresh/entries_station_15m.parquet'
        frame = pd.read_parquet(refreshed if refreshed.exists() else ROOT/'data/processed/metro/entries_station_15m.parquet')
    public = dict(scenario_id=scenario_id, title=scenario_id,
        evaluation_mode='historical_entries_injected_incident_assumed_timetable',
        incident=incident, injected_actual_end_second=incident['expected_end_second'], estimate_mode='known',
        decisions=[], comparisons=[], full_day=meta['full_day'], selection=decision['selection'],
        elapsed_seconds=0, positive_historical_economy_confirmed=False, release_authorized=False)
    public['elapsed_scope'] = 'archive reconstruction and remaining control; initial alternatives excluded'
    floor = max(1, meta['train_count']-2) if minimum_remaining is None else minimum_remaining
    return _extend_saved(output, scenario_id, public, steps=steps, minimum_remaining=floor,
                         frame=frame, incident=incident)


def _comparison(record, economy, candidate_id, strategy, status, elapsed, series, source):
    return dict(strategy=strategy, candidate_id=candidate_id, status=status, elapsed_seconds=elapsed,
        metrics=dict(mean_wait_seconds=record['metrics']['cohort']['mean_wait_seconds'],
            p95_wait_seconds=record['metrics']['cohort']['p95_wait_seconds'],
            max_car_load=record['metrics']['max_car_load'],
            exceptional_time_share=record['metrics']['policy_exposure']['global']['exceptional_time_share'],
            train_km=record['metrics']['resources']['train_km'], saving_km=economy['totals']['saving_km'],
            to1_resource_value_rub=economy['totals']['to1_resource_value_rub'], lost_revenue_rub=None),
        series=series,
        per_train=[{key:row[key] for key in ('train_id','train_type','baseline_km','candidate_km','saving_km')}
                   for row in economy['per_train'].values()], source=source, release_authorized=False)


def _extend_saved(output, scenario_id, public, *, steps, minimum_remaining, count_per_minute=2,
                  frame=None, incident=None):
    began = perf_counter()
    output = Path(output).resolve()
    folder = output/'scenarios'/scenario_id
    if type(steps) is not int or steps < 2:
        raise ValueError('extension requires at least two control steps')
    decision = _read(folder/'decision.json')
    bundle, ledger = _read(folder/'forecast.json'), _read(folder/'notifications.json')
    roster, meta = _read(folder/'fleet_registry.json'), _read(folder/'assumptions.json')
    baselines = {}
    for path in (output/'runs').glob(f'{scenario_id}_*/summary.json'):
        summary = _read(path)
        if summary['scenario_id'] == scenario_id and summary['strategy'] == 'KEEP':
            baselines[summary['od']] = (path.parent, summary)
    if set(baselines) != {'S1','S2','S3'}:
        raise ValueError('saved KEEP baseline must cover all three OD scenarios')
    seed = baselines['S1'][1]['seed']
    new_ids = [f'{scenario_id}_ROLLING_SELECTED_{od}_{seed}' for od in baselines]
    if any((output/'runs'/run_id).exists() for run_id in new_ids):
        raise ValueError('rolling extension already exists')
    old_files = {path: sha256(path.read_bytes()).hexdigest()
                 for path in (output/'runs').glob(f'{scenario_id}_*/*') if path.is_file()}
    data = _read(baselines['S1'][0]/'movement_input.json')
    policy = data['headway_policy']
    profiles = _read(baselines['S1'][0]/'profiles.json')
    groups = {od: _read(path/'groups.json.gz') for od,(path,_) in baselines.items()}
    station_ids = [row['station_id'] for row in data['stations'] if row['kind'] == 'line']
    if frame is None:
        frame = fixture_history(station_ids, count_per_minute=count_per_minute)
    expected = actual_groups(frame, data, meta['service_end_second'])
    if any(_entry_digest(groups[od]) != _entry_digest(expected[od]) for od in groups):
        raise ValueError('saved actual groups do not match the declared entry source')
    sources = {od: GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], rows, profiles)
               for od,rows in groups.items()}
    second = decision['decision_second']
    for od, source in sources.items():
        saved_motion = _read(baselines[od][0]/'motion.json')
        original = [row for row in saved_motion['events'] if row['second'] < second]
        rebuilt = [row for row in source.motion['events'] if row['second'] < second]
        if original != rebuilt:
            raise ValueError('rebuilt movement does not match the authoritative saved past')
    snapshots = {od: runner.run(end_second=second)['checkpoint'] for od,runner in sources.items()}
    candidates = []
    for row in decision['candidate_table']:
        suffix = row['candidate_id'].replace(':','_').replace('+','-')
        path = output/'runs'/f'{scenario_id}_{suffix}_S1_{seed}'/'movement_input.json'
        candidate = {key:copy.deepcopy(row[key]) for key in ('candidate_id','strategy','status','action','timing') if key in row}
        candidates.append(dict(candidate, _data=_read(path) if path.exists() else None))
    chosen = decision['selection'].get('selected_candidate_id') or decision['selection'].get('mitigation_candidate_id')
    if chosen is not None and not any(row['candidate_id'] == chosen and row['_data'] for row in candidates):
        raise ValueError('saved selected candidate movement is missing')
    first = dict(decision=decision, candidates=candidates, bundle=bundle, ledger=ledger)
    controlled = run_controller(data, sources, snapshots, profiles, _ready(data, roster, policy), frame, policy,
        scenario_id=scenario_id, minimum_remaining=minimum_remaining, groups=groups,
        incident=incident, actual_end_second=incident['expected_end_second'] if incident else None,
        steps=steps, profile=meta['timing']['profile'], seed=seed, first_live=first)
    records, contexts = [], {}
    for od, runner in controlled['sources'].items():
        state = controlled['states'][od]
        record = save_run(output, f'{scenario_id}_ROLLING_SELECTED_{od}_{seed}', controlled['data'], runner,
            state, roster, policy, scenario_id, 'ROLLING_SELECTED', od, seed, full_day=meta['full_day'],
            cohort_start=second, cohort_end=second+7200, service_end_second=meta['service_end_second'])
        records.append(record)
        contexts[od] = mileage_context(controlled['data'], runner.motion, roster, digest(groups[od]),
                                       scenario_id if incident else 'none', seed, full_day=meta['full_day'])
    baseline_context = mileage_context(data, _read(baselines['S1'][0]/'motion.json'), roster,
                                       digest(groups['S1']), scenario_id if incident else 'none', seed, full_day=meta['full_day'])
    fleet = {row['train_id']:row['type'] for row in roster}
    economic_path = folder/'economics.json'
    economic_rows = _read(economic_path) if economic_path.exists() else []
    if incident and not public['comparisons']:
        for candidate in candidates:
            suffix = candidate['candidate_id'].replace(':','_').replace('+','-')
            saved = output/'runs'/f'{scenario_id}_{suffix}_S1_{seed}'
            if not (saved/'summary.json').exists():
                continue
            old_summary, old_motion = _read(saved/'summary.json'), _read(saved/'motion.json')
            context = mileage_context(candidate['_data'], old_motion, roster, digest(groups['S1']), scenario_id,
                                       seed, full_day=meta['full_day'])
            old_economy = compare_train_economics(baseline_context, context, fleet, policy=policy)
            old_runner = GroupedPassengerSimulation(candidate['_data'], candidate['_data']['plans'][0]['plan_id'], groups['S1'], profiles)
            old_state = _read(saved/'passenger_checkpoint.json.gz')
            series = _series(old_runner, old_state, second, second+7200, bundle, candidate['_data'])
            public['comparisons'].append(_comparison(old_summary, old_economy, candidate['candidate_id'],
                candidate['strategy'], candidate['status'], decision['elapsed_seconds'], series, public['evaluation_mode']))
            economic_rows.append(dict(candidate_id=candidate['candidate_id'], **old_economy))
    economy = compare_train_economics(baseline_context, contexts['S1'], fleet, policy=policy)
    record = next(row for row in records if row['od'] == 'S1')
    series = _series(controlled['sources']['S1'], controlled['states']['S1'], second, second+7200, bundle, controlled['data'])
    comparison = _comparison(record, economy, 'ROLLING_SELECTED', 'ROLLING_SELECTED',
        controlled['decisions'][-1]['selection']['status'], max(row['elapsed_seconds'] for row in controlled['decisions']),
        series, public['evaluation_mode'])
    public['comparisons'] = [row for row in public['comparisons'] if row['candidate_id'] != 'ROLLING_SELECTED']+[comparison]
    public.update(decisions=controlled['decisions'], rolling_selection=controlled['decisions'][-1]['selection'],
                  extension_elapsed_seconds=perf_counter()-began, saved_first_decision_reused=True)
    public['elapsed_seconds'] += public['extension_elapsed_seconds']
    economic_rows = [row for row in economic_rows if row['candidate_id'] != 'ROLLING_SELECTED']
    economic_rows.append(dict(candidate_id='ROLLING_SELECTED', **economy))
    for name, value in [('decisions',controlled['decisions']), ('forecasts',controlled['bundles']),
                        ('controller_trace',controlled['trace']), ('notifications',controlled['ledger']),
                        ('notification_analytics',analytics(controlled['ledger'])),
                        ('economics',economic_rows), ('summary',public)]:
        write_json(folder/f'{name}.json',value)
    if any(sha256(path.read_bytes()).hexdigest() != value for path,value in old_files.items()):
        raise AssertionError('extension changed an existing saved run')
    result = dict(summary=public, records=records, past_verified=True, saved_first_decision_reused=True,
                  original_run_files_unchanged=True, elapsed_seconds=perf_counter()-began)
    write_json(folder/'extension.json', {key:value for key,value in result.items() if key not in {'summary','records'}})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/processed/goal4_3'))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--resume-stop',metavar='SCENARIO')
    mode.add_argument('--refresh-series',action='store_true')
    args=parser.parse_args()
    if args.refresh_series:
        print(json.dumps(refresh_saved_series(args.output)))
        raise SystemExit(0)
    result=resume_stop(args.output,args.resume_stop) if args.resume_stop else extend_fixture(args.output)
    print(json.dumps(dict(seconds=result['elapsed_seconds'],runs=len(result['records']),
        decision_seconds=[row['elapsed_seconds'] for row in result['summary']['decisions']],
        selection=result['summary']['rolling_selection'],metrics=result['summary']['comparisons'][-1]['metrics'])))
