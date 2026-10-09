"""Independent checks of saved cohorts, ledgers, comparisons and preservation."""

import gzip
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd

from decision_engine import prepare_demand
from dwell import compile_timing
from line1 import car_profiles, load_config
from metro_replay import check
from passenger_groups import GroupedPassengerSimulation
from passengers import PassengerSimulation
from simulation import simulate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/processed/goal4_2'
PROOF = ROOT / 'data/processed/goal4_2_verification'


def read(path):
    content = path.read_bytes()
    return json.loads(gzip.decompress(content) if path.suffix == '.gz' else content.decode('utf-8-sig'))


def independent_overload(state, start=7200, end=21600):
    by_car = defaultdict(list)
    for row in state['load_events']: by_car[row['train_id'], row['car_id']].append(row)
    seconds = occupied = excess = peak = 0
    for rows in by_car.values():
        for i, row in enumerate(rows):
            stop = rows[i+1]['second'] if i+1 < len(rows) else end
            width = max(0, min(stop, end)-max(row['second'], start))
            if width: peak = max(peak, row['load'])
            if row['load'] > row['normal_threshold']:
                seconds += width
                occupied += width*row['load']
                excess += width*(row['load']-row['normal_threshold'])
    return dict(max_car_load=peak, exceeded_car_seconds=seconds,
        passenger_seconds_in_exceeded_cars=occupied, excess_passenger_seconds=excess)


def independent_cohort(state, ids, arrived):
    histogram = defaultdict(int)
    for row in state['board_events']:
        if row['group_id'] in ids: histogram[row['wait_seconds']] += row['count']
    boarded = sum(histogram.values())
    assert boarded <= arrived
    waiting = sum(w*n for w,n in histogram.items())
    residual = sum(g['count']*(state['second']-g['platform_second'])
        for queue in state['queues'].values() for g in queue if g['group_id'] in ids)
    p95 = None
    if boarded == arrived and arrived:
        cumulative = 0
        for wait, count in sorted(histogram.items()):
            cumulative += count
            if cumulative*100 >= arrived*95: p95 = wait; break
    return dict(arrived=arrived, boarded=boarded, unboarded=arrived-boarded,
        waiting_passenger_seconds_lower_bound=waiting+residual,
        mean_wait_seconds=waiting/arrived if boarded == arrived and arrived else None,
        p95_wait_seconds=p95)


def verify_motion(data, motion):
    by_track = defaultdict(list)
    for row in data.get('previous_departures', []): by_track[row['track_id']].append(row['departure_second'])
    for train in data['trains']:
        if train['state'] == 'moving':
            track = next(t for t in data['tracks'] if t['track_id'] == train['location'])
            by_track[train['location']].append(train['initial_movement']['arrival_second']-track['travel_seconds'])
    for event in motion['events']:
        if event['event_type'] == 'departure': by_track[event['location']].append(event['second'])
    gaps = [b-a for times in by_track.values() for a,b in zip(sorted(set(times)), sorted(set(times))[1:])]
    assert min(gaps) >= 113
    assert motion['checks']['peak_trains_on_line'] <= 53
    return min(gaps)


def verify_work(data, motion, resources, start=7200, end=21600):
    tracks = {t['track_id']:t for t in data['tracks']}
    trips = {t['trip_id']:t for p in data['plans'] for t in p['trips']}
    trains = {t['train_id']:t for t in data['trains']}
    current = {t['train_id']:t['location'] for t in data['trains'] if t['state'] == 'moving'}
    work = defaultdict(Decimal)
    for event in motion['events']:
        train = event['train_id']
        if event['event_type'] == 'departure': current[train] = event['location']
        elif event['event_type'] == 'arrival' and start < event['second'] <= end:
            passenger = trips[event['trip_id']]['passenger'] if event['trip_id'] != '@initial' else trains[train]['initial_movement']['passenger']
            prefix = 'passenger' if passenger else 'nonpassenger'
            distance = Decimal(str(tracks[current[train]]['length_km']))
            work[prefix+'_train_km'] += distance
            work[prefix+'_car_km'] += distance*trains[train]['cars']
    for key in ['passenger_train_km','nonpassenger_train_km','passenger_car_km','nonpassenger_car_km']:
        assert Decimal(str(resources[key])) == work[key]
    assert sum(resources['state_seconds'].values()) == len(trains)*(end-start)


def verify_saved_run(folder, groups):
    state = read(folder/'final_checkpoint.json.gz'); summary = read(folder/'summary.json')
    assert state['sha256'] == PassengerSimulation._seal(state)
    assert state['entered'] == state['approaching']+state['queued']+state['onboard_count']+state['completed']
    assert state['queued'] == sum(g['count'] for q in state['queues'].values() for g in q)
    assert state['onboard_count'] == sum(sum(loads) for loads in state['loads'].values())
    assert all(row['load'] <= row['max_capacity'] for row in state['load_events'])
    ids = {g['group_id'] for g in groups if 7200 <= g['turnstile_second'] < 14400}
    arrived = sum(g['count'] for g in groups if g['group_id'] in ids)
    for key, value in independent_cohort(state, ids, arrived).items(): assert summary['cohort'][key] == value
    for key, value in independent_overload(state).items(): assert summary['overload'][key] == value
    assert summary['resources']['cash_total_RUB'] is None and not summary['release_authorized']
    return arrived


def main():
    result = dict(manifest=check(OUTPUT), preserved={}, verified_runs=0)
    before = read(PROOF/'preserved_before.json')
    prefixes = ('data/processed/economics', 'data/processed/simulation', 'data/processed/passengers',
        'data/processed/line1', 'data/processed/decision/', 'data/processed/decision_repeat/', 'data/raw/metro/')
    for name, expected in before.items():
        if not name.startswith(prefixes): continue
        path = ROOT/name
        if not path.exists() and 'Данные Сириус_old' in name:
            path = ROOT/name.replace('data/raw/metro/Данные Сириус_old', 'data/archive/Данные Сириус_old')
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        group = name.split('/')[2]
        result['preserved'][group] = result['preserved'].get(group, 0)+1
    config = load_config(); frame = pd.read_parquet(ROOT/'data/processed/metro/entries_station_15m.parquet')
    groups, _, _ = prepare_demand(frame, config)
    parameter = read(ROOT/'data/processed/metro/parameters.json')['parameters']['reference_roundtrip']
    assert parameter['value'] is None and parameter['status'] == 'withdrawn'
    table = pd.read_csv(OUTPUT/'forecast_vs_actual.csv')
    actual = frame.set_index(['station_id','interval_start']).entry_count
    for row in table.itertuples(): assert int(actual.loc[row.station_id,pd.Timestamp(row.interval_start)]) == row.actual
    wape = float((table.actual-table.predicted).abs().sum()/table.actual.sum())
    assert len(table) == 152 and table.actual.sum() == 116588
    assert wape == read(OUTPUT/'forecast_evaluation.json')['WAPE']
    result['forecast_WAPE'] = wape
    modes = ['A_fixed','B_fixed','A_mild','B_mild','D_mild']
    for mode in modes:
        folder = OUTPUT/'episodes'/mode
        assert len(list(folder.glob('decision_*.json'))) == 8
        data, motion = read(folder/'plan.json'), read(folder/'motion.json')
        verify_motion(data, motion)
        for scenario in groups:
            assert verify_saved_run(folder/scenario, groups[scenario]) == 116588
            verify_work(data,motion,read(folder/scenario/'summary.json')['resources'])
            result['verified_runs'] += 1
            for name in ['checkpoint_08.json.gz','checkpoint_10.json.gz']:
                cp = read(folder/scenario/name)
                assert cp['entered'] == cp['approaching']+cp['queued']+cp['onboard_count']+cp['completed']
        for path in folder.glob('decision_*.json'):
            decision = read(path)
            if 'forecast_issued_at' in decision:
                assert pd.Timestamp(decision['forecast_issued_at']) <= pd.Timestamp(decision['decision_time'])
            for row in decision['candidate_table']:
                if row['status'] == 'evaluated':
                    assert row['trials'] == len(row['tested_seeds'])*3
                else: assert row['cash_delta_RUB'] is None
    for scenario in groups:
        for equivalent in [('A_fixed','B_fixed'),('A_mild','B_mild','D_mild')]:
            first = read(OUTPUT/'episodes'/equivalent[0]/scenario/'checkpoint_08.json.gz')
            assert all(read(OUTPUT/'episodes'/m/scenario/'checkpoint_08.json.gz') == first for m in equivalent[1:])
    seed_rows = read(OUTPUT/'seed_results.json')
    assert len(seed_rows) == 120
    preliminary = {r['seed'] for r in seed_rows if r['stage'] == 'preliminary'}
    validation = {r['seed'] for r in seed_rows if r['stage'] == 'validation'}
    assert len(preliminary) == 10 and len(validation) == 20 and not preliminary & validation
    assert len({(r['stage'],r['profile'],r['seed'],r['scenario']) for r in seed_rows}) == 120
    for row in seed_rows:
        verify_saved_run(OUTPUT/'seed_campaign'/f"{row['stage']}_{row['profile']}_{row['seed']}"/row['scenario'], groups[row['scenario']])
        result['verified_runs'] += 1
    late_folder = PROOF/'selected_validation'
    late_manifest = read(late_folder/'build_manifest.json')
    assert late_manifest['status'] == 'success'
    for name, expected in late_manifest['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    for name, expected in late_manifest['outputs'].items():
        assert hashlib.sha256((late_folder/name).read_bytes()).hexdigest() == expected
    late_rows = read(late_folder/'results.json')
    assert len(late_rows) == 60
    assert {r['seed'] for r in late_rows} == validation
    assert len({(r['seed'],r['scenario']) for r in late_rows}) == 60
    late_data = read(OUTPUT/'episodes/D_mild/plan.json')
    late_profiles = car_profiles(config,[t['train_id'] for t in late_data['trains']])
    for scenario in groups:
        snapshot = read(late_folder/f'{scenario}_snapshot_0945.json.gz')
        runner = GroupedPassengerSimulation(late_data,'line1-baseline',groups[scenario],late_profiles)
        observed = runner.run(end_second=13500)['checkpoint']
        assert snapshot == observed
    late_motion = {}
    for row in late_rows:
        state = read(late_folder/f"{row['seed']}_{row['scenario']}.json.gz")
        assert state['sha256'] == PassengerSimulation._seal(state)
        assert state['entered'] == state['approaching']+state['queued']+state['onboard_count']+state['completed']
        assert all(e['load'] <= e['max_capacity'] for e in state['load_events'])
        assert row['selected_candidate_id'] == row['baseline_candidate_id'] == 'KEEP:plan'
        assert row['paired_actions_identical'] and row['cash_delta_RUB'] is None
        assert row['minimum_headway_seconds'] >= 113 and not row['release_authorized']
        selected_groups = groups[row['scenario']]
        for key, start in [('main_cohort',7200),('late_cohort',13500)]:
            ids = {g['group_id'] for g in selected_groups if start <= g['turnstile_second'] < 14400}
            arrived = sum(g['count'] for g in selected_groups if g['group_id'] in ids)
            for name, value in independent_cohort(state,ids,arrived).items(): assert row[key][name] == value
        for key, end in [('horizon_overload',row['forecast_horizon_end_second']),('tail_overload',21600)]:
            for name, value in independent_overload(state,13500,end).items(): assert row[key][name] == value
        if row['seed'] not in late_motion:
            changed,_ = compile_timing(late_data,'line1-baseline','mild',row['seed'],
                committed=late_data['leg_timing'],as_of=13500)
            late_motion[row['seed']] = changed,simulate(changed,'line1-baseline')
        changed,motion = late_motion[row['seed']]
        assert row['minimum_headway_seconds'] == verify_motion(changed,motion)
        verify_work(changed,motion,row['resources'],13500,21600)
        result['verified_runs'] += 1
    for kind in ['KEEP','ADD']:
        folder = OUTPUT/'counterfactual'/kind
        data,motion = read(folder/'plan.json'),read(folder/'motion.json')
        verify_motion(data,motion)
        for scenario in groups:
            verify_saved_run(folder/scenario, groups[scenario])
            verify_work(data,motion,read(folder/scenario/'summary.json')['resources'])
            result['verified_runs'] += 1
    static = read(OUTPUT/'static_diagnostic.json')
    assert len(static) == 30
    for case in {r['case'] for r in static}:
        folder = OUTPUT/'diagnostic'/case
        data,motion = read(folder/'plan.json'),read(folder/'motion.json')
        verify_motion(data,motion)
        for scenario in groups:
            verify_saved_run(folder/scenario,groups[scenario])
            verify_work(data,motion,read(folder/scenario/'summary.json')['resources'])
            result['verified_runs'] += 1
    meta = read(OUTPUT/'diagnostic/N53_phase0/parameters.json')
    assert meta['cycle_seconds'] == 53*113 and meta['extra_terminal_hold_seconds'] == [24,25]
    # Continuous execution must reproduce the saved sequence of decisions and resumes.
    folder = OUTPUT/'episodes/A_mild'
    data = read(folder/'plan.json')
    profiles = car_profiles(config,[t['train_id'] for t in data['trains']])
    runner = GroupedPassengerSimulation(data,'line1-baseline',groups['S3'],profiles)
    continuous = runner.run()['checkpoint']
    assert continuous == read(folder/'S3/final_checkpoint.json.gz')
    result.update(status='passed', cohort_size=116588, preliminary_seeds=sorted(preliminary),
        validation_seeds=sorted(validation), selected_validation_runs=len(late_rows),
        continuous_resume_exact=True, release_authorized=False)
    (PROOF/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()
