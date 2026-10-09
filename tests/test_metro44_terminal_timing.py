import copy

import pytest

from goal43_policy import load_policy
from metro44_planner_noidle import terminal_total_audit
from metro44_terminal_timing import compile_terminal_timing, rename_tracks
from simulation import simulate


def fixture():
    policy = load_policy()
    policy['demo']['platform_capacity_by_station']['A'] = 3
    data = dict(schema_version='1.0', source='terminal grouping fixture',
        period_start='2026-09-30T08:00:00+03:00', period_end='2026-09-30T09:00:00+03:00',
        stations=[dict(station_id=s, kind='depot' if s == 'D' else 'line', stop_seconds=0,
                       turnaround_seconds=180 if s == 'L1_01' else 0) for s in ('A', 'L1_01', 'D')],
        tracks=[dict(track_id=t, from_station=a, to_station=b, travel_seconds=seconds, length_km='1')
                for t,a,b,seconds in [('P01_down','A','L1_01',100),
                    ('P01_up','L1_01','A',100), ('ENTRY_AVTOVO','D','L1_01',20),
                    ('EXIT_NORTHERN','A','D',20)]],
        trains=[dict(train_id=str(i), train_type='fixture', cars=8, state='standby',
                     location='D' if i == 2 else 'A', ready_after_seconds=0) for i in range(3)],
        previous_departures=[], max_trains_on_line=53, headway_policy=policy)
    trips = [dict(trip_id=f'{i}:service', train_id=str(i), departure_second=i*113,
                  route=['P01_down','P01_up'], passenger=True) for i in range(2)]
    trips += [dict(trip_id='2:entry', train_id='2', departure_second=200,
                   route=['ENTRY_AVTOVO'], passenger=False),
              dict(trip_id='2:service', train_id='2', departure_second=400,
                   route=['P01_up'], passenger=True)]
    trips += [dict(trip_id=f'{i}:exit', train_id=str(i), departure_second=700+i*113,
                   route=['EXIT_NORTHERN'], passenger=False) for i in range(3)]
    data['plans'] = [dict(plan_id='P', trips=trips)]
    return data, policy


def test_shared_terminal_slots_keep_original_tracks_and_input():
    data, policy = fixture()
    before = copy.deepcopy(data)
    compiled, audit = compile_terminal_timing(data, 'P', policy=policy)
    assert compiled['tracks'] == before['tracks'] and data == before
    assert terminal_total_audit(compiled, simulate(compiled, 'P'))['passed']
    entry = next(row for row in audit['platform_occupancy'] if row['train_id'] == '2' and row['platform'][0] == 'L1_01')
    assert entry['platform'] == ['L1_01', 'down']
    assert entry['arrival_second'] >= 281
    assert 'ENTRY_AVTOVO_down' not in repr(compiled)


def test_track_bijection_restores_keys_values_and_exact_strings_only():
    data = {'ENTRY_AVTOVO': ['ENTRY_AVTOVO', 'trip:ENTRY_AVTOVO'],
            'train': {'location': 'ENTRY_NORTHERN'}}
    mapping = {'ENTRY_AVTOVO': 'ENTRY_AVTOVO_down', 'ENTRY_NORTHERN': 'ENTRY_NORTHERN_up'}
    renamed = rename_tracks(data, mapping)
    assert renamed['ENTRY_AVTOVO_down'][1] == 'trip:ENTRY_AVTOVO'
    assert rename_tracks(renamed, {v:k for k,v in mapping.items()}) == data


def test_alias_collision_is_rejected_before_compile():
    data, policy = fixture()
    data['tracks'].append(dict(data['tracks'][0], track_id='ENTRY_AVTOVO_down'))
    with pytest.raises(ValueError, match='alias collision'):
        compile_terminal_timing(data, 'P', policy=policy)


def test_started_arrival_and_past_events_are_preserved():
    data, policy = fixture()
    baseline, _ = compile_terminal_timing(data, 'P', policy=policy)
    compiled, _ = compile_terminal_timing(baseline, 'P', policy=policy, as_of=150)
    before, after = simulate(baseline, 'P'), simulate(compiled, 'P')
    assert [e for e in before['events'] if e['second'] < 150] == [e for e in after['events'] if e['second'] < 150]
    assert next(e for e in after['events'] if e['train_id'] == '1' and e['event_type'] == 'arrival')['second'] == 213


def test_compiler_may_not_delay_already_started_arrival(monkeypatch):
    import metro44_terminal_timing as adapter
    data, policy = fixture()
    data['leg_timing'] = [dict(train_id='0', trip_id='0:service', leg_index=0,
                              departure_second=0, service_seconds=180, approach_holding_seconds=0)]
    def altered(data, *args, **kwargs):
        changed = copy.deepcopy(data)
        changed['leg_timing'][0]['approach_holding_seconds'] = 1
        return changed, {}
    monkeypatch.setattr(adapter, 'compile_operational_timing', altered)
    with pytest.raises(ValueError, match='already started'):
        compile_terminal_timing(data, 'P', policy=policy, as_of=50)
