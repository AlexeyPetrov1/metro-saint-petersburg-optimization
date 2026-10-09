from copy import deepcopy
from decimal import Decimal

import pytest

from withdrawal_stabling import (attach_stabling, assign_routes, is_withdrawal_trip,
                                report, routing, withdrawal_count)


def line():
    stations = [dict(station_id=f'L1_{i:02}', kind='line', stop_seconds=0,
                     turnaround_seconds=180 if i in (1, 19) else 0) for i in range(1, 20)]
    tracks = []
    for i in range(1, 19):
        for direction, a, b in [('up', i, i + 1), ('down', i + 1, i)]:
            tracks.append(dict(track_id=f'T{i:02}_{direction}', from_station=f'L1_{a:02}',
                               to_station=f'L1_{b:02}', length_km='0.5', travel_seconds=60))
    for depot, terminal in [('AVTOVO', 'L1_01'), ('NORTHERN', 'L1_19')]:
        stations.append(dict(station_id=f'DEPOT_{depot}', kind='depot', stop_seconds=0, turnaround_seconds=0))
        for prefix, a, b in [('EXIT', terminal, f'DEPOT_{depot}'), ('ENTRY', f'DEPOT_{depot}', terminal)]:
            tracks.append(dict(track_id=f'{prefix}_{depot}', from_station=a, to_station=b,
                               length_km='1', travel_seconds=120))
    return dict(stations=stations, tracks=tracks, plans=[dict(trips=[])])


def test_attach_is_idempotent_and_does_not_mutate_source():
    data = line(); before = deepcopy(data)
    attached = attach_stabling(data)
    assert data == before and attached is not data
    assert attach_stabling(attached) == attached
    sidings = [s for s in attached['stations'] if s['kind'] == 'siding']
    assert len(sidings) == 3
    assert {s['station_id'] for s in attached['withdrawal_stabling']['sites']} == {'L1_01', 'L1_12', 'L1_17'}
    assert all(s['turnaround_seconds'] == 180 for s in sidings)
    for track in attached['tracks']:
        if 'AVTOVO' in track['track_id']:
            assert track['length_km'] == '1.35' and track['travel_seconds'] == 120
        if 'SIDING_' in track['track_id']:
            assert track['length_km'] == '0.35' and track['travel_seconds'] == 42


def test_twenty_committed_slots_have_seven_sidings_and_thirteen_depots():
    data = attach_stabling(line())
    rows = [routing(data, f'B{i:03}', 'L1_01', i) for i in range(20)]
    assert sum(r['kind'] == 'siding' for r in rows) == 7
    assert sum(r['kind'] == 'depot' for r in rows) == 13
    assert rows == [routing(data, f'B{i:03}', 'L1_01', i) for i in range(20)]


def test_siding_access_at_named_station_is_only_350_metres_each_way():
    data = attach_stabling(line())
    for origin in ['L1_01', 'L1_12', 'L1_17']:
        row = routing(data, 'B001', origin, 1)
        assert row['kind'] == 'siding'
        assert row['exit_km'] == row['entry_km'] == '0.35'
        assert len(row['exit_route']) == len(row['entry_route']) == 1
        assert row['exit_seconds'] == row['entry_seconds'] == 42


def test_northern_terminal_siding_route_contains_physical_line_transfer():
    data = attach_stabling(line()); row = routing(data, 'B001', 'L1_19', 1)
    tracks = {t['track_id']: t for t in data['tracks']}
    assert row['destination'] == 'SIDING_AKADEMICHESKAYA'
    assert row['exit_route'] == ['T18_down', 'T17_down', 'EXIT_SIDING_AKADEMICHESKAYA']
    assert row['entry_route'] == ['ENTRY_SIDING_AKADEMICHESKAYA', 'T17_up', 'T18_up']
    assert Decimal(row['exit_km']) == Decimal('1.35')
    for route, start, end in [(row['exit_route'], 'L1_19', row['destination']),
                              (row['entry_route'], row['destination'], 'L1_19')]:
        assert tracks[route[0]]['from_station'] == start
        assert tracks[route[-1]]['to_station'] == end
        assert all(tracks[a]['to_station'] == tracks[b]['from_station'] for a, b in zip(route, route[1:]))


def test_depot_distances_and_existing_release_minutes_are_separate():
    data = attach_stabling(line())
    for origin, destination, km, release in [('L1_01', 'DEPOT_AVTOVO', '1.35', 720),
                                            ('L1_19', 'DEPOT_NORTHERN', '1', 780)]:
        row = routing(data, 'B001', origin, 0)
        assert row['kind'] == 'depot' and row['destination'] == destination
        assert row['exit_km'] == row['entry_km'] == km
        assert row['release_seconds'] == release


def test_assignment_uses_selected_sorted_ids_and_only_prior_actual_withdrawals():
    data = attach_stabling(line())
    data['plans'][0]['trips'] = [dict(trip_id=identity) for identity in
        ['G48TEMP:0:B001:EXIT', 'G48TEMP:0:B001:ENTRY', 'G47CLOSE:999:Y001:EXIT', 'B001:DAYEXIT']]
    rows = assign_routes(data, ['Y002', 'B002'], {'Y002': 'L1_19', 'B002': 'L1_01'})
    assert list(rows) == ['B002', 'Y002']
    assert rows['B002']['sequence'] == 1 and rows['B002']['kind'] == 'siding'
    assert rows['Y002']['sequence'] == 2 and rows['Y002']['kind'] == 'depot'
    assert assign_routes(data, ['B002', 'Y002'], {'Y002': 'L1_19', 'B002': 'L1_01'}) == rows


def test_legacy_snapshot_keeps_original_depot_route_and_distance():
    data = line(); before = deepcopy(data)
    row = routing(data, 'B001', 'L1_01', 1)
    assert row['kind'] == 'depot' and row['exit_route'] == ['EXIT_AVTOVO']
    assert row['exit_km'] == '1' and data == before


@pytest.mark.parametrize('sequence', [-1, True, 0.5])
def test_invalid_sequence_rejected(sequence):
    with pytest.raises(ValueError, match='sequence'):
        routing(attach_stabling(line()), 'B001', 'L1_01', sequence)


def test_generic_withdrawals_count_but_entries_and_closure_do_not():
    included = ['B001:EXIT123', 'G47WITHDRAW:0:B002:EXIT', 'G48TEMP:0:Y001:EXIT']
    excluded = ['B001:DAYEXIT', 'G47CLOSE:99:B001:EXIT', 'G48TEMP:0:Y001:ENTRY',
                'B001:EXIT', 'B001:EXIT-1', 'B001:ACTION_FINAL_EXIT']
    assert all(is_withdrawal_trip(identity) for identity in included)
    assert not any(is_withdrawal_trip(identity) for identity in excluded)
    data = line(); data['plans'][0]['trips'] = [dict(trip_id=s) for s in included + excluded]
    assert withdrawal_count(data) == 3


def test_report_separates_pending_arrivals_completed_shares_and_current_parking():
    data = attach_stabling(line())
    data['plans'][0]['trips'] = [
        dict(trip_id='B001:EXIT0', train_id='B001', departure_second=0, passenger=False, route=['EXIT_AVTOVO']),
        dict(trip_id='G48TEMP:0:B002:EXIT', train_id='B002', departure_second=0, passenger=False,
             route=['EXIT_SIDING_VETERANS']),
        dict(trip_id='B003:EXIT200', train_id='B003', departure_second=200, passenger=False, route=['EXIT_NORTHERN']),
        dict(trip_id='B004:DAYEXIT', train_id='B004', departure_second=0, passenger=False, route=['EXIT_AVTOVO'])]
    motion = dict(events=[
        dict(train_id='B001', trip_id='B001:EXIT0', event_type='arrival', second=120, location='DEPOT_AVTOVO'),
        dict(train_id='B002', trip_id='G48TEMP:0:B002:EXIT', event_type='arrival', second=42, location='SIDING_VETERANS')],
        final_state=[dict(train_id='B001', state='standby', location='DEPOT_AVTOVO'),
                     dict(train_id='B002', state='moving', location='T01_up'),
                     dict(train_id='B003', state='moving', location='EXIT_NORTHERN'),
                     dict(train_id='B004', state='standby', location='DEPOT_AVTOVO')])
    before = deepcopy(data); result = report(data, motion)
    assert data == before
    assert result['completed_counts'] == dict(siding=1, depot=1)
    assert result['completed_shares'] == dict(siding=0.5, depot=0.5)
    assert result['policy_targets'] == dict(siding=0.35, depot=0.65)
    assert result['pending_count'] == 1
    assert [r['train_id'] for r in result['parked_trains']] == ['B001']
    rows = {r['train_id']: r for r in result['planned_withdrawals']}
    assert rows['B001']['arrival_second'] == 120 and rows['B001']['exit_km'] == '1.35'
    assert rows['B002']['arrival_second'] == 42 and rows['B002']['label'] == SITES_LABEL
    assert rows['B003']['arrival_second'] is None and rows['B003']['pending'] is True
    assert rows['B003']['exit_km'] == '1'


SITES_LABEL = 'Тупики за станцией «Проспект Ветеранов»'


def test_report_no_completed_withdrawal_does_not_invent_measured_share():
    result = report(attach_stabling(line()), dict(events=[], final_state=[]))
    assert result['completed_counts'] == dict(siding=0, depot=0)
    assert result['completed_shares'] == dict(siding=None, depot=None)
    assert result['planned_withdrawals'] == result['parked_trains'] == []
