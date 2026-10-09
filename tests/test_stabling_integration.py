"""Small movement fixtures for withdrawal parking and return mileage."""

from copy import deepcopy
from decimal import Decimal

import pytest

from goal43_actions import expand_operational_action
from goal43_policy import load_policy, parameter
from metro47_audit_run import preparation_gate
from metro47_withdrawal import apply_withdrawals, withdrawal_options
from metro48_audit import transition_gate
from metro48_temporary import apply_temporary_withdrawals, temporary_options
from simulation import simulate
from withdrawal_stabling import attach_stabling, routing


def movement(origin='L1_01'):
    policy = load_policy()
    policy['demo']['terminal_parking_capacity_by_station'] = {'L1_01': 1, 'L1_19': 1}
    policy['demo']['platform_capacity_by_station'] = {'L1_01': 2, 'L1_19': 2}
    stations = [dict(station_id=s, kind='line', stop_seconds=20 if s == 'L1_17' else 0,
                     turnaround_seconds=180 if s in ('L1_01', 'L1_19') else 0)
                for s in ('L1_01', 'L1_12', 'L1_17', 'L1_19')]
    tracks = []
    for i, (a, b, km, seconds) in enumerate([
            ('L1_01', 'L1_12', '1', 100), ('L1_12', 'L1_17', '1', 100),
            ('L1_17', 'L1_19', '2', 200)]):
        for direction, start, end in [('up', a, b), ('down', b, a)]:
            tracks.append(dict(track_id=f'T{i}_{direction}', from_station=start, to_station=end,
                               length_km=km, travel_seconds=seconds))
    for depot, terminal in [('AVTOVO', 'L1_01'), ('NORTHERN', 'L1_19')]:
        name = f'DEPOT_{depot}'
        stations.append(dict(station_id=name, kind='depot', stop_seconds=0, turnaround_seconds=0))
        for prefix, start, end in [('EXIT', terminal, name), ('ENTRY', name, terminal)]:
            tracks.append(dict(track_id=f'{prefix}_{depot}', from_station=start, to_station=end,
                               length_km='1', travel_seconds=120))
    return dict(schema_version='1.0', source='focused stabling integration fixture',
                period_start='2026-09-30T06:00:00+03:00', period_end='2026-09-30T07:40:00+03:00',
                stations=stations, tracks=tracks, trains=[dict(train_id='A', train_type='Yubileyny',
                cars=8, state='standby', location=origin, ready_after_seconds=0)],
                plans=[dict(plan_id='P', trips=[])], previous_departures=[],
                max_trains_on_line=53, headway_policy=policy, leg_timing=[])


def append_trip(data, identity, second, route, *, train='A', passenger=False):
    tracks = {r['track_id']: r for r in data['tracks']}
    stations = {r['station_id']: r for r in data['stations']}
    data['plans'][0]['trips'].append(dict(trip_id=identity, train_id=train,
        departure_second=second, route=route, passenger=passenger))
    for index, name in enumerate(route):
        track = tracks[name]
        destination = stations[track['to_station']]
        service = destination['stop_seconds'] + destination['turnaround_seconds']
        data['leg_timing'].append(dict(train_id=train, trip_id=identity, leg_index=index,
            departure_second=second, service_seconds=service))
        second += track['travel_seconds'] + service


def two_train_duties():
    data = attach_stabling(movement())
    data['trains'].append(dict(train_id='B', train_type='Yubileyny', cars=8,
        state='standby', location='L1_19', ready_after_seconds=0))
    up, down = ['T0_up', 'T1_up', 'T2_up'], ['T2_down', 'T1_down', 'T0_down']
    for train, trips in [('A', [('U', 1000, up), ('D', 2000, down)]),
                         ('B', [('D', 1200, down), ('U', 2200, up)])]:
        for identity, second, route in trips:
            append_trip(data, f'{train}:{identity}', second, route, train=train, passenger=True)
        exit_track = 'EXIT_AVTOVO' if train == 'A' else 'EXIT_NORTHERN'
        append_trip(data, f'{train}:DAYEXIT', 2700 if train == 'A' else 2900,
                    [exit_track], train=train)
    return data


def ledger_to1_cost(data):
    prices = parameter(data['headway_policy'], 'maintenance_prices_rub_by_type')
    interval = Decimal(str(parameter(data['headway_policy'], 'to1_interval_km')))
    result = simulate(data, data['plans'][0]['plan_id'])
    return sum((Decimal(row['train_km']) * Decimal(str(prices[row['train_type']]['TO1'])) / interval
                for row in result['transport_work']), Decimal(0))


@pytest.mark.parametrize('origin,sequence,destination,km', [
    ('L1_01', 1, 'SIDING_VETERANS', '0.35'),
    ('L1_01', 0, 'DEPOT_AVTOVO', '1.35'),
    ('L1_19', 0, 'DEPOT_NORTHERN', '1'),
])
def test_withdrawal_distance_is_recorded_at_actual_destination(origin, sequence, destination, km):
    source = movement(origin)
    before = deepcopy(source)
    data = attach_stabling(source)
    row = routing(data, 'A', origin, sequence)
    append_trip(data, 'EXIT', 100, row['exit_route'])
    result = simulate(data, 'P')
    final = result['final_state'][0]
    assert final['location'] == destination and final['state'] == 'standby'
    assert Decimal(result['transport_work'][0]['nonpassenger_train_km']) == Decimal(km)
    assert result['obligations_closed'] and source == before


def test_siding_standby_frees_line_capacity_before_another_train_enters():
    data = attach_stabling(movement())
    data['max_trains_on_line'] = 1
    data['trains'].append(dict(train_id='C', train_type='Yubileyny', cars=8,
        state='standby', location='DEPOT_AVTOVO', ready_after_seconds=0))
    exit_row = routing(data, 'A', 'L1_01', 1)
    entry_row = routing(data, 'C', 'L1_01', 0)
    append_trip(data, 'EXIT', 0, exit_row['exit_route'])
    append_trip(data, 'ENTRY', 500, entry_row['entry_route'], train='C')
    result = simulate(data, 'P')
    final = {r['train_id']: r for r in result['final_state']}
    assert final['A']['location'] == exit_row['destination']
    assert final['C']['location'] == 'L1_01'
    assert result['checks']['peak_trains_on_line'] == 1


def test_northern_siding_transfer_is_simulated_before_350_metre_access():
    data = attach_stabling(movement('L1_19'))
    row = routing(data, 'A', 'L1_19', 1)
    assert row['destination'] == 'SIDING_AKADEMICHESKAYA'
    append_trip(data, 'EXIT', 100, row['exit_route'])
    result = simulate(data, 'P')
    arrivals = [r for r in result['events'] if r['event_type'] == 'arrival']
    assert [(r['location'], r['second']) for r in arrivals] == [
        ('L1_17', 300), ('SIDING_AKADEMICHESKAYA', 362)]
    assert Decimal(result['transport_work'][0]['nonpassenger_train_km']) == Decimal('2.35')


def test_temporary_withdrawal_returns_from_siding_and_charges_both_access_legs():
    data = attach_stabling(movement())
    data['trains'].append(dict(train_id='B', train_type='Yubileyny', cars=8,
        state='standby', location='L1_01', ready_after_seconds=0))
    # One completed removal makes the next actual selection use siding slot 1.
    append_trip(data, 'G47WITHDRAW:0:B:EXIT', 0, ['EXIT_AVTOVO'], train='B')
    up = ['T0_up', 'T1_up', 'T2_up']
    down = ['T2_down', 'T1_down', 'T0_down']
    for identity, second, route in [('U1', 1000, up), ('D1', 2000, down), ('U2', 3000, up)]:
        append_trip(data, identity, second, route, passenger=True)
    append_trip(data, 'A:DAYEXIT', 3600, ['EXIT_NORTHERN'])
    before = deepcopy(data)
    option = next(r for r in temporary_options(data, 500, 5000, return_by_second=3500)
                  if r['train_id'] == 'A')
    changed, audit = apply_temporary_withdrawals(data, [option], 500, 5000,
                                               return_by_second=3500)
    added = set(audit['appended_exit_ids'] + audit['appended_entry_ids'])
    trips = [t for t in changed['plans'][0]['trips'] if t['trip_id'] in added]
    tracks = {r['track_id']: r for r in changed['tracks']}
    access_km = sum((Decimal(tracks[name]['length_km']) for trip in trips for name in trip['route']), Decimal(0))
    assert access_km == Decimal('0.70')
    assert tracks[trips[0]['route'][-1]]['to_station'] == 'SIDING_VETERANS'
    assert tracks[trips[1]['route'][0]]['from_station'] == 'SIDING_VETERANS'
    assert audit['obligations_closed'] and audit['executed_prefix_preserved']
    assert audit['depot_preparation']['cycles'][0]['release_seconds'] == 0
    assert preparation_gate(data, changed, 500, parameter)['passed']
    assert transition_gate(data, changed, 500)['passed']
    assert data == before


def test_applied_northern_withdrawal_keeps_every_transfer_leg_and_source_clock():
    data = attach_stabling(movement('L1_19'))
    data['trains'].append(dict(train_id='B', train_type='Yubileyny', cars=8,
        state='standby', location='L1_01', ready_after_seconds=0))
    append_trip(data, 'G47WITHDRAW:0:B:EXIT', 0, ['EXIT_AVTOVO'], train='B')
    append_trip(data, 'D1', 1000, ['T2_down', 'T1_down', 'T0_down'], passenger=True)
    append_trip(data, 'U1', 2000, ['T0_up', 'T1_up', 'T2_up'], passenger=True)
    append_trip(data, 'A:DAYEXIT', 2700, ['EXIT_NORTHERN'])
    before = deepcopy(data)
    option = next(r for r in withdrawal_options(data, 500, 5000) if r['train_id'] == 'A')
    changed, audit = apply_withdrawals(data, [option], 500, 5000)
    result = simulate(changed, 'P')
    exit_trip = next(t for t in changed['plans'][0]['trips'] if t['trip_id'] in audit['appended_exit_ids'])
    assert exit_trip['route'] == ['T2_down', 'EXIT_SIDING_AKADEMICHESKAYA']
    work = next(r for r in result['transport_work'] if r['train_id'] == 'A')
    assert Decimal(work['nonpassenger_train_km']) == Decimal('2.35')
    assert next(r for r in result['final_state'] if r['train_id'] == 'A')['location'] == 'SIDING_AKADEMICHESKAYA'
    assert audit['obligations_closed'] and audit['executed_prefix_preserved']
    assert data == before


def test_simulator_exposes_actual_parking_and_destination_cost_distance():
    data = attach_stabling(movement())
    row = routing(data, 'A', 'L1_01', 1)
    append_trip(data, 'G47WITHDRAW:100:A:EXIT', 100, row['exit_route'])
    data['plans'][0]['trips'][0]['departure_second'] = 50
    data['plans'].append(dict(plan_id='OTHER', trips=[dict(trip_id='G47WITHDRAW:200:A:EXIT',
        train_id='A', departure_second=200, route=['EXIT_AVTOVO'], passenger=False)]))
    result = simulate(data, 'P')
    parking = result['withdrawal_stabling']
    assert parking['policy_targets'] == {'siding': 0.35, 'depot': 0.65}
    assert parking['completed_counts'] == {'siding': 1, 'depot': 0}
    assert parking['parked_trains'][0]['destination'] == 'SIDING_VETERANS'
    assert 'Проспект Ветеранов' in parking['parked_trains'][0]['label']
    assert parking['planned_withdrawals'][0]['arrival_second'] == 142
    assert parking['planned_withdrawals'][0]['departure_second'] == 100
    assert parking['planned_counts'] == {'siding': 1, 'depot': 0}
    assert Decimal(parking['planned_withdrawals'][0]['exit_km']) == Decimal('0.35')


def test_selected_batch_uses_one_depot_one_siding_and_actual_mileage_saving():
    data = two_train_duties()
    before = deepcopy(data)
    options = withdrawal_options(data, 500, 5000)
    assert {r['train_id'] for r in options} == {'A', 'B'}
    changed, audit = apply_withdrawals(data, options, 500, 5000)
    destinations = {r['train_id']: r for r in audit['withdrawal_destinations']}
    assert destinations['A']['kind'] == 'depot'
    assert destinations['B']['kind'] == 'siding'
    assert Decimal(audit['estimated_to1_resource_saving_rub']) == ledger_to1_cost(data) - ledger_to1_cost(changed)
    assert transition_gate(data, changed, 500)['passed'] and data == before


def test_generic_actions_count_the_first_withdrawal_for_the_second_selection():
    data = two_train_duties()
    before = deepcopy(data)
    policy = data['headway_policy']
    first, first_action = expand_operational_action(data, 'WITHDRAW', 'A', 500, {}, policy=policy)
    second, second_action = expand_operational_action(first, 'WITHDRAW', 'B', 500, {}, policy=policy)
    assert first_action['withdrawal_destination']['kind'] == 'depot'
    assert second_action['withdrawal_destination']['kind'] == 'siding'
    assert second_action['withdrawal_destination']['sequence'] == 1
    assert first_action['withdrawal_return_closed'] and second_action['withdrawal_return_closed']
    assert simulate(second, 'P')['obligations_closed'] and data == before
