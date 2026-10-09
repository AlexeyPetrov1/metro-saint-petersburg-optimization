import copy
import gzip
import json
from pathlib import Path
from time import monotonic

import pytest

from dwell import timing_key
from metro46_partial_trip import ready_bounds, retime_partial_trips


def geometry_case():
    return dict(
        stations=[dict(station_id=x, stop_seconds=20 if x != 'D' else 0,
                       turnaround_seconds=180 if x == 'D' else 0) for x in 'ABCD'],
        tracks=[dict(track_id=f'P{i}', from_station=a, to_station=b, travel_seconds=100)
                for i, (a, b) in enumerate(zip('ABC', 'BCD'))],
        trains=[dict(train_id='T')],
        plans=[dict(plan_id='P', trips=[dict(trip_id='J', train_id='T',
            departure_second=0, route=['P0', 'P1', 'P2'], passenger=True)])],
        leg_timing=[
            dict(train_id='T', trip_id='J', leg_index=0, departure_second=0,
                 approach_holding_seconds=0, service_seconds=20),
            dict(train_id='T', trip_id='J', leg_index=1, departure_second=120,
                 approach_holding_seconds=30, service_seconds=20),
            dict(train_id='T', trip_id='J', leg_index=2, departure_second=270,
                 approach_holding_seconds=40, service_seconds=180)])


def test_current_approach_hold_is_an_immutable_arrival_not_removable_slack():
    data = geometry_case()
    before = copy.deepcopy(data)
    row = ready_bounds(data, 200, train_ids=['T'])[0]
    assert data == before
    assert row['immutable_last_arrival_second'] == 250
    assert row['source_ready_second'] == 590
    assert row['earliest_ready_second'] == 550
    assert row['maximum_geometry_advance_seconds'] == 40
    assert row['future_leg_indices'] == [2]


def test_completed_route_cannot_reduce_required_terminal_turn():
    row = ready_bounds(geometry_case(), 450, train_ids=['T'])[0]
    assert row['future_leg_indices'] == []
    assert row['earliest_ready_second'] == row['source_ready_second'] == 590
    assert row['maximum_geometry_advance_seconds'] == 0


@pytest.mark.parametrize('as_of', [True, -1, 1.5])
def test_invalid_clock_is_rejected(as_of):
    with pytest.raises(ValueError):
        ready_bounds(geometry_case(), as_of)


def test_unknown_train_is_rejected():
    with pytest.raises(ValueError, match='unknown train'):
        ready_bounds(geometry_case(), 200, train_ids=['missing'])


@pytest.fixture(scope='module')
def morning_case():
    path = Path('data/processed/metro44_wait_calendar_physical_20261009/'
                'wait_calendar_reference_plan.json.gz')
    return json.loads(gzip.decompress(path.read_bytes()))


def test_real_morning_ready_times_are_already_at_the_geometry_lower_bound(morning_case):
    rows = {r['train_id']: r for r in ready_bounds(morning_case, 7200,
                                                 train_ids=['B011', 'Y005', 'B006'])}
    assert [(rows[t]['source_ready_second'], rows[t]['earliest_ready_second'])
            for t in ('B011', 'Y005', 'B006')] == [(7285, 7285), (8125, 8125), (8210, 8210)]
    assert rows['B011']['future_leg_indices'] == []
    assert rows['Y005']['immutable_last_arrival_second'] == 7305
    assert rows['B006']['immutable_last_arrival_second'] == 7230


def test_real_no_slack_plan_retains_every_leg_and_runs_the_exact_physical_gate(morning_case):
    before = copy.deepcopy(morning_case)
    changed, audit = retime_partial_trips(morning_case, 7200, 64800,
        train_ids=['B011', 'Y005', 'B006'], deadline=monotonic() + 60)
    assert morning_case == before and changed == morning_case
    assert audit['movement_passed'] and audit['executed_prefix_preserved']
    assert audit['current_movements_preserved'] and audit['identity_preserved']
    assert audit['changed_future_legs'] == []
    assert audit['passengers_checked'] is False and audit['cash_saving_rub'] is None


def test_expired_budget_stops_before_geometry_or_compile(morning_case):
    with pytest.raises(TimeoutError):
        retime_partial_trips(morning_case, 7200, 64800, deadline=monotonic() - 1)


def test_real_compiler_does_not_turn_optimistic_geometry_slack_into_false_gain():
    path = Path('data/processed/metro44_bridge_off160_actual_6h_20261009_v2/'
                'runs/OFF160_BRIDGE_S1_7/movement_input.json')
    data = json.loads(path.read_bytes())
    before = copy.deepcopy(data)
    changed, audit = retime_partial_trips(data, 6300, 22200,
        train_ids=['Y020'], deadline=monotonic() + 60)
    assert data == before and changed == data
    assert audit['ready_bounds'][0]['maximum_geometry_advance_seconds'] == 35
    assert audit['compiler'] is not None and audit['movement_passed']
    assert audit['changed_future_legs'] == []


def test_future_hold_removal_restores_started_arrival_and_limits_changes_to_future_legs(monkeypatch):
    import metro46_partial_trip as module

    data = geometry_case()
    data['headway_policy'] = {}
    calls = []

    def compile_proposal(source, *args, **kwargs):
        out = copy.deepcopy(source)
        out['leg_timing'][1]['approach_holding_seconds'] = 0
        out['leg_timing'][2]['approach_holding_seconds'] = 0
        calls.append(kwargs)
        return out, dict(test_compiler=True)

    def full_gate(candidate, *args, **kwargs):
        assert candidate['leg_timing'][1] == data['leg_timing'][1]
        assert candidate['leg_timing'][0] == data['leg_timing'][0]
        assert candidate['leg_timing'][2]['approach_holding_seconds'] == 0
        return dict(movement_passed=True, executed_prefix_preserved=True,
                    current_movements_preserved=True)

    monkeypatch.setattr(module, 'compile_operational_timing', compile_proposal)
    monkeypatch.setattr(module, 'physical_gate', full_gate)
    changed, audit = retime_partial_trips(data, 200, 900, train_ids=['T'])
    assert len(calls) == 1
    assert timing_key('T', 'J', 1) in calls[0]['departure_not_before']
    assert calls[0]['departure_not_before'][timing_key('T', 'J', 2)] == 270
    assert audit['changed_future_legs'][0]['arrival_advance_seconds'] == 40
    assert audit['ready_bounds'][0]['maximum_geometry_advance_seconds'] == 40
    assert data['leg_timing'][2]['approach_holding_seconds'] == 40
