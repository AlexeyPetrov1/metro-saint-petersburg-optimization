from types import SimpleNamespace

import pytest

from metro44_journey import cohort_journey


def runner():
    group = dict(group_id='g', turnstile_second=0, platform_second=2, count=3,
                 destinations=[dict(station_id='B', count=1), dict(station_id='C', count=2)])
    events = [dict(event_id='d', event_type='departure', train_id='T', trip_id='trip', second=5,
                   location='AB', remaining_route=['AB', 'BC']),
              dict(event_id='a', event_type='arrival', train_id='T', trip_id='trip', second=15, location='B'),
              dict(event_id='b', event_type='arrival', train_id='T', trip_id='trip', second=30, location='C')]
    return SimpleNamespace(by_id={'g': group}, tracks={'AB': dict(from_station='A', to_station='B'),
        'BC': dict(from_station='B', to_station='C')}, motion_by_id={e['event_id']: e for e in events})


def state(second=40):
    return dict(second=second, board_events=[dict(group_id='g', motion_event_id='d', second=5, count=1),
                                            dict(group_id='g', motion_event_id='d', second=5, count=2)])


def test_destination_fifo_recovers_wait_ride_and_gate_to_exit():
    result = cohort_journey(runner(), state(), 0, 1)
    assert result['complete']
    assert result['mean_ride_seconds'] == 20
    assert result['mean_platform_to_exit_seconds'] == 23
    assert result['mean_gate_to_exit_seconds'] == 25
    assert result['p95_gate_to_exit_seconds'] == 30


def test_unfinished_alighting_keeps_means_unknown():
    result = cohort_journey(runner(), state(20), 0, 1)
    assert not result['complete']
    assert result['completed'] == 1
    assert result['mean_gate_to_exit_seconds'] is None


def test_missing_boarding_keeps_cohort_incomplete():
    value = state(); value['board_events'].pop()
    result = cohort_journey(runner(), value, 0, 1)
    assert not result['complete'] and result['uncompleted'] == 2


def test_duplicate_boarding_cannot_invent_a_destination():
    value = state(); value['board_events'].append(value['board_events'][0])
    with pytest.raises(ValueError, match='destination'):
        cohort_journey(runner(), value, 0, 1)


def test_empty_cohort_is_reported_without_numeric_means():
    result = cohort_journey(runner(), state(), 1, 2)
    assert result['arrived'] == 0
    assert result['mean_ride_seconds'] is None
