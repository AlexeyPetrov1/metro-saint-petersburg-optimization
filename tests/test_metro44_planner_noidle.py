import math

import pytest

from metro44_planner_noidle import remove_idle_cutoff, terminal_total_audit


SOURCE = "def cap(h, repair_future_platforms):\n    max_idle=max(0,2*h-parameter(policy,'turnaround_s')-2)\n    return max_idle\n"


def test_noidle_preserves_original_mode():
    scope = {'parameter': lambda policy, key: 180, 'policy': {}}
    exec(remove_idle_cutoff(SOURCE), scope)
    assert scope['cap'](125, False) == 68
    assert math.isinf(scope['cap'](125, True))


def test_changed_source_is_rejected():
    with pytest.raises(ValueError, match='one idle cutoff'):
        remove_idle_cutoff(SOURCE.replace('max_idle=', 'different='))


def test_duplicate_source_is_rejected():
    with pytest.raises(ValueError, match='one idle cutoff'):
        remove_idle_cutoff(SOURCE + SOURCE)


def test_total_terminal_occupancy_includes_service_and_standby():
    data = {'headway_policy': {'demo': {'terminal_parking_capacity_by_station': {'T': 1}}},
            'trains': [{'train_id': str(i), 'state': 'moving', 'location': 'track'} for i in range(3)]}
    motion = {'events': [{'train_id': str(i), 'second': i, 'next_state': 'service', 'location': 'T'} for i in range(3)]}
    assert terminal_total_audit(data, motion)['passed'] is False


def test_terminal_intervals_are_half_open():
    data = {'headway_policy': {'demo': {'terminal_parking_capacity_by_station': {'T': 1}}},
            'trains': [{'train_id': str(i), 'state': 'service', 'location': 'T'} for i in range(2)] +
                      [{'train_id': '2', 'state': 'moving', 'location': 'track'}]}
    motion = {'events': [{'train_id': '2', 'second': 100, 'next_state': 'service', 'location': 'T'},
                         {'train_id': '0', 'second': 100, 'next_state': 'moving', 'location': 'track'}]}
    assert terminal_total_audit(data, motion)['passed'] is True
