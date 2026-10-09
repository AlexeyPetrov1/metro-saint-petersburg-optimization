import gzip
import json
from time import monotonic

import pytest

from metro48_daily_reference import build_daily_reference, terminal_slot


def test_terminal_slot_releases_at_half_open_boundary():
    assert terminal_slot([[0, 180], [100, 280]], 150) == 180
    assert terminal_slot([[0, 180], [100, 280]], 280) == 280


def test_terminal_slot_does_not_delay_one_overlapping_train():
    assert terminal_slot([[0, 500]], 100) == 100


def test_entry_reservation_needs_only_overlap_clearance():
    incoming = [7319, 7499]
    assert terminal_slot([[7169, 7349], incoming, [7301, 7481]], 7319, replaced=incoming) == 7349


def test_expired_deadline_stops_before_construction():
    with pytest.raises(TimeoutError):
        build_daily_reference({}, [], 65400, deadline=monotonic() - 1)


def test_direct_calendar_preserves_initial_and_bounds_approach_holds():
    with gzip.open('data/processed/metro_daily_experiments_20261009/baseline.json.gz', 'rt', encoding='utf8') as stream:
        baseline = json.load(stream)
    changed, audit = build_daily_reference(baseline['data'], baseline['roster'], 65400,
        deadline=monotonic() + 240)
    assert audit['movement_passed'], audit
    assert audit['current_movements_preserved']
    assert audit['peak_trains_on_line'] <= 53
    assert audit['peak_trains_on_line'] > 41
    assert audit['maximum_new_approach_holding_seconds'] <= 150
    assert audit['approach_holding_limit_seconds'] == 150
    assert changed['trains'] == baseline['data']['trains']
    timings = {(r['train_id'], r['trip_id'], r['leg_index']): r for r in changed['leg_timing']}
    trips = {(t['train_id'], t['trip_id']): t for t in changed['plans'][0]['trips']}
    for train in baseline['data']['trains']:
        if train['location'].startswith('DEPOT_'):
            continue
        first = min((t for t in baseline['data']['plans'][0]['trips'] if t['train_id'] == train['train_id'] and t['passenger']), key=lambda t: t['departure_second'])
        assert trips[train['train_id'], first['trip_id']] == first
        for row in baseline['data']['leg_timing']:
            if row['train_id'] == train['train_id'] and row['trip_id'] in ('@initial', first['trip_id']):
                assert timings[row['train_id'], row['trip_id'], row['leg_index']] == row
    assert audit['future_passenger_truth_used'] is False
    assert next(r['count'] for r in audit['line_counts'] if r['second'] == 7200) >= 49
    late_peak = [r['second'] for r in audit['dispatch'] if r['terminal'] == 'L1_01' and 12000 <= r['second'] < 12600]
    assert min(b - a for a, b in zip(late_peak, late_peak[1:])) < 150
    peak = [r['second'] for r in audit['dispatch'] if r['terminal'] == 'L1_01' and 7200 <= r['second'] < 14000]
    assert min(b - a for a, b in zip(peak, peak[1:])) == 113
