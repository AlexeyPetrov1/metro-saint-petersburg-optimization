import copy
import json
from decimal import Decimal
from pathlib import Path
from time import monotonic

import pytest

from dwell import timing_key
from metro44_catalog import physical_gate
from metro48_temporary import (apply_temporary_withdrawals, build_temporary_candidate,
                               temporary_options)
from simulation import simulate


@pytest.fixture(scope='module')
def source():
    path = Path('data/processed/metro44_bridge_off160_actual_6h_20261009_v2/runs/OFF160_BRIDGE_S1_7/movement_input.json')
    return json.loads(path.read_text(encoding='utf-8'))


def options(source):
    return temporary_options(source, 6300, 22200, return_by_second=17000)


def test_returns_same_depot_before_bound_without_touching_future_peak(source):
    original = copy.deepcopy(source)
    rows = options(source)
    assert rows
    tracks = {t['track_id']: t for t in source['tracks']}
    for row in rows:
        assert 6300 <= row['exit_second'] <= 7200
        assert row['resume_second'] <= 17000
        assert row['cancel_service_ids'] and len(row['cancel_service_ids']) % 2 == 0
        assert tracks[row['exit_track']]['to_station'] == tracks[row['entry_track']]['from_station']
        assert row['resume_trip_id'] not in row['cancel_service_ids']
        assert row['resume_second'] >= row['depot_ready_second'] + row['release_seconds']
        assert row['entry_second'] >= row['depot_ready_second'] + row['release_seconds'] - row['entry_lead_seconds']
    assert source == original


def test_two_hour_bound_does_not_invent_transfer_between_depots(source):
    assert not temporary_options(source, 6300, 22200, return_by_second=13500)


def test_return_preserves_every_kept_clock_and_passes_independent_physical_gate(source):
    original = copy.deepcopy(source)
    row = options(source)[0]
    changed, audit = apply_temporary_withdrawals(source, [row], 6300, 22200, return_by_second=17000)
    removed, added = set(row['cancel_service_ids']), set(audit['appended_exit_ids'] + audit['appended_entry_ids'])
    assert [t for t in changed['plans'][0]['trips'] if t['trip_id'] not in added] == [
        t for t in source['plans'][0]['trips'] if t['trip_id'] not in removed]
    assert [r for r in changed['leg_timing'] if r['trip_id'] not in added] == [
        r for r in source['leg_timing'] if r['trip_id'] not in removed]
    gate = physical_gate(changed, 6300, 22200, reference=source)
    assert gate['movement_passed'] and gate['peak_trains_on_line'] <= 53
    assert audit['depot_preparation']['passed'] and audit['executed_prefix_preserved']
    assert audit['current_movements_preserved'] and audit['obligations_closed']
    assert Decimal(audit['estimated_to1_resource_saving_rub']) == Decimal(row['to1_resource_saving_rub'])
    assert audit['cash_saving_rub'] is None and audit['release_authorized'] is False
    assert source == original


def test_release_lead_and_returnaround_are_both_enforced(source):
    rows = options(source)
    assert {r['release_seconds'] for r in rows} == {720, 780}
    tracks = {t['track_id']: t for t in source['tracks']}
    stations = {s['station_id']: s for s in source['stations']}
    for row in rows:
        assert row['entry_lead_seconds'] == tracks[row['entry_track']]['travel_seconds'] + 180
        assert row['entry_arrival_second'] + 180 <= row['resume_second']
    changed = copy.deepcopy(source)
    changed['headway_policy']['parameters']['depot_release_s']['value'] = dict(Avtovo=20000, Northern=20000)
    assert not options(changed)
    assert all(stations[tracks[r['exit_track']]['to_station']]['turnaround_seconds'] == 0 for r in rows)


def test_confirmed_preparation_supports_source_zero_depot_turn_without_geometry_change(source):
    original = copy.deepcopy(source)
    changed, audit = apply_temporary_withdrawals(source, [options(source)[0]], 6300, 22200,
                                                return_by_second=17000)
    assert changed['stations'] == source['stations']
    assert audit['movement_passed'] and audit['depot_preparation']['passed']
    assert source == original


def test_one_second_short_of_preparation_is_rejected_independently(source):
    row = options(source)[0]
    changed, audit = apply_temporary_withdrawals(source, [row], 6300, 22200, return_by_second=17000)
    identity = audit['appended_entry_ids'][0]
    early = row['depot_ready_second'] + row['release_seconds'] - row['entry_lead_seconds'] - 1
    next(t for t in changed['plans'][0]['trips'] if t['trip_id'] == identity)['departure_second'] = early
    next(t for t in changed['leg_timing'] if t['trip_id'] == identity)['departure_second'] = early
    with pytest.raises(ValueError, match='depot preparation'):
        simulate(changed, changed['plans'][0]['plan_id'])


def test_source_nominal_trip_bound_never_reorders_actual_return_movements():
    path = Path('data/processed/metro47_withdrawal_normalized_actual_20261009/runs/NORMALIZED_REFERENCE_S1_7/movement_input.json')
    source = json.loads(path.read_text(encoding='utf-8'))
    row = next(r for r in temporary_options(source, 3600, 22200, return_by_second=21600)
               if r['depot'] == 'DEPOT_AVTOVO')
    resume = next(t for t in source['plans'][0]['trips'] if t['trip_id'] == row['resume_trip_id'])
    assert resume['departure_second'] < row['entry_second'] < row['resume_second']
    changed, audit = apply_temporary_withdrawals(source, [row], 3600, 22200, return_by_second=21600)
    assert audit['movement_passed'] and audit['kept_clocks_preserved']
    entry = next(t for t in changed['plans'][0]['trips'] if t['trip_id'] in audit['appended_entry_ids'])
    actual = next(r for r in changed['leg_timing'] if r['trip_id'] == entry['trip_id'])
    assert entry['departure_second'] < resume['departure_second']
    assert actual['departure_second'] == row['entry_second']


@pytest.mark.parametrize('field', ['entry_second', 'resume_second', 'depot_ready_second', 'cancel_service_ids'])
def test_rejects_fabricated_preparation_or_begun_service_cancellation(source, field):
    row = options(source)[0]
    tampered = dict(row, **{field: row[field] - 1 if isinstance(row[field], int) else row['begun_service_ids']})
    with pytest.raises(ValueError, match='canonical'):
        apply_temporary_withdrawals(source, [tampered], 6300, 22200, return_by_second=17000)


def test_builder_rejects_duplicate_unknown_train_and_phase(source):
    row = options(source)[0]
    for names in [[row['train_id']] * 2, ['UNKNOWN']]:
        with pytest.raises(ValueError):
            build_temporary_candidate(source, 6300, 22200, selected_train_ids=names, return_by_second=17000)
    with pytest.raises(ValueError, match='phase'):
        build_temporary_candidate(source, 6300, 22200, selected_train_ids=[], return_by_second=17000, phase_seconds=1)


def test_expired_deadline_and_invalid_return_bound_are_rejected(source):
    with pytest.raises(TimeoutError):
        temporary_options(source, 6300, 22200, return_by_second=17000, deadline=monotonic()-1)
    for bound in [True, 6300, 22201]:
        with pytest.raises(ValueError, match='return'):
            temporary_options(source, 6300, 22200, return_by_second=bound)
