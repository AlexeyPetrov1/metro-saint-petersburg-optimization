import copy
import json
from pathlib import Path
from time import monotonic

import pytest

from dwell import timing_key
from metro47_withdrawal import (apply_withdrawals, build_withdrawal_candidate,
                                close_service_at, withdrawal_options)
from simulation import simulate


@pytest.fixture(scope='module')
def source():
    path = Path('data/processed/metro44_bridge_off160_actual_6h_20261009_v2/runs/OFF160_BRIDGE_S1_7/movement_input.json')
    return json.loads(path.read_text(encoding='utf-8'))


def test_options_use_whole_begun_trip_and_known_next_terminal(source):
    original = copy.deepcopy(source)
    options = withdrawal_options(source, 6300, 22200)
    assert options
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in source['leg_timing']}
    trips = {t['trip_id']: t for t in source['plans'][0]['trips']}
    for option in options:
        assert 6300 <= option['ready_second'] <= 7200
        assert option['exit_second'] == option['ready_second']
        assert option['cancel_service_ids']
        assert not set(option['begun_service_ids']) & set(option['cancel_service_ids'])
        assert all(timing[timing_key(option['train_id'], identity, 0)]['departure_second'] >= option['ready_second']
                   for identity in option['cancel_service_ids'])
        assert all(trips[i]['train_id'] == option['train_id'] for i in option['cancel_service_ids'])
    assert source == original


def test_withdrawal_keeps_every_other_row_and_current_arrival_exact(source):
    original = copy.deepcopy(source)
    option = withdrawal_options(source, 6300, 22200)[0]
    changed, audit = apply_withdrawals(source, [option], 6300, 22200)
    removed = set(option['cancel_service_ids'] + option['cancel_exit_ids'] + option['cancel_entry_ids'])
    new_id = audit['appended_exit_ids'][0]
    assert [t for t in changed['plans'][0]['trips'] if t['trip_id'] != new_id] == [
        t for t in source['plans'][0]['trips'] if t['trip_id'] not in removed]
    assert [r for r in changed['leg_timing'] if r['trip_id'] != new_id] == [
        r for r in source['leg_timing'] if r['trip_id'] not in removed]
    assert audit['executed_prefix_preserved'] and audit['current_movements_preserved']
    assert audit['kept_clocks_preserved'] and audit['obligations_closed']
    assert audit['terminal_resources']['passed'] and audit['peak_trains_on_line'] <= 53
    exit_trip = next(t for t in changed['plans'][0]['trips'] if t['trip_id'] == new_id)
    assert not exit_trip['passenger'] and exit_trip['departure_second'] == option['ready_second']
    before = simulate(source, source['plans'][0]['plan_id'])
    after = simulate(changed, changed['plans'][0]['plan_id'])
    assert [e for e in before['events'] if e['second'] < 6300] == [e for e in after['events'] if e['second'] < 6300]
    assert float(audit['estimated_to1_resource_saving_rub']) > 0
    assert source == original


def test_canonical_option_rejects_fabricated_readiness_or_past_cancellation(source):
    option = withdrawal_options(source, 6300, 22200)[0]
    for key, value in [('ready_second', option['ready_second'] - 1), ('cancel_service_ids', option['begun_service_ids'])]:
        tampered = dict(option, **{key: value})
        with pytest.raises(ValueError, match='source|canonical'):
            apply_withdrawals(source, [tampered], 6300, 22200)


def test_builder_rejects_unknown_duplicate_train_and_retiming(source):
    option = withdrawal_options(source, 6300, 22200)[0]
    for names in [['NO_SUCH_TRAIN'], [option['train_id']] * 2]:
        with pytest.raises(ValueError):
            build_withdrawal_candidate(source, 6300, 22200, selected_train_ids=names)
    with pytest.raises(ValueError, match='phase'):
        build_withdrawal_candidate(source, 6300, 22200, selected_train_ids=[option['train_id']], phase_seconds=1)


def test_expired_deadline_does_no_work(source):
    with pytest.raises(TimeoutError):
        withdrawal_options(source, 6300, 22200, deadline=monotonic() - 1)


def test_close_service_cutoff_keeps_begun_and_all_kept_clocks(source):
    original = copy.deepcopy(source)
    changed, audit = close_service_at(source, 3600, 18000)
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in changed['leg_timing']}
    assert all(timing[timing_key(t['train_id'], t['trip_id'], 0)]['departure_second'] < 18000
               for t in changed['plans'][0]['trips'] if t['passenger'])
    removed = set(audit['cancelled_trip_ids']); added = set(audit['appended_exit_ids'])
    assert [t for t in changed['plans'][0]['trips'] if t['trip_id'] not in added] == [
        t for t in source['plans'][0]['trips'] if t['trip_id'] not in removed]
    assert [r for r in changed['leg_timing'] if r['trip_id'] not in added] == [
        r for r in source['leg_timing'] if r['trip_id'] not in removed]
    assert audit['executed_prefix_preserved'] and audit['current_movements_preserved']
    assert audit['obligations_closed'] and audit['terminal_resources']['passed']
    assert audit['peak_trains_on_line'] <= 53
    assert source == original


def test_cutoff_normalization_is_idempotent(source):
    changed, _ = close_service_at(source, 3600, 18000)
    twice, audit = close_service_at(changed, 3600, 18000)
    assert twice == changed
    assert not audit['cancelled_trip_ids'] and not audit['appended_exit_ids']
