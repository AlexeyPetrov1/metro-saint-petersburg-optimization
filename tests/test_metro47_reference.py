import copy
from time import monotonic

import pytest

from metro47_reference import preparation_profile, build_reference, validate_readiness


@pytest.fixture(scope='module')
def day():
    from metro_daily import prepare_day
    return prepare_day('2026-09-30', service_hours=6)[0]


def test_preparation_profile_obeys_peak_and_offpeak_limits(day):
    rows = preparation_profile(day, 21600)
    assert rows[0]['headway_seconds'] == 150
    peak = [r for r in rows if 7200 <= r['start_second'] < 14400]
    assert peak and all(r['headway_seconds'] == 115 for r in peak)
    assert all(150 <= r['headway_seconds'] <= 480 for r in rows if r['start_second'] < 7200)


def test_initial_readiness_is_required_before_any_new_reference(day):
    with pytest.raises(ValueError, match='readiness'):
        build_reference(day, [], 10800, deadline=monotonic() + 60)


def test_unknown_reserve_readiness_is_not_filled_in(day):
    rows = [dict(train_id=t['train_id'], ready_at=day['period_start'], readiness_status='assumed') for t in day['trains']]
    rows[-1]['ready_at'] = None
    with pytest.raises(ValueError, match='readiness'):
        build_reference(day, rows, 10800, deadline=monotonic() + 60)


def test_an_active_initial_train_does_not_need_an_unused_depot_ready_at(day):
    rows = [dict(train_id=t['train_id'], ready_at=day['period_start'], readiness_status='assumed') for t in day['trains']]
    initial = next(t for t in day['trains'] if not t['location'].startswith('DEPOT_'))
    next(r for r in rows if r['train_id'] == initial['train_id']).update(ready_at=None)
    validate_readiness(day, rows)


def test_expired_deadline_stops_before_dispatch(day):
    with pytest.raises(TimeoutError):
        build_reference(day, [], 10800, deadline=monotonic() - 1)
