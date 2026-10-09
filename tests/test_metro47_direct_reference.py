import gzip
import json
from datetime import datetime
from time import monotonic

import pytest

from metro47_direct_reference import build_direct_reference, route_minimum


@pytest.fixture(scope='module')
def baseline():
    with gzip.open('data/processed/metro_daily_experiments_20261009/baseline.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def test_route_minimum_anticipates_offpeak_boundary(baseline):
    data = baseline['data']
    assert route_minimum(data, 'L1_01', 10800) == 113
    assert route_minimum(data, 'L1_01', 13200) == 150


def test_expired_budget_does_not_construct(baseline):
    with pytest.raises(TimeoutError):
        build_direct_reference(baseline['data'], baseline['roster'], 64800, deadline=monotonic() - 1)


def test_direct_daily_reference_preserves_initial_and_passes_full_physics(baseline):
    before = baseline['data']
    changed, audit = build_direct_reference(before, baseline['roster'], 65400, deadline=monotonic() + 240)
    assert audit['movement_passed'], audit
    assert audit['headways']['passed'] and audit['terminal_resources']['passed']
    assert audit['current_movements_preserved'] and audit['obligations_closed']
    assert 33 <= audit['peak_trains_on_line'] <= 53
    assert all(changed[k] == before[k] for k in ('trains', 'stations', 'tracks', 'headway_policy'))
    assert audit['future_passenger_truth_used'] is False


def test_peak_strengthening_preserves_initial_legs_and_maximum_fleet(baseline):
    changed, audit = build_direct_reference(baseline['data'], baseline['roster'], 65400,
        strengthen_peak=True, deadline=monotonic() + 240)
    assert audit['movement_passed'], audit
    assert audit['current_movements_preserved'] and audit['peak_trains_on_line'] <= 53
