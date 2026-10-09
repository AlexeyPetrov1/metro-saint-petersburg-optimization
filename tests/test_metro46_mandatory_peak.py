import copy
from time import monotonic

import pytest

from metro46_mandatory_peak import causal_peak_grid, advance_unstarted_services


def test_peak_grid_joins_the_real_last_departure():
    assert causal_peak_grid(7155, 7200, 7200, 8100, 125, 113) == list(range(7268, 8100, 125))


def test_terminal_readiness_moves_the_phase_without_inventing_a_train():
    assert causal_peak_grid(7100, 7285, 7200, 8100, 125, 113) == list(range(7285, 8100, 125))


def test_peak_grid_does_not_cross_the_selected_window():
    assert causal_peak_grid(0, 900, 0, 900, 125, 113) == []


@pytest.mark.parametrize('target', [True, 112, 481, 125.5])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        causal_peak_grid(0, 0, 0, 900, target, 113)


@pytest.fixture(scope='module')
def source():
    from goal43_policy import load_policy
    from goal43_replay import build_operational_day
    return build_operational_day(load_policy(), count=53,
        start_at='2026-09-30T08:15:00+03:00', service_hours=4)[0]


def test_identity_producer_preserves_all_source_obligations(source):
    old = copy.deepcopy(source)
    changed, audit = advance_unstarted_services(source, 6300, {}, 15000,
                                               deadline=monotonic() + 60)
    assert source == old and changed == source
    assert audit['mandatory_service_jobs_preserved'] and audit['executed_prefix_preserved']
    assert audit['current_movements_preserved']


def test_past_service_cannot_be_reissued(source):
    trip = next(t for t in source['plans'][0]['trips'] if t['passenger'])
    with pytest.raises(ValueError, match='unstarted'):
        advance_unstarted_services(source, 6300, {trip['trip_id']: 6300}, 15000)


def test_unknown_service_id_is_rejected(source):
    with pytest.raises(ValueError, match='unknown'):
        advance_unstarted_services(source, 6300, {'missing': 7000}, 15000)


def test_expired_budget_stops_before_compilation(source):
    with pytest.raises(TimeoutError):
        advance_unstarted_services(source, 6300, {}, 15000, deadline=monotonic() - 1)
