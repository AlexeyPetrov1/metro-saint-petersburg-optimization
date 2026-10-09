import copy
from time import monotonic

import pytest

from dwell import timing_key
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_assignment_plan import _current_movements_preserved
from metro45_phase import rephase_bridge
from simulation import simulate


@pytest.fixture(scope='module')
def source():
    return build_operational_day(load_policy(), count=53,
        start_at='2026-09-30T08:15:00+03:00', service_hours=4)[0]


def test_future_phase_never_rewrites_past_or_current_movement(source):
    original = copy.deepcopy(source)
    changed, audit = rephase_bridge(source, 6300, 15000, shift_seconds=-20,
                                   deadline=monotonic() + 120)
    assert source == original
    before, after = (simulate(d, d['plans'][0]['plan_id']) for d in (source, changed))
    assert [e for e in before['events'] if e['second'] < 6300] == [e for e in after['events'] if e['second'] < 6300]
    assert _current_movements_preserved(source, changed, 6300)
    assert audit['movement_passed'] and audit['future_timing_flex_enabled']
    assert changed['tracks'] == source['tracks'] and changed['trains'] == source['trains']


def test_expired_deadline_stops_before_phase_search(source):
    with pytest.raises(TimeoutError):
        rephase_bridge(source, 6300, 15000, deadline=monotonic() - 1)


@pytest.mark.parametrize('shift', [True, -61, 61, 1.5])
def test_invalid_phase_window_is_rejected(source, shift):
    with pytest.raises(ValueError):
        rephase_bridge(source, 6300, 15000, shift_seconds=shift)
