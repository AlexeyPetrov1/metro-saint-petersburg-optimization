import copy
from time import monotonic

import pytest

from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_planner_terminal import retime_terminal_plan, rebuild_terminal_timetable
from metro44_assignment_plan import terminal_resources,service_clocks
from simulation import simulate


@pytest.fixture(scope='module')
def day():
    return build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]


def test_coupled_compiler_preserves_source_prefix_current_moves_and_physical_tracks(day):
    original=copy.deepcopy(day)
    changed,audit=retime_terminal_plan(day,3600,14400)
    before=simulate(day,day['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert day==original
    assert changed['tracks']==day['tracks']
    assert service_clocks(changed)==service_clocks(day)
    assert [e for e in before['events']if e['second']<3600]==[e for e in after['events']if e['second']<3600]
    assert audit['current_movements_preserved']and audit['movement_passed']
    assert terminal_resources(changed,after)['passed']
    assert after['checks']['peak_trains_on_line']<=53 and after['obligations_closed']


def test_terminal_adapter_checks_expired_deadline_before_generation(day):
    with pytest.raises(TimeoutError):rebuild_terminal_timetable(day,3600,14400,deadline=monotonic()-1)


def test_zero_time_reserve_return_cannot_be_reinvented_by_terminal_adapter(day):
    original=copy.deepcopy(day)
    changed,audit=rebuild_terminal_timetable(day,3600,15000,headway=125,wear_priority=True)
    assert day==original
    assert audit['movement_passed']
    assert audit['terminal_resources']['passed']
    tracks={r['track_id']:r for r in changed['tracks']}
    for train in changed['trains']:
        owned=sorted((t for t in changed['plans'][0]['trips']if t['train_id']==train['train_id']),key=lambda t:t['departure_second'])
        assert not any(a['route'][0].startswith('EXIT_')and b['route'][0].startswith('ENTRY_')for a,b in zip(owned,owned[1:]))
    assert all(r['track_id']in tracks for r in changed['previous_departures'])
