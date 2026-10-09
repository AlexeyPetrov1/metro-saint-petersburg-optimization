import copy
from time import monotonic

import pytest

from goal43_policy import load_policy
from metro_daily import prepare_day
from metro44_assignment_plan import terminal_resources, _current_movements_preserved
from metro44_planner_wait import balanced_phase, next_calendar_slot, reuse_available, rebuild_wait_timetable
from simulation import simulate


def test_long_wait_does_not_force_a_physically_eligible_train_into_depot():
    # Observed regression: Y010 was retired at 7305 despite all three real guards passing.
    assert 7305-7190 > 2*125-180-2
    assert reuse_available(7190,7305,capacity=True,platform=True,parking=True)
    for constraint in ('capacity','platform','parking'):
        limits=dict(capacity=True,platform=True,parking=True);limits[constraint]=False
        assert not reuse_available(7190,7305,**limits)
    assert not reuse_available(7306,7305,capacity=True,platform=True,parking=True)


def test_joint_terminal_phase_balances_real_standby_without_changing_turnaround():
    phase=balanced_phase(3040,125)
    assert phase==62
    waits=((phase-3040)%125,(-phase-3040)%125)
    assert waits==(22,23)
    assert 6080+sum(waits)==49*125
    assert all(180+wait<=2*125 for wait in waits)


def test_peak_calendar_keeps_its_phase_after_retry_and_source_clock_drift():
    calendar=[dict(start_second=7200,end_second=14400,headway_seconds=125,phase_seconds=62)]
    assert next_calendar_slot(7200,calendar)==7262
    assert next_calendar_slot(7262,calendar)==7262
    assert next_calendar_slot(7267,calendar)==7387
    assert next_calendar_slot(14399,calendar)==14400
    assert next_calendar_slot(14430,calendar)==14430
    with pytest.raises(ValueError):next_calendar_slot(7200,[dict(start_second=7200,end_second=14400,headway_seconds=125,phase_seconds=125)])


@pytest.fixture(scope='module')
def day():
    return prepare_day('2026-09-30',service_hours=6)[0]


def test_wait_planner_preserves_source_prefix_current_arrivals_and_real_terminal_limits(day):
    original=copy.deepcopy(day)
    changed,audit=rebuild_wait_timetable(day,3600,10800,headway=150,peak_calendar=False,deadline=monotonic()+60)
    before=simulate(day,day['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert day==original
    assert changed['tracks']==day['tracks'] and changed['trains']==day['trains']
    assert [e for e in before['events']if e['second']<3600]==[e for e in after['events']if e['second']<3600]
    assert _current_movements_preserved(day,changed,3600)
    assert terminal_resources(changed,after)['passed']
    assert after['checks']['peak_trains_on_line']<=53 and after['obligations_closed']
    assert audit['artificial_idle_cutoff_removed']


def test_expired_deadline_precedes_any_dispatch_work(day):
    with pytest.raises(TimeoutError):rebuild_wait_timetable(day,3600,10800,deadline=monotonic()-1)
