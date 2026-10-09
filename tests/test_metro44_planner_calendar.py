from datetime import datetime
from time import monotonic

import pytest

from goal43_policy import load_policy
from metro_daily import prepare_day
from metro44_planner_calendar import segment_calendars, paired_anchor, next_calendar_slot, rebuild_calendar_timetable


def test_offpeak_uses_the_same_joint_calendar_as_peak():
    origin=datetime.fromisoformat('2026-09-30T06:00:00+03:00')
    profile=[dict(start_second=3600,end_second=7200,headway_seconds=150),
             dict(start_second=7200,end_second=14400,headway_seconds=125)]
    rows=segment_calendars(['A','B'],origin,14400,profile,load_policy(),3040)
    a=next(r for r in rows['A']if r['start_second']==3600)
    b=next(r for r in rows['B']if r['start_second']==3600)
    assert a['phase_seconds']==0 and b['phase_seconds']==75
    assert b['end_second']==7200
    assert next_calendar_slot(3600,rows['B'])==3675
    assert next_calendar_slot(3680,rows['B'])==3825
    assert next_calendar_slot(7199,rows['B'])==7200
    assert next_calendar_slot(7200,rows['B'])==7262


def test_equal_adjacent_quarters_share_one_calendar_anchor():
    origin=datetime.fromisoformat('2026-09-30T06:00:00+03:00')
    profile=[dict(start_second=s,end_second=s+900,headway_seconds=150)for s in range(3600,7200,900)]
    rows=segment_calendars(['A','B'],origin,7200,profile,load_policy(),3040)
    assert [r for r in rows['B']if r['start_second']>=3600]==[
        dict(start_second=3600,end_second=7200,headway_seconds=150,phase_seconds=75)]


def test_start_anchor_uses_committed_clocks_and_actual_readiness():
    assert paired_anchor(3665,3600,150,75)==3675
    # B014 is physically ready only at3687; this must supersede the clock-only bound.
    anchor=paired_anchor(3687,3600,150,75)
    assert anchor==3687
    rows=[dict(start_second=3600,end_second=7200,headway_seconds=150,phase_seconds=75,anchor_second=anchor)]
    assert next_calendar_slot(3600,rows)==3612


def test_expired_deadline_aborts_the_extended_calendar_before_work():
    data=prepare_day('2026-09-30',service_hours=3)[0]
    with pytest.raises(TimeoutError):rebuild_calendar_timetable(data,3600,10800,deadline=monotonic()-1)


def test_slot_alignment_cannot_book_a_later_departure_before_the_other_terminal():
    data=prepare_day('2026-09-30',service_hours=3)[0]
    _,audit=rebuild_calendar_timetable(data,3600,10800,headway=150,deadline=monotonic()+60)
    seconds=[r['second']for r in audit['departures']]
    assert seconds==sorted(seconds)
