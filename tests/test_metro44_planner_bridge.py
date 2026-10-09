import copy
from time import monotonic

import pytest

from dwell import timing_key
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_assignment_plan import _current_movements_preserved, terminal_resources
from metro44_planner_bridge import rebuild_bridge_timetable
from simulation import simulate


@pytest.fixture(scope='module')
def source():
    return build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]


@pytest.fixture(scope='module')
def transition(source):
    original=copy.deepcopy(source)
    changed,audit=rebuild_bridge_timetable(source,6300,15000,bridge_seconds=480,
        headway=160,wear_priority=True,deadline=monotonic()+120)
    assert source==original
    return changed,audit


def test_bridge_preserves_source_past_and_started_arrivals(source,transition):
    changed,audit=transition
    before=simulate(source,source['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert [e for e in before['events']if e['second']<6300]==[e for e in after['events']if e['second']<6300]
    assert _current_movements_preserved(source,changed,6300)
    assert changed['tracks']==source['tracks'] and changed['trains']==source['trains']
    assert audit['committed_until_second']==6300
    assert audit['bridge_end_second']==6780


def test_bridge_preserves_near_service_starts_without_freezing_later_schedule(source,transition):
    changed,audit=transition
    old={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in source['leg_timing']}
    new={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in changed['leg_timing']}
    identities={t['trip_id']:t for t in changed['plans'][0]['trips']}
    bridge=[];later=[]
    for t in source['plans'][0]['trips']:
        if not t['passenger']:continue
        key=timing_key(t['train_id'],t['trip_id'],0);at=old[key]['departure_second']
        if 6300<=at<6780:
            bridge.append(t['trip_id'])
            assert identities[t['trip_id']]==t and new[key]['departure_second']==at
        if 6780<=at<7080:later.append(t['trip_id'])
    assert bridge and sorted(bridge)==audit['retained_bridge_service_ids']
    assert later and any(new[timing_key(identities[t]['train_id'],t,0)]['departure_second']>
        old[timing_key(identities[t]['train_id'],t,0)]['departure_second'] for t in later)


def test_bridge_passes_shared_terminal_line_and_clock_constraints(source,transition):
    changed,audit=transition
    motion=simulate(changed,changed['plans'][0]['plan_id'])
    assert audit['movement_passed'] and audit['headways_passed']
    assert audit['maximum_headway']<=480 and audit['minimum_headway']>=150
    assert terminal_resources(changed,motion)['passed']
    assert motion['checks']['peak_trains_on_line']<=53 and motion['obligations_closed']
    assert audit['bridge_departures_preserved']


def test_bridge_stops_before_generation_on_expired_deadline(source):
    with pytest.raises(TimeoutError):
        rebuild_bridge_timetable(source,6300,15000,deadline=monotonic()-1)


@pytest.mark.parametrize('bridge',[True,-1,481,1.5])
def test_bridge_rejects_invalid_transition_duration(source,bridge):
    with pytest.raises(ValueError):rebuild_bridge_timetable(source,6300,15000,bridge_seconds=bridge)
