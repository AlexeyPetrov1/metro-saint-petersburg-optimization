import copy
import gzip
import json
from pathlib import Path
from time import monotonic

import pytest

from dwell import timing_key
from metro44_bridge_reserves import build_reserve_bridge,rebuild_reserve_bridge_timetable,standby_repair_bounds,max_gap_repair_bounds
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_assignment_plan import _current_movements_preserved,terminal_resources
from simulation import simulate


@pytest.fixture(scope='module')
def shared():
    path=Path('data/processed/metro_daily_experiments_20261009/baseline.json.gz')
    return json.loads(gzip.decompress(path.read_bytes()))


def clocks(data):
    return {(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second'] for r in data['leg_timing']}


def test_future_entries_use_explicit_ready_and_full_release(shared):
    source=shared['data'];old=copy.deepcopy(source)
    out,audit=build_reserve_bridge(source,14400,65400,headway=1,readiness=shared['roster'])
    assert source==old and out['trains']==source['trains'] and out['tracks']==source['tracks']
    entries=audit['future_entry_obligations']
    assert entries and {r['release_seconds'] for r in entries}=={720,780}
    trips={t['trip_id']:t for t in out['plans'][0]['trips']}
    for row in entries:
        assert row['first_service_not_before']>=14400+row['release_seconds']
        assert trips[row['entry_trip_id']]['departure_second']==row['first_service_not_before']-row['entry_lead_seconds']
        assert row['readiness_status']=='assumed'


def test_begun_entry_and_first_service_are_committed_without_inferred_preparation(shared):
    source=shared['data'];old=clocks(source)
    entry=next(t for t in source['plans'][0]['trips']if t['trip_id']=='B020:D0:ENTRY:40')
    as_of=old[(entry['train_id'],entry['trip_id'],0)]+1
    out,audit=build_reserve_bridge(source,as_of,65400,bridge_seconds=0,headway=1,readiness=shared['roster'])
    assert entry['trip_id'] in audit['retained_begun_entry_ids']
    identities=audit['retained_begun_entry_first_service_ids']
    assert identities
    source_trips={t['trip_id']:t for t in source['plans'][0]['trips']}
    changed={t['trip_id']:t for t in out['plans'][0]['trips']}
    for identity in identities:
        t=source_trips[identity]
        assert changed[identity]==t
        assert audit['first_departure_bounds'][timing_key(t['train_id'],identity,0)]==old[(t['train_id'],identity,0)]
    assert not audit['documented_preparations']


def test_unknown_readiness_is_not_replaced_by_initial_zero(shared):
    roster=copy.deepcopy(shared['roster'])
    next(r for r in roster if r['train_id']=='B028')['ready_at']=None
    with pytest.raises(ValueError,match='readiness'):
        build_reserve_bridge(shared['data'],14400,65400,headway=1,readiness=roster)


def test_documented_preparation_is_explicit_and_checked(shared):
    source=shared['data'];entry='B019:D0:ENTRY:42'
    _,fresh=build_reserve_bridge(source,3700,65400,bridge_seconds=0,headway=1,readiness=shared['roster'])
    _,prepared=build_reserve_bridge(source,3700,65400,bridge_seconds=0,headway=1,readiness=shared['roster'],
        preparation_started={entry:3300})
    a=next(r for r in fresh['future_entry_obligations']if r['entry_trip_id']==entry)
    b=next(r for r in prepared['future_entry_obligations']if r['entry_trip_id']==entry)
    assert a['release_ready_second']==4420 and b['release_ready_second']==4020
    assert prepared['documented_preparations']=={entry:3300}
    with pytest.raises(ValueError,match='preparation'):
        build_reserve_bridge(source,3700,65400,headway=1,readiness=shared['roster'],preparation_started={entry:3701})


def test_zero_turnaround_depot_reentry_is_rejected(shared):
    data=copy.deepcopy(shared['data'])
    exit_trip=dict(trip_id='test:prior_exit',train_id='B028',departure_second=100,route=['EXIT_NORTHERN'],passenger=False)
    data['plans'][0]['trips'].append(exit_trip)
    data['leg_timing'].append(dict(train_id='B028',trip_id='test:prior_exit',leg_index=0,departure_second=100,service_seconds=0,holding_seconds=0,approach_holding_seconds=0))
    next(s for s in data['stations']if s['station_id']=='DEPOT_NORTHERN')['turnaround_seconds']=0
    with pytest.raises(ValueError,match='zero.*reentry'):
        build_reserve_bridge(data,14400,65400,headway=1,readiness=shared['roster'])


def test_past_depot_return_must_finish_before_fresh_preparation(shared):
    data=copy.deepcopy(shared['data'])
    data['plans'][0]['trips'].append(dict(trip_id='test:return',train_id='B028',departure_second=100,route=['EXIT_NORTHERN'],passenger=False))
    data['leg_timing'].append(dict(train_id='B028',trip_id='test:return',leg_index=0,departure_second=100,
        service_seconds=180,holding_seconds=0,approach_holding_seconds=0))
    _,audit=build_reserve_bridge(data,150,65400,bridge_seconds=0,headway=1,readiness=shared['roster'])
    row=next(r for r in audit['future_entry_obligations']if r['train_id']=='B028')
    assert row['preparation_start_second']==400
    assert row['release_ready_second']==1180


def test_two_hour_profile_uses_clock_fallback_after_its_end(shared):
    profile=[dict(start_second=14400,end_second=18000,headway_seconds=179),
             dict(start_second=18000,end_second=21600,headway_seconds=197)]
    _,audit=build_reserve_bridge(shared['data'],14400,65400,headway=profile,readiness=shared['roster'])
    assert {179,197,203}<=set(audit['profile_resolved_headways'])
    assert max(audit['profile_resolved_headways'])<=480
    assert audit['retained_bridge_service_ids']
    assert audit['allocation_changed'] is False


def test_complete_small_transition_preserves_physics_and_source():
    source=build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]
    original=copy.deepcopy(source)
    changed,audit=rebuild_reserve_bridge_timetable(source,6300,15000,headway=160,deadline=monotonic()+120)
    before=simulate(source,source['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert source==original and audit['movement_passed'] and audit['headways_passed']
    assert [e for e in before['events']if e['second']<6300]==[e for e in after['events']if e['second']<6300]
    assert _current_movements_preserved(source,changed,6300)
    assert terminal_resources(changed,after)['passed'] and after['obligations_closed']
    assert after['checks']['peak_trains_on_line']<=53
    assert audit['retained_bridge_service_ids'] and audit['bridge_departures_preserved']


def test_expired_deadline_stops_before_generation(shared):
    with pytest.raises(TimeoutError):
        rebuild_reserve_bridge_timetable(shared['data'],14400,65400,readiness=shared['roster'],deadline=monotonic()-1)


@pytest.mark.parametrize('bridge',[True,-1,481,1.5])
def test_bridge_duration_is_bounded(shared,bridge):
    with pytest.raises(ValueError):
        build_reserve_bridge(shared['data'],14400,65400,bridge_seconds=bridge,readiness=shared['roster'])


@pytest.fixture
def parking_trace():
    data=dict(headway_policy=dict(demo=dict(terminal_parking_capacity_by_station={'T':1})),
        stations=[dict(station_id='T',turnaround_seconds=180)],tracks=[dict(track_id='approach',travel_seconds=120)],
        plans=[dict(trips=[dict(trip_id=t,train_id=t,route=['approach'])for t in ('A','B')])],
        leg_timing=[dict(train_id=t,trip_id=t,leg_index=0,departure_second=d,service_seconds=180,approach_holding_seconds=0)
                    for t,d in [('A',100),('B',150)]])
    motion=dict(duration_seconds=800,events=[
        dict(second=400,train_id='A',trip_id='A',location='T',next_state='standby'),
        dict(second=450,train_id='B',trip_id='B',location='T',next_state='standby'),
        dict(second=600,train_id='A',trip_id='A',location='approach',next_state='moving'),
        dict(second=700,train_id='B',trip_id='B',location='approach',next_state='moving')])
    return data,motion


def test_parking_repair_delays_only_not_started_inbound_leg(parking_trace):
    data,motion=parking_trace;old=copy.deepcopy(data)
    bounds=standby_repair_bounds(data,motion,50)
    assert bounds=={timing_key('B','B',0):300}
    assert data==old
    assert bounds[timing_key('B','B',0)]+120+180==600


def test_parking_repair_cannot_rewrite_started_approach(parking_trace):
    data,motion=parking_trace
    with pytest.raises(ValueError,match='started'):
        standby_repair_bounds(data,motion,200)


@pytest.fixture
def gap_trace():
    data=dict(headway_policy=load_policy(),plans=[dict(trips=[dict(trip_id=t,train_id=t,route=['T'],passenger=True)
              for t in ('A','B','C')])],leg_timing=[dict(train_id=t,trip_id=t,leg_index=0,departure_second=second)
                for t,second in [('A',10),('B',300),('C',831)]])
    motion=dict(events=[dict(event_type='departure',location='T',second=second,train_id=t,trip_id=t)
                for t,second in [('A',10),('B',300),('C',831)]])
    gap=dict(track_id='T',second=831,seconds=531,kind='completed_interval')
    return data,motion,[gap]


def test_maximum_gap_repair_delays_previous_future_service_by_51(gap_trace):
    data,motion,gaps=gap_trace;before=copy.deepcopy(data)
    bounds=max_gap_repair_bounds(data,motion,50,gaps)
    assert bounds=={timing_key('B','B',0):351}
    assert data==before and 831-bounds[timing_key('B','B',0)]==480
    assert bounds[timing_key('B','B',0)]-10<=480


def test_maximum_gap_repair_rejects_started_service(gap_trace):
    data,motion,gaps=gap_trace
    with pytest.raises(ValueError,match='started'):
        max_gap_repair_bounds(data,motion,301,gaps)


def test_maximum_gap_repair_does_not_create_bad_preceding_gap(gap_trace):
    data,motion,gaps=gap_trace
    motion['events'][0]['second']=-160
    data['leg_timing'][0]['departure_second']=-160
    with pytest.raises(ValueError,match='preceding'):
        max_gap_repair_bounds(data,motion,50,gaps)
