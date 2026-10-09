import copy
from time import monotonic

import pytest

from dwell import timing_key
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_linked_gap_repair import linked_gap_bounds, repair_linked_timetable
from metro44_assignment_plan import _current_movements_preserved,terminal_resources
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import simulate


def fixture():
    trips=[dict(train_id=f'T{i}',trip_id=f'S{i}',passenger=True,departure_second=at,route=['R','S'])
           for i,at in enumerate((16478,16928,17528))]
    data=dict(headway_policy=load_policy(),plans=[dict(plan_id='P',trips=trips)],
        stations=[dict(station_id=x,kind='line')for x in ('A','B','C')],
        tracks=[dict(track_id=r,from_station=a,to_station=b)for r,a,b in [('R','A','B'),('S','B','C')]],
        previous_departures=[],leg_timing=[])
    events=[]
    for t in trips:
        for i,r in enumerate(t['route']):
            at=t['departure_second']+100*i
            data['leg_timing'].append(dict(train_id=t['train_id'],trip_id=t['trip_id'],leg_index=i,departure_second=at))
            events.append(dict(event_type='departure',location=r,second=at,trip_id=t['trip_id'],train_id=t['train_id']))
    return data,dict(events=events)


def test_exact_450_600_conflict_moves_both_predecessors_and_preserves_vehicle_jobs():
    data,motion=fixture();old=copy.deepcopy(data)
    bounds=linked_gap_bounds(data,motion,16000,17628)
    assert bounds=={timing_key('T0','S0',0):16568,timing_key('T1','S1',0):17048}
    assert data==old
    assert 17048-16568==17528-17048==480


def test_linked_propagation_cannot_move_a_begun_service_even_for_a_later_track():
    data,motion=fixture()
    with pytest.raises(ValueError,match='started SERVICE'):
        linked_gap_bounds(data,motion,16500,17628)


def test_boundary_end_propagates_to_the_previous_neighbor_as_well():
    data,motion=fixture()
    bounds=linked_gap_bounds(data,motion,16000,18328)
    assert bounds[timing_key('T2','S2',0)]==17848
    assert bounds[timing_key('T1','S1',0)]==17368
    assert bounds[timing_key('T0','S0',0)]==16888


def test_optional_future_leg_hold_preserves_the_already_running_leg():
    data,motion=fixture()
    starts=(16478,16800,17200)
    for i,t in enumerate(data['plans'][0]['trips']):
        t['departure_second']=starts[i]
        next(r for r in data['leg_timing']if r['trip_id']==t['trip_id']and r['leg_index']==0)['departure_second']=starts[i]
        next(e for e in motion['events']if e['trip_id']==t['trip_id']and e['location']=='R')['second']=starts[i]
    bounds=linked_gap_bounds(data,motion,16500,17628,allow_future_legs=True)
    assert bounds[timing_key('T0','S0',1)]==16668
    assert timing_key('T0','S0',0)not in bounds
    assert bounds[timing_key('T1','S1',0)]==16920


def test_optional_future_leg_mode_still_refuses_a_departed_leg():
    data,motion=fixture()
    with pytest.raises(ValueError,match='departed leg'):
        linked_gap_bounds(data,motion,16500,17628,allow_future_legs=True)


def test_unchanged_valid_chain_checks_prefix_current_minimum_capacity_and_closure():
    data=build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]
    old=copy.deepcopy(data)
    changed,audit=repair_linked_timetable(data,6300,14400,deadline=monotonic()+60)
    motion=simulate(changed,changed['plans'][0]['plan_id'])
    assert data==old and changed==data
    assert audit['movement_passed']and not audit['repair_passes']
    assert audit['minimum_headway']>=113 and audit['maximum_headway']<=480
    assert _current_movements_preserved(data,changed,6300)
    assert terminal_resources(changed,motion)['passed']and motion['obligations_closed']
    assert motion['checks']['peak_trains_on_line']<=53


def test_compiled_future_leg_hold_preserves_a_real_started_movement():
    data=build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]
    old=copy.deepcopy(data);timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    trip=next(t for t in data['plans'][0]['trips']if t['passenger']and len(t['route'])>1)
    first=timing[timing_key(trip['train_id'],trip['trip_id'],0)]['departure_second'];as_of=first+1
    bounds={key:r['departure_second']for key,r in timing.items()if r['departure_second']>=0}
    future=timing_key(trip['train_id'],trip['trip_id'],1);bounds[future]+=20
    changed,_=compile_terminal_timing(data,data['plans'][0]['plan_id'],policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
    before=simulate(data,data['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert data==old and changed['plans']==data['plans']
    assert _current_movements_preserved(data,changed,as_of)
    assert [e for e in before['events']if e['second']<as_of]==[e for e in after['events']if e['second']<as_of]
    assert next(r for r in changed['leg_timing']if timing_key(r['train_id'],r['trip_id'],r['leg_index'])==future)['departure_second']>=bounds[future]
    assert terminal_resources(changed,after)['passed']and after['obligations_closed']
    assert after['checks']['peak_trains_on_line']<=53
    assert operational_headways(changed,after,as_of,14400)['passed']


@pytest.mark.parametrize('limit',[True,-1,6,1.5])
def test_connected_repair_passes_are_bounded(limit):
    with pytest.raises(ValueError):repair_linked_timetable({},3600,65400,passes=limit)


def test_deadline_precedes_any_repair_work():
    with pytest.raises(TimeoutError):repair_linked_timetable({},3600,65400,deadline=monotonic()-1)
