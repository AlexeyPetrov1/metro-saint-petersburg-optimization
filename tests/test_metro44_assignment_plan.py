import copy
from time import monotonic

import pytest

from dwell import timing_key
from metro_daily import prepare_day
from metro44_assignment_plan import (build_assignment_problem, apply_assignment_plan,
                                     service_clocks, terminal_resources)
from simulation import simulate


@pytest.fixture(scope='module')
def day():
    return prepare_day('2026-09-30',service_hours=6)[0]


def test_fixed_jobs_partition_every_original_service_without_using_demand(day):
    original=copy.deepcopy(day);problem=build_assignment_problem(day,3600)
    retained=set(problem['retained_ids']);jobs=problem['jobs']
    service={t['trip_id']for t in day['plans'][0]['trips']if t['passenger']}
    assert {j['job_id']for j in jobs} | (service&retained) == service
    assert not {j['job_id']for j in jobs} & retained
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in day['leg_timing']}
    assert all(j['departure_second']==timing[timing_key(j['original_train_id'],j['job_id'],0)]['departure_second'] for j in jobs)
    assert day==original


def test_started_entry_retains_its_immediate_service(day):
    entry=next(t for t in day['plans'][0]['trips']if t['route'][0].startswith('ENTRY_'))
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in day['leg_timing']}
    second=timing[timing_key(entry['train_id'],entry['trip_id'],0)]['departure_second']+1
    owned=sorted((t for t in day['plans'][0]['trips']if t['train_id']==entry['train_id']),key=lambda t:t['departure_second'])
    following=next(t for t in owned[owned.index(entry)+1:]if t['passenger'])
    retained=set(build_assignment_problem(day,second)['retained_ids'])
    assert {entry['trip_id'],following['trip_id']}<=retained


def test_prescheduled_release_with_preparation_before_checkpoint_is_retained(day):
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in day['leg_timing']}
    tracks={r['track_id']:r for r in day['tracks']}
    for entry in day['plans'][0]['trips']:
        if not entry['route'][0].startswith('ENTRY_'):continue
        owned=sorted((t for t in day['plans'][0]['trips']if t['train_id']==entry['train_id']),key=lambda t:t['departure_second'])
        following=next(t for t in owned[owned.index(entry)+1:]if t['passenger'])
        e=timing[timing_key(entry['train_id'],entry['trip_id'],0)]['departure_second']
        s=timing[timing_key(following['train_id'],following['trip_id'],0)]['departure_second']
        release=720 if 'AVTOVO'in tracks[entry['route'][0]]['from_station']else 780
        if e>=3600 and s-release<3600:
            problem=build_assignment_problem(day,3600)
            assert {entry['trip_id'],following['trip_id']}<=set(problem['retained_ids'])
            assert entry['trip_id']in problem['retained_preparation_obligations']
            break
    else:pytest.fail('fixture must contain a prescheduled preparation crossing the checkpoint')


def test_identity_assignment_preserves_every_service_clock_and_current_motion(day):
    original=copy.deepcopy(day);problem=build_assignment_problem(day,3600)
    assignments={j['job_id']:j['original_train_id']for j in problem['jobs']}
    changed,audit=apply_assignment_plan(day,3600,assignments,service_end_second=21600)
    before=simulate(day,day['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    assert service_clocks(day)==service_clocks(changed)
    assert [e for e in before['events']if e['second']<3600]==[e for e in after['events']if e['second']<3600]
    assert audit['current_movements_preserved']
    assert audit['movement_passed'] and after['obligations_closed']
    assert audit['terminal_resources']['passed']
    assert after['checks']['peak_trains_on_line']<=53
    assert day==original


def test_unprepared_reserve_is_rejected_before_compilation(day):
    problem=build_assignment_problem(day,3600)
    assignments={j['job_id']:j['original_train_id']for j in problem['jobs']}
    first=min(problem['jobs'],key=lambda j:j['departure_second'])
    reserve=next(t for t in problem['availability']if not t['on_line']and t['location']==first['origin'])
    assignments[first['job_id']]=reserve['train_id']
    with pytest.raises(ValueError,match='ready|preparation'):
        apply_assignment_plan(day,3600,assignments)


def test_terminal_resources_share_entry_and_inbound_slots():
    data={'headway_policy':{'demo':{'platform_capacity_by_station':{'A':2},'terminal_parking_capacity_by_station':{'A':1}}},
          'trains':[dict(train_id=f'T{i}',state='moving',location=f'R{i}')for i in range(3)]}
    motion={'duration_seconds':40,'events':[
        dict(train_id=f'T{i}',second=i*10,event_type='arrival',location='A',next_state='turnaround')for i in range(3)]}
    result=terminal_resources(data,motion)
    assert not result['passed']
    assert result['maximum_visits_by_terminal']=={'A':3}
    assert result['maximum_standby_by_terminal']=={'A':0}


def test_expired_assignment_deadline_does_no_work(day):
    with pytest.raises(TimeoutError):apply_assignment_plan(day,3600,{},deadline=monotonic()-1)
