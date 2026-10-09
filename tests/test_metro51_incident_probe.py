import copy

import pytest

from dwell import compile_operational_timing
from goal43_policy import load_policy
from passenger_groups import GroupedPassengerSimulation
from simulation import simulate
from test_goal43_motion import line
from metro51_incident_probe import running_stop, runner_from_checked_motion


@pytest.mark.parametrize('minutes',[3,8,15])
def test_running_stop_extends_committed_leg_and_propagates_without_rewriting_past(minutes):
    source,_=compile_operational_timing(line(),'P',policy=load_policy())
    before=copy.deepcopy(source);motion=simulate(source,'P')
    changed,audit=running_stop(source,100,minutes,train_id='T1')
    actual=simulate(changed,'P')
    assert source==before
    assert [e for e in motion['events'] if e['second']<100]==[e for e in actual['events'] if e['second']<100]
    arrival=next(e['second'] for e in actual['events'] if e['train_id']=='T1' and e['event_type']=='arrival' and e['location']=='B')
    departure=next(e['second'] for e in actual['events'] if e['train_id']=='T1' and e['event_type']=='departure' and e['location']=='BC_up')
    follower=next(e['second'] for e in actual['events'] if e['train_id']=='T2' and e['event_type']=='arrival' and e['location']=='B')
    assert arrival==140+minutes*60 and follower>departure
    assert audit['signalling_verified'] is False and audit['external_committed_movement_changed'] is True


def test_incident_checkpoint_resume_matches_continuous_replay_and_conserves_passengers():
    source,_=compile_operational_timing(line(),'P',policy=load_policy())
    cars=[dict(car_id=f'C{i}',car_type='fixture',max_capacity=160,normal_threshold=120,source='test') for i in range(8)]
    profiles={t['train_id']:copy.deepcopy(cars) for t in source['trains']}
    groups=[dict(group_id='G',turnstile_second=10,platform_second=10,origin='A',count=20,
        destinations=[dict(station_id='C',count=20)])]
    baseline=GroupedPassengerSimulation(source,'P',groups,profiles,boarding_seed=7,boarding_capacity_per_car=150)
    checkpoint=baseline.run(end_second=100)['checkpoint']
    changed,_=running_stop(source,100,8,train_id='T1')
    branch=GroupedPassengerSimulation(changed,'P',groups,profiles,boarding_seed=7,boarding_capacity_per_car=150)
    fork=branch.branch_checkpoint(checkpoint,baseline)
    continuous=branch.run(end_second=1500)['checkpoint']
    resumed=branch.run(start_second=100,end_second=1500,checkpoint=fork)['checkpoint']
    assert continuous==resumed
    assert resumed['entered']==resumed['completed']+resumed['queued']+resumed['onboard_count']+resumed['approaching']==20


@pytest.mark.parametrize('second,minutes,train',[(140,3,'T1'),(100,0,'T1'),(True,3,'T1'),(100,3,'absent')])
def test_stop_rejects_nonmoving_trains_and_invalid_arguments(second,minutes,train):
    source,_=compile_operational_timing(line(),'P',policy=load_policy())
    with pytest.raises(ValueError): running_stop(source,second,minutes,train_id=train)


def test_checked_motion_reuse_preserves_complete_events_and_passenger_results():
    data,_=compile_operational_timing(line(),'P',policy=load_policy())
    cars=[dict(car_id=f'C{i}',car_type='fixture',max_capacity=160,normal_threshold=120,source='test') for i in range(8)]
    profiles={t['train_id']:copy.deepcopy(cars) for t in data['trains']}
    groups=[dict(group_id='G',turnstile_second=0,platform_second=0,origin='A',count=20,
        destinations=[dict(station_id='C',count=20)])]
    expected=GroupedPassengerSimulation(data,'P',groups,profiles,boarding_seed=7,boarding_capacity_per_car=150)
    motion=simulate(data,'P');before=copy.deepcopy(motion)
    cached=runner_from_checked_motion(data,groups,profiles,motion)
    assert cached.motion==expected.motion and cached.stream==expected.stream
    assert cached.run()['checkpoint']==expected.run()['checkpoint']
    assert motion==before
