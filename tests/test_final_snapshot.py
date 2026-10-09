import pytest

from decision_engine import build_regular_plan
from finalize_goal4_2 import exact_snapshot


@pytest.mark.parametrize('end,state,time',[(480,'moving',510),(520,'turnaround',520)])
def test_end_snapshot_keeps_boundary_events_pending_and_future_obligations(end,state,time):
    config=dict(station_ids=['A','B'],date='2026-09-01',timezone='Europe/Moscow',length_m=2000,
        travel_seconds=120,stop_seconds=0,turnaround_seconds=10,train_type='fixture',car_types=['fixture'])
    data,_=build_regular_plan(config,2,end_second=end)
    result=exact_snapshot(data,end)
    train=next(t for t in result['trains'] if t['train_id']=='T01')
    assert train['state']==state and result['boundary_events']=='pending'
    obligation=train['unfinished_obligation']
    assert obligation.get('arrival_second',obligation.get('ready_second'))==time
