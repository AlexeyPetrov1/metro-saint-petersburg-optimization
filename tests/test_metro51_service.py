import copy

import pytest

from metro51_service import local_boarding_service, compare_waiting_budget
from test_metro50_peak_service import fixture


@pytest.mark.parametrize('day,hour',[('2026-10-03','08:00:00'),('2026-09-30','12:00:00')])
def test_local_gate_includes_offpeak_and_weekends(day,hour):
    runner,state=fixture(day=day,hour=hour,boards=((60,94),(180,6)))
    result=local_boarding_service(runner,state)
    assert result['status']=='failed' and result['passed'] is False
    assert result['cohorts'][0]['missed_first_count']==6


def test_both_limits_are_inclusive_without_rounding_up_small_cohorts():
    runner,state=fixture(count=1000,boards=((60,950),(180,40),(300,10)),departure_times=(60,180,300))
    result=local_boarding_service(runner,state)
    row=result['cohorts'][0]
    assert result['passed'] is True
    assert row['missed_first_share']==.05 and row['missed_second_share']==.01
    assert row['passenger_count']==1000 and row['missed_second_count']==10
    runner,state=fixture(count=19,boards=((60,18),(180,1)))
    assert local_boarding_service(runner,state)['passed'] is False


def test_one_percent_repeat_denominator_is_entire_unique_cohort():
    runner,state=fixture(count=1000,boards=((60,950),(180,39),(300,11)),departure_times=(60,180,300))
    result=local_boarding_service(runner,state)
    assert result['status']=='failed'
    assert result['cohorts'][0]['checks']['first_limit_passed'] is True
    assert result['cohorts'][0]['checks']['second_limit_passed'] is False


def test_station_gate_cannot_be_hidden_by_line_average():
    runner,state=fixture(count=1000,boards=((60,1000),),stations=('A','B','C'))
    extra=copy.deepcopy(runner.groups[0]);extra.update(group_id='X',count=1,origin='B')
    extra['destinations']=[dict(station_id='C',count=1,direction=1)]
    runner.groups.append(extra);runner.by_id['X']=extra
    for second in (60,180):
        event=copy.deepcopy(runner.stream[0]);event.update(event_id=f'B{second}',second=second,location='BC',remaining_route=['BC'])
        runner.stream.append(event)
    runner.stream.sort(key=lambda e:e['second']);state['next_event_index']=len(runner.stream)
    state['board_events'].append(dict(group_id='X',destination='C',motion_event_id='B180',second=180,count=1))
    state['queues']={}
    result=local_boarding_service(runner,state)
    assert result['metrics']['missed_first_share']<.05 and result['status']=='failed'
    assert result['worst']['first']['station_id']=='B'


def test_repeated_attempts_count_people_once_and_preserve_maximum_misses():
    runner,state=fixture(boards=((60,90),(300,5),(420,5)),departure_times=(60,180,300,420))
    state['denied_boarding_attempts']=9999
    result=local_boarding_service(runner,state,first_limit=.1,second_limit=.1)
    row=result['cohorts'][0]
    assert row['missed_first_count']==row['missed_second_count']==10
    assert row['max_missed_trains']==3
    assert row['missed_opportunity_count']==25 and result['passed'] is True


@pytest.mark.parametrize('boards,departures',[ ((),()),(((60,95),),(60,)) ])
def test_unobserved_opportunities_are_incomplete(boards,departures):
    runner,state=fixture(boards=boards,departure_times=departures)
    result=local_boarding_service(runner,state)
    assert result['status']=='incomplete' and result['passed'] is None
    assert result['checks']['observation_complete'] is False


def test_complete_first_boarding_needs_no_artificial_second_departure():
    runner,state=fixture(boards=((60,100),),departure_times=(60,))
    assert local_boarding_service(runner,state)['passed'] is True


def test_partial_quarter_remains_incomplete_with_later_followup():
    runner,state=fixture(platform=0,boards=((60,100),))
    result=local_boarding_service(runner,state,end_second=600)
    assert result['status']=='incomplete'
    assert result['cohorts'][0]['temporal_coverage_complete'] is False


def test_platform_arrival_controls_cohort_and_later_followup():
    runner,state=fixture(hour='07:59:00',platform=120,boards=((180,95),(300,5)),departure_times=(180,300),end=1800)
    result=local_boarding_service(runner,state,start_second=60,end_second=960)
    assert result['cohorts'][0]['interval_start']=='2026-09-30T08:00:00+03:00'
    assert result['passed'] is True


def test_short_turn_is_not_a_first_opportunity_for_a_far_destination():
    runner,state=fixture(stations=('A','B','C'),boards=((180,100),))
    runner.stream[0]['remaining_route']=['AB']
    assert local_boarding_service(runner,state)['metrics']['missed_first_count']==0


def test_ledger_is_read_only_and_wrong_od_balance_is_rejected():
    runner,state=fixture();before=copy.deepcopy(state)
    local_boarding_service(runner,state)
    assert state==before
    state['board_events'][0]['count']+=1
    with pytest.raises(ValueError,match='balance|ledger'):
        local_boarding_service(runner,state)


@pytest.mark.parametrize('key,value',[('first_limit',True),('first_limit',-.1),('second_limit',float('nan')),
    ('second_limit',1.01),('start_second',True),('end_second',1000)])
def test_invalid_limits_and_windows_are_rejected(key,value):
    runner,state=fixture()
    with pytest.raises(ValueError): local_boarding_service(runner,state,**{key:value})


def test_wait_budget_is_additive_and_reports_whole_window_and_each_peak():
    base,baseline=fixture(boards=((60,100),))
    other,candidate=fixture()
    result=compare_waiting_budget(base,baseline,other,candidate)
    assert result['passed'] is True
    assert result['windows'][0]['mean_delta_seconds']==6
    assert result['windows'][1]['name']=='morning_peak'
    assert result['windows'][1]['p95_delta_seconds']==0


def test_wait_budget_never_passes_an_unboarded_cohort():
    base,baseline=fixture(boards=((60,100),))
    other,candidate=fixture(boards=((60,95),),departure_times=(60,))
    result=compare_waiting_budget(base,baseline,other,candidate)
    assert result['status']=='incomplete' and result['passed'] is None
    assert result['windows'][0]['candidate']['unboarded']==5


def test_wait_budget_rejects_mean_and_p95_regressions():
    base,baseline=fixture(boards=((60,100),))
    other,candidate=fixture(boards=((60,90),(180,10)))
    result=compare_waiting_budget(base,baseline,other,candidate)
    row=result['windows'][0]
    assert result['passed'] is False
    assert row['mean_delta_seconds']==12 and row['p95_delta_seconds']==120


def test_wait_budget_inclusive_boundary_uses_exact_passenger_sums():
    base,baseline=fixture(count=7,boards=((22,6),(23,1)),departure_times=(22,23))
    other,candidate=fixture(count=7,boards=((32,6),(33,1)),departure_times=(32,33))
    assert compare_waiting_budget(base,baseline,other,candidate)['passed'] is True


@pytest.mark.parametrize('field,value',[('second',-1),('second',901),('count',0)])
def test_wait_budget_rejects_invalid_boarding_times_and_counts(field,value):
    base,baseline=fixture(boards=((60,100),))
    other,candidate=fixture(boards=((60,100),))
    candidate['board_events'][0][field]=value
    with pytest.raises(ValueError,match='ledger'):
        compare_waiting_budget(base,baseline,other,candidate)


def test_wait_budget_rejects_cross_group_duplicates_even_if_total_is_preserved():
    base,baseline=fixture(boards=((60,100),))
    other,candidate=fixture(boards=((60,100),))
    for runner,state in ((base,baseline),(other,candidate)):
        group=copy.deepcopy(runner.groups[0]);group['group_id']='X'
        runner.groups.append(group);runner.by_id['X']=group
        board=copy.deepcopy(state['board_events'][0]);board['group_id']='X'
        state['board_events'].append(board)
    candidate['board_events'][0]['count']=101
    candidate['board_events'][1]['count']=99
    with pytest.raises(ValueError,match='ledger'):
        compare_waiting_budget(base,baseline,other,candidate)


def test_opposite_directions_at_one_station_have_separate_denominators():
    runner,state=fixture(count=20,boards=(),stations=('A','B','C'))
    group=runner.groups[0];group.update(origin='B')
    group['destinations']=[dict(station_id='A',count=10,direction=-1),
                           dict(station_id='C',count=10,direction=1)]
    runner.tracks=dict(BA=dict(from_station='B',to_station='A'),BC=dict(from_station='B',to_station='C'))
    runner.stream=[]
    for second in (60,180):
        for track in ('BA','BC'):
            runner.stream.append(dict(event_type='departure',second=second,event_id=f'{track}{second}',
                passenger_service=True,location=track,remaining_route=[track]))
    state.update(next_event_index=4,queues={},board_events=[
        dict(group_id='G',destination='A',motion_event_id='BA60',second=60,count=9),
        dict(group_id='G',destination='C',motion_event_id='BC60',second=60,count=10),
        dict(group_id='G',destination='A',motion_event_id='BA180',second=180,count=1)])
    result=local_boarding_service(runner,state)
    assert result['metrics']['missed_first_share']==.05 and result['status']=='failed'
    assert result['worst']['first']['direction']==-1
    assert result['worst']['first']['passenger_count']==10
