from types import SimpleNamespace
import gzip
import json

from economics import write_json
from goal43_extend import refresh_saved_series
from goal43_replay import _series


def test_headway_series_uses_maximum_across_directions_in_each_minute():
    events=[dict(second=second,event_type='departure',location=track,
                 train_id='T',next_state='moving')
            for second,track in [(0,'P_up'),(20,'P_down'),(130,'P_up'),(160,'P_down')]]
    runner=SimpleNamespace(motion=dict(period_start='2026-09-01T06:00:00+03:00',events=events),by_id={})
    data=dict(stations=[dict(station_id='A',kind='line'),dict(station_id='B',kind='line'),
                        dict(station_id='D',kind='depot')],
              tracks=[dict(track_id='P_up',from_station='A',to_station='B'),
                      dict(track_id='P_down',from_station='B',to_station='A')],
              trains=[dict(train_id='T',state='standby',location='D')])
    series=_series(runner,dict(load_events=[],queue_samples=[]),0,180,dict(rows=[]),data)
    assert series['headway']==[dict(second=120,seconds=140)]
    assert series['headway_label']=='Максимальный интервал по направлениям за минуту'


def test_headway_series_excludes_depot_paths_and_service_return_tail():
    events=[dict(second=second,event_type='departure',location=track,
                 train_id='T',next_state='moving')
            for second,track in [(0,'P_up'),(0,'EXIT_up'),(0,'DEPOT_down'),
                                 (90,'P_up'),(150,'P_up'),(179,'EXIT_up'),(179,'DEPOT_down')]]
    runner=SimpleNamespace(motion=dict(period_start='2026-09-01T06:00:00+03:00',events=events),by_id={})
    data=dict(stations=[dict(station_id='A',kind='line'),dict(station_id='D',kind='depot')],
              tracks=[dict(track_id='P_up',from_station='A',to_station='A'),
                      dict(track_id='EXIT_up',from_station='A',to_station='D'),
                      dict(track_id='DEPOT_down',from_station='D',to_station='A')],
              trains=[dict(train_id='T',state='stop',location='A')])
    state=dict(load_events=[],queue_samples=[])
    assert _series(runner,state,0,180,dict(rows=[]),data)['headway']==[
        dict(second=60,seconds=90),dict(second=120,seconds=60)]
    assert _series(runner,state,0,180,dict(rows=[]),data,service_end_second=120)['headway']==[
        dict(second=60,seconds=90)]


def test_refresh_uses_each_primary_run_including_rolling_selected(tmp_path):
    scenario=tmp_path/'scenarios/case'
    write_json(scenario/'forecast.json',dict(rows=[]))
    write_json(scenario/'summary.json',dict(comparisons=[dict(candidate_id='KEEP',series={}),
                                                        dict(candidate_id='ROLLING_SELECTED',series={})]))
    for candidate, last in [('KEEP',140),('ROLLING_SELECTED',160)]:
        folder=tmp_path/f'runs/case_{candidate}_S1_7'
        write_json(folder/'summary.json',dict(run_id=folder.name,scenario_id='case',od='S1',seed=7,
                   cohort_scope=dict(start_second=0,end_second=180)))
        events=[dict(second=s,event_type='departure',location='P_up',train_id='T',next_state='moving') for s in [0,last]]
        write_json(folder/'movement_input.json',dict(stations=[dict(station_id='A',kind='line')],
                   tracks=[dict(track_id='P_up',from_station='A',to_station='A')],
                   trains=[dict(train_id='T',state='stop',location='A')]))
        write_json(folder/'motion.json',dict(period_start='2026-09-01T06:00:00+03:00',events=events))
        for name,value in [('groups',[]),('passenger_checkpoint',dict(load_events=[],queue_samples=[]))]:
            with gzip.open(folder/f'{name}.json.gz','wt',encoding='utf-8') as handle:
                json.dump(value,handle)
    result=refresh_saved_series(tmp_path)
    assert result['comparisons_updated']==2
    rows=json.loads((scenario/'summary.json').read_text(encoding='utf-8'))['comparisons']
    assert rows[0]['series']['headway']==[dict(second=120,seconds=140)]
    assert rows[1]['series']['headway']==[dict(second=120,seconds=160)]


def test_refresh_uses_saved_service_boundary_before_cohort_end(tmp_path):
    scenario=tmp_path/'scenarios/case'
    write_json(scenario/'forecast.json',dict(rows=[]))
    write_json(scenario/'assumptions.json',dict(service_end_second=120))
    write_json(scenario/'summary.json',dict(comparisons=[dict(candidate_id='KEEP',series={}),
                                                        dict(candidate_id='ROLLING_SELECTED',series={})]))
    for candidate,boundary in [('KEEP',None),('ROLLING_SELECTED',100)]:
        folder=tmp_path/f'runs/case_{candidate}_S1_7'
        header=dict(run_id=folder.name,scenario_id='case',od='S1',seed=7,
                    cohort_scope=dict(start_second=0,end_second=180))
        if boundary is not None:
            header['service_scope']=dict(start_second=0,end_second=boundary)
        write_json(folder/'summary.json',header)
        write_json(folder/'movement_input.json',dict(stations=[dict(station_id='A',kind='line')],
                   tracks=[dict(track_id='P_up',from_station='A',to_station='A')],
                   trains=[dict(train_id='T',state='stop',location='A')]))
        events=[dict(second=s,event_type='departure',location='P_up',train_id='T',next_state='moving') for s in [0,110,160]]
        write_json(folder/'motion.json',dict(period_start='2026-09-01T06:00:00+03:00',events=events))
    refresh_saved_series(tmp_path)
    rows=json.loads((scenario/'summary.json').read_text(encoding='utf-8'))['comparisons']
    assert rows[0]['series']['headway']==[dict(second=60,seconds=110)]
    assert rows[1]['series']['headway']==[]
