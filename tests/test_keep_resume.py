import json
import gzip

import pytest

import metro_replay
from actions import demo_depots
from decision_engine import build_regular_plan
from dwell import compile_timing
from passenger_groups import GroupedPassengerSimulation


@pytest.mark.parametrize('profile',['fixed','mild'])
def test_keep_continues_saved_stream_without_recompiling_future(tmp_path, monkeypatch, profile):
    config=dict(station_ids=['A','B','C'],date='2026-09-01',timezone='Europe/Moscow',length_m=2000,
        travel_seconds=120,stop_seconds=20,turnaround_seconds=10,train_type='fixture',car_types=['fixture'])
    data,_=build_regular_plan(config,2,end_second=21600)
    data,registry=demo_depots(data,config)
    data,timing=compile_timing(data,'line1-baseline',profile,7,committed=[])
    profiles={t['train_id']:[dict(car_id='C1',car_type='fixture',max_capacity=5,
        normal_threshold=5,source='fixture')] for t in data['trains']}
    groups=[dict(group_id='G',origin='A',turnstile_second=8000,platform_second=8120,count=4,
        destinations=[dict(station_id='B',count=4)])]
    source=GroupedPassengerSimulation(data,'line1-baseline',groups,profiles)
    state=source.run(end_second=7200)['checkpoint']
    initial=(data,registry,profiles,{'S1':source},{'S1':state},timing)
    def unnecessary(*args,**kwargs): raise AssertionError('KEEP must reuse the committed plan')
    monkeypatch.setattr(metro_replay,'compile_timing',unnecessary)
    metro_replay.replay_mode(tmp_path,config,{'S1':groups},{},{},'A_control',profile,None,7,initial)
    saved=json.loads(gzip.decompress((tmp_path/'episodes/A_control/S1/final_checkpoint.json.gz').read_bytes()))
    assert saved==source.run()['checkpoint']
