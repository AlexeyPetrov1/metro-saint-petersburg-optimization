import pytest

from dwell import compile_timing
from passenger_groups import GroupedPassengerSimulation
from selected_validation import future_checkpoint
from tests.test_dwell import three_stations


def test_future_only_fork_matches_the_full_branch_validator():
    data,_=compile_timing(three_stations(103),'line1-baseline','mild',0,committed=[])
    profiles={t['train_id']:[dict(car_id='C1',car_type='fixture',max_capacity=5,
        normal_threshold=5,source='fixture')] for t in data['trains']}
    groups=[dict(group_id='G',origin='A',turnstile_second=0,platform_second=0,count=8,
        destinations=[dict(station_id='C',count=8)])]
    source=GroupedPassengerSimulation(data,'line1-baseline',groups,profiles)
    state=source.run(end_second=229)['checkpoint']
    source._validate_checkpoint(state,229)
    changed,_=compile_timing(data,'line1-baseline','mild',9,committed=data['leg_timing'],as_of=229)
    target=GroupedPassengerSimulation(changed,'line1-baseline',groups,profiles)
    fast=future_checkpoint(target,source,state)
    assert fast==target.branch_checkpoint(state,source)
    assert target.run(start_second=229,checkpoint=fast)['metrics']['balance']['passed']
    changed['tracks'][0]['length_km']='2'
    wrong=GroupedPassengerSimulation(changed,'line1-baseline',groups,profiles)
    with pytest.raises(ValueError,match='past'):
        future_checkpoint(wrong,source,state)


def test_observed_headways_allow_unused_depot_paths():
    from selected_validation import observed_headways
    from simulation import simulate
    data = three_stations()
    motion = simulate(data, 'line1-baseline')
    expected = observed_headways(data, motion)
    data['tracks'].append(dict(track_id='ENTRY_A'))
    assert observed_headways(data, motion) == expected
