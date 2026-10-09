"""Contract tests for the pending optimizer patch; production stays frozen."""
from pathlib import Path
from types import ModuleType
import copy
import pytest

ROOT = Path(__file__).resolve().parents[1]


def patched_optimizer():
    source_text=(ROOT/'metro51_optimizer.py').read_text(encoding='utf8')
    if 'def future_trip_controls(' in source_text:
        module=ModuleType('metro51_optimizer_v3_applied');module.__file__=str(ROOT/'metro51_optimizer.py')
        exec(compile(source_text,module.__file__,'exec'),module.__dict__)
        return module
    source = source_text.splitlines(keepends=True)
    patch = (ROOT/'scripts/metro51_optimizer_v3.patch').read_text(encoding='utf8').splitlines(keepends=True)
    output, cursor, index = [], 0, 0
    import re
    while index < len(patch):
        if not patch[index].startswith('@@ '):
            index += 1
            continue
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',patch[index])
        assert match
        start = int(match[1])-1
        output.extend(source[cursor:start]);cursor=start;index+=1
        while index<len(patch) and not patch[index].startswith('@@ '):
            line=patch[index]
            if line.startswith((' ','-')):
                assert source[cursor].rstrip('\r\n')==line[1:].rstrip('\r\n'), 'patch base drift'
                cursor+=1
            if line.startswith((' ','+')):output.append(line[1:])
            index+=1
    output.extend(source[cursor:])
    module=ModuleType('metro51_optimizer_v3_pending');module.__file__=str(ROOT/'metro51_optimizer.py')
    exec(compile(''.join(output),module.__file__,'exec'),module.__dict__)
    return module


@pytest.fixture(scope='module')
def optimizer():return patched_optimizer()


def departures(step=150):
    return [dict(track_id='t_up',direction='up',section_index=0,second=i*step,
                 trip_id=f'j{i}',minimum_headway_seconds=150,terminal_offset_seconds=0)
            for i in range(7)]


def solve(optimizer, rows=None, rates=(), **kw):
    return optimizer.solve_joint_clocks([],rows or departures(),{},list(rates),as_of=1,
        end_second=900,shift_bounds={'j1':(-20,20)},time_limit_seconds=2,**kw)


def test_own_section_prevents_a_global_max_false_overload(optimizer):
    rates=[dict(start_second=0,end_second=900,up_rate=20,down_rate=0,
                up_section_rates=[2,20],down_section_rates=[0,0])]
    result=solve(optimizer,rates=rates)
    assert result['proxy_overload_arc_count']==0
    assert result['section_rate_arc_count']>0
    assert result['aggregate_rate_fallback_arc_count']==0


def test_missing_section_rates_is_an_explicit_legacy_fallback(optimizer):
    result=solve(optimizer,rates=[dict(start_second=0,end_second=900,up_rate=2,down_rate=0)])
    assert result['aggregate_rate_fallback_arc_count']>0
    assert result['section_rate_arc_count']==0


def test_overloaded_proxy_stays_solvable_and_is_never_passenger_certification(optimizer):
    result=solve(optimizer,rates=[dict(start_second=0,end_second=900,up_rate=10,down_rate=0)])
    assert result['status'] in ('optimal','search_incomplete')
    assert result['proxy_overload_arc_count']>0
    assert result['proxy_objective_value']>0
    assert result['passenger_admissible'] is False
    assert result['operational_global_optimum_proven'] is False
    assert result['release_authorized'] is False


def test_repair_prioritizes_overload_over_profitable_removal(optimizer):
    rows=departures();options=[dict(train_id='train',cancel_service_ids=['j2'])]
    result=optimizer.solve_joint_clocks(options,rows,{'train':1_000_000},
        [dict(start_second=0,end_second=900,up_rate=5,down_rate=0)],as_of=1,
        end_second=900,shift_bounds={'j1':(-20,20)},time_limit_seconds=2,mode='repair')
    assert result['selected_train_ids']==[]
    assert result['proxy_overload_arc_count']==0


def test_relative_continuity_allows_joint_shifts_without_advancing_before_ready(optimizer):
    result=optimizer.solve_joint_clocks([],departures(),{},[],as_of=1,end_second=900,
        shift_bounds={'j1':(-120,120),'j2':(-120,120)},
        continuities=[('j1','j2',0)],time_limit_seconds=2)
    assert result['trip_shifts']['j2']>=result['trip_shifts']['j1']


def test_future_entry_shift_is_supported_without_a_passenger_track_clock(optimizer):
    controls={'entry':dict(kind='ENTRY',first_second=700,earliest_second=500)}
    result=optimizer.solve_joint_clocks([],departures(),{},[],as_of=1,end_second=900,
        shift_bounds={'entry':(-200,600),'j1':(-20,20)},trip_controls=controls,time_limit_seconds=2)
    assert 'entry' in result['trip_shifts']
    assert result['entry_shift_variable_count']==1


def test_started_entry_and_entry_before_release_lead_are_rejected(optimizer):
    for control,bounds in ((dict(kind='ENTRY',first_second=0,earliest_second=0),(-1,1)),
                           (dict(kind='ENTRY',first_second=700,earliest_second=600),(-200,600))):
        with pytest.raises(ValueError):
            optimizer.solve_joint_clocks([],departures(),{},[],as_of=1,end_second=900,
                shift_bounds={'entry':bounds},trip_controls={'entry':control},time_limit_seconds=1)


def test_service_cannot_use_entry_sized_shift_bounds(optimizer):
    with pytest.raises(ValueError):
        optimizer.solve_joint_clocks([],departures(),{},[],as_of=1,end_second=900,
            shift_bounds={'j1':(-150,600)},time_limit_seconds=1)


def test_invalid_section_index_is_not_silently_replaced_by_aggregate(optimizer):
    rows=departures();rows[1]['section_index']=4
    with pytest.raises(ValueError):
        solve(optimizer,rows=rows,rates=[dict(start_second=0,end_second=900,
            up_rate=2,down_rate=0,up_section_rates=[2],down_section_rates=[0])])


def test_balanced_local_and_wait_gates_remain_unchanged(optimizer):
    from tests.test_metro51_optimizer import marked
    rows=[marked(),marked('CHEAP',9,91,150),marked('LOCAL',8,local=False)]
    original=copy.deepcopy(rows)
    result=optimizer.choose_balanced(rows)
    assert result['selected_candidate_id']=='KEEP'
    assert rows==original


def test_builder_preserves_future_chain_freedom_and_entry_leads(optimizer, monkeypatch):
    import gzip,json
    path=ROOT/'data/processed/metro51_balanced_pilot_20261009_v2/balanced/common_scenario.json.gz'
    fixture=json.load(gzip.open(path,'rt',encoding='utf8'))['data']
    assert hasattr(optimizer,'future_trip_controls')
    controls=optimizer.future_trip_controls(fixture,3600,18000,locks=())
    assert all(v['first_second']>=3600 for v in controls['trip_controls'].values())
    assert all(-120<=lo<=0<=hi<=120 for name,(lo,hi) in controls['shift_bounds'].items()
               if controls['trip_controls'][name]['kind']=='SERVICE')
    assert controls['shift_bounds']['G48FIFO:B016:112'][1]>0
    entries=[v for v in controls['trip_controls'].values() if v['kind']=='ENTRY']
    assert entries and {v['release_seconds'] for v in entries}<={720,780}
    assert all(v['earliest_second']>=3600 for v in entries)
    assert all('first_service_trip_id' in v for v in entries)
    assert all(controls['trip_controls'][v['first_service_trip_id']]['first_second']+
               controls['shift_bounds'][v['first_service_trip_id']][0]>=3600+v['release_seconds']
               for v in entries)
