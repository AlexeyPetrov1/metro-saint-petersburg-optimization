import copy
import importlib.util
from pathlib import Path
from time import monotonic

import pytest

from metro_daily import prepare_day
from simulation import simulate

# Historical experiment imports prepend snapshots during full-suite collection.
_spec=importlib.util.spec_from_file_location('_workspace_metro44_planner_search',
    Path(__file__).resolve().parents[1]/'metro44_planner_search.py')
planner=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planner)
rebuild_timetable=planner.rebuild_timetable
search_specs=planner.search_specs
effective_peak_headways=planner.effective_peak_headways
evaluate_spec=planner.evaluate_spec
capacity_repair_command=planner.capacity_repair_command
terminal_parking_audit=planner.terminal_parking_audit


def test_planner_subject_is_current_workspace_source():
    source=Path(__file__).resolve().parents[1]/'metro44_planner_search.py'
    assert Path(planner.__file__).resolve()==source
    assert Path(rebuild_timetable.__code__.co_filename).resolve()==source


def test_search_is_a_declared_six_case_physical_probe():
    specs = search_specs()
    assert len(specs) == 6
    assert len({s['candidate_id'] for s in specs}) == 6
    assert {s['occupancy_margin'] for s in specs} == {0, 1, 2}
    assert all(113 <= s['peak_headway'] <= 125 for s in specs)


@pytest.fixture(scope='module')
def day():
    return prepare_day('2026-09-30', service_hours=6)


def test_checked_control_preserves_past_current_motion_and_depot_deadlines(day):
    data, _, _, meta = day
    before = copy.deepcopy(data)
    changed, audit = rebuild_timetable(data, 3600, meta['service_end_second'], headway=150,
        repair_future_platforms=False, allow_future_advance=False)
    old=simulate(data,data['plans'][0]['plan_id']);new=simulate(changed,changed['plans'][0]['plan_id'])
    assert data == before
    assert [e for e in old['events'] if e['second'] < 3600] == [e for e in new['events'] if e['second'] < 3600]
    assert audit['current_motion_preserved']
    assert audit['depot_preparation_passed']
    assert audit['movement_passed'] and new['obligations_closed']
    assert new['checks']['peak_trains_on_line'] <= 53
    assert audit['minimum_headway'] >= 113
    assert audit['maximum_headway'] <= 480


def test_generation_obeys_expired_deadline(day):
    data, _, _, meta = day
    with pytest.raises(TimeoutError):
        rebuild_timetable(data,3600,meta['service_end_second'],deadline=monotonic()-1)


def test_rejected_physical_probe_is_explicit_and_never_returns_an_unchecked_plan(day):
    data, _, _, meta = day
    row, plan = evaluate_spec(data,3600,meta['service_end_second'],search_specs()[0],deadline=monotonic()-1)
    assert row['status'] == 'evaluation_timeout'
    assert row['reason']
    assert plan is None


def test_effective_peak_metrics_use_actual_terminal_departures(day):
    data, _, _, _ = day
    motion=simulate(data,data['plans'][0]['plan_id'])
    summary=effective_peak_headways(data,motion)
    assert summary['scope'] == 'actual terminal departure intervals in weekday peak windows'
    assert len(summary['windows']) == 4
    assert all(r['mean_seconds'] >= 113 for r in summary['windows'] if r['count'])
    assert summary['minimum_seconds'] != 0


def release_trace():
    data = dict(max_trains_on_line=1, headway_policy={'parameters':{'depot_release_s':{'value':{'Northern':780,'Avtovo':720}}}},
        stations=[dict(station_id=s,kind=k) for s,k in [('A','line'),('D','depot')]],
        tracks=[dict(track_id='ENTRY_N',from_station='D',to_station='A'),
                dict(track_id='EXIT_N',from_station='A',to_station='D')],
        trains=[dict(train_id='T',location='A',state='standby'),dict(train_id='R',location='D',state='standby')],
        plans=[dict(trips=[dict(trip_id='entry',train_id='R',route=['ENTRY_N'],departure_second=900,passenger=False),
                           dict(trip_id='service',train_id='R',route=['LINE_up'],departure_second=1200,passenger=True),
                           dict(trip_id='exit',train_id='T',route=['EXIT_N'],departure_second=950,passenger=False)])],
        leg_timing=[dict(train_id='R',trip_id='entry',leg_index=0,departure_second=900),
                    dict(train_id='R',trip_id='service',leg_index=0,departure_second=1200)])
    motion={'events':[dict(second=900,train_id='R',event_type='departure',location='ENTRY_N',next_state='moving',trip_id='entry'),
                      dict(second=950,train_id='T',event_type='departure',location='EXIT_N',next_state='moving',trip_id='exit'),
                      dict(second=1070,train_id='T',event_type='arrival',location='D',next_state='standby',trip_id='exit')]}
    return data,motion


def test_exact_capacity_repair_waits_for_exit_arrival_and_moves_linked_service():
    data,motion=release_trace(); before=copy.deepcopy(data)
    repair=capacity_repair_command(data,motion,400,set())
    assert repair['first_violation_second']==900
    assert repair['entry_not_before_second']==1071
    assert repair['service_not_before_second']==1371
    assert repair['entry_not_before_second']-900 == repair['service_not_before_second']-1200
    assert repair['service_not_before_second']-780 >= 400
    assert data==before


@pytest.mark.parametrize('as_of,retained',[(901,set()),(400,{'entry'})])
def test_capacity_repair_never_moves_a_started_or_retained_entry(as_of,retained):
    data,motion=release_trace()
    with pytest.raises(ValueError,match='committed ENTRY'):
        capacity_repair_command(data,motion,as_of,retained)


def test_capacity_repair_returns_none_when_exact_trace_is_within_limit():
    data,motion=release_trace();data['max_trains_on_line']=2
    assert capacity_repair_command(data,motion,400,set()) is None


def test_terminal_parking_counts_actual_standby_and_half_open_boundary():
    data={'headway_policy':{'demo':{'terminal_parking_capacity_by_station':{'A':1}}},
          'trains':[dict(train_id='T',state='standby',location='A'),dict(train_id='R',state='turnaround',location='A')]}
    motion={'duration_seconds':30,'events':[
        dict(second=10,train_id='R',next_state='standby',location='A'),
        dict(second=10,train_id='T',next_state='moving',location='LINE_up')]}
    result=terminal_parking_audit(data,motion)
    assert result['passed'] and result['maximum_by_terminal']=={'A':1}
    motion['events'][1]['second']=11
    result=terminal_parking_audit(data,motion)
    assert not result['passed']
    assert result['violations'][0]['second']==10


def test_repair_compiler_uses_a_diagnostic_copy_then_checks_original_capacity(day,monkeypatch):
    data,_,_,meta=day;original=copy.deepcopy(data);seen=[]
    compile_original=planner.compile_operational_timing
    simulate_original=planner.simulate
    def compile_spy(candidate,*args,**kwargs):
        seen.append(('compile',candidate.get('max_trains_on_line')))
        return compile_original(candidate,*args,**kwargs)
    def simulate_spy(candidate,*args,**kwargs):
        seen.append(('simulate',candidate.get('max_trains_on_line')))
        return simulate_original(candidate,*args,**kwargs)
    monkeypatch.setattr(planner,'compile_operational_timing',compile_spy)
    monkeypatch.setattr(planner,'simulate',simulate_spy)
    changed,audit=rebuild_timetable(data,3600,meta['service_end_second'],headway=150,
        repair_future_platforms=False,allow_future_advance=False,repair_capacity=True)
    assert ('compile',None)in seen
    assert ('simulate',53)in seen
    assert changed['max_trains_on_line']==53 and audit['movement_passed']
    assert data==original
