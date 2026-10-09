import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from economics import write_json
from goal43_extend import extend_fixture, resume_stop
from goal43_policy import load_policy
from goal43_replay import (actual_groups, build_operational_day, fixture_history,
                           live_decision, save_run)
from passenger_groups import GroupedPassengerSimulation


def saved_fixture(output, incident=False):
    data, roster, ready, profiles, meta = build_operational_day(load_policy(), count=3,
        service_hours=2, small_line=['L1_01', 'L1_07', 'L1_19'])
    meta['service_end_second'] = 3600  # One demand hour, followed by service to clear its passengers.
    policy = data['headway_policy']
    frame = fixture_history(['L1_01', 'L1_07', 'L1_19'])
    groups = actual_groups(frame, data, meta['service_end_second'])
    sources = {od: GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], rows, profiles)
               for od, rows in groups.items()}
    stop = None
    second = 900
    if incident:
        event=next(row for row in sources['S1'].motion['events'] if row['second'] >= 900
                   and row['event_type']=='arrival' and row['location']=='L1_07')
        second=event['second']
        index=sources['S1'].motion['events'].index(event)
        incoming=next(row for row in reversed(sources['S1'].motion['events'][:index])
                      if row['train_id']==event['train_id'] and row['event_type']=='departure')
        direction='up' if incoming['location'].endswith('_up') else 'down'
        stop=dict(incident_id='tiny',train_id=event['train_id'],station_id='L1_07',direction=direction,
                  detected_second=second,expected_end_second=second+60)
    snapshots = {od: runner.run(end_second=second)['checkpoint'] for od, runner in sources.items()}
    live = live_decision(data, sources, snapshots, profiles, ready, frame, policy,
        scenario_id='tiny', minimum_remaining=2,incidents=[stop] if stop else [])
    scenario = output/'scenarios/tiny'
    write_json(scenario/'summary.json', dict(scenario_id='tiny', evaluation_mode='fixture',
        incident=None, full_day=False, decisions=[live['decision']], comparisons=[], elapsed_seconds=0))
    for name, value in [('decision',live['decision']), ('forecast',live['bundle']),
                        ('notifications',live['ledger']), ('assumptions',meta), ('fleet_registry',roster)]:
        write_json(scenario/f'{name}.json',value)
    write_json(scenario/'economics.json', [])
    for candidate in live['candidates']:
        if candidate.get('_data') is None:
            continue
        suffix=candidate['candidate_id'].replace(':','_').replace('+','-')
        write_json(output/f"runs/tiny_{suffix}_S1_7/movement_input.json",candidate['_data'])
    for od, runner in sources.items():
        if stop:
            baseline=next(row['_data'] for row in live['candidates'] if row['candidate_id']=='KEEP')
            runner=GroupedPassengerSimulation(baseline,baseline['plans'][0]['plan_id'],groups[od],profiles)
        else:
            baseline=data
        full=runner.run()['checkpoint']
        save_run(output, f'tiny_KEEP_{od}_7', baseline, runner, full, roster, policy,
            'tiny','KEEP',od,7,full_day=False,cohort_start=second,cohort_end=second+7200,
            service_end_second=meta['service_end_second'])
    return live


def hashes(folder):
    return {str(path.relative_to(folder)):sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob('*') if path.is_file()}


def test_saved_first_decision_extends_without_rewriting_existing_runs(tmp_path):
    initial=saved_fixture(tmp_path)
    before=hashes(tmp_path/'runs')
    first=copy.deepcopy(initial['decision'])
    result=extend_fixture(tmp_path, scenario_id='tiny', minimum_remaining=2)
    assert len(result['records'])==3
    assert {row['strategy'] for row in result['records']}=={'ROLLING_SELECTED'}
    assert len(result['summary']['decisions'])==2
    assert [row['decision_second'] for row in result['summary']['decisions']]==[900,1800]
    assert result['summary']['decisions'][0]['elapsed_seconds']==first['elapsed_seconds']
    after=hashes(tmp_path/'runs')
    assert all(after[name]==value for name,value in before.items())
    assert all(row['entered']==row['completed'] and row['queued']==0 for row in result['records'])
    assert all(row['service_scope']['end_second']==3600 and row['headways']['passed']
               for row in result['records'])
    assert result['past_verified'] and result['saved_first_decision_reused']
    assert not (tmp_path/'runs.json').exists()
    scenario=tmp_path/'scenarios/tiny'
    assert (scenario/'controller_trace.json').is_file()
    assert len(json.loads((scenario/'decisions.json').read_text(encoding='utf-8')))==2
    assert result['summary']['comparisons'][-1]['per_train']
    with pytest.raises(ValueError,match='already'):
        extend_fixture(tmp_path, scenario_id='tiny', minimum_remaining=2)


def test_extension_rejects_other_source_scope_before_calculation(tmp_path):
    write_json(tmp_path/'scenarios/wrong/summary.json',dict(evaluation_mode='historical_entries_assumed_timetable'))
    with pytest.raises(ValueError,match='fixture'):
        extend_fixture(tmp_path,scenario_id='wrong')


def test_extension_rejects_changed_authoritative_past(tmp_path):
    saved_fixture(tmp_path)
    path=tmp_path/'runs/tiny_KEEP_S1_7/motion.json'
    motion=json.loads(path.read_text(encoding='utf-8'))
    motion['events'][0]['second'] += 1
    write_json(path,motion)
    with pytest.raises(ValueError,match='authoritative saved past'):
        extend_fixture(tmp_path,scenario_id='tiny',minimum_remaining=2)
    assert not list((tmp_path/'runs').glob('*ROLLING_SELECTED*'))


def test_stop_resume_creates_missing_summary_from_saved_first_decision(tmp_path):
    initial=saved_fixture(tmp_path,incident=True)
    (tmp_path/'scenarios/tiny/summary.json').unlink()
    before=hashes(tmp_path/'runs')
    frame=fixture_history(['L1_01','L1_07','L1_19'])
    result=resume_stop(tmp_path,'tiny',steps=4,minimum_remaining=2,frame=frame)
    assert len(result['records'])==3 and len(result['summary']['decisions'])==4
    assert result['summary']['decisions'][0]['elapsed_seconds']==initial['decision']['elapsed_seconds']
    assert result['summary']['injected_actual_end_second']==initial['decision']['incident_estimates'][0]['expected_end_second']
    assert all(row['recovery_observed'] for row in result['summary']['decisions'][1:])
    after=hashes(tmp_path/'runs')
    assert all(after[name]==value for name,value in before.items())
    assert {row['candidate_id'] for row in result['summary']['comparisons']} >= {'KEEP','ROLLING_SELECTED'}
    assert result['summary']['evaluation_mode']=='historical_entries_injected_incident_assumed_timetable'
    assert all(row['source']=='historical_entries_injected_incident_assumed_timetable'
               for row in result['summary']['comparisons'])
