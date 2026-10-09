import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from goal43_policy import load_policy
from metro51_run import atomic_json
from passenger_groups import GroupedPassengerSimulation
from test_line1 import tiny_movement


@pytest.fixture
def partial():
    return importlib.import_module('scripts.metro52_partial')


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'full_campaign'
    physics = tmp_path/'physics.py'
    physics.write_bytes(b'# fixed physics source\n')
    source_sha = hashlib.sha256(physics.read_bytes()).hexdigest()
    data = tiny_movement()
    data['period_end'] = '2026-09-01T07:00:00+03:00'
    data['headway_policy'] = load_policy()
    data['plans'][0]['trips'][1]['departure_second'] = 2400
    groups = [dict(group_id=name, turnstile_second=gate, platform_second=platform,
                   origin='A',count=count,destinations=[dict(station_id='B',count=count,direction=1)])
              for name,gate,platform,count in [('past',0,0,5),('approaching',1790,1810,3),('future',1800,1820,7)]]
    profiles = {t:[dict(car_id='C1',car_type='fixture',max_capacity=150,normal_threshold=120,source='fixture')]
                for t in ['T1','T2']}
    common = dict(data=data,groups=groups,profiles=profiles,roster=[dict(train_id=t,type='fixture') for t in ['T1','T2']])
    runner = GroupedPassengerSimulation(data,'service',groups,profiles,boarding_seed=7,boarding_capacity_per_car=150)
    state = runner.run(end_second=1800)['checkpoint']
    from simulation import digest
    for arm in ['reference','balanced']:
        contract = dict(arm=arm.upper(),stage='full_day',full_day=True,seed=7,control_start_second=900,
            cohort_end_second=3600,platform_cohort_end_second=3600,observation_end_second=3600,service_end_second=3600,
            boarding_capacity_per_car=150,groups_digest=digest(groups),source_hashes={str(physics):source_sha},
            period_start=data['period_start'],period_end=data['period_end'],line_id=1,fixed_fleet_size=2,
            first_miss_limit=.05,second_miss_limit=.01)
        atomic_json(root/arm/'contract.json',contract)
        atomic_json(root/arm/'common_scenario.json.gz',common)
        snapshot = root/arm/'execution_sources'
        snapshot.mkdir()
        (snapshot/'physics.py').write_bytes(physics.read_bytes())
    atomic_json(root/'reference/result.json',dict(contract=json.loads((root/'reference/contract.json').read_text()),
        full_day=True,original_reference_sentinel='must-remain-exact'))
    atomic_json(root/'balanced/controller/resume_01800.json.gz',dict(second=1800,data=data,state=state))
    for second in [900,1800,2700]:
        atomic_json(root/'balanced/controller'/f'decision_{second:05}.json',dict(decision_second=second,
            decision_time='2026-09-01T06:15:00+03:00',applied_candidate_id='KEEP',pipeline_seconds=.1,
            selection={'status':'no_feasible_candidate'}))
        for name in ['raw_forecast','forecast']:
            atomic_json(root/'balanced/controller'/f'{name}_{second:05}.json',dict(rows=[],second=second))
        for name in ['facts','explanation']:
            atomic_json(root/'balanced/explanations'/f'{name}_{second:05}.json',dict(second=second))
    atomic_json(root/'model_verification/verification.json',{'passed':True})
    return root


def test_contract_includes_sixtieth_decision_without_daily_extrapolation(partial):
    original = dict(arm='BALANCED',stage='full_day',full_day=True,control_start_second=900,
                    period_start='2026-09-30T06:00:00+03:00',period_end='2026-10-01T02:00:00+03:00')
    contract = partial.partial_contract(original,60)
    assert original['stage'] == 'full_day'
    assert contract['stage'] == 'partial_60' and contract['full_day'] is False
    assert contract['decision_times'] == list(range(900,54001,900))
    assert contract['control_end_second'] == 54001
    for key in ['cohort_end_second','entry_end_second','platform_cohort_end_second','observation_end_second','service_end_second']:
        assert contract[key] == 54000
    assert contract['annual_repeat_days'] is None
    assert contract['observation_followup_seconds'] == 0
    assert contract['period_end'] == '2026-09-30T21:00:00+03:00'
    assert contract['last_decision_future_execution_observed'] is False
    assert contract['checkpoint_boundary_events'] == 'pending'


def test_paired_materialization_preserves_sources_and_censored_stocks(partial, source, tmp_path):
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    output = tmp_path/'derived'
    receipt = partial.materialize(source,output,steps=2,max_reference_seconds=15)
    assert receipt['stage'] == 'materialized_partial'
    results = {arm:partial.read(output/arm/'result.json') for arm in ['reference','balanced']}
    for arm, result in results.items():
        assert result['full_day'] is False
        assert result['metrics']['train_km'] == 1.0
        assert result['metrics']['resources_scope']['end_second'] == 1800
        assert result['metrics']['approaching_at_closure'] == 3
        assert result['metrics']['complete_passenger_accounting'] is False
        assert result['metrics']['passenger_balance_passed'] is True
        assert result['metrics']['cohort']['complete'] is False
        assert result['metrics']['cohort']['mean_wait_seconds'] is None
        assert result['cash_saving_rub'] is None
        run = output/arm/result['run']['folder']
        groups = partial.read(run/'groups.json.gz')
        assert [g['group_id'] for g in groups] == ['past','approaching']
        assert result['run']['resources_scope']['end_second'] == 1800
        assert float(result['run']['metrics']['resources']['train_km']) == 1.0
        summary = partial.read(run/'summary.json')
        assert summary['resources_scope']['end_second'] == 1800
        assert float(summary['metrics']['resources']['train_km']) == 1.0
        assert summary['full_day'] is False
    assert results['reference']['contract']['groups_digest'] == results['balanced']['contract']['groups_digest']
    assert [d['decision_second'] for d in results['balanced']['decisions']] == [900,1800]
    assert results['reference']['decisions'] == []
    assert not (output/'balanced/controller/decision_02700.json').exists()
    assert not (output/'balanced/explanations/facts_02700.json').exists()
    for p, sha in before.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
    assert not list(output.rglob('*.tmp'))


def test_future_group_removal_uses_a_valid_audited_branch(partial, source):
    common = partial.read(source/'balanced/common_scenario.json.gz')
    saved = partial.read(source/'balanced/controller/resume_01800.json.gz')
    original = copy.deepcopy(saved['state'])
    runner, state, groups = partial.branch_observed(common,saved,seed=7,capacity=150)
    runner._validate_checkpoint(state,1800)
    assert groups == [g for g in common['groups'] if g['turnstile_second']<1800]
    assert saved['state'] == original
    for key in original:
        if key not in ['context_id','sha256','next_event_index']:
            assert state[key] == original[key]


@pytest.mark.parametrize('failure',['missing_step','missing_facts','corrupt_checkpoint','different_seed','source_changed','existing_output'])
def test_invalid_partial_is_rejected_before_results_are_written(partial, source, tmp_path, failure):
    output = tmp_path/'derived'
    if failure == 'missing_step':
        (source/'balanced/controller/decision_01800.json').unlink()
    elif failure == 'missing_facts':
        (source/'balanced/explanations/facts_01800.json').unlink()
    elif failure == 'corrupt_checkpoint':
        path = source/'balanced/controller/resume_01800.json.gz'
        saved = partial.read(path);saved['state']['queued'] += 1;atomic_json(path,saved)
    elif failure == 'different_seed':
        path = source/'reference/contract.json'
        contract = partial.read(path);contract['seed'] = 8;atomic_json(path,contract)
    elif failure == 'source_changed':
        (tmp_path/'physics.py').write_bytes(b'# source drift\n')
    else:
        output.mkdir();(output/'preserve.txt').write_bytes(b'do-not-overwrite')
    with pytest.raises((ValueError,FileNotFoundError,FileExistsError)):
        partial.materialize(source,output,steps=2,max_reference_seconds=15)
    assert not list(output.rglob('result.json'))


def test_partial_cannot_be_written_inside_original_campaign(partial, source):
    with pytest.raises(ValueError):
        partial.materialize(source,source/'derived',steps=2)
