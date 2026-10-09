import csv
import json
from pathlib import Path

import pytest

from metro51_report import (collect_explanations, collect_incident_probes,
                           collect_post_execution_forecasts, comfort_metrics, export_pdf, inventory,
                           make_comparison, render_dispatcher, render_reports,
                           summarize_attribution_facts, verify_model_evidence, write_status_report)


def results():
    fleet = [dict(train_id=f'{prefix}{i:03}', train_type=kind, km=100)
             for prefix, kind, count in [('Y', 'Yubileyny', 31), ('B', 'Baltiets', 37)]
             for i in range(1, count+1)]
    baseline = dict(contract={'full_day': True, 'stage': 'full_day', 'line_id': 1},
                    metrics={'per_train': fleet, 'train_km': 6800,
                             'mean_wait_seconds': 70, 'p95_wait_seconds': 140},
                    decisions=[], elapsed_seconds=1)
    changed = json.loads(json.dumps(baseline))
    changed['metrics']['per_train'] = [dict(row, km=95) for row in fleet]
    changed['metrics']['train_km'] = 6460
    return {'reference': baseline, 'balanced': changed}


def audits(*, admitted=True):
    return {name: dict(passed=admitted if name == 'balanced' else True,
                       baseline_valid_for_comparison=True,
                       checks={'independent_physical_gate': True,
                               'local_boarding_service': admitted if name == 'balanced' else True},
                       failed_checks=[] if admitted or name == 'reference' else ['local_boarding_service'],
                       metrics=value['metrics'], local_service={'cohorts': [], 'status': 'passed' if admitted else 'failed'},
                       waiting_budget={'windows': [], 'passed': admitted},
                       controller={'count': 0, 'maximum_seconds': None},
                       physical={'movement_passed': True})
            for name, value in results().items()}


def test_failed_local_gate_keeps_diagnostic_delta_but_never_claims_savings():
    data = make_comparison(results(), audits(admitted=False), paired_inputs_passed=True)
    economy = data['economics']['balanced']
    assert economy['daily']['without_sr_kr']['saving_rub'] > 0
    assert economy['accepted_resource_saving'] is False
    assert economy['meets_one_percent_target'] is False
    assert data['accepted_saving']['balanced']['without_sr_kr'] is None
    assert data['goal_achieved'] is False


def test_pairing_failure_rejects_nominal_positive_economy():
    data = make_comparison(results(), audits(), paired_inputs_passed=False)
    assert data['economics']['balanced']['admissible'] is False
    assert data['accepted_saving']['balanced']['with_sr_kr'] is None


def test_unsafe_baseline_cannot_anchor_cost_saving_claim():
    audit = audits()
    audit['reference']['baseline_valid_for_comparison'] = False
    data = make_comparison(results(), audit, paired_inputs_passed=True)
    assert data['economics']['balanced']['admissible'] is False


def test_reference_local_failure_does_not_hide_candidate_improvement():
    audit = audits()
    audit['reference']['passed'] = False
    audit['reference']['checks']['local_boarding_service'] = False
    data = make_comparison(results(), audit, paired_inputs_passed=True)
    assert data['economics']['balanced']['admissible'] is True
    assert data['economics']['balanced']['meets_one_percent_target'] is True
    assert data['all_results_admissible'] is False


def test_economy_annual_fleet_per_train_and_uncertainty_scope():
    data = make_comparison(results(), audits(), paired_inputs_passed=True, annual_days=250)
    econ = data['economics']['balanced']
    assert len(econ['per_train']) == 68
    for side in ('without_sr_kr', 'with_sr_kr'):
        assert econ['annual'][side]['saving_rub'] == pytest.approx(econ['daily'][side]['saving_rub'] * 250)
        assert sum(row['daily'][side]['saving_rub'] for row in econ['per_train']) == pytest.approx(econ['daily'][side]['saving_rub'])
        assert econ['daily'][side]['mean_saving_per_fixed_train_rub'] == pytest.approx(econ['daily'][side]['saving_rub']/68)
        bounds = econ['daily'][side]['uncertainty']
        assert bounds['saving_rub_min'] < econ['daily'][side]['saving_rub'] < bounds['saving_rub_max']
    assert econ['uncertainty']['computed_mileage_relative_error'] == .05
    assert econ['uncertainty']['sr_kr_price_relative_error'] == .2
    assert econ['uncertainty']['maintenance_intervals_changed'] is False
    assert econ['annual']['scenario_only'] is True
    assert econ['cash_saving_rub'] is None


def test_partial_missing_result_is_rejected():
    with pytest.raises(ValueError, match='same arms'):
        make_comparison(results(), {'reference': audits()['reference']}, paired_inputs_passed=True)


def test_pilot_uses_observed_mileage_and_never_extrapolates_day_or_year():
    source = results()
    for value in source.values():
        value['contract'].update(full_day=False, stage='pilot', observation_end_second=23400)
        value['metrics']['actual_per_train'] = value['metrics']['per_train']
        value['metrics']['per_train'] = [dict(row, km=row['km']*4) for row in value['metrics']['per_train']]
    data = make_comparison(source, audits(), paired_inputs_passed=True)
    econ = data['economics']['balanced']
    assert econ['observed_window']['candidate_km'] == 6460
    assert 'daily' not in econ and 'annual' not in econ
    assert all('annual' not in row for row in econ['per_train'])
    assert data['annual_scenario_emitted'] is False
    assert data['goal_achieved'] is False


def test_pilot_rejects_whole_planned_mileage_when_observed_ledger_missing():
    source = results()
    for value in source.values():
        value['contract'].update(full_day=False, stage='pilot')
    with pytest.raises(ValueError, match='observed mileage'):
        make_comparison(source, audits(), paired_inputs_passed=True)


def test_inventory_hashes_sources_and_excludes_secrets_and_report_output(tmp_path):
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run/result.json').write_text('{}', encoding='utf-8')
    (tmp_path / '.env').write_text('API_KEY=private', encoding='utf-8')
    (tmp_path / '.env.example').write_text('API_KEY=private', encoding='utf-8')
    output = tmp_path / 'review'
    output.mkdir()
    (output / 'old_report.json').write_text('{}', encoding='utf-8')
    listing = inventory(tmp_path, excluded=output)
    assert [r['path'] for r in listing] == ['run/result.json']
    assert len(listing[0]['sha256']) == 64 and listing[0]['bytes'] == 2
    assert 'private' not in json.dumps(listing)


def test_reports_have_paired_columns_diagnostic_warning_and_every_train(tmp_path):
    data = make_comparison(results(), audits(admitted=False), paired_inputs_passed=True)
    manifest = render_reports(data, tmp_path, create_pdf=False)
    md = (tmp_path / 'report.md').read_text(encoding='utf-8')
    doc = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert 'Без СР и КР' in md and 'С СР и КР' in md
    assert 'не считается допустимой экономией' in md
    assert '±5%' in md and '±20%' in md
    assert 'Сценарная оценка' in md
    assert 'Y031' in doc and 'B037' in doc
    assert manifest['pdf_created'] is False
    with (tmp_path / 'per_train.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 68 * 2 * 2 * 2
    assert {row['period'] for row in rows} == {'daily', 'annual'}
    assert all(row['accepted_saving_rub'] == '' for row in rows if row['arm'] == 'balanced')


def test_pdf_renderer_uses_absolute_profile_and_records_sanitized_diagnostics(tmp_path, monkeypatch):
    import metro51_report
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    Path('review').mkdir()
    Path('review/report.html').write_text('<html></html>', encoding='utf-8')
    captured = []
    monkeypatch.setattr(metro51_report.shutil, 'which', lambda name: 'browser')
    monkeypatch.setattr(metro51_report.subprocess, 'run',
                        lambda args, **kwargs: captured.append(args) or SimpleNamespace(returncode=1, stderr=b'render unavailable'))
    with pytest.raises(RuntimeError, match='PDF export failed'):
        export_pdf(Path('review/report.html'), Path('review/report.pdf'))
    profile = next(a.split('=', 1)[1] for a in captured[0] if a.startswith('--user-data-dir='))
    assert Path(profile).is_absolute()
    assert Path('review/pdf_renderer.log').read_text() == 'render unavailable'


def test_explanation_collection_validates_facts_and_retains_only_safe_text(tmp_path):
    from metro51_explanations import build_fact_packet, explain_decision

    folder = tmp_path / 'balanced' / 'explanations'
    folder.mkdir(parents=True)
    p = build_fact_packet(run_id='balanced', decision_id='3600', action={'type': 'KEEP'}, facts=[])
    e = explain_decision(p, use_api=False)
    (folder / 'facts_03600.json').write_text(json.dumps(p), encoding='utf-8')
    (folder / 'explanation_03600.json').write_text(json.dumps(e), encoding='utf-8')
    summary, rows = collect_explanations(tmp_path)
    assert summary['verified_count'] == 1 and summary['template_count'] == 1
    assert summary['invalid_count'] == 0
    assert rows[0]['decision_id'] == '3600'
    e['sections']['action'][0]['text'] += ' private made-up narrative'
    (folder / 'explanation_03600.json').write_text(json.dumps(e), encoding='utf-8')
    summary, rows = collect_explanations(tmp_path)
    assert summary['invalid_count'] == 1 and summary['verified_count'] == 0
    assert 'private made-up narrative' not in json.dumps(rows)


def test_blind_model_verified_flags_cannot_complete_goal():
    source = results()
    source['balanced']['ensemble_contract_verified'] = True
    plain = make_comparison(source, audits(), paired_inputs_passed=True)
    assert plain['goal_achieved'] is False
    fake = make_comparison(source, audits(), paired_inputs_passed=True,
                           model_evidence={'passed': True})
    assert fake['goal_achieved'] is False


def model_evidence_inputs(tmp_path):
    import hashlib
    from metro51_report import REQUIRED_MODEL_TESTS

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    model = tmp_path / 'input'
    weights = {'b1': {}, 'b3': {}}
    files = {}
    for group, folder, names in [('b1', 'lgbm_holdout', ['meta.json', 'q10.txt', 'q50.txt', 'q90.txt', 'station_quantiles.csv']),
                                ('b3', 'stack_gru_2026-09', ['meta.json', 'gru_seed0.pt', 'gru_seed1.pt', 'gru_seed2.pt'])]:
        for name in names:
            path = model / 'models' / folder / name
            weights[group][name] = write(path, b'test artifact')
            files[path.relative_to(model).as_posix()] = weights[group][name]
    for key, name in [('b2', 'b2_quantiles_september.json'), ('stack_params', 'stack_params.json')]:
        weights[key] = write(model / name, b'{}')
    identity = {'data_sha256': {}, 'GRU_sha256': {}}
    for name in ['spb_line1_15min.parquet', 'spb_line1_hourly.parquet', 'spb_line1_15min_reconciliation.parquet']:
        digest = write(model / 'data' / 'interim' / name, b'same data')
        files['data/interim/'+name] = digest
        identity['data_sha256'][name] = {'new': digest, 'old': digest, 'identical': True}
    for name, digest in weights['b3'].items():
        identity['GRU_sha256'][name] = {'new': digest, 'old': digest, 'identical': True}
    contract = {'components': ['b1_lightgbm', 'b2_persistence', 'b3_gru_seeds012', 'frozen_meta_stack'],
                'weights_sha256': weights, 'b1_train_end': '2026-08-24', 'b3_train_end': '2026-07-31',
                'meta_fit_months': [5, 7], 'b2_calibration_months': [2, 5, 7],
                'native_update_seconds': 1800, 'simulator_refresh_seconds': 900,
                'future_truth_used': False, 'retrospective_history_flags': False,
                'feature_order': [str(i) for i in range(26)],
                'vestibule_order': [str(i) for i in range(23)]}
    for name, value in [('contract.json', contract), ('source_identity_audit.json', identity),
                        ('input_manifest.json', {'files': files}),
                        ('model_audit.json', {'b1_native_slot_max_absolute_error': 1e-15, 'native_rows': 92,
                                              'limited_forecast_comparison': [{'model': 'full_stack', 'n': 152, 'wape': .08}]})]:
        write(model / name, json.dumps(value).encode())
    proof = tmp_path / 'model_verification'
    xml = '<testsuite>'+''.join(f'<testcase classname="tests.test_metro51_ensemble" name="{name}"/>' for name in REQUIRED_MODEL_TESTS)+'</testsuite>'
    xml_hash = write(proof/'junit.xml', xml.encode())
    test_source, ensemble_source = tmp_path/'test_metro51_ensemble.py', tmp_path/'metro51_ensemble.py'
    test_hash, ensemble_hash = write(test_source, b'test code'), write(ensemble_source, b'model code')
    evidence = {'exit_code': 0, 'junit_xml': str(proof/'junit.xml'), 'junit_sha256': xml_hash,
                'command': ['python', '-m', 'pytest', 'tests/test_metro51_ensemble.py'],
                'source_sha256': {str(test_source): test_hash, str(ensemble_source): ensemble_hash},
                'input_manifest_sha256': hashlib.sha256((model/'input_manifest.json').read_bytes()).hexdigest()}
    write(proof/'evidence.json', json.dumps(evidence).encode())
    return model, proof/'evidence.json'


def test_model_verification_requires_live_hashes_parity_identity_and_passed_actual_tests(tmp_path):
    inputs, proof = model_evidence_inputs(tmp_path)
    verified = verify_model_evidence(inputs, test_evidence=proof)
    assert verified['passed'] is True
    (inputs/'models/lgbm_holdout/q50.txt').write_text('modified', encoding='utf-8')
    changed = verify_model_evidence(inputs, test_evidence=proof)
    assert changed['passed'] is False
    assert 'model_weight_hashes' in changed['failed_checks']


def test_missing_skipped_or_failed_model_tests_are_not_verification(tmp_path):
    inputs, proof = model_evidence_inputs(tmp_path)
    assert verify_model_evidence(inputs)['passed'] is False
    data = json.loads(proof.read_text())
    xml = Path(data['junit_xml'])
    xml.write_text('<testsuite><testcase name="test_actual_contract_uses_all_native_bases_and_true_cutoffs"><skipped/></testcase></testsuite>', encoding='utf-8')
    import hashlib
    data['junit_sha256'] = hashlib.sha256(xml.read_bytes()).hexdigest()
    proof.write_text(json.dumps(data), encoding='utf-8')
    failed = verify_model_evidence(inputs, test_evidence=proof)
    assert failed['passed'] is False
    assert 'actual_model_tests' in failed['failed_checks']


def test_comfort_exposure_time_is_not_departure_frequency_weighted():
    from types import SimpleNamespace
    runner = SimpleNamespace(motion={'period_start': '2026-09-30T08:00:00+03:00'},
        motion_by_id={'d1': {'passenger_service': True}, 'd2': {'passenger_service': True}})
    state = {'second': 200, 'load_events': [
        {'second': 0, 'train_id': 'T', 'car_id': 'C', 'load': 130, 'event_type': 'departure', 'motion_event_id': 'd1'},
        {'second': 100, 'train_id': 'T', 'car_id': 'C', 'load': 145, 'event_type': 'departure', 'motion_event_id': 'd2'}]}
    m = comfort_metrics(runner, state, end_second=200)
    assert m['whole_window']['observed_car_seconds'] == 200
    assert m['whole_window']['above_120_car_seconds'] == 200
    assert m['whole_window']['above_140_car_seconds'] == 100
    assert m['whole_window']['above_140_time_share'] == .5
    assert m['peak']['p95'] == 145 and m['peak']['p99'] == 145


def test_dispatcher_escapes_evidence_and_marks_post_execution_and_failed_full_day(tmp_path):
    comparison = make_comparison(results(), audits(admitted=False), paired_inputs_passed=True)
    comparison.update(full_day=False, stage='pilot', campaign_status={'full_day_status': 'not_run_due_to_failed_pilot'},
        explanations=[{'arm': 'balanced', 'decision_id': '3600', 'source': 'template', 'model': None,
            'sections': {'risks': [{'text': '<script>alert(1)</script>', 'evidence_ids': ['risk'], 'source': 'packet'}]}}],
        post_execution_forecasts=[{'arm': 'balanced', 'origin': '2026-09-30T07:00:00+03:00',
                                  'actual_sum': 10, 'predicted_sum': 8, 'wape': .2}])
    output = render_dispatcher(comparison, tmp_path)
    doc = output.read_text(encoding='utf-8')
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in doc
    assert '<script>alert(1)</script>' not in doc
    assert 'Полный суточный прогон не выполнен' in doc
    assert 'После исполнения' in doc
    assert '20.00' in doc


def test_dispatcher_never_dumps_full_attribution_json_in_any_section(tmp_path):
    comparison = make_comparison(results(), audits(), paired_inputs_passed=True)
    comparison['explanations'] = [dict(arm='balanced', decision_id='3600', source='yandexgpt', model='m',
        sections={'rationale': [dict(kind='attribution', text='FULL_RAW_JSON_raw_q50_z', evidence_ids=['a'])]},
        attribution_groups=[dict(station_id='A', target_hour='08:00', horizon_hours=1,
            features=[dict(vestibule_id='A1', name='level', value=5, contribution_z=.2)])])]
    doc = render_dispatcher(comparison, tmp_path).read_text(encoding='utf-8')
    assert 'FULL_RAW_JSON' not in doc and 'raw_q50_z' not in doc
    assert 'level' in doc and '+0.200000' in doc


def test_dispatcher_formats_verified_keep_action_as_dispatcher_text(tmp_path):
    comparison = make_comparison(results(), audits(), paired_inputs_passed=True)
    comparison['explanations'] = [dict(arm='balanced', decision_id='3600', source='template', model='m',
        action=dict(candidate_id='KEEP', decision_time='2026-09-30T07:00:00+03:00',
                    withdraw_train_ids=[], trip_shifts={}, forecast_admissible=False),
        sections={'action': [dict(kind='action', text='candidate_id=KEEP internal JSON', evidence_ids=['decision.action'])]})]
    doc = render_dispatcher(comparison, tmp_path).read_text(encoding='utf-8')
    assert 'Сохранить текущий график' in doc
    assert 'Прогнозная допустимость: не подтверждена' in doc
    assert 'Данные решения' in doc


def test_attribution_groups_keep_vestibule_z_units_without_summing_contributions():
    facts = [{'id': 'attribution', 'kind': 'attribution', 'value': [
        {'station_id': 'A', 'vestibule_id': 'A1', 'target_hour': '08:00', 'horizon_hours': 1,
         'raw_q50_z': .1, 'features': [{'name': 'level', 'value': 5, 'contribution_z': .2}]},
        {'station_id': 'A', 'vestibule_id': 'A2', 'target_hour': '08:00', 'horizon_hours': 1,
         'raw_q50_z': .2, 'features': [{'name': 'level', 'value': 8, 'contribution_z': -.4}]}]}]
    groups = summarize_attribution_facts(facts)
    assert len(groups) == 1
    assert groups[0]['station_id'] == 'A'
    assert groups[0]['features'][0]['vestibule_id'] == 'A2'
    assert groups[0]['features'][0]['contribution_z'] == -.4
    assert 'summed_contribution_z' not in groups[0]
    assert groups[0]['unit'] == 'raw z = log((y+1)/(hourly_reference+1))'


def test_forecast_review_uses_only_complete_truth_and_marks_after_execution(tmp_path):
    folder = tmp_path/'forecast_review'
    folder.mkdir()
    rows = [dict(arm='balanced', origin='2026-09-30T08:00:00+03:00',
                 interval_end='2026-09-30T08:30:00+03:00', truth_status='complete',
                 truth=10, predicted=8, decision_predicted=12),
            dict(arm='balanced', origin='2026-09-30T08:00:00+03:00',
                 interval_end='2026-09-30T08:45:00+03:00', truth_status='missing',
                 truth='', predicted=100, decision_predicted=200)]
    with (folder/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    before = (folder/'predictions.csv').read_bytes()
    [review] = collect_post_execution_forecasts(tmp_path)
    assert review['n'] == 1 and review['target_rows'] == 2
    assert review['actual_sum'] == 10 and review['predicted_sum'] == 8
    assert review['decision_predicted_sum'] == 12 and review['wape'] == .2
    assert review['known_after_execution'] is True and review['used_for_dispatch'] is False
    assert review['known_after'] == '2026-09-30T08:30:00+03:00'
    assert (folder/'predictions.csv').read_bytes() == before


def test_incident_probe_is_diagnostic_and_never_promoted_to_causal_safe_saving(tmp_path):
    case = dict(duration_minutes=8, synthetic=True, external_delivery_measured=False,
        receipt=dict(status='evaluation_incomplete', elapsed_seconds=12, within_budget=True),
        diagnostics=dict(incident=dict(signalling_verified=False, blocked_track_all_trains_verified=False),
            movement=dict(movement_passed=False, minimum_headway=113, maximum_headway=593),
            local_service=dict(status='failed', cohorts=[dict(missed_first_share=.4, missed_second_share=.1)]),
            waiting_budget=dict(windows=[dict(name='whole_period', mean_delta_seconds=35, p95_delta_seconds=301)]),
            max_station_direction_queue=1272, remaining_queued=0, remaining_onboard=13,
            passenger_recovery_complete=False, future_actual_used_for_evaluation=True,
            causal_operational_recommendation=False, action='retain fleet', economic_gain_proven=False,
            cash_saving_rub=None))
    folder = tmp_path/'stop_8'
    folder.mkdir()
    (folder/'result.json').write_text(json.dumps(case), encoding='utf-8')
    rows = collect_incident_probes([tmp_path])
    assert len(rows) == 1
    row = rows[0]
    assert row['duration_minutes'] == 8 and row['maximum_headway'] == 593
    assert row['recovery_complete'] is False and row['remaining_onboard'] == 13
    assert row['external_delivery_measured'] is False
    assert row['causal_operational_recommendation'] is False
    assert row['accepted_resource_saving'] is False and row['cash_saving_rub'] is None
    assert len(row['source_sha256']) == 64
    data = make_comparison(results(), audits(admitted=False), paired_inputs_passed=True)
    data['incident_probes'] = rows
    render_reports(data, tmp_path/'review', create_pdf=False)
    md = (tmp_path/'review/report.md').read_text(encoding='utf-8')
    assert 'Диагностические остановки составов' in md
    assert '593' in md and 'не завершено' in md


def test_short_status_report_never_runs_engine_audit_or_claims_a_full_day(tmp_path, monkeypatch):
    import metro51_report
    monkeypatch.setattr(metro51_report, 'audit_arm', lambda *a: (_ for _ in ()).throw(AssertionError('engine audit forbidden')))
    for arm, result in results().items():
        result['contract'].update(full_day=False, stage='pilot')
        result['metrics']['actual_per_train'] = result['metrics']['per_train']
        folder = tmp_path/arm
        folder.mkdir()
        (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    output = tmp_path/'status'
    manifest = write_status_report(tmp_path, output, create_pdf=False)
    md = (output/'status.md').read_text(encoding='utf-8')
    assert manifest['engine_audit_performed'] is False and manifest['goal_achieved'] is False
    assert 'Полный суточный прогон не выполнен' in md
    assert 'Без СР и КР' in md and 'С СР и КР' in md
    assert 'годовую экономию' in md
    ui = (output/'dispatcher.html').read_text(encoding='utf-8')
    assert 'href="status.html"' in ui and 'href="status.pdf"' in ui
    assert (output/'explanations.csv').is_file()
