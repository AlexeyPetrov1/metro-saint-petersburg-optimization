from types import SimpleNamespace

import pytest

from metro50_review import (audit_receipts, html_report, nearest_rank, peak_car_quantiles,
                            require_finished_results, report_saving)


def result(arm='GRU'):
    times = list(range(3600, 64800, 900)) if arm != 'REFERENCE' else []
    return dict(contract=dict(arm=arm, decision_times=times, all_control_steps=True),
                decisions=[dict(decision_second=s, pipeline_seconds=30.5,
                                local_recommendation_within_480=True) for s in times])


def test_exact_68_completed_receipts_and_inclusive_480_second_deadline():
    source = result(); source['decisions'][0]['pipeline_seconds'] = 480
    audit = audit_receipts(source)
    assert audit['passed'] is True and audit['count'] == 68
    assert audit['maximum_seconds'] == 480


@pytest.mark.parametrize('damage', ['missing', 'repeat', 'unordered', 'late', 'negative', 'nan', 'flag', 'contract'])
def test_incomplete_or_ambiguous_controller_evidence_is_rejected(damage):
    source = result()
    if damage == 'missing': source['decisions'].pop()
    if damage == 'repeat': source['decisions'][1]['decision_second'] = 3600
    if damage == 'unordered': source['decisions'].reverse()
    if damage == 'late': source['decisions'][0]['pipeline_seconds'] = 480.001
    if damage == 'negative': source['decisions'][0]['pipeline_seconds'] = -1
    if damage == 'nan': source['decisions'][0]['pipeline_seconds'] = float('nan')
    if damage == 'flag': source['decisions'][0]['local_recommendation_within_480'] = False
    if damage == 'contract': source['contract']['decision_times'].pop()
    assert audit_receipts(source)['passed'] is False


def test_reference_has_no_required_controller_decisions():
    assert audit_receipts(result('REFERENCE'))['passed'] is True


def test_nearest_rank_uses_ceil_and_includes_zero_loads():
    assert nearest_rank(list(range(100)), .95) == 94
    assert nearest_rank([0]*99+[150], .99) == 0
    assert nearest_rank([0]*99+[150], .999) == 150
    assert nearest_rank([], .95) is None


def test_peak_quantiles_exclude_arrival_nonpassenger_offpeak_and_weekend_rows():
    runner = SimpleNamespace(motion={'period_start': '2026-09-30T06:00:00+03:00'},
        motion_by_id={'P': {'passenger_service': True}, 'N': {'passenger_service': False}},
        profiles={'T': [{'car_id': '1'}, {'car_id': '2'}]})
    rows = [dict(second=s, load=n, event_type=event, motion_event_id=i, train_id='T', car_id=c)
            for s, n, event, i, c in [(7200, 0, 'departure', 'P', '1'),
                                      (7200, 150, 'departure', 'P', '2'),
                                      (7200, 160, 'arrival', 'P', '1'),
                                      (7200, 160, 'departure', 'N', '1'),
                                      (14400, 160, 'departure', 'P', '1')]]
    audit = peak_car_quantiles(runner, {'load_events': rows})
    assert audit['observations'] == 2 and audit['p95'] == audit['p99'] == 150
    assert audit['maximum'] == 150 and audit['zero_observations'] == 1
    assert audit['complete_car_coverage'] is True
    assert peak_car_quantiles(runner, {'load_events': rows+[rows[0]]})['complete_car_coverage'] is False
    runner.motion['period_start'] = '2026-10-03T06:00:00+03:00'
    assert peak_car_quantiles(runner, {'load_events': rows})['observations'] == 0


def test_missing_final_results_stops_before_review_or_mutation(tmp_path):
    (tmp_path/'reference').mkdir()
    (tmp_path/'reference'/'result.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='current5.*dense10'):
        require_finished_results(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['reference']


def test_positive_diagnostic_savings_are_not_published_for_rejected_candidates():
    side = dict(saving_rub=100, saving_percent=2)
    assert report_saving(side, False, 'saving_rub') is None
    assert report_saving(side, True, 'saving_rub') == 100
    assert report_saving(side, True, 'saving_percent') == 2


def test_html_keeps_both_accounting_columns_and_labels_annual_assumptions():
    from metro50_economics import economy_comparison
    rows = [dict(train_id=f'Y{i}', train_type='Yubileyny', km=100) for i in range(31)]
    rows += [dict(train_id=f'B{i}', train_type='Baltiets', km=100) for i in range(37)]
    economy = economy_comparison({'per_train': rows}, {'per_train': [dict(r, km=98) for r in rows]}, admissible=True)
    metrics = dict(entered=758936, completed=758936, train_km=6800, mean_wait_seconds=100,
                   p95_wait_seconds=250, max_platform_direction_queue=500,
                   peak_car_quantiles=dict(p95=130, p99=145, maximum=150),
                   peak_boarding_service=dict(intervals=[], aggregate={'second_train_missed_count': 4}))
    source = dict(paired_inputs_passed=True, audits={'current5': dict(passed=True, metrics=metrics,
                    controller={'maximum_seconds': 100})}, economics={'current5': economy})
    report = html_report(source)
    assert 'Без СР и КР' in report and 'С СР и КР' in report
    assert '250 повторений' in report and '±5%' in report and '±20%' in report
    assert 'кассовая экономия' in report and 'не доверительными' in report
    source['audits']['current5']['passed'] = source['economics']['current5']['admissible'] = False
    assert 'не считается допустимой экономией' in html_report(source)
