import copy
import importlib.util
from pathlib import Path

import pytest

# Historical experiment imports prepend snapshots to sys.path during collection.
_spec = importlib.util.spec_from_file_location('_workspace_metro44_report', Path(__file__).resolve().parents[1]/'metro44_report.py')
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)
compare_results = _report.compare_results


def test_report_subject_is_current_workspace_source():
    assert Path(compare_results.__code__.co_filename).resolve() == Path(__file__).resolve().parents[1]/'metro44_report.py'


def result(arm, km=100, mean=90, p95=180, valid=True, pending=0):
    fleet = [dict(train_id=f'T{i:02}', train_type='Baltiets' if i < 37 else 'Yubileyny', km=km)
             for i in range(68)]
    return dict(strategy=arm, groups_digest='same', checkpoint_digest='same',
        metrics=dict(per_train=fleet, train_km=km*68, actual_service_admissible=valid,
            terminal_resources_passed=True,
            movement_obligations_closed=True, mean_wait_seconds=mean, p95_wait_seconds=p95,
            cohort=dict(complete=pending == 0, unboarded=pending),
            mature_cohort=dict(complete=True), headways=dict(passed=True),
            policy_exposure=dict(admissible=valid, windows=[])))


def test_safe_gain_uses_fixed_68_and_type_weighted_to1_cost():
    report = compare_results([result('STANDARD'), result('MPC_ML', km=90)])
    row = report['comparisons'][0]
    assert row['accepted_improvement']
    assert row['observed_reduction_train_km'] == 680
    assert row['observed_reduction_km_per_fixed_train'] == 10
    assert row['to1_resource_equivalent_rub'] == pytest.approx(37*10*9.2+31*10*7.4)
    assert row['cash_saving_rub'] is None
    assert report['development_period_used']


@pytest.mark.parametrize('kind', ['unsafe_reference', 'unsafe_candidate', 'wait', 'p95', 'pending'])
def test_lower_mileage_cannot_hide_service_failure(kind):
    baseline, candidate = result('STANDARD'), result('MPC_ML', km=90)
    if kind == 'unsafe_reference': baseline['metrics']['actual_service_admissible'] = False
    if kind == 'unsafe_candidate': candidate['metrics']['actual_service_admissible'] = False
    if kind == 'wait': candidate['metrics']['mean_wait_seconds'] = 91
    if kind == 'p95': candidate['metrics']['p95_wait_seconds'] = 181
    if kind == 'pending': candidate['metrics']['cohort'] = dict(complete=False, unboarded=1)
    row = compare_results([baseline, candidate])['comparisons'][0]
    assert not row['accepted_improvement']
    assert row['accepted_saving_train_km'] is None
    assert row['net_profit_effect_rub'] is None


def test_mismatched_passengers_or_initial_state_are_not_comparable():
    a, b = result('STANDARD'), result('MPC_ML')
    b['checkpoint_digest'] = 'other'
    with pytest.raises(ValueError, match='shared'):
        compare_results([a, b])


def test_all_unsafe_results_are_reported_without_a_fictitious_best():
    report = compare_results([result('STANDARD', valid=False), result('MPC_ML', km=80, valid=False)])
    assert report['best_accepted_strategy'] is None
    assert not report['confirmed_cash_savings']
    assert report['incomplete_acceptance']


def test_expensive_type_allocation_can_increase_wear_despite_less_distance():
    a, b = result('STANDARD'), result('MPC_ML')
    for r in a['metrics']['per_train']: r['km'] = 200 if r['train_type'] == 'Yubileyny' else 0
    for r in b['metrics']['per_train']: r['km'] = 160 if r['train_type'] == 'Baltiets' else 0
    a['metrics']['train_km'] = 6200
    b['metrics']['train_km'] = 5920
    report = compare_results([a, b])
    assert report['comparisons'][0]['observed_reduction_train_km'] == 280
    assert report['comparisons'][0]['to1_resource_equivalent_rub'] < 0
    assert not report['comparisons'][0]['accepted_improvement']


def test_input_results_are_never_rewritten():
    rows = [result('STANDARD'), result('MPC_ML', km=90)]
    before = copy.deepcopy(rows)
    compare_results(rows)
    assert rows == before


def test_finite_cohort_tail_saving_requires_common_continuation():
    baseline, candidate = result('STANDARD'), result('MPC_ML', km=90)
    for row in (baseline, candidate):
        row['economic_claim_requires_common_continuation'] = True
        row['boundary_end_is_operational_closure'] = False
    compared = compare_results([baseline, candidate])
    assert not compared['comparisons'][0]['accepted_improvement']
    assert 'common_continuation_not_verified' in compared['comparisons'][0]['reasons']
    assert compared['best_accepted_strategy'] is None


def test_common_continuation_receipt_must_match_both_arms():
    baseline, candidate = result('STANDARD'), result('MPC_ML', km=90)
    for row in (baseline, candidate):
        row['economic_claim_requires_common_continuation'] = True
        row['common_continuation'] = dict(verified=True, contract_digest='same')
    assert compare_results([baseline, candidate])['comparisons'][0]['accepted_improvement']
    candidate['common_continuation']['contract_digest'] = 'other'
    assert not compare_results([baseline, candidate])['comparisons'][0]['accepted_improvement']


def test_terminal_resource_failure_cannot_be_a_service_valid_gain():
    candidate = result('MPC_ML', km=90)
    candidate['metrics']['terminal_resources_passed'] = False
    compared = compare_results([result('STANDARD'), candidate])
    assert not compared['rows'][1]['service_valid']
    assert not compared['comparisons'][0]['accepted_improvement']


@pytest.mark.parametrize('arm', ['STANDARD', 'MPC_ML'])
@pytest.mark.parametrize('proof', ['missing', None, 1, 'true'])
def test_terminal_check_requires_an_explicit_true_receipt(arm, proof):
    rows = [result('STANDARD', mean=70), result('MPC_ML', km=90, mean=80)]
    row = next(r for r in rows if r['strategy'] == arm)
    if proof == 'missing':
        row['metrics'].pop('terminal_resources_passed')
    else:
        row['metrics']['terminal_resources_passed'] = proof
    rows[1]['window_savings_bounds'] = dict(verified=True, saving_train_km_lower_bound=10,
        saving_to1_resource_rub_lower_bound=74)
    compared = compare_results(rows, alternate_wait_budget=dict(mean_seconds=90, p95_seconds=180))
    assert not next(r for r in compared['rows'] if r['strategy'] == arm)['service_valid']
    assert compared['best_conditional_strategy'] is None


def test_explicit_development_tradeoff_is_separate_from_strict_improvement():
    baseline, candidate = result('STANDARD', mean=70, p95=138), result('MPC_ML', km=90, mean=80, p95=150)
    candidate['window_savings_bounds'] = dict(verified=True, saving_train_km_lower_bound=10,
        saving_to1_resource_rub_lower_bound=74)
    compared = compare_results([baseline, candidate], alternate_wait_budget=dict(mean_seconds=90, p95_seconds=180))
    row = compared['comparisons'][0]
    assert not row['accepted_improvement']
    assert row['conditional_development_improvement']
    assert compared['best_accepted_strategy'] is None
    assert compared['best_conditional_strategy'] == 'MPC_ML'
    assert not compared['confirmed_cash_savings']


@pytest.mark.parametrize('kind', ['terminal', 'wait', 'boundary', 'wear'])
def test_conditional_result_still_needs_physics_wait_and_positive_window_bounds(kind):
    baseline, candidate = result('STANDARD', mean=70), result('MPC_ML', km=90, mean=80)
    candidate['window_savings_bounds'] = dict(verified=True, saving_train_km_lower_bound=10,
        saving_to1_resource_rub_lower_bound=74)
    if kind == 'terminal': candidate['metrics']['terminal_resources_passed'] = False
    if kind == 'wait': candidate['metrics']['mean_wait_seconds'] = 91
    if kind == 'boundary': candidate['window_savings_bounds']['verified'] = False
    if kind == 'wear': candidate['window_savings_bounds']['saving_to1_resource_rub_lower_bound'] = -1
    compared = compare_results([baseline, candidate], alternate_wait_budget=dict(mean_seconds=90, p95_seconds=180))
    assert not compared['comparisons'][0]['conditional_development_improvement']
