from copy import deepcopy
from decimal import Decimal

import pytest

from metro47_acceptance import assess_pair


def case(scale='1'):
    scale = Decimal(scale)
    fleet = [('B', 'Baltiets', 37), ('Y', 'Yubileyny', 31)]
    work = [dict(train_id=f'{prefix}{i:03}', train_type=kind,
                 passenger_train_km=str(100 * scale), nonpassenger_train_km=str(10 * scale))
            for prefix, kind, size in fleet for i in range(1, size + 1)]
    summary = dict(passenger_car_seconds=100, exceptional_car_seconds=1,
                   exceptional_time_share=.01, max_load=160, hard_limit_car_seconds=0)
    return dict(simulated_line=1, period_start='2026-09-30T08:15:00+03:00',
        cohort_digest='same passengers', profiles_digest='same car profiles',
        common_start_digest='same physical state',
        cohort_scope=dict(start_second=3600, end_second=21600),
        service_scope=dict(start_second=3600, end_second=22200),
        resource_scope=dict(start_second=0, end_second=32400, includes_entry_exit=True),
        closure_second=32400, movement_obligations_closed=True,
        physical=dict(movement_passed=True, headways_passed=True, terminal_passed=True,
            depot_preparation_passed=True, executed_prefix_preserved=True,
            current_movements_preserved=True, peak_trains_on_line=53),
        entered=247117, completed=247117, queued=0, approaching=0, onboard=0,
        all_service_arrivals_completed=True,
        exposure=dict(admissible=True, global_summary=deepcopy(summary),
            windows=[dict(summary, start=0, end=32400)],
            per_train={'B001': deepcopy(summary)}, per_car={'B001:C1': deepcopy(summary)}),
        transport_work=work, mean_wait_seconds=70, p95_wait_seconds=140)


def test_exact_one_percent_is_accepted_without_waiting_budgets():
    old, new = case(), case('.99')
    new.update(mean_wait_seconds=170, p95_wait_seconds=500)
    result = assess_pair(old, new)
    assert result['accepted'] and result['saving_fraction_lower_bound'] == pytest.approx(.01)
    assert result['waiting_budget_applied'] is False
    assert result['cash_saving_rub'] is result['lost_revenue_rub'] is None


def test_threshold_below_one_percent_is_rejected():
    result = assess_pair(case(), case('.9901'))
    assert not result['accepted'] and 'saving_below_threshold' in result['reasons']


def test_type_specific_resource_cost_uses_train_km_and_fixed_68():
    result = assess_pair(case(), case('.98'))
    expected = (Decimal(37) * Decimal('9.2') + Decimal(31) * Decimal('7.4')) * 110
    assert result['reference_to1_resource_rub_upper_bound'] == float(expected)
    assert result['fixed_fleet_size'] == 68
    assert result['saving_km_per_fixed_train_lower_bound'] == pytest.approx(2.2)


def test_empty_entry_exit_work_is_included_in_the_threshold():
    old, new = case(), case()
    for row in new['transport_work']:
        row['nonpassenger_train_km'] = '9'
    result = assess_pair(old, new)
    assert result['saving_train_km_lower_bound'] == 68
    assert result['saving_fraction_lower_bound'] == pytest.approx(1 / 110)
    assert not result['accepted']


@pytest.mark.parametrize('key', ['cohort_digest', 'profiles_digest', 'common_start_digest'])
def test_mismatched_pair_provenance_is_rejected(key):
    new = case('.98'); new[key] = 'different'
    assert 'pair_context_mismatch' in assess_pair(case(), new)['reasons']


@pytest.mark.parametrize('key', ['cohort_scope', 'resource_scope', 'service_scope'])
def test_time_boundaries_must_match(key):
    new = case('.98'); new[key]['end_second'] += 1
    assert 'pair_context_mismatch' in assess_pair(case(), new)['reasons']


def test_horizon_must_include_entry_and_exit():
    new = case('.98'); new['resource_scope']['includes_entry_exit'] = False
    assert 'entry_exit_scope_unverified' in assess_pair(case(), new)['reasons']


def test_only_line_one_is_accepted():
    new = case('.98'); new['simulated_line'] = 2
    assert 'line_one_required' in assess_pair(case(), new)['reasons']


def test_missing_period_or_invalid_scopes_are_not_a_common_closed_horizon():
    old, new = case(), case('.98')
    old.pop('period_start'); new.pop('period_start')
    assert not assess_pair(old, new)['accepted']
    new = case('.98'); new['cohort_scope']['end_second'] = 33000
    assert 'candidate_full_horizon_unverified' in assess_pair(case(), new)['reasons']


def test_mismatched_start_does_not_receive_a_guaranteed_saving_fraction():
    new = case('.98'); new['common_start_digest'] = 'another starting movement'
    assert assess_pair(case(), new)['saving_fraction_lower_bound'] is None


@pytest.mark.parametrize('side', ['reference', 'candidate'])
def test_capacity_failure_prevents_economic_acceptance(side):
    old, new = case(), case('.98')
    selected = old if side == 'reference' else new
    selected['exposure']['global_summary']['max_load'] = 161
    assert f'{side}_passenger_capacity' in assess_pair(old, new)['reasons']


def test_window_exception_share_cannot_hide_behind_global_admissible():
    new = case('.98')
    selected = new['exposure']['windows'][0]
    selected['exceptional_time_share'] = .01001
    assert 'candidate_passenger_capacity' in assess_pair(case(), new)['reasons']


@pytest.mark.parametrize('component', ['per_train', 'per_car'])
def test_per_train_and_car_exception_share_is_diagnostic_when_global_windows_pass(component):
    new = case('.98')
    prototype = new['exposure']['global_summary']
    new['exposure'][component] = [dict(prototype, passenger_car_seconds=50,
        exceptional_car_seconds=1, exceptional_time_share=.02),
        dict(prototype, passenger_car_seconds=50, exceptional_car_seconds=0,
             exceptional_time_share=0)]
    assert assess_pair(case(), new)['accepted']


def test_individual_car_hard_limit_still_rejects_a_globally_safe_claim():
    new = case('.98')
    new['exposure']['per_car']['B001:C1']['max_load'] = 161
    assert 'candidate_passenger_capacity' in assess_pair(case(), new)['reasons']


def test_empty_exposure_window_is_allowed_but_nonempty_unknown_is_not():
    new = case('.98')
    new['exposure']['windows'].append(dict(passenger_car_seconds=0,
        exceptional_car_seconds=0, exceptional_time_share=None, max_load=0,
        hard_limit_car_seconds=0))
    assert assess_pair(case(), new)['accepted']
    new['exposure']['windows'][-1]['passenger_car_seconds'] = 1
    assert not assess_pair(case(), new)['accepted']


def test_missing_local_exposure_proof_is_not_accepted():
    new = case('.98'); new['exposure'].pop('per_car')
    assert 'candidate_passenger_capacity' in assess_pair(case(), new)['reasons']


def test_reported_exception_share_must_match_the_measured_duration():
    new = case('.98')
    new['exposure']['global_summary']['exceptional_car_seconds'] = 2
    assert 'candidate_passenger_capacity' in assess_pair(case(), new)['reasons']


def test_line_capacity_and_physical_gates_are_required():
    new = case('.98'); new['physical']['peak_trains_on_line'] = 54
    assert 'candidate_physical' in assess_pair(case(), new)['reasons']
    new = case('.98'); new['physical'].pop('executed_prefix_preserved')
    assert 'candidate_physical' in assess_pair(case(), new)['reasons']


def test_unfinished_passenger_stock_or_different_cohort_is_rejected():
    new = case('.98'); new['onboard'] = 1
    assert 'candidate_passengers_unfinished' in assess_pair(case(), new)['reasons']
    new = case('.98'); new['entered'] -= 1; new['completed'] -= 1
    assert 'passenger_total_mismatch' in assess_pair(case(), new)['reasons']


def test_open_endpoint_retains_independent_uncertainty_and_is_not_accepted():
    old, new = case(), case('.98')
    new['movement_obligations_closed'] = False
    new['endpoint_km_upper_bound'] = {r['train_id']: '0' for r in new['transport_work']}
    new['endpoint_km_upper_bound']['B001'] = '10'
    result = assess_pair(old, new)
    assert not result['accepted'] and 'candidate_movement_unclosed' in result['reasons']
    assert result['saving_train_km_lower_bound'] == pytest.approx(149.6 - 10)
    assert result['saving_to1_resource_rub_lower_bound'] == pytest.approx(1253.56 - 92)
    assert result['position_interpolation_assumed'] is False


def test_unclosed_endpoint_without_a_bound_is_not_fabricated():
    new = case('.98'); new['movement_obligations_closed'] = False
    result = assess_pair(case(), new)
    assert result['saving_fraction_lower_bound'] is None and not result['accepted']


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-1', True])
def test_invalid_mileage_is_rejected(value):
    new = case('.98'); new['transport_work'][0]['passenger_train_km'] = value
    with pytest.raises(ValueError): assess_pair(case(), new)


def test_fixed_fleet_requires_all_rows_and_matching_train_types():
    new = case('.98'); new['transport_work'].pop()
    with pytest.raises(ValueError, match='68'): assess_pair(case(), new)
    new = case('.98'); new['transport_work'][0]['train_type'] = 'Yubileyny'
    with pytest.raises(ValueError, match='fleet'): assess_pair(case(), new)


def test_inputs_are_not_mutated():
    old, new = case(), case('.98'); preserved = deepcopy((old, new))
    assess_pair(old, new)
    assert (old, new) == preserved
