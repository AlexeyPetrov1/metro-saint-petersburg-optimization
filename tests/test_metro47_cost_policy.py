import copy

import pytest

from metro47_cost_policy import choose_by_cost


def row(name='KEEP', km=10, mean=80, p95=150):
    exposure = {'admissible': True, 'global': dict(exceptional_time_share=0,
        exceptional_car_seconds=0, passenger_car_seconds=1000,
        hard_limit_car_seconds=0, max_load=150), 'windows': []}
    return dict(candidate_id=name, status='evaluated', passengers_checked=True,
        physical_passed=True, service_passed=True, committed_per_train=[
            dict(train_id=f'T{i:02}', train_type='Baltiets' if i < 37 else 'Yubileyny', km=km)
            for i in range(68)], trials=[dict(policy_exposure=exposure,
                headways=dict(passed=True), mature_cohort=dict(complete=True,
                    arrived=100, unboarded=0, mean_wait_seconds=mean, p95_wait_seconds=p95),
                initial_obligations=dict(complete=True, pending_at_end=0))])


def test_cost_can_trade_higher_wait_for_admissible_service():
    result = choose_by_cost([row(), row('CHEAP', km=9, mean=200, p95=470)],
        mean_wait_budget_s=80, p95_wait_budget_s=150)
    assert result['selected_candidate_id'] == 'CHEAP'
    assert result['waiting_budgets'] == dict(mean_seconds=None, p95_seconds=None)
    assert result['economic_improvement_claimed'] is False


def test_type_specific_cost_can_prefer_more_total_km():
    balt, yub = row('B', km=0), row('Y', km=0)
    balt['committed_per_train'][0]['km'] = 100
    yub['committed_per_train'][37]['km'] = 110
    assert choose_by_cost([balt, yub])['selected_candidate_id'] == 'Y'


@pytest.mark.parametrize('failure', ['headway', 'capacity', 'closure', 'unchecked'])
def test_cheapest_plan_cannot_bypass_constraints(failure):
    cheap = row('CHEAP', km=0)
    if failure == 'headway': cheap['trials'][0]['headways']['passed'] = False
    if failure == 'capacity': cheap['trials'][0]['policy_exposure']['admissible'] = False
    if failure == 'closure': cheap['trials'][0]['mature_cohort']['complete'] = False
    if failure == 'unchecked': cheap['passengers_checked'] = False
    assert choose_by_cost([row(), cheap])['selected_candidate_id'] == 'KEEP'


def test_equal_cost_keeps_current_schedule():
    assert choose_by_cost([row('CHANGE'), row()])['selected_candidate_id'] == 'KEEP'


def test_matching_fleet_is_required():
    bad = row('CHEAP', km=0)
    bad['committed_per_train'][0]['train_id'] = 'outsider'
    with pytest.raises(ValueError, match='fleet'):
        choose_by_cost([row(), bad])


def test_no_valid_plan_is_reported_without_approval():
    bad = row()
    bad['physical_passed'] = False
    result = choose_by_cost([bad])
    assert result['selected_candidate_id'] is None
    assert result['release_authorized'] is False


def test_selection_does_not_mutate_evidence():
    rows = [row(), row('CHEAP', km=9)]
    saved = copy.deepcopy(rows)
    choose_by_cost(rows)
    assert rows == saved


def test_full_maintenance_objective_changes_type_choice_and_keeps_TO1_separate():
    from metro50_economics import maintenance_rates
    rates = {kind: v['without_sr_kr'] for kind, v in maintenance_rates().items()}
    balt, yub = row('B', km=0), row('Y', km=0)
    balt['committed_per_train'][0]['km'] = 100
    yub['committed_per_train'][37]['km'] = 110
    assert choose_by_cost([balt, yub])['selected_candidate_id'] == 'Y'
    result = choose_by_cost([balt, yub], maintenance_rates=rates)
    assert result['selected_candidate_id'] == 'B'
    assert result['resources']['maintenance_resource_equivalent_rub'] == pytest.approx(100 * rates['Baltiets'])
    assert result['resources']['to1_resource_equivalent_rub'] == pytest.approx(920)
    assert result['resources']['per_train'][0]['maintenance_rub_per_km'] == rates['Baltiets']
    assert 'TO1+TO2+TR1+TR2+TR3' in result['objective']
    assert result['cash_saving_rub'] is None


@pytest.mark.parametrize('rates', [{'Baltiets': 113}, {'Baltiets': 113, 'Yubileyny': 0},
    {'Baltiets': 113, 'Yubileyny': float('nan')}, {'Baltiets': True, 'Yubileyny': 104},
    {'unknown': 113, 'Yubileyny': 104}])
def test_invalid_full_maintenance_rates_fail_even_when_no_candidate(rates):
    with pytest.raises(ValueError, match='maintenance'):
        choose_by_cost([], maintenance_rates=rates)


def test_full_maintenance_rates_support_existing_type_aliases():
    result = choose_by_cost([row()], maintenance_rates={'Балтиец': 113, 'юбилейный': 104})
    assert result['resources']['maintenance_resource_equivalent_rub'] == pytest.approx(37 * 10 * 113 + 31 * 10 * 104)


def test_full_maintenance_objective_requires_37_and_31_type_counts():
    changed = row()
    changed['committed_per_train'][0]['train_type'] = 'Yubileyny'
    with pytest.raises(ValueError, match='fleet'):
        choose_by_cost([changed], maintenance_rates={'Baltiets': 113, 'Yubileyny': 104})
