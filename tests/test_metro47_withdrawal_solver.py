from copy import deepcopy
from time import monotonic

import pytest

from metro47_withdrawal_solver import solve_withdrawal_options


def case():
    options = [dict(train_id=t, cancel_service_ids=[t+'0', t+'1', t+'2']) for t in ('A', 'B', 'C')]
    values = {'A': 920, 'B': 1840, 'C': 740}
    departures = [dict(track_id='P', direction='up', terminal_offset_seconds=0,
                       second=0, trip_id='past')]
    for i, t in enumerate(('A', 'B', 'C')):
        for j in range(3):
            departures.append(dict(track_id='P', direction='up', terminal_offset_seconds=0,
                second=150*(i+1)+450*j, trip_id=t+str(j)))
    return options, departures, values


def solve(options=None, departures=None, values=None, **kwargs):
    base = case()
    return solve_withdrawal_options(options or base[0], departures or base[1],
        option_values=values or base[2], rate_slots=kwargs.pop('rate_slots', []),
        as_of=150, service_end_second=1500, **kwargs)


def test_solver_selects_concrete_highest_value_subset_with_480_second_coverage():
    result = solve()
    assert result['status'] == 'optimal'
    assert result['selected_train_ids'] == ['A', 'B']
    assert result['estimated_to1_resource_saving_rub'] == 2760
    assert result['withdrawal_count'] == 2
    assert result['waiting_in_objective'] is False


def test_forecast_capacity_can_prevent_a_gap_that_the_480_limit_would_allow():
    rates = [dict(start_second=150, end_second=1050, up_rate=6, down_rate=0)]
    result = solve(rate_slots=rates)
    assert result['status'] == 'optimal' and result['selected_train_ids'] == []
    assert result['forecast_capacity_constraints'] > 0


def test_infeasible_existing_capacity_is_not_called_an_approved_keep():
    rates = [dict(start_second=150, end_second=1050, up_rate=20, down_rate=0)]
    result = solve(rate_slots=rates)
    assert result['status'] == 'no_feasible_candidate'
    assert result['selected_train_ids'] == []
    assert result['passenger_admissible'] is False


def test_max_withdrawal_feedback_bounds_cardinality_without_changing_the_grid():
    result = solve(max_withdrawals=1)
    assert result['selected_train_ids'] == ['B']
    assert solve(max_withdrawals=0)['selected_train_ids'] == []


def test_no_good_cut_excludes_only_the_exact_previous_subset():
    options, departures, values = case()
    result = solve(no_good_cuts=[['A', 'B']])
    assert result['selected_train_ids'] == ['B', 'C']
    # A cut excluding A alone must still allow A+B. That superset has a distinct
    # physical plan and may release a terminal blocking the failed A-only plan.
    assert solve(no_good_cuts=[['A']])['selected_train_ids'] == ['A', 'B']


def test_wholly_begun_trip_departures_remain_mandatory_even_for_a_removed_train():
    options, departures, values = case()
    options[0]['cancel_service_ids'] = ['A1', 'A2']
    result = solve(options, departures, values)
    assert result['selected_train_ids'] == ['A', 'B']
    assert result['immutable_departure_count'] == 2


def test_tail_has_no_fabricated_demand_and_no_money_is_claimed():
    result = solve()
    assert result['forecast_capacity_constraints'] == 0
    assert result['tail_demand_scope'] == 'no supplied forecast: fixed-clock maximum-gap coverage only'
    assert result['cash_saving_rub'] is None and result['release_authorized'] is False


def test_preserved_inputs_and_positive_type_cost_only_objective():
    options, departures, values = case()
    before = deepcopy((options, departures, values))
    result = solve(options, departures, values)
    assert (options, departures, values) == before
    assert result['objective'] == 'cancelled_known_train_km_times_confirmed_TO1_rates'


def test_unknown_cut_or_duplicate_trip_ownership_is_rejected():
    with pytest.raises(ValueError, match='unknown cut'):
        solve(no_good_cuts=[['unknown']])
    options, departures, values = case()
    options[1]['cancel_service_ids'].append('A0')
    with pytest.raises(ValueError, match='trip ownership'):
        solve(options, departures, values)


def test_boundary_end_and_intermediate_tracks_require_a_retained_service():
    options, departures, values = case()
    departures.append(dict(track_id='Q', direction='down', terminal_offset_seconds=0,
                           second=150, trip_id='immutable'))
    result = solve(options, departures, values)
    assert result['status'] == 'no_feasible_candidate'


def test_forecast_capacity_also_covers_the_last_480_seconds_of_the_window():
    options = [dict(train_id='A', cancel_service_ids=['A750'])]
    rows = [dict(track_id='P', direction='up', terminal_offset_seconds=0,
                 second=t, trip_id='A750' if t == 750 else str(t))
            for t in range(0, 901, 150)]
    rates = [dict(start_second=600, end_second=1500, up_rate=6, down_rate=0)]
    result = solve_withdrawal_options(options, rows, option_values={'A': 100},
        rate_slots=rates, as_of=450, service_end_second=900)
    assert result['selected_train_ids'] == []


def test_expired_budget_returns_no_withdrawal():
    result = solve(deadline=monotonic()-1)
    assert result['status'] == 'evaluation_timeout' and result['selected_train_ids'] == []


@pytest.mark.parametrize('maximum', [True, -1, 1.5])
def test_invalid_feedback_cardinality_is_rejected(maximum):
    with pytest.raises(ValueError):
        solve(max_withdrawals=maximum)
