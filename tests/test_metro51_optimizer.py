import copy

import pytest

from tests.test_metro47_cost_policy import row
from metro51_optimizer import choose_balanced, solve_joint_clocks


def marked(name='KEEP', km=10, mean=80, p95=150, local=True):
    value = row(name, km, mean, p95)
    value['trials'][0]['scenario'] = 'S1'
    value['trials'][0]['peak_boarding_service'] = {'passed': local, 'status': 'passed' if local else 'failed'}
    return value


def test_wait_budget_rejects_cheaper_candidate():
    result = choose_balanced([marked(), marked('CHEAP', 9, 91, 150)])
    assert result['selected_candidate_id'] == 'KEEP'
    assert result['waiting_budgets'] == {'mean_seconds': 90, 'p95_seconds': 190}


def test_p95_and_local_failures_cannot_be_hidden_by_aggregate_flags():
    rows = [marked(), marked('P95', 8, 80, 191), marked('LOCAL', 7, local=False)]
    assert choose_balanced(rows)['selected_candidate_id'] == 'KEEP'


def test_budget_boundaries_are_inclusive_and_input_is_preserved():
    rows = [marked(), marked('CHEAP', 9, 90, 190)]
    saved = copy.deepcopy(rows)
    assert choose_balanced(rows)['selected_candidate_id'] == 'CHEAP'
    assert rows == saved


def test_failed_keep_is_not_certified_as_fallback():
    answer = choose_balanced([marked(local=False)])
    assert answer['selected_candidate_id'] is None
    assert answer['fallback_service_verified'] is False


def clocks():
    return [dict(track_id='t_up', direction='up', second=s, trip_id=f'j{i}',
                 minimum_headway_seconds=150, terminal_offset_seconds=0)
            for i, s in enumerate((0, 150, 300, 450, 600, 750, 900))]


def test_continuous_clock_search_keeps_minimum_and_maximum():
    departures = clocks()
    shifts = {f'j{i}': (-20, 20) for i in range(1, 6)}
    answer = solve_joint_clocks([], departures, {}, [], as_of=1, end_second=900,
                                shift_bounds=shifts, time_limit_seconds=2)
    assert answer['status'] in ('optimal', 'search_incomplete')
    actual = [r['second'] + answer['trip_shifts'].get(r['trip_id'], 0) for r in departures]
    assert all(150 <= b-a <= 480 for a, b in zip(actual, actual[1:]))
    assert answer['fixed_headway_grid_used'] is False


def test_withdrawal_and_retiming_are_solved_together():
    departures = [dict(track_id='t_up', direction='up', second=s, trip_id=f'j{i}',
                       minimum_headway_seconds=150, terminal_offset_seconds=0)
                  for i, s in enumerate((0, 200, 400, 600, 800))]
    options = [dict(train_id='train', cancel_service_ids=['j2'])]
    answer = solve_joint_clocks(options, departures, {'train': 100}, [], as_of=1,
        end_second=800, shift_bounds={'j1': (-30, 30), 'j3': (-30, 30)}, time_limit_seconds=2)
    assert answer['selected_train_ids'] == ['train']
    assert answer['estimated_maintenance_resource_saving_rub'] == 100
    assert 'trip_shifts' in answer


def test_forecast_capacity_blocks_profitable_withdrawal():
    departures = clocks()
    options = [dict(train_id='train', cancel_service_ids=['j2'])]
    rates = [dict(start_second=0, end_second=900, up_rate=5, down_rate=0)]
    answer = solve_joint_clocks(options, departures, {'train': 100}, rates,
        as_of=1, end_second=900, shift_bounds={}, capacity=1200, time_limit_seconds=2)
    assert answer['selected_train_ids'] == []


def test_started_duties_cannot_be_retimed():
    with pytest.raises(ValueError, match='committed'):
        solve_joint_clocks([], clocks(), {}, [], as_of=200, end_second=900,
            shift_bounds={'j1': (-10, 10)}, time_limit_seconds=1)


def test_train_continuity_limits_differential_shifts():
    answer = solve_joint_clocks([], clocks(), {}, [], as_of=1, end_second=900,
        shift_bounds={'j1': (-20,20), 'j2': (-20,20)},
        continuities=[('j1','j2',0)], time_limit_seconds=2)
    assert answer['trip_shifts']['j2'] >= answer['trip_shifts']['j1']


def test_missing_successor_is_infeasible_instead_of_an_unchecked_gap():
    departures = [dict(track_id='t_up',direction='up',second=s,trip_id=f'j{i}',
                       minimum_headway_seconds=150) for i,s in enumerate((0,900,1800))]
    answer = solve_joint_clocks([],departures,{},[],as_of=1,end_second=1800,
                               shift_bounds={'j1':(-20,20)},time_limit_seconds=1)
    assert answer['status'] == 'no_feasible_candidate'


def test_fixed_reference_budget_prevents_allowance_accumulation():
    original = marked(mean=80,p95=150)['trials']
    rows = [marked(mean=90,p95=190),marked('CHEAP',9,100,230)]
    assert choose_balanced(rows,reference_trials=original)['selected_candidate_id'] == 'KEEP'


def test_peak_boundary_uses_later_departure_policy():
    departures=[dict(track_id='t_up',direction='up',second=s,trip_id=f'j{i}',
                    minimum_headway_seconds=m) for i,(s,m) in enumerate(((0,150),(113,113),(226,113)))]
    answer=solve_joint_clocks([],departures,{},[],as_of=1,end_second=226,
        shift_bounds={'j1':(0,0)},time_limit_seconds=1)
    assert answer['status'] == 'optimal'
