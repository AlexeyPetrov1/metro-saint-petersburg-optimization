import copy
import itertools
import time

import pytest

from goal43_optimizer import solve_departures, solve_withdrawals


def departure(event_id, scheduled, earliest=None, **extra):
    row = dict(event_id=event_id, train_id=f"T{event_id}", station_id="A", direction="up",
               scheduled_seconds=scheduled, earliest_seconds=scheduled if earliest is None else earliest)
    return dict(row, **extra)


def test_repairs_intervals_without_changing_committed_departure():
    rows = [departure("a", 0), departure("b", 100), departure("c", 200)]
    result = solve_departures(rows, fixed_departures={"a": 0}, max_hold_seconds=60)
    assert result.status == "optimal"
    assert dict(result.times) == {"a": 0, "b": 113, "c": 226}
    assert result.search_complete and result.validated_scope == "aggregate_departures_only"


def test_no_solution_does_not_return_partial_plan():
    rows = [departure("a", 0), departure("b", 100)]
    result = solve_departures(rows, fixed_departures={"a": 0}, max_hold_seconds=12)
    assert result.status == "no_feasible_candidate"
    assert not result.times and result.search_complete


def test_maximum_gap_is_a_real_constraint():
    rows = [departure("a", 0), departure("b", 600)]
    result = solve_departures(rows, fixed_departures={"a": 0})
    assert result.status == "no_feasible_candidate"


def test_result_matches_independent_enumeration_of_small_regularization_problem():
    rows = [departure("a", 0), departure("b", 3), departure("c", 8)]
    feasible = [(a, b, c) for a, b, c in itertools.product(range(4), range(3, 7), range(8, 12))
                if 2 <= b-a <= 7 and 2 <= c-b <= 7]
    objective = lambda x: (abs(x[1]-x[0]-4)+abs(x[2]-x[1]-4), sum(x)-11)
    optimum = min(map(objective, feasible))
    result = solve_departures(rows, min_headway_seconds=2, max_headway_seconds=7,
                              max_hold_seconds=3, target_headway_seconds=4)
    assert result.status == "optimal"
    assert objective(tuple(result.times[event] for event in "abc")) == optimum
    assert result.objective == optimum


def test_inputs_and_outputs_are_immutable():
    rows = [departure("a", 0), departure("b", 100)]
    committed = {"a": 0}
    before = copy.deepcopy((rows, committed))
    result = solve_departures(rows, fixed_departures=committed)
    assert (rows, committed) == before
    with pytest.raises(TypeError):
        result.times["a"] = 4
    with pytest.raises(AttributeError):
        result.status = "fake"


def test_platforms_and_directions_have_independent_departure_order():
    rows = [departure("a", 0), departure("b", 1, direction="down"),
            departure("c", 2, station_id="B")]
    result = solve_departures(rows)
    assert dict(result.times) == {"a": 0, "b": 1, "c": 2}


def test_train_readiness_links_include_turnaround():
    rows = [departure("a", 0), departure("b", 100, station_id="B", train_id="Ta")]
    result = solve_departures(rows, fixed_departures={"a": 0}, links=[("a", "b", 180)])
    assert result.times["b"] == 180


def test_latest_time_and_horizon_are_not_silently_clipped_or_dropped():
    rows = [departure("a", 0), departure("b", 100, earliest=180)]
    result = solve_departures(rows, fixed_departures={"a": 0}, horizon_end_seconds=179)
    assert result.status == "no_feasible_candidate" and not result.times


def test_actual_departure_clock_selects_offpeak_minimum():
    rows = [departure("a", 0), departure("b", 140)]
    result = solve_departures(rows, fixed_departures={"a": 0},
                              headway_windows=[(0, 130, 113), (130, 1000, 150)])
    assert result.times["b"] == 150


def test_already_expired_budget_does_not_claim_verification():
    result = solve_departures([departure("a", 0)], deadline=time.monotonic()-1)
    assert result.status == "evaluation_timeout"
    assert not result.times and not result.search_complete


def test_tiny_search_budget_returns_timeout_or_complete_checked_solution():
    rows = [departure(str(i), i*100) for i in range(100)]
    result = solve_departures(rows, time_limit_seconds=1e-9, max_hold_seconds=2000)
    assert result.status in {"evaluation_timeout", "optimal", "search_incomplete"}
    if result.times:
        assert all(113 <= result.times[str(i+1)]-result.times[str(i)] <= 480 for i in range(99))
    else:
        assert not result.search_complete


@pytest.mark.parametrize("extra", [{"scheduled_seconds": 0.5}, {"earliest_seconds": True}])
def test_fractional_or_boolean_clock_is_rejected(extra):
    row = departure("a", 0)
    row.update(extra)
    with pytest.raises(ValueError):
        solve_departures([row])


def test_duplicate_ids_and_unknown_links_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        solve_departures([departure("a", 0), departure("a", 120)])
    with pytest.raises(ValueError, match="unknown"):
        solve_departures([departure("a", 0)], links=[("a", "missing", 180)])


def test_overlapping_headway_windows_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        solve_departures([departure("a", 0)], headway_windows=[(0, 100, 113), (99, 1000, 150)])


def test_equal_score_results_are_repeatable_after_input_reordering():
    rows = [dict(train_id="b", saving_units=1), dict(train_id="a", saving_units=1)]
    first = solve_withdrawals(rows, initial_count=2, minimum_remaining=1)
    second = solve_withdrawals(rows[::-1], initial_count=2, minimum_remaining=1)
    assert first.selected_train_ids == second.selected_train_ids


def test_zero_holding_and_missing_headway_clock_coverage_are_infeasible():
    rows = [departure("a", 0), departure("b", 100)]
    result = solve_departures(rows, max_hold_seconds=0)
    assert result.status == "no_feasible_candidate"
    result = solve_departures(rows, headway_windows=[(1000, 2000, 150)])
    assert result.status == "no_feasible_candidate"


def test_fixed_departure_conflicting_with_earliest_is_infeasible():
    result = solve_departures([departure("a", 100, earliest=120)], fixed_departures={"a": 100})
    assert result.status == "no_feasible_candidate" and not result.times


def test_row_readiness_and_row_minimum_are_hard_bounds():
    result = solve_departures([departure("a", 0), departure("b", 100, earliest=170,
                                                               min_headway_seconds=180)],
                              fixed_departures={"a": 0})
    assert result.times["b"] == 180


def test_empty_plan_and_empty_withdrawals_have_no_fabricated_activity():
    departures = solve_departures([])
    fleet = solve_withdrawals([], initial_count=53, minimum_remaining=53)
    assert departures.status == "optimal" and not departures.times
    assert fleet.status == "optimal" and fleet.remaining_count == 53
    assert fleet.saving_units == 0 and not fleet.selected_train_ids


def test_withdrawals_match_brute_force_under_explicit_service_and_cut_bounds():
    rows = [dict(train_id=f"T{i}", saving_units=value) for i, value in enumerate([5, 12, 8, 0])]
    result = solve_withdrawals(rows, initial_count=10, minimum_remaining=8, max_withdrawals=3)
    scores = [sum(row["saving_units"] for row in rows if row["train_id"] in chosen)
              for n in range(3) for chosen in itertools.combinations([row["train_id"] for row in rows], n)]
    assert result.status == "optimal" and result.saving_units == max(scores)
    assert result.selected_train_ids == ("T1", "T2")
    assert result.remaining_count == 8
    assert result.validated_scope == "aggregate_fleet_only"


def test_fleet_overflow_is_rejected_before_solve():
    with pytest.raises(ValueError, match="53"):
        solve_withdrawals([], initial_count=54, minimum_remaining=20)


def test_forbidden_and_negative_saving_withdrawals_are_never_selected():
    rows = [dict(train_id="a", saving_units=100, eligible=False), dict(train_id="b", saving_units=-1)]
    result = solve_withdrawals(rows, initial_count=2, minimum_remaining=0)
    assert not result.selected_train_ids and result.saving_units == 0


def test_withdrawal_result_does_not_mutate_candidates_and_timeout_returns_no_ids():
    rows = [dict(train_id="a", saving_units=10)]
    before = copy.deepcopy(rows)
    result = solve_withdrawals(rows, initial_count=1, minimum_remaining=0, deadline=time.monotonic()-1)
    assert rows == before and not result.selected_train_ids
    assert result.status == "evaluation_timeout"


def test_withdrawal_exclusion_produces_next_best_distinct_subset():
    rows = [dict(train_id="a", saving_units=10), dict(train_id="b", saving_units=8),
            dict(train_id="c", saving_units=6)]
    first = solve_withdrawals(rows, initial_count=3, minimum_remaining=1, max_withdrawals=2)
    second = solve_withdrawals(rows, initial_count=3, minimum_remaining=1, max_withdrawals=2,
                               excluded_subsets=[first.selected_train_ids])
    assert first.selected_train_ids == ("a", "b")
    assert second.selected_train_ids == ("a", "c") and second.saving_units == 16
