import copy
from time import monotonic

import pytest

from metro47_demand_optimizer import solve_demand_profile, optimize_demand_profile


def slot(start=0, *, up=5.05, down=2.71, minimum=150, ready=53):
    return dict(start_second=start, end_second=start+900, up_rate=up,
                down_rate=down, minimum_headway_seconds=minimum, ready_capacity=ready)


def test_cost_only_optimizer_selects_integer_intervals_outside_the_old_grid():
    result = solve_demand_profile([slot()], one_way_km=29.6, cost_rub_per_km=9.2)
    assert result['status'] == 'optimal'
    row = result['slots'][0]
    assert (row['up_headway_seconds'], row['down_headway_seconds']) == (237, 442)
    assert row['up_rate']*237 <= 1200 < row['up_rate']*238
    assert row['down_rate']*442 <= 1200 < row['down_rate']*443
    assert result['objective'] == 'to1_resource_cost_only'
    assert result['waiting_in_objective'] is False


def test_normal_per_car_concentration_tightens_capacity():
    result = solve_demand_profile([slot()], one_way_km=29.6,
                                  cost_rub_per_km=7.4, car_concentration=1.25)
    assert result['capacity_passengers'] == 960
    assert (result['slots'][0]['up_headway_seconds'],
            result['slots'][0]['down_headway_seconds']) == (190, 354)


def test_zero_demand_keeps_the_user_maximum_interval():
    row = solve_demand_profile([slot(up=0, down=0)], one_way_km=29.6,
                              cost_rub_per_km=7.4)['slots'][0]
    assert row['up_headway_seconds'] == row['down_headway_seconds'] == 480


def test_infeasible_demand_is_not_relaxed_into_an_overloaded_candidate():
    result = solve_demand_profile([slot(up=20)], one_way_km=29.6, cost_rub_per_km=9.2)
    assert result['status'] == 'no_feasible_candidate'
    assert result['slots'] == []
    assert result['passenger_admissible'] is False


def test_exact_joint_fleet_bound_avoids_rounding_each_direction_up_to_54():
    result = solve_demand_profile([slot(up=10.4, down=10.4, minimum=113)],
                                  one_way_km=29.6, cost_rub_per_km=7.4)
    row = result['slots'][0]
    assert row['up_headway_seconds'] == row['down_headway_seconds'] == 115
    assert row['train_count'] == 53


def test_peak_geometry_cannot_teleport_unavailable_trains():
    result = solve_demand_profile([slot(up=10.4, down=10.4, minimum=113, ready=40)],
                                  one_way_km=29.6, cost_rub_per_km=9.2)
    assert result['status'] == 'no_feasible_candidate'


def test_type_cost_changes_reported_resource_not_capacity_or_waiting_objective():
    cheap = solve_demand_profile([slot()], one_way_km=29.6, cost_rub_per_km=7.4)
    expensive = solve_demand_profile([slot()], one_way_km=29.6, cost_rub_per_km=9.2)
    assert cheap['slots'] == expensive['slots']
    assert expensive['estimated_to1_resource_rub']/cheap['estimated_to1_resource_rub'] == pytest.approx(9.2/7.4)
    assert cheap['cash_saving_rub'] is None and cheap['release_authorized'] is False


def test_completed_feasibility_feedback_can_raise_future_floor_rates():
    feedback = [dict(start_second=0, up_rate=6, down_rate=0,
        observed_as_of_second=0, source_id='physical_feedback_001', future_actual_used=False)]
    result = solve_demand_profile([slot()], one_way_km=29.6, cost_rub_per_km=7.4,
                                  floor_rates=feedback)
    assert result['slots'][0]['up_headway_seconds'] == 200
    assert result['feedback_digest'] is not None


@pytest.mark.parametrize('change', [dict(observed_as_of_second=1), dict(future_actual_used=True),
                                  dict(source_id=''), dict(up_rate=float('nan'))])
def test_unavailable_or_unprovenance_feedback_is_rejected(change):
    feedback = dict(start_second=0, up_rate=6, down_rate=0,
        observed_as_of_second=0, source_id='physical_feedback_001', future_actual_used=False)
    feedback.update(change)
    with pytest.raises(ValueError):
        solve_demand_profile([slot()], one_way_km=29.6, cost_rub_per_km=7.4,
                             floor_rates=[feedback])


def test_expired_deadline_returns_no_action():
    result = solve_demand_profile([slot()], one_way_km=29.6,
        cost_rub_per_km=7.4, deadline=monotonic()-1)
    assert result['status'] == 'evaluation_timeout' and result['slots'] == []


def test_input_snapshots_stay_exact_and_no_unrequested_quality_budget_appears():
    rows = [slot(), slot(900, up=1, down=2)]
    original = copy.deepcopy(rows)
    result = solve_demand_profile(rows, one_way_km=29.6, cost_rub_per_km=7.4)
    assert rows == original
    assert result['mean_wait_budget_seconds'] is None
    assert result['scope'] == 'forecast frequency/capacity surrogate; exact motion/passengers separate'


@pytest.mark.parametrize('rate', [-1, float('inf'), float('nan'), True])
def test_invalid_demand_is_rejected(rate):
    with pytest.raises(ValueError):
        solve_demand_profile([slot(up=rate)], one_way_km=29.6, cost_rub_per_km=7.4)


@pytest.mark.parametrize('clock', [14400, '2026-09-30T10:00:00+03:00'])
@pytest.mark.parametrize('available_slots', [5, 8])
def test_wrapper_uses_queue_stock_causally_and_clamps_to_real_future_coverage(monkeypatch, clock, available_slots):
    import metro47_demand_optimizer as module
    from goal43_policy import load_policy

    data = dict(period_start='2026-09-30T06:00:00+03:00', headway_policy=load_policy(),
        stations=[dict(station_id=s, kind='line') for s in ('A', 'B')])
    bundle = dict(issued_at='2026-09-30T10:00:00+03:00', rows=[])
    state = dict(second=14400, queues={'A': [dict(origin='A', destination='B', count=900)]})
    captured = {}
    def rates(*args, **kwargs):
        captured.update(kwargs)
        return [slot(14400+900*i, up=1+(i == 0), down=1) for i in range(available_slots)]
    monkeypatch.setattr(module, 'validate_bundle', lambda *args: {})
    monkeypatch.setattr(module, 'forecast_section_rates', rates)
    monkeypatch.setattr(module, 'station_offsets', lambda data: {'up': [0, 0], 'down': [0, 0]})
    monkeypatch.setattr(module, '_geometry', lambda data: ({}, {}, {}, 3040))
    result = optimize_demand_profile(data, bundle, state, clock, one_way_km=29.6,
                                     cost_rub_per_km=9.2, service_end_second=22200)
    assert captured['checkpoint'] is state
    assert len(result['profile']) == 9+available_slots
    assert len(result['slots']) == available_slots
    assert result['forecast_horizon_seconds'] == available_slots*900
    assert result['requested_horizon_seconds'] == 7200
    assert result['profile'][0]['terminal'] == 'A'
    assert result['profile'][1]['terminal'] == 'B'
    assert result['profile'][-2]['end_second'] == 21600
    assert result['profile'][-1] == dict(terminal='*', start_second=21600,
        end_second=22200, headway_seconds=150)
    assert result['future_actual_used'] is False
