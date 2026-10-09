import itertools

import pytest

from metro44_mpc import solve_joint_profile, forecast_section_rates


def slot(second=0, up=1, down=1, minimum=150, ready=53):
    return dict(start_second=second, end_second=second+900,
                up_rate=up, down_rate=down, up_entries=100, down_entries=100,
                minimum_headway_seconds=minimum, ready_capacity=ready)


def test_joint_frequency_and_fleet_match_independent_enumeration():
    choices = [150, 180, 240, 300]
    rows = [slot(up=4, down=2)]
    feasible = [(a, b) for a, b in itertools.product(choices, repeat=2)
                if a*4 <= 1200 and b*2 <= 1200 and 3040*(1/a+1/b) <= 53]
    optimum = min((1/a+1/b, a+b) for a, b in feasible)
    result = solve_joint_profile(rows, headway_choices=choices, half_cycle_seconds=3040)
    assert result['status'] == 'optimal'
    row = result['slots'][0]
    assert (1/row['up_headway_seconds']+1/row['down_headway_seconds'],
            row['up_headway_seconds']+row['down_headway_seconds']) == optimum
    assert row['train_count'] <= 53


def test_wait_budget_prevents_saving_km_by_arbitrary_slow_service():
    result = solve_joint_profile([slot()], headway_choices=[150, 180, 240, 480],
                                mean_wait_budget_seconds=90)
    assert result['estimated_mean_wait_seconds'] <= 90
    assert result['slots'][0]['up_headway_seconds'] == 180


def test_capacity_shortfall_is_mitigation_not_a_valid_plan():
    result = solve_joint_profile([slot(up=20)], headway_choices=[150, 180, 240])
    assert result['status'] == 'capacity_shortfall'
    assert result['slots'][0]['up_headway_seconds'] == 150
    assert not result['passenger_admissible']
    assert not result['release_authorized']


def test_reserve_unavailable_in_first_slot_is_not_teleported():
    result = solve_joint_profile([slot(ready=20), slot(900, ready=53)],
                                headway_choices=[150, 300, 480])
    assert result['slots'][0]['train_count'] <= 20
    assert all(r['train_count'] <= 53 for r in result['slots'])


def test_expired_budget_never_returns_unchecked_partial_assignment():
    import time
    result = solve_joint_profile([slot()], deadline=time.monotonic()-1)
    assert result['status'] == 'evaluation_timeout'
    assert result['slots'] == []


def test_a_later_forecast_issue_or_duplicate_quarter_is_rejected():
    bundle = dict(issued_at='2026-09-30T08:00:00+03:00',
                  rows=[dict(station_id=s, interval_start='2026-09-30T08:00:00+03:00',
                             predicted_count=90) for s in ('A', 'B', 'C')])
    with pytest.raises(ValueError, match='future'):
        forecast_section_rates(bundle, ['A', 'B', 'C'], '2026-09-30T07:00:00+03:00')
    bundle['rows'].append(bundle['rows'][0].copy())
    with pytest.raises(ValueError, match='duplicate'):
        forecast_section_rates(bundle, ['A', 'B', 'C'], '2026-09-30T08:00:00+03:00')


def test_entries_are_converted_to_directional_crossings_not_platform_stock():
    bundle = dict(issued_at='2026-09-30T08:00:00+03:00', rows=[
        dict(station_id=s, interval_start='2026-09-30T08:00:00+03:00', predicted_count=n)
        for s, n in [('A', 900), ('B', 0), ('C', 0)]])
    rows = forecast_section_rates(bundle, ['A', 'B', 'C'], '2026-09-30T08:00:00+03:00')
    assert rows[0]['up_rate'] == 1
    assert rows[0]['down_rate'] == 0
    assert rows[0]['up_entries'] == 900
    assert rows[0]['down_entries'] == 0


def test_inputs_remain_unchanged_and_no_service_value_is_invented():
    import copy
    rows = [slot()]
    before = copy.deepcopy(rows)
    result = solve_joint_profile(rows, headway_choices=[150, 240])
    assert rows == before
    assert result['cash_saving_rub'] is None
    assert result['validated_scope'] == 'aggregate_demand_frequency_fleet_only'


def test_crossing_demand_uses_each_origins_arrival_clock():
    bundle = dict(issued_at='2026-09-30T08:00:00+03:00', rows=[
        dict(station_id=s, interval_start=f'2026-09-30T08:{q*15:02}:00+03:00', predicted_count=n)
        for q, counts in enumerate(((900, 0, 0), (900, 1800, 0)))
        for s, n in zip(('A', 'B', 'C'), counts)])
    result = forecast_section_rates(bundle, ['A', 'B', 'C'], bundle['issued_at'],
        station_offsets_seconds={'up': [0, 900, 1800], 'down': [1800, 900, 0]})
    assert result[0]['up_rate'] == 1.5
    assert result[0]['spatial_shift_applied']
def test_explicit_waiting_weights_exclude_late_entries_but_keep_capacity_rates():
    from metro44_mpc import solve_joint_profile
    rows = [dict(start_second=0, end_second=900, minimum_headway_seconds=150, ready_capacity=53,
                 up_rate=0.1, down_rate=0.1, up_entries=1, down_entries=1,
                 up_wait_entries=100, down_wait_entries=100),
            dict(start_second=900, end_second=1800, minimum_headway_seconds=150, ready_capacity=53,
                 up_rate=0.1, down_rate=0.1, up_entries=1000, down_entries=1000,
                 up_wait_entries=0, down_wait_entries=0)]
    original = [{k: v for k, v in row.items() if k not in ("up_wait_entries", "down_wait_entries")} for row in rows]
    legacy = solve_joint_profile(original, mean_wait_budget_seconds=90)
    changed = solve_joint_profile(rows, mean_wait_budget_seconds=90)
    assert changed["status"] == "optimal"
    assert changed["estimated_mean_wait_seconds"] == 90
    assert changed["slots"][0]["up_headway_seconds"] == changed["slots"][0]["down_headway_seconds"] == 180
    assert changed["slots"][1]["up_headway_seconds"] == changed["slots"][1]["down_headway_seconds"] == 480
    assert legacy["slots"][0]["up_headway_seconds"] == legacy["slots"][0]["down_headway_seconds"] == 480


def test_zero_late_wait_weight_does_not_remove_late_capacity_demand():
    row = slot(up=4, down=4)
    row.update(up_wait_entries=0, down_wait_entries=0)
    result = solve_joint_profile([row], mean_wait_budget_seconds=90)
    assert result["status"] == "optimal"
    assert result["slots"][0]["up_headway_seconds"] == result["slots"][0]["down_headway_seconds"] == 300
    assert result["estimated_mean_wait_seconds"] is None


def test_explicit_wait_weights_do_not_require_unused_legacy_entry_fields():
    row = slot()
    del row["up_entries"], row["down_entries"]
    row.update(up_wait_entries=100, down_wait_entries=100)
    assert solve_joint_profile([row], mean_wait_budget_seconds=90)["estimated_mean_wait_seconds"] == 90


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan")])
def test_invalid_explicit_wait_weights_are_rejected(bad):
    row = slot()
    row.update(up_wait_entries=bad, down_wait_entries=100)
    with pytest.raises(ValueError, match="waiting cohort"):
        solve_joint_profile([row], mean_wait_budget_seconds=90)

