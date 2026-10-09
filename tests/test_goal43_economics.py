import copy
import json
from decimal import Decimal

import pytest

from economics import Article
from goal43_economics import (
    compare_train_economics, maintenance_registry, revenue_sensitivity, to1_deferral,
)


FLEET = {"Y01": "yubileyny", "B01": "baltiets"}


def plan(**changes):
    row = {
        "plan_id": "KEEP", "demand_id": "demand", "initial_state_id": "state",
        "incident_id": "same-incident", "random_world_id": "seed-11",
        "period_start": "2026-09-01T05:30:00+03:00",
        "period_end": "2026-09-02T01:00:00+03:00",
        "closure_end": "2026-09-02T02:00:00+03:00",
        "coverage_complete": True, "obligations_closed": True,
        "trains": {
            "Y01": {"passenger_km": "1000", "nonpassenger_km": "10",
                    "moving_seconds": "5000", "standby_seconds": "500", "obligations": []},
            "B01": {"passenger_km": "500", "nonpassenger_km": "20",
                    "moving_seconds": "2500", "standby_seconds": "1000", "obligations": []},
        },
    }
    row.update(changes)
    return row


def reduced():
    row = plan(plan_id="WITHDRAW")
    row["trains"]["Y01"]["passenger_km"] = "0"
    row["trains"]["Y01"]["nonpassenger_km"] = "30"
    row["trains"]["B01"]["passenger_km"] = "400"
    return row


def test_fixed_fleet_return_mileage_and_type_rates():
    result = compare_train_economics(plan(), reduced(), FLEET)
    assert result["status"] == "complete"
    assert result["fixed_train_ids"] == ["B01", "Y01"]
    assert result["per_train"]["Y01"]["saving_passenger_km"] == "1000"
    assert result["per_train"]["Y01"]["saving_nonpassenger_km"] == "-20"
    assert result["per_train"]["Y01"]["saving_km"] == "980"
    assert result["per_train"]["Y01"]["to1_resource_value_rub"] == "7252"
    assert result["per_train"]["B01"]["to1_resource_value_rub"] == "920"
    assert result["totals"]["saving_km"] == "1080"
    assert result["totals"]["mean_saving_km_per_train"] == "540"
    assert result["totals"]["to1_resource_value_rub"] == "8172"
    assert result["cash_savings_rub"] is None
    assert result["net_profit_effect_rub"] is None
    assert "total_savings_rub" not in result
    json.dumps(result, allow_nan=False)


def test_train_order_and_inactive_trains_do_not_change_denominator():
    candidate = reduced()
    candidate["trains"]["B01"].update(passenger_km=0, nonpassenger_km=0)
    first = compare_train_economics(plan(), candidate, FLEET)
    second = compare_train_economics(plan(), candidate, dict(reversed(list(FLEET.items()))))
    assert first == second
    assert first["totals"]["mean_saving_km_per_train"] == "750"


def test_added_return_can_make_savings_negative():
    candidate = plan()
    candidate["trains"]["Y01"]["nonpassenger_km"] = "50"
    result = compare_train_economics(plan(), candidate, FLEET)
    assert result["totals"]["saving_km"] == "-40"
    assert result["totals"]["to1_resource_value_rub"] == "-296"


@pytest.mark.parametrize("field", ["demand_id", "initial_state_id", "incident_id", "random_world_id", "period_end", "closure_end"])
def test_different_incident_or_context_is_not_a_comparison(field):
    with pytest.raises(ValueError, match="same context"):
        compare_train_economics(plan(), plan(**{field: "different"}), FLEET)


@pytest.mark.parametrize("change", [{"coverage_complete": False}, {"obligations_closed": False}])
def test_partial_coverage_cannot_claim_final_savings(change):
    result = compare_train_economics(plan(), {**reduced(), **change}, FLEET)
    assert result["status"] == "partial"
    assert result["totals"]["observed_km_delta"] == "1080"
    assert result["totals"]["saving_km"] is None
    assert result["totals"]["to1_resource_value_rub"] is None


def test_unfinished_return_obligation_survives_short_card():
    candidate = reduced()
    candidate["trains"]["Y01"]["obligations"] = [{"kind": "return_to_depot", "remaining_km": "30"}]
    result = compare_train_economics(plan(), candidate, FLEET)
    assert result["status"] == "partial"
    assert result["per_train"]["Y01"]["candidate_obligations"][0]["remaining_km"] == "30"
    assert result["per_train"]["Y01"]["saving_km"] is None


@pytest.mark.parametrize("field", ["passenger_km", "nonpassenger_km"])
def test_missing_mileage_is_unknown_not_zero(field):
    candidate = reduced()
    del candidate["trains"]["Y01"][field]
    result = compare_train_economics(plan(), candidate, FLEET)
    assert result["status"] == "partial"
    assert result["totals"]["saving_km"] is None
    assert result["per_train"]["Y01"]["candidate_km"] is None


def test_removed_train_id_is_rejected():
    candidate = reduced()
    del candidate["trains"]["Y01"]
    with pytest.raises(ValueError, match="fixed train"):
        compare_train_economics(plan(), candidate, FLEET)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", True])
def test_invalid_mileage_rejected(value):
    candidate = plan()
    candidate["trains"]["Y01"]["passenger_km"] = value
    with pytest.raises(ValueError):
        compare_train_economics(plan(), candidate, FLEET)


def test_exact_decimal_mileage_is_not_reduced_to_float():
    baseline, candidate = plan(), plan()
    baseline["trains"]["Y01"]["passenger_km"] = "100000000000000000000.01"
    candidate["trains"]["Y01"]["passenger_km"] = "100000000000000000000"
    result = compare_train_economics(baseline, candidate, FLEET)
    assert result["per_train"]["Y01"]["saving_km"] == "0.01"
    assert result["per_train"]["Y01"]["to1_resource_value_rub"] == "0.074"


def test_registry_prices_per_train_and_annual_service_events():
    registry = maintenance_registry()
    assert registry["prices_rub_per_train"]["yubileyny"] == {
        "TO1": "111000", "TO2": "313000", "TR1": "714000", "TR2": "3181000", "TR3": "46651000",
    }
    assert registry["prices_rub_per_train"]["baltiets"]["TR3"] == "47951000"
    assert registry["annual_service_equivalents"]["yubileyny"] == {"TO1": "291", "TO2": "22", "TR1": "16"}
    assert registry["annual_service_equivalents"]["baltiets"] == {"TO1": "363", "TO2": "103", "TR1": "14"}
    assert registry["annual_volume_period"] is None
    assert registry["annual_volume_line"] is None
    assert registry["blue_line_2025"]["trips"] == 200400000
    assert registry["blue_line_2025"]["gross_revenue_rub"] == "10200360000"
    assert registry["blue_line_2025"]["scale_line1"] is False
    assert registry["repair_km_rates"] is None


def test_to1_delay_uses_known_odometer_and_future_plan():
    result = to1_deferral({
        "km_since_to1": "14000", "baseline_evaluation_km": "500",
        "candidate_evaluation_km": "250", "future_daily_km": "100",
        "at": "2026-09-02T02:00:00+03:00",
    })
    assert result["baseline_days_until_to1"] == "5"
    assert result["candidate_days_until_to1"] == "7.5"
    assert result["deferral_days"] == "2.5"
    assert result["cash_savings_rub"] is None
    assert result["permanent_cancellation"] is False
    assert result["baseline_to1_at"] == "2026-09-07T02:00:00+03:00"
    assert result["candidate_to1_at"] == "2026-09-09T14:00:00+03:00"


@pytest.mark.parametrize("row", [
    {}, {"km_since_to1": "14000"},
    {"km_since_to1": "14000", "baseline_evaluation_km": "500", "candidate_evaluation_km": "250", "future_daily_km": 0},
])
def test_unknown_or_zero_future_mileage_has_unknown_to1_date(row):
    result = to1_deferral(row)
    assert result["deferral_days"] is None
    assert result["candidate_to1_at"] is None


def test_already_crossed_to1_threshold_is_due_not_modulo_reset():
    result = to1_deferral({"km_since_to1": 14900, "baseline_evaluation_km": 200,
                           "candidate_evaluation_km": 150, "future_daily_km": 100})
    assert result["status"] == "due"
    assert result["baseline_days_until_to1"] == "0"
    assert result["candidate_days_until_to1"] == "0"


def test_paid_waiting_is_not_lost_revenue():
    result = compare_train_economics(plan(), reduced(), FLEET,
                                    revenue={"paid_waiting_passengers": 1000})
    assert result["revenue"]["lost_revenue_rub"] is None
    assert result["revenue"]["paid_waiting_revenue_loss_rub"] == "0"
    assert result["revenue"]["lost_profit_rub"] is None


def test_revenue_and_refunds_are_separate_from_resource_value():
    result = compare_train_economics(plan(), reduced(), FLEET, revenue={
        "additional_unpaid_trips": 10, "refunds_rub": "100", "loss_categories_disjoint": True,
    })
    assert result["revenue"]["foregone_trip_revenue_rub"] == "509"
    assert result["revenue"]["refunds_rub"] == "100"
    assert result["revenue"]["lost_revenue_rub"] == "609"
    assert result["revenue"]["lost_profit_rub"] is None
    assert result["totals"]["to1_resource_value_rub"] == "8172"
    assert result["net_profit_effect_rub"] is None


def test_known_contribution_margin_is_required_for_lost_profit():
    result = compare_train_economics(plan(), reduced(), FLEET, revenue={
        "additional_unpaid_trips": 10, "refunds_rub": 100,
        "loss_categories_disjoint": True, "contribution_margin_per_trip": "30",
    })
    assert result["revenue"]["lost_profit_rub"] == "400"
    assert result["revenue"]["profit_status"] == "configured_margin_only"


def test_loss_categories_must_not_double_count():
    with pytest.raises(ValueError, match="disjoint"):
        compare_train_economics(plan(), reduced(), FLEET,
                                revenue={"additional_unpaid_trips": 10, "refunds_rub": 100})
    with pytest.raises(ValueError, match="overlap"):
        compare_train_economics(plan(), reduced(), FLEET, revenue={
            "additional_unpaid_trips": 1, "refunds_rub": 50.9,
            "loss_categories_disjoint": True,
            "unpaid_loss_ids": ["passenger-1"], "refund_loss_ids": ["passenger-1"],
        })


def test_resource_and_same_to1_cash_are_not_summed_twice():
    item = Article("same_TO1", "yubileyny_train_km", "7.4", "RUB/train_km",
                   ("maintenance:TO1:yubileyny",), "cash", "fixture", "assumed")
    with pytest.raises(ValueError, match="overlap"):
        compare_train_economics(plan(), reduced(), FLEET, cash_articles=[item])


def test_incomplete_resource_cycle_cost_guard_remains_in_force():
    with pytest.raises(ValueError, match="complete cycle"):
        compare_train_economics(plan(), reduced(), FLEET, cash_articles=[{
            "name": "TR", "quantity_key": "yubileyny_train_km", "rate": "1",
            "unit": "RUB/train_km", "covered_components": ["TR"],
            "accounting_kind": "resource_accrual", "source": "fixture", "status": "assumed",
        }])


def test_sensitivity_labels_assumptions_and_fare_break_even():
    result = revenue_sensitivity({"benefit_rub": "7400", "benefit_kind": "TO1_resource_accrual",
                                  "potential_unpaid_trips": 1000, "abandonment_shares": [0, "0.01", "0.2"]})
    assert Decimal(result["break_even_additional_unpaid_trips"]) == Decimal(7400) / Decimal("50.9")
    assert result["scenarios"][1]["lost_revenue_rub"] == "509"
    assert result["scenarios"][2]["comparison_rub"] == "-2780"
    assert all(row["status"] == "assumed" for row in result["scenarios"])
    assert result["comparison_is_cash_profit"] is False


def test_prices_are_connected_to_the_versioned_policy():
    from goal43_policy import load_policy
    policy = load_policy()
    policy["parameters"]["maintenance_prices_rub_by_type"]["value"]["Yubileyny"]["TO1"] = 150000
    result = compare_train_economics(plan(), reduced(), FLEET, policy=policy)
    assert result["per_train"]["Y01"]["to1_rate_rub_per_train_km"] == "10"
    assert result["per_train"]["Y01"]["to1_resource_value_rub"] == "9800"


def test_unknown_cash_is_not_a_zero_inside_detailed_accounting():
    result = compare_train_economics(plan(), reduced(), FLEET)
    assert result["cost_comparison"]["groups"]["cash"]["savings"] is None
    assert result["economic_status"] == "partial_unknown_operating_costs"


def test_loss_recovery_can_produce_negative_lost_revenue_delta():
    result = compare_train_economics(plan(), reduced(), FLEET,
                                    revenue={"additional_unpaid_trips": -10, "refunds_rub": 0})
    assert result["revenue"]["lost_revenue_rub"] == "-509"


def test_to1_due_status_does_not_require_future_mileage():
    result = to1_deferral({"km_since_to1": 15100, "baseline_evaluation_km": 0,
                           "candidate_evaluation_km": 0, "future_daily_km": 0})
    assert result["status"] == "due"
    assert result["baseline_days_until_to1"] == "0"
    assert result["candidate_days_until_to1"] == "0"


def test_inputs_are_not_mutated():
    baseline, candidate = plan(), reduced()
    before = copy.deepcopy((baseline, candidate, FLEET))
    compare_train_economics(baseline, candidate, FLEET)
    assert (baseline, candidate, FLEET) == before
