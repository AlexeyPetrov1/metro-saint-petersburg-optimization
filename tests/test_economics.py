from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from economics import (
    Article, Evaluation, Event, Quantity, aggregate_days, break_even, compare,
    compare_scenarios, load_costs, maintenance_resources, natural_report,
    price_frontier, round_trip,
)


ROOT = Path(__file__).resolve().parents[1]


def quantity(value, unit="train_km"):
    return Quantity(value, unit, "fixture", "assumed")


def evaluation(value="59.2", **changes):
    plan = Evaluation(
        "base", "same-demand", "same-state",
        "2026-05-01T03:00:00+03:00", "2026-05-01T05:00:00+03:00",
        "2026-05-01T06:00:00+03:00",
        {"passenger_train_km": quantity(value),
         "paid_person_hours": quantity("8", "person_h")},
        obligations_closed=True,
    )
    return replace(plan, **changes)


def article(rate="10", **changes):
    item = Article("traction", "passenger_train_km", rate, "RUB/train_km",
                   ("traction",), "cash", "fixture", "assumed")
    return replace(item, **changes)


def test_full_round_trip_counts_eight_cars_once():
    q = round_trip("29.6", 8, "99", "fixture")
    assert q["passenger_train_km"].amount == Decimal("59.2")
    assert q["passenger_car_km"].amount == Decimal("473.6")
    assert q["moving_hours"].amount == Decimal("1.65")
    plans = [replace(evaluation(), quantities=q),
             replace(evaluation(), quantities=round_trip("0", 8, "0", "fixture"))]
    out = compare(*plans, [article(unit="RUB/car_km", quantity_key="passenger_car_km")])
    assert out["groups"]["cash"]["savings"] == "4736"


def test_equal_plans_zero_each_delta_even_with_unknown_rate():
    out = compare(evaluation(), evaluation(), [article(None)])
    assert out["articles"][0]["savings"] == "0"
    assert all(row["delta"] == "0" for row in out["natural_deltas"].values())
    assert out["groups"]["cash"]["baseline"] is None
    assert out["status"] == "partial"


def test_negative_savings_is_preserved():
    out = compare(evaluation("1"), evaluation("2"), [article()])
    assert out["groups"]["cash"]["savings"] == "-10"
    assert out["groups"]["cash"]["savings_pct"] == "-100"


def test_unknown_rate_nonzero_is_partial_and_known_zero_is_zero():
    out = compare(evaluation("10"), evaluation("0"), [article(None)])
    row = out["articles"][0]
    assert row["baseline"]["amount"] is None
    assert row["candidate"]["amount"] == "0"
    assert row["savings"] is None
    assert row["status"] == "partial"


def test_missing_quantity_is_unknown_not_zero():
    out = compare(evaluation(), replace(evaluation(), quantities={}), [article()])
    assert out["articles"][0]["candidate"]["amount"] is None
    assert out["natural_deltas"]["passenger_train_km"]["delta"] is None


def test_zero_baseline_has_no_percentage():
    assert compare(evaluation("0"), evaluation("1"), [article()])["groups"]["cash"]["savings_pct"] is None


@pytest.mark.parametrize("unit", ["kRUB/train_km", "RUB/car_km"])
def test_silent_currency_or_quantity_unit_change_is_rejected(unit):
    with pytest.raises(ValueError, match="unit"):
        compare(evaluation(), evaluation(), [article(unit=unit)])


def test_overlapping_cash_components_are_rejected():
    bundled = article(name="bundled", covered_components=("traction", "maintenance"))
    with pytest.raises(ValueError, match="overlap"):
        compare(evaluation(), evaluation(), [article(), bundled])


def test_accounting_groups_remain_separate():
    capital = article(name="equipment", accounting_kind="capital", covered_components=("equipment",))
    out = compare(evaluation("2"), evaluation("1"), [article(), capital])
    assert out["groups"]["cash"]["savings"] == "10"
    assert out["groups"]["capital"]["savings"] == "10"
    assert "total_savings" not in out
    with pytest.raises(ValueError, match="cycle"):
        compare(evaluation(), evaluation(), [article(accounting_kind="resource_accrual")])


def test_maneuver_and_return_after_120_minutes_reduce_savings():
    rates = [article(), article(name="maneuver", quantity_key="nonpassenger_train_km",
                               covered_components=("maneuver_traction",))]
    base = replace(evaluation(), quantities={"passenger_train_km": quantity("59.2"),
                                            "nonpassenger_train_km": quantity("0")})
    candidate = replace(base, plan_id="candidate", quantities={
        "passenger_train_km": quantity("0"), "nonpassenger_train_km": quantity("0")})
    plain = compare(base, candidate, rates)["groups"]["cash"]["savings"]
    tail = Event("return", "2026-05-01T05:30:00+03:00",
                 {"nonpassenger_train_km": quantity("5")}, "fixture")
    candidate = replace(candidate, events=(tail,))
    out = compare(base, candidate, rates)
    assert plain == "592"
    assert out["groups"]["cash"]["savings"] == "542"
    assert out["boundary"]["closure_end"] == "2026-05-01T06:00:00+03:00"
    assert out["events"]["candidate"][0]["event_id"] == "return"


def test_same_paid_shift_gives_zero_labor_savings():
    wage = article(name="labor", quantity_key="paid_person_hours", unit="RUB/person_h",
                   covered_components=("paid_labor",))
    out = compare(evaluation(), evaluation("0"), [wage])
    assert out["groups"]["cash"]["savings"] == "0"


@pytest.mark.parametrize("changes", [
    {"demand_id": "different"}, {"initial_state_id": "different"},
    {"closure_end": "2026-05-01T07:00:00+03:00"},
    {"model_version": "different"}, {"obligations_closed": False},
    {"obligations_closed": None},
])
def test_comparisons_require_same_context_and_closed_obligations(changes):
    with pytest.raises(ValueError):
        compare(evaluation(), replace(evaluation(), **changes), [article()])


def test_unknown_operational_status_does_not_become_passed():
    out = compare(evaluation(), evaluation("0"), [article()])
    assert out["operational_status"] == {"baseline": "unknown", "candidate": "unknown"}
    assert out["evaluation_mode"] == "fixture"
    assert out["release_authorized"] is False


@pytest.mark.parametrize("running,standby", [("10", "10"), ("5", "10")])
def test_no_finite_break_even_when_running_not_more_expensive(running, standby):
    assert break_even("100", running, standby, "fixture")["threshold_hours"] is None


def test_break_even_and_sign_change_frontier():
    assert break_even("100", "30", "10", "fixture")["threshold_hours"] == "5"
    out = price_frontier("10", "-100", "RUB/train_km", "fixture")
    assert out["rate"] == "10"
    assert out["profitable_when"] == "rate > boundary"
    base, candidate = evaluation("10"), evaluation("0")
    cases = [dict(name="low", baseline=base, candidate=candidate, articles=[article("5")], delay_minutes="0"),
             dict(name="high", baseline=base, candidate=candidate, articles=[article("15")], delay_minutes="30")]
    out = compare_scenarios(cases)
    assert [x["result"]["groups"]["cash"]["savings"] for x in out["scenarios"]] == ["50", "150"]


def test_days_require_whole_nonoverlapping_days():
    day = replace(evaluation(), period_end="2026-05-02T03:00:00+03:00",
                  closure_end="2026-05-02T03:00:00+03:00")
    result = compare(day, replace(day, quantities={"passenger_train_km": quantity("0")}), [article()])
    out = aggregate_days([result])
    assert out["days"] == 1 and out["groups"]["cash"]["savings"] == "592"
    with pytest.raises(ValueError, match="overlap"):
        aggregate_days([result, result])
    with pytest.raises(ValueError, match="whole"):
        aggregate_days([compare(evaluation(), evaluation(), [article()])])


def test_photo_checked_resource_per_wagon_does_not_invent_history_or_rubles():
    rules = ROOT / "data/processed/metro/maintenance_rules.csv"
    wagons = [{"wagon_id": "B1", "series": "81-725/726/727", "planned_km": "100",
               "since_maintenance_km": {"ТО-2": "29000"}, "source": "fixture"}]
    out = maintenance_resources(rules, wagons, ROOT / "docs/reference/maintenance_photo_review.json")
    assert len(out["rows"]) == 7
    to2 = next(r for r in out["rows"] if r["maintenance_type"] == "ТО-2")
    assert to2["remaining_nominal_km"] == "5900"
    assert to2["lower_km"] == "30000" and to2["upper_km"] == "43000"
    assert to2["source_verification"] == "visual_review_chat_attachment"
    assert to2["regulation_applicability"] == "unknown"
    to1 = next(r for r in out["rows"] if r["maintenance_type"] == "ТО-1")
    assert to1["remaining_nominal_km"] is None
    assert "салонных воздушных фильтров" in to1["note"]
    assert all(r["monetary_amount"] is None for r in out["rows"])
    assert out["eo"]["81-725/726/727"]["inspection_validity_h"] == 20
    assert out["eo"]["81-725/726/727"]["depot_time_accounting"] == "unknown"
    assert out["release_authorized"] is False


def test_real_mo1_natural_report_preserves_units_and_discrepancies():
    out = natural_report(ROOT / "data/processed/metro/transport_work.csv")
    assert len(out["periods"]) == 3
    annual = out["periods"][0]
    assert annual["quantities"]["passenger_car_km"]["amount"] == "65459168"
    assert annual["quantities"]["nonpassenger_car_km"]["amount"] == "3049454.71"
    assert annual["checks"]["car_km_total_delta"] == "0"
    assert annual["checks"]["train_hours_delta"] == "0.03"
    assert annual["quantities"]["nonpassenger_train_km"]["amount"] is None
    assert annual["cash_amount"] is None
    assert out["evaluation_mode"] == "observed_aggregate"


def test_default_costs_are_unknown_not_fixture_prices():
    rates = load_costs(ROOT / "config/costs/default.json")
    assert rates and all(a.rate is None for a in rates)


def test_report_fixtures_have_actual_sign_change_and_tail_costs():
    import json
    from economics import prepare_reports
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        prepare_reports(ROOT, Path(folder))
        result = json.loads((Path(folder) / "fixture_scenarios.json").read_text(encoding="utf-8"))
        assert result["sign_change"] is True
        assert {r["rate"] for r in result["frontiers"]} == {"25", "75"}
        assert all(r["period"]["start"] == "2026-05-01T03:00:00+03:00" for r in result["frontiers"])
        assert all(r["result"]["evaluation_mode"] == "fixture" for r in result["scenarios"])
        for row in result["scenarios"]:
            labor = next(a for a in row["result"]["articles"] if a["name"] == "paid_labor")
            assert labor["savings"] == "0"
        days = json.loads((Path(folder) / "fixture_days.json").read_text(encoding="utf-8"))
        assert days["days"] == 2 and days["annualized"] is False
        resources = json.loads((Path(folder) / "maintenance_resources.json").read_text(encoding="utf-8"))
        assert len(resources["rules"]) == 14 and resources["rows"] == []


def test_prepare_rejects_tampered_metro_source(tmp_path):
    import shutil
    from economics import prepare_reports

    for folder in ["data/processed/metro", "docs/reference", "config", "examples", "metro_data"]:
        shutil.copytree(ROOT / folder, tmp_path / folder)
    for name in ["requirements.txt", "economics.py"]:
        shutil.copyfile(ROOT / name, tmp_path / name)
    transport = tmp_path / "data/processed/metro/transport_work.csv"
    transport.write_bytes(transport.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        prepare_reports(tmp_path, tmp_path / "results")


def test_days_keep_versions_and_itemized_sources():
    day = replace(evaluation(), period_end="2026-05-02T03:00:00+03:00",
                  closure_end="2026-05-02T03:00:00+03:00")
    result = compare(day, day, [article()])
    assert aggregate_days([result])["day_results"][0]["articles"][0]["baseline"]["rate_source"] == "fixture"
    changed = {**result, "model_version": "other", "boundary": {
        **result["boundary"], "period_start": "2026-05-02T03:00:00+03:00",
        "period_end": "2026-05-03T03:00:00+03:00", "closure_end": "2026-05-03T03:00:00+03:00"}}
    with pytest.raises(ValueError, match="version"):
        aggregate_days([result, changed])


def test_sensitivity_does_not_mix_demand_contexts():
    base = evaluation()
    other = replace(base, demand_id="another-demand")
    cases = [dict(name="first", baseline=base, candidate=base, articles=[article()], delay_minutes="0"),
             dict(name="other", baseline=other, candidate=other, articles=[article()], delay_minutes="0")]
    with pytest.raises(ValueError, match="context"):
        compare_scenarios(cases)


@pytest.mark.parametrize("delta,offset,expected", [("1", "5", "all rates"), ("-1", "-5", "no rates")])
def test_price_frontier_handles_boundary_outside_nonnegative_rate_domain(delta, offset, expected):
    result = price_frontier(delta, offset, "RUB/train_km", "fixture")
    assert result["rate"] is None and result["profitable_when"] == expected


def test_empty_history_does_not_skip_photo_registry_verification(tmp_path):
    text = (ROOT / "data/processed/metro/maintenance_rules.csv").read_text(encoding="utf-8")
    bad = tmp_path / "rules.csv"
    bad.write_text(text.replace(",35000,30000,43000,", ",35000,30000,45000,"), encoding="utf-8")
    with pytest.raises(ValueError, match="photo"):
        maintenance_resources(bad, [], ROOT / "docs/reference/maintenance_photo_review.json")


def test_economic_manifest_detects_modified_result(tmp_path):
    from economics import check_reports, prepare_reports

    prepare_reports(ROOT, tmp_path)
    assert check_reports(ROOT, tmp_path)["files"] == 8
    result = tmp_path / "natural_transport.json"
    result.write_bytes(result.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Output hash"):
        check_reports(ROOT, tmp_path)


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity"])
def test_invalid_quantities_are_rejected(amount):
    with pytest.raises(ValueError):
        quantity(amount)
