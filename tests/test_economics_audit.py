"""Independent expected values: literal hand calculations, never target helpers."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from decimal import localcontext
from pathlib import Path

import pytest

from economics import (
    Article, Evaluation, Event, Quantity, aggregate_days, compare, compare_scenarios,
    evaluation_from_dict, load_costs, maintenance_resources, prepare_reports, round_trip,
    natural_report,
)

ROOT = Path(__file__).resolve().parents[1]


def q(value, unit="kWh", **kwargs):
    return Quantity(value, unit, "independent fixture", "assumed", **kwargs)


def plan(value="1000", **kwargs):
    result = Evaluation("baseline", "audit-demand", "audit-state",
                        "2026-05-01T03:00:00+03:00", "2026-05-01T05:00:00+03:00",
                        "2026-05-01T06:00:00+03:00", {"energy": q(value)},
                        obligations_closed=True)
    return replace(result, **kwargs)


def rate(value="7", **kwargs):
    result = Article("energy", "energy", value, "RUB/kWh", ("traction",),
                     "cash", "independent fixture", "assumed")
    return replace(result, **kwargs)


def money(result):
    return result["groups"]["cash"]


def test_A_energy_and_two_disjoint_fees():
    base, candidate = plan(), plan("850")
    result = compare(base, candidate, [rate()])
    assert [money(result)[key] for key in ("baseline", "candidate", "savings")] == ["7000", "5950", "1050"]
    base = replace(base, quantities={**base.quantities, "fee1": q("0", "event"), "fee2": q("0", "event")})
    candidate = replace(candidate, quantities={**candidate.quantities, "fee1": q("1", "event"), "fee2": q("1", "event")})
    fees = [rate("100", name="fee1", quantity_key="fee1", unit="RUB/event", covered_components=("fee1",)),
            rate("200", name="fee2", quantity_key="fee2", unit="RUB/event", covered_components=("fee2",))]
    result = compare(base, candidate, [rate(), *fees])
    assert [money(result)[key] for key in ("baseline", "candidate", "savings")] == ["7000", "6250", "750"]


def test_B_tariff_profile_is_explicitly_rejected(tmp_path):
    # Manual oracle: 75*6+25*10=700; 50*6+50*10=800; saving=-100.
    config = {"currency": "RUB", "articles": [{**asdict(rate()),
              "tariff_periods": [{"rate": "6"}, {"rate": "10"}]}]}
    path = tmp_path / "tariffs.json"
    path.write_text(json.dumps(config, default=str), encoding="utf-8")
    with pytest.raises(ValueError, match="tariff_periods"):
        load_costs(path)


def test_C_explicit_car_km_and_trip_both_cost_160():
    trip = round_trip("5", 8, "10", "independent fixture")
    assert trip["passenger_train_km"].amount == 10
    assert trip["passenger_car_km"].amount == 80
    a = rate("2", quantity_key="passenger_car_km", unit="RUB/car_km")
    for quantities in (trip, {"passenger_car_km": q("80", "car_km")}):
        out = compare(plan(quantities=quantities), plan(quantities={"passenger_car_km": q("0", "car_km")}), [a])
        assert money(out)["baseline"] == "160"


def test_sign_symmetry_equal_plans_and_undefined_percentage():
    base, candidate = plan(), plan("850")
    assert money(compare(candidate, base, [rate()]))["savings"] == "-1050"
    assert money(compare(base, base, [rate()]))["savings"] == "0"
    assert money(compare(plan("0"), candidate, [rate()]))["savings_pct"] is None
    assert money(compare(base, candidate, [rate(None)]))["savings_pct"] is None


def test_decimal_totals_are_independent_of_display_precision_and_order():
    with localcontext() as context:
        context.prec = 6
        out = compare(plan("10000.01"), plan("10000"), [rate()])
    assert [money(out)[key] for key in ("baseline", "candidate", "savings")] == ["70000.07", "70000", "0.07"]
    base = plan("10000000000000000000000000000", quantities={
        "energy": q("10000000000000000000000000000"), "fee": q("1", "event")})
    candidate = plan(quantities={"energy": q("0"), "fee": q("0", "event")})
    rates = [rate("1"), rate("1", name="fee", quantity_key="fee", unit="RUB/event", covered_components=("fee",))]
    for articles in (rates, rates[::-1]):
        assert money(compare(base, candidate, articles))["baseline"] == "10000000000000000000000000001"


def test_unknown_missing_zero_and_not_applicable_are_distinct():
    unknown_rate = compare(plan(), plan("850"), [rate(None)])
    assert money(unknown_rate)["savings"] is None
    assert unknown_rate["articles"][0]["baseline"]["reason"] == "unknown_rate"
    unknown_qty = compare(plan(), plan(None), [rate()])
    assert unknown_qty["articles"][0]["candidate"]["reason"] == "unknown_quantity"
    zero = compare(plan(), plan("0"), [rate(None)])
    assert zero["articles"][0]["candidate"]["amount"] == "0"
    na = plan(quantities={"energy": q(None, applicability="not_applicable")})
    out = compare(na, na, [rate(None)])
    assert out["articles"][0]["baseline"]["reason"] == "not_applicable"
    assert money(out)["baseline"] == "0"
    assert out["natural_deltas"]["energy"]["baseline"]["applicability"] == "not_applicable"


def test_missing_required_cash_article_blocks_complete_saving():
    base, candidate = plan(required_cash_articles=("energy", "return")), plan("850", required_cash_articles=("energy", "return"))
    out = compare(base, candidate, [rate()])
    assert out["missing_required_articles"] == ["return"]
    assert money(out)["known_subtotal_baseline"] == "7000"
    assert money(out)["savings"] is None and out["status"] == "partial"


def test_equal_partial_observations_do_not_cancel_unknown_remainders():
    partial = Quantity("100", "kWh", "known portion only", "partial")
    out = compare(plan(quantities={"energy": partial}), plan(quantities={"energy": partial}), [rate()])
    assert money(out)["baseline"] is None and money(out)["savings"] is None
    assert money(out)["known_subtotal_baseline"] == "700"
    out = compare(plan(quantities={"energy": partial}), plan(quantities={"energy": partial}), [rate(None)])
    assert out["articles"][0]["savings"] is None


@pytest.mark.parametrize("components", [("traction",), ("traction", "maneuver"), ("traction", "labor")])
def test_overlapping_energy_model_maneuver_or_operation_cost_is_rejected(components):
    bundled = rate(name="bundled", covered_components=components)
    with pytest.raises(ValueError, match="overlap"):
        compare(plan(), plan(), [rate(), bundled])


def test_cash_maintenance_cannot_be_repeated_as_priced_resource():
    cash = rate(covered_components=("maintenance",))
    accrual = rate(name="wear", covered_components=("maintenance",), accounting_kind="resource_accrual", cycle_cost_complete=True)
    with pytest.raises(ValueError, match="overlap"):
        compare(plan(), plan(), [cash, accrual])


def test_vat_bases_cannot_be_mixed():
    fees = rate(name="fee", covered_components=("fee",), vat_basis="excluded")
    with pytest.raises(ValueError, match="VAT"):
        compare(plan(), plan(), [rate(vat_basis="included"), fees])


def test_undeclared_currency_scale_is_not_ignored(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"currency": "RUB", "currency_scale": 1000, "articles": [asdict(rate())]}, default=str), encoding="utf-8")
    with pytest.raises(ValueError, match="currency_scale"):
        load_costs(path)


def test_minutes_are_not_hours_and_salary_tracks_paid_shift():
    hourly = rate("10", quantity_key="time", unit="RUB/train_h")
    with pytest.raises(ValueError, match="unit"):
        compare(plan(quantities={"time": q("60", "min")}), plan(quantities={"time": q("0", "min")}), [hourly])
    a = rate("500", name="paid_labor", quantity_key="paid", unit="RUB/person_h", covered_components=("paid_labor",))
    base = plan(quantities={"paid": q("8", "person_h"), "moving": q("2", "train_h")})
    candidate = plan(quantities={"paid": q("8", "person_h"), "moving": q("0", "train_h")})
    assert money(compare(base, candidate, [a]))["savings"] == "0"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1"])
def test_invalid_physical_quantities_and_rates_are_rejected(value):
    with pytest.raises(ValueError):
        q(value)
    with pytest.raises(ValueError):
        rate(value)


def return_case():
    other = q(None, "event", applicability="not_applicable")
    required = ("running", "return", "other")
    base = plan(quantities={"running": q("10", "train_h"), "return": q("0", "event"), "other": other}, required_cash_articles=required)
    candidate = plan(quantities={"running": q("8.5", "train_h"), "return": q("0", "event"), "other": other}, required_cash_articles=required,
                     events=(Event("return", "2026-05-01T05:30:00+03:00", {"return": q("1", "event")}, "fixture"),))
    rates = [rate("1000", name="running", quantity_key="running", unit="RUB/train_h", covered_components=("main_operations",)),
             rate("2000", name="return", quantity_key="return", unit="RUB/event", covered_components=("return",)),
             rate(None, name="other", quantity_key="other", unit="RUB/event", covered_components=("all_other_costs",))]
    return base, candidate, rates


def test_return_changes_1500_window_saving_to_minus_500():
    base, candidate, rates = return_case()
    out = compare(base, candidate, rates)
    assert [money(out)[key] for key in ("baseline", "candidate", "savings")] == ["10000", "10500", "-500"]
    base = replace(base, events=(Event("base-tail", "2026-05-01T05:45:00+03:00", {"running": q("1", "train_h")}, "fixture"),))
    assert money(compare(base, candidate, rates))["savings"] == "500"
    rates[1] = replace(rates[1], rate=None, status="unknown")
    assert money(compare(base, candidate, rates))["savings"] is None


def test_return_after_midnight_and_incompatible_boundaries():
    base, candidate, rates = return_case()
    dates = dict(period_start="2026-05-01T23:00:00+03:00", period_end="2026-05-02T01:00:00+03:00", closure_end="2026-05-02T02:00:00+03:00")
    base, candidate = replace(base, **dates), replace(candidate, **dates)
    candidate = replace(candidate, events=(replace(candidate.events[0], at="2026-05-02T01:30:00+03:00"),))
    assert money(compare(base, candidate, rates))["savings"] == "-500"
    with pytest.raises(ValueError):
        compare(base, replace(candidate, closure_end="2026-05-02T03:00:00+03:00"), rates)


def test_missing_closure_evidence_is_not_defaulted_to_true():
    payload = asdict(plan())
    payload.pop("obligations_closed")
    candidate = evaluation_from_dict(payload)
    with pytest.raises(ValueError, match="obligations"):
        compare(plan(), candidate, [rate()])


def test_duplicate_return_identity_is_detected_even_with_different_event_labels():
    base, candidate, rates = return_case()
    event = replace(candidate.events[0], operation_id="actual-return-1")
    duplicate = replace(event, event_id="return-copy")
    with pytest.raises(ValueError, match="unique"):
        compare(base, replace(candidate, events=(event, duplicate)), rates)


def scenarios():
    return [dict(name="low", baseline=plan("10"), candidate=plan("0"), articles=[rate("5")], delay_minutes="0"),
            dict(name="high", baseline=plan("10"), candidate=plan("0"), articles=[rate("20")], delay_minutes="30")]


def test_scenarios_are_isolated_order_independent_and_have_no_unweighted_expectation():
    cases = scenarios()
    before = copy.deepcopy(cases)
    first, second = compare_scenarios(cases), compare_scenarios(cases[::-1])
    assert {r["name"]: money(r["result"])["savings"] for r in first["scenarios"]} == {"low": "50", "high": "200"}
    assert {r["name"]: r["result"] for r in first["scenarios"]} == {r["name"]: r["result"] for r in second["scenarios"]}
    assert cases == before and "expected_savings" not in first
    assert first["cash_savings_range"]["min"] == "50" and first["cash_savings_range"]["max"] == "200"
    cases[0]["articles"] = [rate("1")]
    assert money(compare_scenarios(cases)["scenarios"][1]["result"])["savings"] == "200"
    assert first["evaluation_mode"] == "fixture"


@pytest.mark.parametrize("probabilities", [("-0.1", "1.1"), ("0.2", "0.2"), ("NaN", "1"), ("1", None)])
def test_invalid_or_incomplete_scenario_probabilities_are_rejected(probabilities):
    cases = scenarios()
    for case, probability in zip(cases, probabilities):
        if probability is not None:
            case["probability"] = probability
    with pytest.raises(ValueError, match="probabilit"):
        compare_scenarios(cases)


def test_explicit_probabilities_and_unknown_scenario_do_not_create_false_expectation():
    cases = scenarios()
    cases[0]["probability"], cases[1]["probability"] = "0.25", "0.75"
    out = compare_scenarios(cases)
    assert out["expected_savings"]["amount"] == "162.5"
    cases[1]["articles"] = [rate(None)]
    out = compare_scenarios(cases)
    assert out["expected_savings"]["amount"] is None
    assert out["cash_savings_range"]["min"] is None


def whole_day(value, day=1):
    return plan(value, period_start=f"2026-05-0{day}T03:00:00+03:00", period_end=f"2026-05-0{day+1}T03:00:00+03:00",
                closure_end=f"2026-05-0{day+1}T03:00:00+03:00")


def test_disjoint_parts_match_direct_day_and_windows_are_not_days():
    morning = dict(period_start="2026-05-01T03:00:00+03:00", period_end="2026-05-01T15:00:00+03:00", closure_end="2026-05-01T15:00:00+03:00")
    evening = dict(period_start="2026-05-01T15:00:00+03:00", period_end="2026-05-02T03:00:00+03:00", closure_end="2026-05-02T03:00:00+03:00")
    parts = [compare(plan("10", **morning), plan("8", **morning), [rate()]),
             compare(plan("20", **evening), plan("15", **evening), [rate()])]
    assert sum(int(money(p)["savings"]) for p in parts) == 49
    direct = compare(whole_day("30"), whole_day("23"), [rate()])
    out = aggregate_days([direct])
    assert [money(direct)[key] for key in ("baseline", "candidate", "savings")] == ["210", "161", "49"]
    assert out["evaluation_mode"] == "fixture" and out["annualized"] is False
    with pytest.raises(ValueError, match="whole"):
        aggregate_days(parts)
    with pytest.raises(ValueError, match="overlap"):
        aggregate_days([direct, direct])


def test_boundary_event_is_counted_once_and_timezones_are_consistent():
    a = rate("2000", quantity_key="return", unit="RUB/event")
    event = Event("return", "2026-05-02T03:00:00+03:00", {"return": q("1", "event")}, "fixture")
    days = []
    for day in (1, 2):
        base = replace(whole_day("0", day), quantities={"return": q("0", "event")})
        candidate = replace(base, events=(event,) if day == 2 else ())
        days.append(compare(base, candidate, [a]))
    assert money(aggregate_days(days))["savings"] == "-2000"
    with pytest.raises(ValueError, match="outside"):
        compare(whole_day("0"), replace(whole_day("0"), events=(event,)), [rate()])
    event = replace(event, at="2026-05-02T01:00:00+00:00")
    with pytest.raises(ValueError, match="timezone"):
        compare(base, replace(base, events=(event,)), [a])


def test_day_aggregation_requires_one_cost_scope_and_no_output_aliasing():
    first = compare(whole_day("10"), whole_day("0"), [rate()])
    second = compare(whole_day("10", 2), whole_day("0", 2), [rate(covered_components=("other_cost",))])
    with pytest.raises(ValueError, match="scope"):
        aggregate_days([first, second])


def test_day_summary_does_not_alias_input_results():
    first = compare(whole_day("10"), whole_day("0"), [rate()])
    out = aggregate_days([first])
    out["day_results"][0]["articles"][0]["savings"] = "bad"
    assert first["articles"][0]["savings"] == "70"


def test_all_photo_rules_eo_and_history_remain_scoped():
    wagons = [{"wagon_id": "B1", "series": "81-725/726/727", "planned_km": "100", "odometer_km": "1000000", "source": "fixture"}]
    out = maintenance_resources(ROOT / "data/processed/metro/maintenance_rules.csv", wagons,
                                ROOT / "docs/reference/maintenance_photo_review.json")
    assert len(out["rules"]) == 14 and out["regulation_applicability"] == "unknown"
    assert out["regulation_date"] is None and out["valid_from"] is None
    assert all(r["remaining_nominal_km"] is None and r["monetary_amount"] is None for r in out["rows"])
    to1 = next(r for r in out["rows"] if r["maintenance_type"] == "ТО-1")
    assert "только замена салонных воздушных фильтров" in to1["note"]
    for work in ("ПДР-1", "ПДР-2", "КР"):
        assert "осмотру и наладке" in next(r for r in out["rows"] if r["maintenance_type"] == work)["note"]
    eo = out["eo"]["81-725/726/727"]
    assert (eo["frequency_per_day"], eo["inspection_validity_h"], eo["tolerance_h"]) == (1, 20, 4)
    assert eo["clock_start"] == "С момента выдачи состава на линию с площадки электродепо"
    assert eo["upper_is_operating_target"] is False and eo["depot_time_accounting"] == "unknown"
    assert "81-722/723/724" not in out["eo"] and out["release_authorized"] is False


def cli(*args):
    return subprocess.run([sys.executable, "-m", "economics", *map(str, args)], cwd=ROOT,
                          env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8")


def test_failed_cli_replaces_stale_success_with_explicit_failure(tmp_path):
    inputs = [tmp_path / name for name in ("base.json", "candidate.json", "prices.json")]
    for path, data in zip(inputs, (asdict(plan()), asdict(plan("850")), {"currency": "RUB", "articles": [asdict(rate())]})):
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    output = tmp_path / "result.json"
    args = ("compare", "--baseline", inputs[0], "--candidate", inputs[1], "--costs", inputs[2], "--output", output)
    assert cli(*args).returncode == 0 and money(json.loads(output.read_text()))["savings"] == "1050"
    data = json.loads(inputs[1].read_text())
    data["quantities"]["energy"]["amount"] = "NaN"
    inputs[1].write_text(json.dumps(data), encoding="utf-8")
    before = [path.read_bytes() for path in inputs]
    failed = cli(*args)
    assert failed.returncode != 0 and "finite" in failed.stderr and "Traceback" not in failed.stderr
    assert json.loads(output.read_text())["status"] == "error"
    assert [path.read_bytes() for path in inputs] == before


def test_fresh_output_reproducibility_and_human_table_agree(tmp_path):
    folders = [tmp_path / "first", tmp_path / "repeat"]
    assert all(not p.exists() for p in folders)
    for folder in folders:
        assert cli("prepare", "--output", folder).returncode == 0
        assert cli("check", "--output", folder).returncode == 0
    assert {p.name for p in folders[0].iterdir()} == {p.name for p in folders[1].iterdir()}
    assert all(p.read_bytes() == (folders[1] / p.name).read_bytes() for p in folders[0].iterdir())
    table = (folders[0] / "summary.md").read_text(encoding="utf-8")
    data = json.loads((folders[0] / "fixture_scenarios.json").read_text(encoding="utf-8"))
    expected = {"low_price": ("4592", "5480", "-888"), "higher_price": ("6960", "5480", "1480"),
                "extra_delay_and_maneuver": ("6960", "8440", "-1480")}
    for row in data["scenarios"]:
        b, c, s = expected[row["name"]]
        assert [money(row["result"])[key] for key in ("baseline", "candidate", "savings")] == [b, c, s]
        assert f"| {row['name']} | {b} | {c} | {s} | assumed | fixture |" in table


def test_failed_rebuild_invalidates_previous_manifest(tmp_path):
    import shutil
    for name in ("transport_work.csv", "maintenance_rules.csv", "quality_report.json", "build_manifest.json"):
        dest = tmp_path / "data/processed/metro" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "data/processed/metro" / name, dest)
    for name in ("docs/reference/maintenance_photo_review.json", "examples/economics/withdrawal.json",
                 "config/costs/default.json", "economics.py", "requirements.txt"):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    output = tmp_path / "output"
    prepare_reports(tmp_path, output)
    path = tmp_path / "examples/economics/withdrawal.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    fixture["scenarios"][0]["articles"][0]["rate"] = "NaN"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_reports(tmp_path, output)
    assert json.loads((output / "build_manifest.json").read_text())["status"] == "error"


def test_cli_output_cannot_overwrite_an_input(tmp_path):
    baseline, candidate, prices = [tmp_path / name for name in ("base.json", "candidate.json", "costs.json")]
    for path, data in ((baseline, asdict(plan())), (candidate, asdict(plan("850"))),
                       (prices, {"currency": "RUB", "articles": [asdict(rate())]})):
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    before = baseline.read_bytes()
    result = cli("compare", "--baseline", baseline, "--candidate", candidate, "--costs", prices, "--output", baseline)
    assert result.returncode != 0 and "input" in result.stderr
    assert baseline.read_bytes() == before


def test_report_output_cannot_invalidate_source_manifest(tmp_path):
    folder = tmp_path / "data/processed/metro"
    folder.mkdir(parents=True)
    manifest = folder / "build_manifest.json"
    manifest.write_text('{"protected_source":true}', encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises((ValueError, KeyError, OSError)):
        prepare_reports(tmp_path, folder)
    assert manifest.read_bytes() == before


def test_scenarios_with_different_accounting_scope_are_not_a_price_range():
    cases = scenarios()
    cases[1]["articles"] = [rate("20", covered_components=("other_expense",))]
    with pytest.raises(ValueError, match="scope"):
        compare_scenarios(cases)


def test_same_global_return_operation_cannot_be_charged_in_two_days():
    results = []
    a = rate("2000", quantity_key="return", unit="RUB/event")
    for day in (1, 2):
        base = replace(whole_day("0", day), quantities={"return": q("0", "event")})
        event = Event(f"return-day-{day}", f"2026-05-0{day}T04:00:00+03:00", {"return": q("1", "event")},
                      "fixture", operation_id="same-return")
        results.append(compare(base, replace(base, events=(event,)), [a]))
    with pytest.raises(ValueError, match="operation"):
        aggregate_days(results)


def test_resource_cycle_complete_requires_boolean_true():
    with pytest.raises(ValueError, match="cycle"):
        rate(accounting_kind="resource_accrual", cycle_cost_complete="false")


def test_partial_maintenance_history_cannot_produce_a_full_balance():
    wagons = [{"wagon_id": "B1", "series": "81-725/726/727", "planned_km": "100", "status": "partial",
               "since_maintenance_km": {"ТО-1": "9000"}, "source": "incomplete history"}]
    out = maintenance_resources(ROOT / "data/processed/metro/maintenance_rules.csv", wagons,
                                ROOT / "docs/reference/maintenance_photo_review.json")
    assert all(row["remaining_nominal_km"] is None for row in out["rows"])
    assert all(row["planned_resource_fraction"] is None for row in out["rows"])


def test_source_reconciliation_is_independent_of_decimal_display_context():
    with localcontext() as context:
        context.prec = 6
        report = natural_report(ROOT / "data/processed/metro/transport_work.csv")
    assert [p["checks"]["train_hours_delta"] for p in report["periods"]] == ["0.03", "0.02", "-0.02"]
    assert [p["checks"]["reported_train_km_x8_minus_passenger_car_km"] for p in report["periods"]] == ["532.8", "0", "12.2"]
