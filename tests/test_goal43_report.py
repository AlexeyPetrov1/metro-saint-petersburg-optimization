import csv
import gzip
import json
from pathlib import Path
from collections import Counter
from decimal import Decimal
import pytest

from goal43_report import finalize, _presentation, _saved_queue_age


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(raw) if path.suffix == ".gz" else raw)


def saved_scenario(tmp_path):
    output = tmp_path / "goal43"
    folder = output / "scenarios" / "fixture"
    scenario = dict(scenario_id="fixture", title="Пример", evaluation_mode="fixture", full_day=True,
        decisions=[dict(decision_time="2026-09-01T06:00:00+03:00", elapsed_seconds=3,
            latency_scope="detection through forecast, candidate generation and detailed validation",
            selection=dict(status="conditional_resource_policy", selected_candidate_id="WITHDRAW_ONLY:B001"))],
        comparisons=[dict(candidate_id="KEEP", strategy="KEEP", status="evaluated", elapsed_seconds=3,
            metrics=dict(train_km="20", saving_km="0", to1_resource_value_rub="0", lost_revenue_rub=None),
            per_train=[dict(train_id="B001", train_type="Baltiets", baseline_km="20", candidate_km="20", saving_km="0")]),
            dict(candidate_id="WITHDRAW_ONLY:B001", strategy="WITHDRAW_ONLY", status="evaluated", elapsed_seconds=3,
            metrics=dict(train_km="10", saving_km="10", to1_resource_value_rub="92", lost_revenue_rub=None),
            per_train=[dict(train_id="B001", train_type="Baltiets", baseline_km="20", candidate_km="10", saving_km="10")])])
    dump(folder / "summary.json", scenario)
    dump(folder / "assumptions.json", dict(service_end_second=1800))
    dump(folder / "forecast.json", dict(target="turnstile_entries", issued_at="2026-09-01T06:00:00+03:00", rows=[
        dict(station_id="A", interval_start="2026-09-01T06:00:00+03:00", interval_end="2026-09-01T06:15:00+03:00", predicted_count=20),
        dict(station_id="A", interval_start="2026-09-01T06:15:00+03:00", interval_end="2026-09-01T06:30:00+03:00", predicted_count=20)]))
    runs = []
    for candidate, km in (("KEEP", "20"), ("WITHDRAW_ONLY:B001", "10")):
        run_id = candidate.replace(":", "_")
        path = output / "runs" / run_id
        dump(path / "motion.json", dict(period_start="2026-09-01T06:00:00+03:00", obligations_closed=True))
        dump(path / "groups.json.gz", [dict(group_id="G1", origin="A", turnstile_second=0, count=10),
                                          dict(group_id="G2", origin="A", turnstile_second=900, count=30)])
        runs.append(dict(run_id=run_id, folder=f"runs/{run_id}", scenario_id="fixture", strategy=candidate.split(":")[0],
                         od="S1", seed=7, full_day=True, entered=40, completed=40, queued=0,
                         metrics=dict(policy_exposure=dict(admissible=True), resources=dict(train_km=km)),
                         headways=dict(passed=True)))
    dump(output / "runs.json", runs)
    return output, folder


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_normalizes_decimal_strings_only_for_demo_and_preserves_unknown_money(tmp_path):
    output, _ = saved_scenario(tmp_path)
    result = finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    assert payload["scenarios"][0]["comparisons"][1]["metrics"]["saving_km"] == 10.0
    assert payload["scenarios"][0]["comparisons"][1]["metrics"]["lost_revenue_rub"] is None
    assert payload["confirmed_parameters"] and payload["assumptions"] and payload["unknowns"]
    assert rows(output / "tables/train_km.csv")[1]["saving_km"] == "10"
    assert result["scenario_count"] == 1
    assert result["positive_admissible_examples"] == 1
    assert (output / "demo.html").exists()
    assert result["verification"]["status"] == "not_rechecked"
    assert result["generated_at"].endswith("+03:00")
    assert "прибыль" in (output / "report.md").read_text(encoding="utf-8").lower()


def test_forecast_errors_use_station_entries_and_exact_available_coverage(tmp_path):
    output, _ = saved_scenario(tmp_path)
    result = finalize(output)
    assert result["forecast_errors"]["covered_rows"] == 2
    assert result["forecast_errors"]["wape"] == 0.5
    assert result["forecast_errors"]["absolute_error"] == 20
    assert result["forecast_errors"]["actual_entries"] == 40
    assert [row["actual_entries"] for row in rows(output / "tables/forecast_errors.csv")] == ["10", "30"]


def test_missing_forecast_fact_remains_unknown(tmp_path):
    output, folder = saved_scenario(tmp_path)
    forecast = json.loads((folder / "forecast.json").read_text(encoding="utf-8"))
    forecast["rows"][1]["interval_end"] = "2026-09-01T06:45:00+03:00"
    dump(folder / "forecast.json", forecast)
    result = finalize(output)
    assert result["forecast_errors"]["covered_rows"] == 1
    assert result["forecast_errors"]["missing_fact_rows"] == 1
    assert rows(output / "tables/forecast_errors.csv")[1]["actual_entries"] == ""


def test_inadmissible_km_reduction_is_not_counted_as_positive_example(tmp_path):
    output, _ = saved_scenario(tmp_path)
    data = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    data[1]["headways"]["passed"] = False
    dump(output / "runs.json", data)
    result = finalize(output)
    assert result["positive_admissible_examples"] == 0
    assert result["comparisons"][1]["service_status"] == "violated"
    assert result["comparisons"][1]["saving_km"] == "10"


def test_campaign_scope_and_incomplete_seed_count_are_not_hidden(tmp_path):
    output, _ = saved_scenario(tmp_path)
    dump(output / "runs/KEEP/economics.json", {"per_train": {f"B{index:03}": dict(saving_km="2",
        baseline_km="3", candidate_km="1") for index in range(68)}})
    dump(output / "campaign/validation/manifest.json", dict(stage="validation", seeds=[200], seed_count=1,
         elapsed_seconds=5, scope=dict(evaluation_mode="explicit_synthetic_three_station_fixture", probability_guarantee=False),
         trials=[dict(stage="validation", future_seed=200, od="S1", candidate_id="WITHDRAW_ONLY:B001",
                      strategy="WITHDRAW_ONLY", run_id="KEEP", status="evaluated", service_admissible=True,
                      full_day=True, obligations_closed=True, saving_km="4", train_km="6", to1_resource_value_rub="36.8")]))
    result = finalize(output)
    assert result["campaign"][0]["seed_count"] == 1
    assert result["campaign"][0]["complete_seed_count"] is False
    assert result["campaign"][0]["scope"]["probability_guarantee"] is False
    assert rows(output / "tables/campaign_trials.csv")[0]["saving_km"] == "4"
    assert result["positive_admissible_future_trials"] == 1
    assert result["positive_admissible_examples"] == 2
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    assert payload["scenarios"][-1]["comparisons"][0]["metrics"]["mean_saving_km_per_train"] == 2.0


def test_empty_report_does_not_claim_scenarios_or_economy(tmp_path):
    result = finalize(tmp_path / "empty")
    assert result["status"] == "partial"
    assert result["scenario_count"] == 0
    assert result["positive_admissible_examples"] == 0
    assert result["forecast_errors"]["wape"] is None


def test_campaign_unknown_service_does_not_imply_a_headway_failure(tmp_path):
    output, _ = saved_scenario(tmp_path)
    dump(output / "campaign/validation/manifest.json", dict(stage="validation", seeds=[200],
         trials=[dict(future_seed=200, od="S1", candidate_id="KEEP", strategy="KEEP", status="evaluated")]))
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    comparison = payload["scenarios"][-1]["comparisons"][0]
    assert comparison["status"] == "unknown"
    assert comparison["od_results"][0]["headways_passed"] is None


def test_demo_explains_the_saved_selection_without_claiming_profit(tmp_path):
    output, folder = saved_scenario(tmp_path)
    scenario = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    scenario["decisions"][0]["selection"]["policy_version"] = "closed-schedule-km-comfort-soft-v1"
    dump(folder / "summary.json", scenario)
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    reason = payload["scenarios"][0]["decisions"][0]["reason"]
    assert "проверенных допустимых" in reason
    assert "денежной прибыли" in reason


def test_incomplete_search_does_not_claim_no_feasible_plan_exists(tmp_path):
    output, folder = saved_scenario(tmp_path)
    scenario = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    scenario["decisions"][0].update(search_complete=False, selection=dict(status="no_feasible_candidate",
        selected_candidate_id=None, mitigation_candidate_id="KEEP"))
    dump(folder / "summary.json", scenario)
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    reason = payload["scenarios"][0]["decisions"][0]["reason"]
    assert "Среди проверенных вариантов" in reason
    assert "Поиск не завершён; отсутствие допустимого решения не доказано" in reason


def test_performance_preserves_the_latest_invocation_without_claiming_whole_campaign_runtime(tmp_path):
    output, _ = saved_scenario(tmp_path)
    dump(output / "performance.json", {"total_seconds": 42})
    performance = finalize(output)["performance"]
    assert performance["total_seconds"] == 42
    assert performance["total_seconds_scope"] == "last recorded replay invocation; not the sum of all artifacts"


def test_demo_presentation_separates_actual_duration_from_revised_controller_estimates():
    source = dict(scenario_id="station_stop_5_to_15", title="station_stop_5_to_15",
        incident=dict(detected_second=1843, expected_end_second=2143), injected_actual_end_second=2743,
        decisions=[dict(decision_second=1843, incident_estimates=[dict(expected_end_second=2143)],
                        selection=dict(selected_candidate_id="KEEP")),
                   dict(decision_second=2023, incident_estimates=[dict(expected_end_second=2743)])])
    original = json.dumps(source, sort_keys=True)
    shown = _presentation(source)
    assert "5" in shown["title"] and "15" in shown["title"] and "Пересмотр" in shown["title"]
    assert shown["incident"]["actual_duration_minutes"] == 15
    assert shown["incident"]["initial_estimated_duration_seconds"] == 300
    assert shown["incident"]["controller_estimate_history"][1]["estimates"][0]["expected_end_second"] == 2743
    assert shown["decisions"] == source["decisions"]
    assert json.dumps(source, sort_keys=True) == original


def test_demo_presentation_does_not_replace_unknown_actual_duration_with_an_estimate():
    shown = _presentation(dict(scenario_id="station_stop_unknown", incident=dict(
        detected_second=100, expected_end_second=1000)))
    assert "неизвестным" in shown["title"]
    assert shown["incident"]["actual_duration_minutes"] is None
    assert shown["incident"]["initial_estimated_duration_seconds"] == 900
    saved = _presentation(dict(scenario_id="station_stop_3", incident=dict(duration_seconds=180)))
    assert saved["incident"]["actual_duration_minutes"] == 3
    assert saved["title"] == "Остановка состава на 3 минуты"


@pytest.mark.parametrize("count,missing,expected", [(68, False, 2.0), (67, False, None), (68, True, None)])
def test_demo_mean_saving_requires_complete_fixed_68_train_mileage(tmp_path, count, missing, expected):
    output, folder = saved_scenario(tmp_path)
    scenario = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    comparison = scenario["comparisons"][1]
    comparison["metrics"]["saving_km"] = "136"
    comparison["per_train"] = [dict(train_id=f"B{index:03}", saving_km="2", baseline_km="3", candidate_km="1")
                                for index in range(count)]
    if missing:
        comparison["per_train"][-1]["saving_km"] = None
    dump(folder / "summary.json", scenario)
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    assert payload["scenarios"][0]["comparisons"][1]["metrics"]["mean_saving_km_per_train"] == expected


def test_full_day_audit_exposes_failed_windows_and_residual_queue_separately_from_waiting_cohort(tmp_path):
    output, _ = saved_scenario(tmp_path)
    runs = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    run = runs[1]
    run.update(completed=39, queued=1, cohort_scope=dict(start_second=0, end_second=1800),
               resources_scope=dict(start_second=0, end_second=9000))
    run["metrics"]["policy_exposure"] = dict(admissible=False, global_=None,
        windows=[dict(start=900, end=1800, exceptional_time_share=0.02, status="fail")])
    dump(output / "runs.json", runs)
    result = finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    comparison = payload["scenarios"][0]["comparisons"][1]
    audited = comparison["od_results"][0]
    assert comparison["status"] == "fail" and comparison["audit_full_day"] is True
    assert audited["queued"] == 1 and audited["cohort_scope"]["end_second"] == 1800
    assert audited["failed_15m_windows"] == [dict(start_second=900, end_second=1800, exceptional_time_share=0.02)]
    assert result["positive_admissible_scenario_examples"] == 0


@pytest.mark.parametrize("state,platform,turnstile", [
    ({"second": 72000, "queues": {"A:1": [{"count": 1, "platform_second": 64680, "turnstile_second": 64560},
                                         {"count": 0, "platform_second": 0}]}}, 7320, 7440),
    ({"second": 72000, "queues": {}}, 0, 0),
    ({"second": 72000}, None, None),
    ({"second": 72000, "queues": {"A:1": [{"count": 1, "platform_second": 72001}]}}, None, None),
    ({"second": 72000, "queues": {"A:1": [{"count": 1, "platform_second": "64680"}]}}, None, None),
])
def test_saved_queue_age_uses_positive_rows_and_valid_checkpoint_timestamps(state, platform, turnstile):
    result = _saved_queue_age(state)
    assert result["oldest_queue_age_seconds"] == platform
    assert result["oldest_turnstile_wait_age_seconds"] == turnstile


def test_queue_age_card_reads_the_matching_s1_checkpoint_only(tmp_path):
    output, _ = saved_scenario(tmp_path)
    dump(output / "runs/WITHDRAW_ONLY_B001/passenger_checkpoint.json.gz", dict(second=72000,
        queues={"A:1": [dict(count=1, platform_second=64680, turnstile_second=64560)]}))
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    card = payload["scenarios"][0]["comparisons"][1]
    assert card["metrics"]["oldest_queue_age_seconds"] == 7320
    assert card["queue_age_scope"]["checkpoint_second"] == 72000
    assert card["queue_age_scope"]["od"] == "S1"
    assert card["queue_age_scope"]["includes_closure_tail"] is True


def test_planned_seeds_without_completed_trials_are_incomplete(tmp_path):
    output, _ = saved_scenario(tmp_path)
    dump(output / "campaign/validation/manifest.json", dict(stage="validation", seeds=list(range(200, 220)),
         seed_count=20, scope={}, trials=[]))
    assert finalize(output)["campaign"][0]["complete_seed_count"] is False


def test_missing_actual_journal_does_not_become_zero_entries(tmp_path):
    output, _ = saved_scenario(tmp_path)
    (output / "runs/KEEP/groups.json.gz").unlink()
    result = finalize(output)
    assert result["forecast_errors"]["covered_rows"] == 0
    assert result["forecast_errors"]["missing_fact_rows"] == 2


def test_mean_mileage_uses_fixed_train_set_and_violation_scope_is_visible(tmp_path):
    output, folder = saved_scenario(tmp_path)
    scenario = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    scenario["comparisons"][1]["per_train"].append(dict(train_id="B002", train_type="Baltiets",
        baseline_km="0", candidate_km="0", saving_km="0"))
    dump(folder / "summary.json", scenario)
    data = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    data[1]["headways"] = dict(passed=False, violations=[dict(kind="completed_interval", second=900, seconds=600)])
    data[1]["cohort_scope"] = dict(start_second=0, end_second=1800)
    dump(output / "runs.json", data)
    result = finalize(output)
    assert result["comparisons"][1]["mean_saving_km_per_train"] == "5"
    assert result["service_runs"][1]["cohort_scope"]["end_second"] == 1800
    assert result["service_runs"][1]["recorded_headway_violations"][0]["seconds"] == 600


def test_fare_sensitivity_and_maintenance_dates_remain_explicit_hypotheses(tmp_path):
    output, _ = saved_scenario(tmp_path)
    result = finalize(output)
    sensitivity = json.loads((output / "revenue_sensitivity.json").read_text(encoding="utf-8"))
    row = sensitivity["comparisons"][1]
    assert row["benefit_kind"] == "TO1_resource_accrual"
    assert row["potential_unpaid_trips"] == 1000
    assert [item["lost_revenue_rub"] for item in row["scenarios"]] == ["0", "50.9", "509", "2545"]
    assert Decimal(row["break_even_additional_unpaid_trips"]) == Decimal(92) / Decimal("50.9")
    assert row["comparison_is_cash_profit"] is False
    assert sensitivity["actual_lost_revenue_rub"] is None
    assert all(item["status"] == "assumed" for item in row["scenarios"])
    maintenance = json.loads((output / "maintenance_deferral.json").read_text(encoding="utf-8"))
    assert len(maintenance["fleet"]) == 68
    assert Counter(item["train_type"] for item in maintenance["fleet"]) == {"Baltiets": 37, "Yubileyny": 31}
    assert all(item["km_since_to1"] is None and item["status"] == "unknown"
               and item["candidate_to1_at"] is None for item in maintenance["fleet"])
    assert result["economic_artifacts"]["revenue_sensitivity"] == "revenue_sensitivity.json"
    assert result["comparisons"][1]["lost_revenue_rub"] is None


def test_saved_timetable_effect_is_distinct_from_depot_return_and_not_promised_benefit(tmp_path):
    output, folder = saved_scenario(tmp_path)
    scenario = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    scenario["decisions"][0].update(decision_second=0, candidate_table=[
        dict(candidate_id="KEEP", strategy="KEEP", action=dict(kind="KEEP")),
        dict(candidate_id="WITHDRAW_ONLY:B001", strategy="WITHDRAW_ONLY", action=dict(kind="WITHDRAW_ONLY",
            actions=[dict(kind="WITHDRAW", train_id="B001", actual_effect_second=500)]))])
    dump(folder / "summary.json", scenario)
    for run_id, seconds in (("KEEP", [100, 900]), ("WITHDRAW_ONLY_B001", [200])):
        dump(output / "runs" / run_id / "movement_input.json", dict(
            stations=[dict(station_id="A", kind="line"), dict(station_id="B", kind="line")],
            tracks=[dict(track_id="AB", from_station="A", to_station="B")],
            plans=[dict(trips=[dict(trip_id="H", passenger=True), dict(trip_id="K", passenger=True)])]))
        dump(output / "runs" / run_id / "motion.json", dict(period_start="2026-09-01T06:00:00+03:00",
            obligations_closed=True, events=[dict(event_type="departure", train_id="B001", trip_id=trip,
                location="AB", second=second) for trip, second in zip(("H", "K"), seconds)]))
    finalize(output)
    payload = json.loads((output / "demo_payload.json").read_text(encoding="utf-8"))
    action = payload["scenarios"][0]["decisions"][0]["candidate_table"][1]["action"]
    assert action["actual_effect_second"] == 100
    assert action["actual_effect_by_station"] == {"A": 100}
    assert action["actual_effect_kind"] == "timetable change"
    assert action["guaranteed_improvement"] is False
    assert action["actions"][0]["depot_return_second"] == 500
    assert action["actions"][0]["actual_effect_second"] == 500
    assert payload["scenarios"][0]["comparisons"][1]["od_results"][0]["od"] == "S1"
