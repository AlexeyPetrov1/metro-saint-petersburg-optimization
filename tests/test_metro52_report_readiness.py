"""Report proof must follow the declared clock and verify fallback history."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metro52_readiness_report", ROOT / "metro51_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_full_day_schedule_uses_declared_control_start():
    assert report.expected_control_times({"arm": "BALANCED", "cohort_end_second": 64800,
                                          "control_start_second": 900}) == list(range(900, 64800, 900))
    assert len(report.expected_control_times({"arm": "BALANCED", "cohort_end_second": 64800})) == 68
    assert report.expected_control_times({"arm": "REFERENCE", "cohort_end_second": 64800}) == []
    assert report.expected_control_times({"arm": "BALANCED", "stage": "evening", "cohort_end_second": 48600})[0] == 36000


@pytest.mark.parametrize("status", ["evaluation_error", "evaluation_timeout"])
def test_fallback_prefix_requires_saved_same_history(tmp_path, monkeypatch, status):
    import metro44_focus
    import simulation
    monkeypatch.setattr(metro44_focus, "execution_prefix", lambda motion, second: motion["prefix"])
    monkeypatch.setattr(simulation, "simulate", lambda data, plan: {"prefix": data["prefix"]})
    saved = {"second": 900, "state": {"second": 900}, "data": {"prefix": "same", "plans": [{"plan_id": "p"}]}}
    monkeypatch.setattr(report, "read", lambda path: saved)
    path = tmp_path / "controller/resume_00900.json.gz"
    path.parent.mkdir(); path.touch()
    decision = {"decision_second": 900, "applied_candidate_id": "KEEP",
                "selection": {"status": status, "release_authorized": False, "selected_candidate_id": None}}
    assert report.committed_prefix_preserved(tmp_path, {"prefix": "same"}, [decision])
    assert not report.committed_prefix_preserved(tmp_path, {"prefix": "changed"}, [decision])
    path.unlink()
    assert not report.committed_prefix_preserved(tmp_path, {"prefix": "same"}, [decision])


def test_verified_decision_cannot_omit_prefix(tmp_path):
    decision = {"decision_second": 900, "applied_candidate_id": "COST_SOLVER",
                "selection": {"status": "conditional_resource_policy", "release_authorized": True}}
    assert not report.committed_prefix_preserved(tmp_path, {}, [decision])


def saved_arm(root, name, *, completed=True, contract=True, scenario=False):
    folder = root/name
    folder.mkdir()
    record = {"arm": name.upper(), "full_day": True}
    if contract:
        (folder/"contract.json").write_text(json.dumps(record), encoding="utf-8")
    if completed:
        (folder/"result.json").write_text(json.dumps({"contract": record, "metrics": {}}), encoding="utf-8")
    if scenario:
        (folder/"common_scenario.json.gz").touch()
    return folder


@pytest.fixture
def comparison_dependencies(monkeypatch):
    import metro51_service
    audited = []

    def audit(folder):
        report.read(folder/"contract.json")
        audited.append(folder.name)
        pairing = dict.fromkeys(("groups_digest", "profiles_digest", "roster_digest", "geometry_digest",
                                 "boarding_seed", "entry_end_second", "cohort_end_second", "observation_end_second"), 1)
        return dict(pairing, checks={}, metrics={}), None, None

    monkeypatch.setattr(report, "audit_arm", audit)
    monkeypatch.setattr(metro51_service, "compare_waiting_budget", lambda *a, **k: {
        "passed": True, "windows": [{"name": "whole_period", "candidate": {
            "mean_wait_seconds": 0, "p95_wait_seconds": 0}}]})
    monkeypatch.setattr(report, "make_comparison", lambda results, audits, **k: {
        "full_day": all(r["contract"]["full_day"] for r in results.values())})
    monkeypatch.setattr(report, "collect_explanations", lambda root: ({}, []))
    monkeypatch.setattr(report, "collect_post_execution_forecasts", lambda root: [])
    return audited


@pytest.mark.parametrize("balanced_complete", [True, False])
@pytest.mark.parametrize("aggregate", ['{"cases": []}', "not an arm"])
def test_comparison_uses_two_declared_arms_and_ignores_incident_aggregate(tmp_path, comparison_dependencies, balanced_complete, aggregate):
    saved_arm(tmp_path, "reference")
    saved_arm(tmp_path, "balanced", completed=balanced_complete)
    incidents = tmp_path/"incidents"
    incidents.mkdir()
    (incidents/"result.json").write_text(aggregate, encoding="utf-8")
    result = report.compile_comparison(tmp_path)
    status = result["campaign_status"]
    assert status["expected_arms"] == ["balanced", "reference"]
    assert sorted(status["completed_arms"]) == (["balanced", "reference"] if balanced_complete else ["reference"])
    assert status["unfinished_arms"] == ([] if balanced_complete else ["balanced"])
    assert status["all_arms_finished"] is balanced_complete
    assert status["full_day_completed"] is balanced_complete
    assert sorted(comparison_dependencies) == sorted(status["completed_arms"])


@pytest.mark.parametrize("name", ["balanced", "custom_arm"])
def test_comparison_does_not_hide_missing_real_arm_contract(tmp_path, comparison_dependencies, name):
    saved_arm(tmp_path, "reference")
    saved_arm(tmp_path, name, contract=False, scenario=name == "custom_arm")
    with pytest.raises(FileNotFoundError, match="contract.json"):
        report.compile_comparison(tmp_path)


def test_status_report_ignores_incident_result_and_preserves_pending_arm(tmp_path):
    saved_arm(tmp_path, "reference")
    saved_arm(tmp_path, "balanced", completed=False)
    incidents = tmp_path/"incidents"
    incidents.mkdir()
    (incidents/"result.json").write_text("not an arm", encoding="utf-8")
    status = report.write_status_report(tmp_path, tmp_path/"status")
    assert status["completed_arms"] == ["reference"]
    assert status["unfinished_arms"] == ["balanced"]


def test_partial_60_includes_the_last_control_at_2100():
    contract = {"arm": "BALANCED", "stage": "partial_60", "control_start_second": 900,
                "cohort_end_second": 54000, "stop_after_steps": 60}
    assert report.expected_control_times(contract) == list(range(900, 54001, 900))


def test_partial_campaign_is_completed_without_claiming_a_full_day(tmp_path, comparison_dependencies, monkeypatch):
    for name in ("reference", "balanced"):
        folder = saved_arm(tmp_path, name)
        contract = {"arm": name.upper(), "stage": "partial_60", "full_day": False}
        (folder/"contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (folder/"result.json").write_text(json.dumps({"contract": contract}), encoding="utf-8")
    monkeypatch.setattr(report, "make_comparison", lambda *a, **k: {"full_day": False, "stage": "partial_60"})
    status = report.compile_comparison(tmp_path)["campaign_status"]
    assert status["full_day_status"] == "completed_partial"
    assert status["full_day_completed"] is False


def test_partial_accounting_conserves_unclosed_passengers_without_claiming_completion():
    state = {"entered": 10, "completed": 6, "queued": 1, "approaching": 1, "onboard_count": 2}
    summary = report.passenger_observation(state, [{"count": 10}], partial=True)
    assert summary["accounting_passed"] is True
    assert summary["journeys_complete"] is False and summary["censored"] is True
    state["completed"] = 7
    assert report.passenger_observation(state, [{"count": 10}], partial=True)["accounting_passed"] is False


def test_partial_scope_uses_contract_clock_and_marks_censoring():
    contract = {"stage": "partial_60", "period_start": "2026-09-30T06:00:00+03:00",
                "cohort_end_second": 54000, "observation_end_second": 54000}
    text = report.observation_scope({"stage": "partial_60", "contracts": {"reference": contract}})
    assert "06:00–21:00" in text and "цензур" in text
    assert "10:30" not in text


def test_partial_render_uses_observed_ledger_and_has_no_annual_estimate(tmp_path):
    from test_metro51_report import results, audits
    source = results()
    for record in source.values():
        record["contract"].update(stage="partial_60", full_day=False, stop_after_steps=60,
                                  period_start="2026-09-30T06:00:00+03:00",
                                  cohort_end_second=54000, observation_end_second=54000)
        record["metrics"]["actual_per_train"] = record["metrics"]["per_train"]
        record["metrics"]["per_train"] = [dict(row, km=400) for row in record["metrics"]["per_train"]]
    comparison = report.make_comparison(source, audits(admitted=False), paired_inputs_passed=True)
    comparison["campaign_status"] = {"full_day_status": "completed_partial"}
    assert comparison["economics"]["balanced"]["observed_window"]["candidate_km"] == 6460
    assert "annual" not in comparison["economics"]["balanced"]
    manifest = report.render_reports(comparison, tmp_path, create_pdf=False)
    assert manifest["annual_scenario_emitted"] is False
    assert manifest["side_columns"] == list(report.SIDE_NAMES)
    for name in ("report.md", "dispatcher.html"):
        text = (tmp_path/name).read_text(encoding="utf-8")
        assert "06:00–21:00" in text and "10:30" not in text
        assert "цензур" in text
