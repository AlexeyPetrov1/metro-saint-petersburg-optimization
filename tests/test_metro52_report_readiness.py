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
