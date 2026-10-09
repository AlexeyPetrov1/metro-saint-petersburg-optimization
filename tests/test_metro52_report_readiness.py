"""Report proof must follow the declared clock and verify fallback history."""

import importlib.util
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
