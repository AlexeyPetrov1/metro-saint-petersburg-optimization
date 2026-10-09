import copy
import json
from pathlib import Path

import pytest

from goal43_notifications import (
    analytics, record_calculation_started, record_recommendation, update_incident,
)


AT = "2026-09-01T08:00:00+03:00"


def incident(**changes):
    return {"incident_id": "I1", "train_id": "B001", "station_id": "L1_10",
            "direction": 1, "severity": "warning",
            "estimated_end_at": "2026-09-01T08:05:00+03:00", **changes}


def test_detection_is_immediate_and_never_creates_acknowledgement():
    ledger = update_incident([], incident(), AT)
    report = analytics(ledger)["incidents"]["I1"]
    assert ledger[0]["kind"] == "incident_detected"
    assert ledger[0]["notification_at"] == AT
    assert report["detected_at"] == AT
    assert report["acknowledged_at"] is None
    assert report["calculation_started_at"] is None
    assert report["recommendation_ready_at"] is None
    assert report["notification_delay_seconds"] == 0


def test_repeated_polling_and_hidden_actual_end_do_not_duplicate_notifications():
    ledger = update_incident([], incident(), AT)
    second = update_incident(ledger, incident(actual_end_at="2026-09-01T08:15:00+03:00"),
                             "2026-09-01T08:01:00+03:00")
    assert second == ledger
    assert "actual_end_at" not in second[0]["incident"]


@pytest.mark.parametrize("changes", [
    {"estimated_end_at": "2026-09-01T08:15:00+03:00"}, {"severity": "critical"},
    {"allowed_actions": ["KEEP", "REGULATE_ONLY"]}, {"recovery_at": "2026-09-01T08:25:00+03:00"},
])
def test_material_changes_notify_once(changes):
    ledger = update_incident([], incident(), AT)
    second = update_incident(ledger, incident(**changes), "2026-09-01T08:03:00+03:00")
    third = update_incident(second, incident(**changes), "2026-09-01T08:04:00+03:00")
    assert len(second) == 2
    assert third == second
    assert second[-1]["kind"] == "incident_updated"


def test_unknown_duration_and_estimate_updates_preserve_prior_history():
    ledger = update_incident([], incident(estimated_end_at=None), AT)
    updated = update_incident(ledger, incident(), "2026-09-01T08:01:00+03:00")
    assert ledger[0]["incident"]["estimated_end_at"] is None
    assert updated[0]["incident"]["estimated_end_at"] is None
    assert updated[-1]["incident"]["estimated_end_at"].endswith("08:05:00+03:00")


@pytest.mark.parametrize("elapsed,within", [(0, True), (479.99, True), (480, True), (480.01, False)])
def test_actual_monotonic_budget_boundaries(elapsed, within):
    ledger = update_incident([], incident(), AT)
    result = record_recommendation(ledger, "I1", {"validated": True, "action": "KEEP"},
                                   "2026-09-01T09:00:00+03:00", elapsed)
    row = analytics(result)["incidents"]["I1"]
    assert row["within_budget"] is within
    assert row["elapsed_seconds"] == elapsed
    assert result[-1]["kind"] == ("recommendation_ready" if within else "evaluation_timeout")
    assert row["simulation_elapsed_seconds"] == 3600
    if not within:
        assert result[-1]["recommendation"] is None


def test_partial_validation_never_publishes_a_recommendation():
    ledger = update_incident([], incident(), AT)
    result = record_recommendation(ledger, "I1", {"validated": False, "action": "ADD"}, AT, 1)
    assert result[-1]["kind"] == "evaluation_incomplete"
    assert result[-1]["recommendation"] is None
    assert analytics(result)["incidents"]["I1"]["recommendation_ready_at"] is None


def test_calculation_started_is_recorded_only_when_supplied():
    ledger = update_incident([], incident(), AT)
    started = record_calculation_started(ledger, "I1", "2026-09-01T08:00:01+03:00")
    assert analytics(started)["incidents"]["I1"]["calculation_started_at"].endswith("08:00:01+03:00")


def test_identical_recommendation_is_not_notified_twice():
    ledger = update_incident([], incident(), AT)
    recommendation = {"validated": True, "action": "KEEP"}
    first = record_recommendation(ledger, "I1", recommendation, AT, 1)
    second = record_recommendation(first, "I1", recommendation, "2026-09-01T08:00:02+03:00", 2)
    assert len(second) == len(first) + 1
    assert second[-1]["notification_at"] is None
    assert analytics(second)["notification_count"] == 2


def test_a_changed_incident_invalidates_old_recommendation_scope():
    ledger = update_incident([], incident(), AT)
    ledger = record_recommendation(ledger, "I1", {"validated": True, "action": "KEEP"}, AT, 1)
    changed = update_incident(ledger, incident(estimated_end_at="2026-09-01T08:15:00+03:00"),
                              "2026-09-01T08:03:00+03:00")
    report = analytics(changed)["incidents"]["I1"]
    assert report["recommendation_current"] is False
    assert report["recalculation_required"] is True


def test_explicit_real_acknowledgement_and_effect_are_preserved():
    ledger = update_incident([], incident(), AT)
    updated = update_incident(ledger, incident(
        acknowledged_at="2026-09-01T08:01:00+03:00", acknowledgement_source="dispatcher input",
        action_started_at="2026-09-01T08:02:00+03:00", first_effect_at="2026-09-01T08:14:00+03:00",
        resolved_at="2026-09-01T08:20:00+03:00"), "2026-09-01T08:20:00+03:00")
    report = analytics(updated)["incidents"]["I1"]
    assert report["acknowledged_at"].endswith("08:01:00+03:00")
    assert report["first_effect_at"].endswith("08:14:00+03:00")


def test_generated_acknowledgement_is_rejected():
    with pytest.raises(ValueError, match="acknowledgement"):
        update_incident([], incident(acknowledged_at=AT), AT)


@pytest.mark.parametrize("elapsed", [-1, True, float("nan"), float("inf")])
def test_invalid_latency_is_rejected(elapsed):
    with pytest.raises(ValueError):
        record_recommendation(update_incident([], incident(), AT), "I1", {"validated": True}, AT, elapsed)


def test_not_started_incident_and_rewriting_past_are_rejected():
    with pytest.raises(ValueError, match="unknown incident"):
        record_recommendation([], "missing", {"validated": True}, AT, 1)
    with pytest.raises(ValueError, match="chronology"):
        update_incident(update_incident([], incident(), AT), incident(severity="critical"),
                        "2026-09-01T07:59:00+03:00")


def test_ledger_and_incident_inputs_are_immutable_and_json_safe():
    state = incident()
    before = copy.deepcopy(state)
    initial = []
    result = update_incident(initial, state, AT)
    assert initial == [] and state == before
    json.dumps(analytics(result), allow_nan=False)


def test_historical_report_to_broadcast_is_not_recommendation_latency():
    source = json.loads((Path(__file__).resolve().parents[1] / "docs/goal43_incident_sources.json").read_text(encoding="utf-8"))
    assert source["timestamps"]["fault_reported_at"] == "2026-08-31T07:52:00+03:00"
    assert source["timestamps"]["broadcast_started_at"] == "2026-08-31T08:06:00+03:00"
    assert source["analytics"]["report_to_broadcast_minutes"] == 14
    assert source["timestamps"]["detected_at"] is None
    assert source["timestamps"]["recommendation_ready_at"] is None
    assert source["analytics"]["recommendation_sla_compliance"] is None
