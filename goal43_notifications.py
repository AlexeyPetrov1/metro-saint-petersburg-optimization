"""Local immutable incident ledger; no external dispatch or messaging."""

import copy
from datetime import datetime
import hashlib
import json
import math
from zoneinfo import ZoneInfo


MOSCOW = ZoneInfo("Europe/Moscow")
PUBLIC_FIELDS = {
    "incident_id", "train_id", "station_id", "direction", "severity", "description",
    "estimated_end_at", "estimated_end_s", "expected_duration_minutes", "allowed_actions",
    "recovery_at", "recovery_eta_s", "acknowledged_at", "acknowledgement_source",
    "action_started_at", "first_effect_at", "resolved_at", "source", "status",
}
DECISION_FIELDS = PUBLIC_FIELDS - {
    "acknowledged_at", "acknowledgement_source", "action_started_at", "first_effect_at", "source",
}
STAGES = ("detected_at", "calculation_started_at", "recommendation_ready_at", "acknowledged_at",
          "action_started_at", "first_effect_at", "resolved_at")


def _instant(at):
    result = datetime.fromisoformat(at)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("incident timestamp must include a timezone")
    return result.astimezone(MOSCOW)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _latest(ledger, incident_id):
    return next((event for event in reversed(ledger)
                 if event["incident_id"] == incident_id and "incident" in event), None)


def _append(ledger, event):
    at = _instant(event["at"])
    previous = next((row for row in reversed(ledger) if row["incident_id"] == event["incident_id"]), None)
    if previous and at < _instant(previous["at"]):
        raise ValueError("incident chronology cannot rewrite past events")
    return [*copy.deepcopy(ledger), copy.deepcopy(event)]


def update_incident(ledger, incident, at):
    """Notify only on a material change; future factual end is never copied."""
    incident_id = incident.get("incident_id")
    if not isinstance(incident_id, str) or not incident_id:
        raise ValueError("incident needs an identifier")
    at = _instant(at).isoformat()
    previous = _latest(ledger, incident_id)
    snapshot = copy.deepcopy(previous["incident"]) if previous else {"estimated_end_at": None}
    snapshot.update({key: copy.deepcopy(value) for key, value in incident.items() if key in PUBLIC_FIELDS})
    if snapshot.get("allowed_actions") is not None:
        snapshot["allowed_actions"] = sorted(set(snapshot["allowed_actions"]))
    for key in ("estimated_end_at", "recovery_at", "acknowledged_at", "action_started_at", "first_effect_at", "resolved_at"):
        if snapshot.get(key) is not None:
            snapshot[key] = _instant(snapshot[key]).isoformat()
            if key not in {"estimated_end_at", "recovery_at"} and _instant(snapshot[key]) > _instant(at):
                raise ValueError("observed incident stage cannot be in the future")
    if snapshot.get("acknowledged_at") is not None and not snapshot.get("acknowledgement_source"):
        raise ValueError("acknowledgement needs an explicit real source")
    signature = _fingerprint({key: snapshot[key] for key in sorted(DECISION_FIELDS) if key in snapshot})
    if previous and snapshot == previous["incident"]:
        return copy.deepcopy(ledger)
    changed = not previous or signature != previous["incident_signature"]
    event = {"kind": "incident_detected" if previous is None else "incident_updated",
             "incident_id": incident_id, "at": at, "incident": snapshot,
             "incident_signature": signature, "revision": 1 if previous is None else previous["revision"] + int(changed),
             "notification_at": at if changed else None, "recalculation_required": changed,
             "external_message_sent": False}
    return _append(ledger, event)


def record_calculation_started(ledger, incident_id, at):
    previous = _latest(ledger, incident_id)
    if previous is None:
        raise ValueError("unknown incident")
    return _append(ledger, {"kind": "calculation_started", "incident_id": incident_id,
                            "at": _instant(at).isoformat(), "revision": previous["revision"],
                            "notification_at": None, "external_message_sent": False})


def record_recommendation(ledger, incident_id, recommendation, at, elapsed_seconds, budget_seconds=480):
    """elapsed_seconds is measured from detection with a monotonic clock."""
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("invalid monotonic elapsed seconds")
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("invalid response budget")
    previous = _latest(ledger, incident_id)
    if previous is None:
        raise ValueError("unknown incident")
    at = _instant(at).isoformat()
    within = elapsed_seconds <= budget_seconds
    validated = recommendation.get("validated") is True
    kind = "evaluation_timeout" if not within or recommendation.get("status") == "evaluation_timeout" else "recommendation_ready" if validated else "evaluation_incomplete"
    delivered = copy.deepcopy(recommendation) if kind == "recommendation_ready" else None
    signature = _fingerprint({key: value for key, value in recommendation.items()
                              if key not in {"elapsed_seconds", "calculation_seconds", "compute_ms"}})
    prior = next((event for event in reversed(ledger) if event["incident_id"] == incident_id
                  and event["kind"] in {"recommendation_ready", "evaluation_timeout", "evaluation_incomplete"}), None)
    notify = not prior or any((prior["recommendation_signature"] != signature,
                              prior["revision"] != previous["revision"], prior["kind"] != kind))
    return _append(ledger, {"kind": kind, "incident_id": incident_id, "at": at,
                            "revision": previous["revision"], "recommendation": delivered,
                            "recommendation_signature": signature, "elapsed_seconds": elapsed_seconds,
                            "latency_basis": "monotonic detection-to-result", "budget_seconds": budget_seconds,
                            "within_budget": within, "validated": validated,
                            "notification_at": at if notify else None, "external_message_sent": False})


def analytics(ledger):
    incidents = {}
    for event in ledger:
        incident_id = event["incident_id"]
        row = incidents.setdefault(incident_id, {**{key: None for key in STAGES}, "elapsed_seconds": None,
                                                "within_budget": None, "simulation_elapsed_seconds": None,
                                                "notification_delay_seconds": None, "notification_count": 0,
                                                "recommendation_current": False, "recalculation_required": True})
        if event.get("notification_at") is not None:
            row["notification_count"] += 1
        if event["kind"] == "incident_detected":
            row["detected_at"] = event["at"]
            row["notification_delay_seconds"] = (_instant(event["notification_at"]) - _instant(event["at"])).total_seconds()
        if "incident" in event:
            row["incident"] = copy.deepcopy(event["incident"])
            for key in ("acknowledged_at", "action_started_at", "first_effect_at", "resolved_at"):
                row[key] = event["incident"].get(key)
            if event["recalculation_required"]:
                row["recommendation_current"] = False
                row["recalculation_required"] = True
        elif event["kind"] == "calculation_started":
            row["calculation_started_at"] = event["at"]
        else:
            row["elapsed_seconds"] = event["elapsed_seconds"]
            row["within_budget"] = event["within_budget"]
            row["simulation_elapsed_seconds"] = (_instant(event["at"]) - _instant(row["detected_at"])).total_seconds()
            row["result_status"] = event["kind"]
            row["recommendation_current"] = event["kind"] == "recommendation_ready"
            row["recalculation_required"] = not row["recommendation_current"]
            if row["recommendation_current"]:
                row["recommendation_ready_at"] = event["at"]
    return {"incidents": incidents, "event_count": len(ledger),
            "notification_count": sum(row["notification_count"] for row in incidents.values()),
            "latency_basis": "monotonic detection-to-result; simulated timestamps shown separately",
            "external_message_sent": False}
