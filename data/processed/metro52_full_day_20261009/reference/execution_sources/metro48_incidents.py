"""Immediate recomputation on a material incident or revised recovery estimate."""

from datetime import datetime, timedelta

from goal43_notifications import update_incident, record_calculation_started, record_recommendation
from metro44_run import bounded_decision


def process_incident(ledger, incident, at, calculate, *, budget_seconds=480, bounded=bounded_decision):
    updated = update_incident(ledger,incident,at)
    if len(updated)==len(ledger) or not updated[-1]['recalculation_required']:
        return updated,dict(recomputed=False,status='unchanged')
    known = updated[-1]['incident']; identity=known['incident_id']
    updated=record_calculation_started(updated,identity,at)
    receipt=bounded(lambda deadline,cancelled:calculate(known,at,deadline,cancelled),budget_seconds)
    recommendation=receipt['result'] if receipt['completed'] else dict(validated=False,status='evaluation_timeout')
    ready=(datetime.fromisoformat(at)+timedelta(seconds=receipt['elapsed_seconds'])).isoformat()
    updated=record_recommendation(updated,identity,recommendation,ready,receipt['elapsed_seconds'],budget_seconds)
    return updated,dict(recomputed=True,status=updated[-1]['kind'],elapsed_seconds=receipt['elapsed_seconds'],
        within_budget=updated[-1]['within_budget'],revision=updated[-1]['revision'],
        external_delivery_measured=False,future_factual_end_used=False)
