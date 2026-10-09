"""Bounded dispatch checks with an explicit, uncertified error fallback."""

from math import isfinite
from threading import Event
from time import monotonic

from metro44_run import bounded_decision


def safe_recommendation(callback, budget_seconds=480, *, cancel_event=None):
    if isinstance(budget_seconds, bool) or not isfinite(budget_seconds) or not 0 < budget_seconds <= 480:
        raise ValueError('dispatch budget must be in (0,480] seconds')
    started = monotonic()
    cancelled = cancel_event if cancel_event is not None else Event()
    try:
        answer = bounded_decision(callback, budget_seconds, cancel_event=cancelled)
    except Exception as error:
        cancelled.set()
        code = ('source_unavailable' if isinstance(error, ConnectionError) else
                'input_contract_error' if isinstance(error, ValueError) else 'evaluation_failed')
        answer = dict(completed=False, result=None, elapsed_seconds=monotonic()-started,
            error_code=code, selection=dict(status='evaluation_error', selected_candidate_id=None,
                mitigation_candidate_id=None, fallback_candidate_id='KEEP', release_authorized=False,
                economic_optimality=False, economic_improvement_claimed=False))
    if not answer['completed']:
        answer['fallback_service_verified'] = False
    return answer
