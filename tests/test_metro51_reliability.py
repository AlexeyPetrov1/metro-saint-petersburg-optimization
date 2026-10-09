from threading import Event

import pytest

from metro51_reliability import safe_recommendation


def test_forecast_failure_retains_an_uncertified_fallback_without_exposing_error_text():
    cancelled = Event()

    def failed(deadline, signal):
        raise ConnectionError('private credential in transport error')

    answer = safe_recommendation(failed, 1, cancel_event=cancelled)
    assert answer['completed'] is False
    assert answer['selection']['status'] == 'evaluation_error'
    assert answer['error_code'] == 'source_unavailable'
    assert answer['selection']['selected_candidate_id'] is None
    assert answer['fallback_service_verified'] is False
    assert cancelled.is_set()
    assert 'credential' not in str(answer)


def test_success_preserves_the_checked_result():
    result = {'selected': 'checked'}
    answer = safe_recommendation(lambda *args: result, 1)
    assert answer['completed'] is True
    assert answer['result'] is result


def test_timeout_cannot_certify_keep_or_consume_a_late_result():
    release = Event()
    cancelled = Event()
    answer = safe_recommendation(lambda *args: release.wait(.5), .01, cancel_event=cancelled)
    release.set()
    assert answer['completed'] is False
    assert answer['selection']['status'] == 'evaluation_timeout'
    assert answer['fallback_service_verified'] is False
    assert cancelled.is_set()


@pytest.mark.parametrize('budget', [0, 481, float('nan'), True])
def test_invalid_dispatch_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        safe_recommendation(lambda *args: {}, budget)
