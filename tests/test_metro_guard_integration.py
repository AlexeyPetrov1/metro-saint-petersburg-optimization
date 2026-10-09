import copy

import pandas as pd

import metro_monitoring
from rolling import decide_operational
from test_goal43_rolling_boundaries import regular_case


def blocking_guard():
    return dict(reduction_allowed=False, reasons=['demand_censored'], event_ids=['closure'],
                motion_coverage='none', latest_movement_at=None, demand_status='censored',
                unknowns=['movement_not_supplied'])


def context():
    return dict(observations=object(), event_registry=object(), movement=None,
                knowledge_mode='strict', require_movement_coverage=False)


def test_controller_applies_guard_before_simulation_and_cannot_select_blocked_withdrawal(monkeypatch):
    args = regular_case()
    withdrawal = copy.deepcopy(args[7][0])
    withdrawal.update(candidate_id='WITHDRAW_ONLY:T1', action=dict(kind='WITHDRAW_ONLY'),
                      _data={'must_not_be_simulated': True})
    supplied = args[7]+[withdrawal]
    original = copy.deepcopy(supplied)
    calls = []

    def guarded(observations, event_registry, as_of, station_ids, **kwargs):
        calls.append((observations, event_registry, as_of, station_ids, kwargs))
        return blocking_guard()

    monkeypatch.setattr(metro_monitoring, 'control_guard', guarded)
    monitoring = context()
    result = decide_operational(*args[:7], candidates=supplied, monitoring_context=monitoring)
    now = pd.Timestamp(args[0]['period_start'])+pd.Timedelta(seconds=args[2]['S1']['second'])
    assert len(calls) == 1
    assert pd.Timestamp(calls[0][2]) == now
    assert calls[0][3] == list('ABC')
    assert calls[0][4]['require_movement_coverage'] is False
    assert calls[0][4]['horizon_seconds'] == result['available_horizon_seconds']
    assert result['anomaly_guard'] == blocking_guard()
    rejected = next(row for row in result['candidate_table'] if row['candidate_id'] == 'WITHDRAW_ONLY:T1')
    assert rejected['status'] == 'rejected'
    assert rejected['reason'] == 'control_guard: demand_censored'
    assert rejected.get('passengers_checked') is not True
    assert result['selection']['selected_candidate_id'] == 'KEEP'
    assert result['selection']['mitigation_candidate_id'] is None
    assert supplied == original


def test_blocked_reduction_cannot_be_returned_as_mitigation(monkeypatch):
    args = regular_case()
    reduction = dict(candidate_id='REPLAN:-2', count_delta=-2, status='movement_checked', _data=args[0])
    monkeypatch.setattr(metro_monitoring, 'control_guard', lambda *a, **k: blocking_guard())
    result = decide_operational(*args[:7], candidates=[reduction], monitoring_context=context())
    assert result['candidate_table'][0]['status'] == 'rejected'
    assert result['selection']['selected_candidate_id'] is None
    assert result['selection']['mitigation_candidate_id'] is None
    assert result['selection']['status'] == 'no_feasible_candidate'


def test_absent_context_preserves_controller_behavior_and_avoids_monitoring(monkeypatch):
    args = regular_case()

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy invocation must not apply monitoring')

    monkeypatch.setattr(metro_monitoring, 'control_guard', forbidden)
    result = decide_operational(*args[:7], candidates=args[7])
    assert 'anomaly_guard' not in result
    assert result['selection']['selected_candidate_id'] == 'KEEP'


def test_exhausted_deadline_never_calls_guard_or_authorizes_a_candidate(monkeypatch):
    args = regular_case()

    def forbidden(*args, **kwargs):
        raise AssertionError('exhausted deadline must return immediately')

    monkeypatch.setattr(metro_monitoring, 'control_guard', forbidden)
    result = decide_operational(*args[:7], candidates=args[7], monitoring_context=context(), budget_seconds=0)
    assert result['selection']['status'] == 'evaluation_timeout'
    assert result['selection']['selected_candidate_id'] is None
    assert result['selection']['mitigation_candidate_id'] is None


def test_guard_rejection_does_not_erase_candidate_generation_timeout(monkeypatch):
    args = regular_case()
    reduction = dict(candidate_id='REPLAN:-2', count_delta=-2, status='evaluation_timeout')
    monkeypatch.setattr(metro_monitoring, 'control_guard', lambda *a, **k: blocking_guard())
    result = decide_operational(*args[:7], candidates=[reduction], monitoring_context=context())
    assert result['selection']['status'] == 'evaluation_timeout'
    assert result['search_complete'] is False


def test_available_monitor_diagnostics_are_recorded_without_turning_candidates_into_incidents(monkeypatch):
    import metro_anomaly_monitor
    args = regular_case()
    monitoring = context()
    monitoring['enable_anomaly_monitor'] = True
    diagnostics = dict(alert_rows=[dict(classification='flow_anomaly_candidate',causal_claim=None)],context={})
    calls = []
    def monitor(frame, registry, at, **kwargs):
        calls.append((frame, registry, pd.Timestamp(at)))
        return diagnostics
    monkeypatch.setattr(metro_monitoring,'control_guard',lambda *a,**k:blocking_guard())
    monkeypatch.setattr(metro_anomaly_monitor,'monitor_anomalies',monitor)
    result = decide_operational(*args[:7],candidates=args[7],monitoring_context=monitoring)
    assert result['anomaly_monitor'] == diagnostics
    assert calls[0][2] == pd.Timestamp(result['decision_time'])
    assert result['selection']['selected_candidate_id'] == 'KEEP'
