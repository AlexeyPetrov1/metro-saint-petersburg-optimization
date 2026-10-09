from time import monotonic
from types import SimpleNamespace

import pytest

import metro47_catalog as module


def data():
    return dict(period_start='2026-09-30T08:15:00+03:00',
        stations=[dict(station_id='A', kind='line'), dict(station_id='B', kind='line')],
        plans=[dict(plan_id='P')], headway_policy={})


def test_future_truth_is_rejected_before_planning():
    with pytest.raises(ValueError, match='causal'):
        module.calculate_cost_catalog(data(), None, {}, {}, [],
            dict(source_versions=dict(future_truth_intentionally_used=True)),
            3600, 22200, monotonic()+10, None)


def test_deadline_stops_before_forecast_validation():
    with pytest.raises(TimeoutError):
        module.calculate_cost_catalog(data(), None, {}, {}, [], {},
            3600, 22200, monotonic()-1, None)


def test_bad_physical_proposal_never_reaches_passenger_validation(monkeypatch):
    seen = []
    monkeypatch.setattr(module, 'validate_bundle', lambda *args: None)
    monkeypatch.setattr(module, 'physical_gate', lambda value, *args, **kwargs:
        dict(movement_passed=not value.get('bad'), _transport_work=[]))
    monkeypatch.setattr(module, 'closed_per_train', lambda *args, **kwargs: [])
    monkeypatch.setattr(module, 'digest', lambda value: str(value.get('bad', False)))

    def validate(*args, candidates, **kwargs):
        seen.extend(r['candidate_id'] for r in candidates)
        return dict(candidate_table=[dict(candidate_id='KEEP', status='evaluated',
            passengers_checked=True, physical_passed=True)], search_complete=True)

    monkeypatch.setattr(module, 'decide_operational', validate)
    monkeypatch.setattr(module, 'choose_by_cost', lambda *args, **kwargs:
        dict(selected_candidate_id='KEEP'))
    result = module.calculate_cost_catalog(data(), SimpleNamespace(motion={}), {}, {}, [], {},
        3600, 22200, monotonic()+10, None,
        proposal=dict(headway=200), builder=lambda *args, **kwargs: ({**data(), 'bad': True}, {}))
    assert seen == ['KEEP']
    assert result['decision']['waiting_constraints'] is None
    assert result['decision']['candidate_table'][1]['status'] == 'rejected'
