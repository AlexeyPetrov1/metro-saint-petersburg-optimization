from copy import deepcopy
from time import monotonic

import pytest

import metro45_controller as control
from goal43_policy import load_policy


def inputs(hour=10, minute=0, second=0):
    at = f'2026-09-30T{hour:02}:{minute:02}:00+03:00'
    data = dict(period_start=at, headway_policy=load_policy())
    bundle = dict(issued_at=at, rows=[dict(station_id=s, interval_start=at,
        predicted_count=180) for s in ['S1', 'S2']])
    return data, bundle, dict(second=second, queues={})


def test_control_rebuilds_from_current_forecast_and_checkpoint():
    data, bundle, state = inputs()
    before = deepcopy((data, bundle, state))
    specs, metadata = control.prepare_candidates(data, bundle, state, ['S1', 'S2'], 0, 1800)
    assert (data, bundle, state) == before
    assert len(specs) == 2
    assert specs[0]['headway'][0]['headway_seconds'] == 150
    assert metadata['feedback_checkpoint_second'] == 0
    assert metadata['feedback_interval_seconds'] == 900
    assert metadata['future_actual_demand_used'] is False


def test_peak_does_not_apply_offpeak_candidate():
    data, bundle, state = inputs(9, 45)
    specs, metadata = control.prepare_candidates(data, bundle, state, ['S1', 'S2'], 0, 1800)
    assert specs == []
    assert metadata['offpeak_candidate_applicable'] is False


def test_online_callback_keeps_total_deadline_and_exact_validation(monkeypatch):
    data, bundle, state = inputs()
    captured = {}
    monkeypatch.setattr(control, 'read_forecast', lambda *args: bundle)
    def checked(*args, **kwargs):
        captured.update(kwargs)
        return dict(decision=dict(selected_candidate_id='KEEP'), candidates=[])
    monkeypatch.setattr(control, 'calculate_catalog', checked)
    deadline = monotonic() + 20
    result = control.calculate(data, None, state, {}, [], None, 'MPC_ML', ['S1', 'S2'],
                               0, 1800, 2, deadline, None)
    assert captured['quality_policy'] == 'strict'
    assert len(captured['profile_specs']) == 2
    assert captured['bundle_override'] == bundle
    assert result['decision']['feedback45']['future_actual_demand_used'] is False


def test_guard_requires_observations_and_rejects_timeout():
    data, _, state = inputs()
    with pytest.raises(TimeoutError):
        control.calculate(data, None, state, {}, [], None, 'MPC_ML', ['S1', 'S2'],
                          0, 1800, 2, monotonic() - 1, None)
