from passenger_groups import GroupedPassengerSimulation
from rolling import decide_operational
from test_goal43_rolling import setup


def test_forecast_trial_inherits_boarding_and_applies_service_check():
    data, sources, states, profiles, bundle, policy, candidates = setup()
    source = sources['S1']
    source = GroupedPassengerSimulation(data, 'P', list(source.by_id.values()), profiles,
                                        boarding_seed=7, boarding_capacity_per_car=150)
    state = source.run(end_second=300)['checkpoint']
    seen = []

    def check(trial_runner, trial_state, start, end):
        seen.append((trial_runner.boarding_seed, trial_runner.boarding_capacity_per_car))
        return {'passed': False, 'status': 'failed'}

    answer = decide_operational(data, {'S1': source}, {'S1': state}, profiles, {}, bundle,
                               policy, candidates=candidates, include_recovery=False,
                               passenger_service_check=check)
    assert seen == [(7, 150)]
    row = answer['candidate_table'][0]
    assert row['passengers_checked'] is True
    assert row['trials'][0]['peak_boarding_service']['passed'] is False
    assert row['service_passed'] is False
    assert answer['selection']['selected_candidate_id'] is None


def test_invalid_forecast_horizon_cannot_remove_a_validation_bound():
    import pytest
    data, sources, states, profiles, bundle, policy, candidates = setup()
    for horizon in (0, -1, True, float('inf')):
        with pytest.raises(ValueError):
            decide_operational(data, sources, states, profiles, {}, bundle, policy,
                               candidates=candidates, forecast_horizon_seconds=horizon)
    for end in (0, 300, -1, True, float('inf'), sources['S1'].duration + 1, 1790000000):
        with pytest.raises(ValueError):
            decide_operational(data, sources, states, profiles, {}, bundle, policy,
                               candidates=candidates, service_end_second=end)


def test_extended_horizon_is_replayed_and_headways_stop_at_service_close(monkeypatch):
    import copy
    from datetime import datetime, timedelta
    import rolling

    data, sources, states, profiles, bundle, policy, candidates = setup()
    data['period_end'] = '2026-09-01T12:00:00+03:00'
    source = GroupedPassengerSimulation(data, 'P', list(sources['S1'].by_id.values()), profiles)
    state = source.run(end_second=300)['checkpoint']
    for row in copy.deepcopy(bundle['rows']):
        for key in ('interval_start', 'interval_end'):
            row[key] = (datetime.fromisoformat(row[key]) + timedelta(hours=2)).isoformat()
        bundle['rows'].append(row)
    observed, checkpoints = [], []
    original = rolling.operational_headways

    def headways(data, motion, start, end):
        observed.append(end)
        return original(data, motion, start, end)

    monkeypatch.setattr(rolling, 'operational_headways', headways)
    def service(runner, snapshot, start, end):
        checkpoints.append(snapshot['second'])
        return {'passed': True}
    answer = decide_operational(data, {'S1': source}, {'S1': state}, profiles, {}, bundle,
        policy, candidates=candidates, forecast_horizon_seconds=14400, service_end_second=10800,
        passenger_service_check=service)
    assert answer['available_horizon_seconds'] == 14100
    assert observed == [10800]
    assert checkpoints == [14400]
