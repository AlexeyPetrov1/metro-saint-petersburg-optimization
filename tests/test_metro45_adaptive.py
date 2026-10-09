from copy import deepcopy

import pytest

from metro45_adaptive import adaptive_profile, strict_improvement


def bundle():
    return dict(issued_at='2026-09-30T10:00:00+03:00', rows=[
        dict(station_id=f'S{i}', interval_start=f'2026-09-30T10:{minute:02}:00+03:00',
             predicted_count=count)
        for minute, count in ((0, 180), (15, 360), (30, 90), (45, 180))
        for i in (1, 2)])


def test_forecast_changes_frequency_without_changing_source():
    source = bundle(); before = deepcopy(source)
    result = adaptive_profile(source, ['S1', 'S2'], source['issued_at'],
        0, 5400, checkpoint=dict(second=0, queues={}))
    assert source == before
    assert [r['headway_seconds'] for r in result['profile']] == [150, 150, 160, 150, 160, 160]
    assert result['profile'][-1]['end_second'] == 5400
    assert result['future_actual_demand_used'] is False
    assert result['continuation_headway_seconds'] == 160


def test_queue_age_forces_only_the_first_slot_to_shorter_interval():
    b = bundle(); b['rows'] = [dict(r, predicted_count=0) for r in b['rows']]
    state = dict(second=900, queues={'S1': [dict(count=1, platform_second=700,
                                               origin='S1', destination='S2')]})
    result = adaptive_profile(b, ['S1', 'S2'], '2026-09-30T10:15:00+03:00', 900, 2700,
                              checkpoint=state)
    assert [r['headway_seconds'] for r in result['profile']] == [150, 160]
    assert result['observed_queue_urgent']


def test_future_issue_and_checkpoint_do_not_enter_policy():
    with pytest.raises(ValueError, match='future'):
        adaptive_profile(bundle(), ['S1', 'S2'], '2026-09-30T09:45:00+03:00', 0, 1800)
    with pytest.raises(ValueError, match='checkpoint'):
        adaptive_profile(bundle(), ['S1', 'S2'], bundle()['issued_at'], 0, 1800,
                         checkpoint=dict(second=900, queues={}))


def test_expired_forecast_uses_declared_continuation_without_future_truth():
    result = adaptive_profile(bundle(), ['S1', 'S2'], '2026-09-30T11:00:00+03:00', 3600, 5400)
    assert [r['headway_seconds'] for r in result['profile']] == [160, 160]
    assert result['forecast_covered_slot_count'] == 0


def test_economic_acceptance_requires_physics_service_and_positive_lower_bound():
    base = dict(mean_wait_seconds=75., p95_wait_seconds=180.)
    metrics = dict(mean_wait_seconds=75., p95_wait_seconds=180., actual_service_admissible=True,
        all_service_arrivals_completed=True, movement_obligations_closed=True,
        terminal_resources_passed=True)
    bounds = dict(saving_train_km_lower_bound=1., saving_to1_resource_rub_lower_bound=7.4)
    assert strict_improvement(base, metrics, bounds, physical_passed=True)
    for key in ('actual_service_admissible', 'all_service_arrivals_completed',
                'movement_obligations_closed', 'terminal_resources_passed'):
        assert not strict_improvement(base, {**metrics, key: False}, bounds, physical_passed=True)
    assert not strict_improvement(base, metrics, bounds, physical_passed=False)
    assert not strict_improvement(base, {**metrics, 'p95_wait_seconds': 181}, bounds, physical_passed=True)
    assert not strict_improvement(base, metrics, {**bounds, 'saving_train_km_lower_bound': 0}, physical_passed=True)
    assert not strict_improvement(base, {**metrics, 'mean_wait_seconds': float('nan')}, bounds, physical_passed=True)
