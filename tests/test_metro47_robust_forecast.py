import copy
from datetime import datetime, timedelta

import pytest

from forecasting import validate_bundle
from metro47_robust_forecast import apply_safety_margin
from simulation import digest


NOW = '2026-09-30T10:45:00+03:00'


def bundle():
    start = datetime.fromisoformat('2026-09-30T10:00:00+03:00')
    return dict(schema_version='1.0', model_version='frozen-q50', feature_version='causal-v1',
        issued_at=start.isoformat(), as_of=start.isoformat(), available_at=start.isoformat(),
        features_available_at=start.isoformat(), target='turnstile_entries',
        unit='passengers_per_15m', coverage='complete', included_event_ids=[],
        refresh_seconds=3600, source_versions={'model_sha256': {'q50': 'unchanged'}},
        rows=[dict(station_id=s, interval_start=(start+timedelta(seconds=900*q)).isoformat(),
            interval_end=(start+timedelta(seconds=900*(q+1))).isoformat(),
            predicted_count=[0, 1, 3, 4, 101, 100, 0, 0][q], coverage='complete')
            for s in ('A', 'B') for q in range(8)])


def test_uniform_comfort_margin_rounds_up_and_keeps_finite_tail_zero():
    result = apply_safety_margin(bundle(), NOW)
    assert [r['predicted_count'] for r in result['rows'][:8]] == [0, 2, 4, 5, 127, 125, 0, 0]
    assert validate_bundle(result, ['A', 'B'], NOW)['available_horizon_seconds'] == 4500
    meta = result['source_versions']['engineering_safety_margin']
    assert meta['factor'] == 1.25
    assert meta['normal_per_car'] == 150 and meta['design_per_car'] == 120
    assert meta['statistical_coverage_guarantee'] is False
    assert meta['future_actual_used'] is False
    assert meta['passenger_policy_changed'] is False
    assert meta['parameters_selected_from_current_run'] is False


def test_original_weights_clock_counts_and_nested_metadata_are_preserved():
    original = bundle(); before = copy.deepcopy(original)
    result = apply_safety_margin(original, NOW)
    assert original == before
    for key in ('issued_at', 'as_of', 'available_at', 'features_available_at', 'model_version'):
        assert result[key] == original[key]
    meta = result['source_versions']['engineering_safety_margin']
    assert meta['original_bundle_digest'] == digest(original)
    assert meta['decision_time'] == NOW
    assert result['source_versions']['model_sha256'] == original['source_versions']['model_sha256']
    result['source_versions']['model_sha256']['q50'] = 'changed-result'
    assert original == before


def test_margin_cannot_be_applied_twice():
    with pytest.raises(ValueError, match='already'):
        apply_safety_margin(apply_safety_margin(bundle(), NOW), NOW)


@pytest.mark.parametrize('damage', ['future_features', 'unavailable', 'oracle', 'stale', 'incomplete'])
def test_future_unavailable_or_invalid_forecast_cannot_be_guarded(damage):
    source = bundle()
    if damage == 'future_features': source['features_available_at'] = '2026-09-30T11:00:00+03:00'
    elif damage == 'unavailable': source['source_versions']['operationally_available'] = False
    elif damage == 'oracle': source['source_versions']['future_truth_intentionally_used'] = True
    elif damage == 'stale': source['issued_at'] = source['available_at'] = '2026-09-30T09:00:00+03:00'
    elif damage == 'incomplete': source['rows'].pop()
    with pytest.raises(ValueError): apply_safety_margin(source, NOW)


@pytest.mark.parametrize('normal,design', [(150, 151), (150, 0), (True, 120), (150, 120.0)])
def test_capacity_design_cannot_reduce_forecast_or_accept_ambiguous_parameters(normal, design):
    with pytest.raises(ValueError):
        apply_safety_margin(bundle(), NOW, normal_per_car=normal, design_per_car=design)


def test_fractional_input_is_rounded_conservatively_without_float_underflow():
    source = bundle(); source['rows'][0]['predicted_count'] = 0.01
    assert apply_safety_margin(source, NOW)['rows'][0]['predicted_count'] == 1


def test_equal_design_is_explicit_identity_but_still_not_a_safety_guarantee():
    original = bundle(); result = apply_safety_margin(original, NOW, design_per_car=150)
    assert result['rows'] == original['rows']
    assert result['source_versions']['engineering_safety_margin']['statistical_coverage_guarantee'] is False
