import copy

import pandas as pd
import pytest

from forecasting import validate_bundle
from metro45_nowcast import adjust_bundle
from simulation import digest


NOW = '2026-09-30T10:15:00+03:00'


def bundle(count=100):
    start = pd.Timestamp('2026-09-30T10:00:00+03:00')
    return dict(schema_version='1.0', model_version='frozen-model', feature_version='hourly-v1',
        issued_at=start.isoformat(), as_of=start.isoformat(), available_at=start.isoformat(),
        features_available_at=start.isoformat(), target='turnstile_entries', unit='passengers_per_15m',
        coverage='complete', included_event_ids=[], source_versions={'model_sha256': {'q50': 'original'}},
        refresh_seconds=3600, rows=[dict(station_id=s, interval_start=(start+pd.Timedelta(minutes=15*q)).isoformat(),
            interval_end=(start+pd.Timedelta(minutes=15*(q+1))).isoformat(), predicted_count=count,
            coverage='complete') for s in ('A', 'B') for q in range(8)])


def history(recent=(200, 200)):
    rows = []
    for day in ('2026-09-23', '2026-09-24', '2026-09-25', '2026-09-30'):
        for s in ('A', 'B'):
            for q, clock in enumerate(('09:45', '10:00')):
                start = pd.Timestamp(f'{day}T{clock}:00+03:00')
                rows.append(dict(station_id=s, interval_start=start, interval_end=start+pd.Timedelta(minutes=15),
                    entry_count=recent[q] if day == '2026-09-30' else 100,
                    is_complete=True, source_id='observed'))
    return pd.DataFrame(rows)


def counts(out, station='A'):
    return [r['predicted_count'] for r in out['rows'] if r['station_id'] == station]


def meta(out):
    return out['source_versions']['nowcast']


def test_complete_recent_intervals_adjust_only_future_and_preserve_original_provenance():
    base, frame = bundle(), history()
    original, original_frame = copy.deepcopy(base), frame.copy(deep=True)
    out = adjust_bundle(base, frame, NOW)
    assert counts(out) == [125]*7
    assert all(pd.Timestamp(r['interval_start']) >= pd.Timestamp(NOW) for r in out['rows'])
    assert out['as_of'] == out['issued_at'] == out['available_at'] == NOW
    assert out['model_version'] == base['model_version']
    assert out['source_versions']['model_sha256'] == base['source_versions']['model_sha256']
    assert meta(out)['original_bundle_digest'] == digest(base)
    assert meta(out)['original_timestamps']['issued_at'] == base['issued_at']
    assert meta(out)['station_adjustments']['A']['recent_intervals'] == 2
    assert meta(out)['station_adjustments']['A']['expected_recent_count'] == 200
    assert base == original
    pd.testing.assert_frame_equal(frame, original_frame)
    assert validate_bundle(out, ['A', 'B'], NOW)['status'] == 'passed'


def test_future_labels_and_unavailable_rows_cannot_change_prediction_or_provenance():
    frame = history()
    future = frame.iloc[-2:].copy()
    future['interval_start'] += pd.Timedelta(minutes=30)
    future['interval_end'] += pd.Timedelta(minutes=30)
    future['entry_count'] = -999999
    poisoned = pd.concat([frame, future], ignore_index=True)
    assert adjust_bundle(bundle(), frame, NOW) == adjust_bundle(bundle(), poisoned, NOW)


def test_unfinished_quarter_is_not_observed_and_started_forecast_is_not_republished():
    frame = history((200, 999999))
    out = adjust_bundle(bundle(), frame, '2026-09-30T10:07:00+03:00')
    assert counts(out) == [125]*7
    assert meta(out)['station_adjustments']['A']['recent_intervals'] == 1
    assert out['rows'][0]['interval_start'] == NOW


@pytest.mark.parametrize('column', ['available_at', 'received_at'])
@pytest.mark.parametrize('stamp', ['2026-09-30T10:16:00+03:00', None])
def test_late_or_unknown_publication_is_excluded_even_when_interval_is_complete(column, stamp):
    frame = history((100, 200))
    frame[column] = frame.interval_end.copy()
    latest = frame.interval_start.eq(pd.Timestamp('2026-09-30T10:00:00+03:00'))
    frame.loc[latest, column] = pd.Timestamp(stamp) if stamp else pd.NaT
    out = adjust_bundle(bundle(), frame, NOW)
    assert counts(out) == [100]*7
    assert meta(out)['station_adjustments']['A']['recent_intervals'] == 1
    assert column in meta(out)['availability_assumption']


def test_newly_received_observation_changes_only_the_next_issued_bundle():
    frame = history((100, 200))
    frame['received_at'] = frame.interval_end.copy()
    latest = frame.interval_start.eq(pd.Timestamp('2026-09-30T10:00:00+03:00'))
    frame.loc[latest, 'received_at'] = pd.Timestamp('2026-09-30T10:16:00+03:00')
    assert counts(adjust_bundle(bundle(), frame, NOW)) == [100]*7
    assert counts(adjust_bundle(bundle(), frame, '2026-09-30T10:16:00+03:00')) == [125]*6


def test_all_publication_fields_and_historical_reference_availability_are_required():
    frame = history((100, 200))
    frame['available_at'] = frame.interval_end.copy()
    frame['received_at'] = frame.interval_end.copy()
    latest = frame.interval_start.eq(pd.Timestamp('2026-09-30T10:00:00+03:00'))
    frame.loc[latest, 'received_at'] = pd.NaT
    assert counts(adjust_bundle(bundle(), frame, NOW)) == [100]*7
    frame = history()
    frame['available_at'] = frame.interval_end.copy()
    frame.loc[frame.interval_start.dt.day.ne(30), 'available_at'] = pd.Timestamp('2026-09-30T10:16:00+03:00')
    out = adjust_bundle(bundle(), frame, NOW)
    assert counts(out) == [100]*7
    assert meta(out)['station_adjustments']['A']['status'] == 'insufficient_history'


def test_incomplete_observations_and_insufficient_reference_history_leave_forecast_unchanged():
    frame = history()
    frame.loc[frame.interval_start.dt.date.eq(pd.Timestamp(NOW).date()), 'is_complete'] = False
    out = adjust_bundle(bundle(), frame, NOW)
    assert counts(out) == [100]*7
    assert meta(out)['station_adjustments']['A']['status'] == 'no_recent_observations'
    frame = history().loc[lambda f: ~f.interval_start.dt.day.eq(23)]
    out = adjust_bundle(bundle(), frame, NOW)
    assert counts(out) == [100]*7
    assert meta(out)['station_adjustments']['A']['status'] == 'insufficient_history'


def test_reference_profile_uses_prior_matching_day_type_and_excludes_uniform_warmup():
    frame = history()
    weekend = frame[frame.interval_start.dt.day.eq(23)].copy()
    weekend['interval_start'] += pd.Timedelta(days=3)
    weekend['interval_end'] += pd.Timedelta(days=3)
    weekend['entry_count'] = 10000
    assert counts(adjust_bundle(bundle(), pd.concat([frame, weekend]), NOW)) == [125]*7
    frame.loc[frame.interval_start.dt.day.ne(30), 'source_id'] = 'uploaded-hourly-uniform-warmup'
    assert counts(adjust_bundle(bundle(), frame, NOW)) == [100]*7


def test_quality_rejection_is_applied_only_once_the_quality_assessment_is_available():
    frame = history((100, 200))
    frame['quality_status'] = 'ok'
    frame['quality_available_at'] = frame.interval_end.copy()
    latest = frame.interval_start.eq(pd.Timestamp('2026-09-30T10:00:00+03:00'))
    frame.loc[latest, 'quality_status'] = 'source_mismatch'
    assert counts(adjust_bundle(bundle(), frame, NOW)) == [100]*7
    frame.loc[latest, 'quality_available_at'] = pd.Timestamp('2026-09-30T10:16:00+03:00')
    assert counts(adjust_bundle(bundle(), frame, NOW)) == [125]*7


def test_half_up_rounding_remains_integer_nonnegative_and_inside_final_relative_bounds():
    out = adjust_bundle(bundle(5), history((120, 120)), NOW)
    assert counts(out) == [6]*7
    assert counts(adjust_bundle(bundle(2), history(), NOW)) == [2]*7
    assert counts(adjust_bundle(bundle(100), history((0, 0)), NOW)) == [75]*7
    assert counts(adjust_bundle(bundle(0), history(), NOW)) == [0]*7
    assert all(type(r['predicted_count']) is int for r in out['rows'])


def test_zero_reference_volume_does_not_create_an_infinite_or_invented_correction():
    frame = history()
    frame.loc[frame.interval_start.dt.day.ne(30), 'entry_count'] = 0
    out = adjust_bundle(bundle(), frame, NOW)
    assert counts(out) == [100]*7
    assert meta(out)['station_adjustments']['A']['status'] == 'zero_expected_volume'


def test_no_double_adjustment_or_stale_hourly_input_can_be_promoted_as_fresh_forecast():
    out = adjust_bundle(bundle(), history(), NOW)
    with pytest.raises(ValueError, match='already'):
        adjust_bundle(out, history(), NOW)
    with pytest.raises(ValueError, match='stale'):
        adjust_bundle(bundle(), history(), '2026-09-30T11:00:00+03:00')


@pytest.mark.parametrize('kwargs', [dict(recent_minutes=10), dict(gain=-.1),
    dict(gain=float('nan')), dict(max_relative_change=1.1), dict(min_history_days=0)])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        adjust_bundle(bundle(), history(), NOW, **kwargs)


def test_missing_schema_naive_time_duplicate_and_bad_available_counts_are_rejected():
    with pytest.raises(ValueError, match='schema'):
        adjust_bundle(bundle(), history().drop(columns='source_id'), NOW)
    with pytest.raises(ValueError, match='timezone'):
        adjust_bundle(bundle(), history(), '2026-09-30T10:15:00')
    frame = history()
    with pytest.raises(ValueError, match='duplicate'):
        adjust_bundle(bundle(), pd.concat([frame, frame.iloc[:1]]), NOW)
    frame.loc[frame.interval_start.dt.day.eq(30), 'entry_count'] = -1
    with pytest.raises(ValueError, match='counts'):
        adjust_bundle(bundle(), frame, NOW)
