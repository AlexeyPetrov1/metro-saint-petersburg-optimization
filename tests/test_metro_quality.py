import pandas as pd
import pytest

from metro_quality import annotate_quality, correct_service_hour_labels, filter_available_quality, reconcile_sources


def sources(totals=(1000,), hourly_counts=None):
    starts = pd.date_range('2026-09-09 08:00', periods=len(totals), freq='h', tz='Europe/Moscow')
    q, h = [], []
    for i, start in enumerate(starts):
        h.append(dict(station_id='L1_16', vestibule_id='V16', interval_start=start,
                      interval_end=start + pd.Timedelta(hours=1), entry_count=(hourly_counts or totals)[i],
                      source_id='hourly'))
        for slot in range(4):
            ts = start + pd.Timedelta(minutes=15 * slot)
            base, remainder = divmod(totals[i], 4)
            q.append(dict(station_id='L1_16', vestibule_id='V16', interval_start=ts,
                          interval_end=ts + pd.Timedelta(minutes=15), entry_count=base + (slot < remainder),
                          source_id='quarters'))
    return pd.DataFrame(q), pd.DataFrame(h)


def test_compensating_hours_remain_mismatches_and_keep_raw_counts():
    q, h = sources((2413, 605), (1000, 2000))
    result = reconcile_sources(q, h)
    assert result.delta.tolist() == [1413, -1395]
    assert result.quality_status.tolist() == ['source_mismatch'] * 2
    assert result.compensating_adjacent_hour.tolist() == [True, True]
    assert result.hourly_count.tolist() == [1000, 2000]
    assert result.quarter_total.tolist() == [2413, 605]
    assert result.quality_available_at.tolist() == h.interval_end.tolist()
    assert q.entry_count.sum() == 3018


def test_missing_hour_and_incomplete_quarters_stay_unknown():
    q, h = sources((1000, 1200))
    result = reconcile_sources(q.iloc[:-1], h.iloc[:1])
    assert result.quality_status.tolist() == ['ok', 'unknown']
    assert pd.isna(result.hourly_count.iloc[1])
    assert pd.isna(result.quarter_total.iloc[1])
    assert result.quarter_observed_total.iloc[1] == 900


@pytest.mark.parametrize('damage', ['naive', 'duplicate', 'same_source', 'misaligned', 'fractional'])
def test_invalid_or_dependent_sources_are_rejected(damage):
    q, h = sources()
    if damage == 'naive':
        q['interval_start'] = q.interval_start.dt.tz_localize(None)
    elif damage == 'duplicate':
        q = pd.concat([q, q.iloc[:1]], ignore_index=True)
    elif damage == 'same_source':
        h['source_id'] = 'quarters'
    elif damage == 'misaligned':
        q.loc[0, 'interval_start'] += pd.Timedelta(minutes=1)
    else:
        q['entry_count'] = q.entry_count.astype(float)
        q.loc[0, 'entry_count'] += .5
    with pytest.raises(ValueError):
        reconcile_sources(q, h)


def test_service_date_correction_is_explicit_and_cannot_apply_twice():
    q, h = sources()
    h['interval_start'] -= pd.Timedelta(hours=7)
    h['interval_end'] -= pd.Timedelta(hours=7)
    corrected = correct_service_hour_labels(h)
    assert corrected.interval_start.iloc[0] == h.interval_start.iloc[0] + pd.Timedelta(days=1)
    assert corrected.raw_interval_start.iloc[0] == h.interval_start.iloc[0]
    assert reconcile_sources(q, h).hour_start.min() == h.interval_start.iloc[0]
    with pytest.raises(ValueError, match='already'):
        correct_service_hour_labels(corrected)


def test_service_label_correction_uses_moscow_clock_even_after_utc_serialization():
    _, h = sources()
    h['interval_start'] = pd.Series([pd.Timestamp('2026-09-09T00:00:00Z')])
    h['interval_end'] = h.interval_start + pd.Timedelta(hours=1)
    # 00:00 UTC is 03:00 Moscow and belongs to the current calendar day.
    assert correct_service_hour_labels(h).interval_start.equals(h.interval_start)


def test_quality_availability_requires_both_sources_and_interval_end():
    q, h = sources()
    q['available_at'] = q.interval_end
    q['received_at'] = q.interval_end
    h['available_at'] = h.interval_start
    h['received_at'] = h.interval_end + pd.Timedelta(hours=2)
    result = reconcile_sources(q, h)
    assert result.quality_available_at.iloc[0] == h.received_at.iloc[0]
    station = q.drop(columns='vestibule_id')
    annotated = annotate_quality(station, result)
    pd.testing.assert_series_equal(annotated.entry_count, station.entry_count)
    assert annotated.quality_available_at.eq(h.received_at.iloc[0]).all()
    h.loc[0, 'received_at'] = pd.NaT
    assert pd.isna(reconcile_sources(q, h).quality_available_at.iloc[0])


def test_annotation_uses_worst_vestibule_without_multiplying_station_rows():
    q, h = sources((1000,), (500,))
    q2, h2 = sources()
    q2['vestibule_id'] = h2['vestibule_id'] = 'V17'
    result = reconcile_sources(pd.concat([q, q2]), pd.concat([h, h2]))
    station = q.drop(columns='vestibule_id').copy()
    station['entry_count'] += q2.entry_count.to_numpy()
    out = annotate_quality(station, result)
    assert len(out) == len(station)
    assert out.quality_status.eq('source_mismatch').all()
    assert out.entry_count.tolist() == station.entry_count.tolist()


def test_online_quality_filter_does_not_read_future_flags_or_discard_unknowns():
    q, h = sources((1000,), (500,))
    annotated = annotate_quality(q.drop(columns='vestibule_id'), reconcile_sources(q, h))
    cutoff = h.interval_start.iloc[0] + pd.Timedelta(minutes=30)
    assert len(filter_available_quality(annotated, cutoff)) == 4
    assert filter_available_quality(annotated, h.interval_end.iloc[0]).empty
    annotated['quality_status'], annotated['quality_available_at'] = 'unknown', pd.NaT
    annotated['quality_available_at'] = pd.to_datetime(annotated.quality_available_at, utc=True)
    assert len(filter_available_quality(annotated, h.interval_end.iloc[0])) == 4
    annotated['quality_status'] = 'source_mismatch'
    with pytest.raises(ValueError, match='availability'):
        filter_available_quality(annotated, cutoff)
