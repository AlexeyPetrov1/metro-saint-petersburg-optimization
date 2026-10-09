"""Reconcile independent entry sources without imputing or changing observations."""
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_HOURLY_SOURCE = Path(__file__).resolve().parent / 'data/processed/goal4_3_source_refresh/entries_vestibule_1h.parquet'


def _aware(frame, columns):
    for column in columns:
        if column not in frame or not isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(f'{column} must contain timezone-aware timestamps')


def _validate(frame, duration):
    required = {'vestibule_id', 'station_id', 'interval_start', 'interval_end', 'entry_count', 'source_id'}
    if not required <= set(frame):
        raise ValueError(f'missing source columns: {sorted(required - set(frame))}')
    _aware(frame, ['interval_start', 'interval_end'])
    if frame[['vestibule_id', 'station_id', 'interval_start', 'interval_end', 'source_id']].isna().any().any():
        raise ValueError('source keys and interval boundaries cannot be missing')
    if frame.duplicated(['vestibule_id', 'interval_start']).any():
        raise ValueError('duplicate source intervals')
    aligned = frame.interval_start.dt.floor(f'{duration}min').eq(frame.interval_start)
    if not (aligned & frame.interval_end.sub(frame.interval_start).eq(pd.Timedelta(minutes=duration))).all():
        raise ValueError('source intervals must be aligned and have the declared duration')
    counts = pd.to_numeric(frame.entry_count, errors='raise').to_numpy(float, na_value=np.nan)
    if np.isinf(counts).any() or (counts < 0).any() or (counts[np.isfinite(counts)] % 1 != 0).any():
        raise ValueError('entry counts must be nonnegative integers or missing')
    _aware(frame, [c for c in ('available_at', 'received_at') if c in frame])


def _availability(frame):
    columns = ['interval_end', *[c for c in ('available_at', 'received_at') if c in frame]]
    values = frame[columns].copy()
    for column in columns:
        values[column] = values[column].dt.tz_convert('UTC')
    return values.max(axis=1).where(values.notna().all(axis=1))


def correct_service_hour_labels(frame):
    """Explicitly interpret 00:00–02:00 as the following service-calendar day."""
    _aware(frame, ['interval_start', 'interval_end'])
    if ('service_hour_labels_corrected' in frame and frame.service_hour_labels_corrected.any()) or 'raw_interval_start' in frame:
        raise ValueError('service hour labels have already been corrected')
    result = frame.copy()
    result['raw_interval_start'], result['raw_interval_end'] = frame.interval_start, frame.interval_end
    shift = pd.to_timedelta(frame.interval_start.dt.tz_convert('Europe/Moscow').dt.hour.lt(3).astype(int), unit='D')
    result['interval_start'] += shift
    result['interval_end'] += shift
    result['service_hour_labels_corrected'] = True
    return result


def reconcile_sources(quarters, hourly, *, service_hour_labels=False, absolute_threshold=100,
                      relative_threshold=.2):
    """Return every source hour, including absent/incomplete comparisons as unknown.

    When publication timestamps do not exist, availability is explicitly assumed
    at interval end. An existing but missing timestamp never receives this assumption.
    """
    if not np.isfinite([absolute_threshold, relative_threshold]).all() or absolute_threshold < 0 or relative_threshold < 0:
        raise ValueError('source thresholds must be nonnegative')
    q = quarters.copy()
    h = correct_service_hour_labels(hourly) if service_hour_labels else hourly.copy()
    _validate(q, 15)
    _validate(h, 60)
    mapping = pd.concat([q[['vestibule_id', 'station_id']], h[['vestibule_id', 'station_id']]])
    if mapping.groupby('vestibule_id').station_id.nunique().gt(1).any():
        raise ValueError('a vestibule cannot belong to different stations across sources')
    q['hour_start'] = q.interval_start.dt.floor('h')
    q['_available'] = _availability(q)
    q['_availability_known'] = q._available.notna()
    q['_complete'] = q.entry_count.notna() & (q.is_complete.fillna(False) if 'is_complete' in q else True)
    keys = ['vestibule_id', 'station_id', 'hour_start']
    quarter = q.groupby(keys, observed=True, sort=False).agg(
        quarter_observed_total=('entry_count', 'sum'),
        quarter_rows=('entry_count', 'size'), quarter_complete_rows=('_complete', 'sum'),
        quarter_source_ids=('source_id', lambda s: tuple(sorted(set(s)))),
        quarter_interval_starts=('interval_start', tuple), quarter_interval_ends=('interval_end', tuple),
        quarter_available_at=('_available', 'max'), quarter_availability_known=('_availability_known', 'all'))
    quarter['quarter_observed_total'] = q.groupby(keys, observed=True).entry_count.sum(min_count=1)
    quarter['quarter_complete'] = quarter.quarter_rows.eq(4) & quarter.quarter_complete_rows.eq(4)
    quarter['quarter_total'] = quarter.quarter_observed_total.where(quarter.quarter_complete)
    quarter['quarter_available_at'] = quarter.quarter_available_at.where(quarter.quarter_availability_known)
    quarter = quarter.reset_index()
    h['hour_start'] = h.interval_start
    h['hourly_available_at'] = _availability(h)
    h['hourly_complete'] = h.entry_count.notna() & (h.is_complete.fillna(False) if 'is_complete' in h else True)
    h['hourly_raw_interval_start'] = h.raw_interval_start if 'raw_interval_start' in h else h.interval_start
    h['hourly_raw_interval_end'] = h.raw_interval_end if 'raw_interval_end' in h else h.interval_end
    keep = [*keys, 'entry_count', 'source_id', 'hourly_complete', 'hourly_available_at',
            'hourly_raw_interval_start', 'hourly_raw_interval_end']
    result = quarter.merge(h[keep].rename(columns={'entry_count': 'hourly_count', 'source_id': 'hourly_source_id'}),
                           on=keys, how='outer', validate='one_to_one')
    comparable = result.quarter_complete.fillna(False) & result.hourly_complete.fillna(False)
    for row in result.loc[comparable, ['quarter_source_ids', 'hourly_source_id']].itertuples(index=False):
        if row.hourly_source_id in row.quarter_source_ids:
            raise ValueError('comparison requires independent source IDs')
    result['hour_end'] = result.hour_start + pd.Timedelta(hours=1)
    result['delta'] = (result.quarter_total - result.hourly_count).where(comparable)
    result['mismatch_threshold'] = np.maximum(absolute_threshold, relative_threshold * result.hourly_count)
    result['quality_status'] = np.select([comparable & result.delta.abs().gt(result.mismatch_threshold), comparable],
                                        ['source_mismatch', 'ok'], default='unknown')
    availability = pd.concat([result.quarter_available_at, result.hourly_available_at,
                              result.hour_end.dt.tz_convert('UTC')], axis=1)
    result['quality_available_at'] = availability.max(axis=1).where(comparable & availability.notna().all(axis=1))
    result['availability_assumption'] = ('interval_end_when_source_publication_absent' if
        any(not ({'available_at', 'received_at'} & set(f)) for f in (q, h)) else 'provided_timestamps_enforced')
    result = result.sort_values(['vestibule_id', 'hour_start']).reset_index(drop=True)
    previous = result.groupby('vestibule_id', observed=True).shift(1)
    pair = (result.quality_status.eq('source_mismatch') & previous.quality_status.eq('source_mismatch') &
            result.hour_start.sub(previous.hour_start).eq(pd.Timedelta(hours=1)) &
            result.delta.mul(previous.delta).lt(0) & result.delta.add(previous.delta).abs().le(absolute_threshold))
    result['compensating_adjacent_hour'] = pair | pair.shift(-1, fill_value=False)
    result['compensation_available_at'] = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns, UTC]')
    for index in result.index[pair]:
        times = result.loc[[index - 1, index], 'quality_available_at']
        if times.notna().all():
            result.loc[[index - 1, index], 'compensation_available_at'] = times.max()
    return result


def annotate_quality(frame, reconciliation):
    """Add station-hour quality metadata; retain each raw station-quarter count."""
    _aware(frame, ['interval_start', 'interval_end'])
    if frame.duplicated(['station_id', 'interval_start']).any():
        raise ValueError('duplicate station intervals')
    result = frame.copy()
    if len(reconciliation):
        r = reconciliation.copy()
        r['_severity'] = r.quality_status.map({'ok': 0, 'unknown': 1, 'source_mismatch': 2})
        keys = ['station_id', 'hour_start']
        r['_selected'] = r._severity.eq(r.groupby(keys, observed=True)._severity.transform('max'))
        r = r[r._selected].copy()
        r['_known'] = r.quality_available_at.notna()
        metadata = r.groupby(keys, observed=True).agg(severity=('_severity', 'max'),
            quality_available_at=('quality_available_at', 'max'), availability_known=('_known', 'all')).reset_index()
        metadata['quality_status'] = metadata.severity.map({0: 'ok', 1: 'unknown', 2: 'source_mismatch'})
        metadata['quality_available_at'] = metadata.quality_available_at.where(metadata.availability_known & metadata.severity.ne(1))
        metadata = metadata[[*keys, 'quality_status', 'quality_available_at']]
        join = result[['station_id', 'interval_start']].assign(hour_start=result.interval_start.dt.floor('h'))
        join = join.merge(metadata, on=['station_id', 'hour_start'], how='left', validate='many_to_one')
        result['quality_status'] = join.quality_status.fillna('unknown').to_numpy()
        result['quality_available_at'] = join.quality_available_at.array
    else:
        result['quality_status'] = 'unknown'
        result['quality_available_at'] = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns, UTC]')
    return result


def filter_available_quality(frame, as_of):
    """Exclude known source mismatches; keep unassessed and future quality states.

    Unknown independent-source coverage does not invalidate real complete counts.
    Callers must enforce observation availability/is_complete separately.
    """
    if 'quality_status' not in frame:
        return frame.copy()
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('quality cutoff requires a timezone')
    _aware(frame, ['quality_available_at'])
    if not frame.quality_status.isin(['ok', 'source_mismatch', 'unknown']).all():
        raise ValueError('unknown quality status')
    mismatch = frame.quality_status.eq('source_mismatch')
    if (mismatch & frame.quality_available_at.isna()).any():
        raise ValueError('source mismatch requires nonmissing availability')
    return frame.loc[~(mismatch & frame.quality_available_at.le(cutoff))].copy()
