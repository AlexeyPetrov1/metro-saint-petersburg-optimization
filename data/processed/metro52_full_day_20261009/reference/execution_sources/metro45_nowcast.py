"""Causal, bounded entry nowcast over an unchanged hourly model."""
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math

import pandas as pd

from forecasting import validate_bundle
from simulation import digest


VERSION = 'station-entry-ratio-nowcast-v1'
REQUIRED = {'station_id', 'interval_start', 'interval_end', 'entry_count', 'is_complete', 'source_id'}


def _observations(frame, now, stations, lookback_days):
    if not REQUIRED <= set(frame):
        raise ValueError('observed schema lacks complete station entry intervals')
    publication = [key for key in ('available_at', 'received_at') if key in frame]
    for key in ('interval_start', 'interval_end', *publication):
        if not isinstance(frame[key].dtype, pd.DatetimeTZDtype):
            raise ValueError(f'{key} requires a timezone')
    if not pd.api.types.is_bool_dtype(frame.is_complete.dtype):
        raise ValueError('is_complete must be boolean')
    oldest = now.normalize()-pd.Timedelta(days=lookback_days)
    eligible = (frame.interval_start.ge(oldest) & frame.interval_end.le(now)
                & frame.is_complete.eq(True) & frame.station_id.isin(stations))
    for key in publication:
        eligible &= frame[key].notna() & frame[key].le(now)
    data = frame.loc[eligible].copy()
    if 'quality_status' in data:
        from metro_quality import filter_available_quality
        data = filter_available_quality(data, now)
    data = data[~data.source_id.eq('uploaded-hourly-uniform-warmup')].copy()
    if data.duplicated(['station_id', 'interval_start']).any():
        raise ValueError('duplicate available observations')
    aligned = data.interval_start.dt.floor('15min').eq(data.interval_start)
    if not (aligned & data.interval_end.sub(data.interval_start).eq(pd.Timedelta(minutes=15))).all():
        raise ValueError('observations require aligned quarter-hour intervals')
    if (not pd.api.types.is_integer_dtype(data.entry_count.dtype)
            or pd.api.types.is_bool_dtype(data.entry_count.dtype)
            or data.entry_count.isna().any() or data.entry_count.lt(0).any()):
        raise ValueError('observed counts must be nonnegative integers')
    if data[['station_id', 'source_id']].isna().any().any():
        raise ValueError('observation identities cannot be missing')
    local = data.interval_start.dt.tz_convert('Europe/Moscow')
    data['_clock'] = local.dt.hour*60+local.dt.minute
    data['_weekday'] = local.dt.dayofweek.lt(5)
    assumption = ('provided '+ '/'.join(publication) +' is enforced' if publication else
                  'assumed: complete intervals are available at interval end; publication times are absent')
    return data, publication, assumption


def _records(data, publication):
    return [dict(station_id=r['station_id'], entry_count=int(r['entry_count']), source_id=str(r['source_id']),
                 **{key: r[key].isoformat() for key in ('interval_start', 'interval_end', *publication)})
            for r in data.sort_values(['station_id', 'interval_start']).to_dict('records')]


def _rounded(count, multiplier, cap):
    original = Decimal(str(count))
    low = int((original*(1-cap)).to_integral_value(rounding=ROUND_CEILING))
    high = int((original*(1+cap)).to_integral_value(rounding=ROUND_FLOOR))
    rounded = int((original*multiplier).to_integral_value(rounding=ROUND_HALF_UP))
    return max(low, min(high, rounded))


def adjust_bundle(bundle, observed_frame, as_of, *, recent_minutes=30, lookback_days=28,
                  gain=.5, max_relative_change=.25, min_history_days=3):
    """Publish future quarters only; do not fit weights or read future entry labels.

    Each station's recent entry total is compared with the prior weekday/weekend
    median at the same local quarters. The bounded ratio is assumed to persist
    over the remaining original horizon. Default gain/cap are development choices,
    not operational limits. Missing references produce an unchanged forecast.
    """
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError('nowcast cutoff requires a timezone')
    now = now.tz_convert('Europe/Moscow')
    if (recent_minutes not in (15, 30) or type(lookback_days) is not int or lookback_days < 1
            or type(min_history_days) is not int or min_history_days < 1
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in (gain, max_relative_change))):
        raise ValueError('invalid nowcast parameters')
    if 'nowcast' in bundle.get('source_versions', {}):
        raise ValueError('bundle is already adjusted; use the original hourly forecast')
    stations = sorted({r['station_id'] for r in bundle['rows']})
    validate_bundle(bundle, stations, now.isoformat())
    if any(type(r['predicted_count']) is not int for r in bundle['rows']):
        raise ValueError('hourly forecast counts must be integers')
    data, publication, availability = _observations(observed_frame, now, stations, lookback_days)
    window_start = now-pd.Timedelta(minutes=recent_minutes)
    recent = data[data.interval_start.ge(window_start)].copy()
    keys = set(zip(recent.station_id, recent._clock, recent._weekday))
    history = data[data.interval_end.le(window_start)].copy()
    history = history.loc[pd.Series([k in keys for k in zip(history.station_id, history._clock, history._weekday)],
                                   index=history.index, dtype=bool)]
    reference = history.groupby(['station_id', '_clock', '_weekday'], sort=True).entry_count.agg(['median', 'size'])
    adjustments, multipliers = {}, {}
    gain_d, cap = Decimal(str(gain)), Decimal(str(max_relative_change))
    for station in stations:
        rows = recent[recent.station_id.eq(station)]
        actual, expected, supported = 0, Decimal(0), 0
        history_days = []
        for row in rows.to_dict('records'):
            key = (station, row['_clock'], row['_weekday'])
            if key not in reference.index or reference.loc[key, 'size'] < min_history_days:
                continue
            actual += int(row['entry_count'])
            expected += Decimal(str(reference.loc[key, 'median']))
            supported += 1
            history_days.append(int(reference.loc[key, 'size']))
        status = ('no_recent_observations' if rows.empty else 'insufficient_history' if not supported else
                  'zero_expected_volume' if expected == 0 else 'adjusted')
        multiplier = Decimal(1)
        if status == 'adjusted':
            multiplier = max(1-cap, min(1+cap, 1+gain_d*(Decimal(actual)/expected-1)))
        multipliers[station] = multiplier
        adjustments[station] = dict(status=status, recent_intervals=len(rows), supported_intervals=supported,
            observed_recent_count=actual, expected_recent_count=float(expected),
            minimum_reference_days=min(history_days) if history_days else 0, multiplier=float(multiplier))
    result = deepcopy(bundle)
    result['rows'] = [{**r, 'predicted_count': _rounded(r['predicted_count'], multipliers[r['station_id']], cap)}
                      for r in result['rows'] if pd.Timestamp(r['interval_start']) >= now]
    result.update(issued_at=now.isoformat(), as_of=now.isoformat(), available_at=now.isoformat(),
        features_available_at=now.isoformat(), refresh_seconds=900,
        feature_version=bundle['feature_version']+'+'+VERSION,
        rounding='ROUND_HALF_UP per quarter, clipped to integer relative bounds',
        evaluation_mode='causal_entry_nowcast_development_parameters')
    result['source_versions']['nowcast'] = dict(version=VERSION, as_of=now.isoformat(),
        original_bundle_digest=digest(bundle), original_model_version=bundle['model_version'],
        original_feature_version=bundle['feature_version'],
        original_timestamps={k: bundle[k] for k in ('issued_at', 'as_of', 'available_at', 'features_available_at')},
        original_row_count=len(bundle['rows']), published_row_count=len(result['rows']),
        parameters=dict(recent_minutes=recent_minutes, lookback_days=lookback_days, gain=gain,
                        max_relative_change=max_relative_change, min_history_days=min_history_days),
        parameters_are_user_requirements=False, target='turnstile_entries', train_load_inferred=False,
        reference='prior matching weekday/weekend and local quarter medians; uniform warmup excluded',
        availability_assumption=availability, persistence_assumption='same station multiplier throughout remaining horizon',
        recent_observations_digest=digest(_records(recent, publication)),
        reference_observations_digest=digest(_records(history, publication)), station_adjustments=adjustments,
        observed_latest_available_at=max([pd.Timestamp(bundle['features_available_at']),
            *[data[k].max() for k in ('interval_end', *publication) if not data.empty]]).isoformat())
    result['availability_assumption'] = availability
    validate_bundle(result, stations, now.isoformat())
    return result
