"""Bounded entry correction from previously issued forecasts, without fitting."""
from copy import deepcopy
from decimal import Decimal
import math

import pandas as pd

from forecasting import validate_bundle
from metro45_nowcast import _observations, _records, _rounded
from simulation import digest


VERSION = 'prior-forecast-entry-residual-nowcast-v1'
ADJUSTMENTS = {'nowcast', 'residual_nowcast'}


def _prior_predictions(past_bundles, recent, current):
    if recent.empty:
        return {}
    latest = recent.interval_start.max()
    needed = set(zip(recent.station_id, recent.interval_start))
    chosen = {}
    for bundle in past_bundles:
        times = {k: pd.Timestamp(bundle[k]) for k in
                 ('issued_at', 'available_at', 'features_available_at', 'as_of')}
        if any(t.tzinfo is None for t in times.values()):
            raise ValueError('archived forecast timestamps require timezone')
        if any(t > latest for t in times.values()):
            continue
        if ADJUSTMENTS & set(bundle.get('source_versions', {})):
            raise ValueError('past bundles must be original model forecasts')
        if (bundle['model_version'] != current['model_version'] or
                bundle['source_versions'].get('model_sha256') != current['source_versions'].get('model_sha256')):
            continue
        stations = sorted({r['station_id'] for r in bundle['rows']})
        validate_bundle(bundle, stations, times['available_at'].isoformat())
        bundle_hash = digest(bundle)
        for row in bundle['rows']:
            start = pd.Timestamp(row['interval_start'])
            key = (row['station_id'], start)
            if key not in needed or any(t > start for t in times.values()):
                continue
            if type(row['predicted_count']) is not int:
                raise ValueError('archived prediction counts must be integers')
            rank = (times['issued_at'], times['available_at'])
            candidate = dict(station_id=row['station_id'], interval_start=start.isoformat(),
                predicted_count=row['predicted_count'], forecast_issued_at=bundle['issued_at'],
                forecast_available_at=bundle['available_at'], features_available_at=bundle['features_available_at'],
                forecast_bundle_digest=bundle_hash, model_version=bundle['model_version'])
            if key in chosen and rank == chosen[key][0]:
                previous = chosen[key][1]
                if previous['predicted_count'] != candidate['predicted_count']:
                    raise ValueError('conflicting forecasts at the same publication time')
                if previous['forecast_bundle_digest'] < bundle_hash:
                    continue
            if key not in chosen or rank >= chosen[key][0]:
                chosen[key] = (rank, candidate)
    return {key: value for key, (_, value) in chosen.items()}


def adjust_bundle(bundle, observed_frame, as_of, *, past_bundles, recent_minutes=30,
                  gain=.25, max_relative_change=.15, decay_minutes=60):
    """Use completed, received entries against ML forecasts available before entry.

    Pick the latest original forecast per observed quarter. The station ratio is
    shrunk and clipped, then decays linearly to zero at 60 minutes by default.
    These fixed parameters are development choices, not operational requirements.
    """
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError('residual cutoff requires a timezone')
    now = now.tz_convert('Europe/Moscow')
    if (recent_minutes not in (15, 30) or type(decay_minutes) is not int or decay_minutes < 1 or
            not all(math.isfinite(v) and 0 <= v <= 1 for v in (gain, max_relative_change))):
        raise ValueError('invalid residual nowcast parameters')
    if ADJUSTMENTS & set(bundle.get('source_versions', {})):
        raise ValueError('bundle is already adjusted; use the original hourly forecast')
    stations = sorted({r['station_id'] for r in bundle['rows']})
    validate_bundle(bundle, stations, now.isoformat())
    if any(type(r['predicted_count']) is not int for r in bundle['rows']):
        raise ValueError('hourly prediction counts must be integers')
    if ('interval_start' not in observed_frame or
            not isinstance(observed_frame.interval_start.dtype, pd.DatetimeTZDtype)):
        raise ValueError('observation interval_start requires timezone')
    recent_frame = observed_frame[observed_frame.interval_start.ge(now-pd.Timedelta(minutes=recent_minutes))]
    recent, publication, availability = _observations(recent_frame, now, stations, lookback_days=1)
    predictions = _prior_predictions(past_bundles, recent, bundle)
    adjustments, multipliers = {}, {}
    cap, gain_d = Decimal(str(max_relative_change)), Decimal(str(gain))
    used = []
    for station in stations:
        rows = recent[recent.station_id.eq(station)]
        actual, expected, supported = 0, 0, 0
        for row in rows.to_dict('records'):
            prediction = predictions.get((station, row['interval_start']))
            if prediction is None:
                continue
            actual += int(row['entry_count'])
            expected += prediction['predicted_count']
            supported += 1
            used.append(prediction)
        status = ('no_recent_observations' if rows.empty else 'no_prior_predictions' if not supported else
                  'zero_predicted_volume' if expected == 0 else 'adjusted')
        multiplier = Decimal(1)
        if status == 'adjusted':
            multiplier = max(1-cap, min(1+cap, 1+gain_d*(Decimal(actual)/Decimal(expected)-1)))
        multipliers[station] = multiplier
        adjustments[station] = dict(status=status, recent_intervals=len(rows), supported_intervals=supported,
            observed_recent_count=actual, predicted_recent_count=expected, multiplier=float(multiplier))
    result = deepcopy(bundle)
    result['rows'] = []
    duration = Decimal(decay_minutes*60)
    for row in bundle['rows']:
        lead = (pd.Timestamp(row['interval_start'])-now).total_seconds()
        if lead < 0:
            continue
        decay = max(Decimal(0), 1-Decimal(str(lead))/duration)
        multiplier = 1+(multipliers[row['station_id']]-1)*decay
        result['rows'].append({**row, 'predicted_count': _rounded(row['predicted_count'], multiplier, cap)})
    result.update(issued_at=now.isoformat(), as_of=now.isoformat(), available_at=now.isoformat(),
        features_available_at=now.isoformat(), refresh_seconds=900,
        feature_version=bundle['feature_version']+'+'+VERSION,
        rounding='ROUND_HALF_UP per quarter; final counts clipped to relative integer bounds',
        evaluation_mode='causal_prior_forecast_residual_development_parameters', availability_assumption=availability)
    result['source_versions']['residual_nowcast'] = dict(version=VERSION, as_of=now.isoformat(),
        original_bundle_digest=digest(bundle), original_model_version=bundle['model_version'],
        original_feature_version=bundle['feature_version'],
        original_timestamps={k:bundle[k] for k in ('issued_at','as_of','available_at','features_available_at')},
        parameters=dict(recent_minutes=recent_minutes, gain=gain, max_relative_change=max_relative_change,
                        decay_minutes=decay_minutes), parameters_are_user_requirements=False,
        target='turnstile_entries', train_load_inferred=False,
        availability_assumption=availability, used_predictions=sorted(used, key=lambda r:(r['station_id'],r['interval_start'])),
        reference='latest same-weight original ML forecast issued and available no later than observed quarter start',
        recent_observations_digest=digest(_records(recent, publication)),
        station_adjustments=adjustments,
        observed_latest_available_at=max([pd.Timestamp(bundle['features_available_at']),
            *[recent[k].max() for k in ('interval_end',*publication) if not recent.empty]]).isoformat(),
        decay='linear by future quarter start; original prediction at and beyond decay_minutes')
    validate_bundle(result, stations, now.isoformat())
    return result
