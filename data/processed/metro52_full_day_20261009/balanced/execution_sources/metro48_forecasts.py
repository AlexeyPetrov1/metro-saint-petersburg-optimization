"""Causal quarter profiles, a labelled historical tail, and honest error scores."""

from copy import deepcopy
from math import ceil

import numpy as np
import pandas as pd

from forecasting import hourly_median, validate_bundle
from line1 import largest_remainders
from metro_ml import observed_history, _profile_from_observed
from simulation import digest


def profiled_median(frame, station_ids, as_of, horizon_hours=2, *, profile_provider=None):
    now = pd.Timestamp(as_of)
    if now.minute % 15 or now.second or type(horizon_hours) is not int or horizon_hours < 1:
        raise ValueError('aligned quarter and positive integer horizon required')
    floor = now.floor('h'); end = now + pd.Timedelta(hours=horizon_hours)
    source = hourly_median(frame, station_ids, floor.isoformat(),
                          int(ceil((end-floor).total_seconds()/3600)))
    past = observed_history(frame, now)
    groups = {}
    for row in source['rows']:
        key = row['station_id'], pd.Timestamp(row['interval_start']).floor('h')
        groups.setdefault(key, []).append(row)
    rows = []
    for (station, hour), group in groups.items():
        profile = np.asarray(profile_provider(station,hour) if profile_provider else
                             _profile_from_observed(past, station, hour))
        counts = largest_remainders(sum(r['predicted_count'] for r in group),
            np.maximum(0, np.rint(profile*1_000_000_000)).astype('int64').tolist())
        for quarter, count in enumerate(counts):
            start = hour + pd.Timedelta(minutes=15*quarter)
            if now <= start < end:
                rows.append(dict(station_id=station, interval_start=start.isoformat(),
                    interval_end=(start+pd.Timedelta(minutes=15)).isoformat(),
                    predicted_count=int(count), coverage='complete'))
    result = deepcopy(source)
    result.update(model_version='station-hour-median-past-quarter-profile-v2',
        feature_version='complete-hours-plus-past-only-quarter-profile',
        issued_at=as_of, as_of=as_of, available_at=as_of, refresh_seconds=900,
        features_available_at=past.interval_end.max().isoformat(), rows=rows,
        rounding='round historical hourly median; past-only profile and largest remainders')
    result['source_versions'].update(profile_observation_cutoff=as_of,
        original_hourly_bundle_digest=digest(source), future_actual_used=False)
    validate_bundle(result, station_ids, as_of)
    return result


def reprofile_hourly(bundle, profile_provider):
    result = deepcopy(bundle); groups = {}
    for row in result['rows']:
        groups.setdefault((row['station_id'],pd.Timestamp(row['interval_start']).floor('h')),[]).append(row)
    for (station,hour), rows in groups.items():
        rows.sort(key=lambda r:r['interval_start'])
        if len(rows) != 4:
            raise ValueError('shared profile requires complete model hours')
        weights = np.asarray(profile_provider(station,hour))
        counts = largest_remainders(sum(r['predicted_count'] for r in rows),
            np.maximum(0,np.rint(weights*1_000_000_000)).astype('int64').tolist())
        for row,count in zip(rows,counts):
            row['predicted_count'] = int(count)
    result['source_versions']['shared_profile_original_digest'] = digest(bundle)
    result['feature_version'] += '+shared-GRU-causal-normalization-profile'
    return result


def complete_tail(bundle, frame, station_ids, decision_time, entry_end):
    """Protect a committed return beyond the model's two-hour forecast."""
    validate_bundle(bundle, station_ids, decision_time)
    now, end = pd.Timestamp(decision_time), pd.Timestamp(entry_end)
    if end <= now or (end-now).total_seconds() % 900:
        raise ValueError('entry boundary must follow an aligned decision')
    base = profiled_median(frame, station_ids, decision_time,
                          max(1, ceil((end-now).total_seconds()/3600)))
    original = {(r['station_id'], pd.Timestamp(r['interval_start'])): r for r in bundle['rows']}
    rows, fallback = [], []
    for row in base['rows']:
        start = pd.Timestamp(row['interval_start'])
        if start >= end:
            continue
        key = row['station_id'], start
        rows.append(deepcopy(original.get(key, row)))
        if key not in original:
            fallback.append([row['station_id'], row['interval_start']])
    result = deepcopy(bundle)
    result.update(issued_at=decision_time, as_of=decision_time, available_at=decision_time,
        features_available_at=max(pd.Timestamp(bundle['features_available_at']),
                                  pd.Timestamp(base['features_available_at'])).isoformat(), rows=rows)
    result['feature_version'] += '+past-median-commitment-tail-v1'
    result['source_versions']['historical_tail'] = dict(original_model_digest=digest(bundle),
        original_model_timestamps={k: bundle[k] for k in ('issued_at','as_of','available_at')},
        median_digest=digest(base), fallback_intervals=fallback, entry_end=end.isoformat(),
        reason='cover withdrawal and scheduled return beyond native model horizon',
        future_actual_used=False, statistical_coverage_guarantee=False)
    validate_bundle(result, station_ids, decision_time)
    return result


def score_rows(rows, frame):
    predicted = pd.DataFrame(rows).copy()
    for key in ('interval_start', 'issued_at'):
        predicted[key] = pd.to_datetime(predicted[key], utc=True)
    if predicted.issued_at.gt(predicted.interval_start).any():
        raise ValueError('prediction was issued after its target began')
    if predicted.duplicated(['station_id','interval_start','issued_at']).any():
        raise ValueError('duplicate prediction identity')
    joined = predicted.merge(frame[['station_id','interval_start','entry_count','is_complete','source_id']],
        on=['station_id','interval_start'], how='left', validate='many_to_one')
    if (joined.entry_count.isna().any() or not joined.is_complete.eq(True).all()
            or not joined.source_id.eq('uploaded-spb-line1-15min').all()):
        raise ValueError('scores require complete real quarter targets')
    values = joined.predicted_count.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid predicted counts')
    error = joined.predicted_count - joined.entry_count
    total = int(joined.entry_count.sum())
    return dict(rows=len(joined), actual_entries=total,
        predicted_entries=float(joined.predicted_count.sum()),
        wape_percent=100*float(error.abs().sum())/total if total else None,
        signed_bias_percent=100*float(error.sum())/total if total else None,
        underpredicted_entries=float((-error).clip(lower=0).sum()),
        overpredicted_entries=float(error.clip(lower=0).sum()),
        mae=float(error.abs().mean()), rmse=float(np.sqrt(error.pow(2).mean())))


def calibrate_reserve(rows, evaluation_start, coverage=.99):
    if not 0 < coverage < 1 or rows.empty:
        raise ValueError('nonempty calibration and coverage between zero and one required')
    end = pd.to_datetime(rows.interval_end, utc=True)
    issued = pd.to_datetime(rows.issued_at, utc=True)
    target = pd.to_datetime(rows.interval_start, utc=True)
    if end.ge(pd.Timestamp(evaluation_start)).any() or issued.gt(target).any():
        raise ValueError('calibration must be completed before evaluation')
    predicted = rows.predicted_count.to_numpy(float); actual = rows.actual_count.to_numpy(float)
    if not np.isfinite([predicted, actual]).all() or (predicted < 0).any() or (actual < 0).any():
        raise ValueError('finite nonnegative calibration counts required')
    residual = (actual-predicted)/np.maximum(predicted, 1)
    rank = ceil((len(rows)+1)*coverage)
    reserve = max(0, float(np.sort(residual)[rank-1])) if rank <= len(rows) else None
    return dict(factor=None if reserve is None else 1+reserve,
        normalized_residual=reserve, coverage_target=coverage, finite_sample_rank=rank,
        calibration_rows=len(rows), latest_calibration_end=end.max().isoformat(),
        evaluation_start=evaluation_start, evaluation_actual_used=False,
        formula='prediction + ceil(reserve * max(prediction, 1))',
        guarantee='marginal only under exchangeability; temporal drift not guaranteed',
        operational_capacity_guarantee=False, enabled_in_controller=False)
