"""Score raw entry truth on ordinary, affected and recovery intervals."""
import numpy as np
import pandas as pd


SLICES = ('ordinary', 'repair', 'disruption', 'recovery', 'holiday', 'data_quality')


def _time(series, name):
    if series.empty:
        return pd.Series(index=series.index, dtype='datetime64[ns, UTC]', name=series.name)
    if not isinstance(series.dtype, pd.DatetimeTZDtype) or series.isna().any():
        raise ValueError(f'{name} must contain nonmissing timezone-aware timestamps')
    return series.dt.tz_convert('UTC')


def _metrics(part):
    y = pd.to_numeric(part.actual_count, errors='raise').to_numpy(float, na_value=np.nan)
    f = pd.to_numeric(part.predicted_count, errors='raise').to_numpy(float, na_value=np.nan)
    complete = part.actual_is_complete.fillna(False).to_numpy(bool)
    valid = np.isfinite(y) & np.isfinite(f) & complete
    errors = np.abs(f[valid] - y[valid])
    total = float(y[valid].sum())
    coverage, quantile_rows = None, 0
    if {'q10', 'q90'} <= set(part):
        low, high = (pd.to_numeric(part[c], errors='raise').to_numpy(float, na_value=np.nan) for c in ('q10', 'q90'))
        measured = valid & np.isfinite(low) & np.isfinite(high)
        quantile_rows = int(measured.sum())
        if quantile_rows:
            coverage = float(((y[measured] >= low[measured]) & (y[measured] <= high[measured])).mean())
    return dict(rows=len(part), n=int(valid.sum()), excluded_incomplete=int((~complete).sum()),
                actual_total=total if valid.any() else None,
                wape=float(errors.sum() / total) if total > 0 else None,
                mae=float(errors.mean()) if len(errors) else None,
                q10_q90_coverage=coverage, quantile_rows=quantile_rows)


def _event_masks(frame, registry, recovery_window):
    masks = {name: pd.Series(False, index=frame.index) for name in SLICES}
    if 'evaluation_slice' in frame:
        for name in SLICES:
            masks[name] |= frame.evaluation_slice.eq(name)
    if 'is_holiday' in frame:
        masks['holiday'] |= frame.is_holiday.fillna(False).astype(bool)
    if 'quality_status' in frame:
        masks['data_quality'] |= frame.quality_status.eq('source_mismatch')
    if 'quality_flag' in frame:
        masks['data_quality'] |= frame.quality_flag.fillna('').str.contains('mismatch|invalid|missing|conflict|anomaly', case=False)
    labels = []
    if registry is not None:
        cache = {}
        for row in frame.itertuples():
            key = (row.station_id, row.interval_start, getattr(row, 'interval_end', None), getattr(row, 'vestibule_id', None))
            if key not in cache:
                if key[2] is not None and hasattr(registry, 'label_during'):
                    cache[key] = registry.label_during(key[0], key[1], key[2], vestibule_id=key[3])
                else:
                    cache[key] = registry.label_at(key[0], key[1], vestibule_id=key[3])
            label = cache[key]
            kinds = set(label.get('event_kinds', []))
            masks['repair'].loc[row.Index] |= any('repair' in kind or 'reconstruction' in kind or
                kind in ('planned_entrance_closure', 'planned_entrance_restriction') for kind in kinds)
            masks['disruption'].loc[row.Index] |= any('disruption' in kind or 'unplanned' in kind or 'passenger_incident' in kind for kind in kinds)
            masks['holiday'].loc[row.Index] |= bool({'planned_night_service', 'planned_event_entrance_closure'} & kinds)
            masks['recovery'].loc[row.Index] |= 'recovery' in kinds or label.get('evaluation_slice') == 'recovery'
            labels.append(label)
        latest = {}
        for record in getattr(registry, 'records', []):
            identity = record['event_id']
            if identity not in latest or record.get('version', 1) > latest[identity].get('version', 1):
                latest[identity] = record
        for record in latest.values():
            timing = record.get('timing', {}).get('observed', {})
            if record.get('confidence') != 'confirmed' or 'disruption' not in record.get('kind', '') or not timing.get('end'):
                continue
            end = pd.Timestamp(timing['end'])
            scope = record.get('scope', {})
            applies = (frame.station_id.isin(scope['station_ids']) if scope.get('station_ids') else
                       frame.station_id.str.startswith(record.get('line_id', '') + '_'))
            if scope.get('level') == 'network':
                applies = pd.Series(True, index=frame.index)
            masks['recovery'] |= applies & frame.interval_start.ge(end) & frame.interval_start.lt(end + recovery_window)
    masks['ordinary'] = ~pd.concat([masks[n] for n in SLICES if n != 'ordinary'], axis=1).any(axis=1)
    return masks, labels


def evaluate_forecasts(predictions, actuals, event_registry=None, *, decision_time=None, recovery_window='2h'):
    """Evaluate stored forecasts against raw truth, including formerly flagged rows.

    Features must exist at as_of. A forecast must be published by decision time
    (or target start when no decision time is supplied). Truth is retrospective.
    Slices can overlap (a disrupted hour can also have a source mismatch).
    """
    p, a = predictions.copy(), actuals.copy()
    recovery_window = pd.Timedelta(recovery_window)
    if recovery_window < pd.Timedelta(0):
        raise ValueError('recovery_window cannot be negative')
    p = p.rename(columns={c: target for c, target in [('target_start', 'interval_start'), ('target_hour', 'interval_start'),
        ('origin', 'as_of'), ('q50', 'predicted_count')] if c in p and target not in p})
    required = {'station_id', 'interval_start', 'predicted_count', 'as_of', 'available_at'}
    if not required <= set(p) or not {'station_id', 'interval_start', 'entry_count'} <= set(a):
        raise ValueError('forecast/truth schema lacks station, interval, counts or forecast availability')
    for c in ('interval_start', 'as_of', 'available_at'):
        p[c] = _time(p[c], c)
    a['interval_start'] = _time(a.interval_start, 'actual interval_start')
    if 'features_available_at' in p:
        p['features_available_at'] = _time(p.features_available_at, 'features_available_at')
        if p.features_available_at.gt(p.as_of).any():
            raise ValueError('forecast features_available_at exceeds as_of')
    if p.available_at.lt(p.as_of).any():
        raise ValueError('forecast available_at precedes as_of')
    if 'issued_at' in p:
        p['issued_at'] = _time(p.issued_at, 'issued_at')
        if (p.issued_at.lt(p.as_of) | p.available_at.lt(p.issued_at)).any():
            raise ValueError('forecast publication precedes as_of/issued_at')
    if decision_time is not None:
        limit = pd.Timestamp(decision_time)
        if limit.tzinfo is None:
            raise ValueError('forecast decision_time needs a timezone')
    elif 'decision_time' in p:
        p['decision_time'] = _time(p.decision_time, 'decision_time')
        limit = p.decision_time
    else:
        limit = p.interval_start
    if p.available_at.gt(limit).any():
        raise ValueError('forecast availability exceeds decision time or target start')
    if p.interval_start.lt(p.as_of).any():
        raise ValueError('forecast target precedes as_of')
    keys = ['station_id', 'interval_start']
    if a.duplicated(keys).any():
        raise ValueError('duplicate actual intervals')
    forecast_keys = [*keys, 'as_of', *[c for c in ('mode', 'model_version') if c in p]]
    if p.duplicated(forecast_keys).any():
        raise ValueError('duplicate forecast intervals')
    for column, values in [('predicted_count', p.predicted_count), ('entry_count', a.entry_count)]:
        counts = pd.to_numeric(values, errors='raise').to_numpy(float, na_value=np.nan)
        if np.isinf(counts).any() or (counts < 0).any():
            raise ValueError(f'{column} must contain nonnegative finite values or missing')
    if {'q10', 'q90'} <= set(p):
        numeric = p[['q10', 'predicted_count', 'q90']].apply(pd.to_numeric, errors='raise')
        complete = numeric.notna().all(axis=1)
        if ((numeric.q10 > numeric.predicted_count) | (numeric.predicted_count > numeric.q90))[complete].any():
            raise ValueError('forecast quantiles must satisfy q10 <= prediction <= q90')
    # Prediction-side copies of truth/labels cannot replace independent actuals.
    truth = a.rename(columns={'entry_count': 'actual_count'})
    truth['actual_is_complete'] = truth.is_complete.fillna(False).astype(bool) if 'is_complete' in truth else True
    overlap = (set(p) & set(truth)) - set(keys)
    forecast_metadata = {'available_at', 'received_at', 'features_available_at', 'source_id'}
    truth = truth.rename(columns={c: f'actual_{c}' for c in overlap & forecast_metadata})
    p = p.drop(columns=list(overlap - forecast_metadata - {'interval_end'}))
    if 'interval_end' in p and 'interval_end' in truth:
        truth = truth.rename(columns={'interval_end': 'actual_interval_end'})
    joined = p.merge(truth, on=keys, how='left', validate='many_to_one')
    if {'interval_end', 'actual_interval_end'} <= set(joined):
        start, end = _time(joined.interval_end, 'forecast interval_end'), pd.to_datetime(joined.actual_interval_end, utc=True)
        if (end.notna() & start.ne(end)).any():
            raise ValueError('forecast and truth interval ends disagree')
    masks, labels = _event_masks(joined, event_registry, recovery_window)
    rows = joined.to_dict('records')
    for index, row in enumerate(rows):
        row['evaluation_eligible'] = bool(pd.notna(row['actual_is_complete']) and row['actual_is_complete'] and
            pd.notna(row['actual_count']) and pd.notna(row['predicted_count']))
        row['evaluation_slices'] = [name for name in SLICES if masks[name].iloc[index]]
        if labels:
            row['event_label'] = labels[index]
    return dict(all=_metrics(joined), slices={name: _metrics(joined.loc[mask]) for name, mask in masks.items()},
                rows=rows, event_detection=evaluate_event_alerts(None, None, None),
                limitations=['Entry-count error does not measure unmet demand, train intervals or platform queues.',
                             'Event slices are descriptive; the registry is not exhaustive incident truth.',
                             f'Recovery slice: {recovery_window} after exact confirmed observed disruption end.'])


def evaluate_event_alerts(alerts, events, exposure):
    """Incident scores require an explicitly exhaustive, bounded truth exposure.

    Precision is per alert; recall and delay are per unique event. Repeated alerts
    for a known event affect alert counts but never multiply the detected events.
    """
    result = dict(precision=None, recall=None, mean_detection_delay_seconds=None,
                  false_alerts_per_station_hour=None, detected_events=None, eligible_events=None,
                  false_alerts=None, alerts=None)
    if not isinstance(exposure, dict) or exposure.get('truth_exhaustive') is not True:
        return result | {'reason': 'exhaustive incident truth and observation exposure are unavailable'}
    if alerts is None or events is None:
        return result | {'reason': 'incident truth or alert observations are unavailable'}
    stations = set(exposure.get('station_ids', []))
    start, end = (pd.Timestamp(exposure[k]) for k in ('start', 'end'))
    if not stations or start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError('exposure needs stations and a positive timezone-aware interval')
    alert = alerts.copy()
    truth = events.copy()
    if not {'alert_id', 'station_id', 'alerted_at'} <= set(alert) or not {'event_id', 'station_id', 'start', 'end'} <= set(truth):
        raise ValueError('event metrics require identified station alerts and bounded events')
    alert['alerted_at'] = _time(alert.alerted_at, 'alerted_at')
    truth['start'], truth['end'] = _time(truth.start, 'event start'), _time(truth.end, 'event end')
    if alert.alert_id.duplicated().any() or truth.event_id.duplicated().any():
        raise ValueError('duplicate alert/event identity')
    if truth.end.le(truth.start).any():
        raise ValueError('event duration must be positive')
    alert = alert[alert.station_id.isin(stations) & alert.alerted_at.ge(start) & alert.alerted_at.lt(end)]
    truth = truth[truth.station_id.isin(stations) & truth.start.lt(end) & truth.end.gt(start)]
    # Partial events explain alerts, but do not contribute recall or delay.
    eligible = truth.start.ge(start) & truth.end.le(end)
    matched = pd.Series(False, index=alert.index)
    delays = []
    for event in truth.itertuples():
        hit = alert.station_id.eq(event.station_id) & alert.alerted_at.ge(event.start) & alert.alerted_at.lt(event.end)
        matched |= hit
        if hit.any() and eligible.loc[event.Index]:
            delays.append((alert.loc[hit, 'alerted_at'].min() - event.start).total_seconds())
    station_hours = len(stations) * (end - start).total_seconds() / 3600
    false = int((~matched).sum())
    return dict(precision=float(matched.mean()) if len(alert) else None,
                recall=len(delays) / int(eligible.sum()) if eligible.any() else None,
                mean_detection_delay_seconds=float(np.mean(delays)) if delays else None,
                false_alerts_per_station_hour=false / station_hours, detected_events=len(delays),
                eligible_events=int(eligible.sum()), false_alerts=false, alerts=len(alert),
                station_hours=station_hours, reason=None)
