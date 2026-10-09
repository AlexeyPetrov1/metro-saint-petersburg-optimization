"""Causal residual candidates, access/quality diagnostics and measured motion."""
import math
from numbers import Real

import pandas as pd

from metro_monitoring import STATION_IDS, _timestamp, movement_state_at


LIMITATION = 'Flow anomalies are candidates; weekday/daytype reference excludes holiday and school-term modeling. Incident precision and unmet demand require independent evidence.'


def _observations(frame):
    data = pd.DataFrame(frame).copy()
    if not {'station_id', 'interval_start', 'interval_end', 'entry_count'} <= set(data):
        raise ValueError('monitor needs identified intervals and entry_count')
    data = data[data.station_id.isin(STATION_IDS)].copy()
    if 'source_id' in data:
        data = data[~data.source_id.eq('uploaded-hourly-uniform-warmup')].copy()
    for key in ('interval_start', 'interval_end'):
        data[key] = pd.to_datetime([_timestamp(value, key) for value in data[key]], utc=True)
    if not (data.interval_end-data.interval_start).eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('monitor requires quarter-hour intervals')
    availability = data.interval_end.copy()
    for field in ('available_at', 'received_at'):
        if field in data:
            supplied = pd.Series(pd.to_datetime([_timestamp(value, field) if pd.notna(value) else pd.NaT
                for value in data[field]], utc=True), index=data.index)
            availability = pd.concat([availability, supplied], axis=1).max(axis=1).where(
                availability.notna() & supplied.notna())
    data['available_at'] = availability
    data['complete'] = data.get('is_complete', pd.Series(True, index=data.index)).eq(True)
    data['local'] = data.interval_start.dt.tz_convert('Europe/Moscow')
    keys = ['station_id', *(['vestibule_id'] if 'vestibule_id' in data else []), 'interval_start']
    if data.duplicated(keys).any():
        raise ValueError('duplicate monitor observations')
    return data.reset_index(drop=True)


def _quality_known(row, now):
    stamp = getattr(row, 'quality_available_at', None)
    return pd.notna(stamp) and _timestamp(stamp, 'quality_available_at') <= now


def _state(registry, sid, start, end, now, vestibule, mode):
    if registry is None:
        return dict(entrance_state='open', movement_state='normal', event_ids=[], demand_censored=False)
    if hasattr(registry, 'state_during'):
        return registry.state_during(sid, start, end, now, vestibule_id=vestibule, knowledge_mode=mode)
    return registry.state_at(sid, start, now, vestibule_id=vestibule, knowledge_mode=mode)


def monitor_anomalies(frame, event_registry, as_of, *, vestibules=None, movement=None,
                      lookback_days=56, min_history=4, threshold=4, absolute_threshold=100,
                      knowledge_mode='strict'):
    now = _timestamp(as_of, 'as_of')
    if (type(lookback_days) is not int or lookback_days <= 0 or type(min_history) is not int or min_history < 1
            or not isinstance(threshold, Real) or not math.isfinite(threshold) or threshold <= 0
            or not isinstance(absolute_threshold, Real) or not math.isfinite(absolute_threshold) or absolute_threshold < 0):
        raise ValueError('positive history/threshold parameters are required')
    if knowledge_mode not in {'strict', 'publication_replay'}:
        raise ValueError('online event knowledge mode is required')
    station = _observations(frame)
    motion = movement_state_at(movement, now, sorted(set(station.station_id)) or sorted(STATION_IDS))
    alerts, residuals, unknowns = [], [], set(motion['unknowns'])
    if vestibules is not None and 'vestibule_id' not in vestibules:
        raise ValueError('vestibule observations need vestibule_id')
    for data in [station]+([] if vestibules is None else [_observations(vestibules)]):
        keys = ['station_id']+(['vestibule_id'] if 'vestibule_id' in data else [])
        known = data[data.interval_end.le(now) & data.available_at.le(now)].copy()
        for _, part in known.groupby(keys, sort=False):
            current = part.sort_values('interval_start').iloc[-1]
            if now-current.interval_end > pd.Timedelta(hours=1):
                unknowns.add('entry_observation_stale'); continue
            sid, vestibule = current.station_id, current.get('vestibule_id')
            state = _state(event_registry, sid, current.interval_start, current.interval_end, now, vestibule, knowledge_mode)
            past = part[part.interval_end.le(current.interval_start) & part.complete & part.entry_count.notna()
                & part.interval_start.ge(current.interval_start-pd.Timedelta(days=lookback_days))]
            past = past[past.local.dt.strftime('%H:%M').eq(current.local.strftime('%H:%M'))]
            eligible = []
            for row in past.itertuples():
                access = _state(event_registry, sid, row.interval_start, row.interval_end, now, vestibule, knowledge_mode)
                if (_quality_known(row, now) and getattr(row, 'quality_status', None) == 'source_mismatch'
                        or access.get('demand_censored') or access.get('movement_state') != 'normal'
                        or access.get('conflicts') or access.get('unknown_timing')):
                    continue
                eligible.append(row.Index)
            reference = past.loc[eligible]
            exact = reference[reference.local.dt.weekday.eq(current.local.weekday())]
            reference = exact if len(exact) >= min_history else reference[
                reference.local.dt.weekday.lt(5).eq(current.local.weekday() < 5)]
            value = current.entry_count
            valid = (current.complete and pd.notna(value) and isinstance(value, Real)
                     and math.isfinite(value) and value >= 0 and int(value) == value)
            median = float(reference.entry_count.median()) if len(reference) >= min_history else None
            delta = float(value)-median if valid and median is not None else None
            scale = max(1, math.sqrt(max(1, median)), 1.4826*float((reference.entry_count-median).abs().median())) if median is not None else None
            score = delta/scale if delta is not None else None
            flow = score is not None and abs(score) >= threshold and abs(delta) >= absolute_threshold
            measured = motion['station_states'].get(sid, 'unknown')
            row = dict(station_id=sid, vestibule_id=vestibule, interval_start=current.interval_start.isoformat(),
                interval_end=current.interval_end.isoformat(), observed_count=int(value) if valid else None,
                reference_count=median, residual_count=delta, residual_score=score, history_count=len(reference),
                reference_mode='same_weekday' if len(exact) >= min_history else 'same_daytype',
                movement_state=measured, event_ids=state.get('event_ids', []), causal_claim=None,
                adjacent_station_ids=[], signal_sources=['turnstile_residual'] if flow else [])
            residuals.append(row.copy())
            quality = _quality_known(current, now) and current.get('quality_status') == 'source_mismatch'
            classification = ('data_quality' if quality else 'censored_admissions' if state.get('demand_censored') else
                'measured_movement_disruption' if measured == 'disrupted' else
                'known_movement_context' if state.get('movement_state') == 'disrupted' else
                'flow_anomaly_candidate' if flow else None)
            if classification:
                source = ('source_quality' if quality else 'event_registry' if classification in
                    {'censored_admissions', 'known_movement_context'} else 'measured_motion' if measured == 'disrupted' else None)
                row.update(classification=classification)
                if source: row['signal_sources'].append(source)
                alerts.append(row)
            if median is None: unknowns.add('reference_coverage_insufficient')
            if not valid: unknowns.add('entry_observation_missing')
    groups = {}
    for row in alerts:
        if (row['classification'] in {'flow_anomaly_candidate', 'measured_movement_disruption', 'known_movement_context'}
                and 'turnstile_residual' in row['signal_sources'] and row['vestibule_id'] is None):
            groups.setdefault((row['interval_start'], row['residual_count'] < 0), []).append(row)
    for rows in groups.values():
        ids = sorted({row['station_id'] for row in rows}); blocks = []
        for sid in ids:
            if not blocks or int(sid[-2:]) != int(blocks[-1][-1][-2:])+1: blocks.append([])
            blocks[-1].append(sid)
        for row in rows:
            block = next(block for block in blocks if row['station_id'] in block)
            if len(block) >= 2:
                row['adjacent_station_ids'] = block
                row['signal_sources'].append('adjacent_stations')
    represented = {row['station_id'] for row in alerts if 'measured_motion' in row['signal_sources']}
    for sid, state in motion['station_states'].items():
        if state == 'disrupted' and sid not in represented:
            alerts.append(dict(station_id=sid, vestibule_id=None, interval_start=motion['latest_movement_at'],
                interval_end=None, observed_count=None, reference_count=None, residual_count=None,
                residual_score=None, history_count=0, reference_mode=None, movement_state=state, event_ids=[],
                causal_claim=None, adjacent_station_ids=[], signal_sources=['measured_motion'],
                classification='measured_movement_disruption'))
    return dict(alert_rows=alerts, context=dict(as_of=now.isoformat(), knowledge_mode=knowledge_mode,
        residual_rows=residuals, movement=motion, unknowns=sorted(unknowns), threshold=threshold,
        absolute_threshold=absolute_threshold, incident_precision=None, limitation=LIMITATION))


def calibrate_alert_budget(reference_residuals, exposure, *, max_alerts_per_station_day=1, absolute_threshold=100):
    """Fit an alert-rate threshold on a declared ordinary reference exposure."""
    if not isinstance(exposure, dict) or exposure.get('ordinary_reference') is not True:
        raise ValueError('explicit ordinary reference exposure is required')
    start, end = (_timestamp(exposure[key], key).tz_convert('Europe/Moscow') for key in ('start', 'end'))
    stations = set(exposure.get('station_ids', []))
    if end <= start or not stations or not stations <= STATION_IDS or max_alerts_per_station_day < 0:
        raise ValueError('positive identified exposure and nonnegative alert budget are required')
    data = pd.DataFrame(reference_residuals).copy()
    if not {'station_id', 'interval_start', 'residual_score', 'residual_count'} <= set(data):
        raise ValueError('reference residuals need station, interval, score and count difference')
    data['interval_start'] = pd.to_datetime([_timestamp(value, 'interval_start') for value in data.interval_start], utc=True)
    if data.duplicated(['station_id', 'interval_start']).any():
        raise ValueError('duplicate reference residuals')
    if not data.station_id.isin(stations).all() or not data.interval_start.between(start, end, inclusive='left').all():
        raise ValueError('reference residuals must belong to the supplied exposure')
    finite = data[['residual_score', 'residual_count']].apply(pd.to_numeric, errors='raise').map(math.isfinite)
    if not finite.all().all():
        raise ValueError('reference residuals must be finite')
    data['day'] = data.interval_start.dt.tz_convert('Europe/Moscow').dt.normalize()
    fractions = {day: (min(day+pd.Timedelta(days=1), end)-max(day, start)).total_seconds()/86400
        for day in pd.date_range(start.normalize(), end-pd.Timedelta(nanoseconds=1), freq='D')}
    scores = sorted(set(data.loc[data.residual_count.abs().ge(absolute_threshold), 'residual_score'].abs()))
    chosen, alerts, rate = None, 0, None
    for threshold in [0.]+scores+([math.nextafter(max(scores), math.inf)] if scores else []):
        hits = data[data.residual_score.abs().ge(threshold) & data.residual_count.abs().ge(absolute_threshold)]
        counts = hits.groupby(['station_id', 'day']).size()
        maximum = max((count/fractions[day] for (_, day), count in counts.items()), default=0.)
        if maximum <= max_alerts_per_station_day:
            chosen, alerts, rate = threshold, len(hits), maximum; break
    return dict(threshold=chosen if len(data) else None, alerts=alerts, maximum_alerts_per_station_day=rate,
        reference_rows=len(data), exposure_station_days=len(stations)*sum(fractions.values()),
        incident_precision=None, limitation='Threshold fits this ordinary reference only; future alert rates remain unverified.')
