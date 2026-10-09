"""Causal medians from comparable entrance regimes, with observed quarter shape."""
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from forecasting import validate_bundle
from line1 import largest_remainders
from metro_calendar import ProductionCalendar, calendar_day
from metro_ml import observed_history
from simulation import digest, instant, integer


def _signature(registry, station, hour, cutoff, knowledge_mode):
    states = [(registry.state_during(station, hour+timedelta(minutes=15*q),
        hour+timedelta(minutes=15*(q+1)), cutoff, knowledge_mode=knowledge_mode)
        if hasattr(registry, 'state_during') else registry.state_at(station,
        hour+timedelta(minutes=15*q), cutoff, knowledge_mode=knowledge_mode)) for q in range(4)]
    modes = tuple((s['entrance_state'], s['movement_state'], s.get('censored_seconds'),
                   s.get('closed_seconds')) for s in states)
    structural = tuple(sorted({event for s in states for event in s.get('event_ids', [])
        if 'connecting_transport_regime' in s.get('event_kinds', [])}))
    uncertain = any(s.get('conflicts') or s.get('unknown_timing') for s in states)
    return (modes, structural), states, uncertain


def regime_hourly_median(frame, station_ids, as_of, *, event_registry,
                         horizon_hours=2, lookback_days=56, knowledge_mode='strict', calendar=None):
    now = instant(as_of)
    if now.minute or now.second or now.microsecond:
        raise ValueError('regime median refresh must occur on the hour')
    if knowledge_mode not in ('strict', 'publication_replay'):
        raise ValueError('online event knowledge mode is required')
    integer(horizon_hours, 'horizon hours', 1)
    integer(lookback_days, 'lookback days', 1)
    if not station_ids or len(station_ids) != len(set(station_ids)):
        raise ValueError('distinct station coverage is required')
    if isinstance(calendar, pd.DataFrame):
        calendar = ProductionCalendar(calendar)
    data = observed_history(frame, pd.Timestamp(now))
    data = data[data.station_id.isin(station_ids) &
                data.interval_start.ge(now-timedelta(days=lookback_days))].copy()
    data['hour_start'] = data.interval_start.dt.floor('h')
    wanted = {(now+timedelta(hours=h)).hour for h in range(horizon_hours)}
    data = data[data.hour_start.dt.hour.isin(wanted)]
    aggregate = data.groupby(['station_id', 'hour_start'], sort=True).agg(
        rows=('entry_count', 'size'), count=('entry_count', 'sum'),
        sources=('source_id', 'nunique'), latest_end=('interval_end', 'max'),
        source_id=('source_id', 'first')).reset_index()
    aggregate = aggregate[aggregate.rows.eq(4) & aggregate.sources.eq(1)].copy()
    signatures = {}
    for r in aggregate.itertuples():
        signature, states, uncertain = _signature(event_registry, r.station_id, r.hour_start, now, knowledge_mode)
        if not uncertain and all(s['movement_state'] == 'normal' for s in states):
            signatures[(r.station_id, r.hour_start)] = signature
    rows, selections, included, feature_times, calendar_targets = [], [], set(), [], []
    for h in range(horizon_hours):
        target = now+timedelta(hours=h)
        for station in station_ids:
            signature, states, uncertain = _signature(event_registry, station, target, now, knowledge_mode)
            if uncertain or any(s['movement_state'] != 'normal' for s in states):
                raise ValueError(f'unknown target regime for {station}/{target.isoformat()}')
            sample = aggregate[aggregate.station_id.eq(station) & aggregate.hour_start.dt.hour.eq(target.hour)]
            sample = sample[[signatures.get((station, at)) == signature for at in sample.hour_start]]
            if sample.empty:
                raise ValueError(f'no complete historical matching regime for {station}/{target.hour:02}:00')
            if calendar is not None:
                target_day = calendar_day(target, calendar)
                kinds = sample.hour_start.map(lambda at: calendar_day(at, calendar)['day_kind'])
                sample = sample[kinds.eq(target_day['day_kind']) & kinds.ne('unknown')]
                if target_day['school_period'] != 'unknown':
                    periods = sample.hour_start.map(lambda at: calendar_day(at, calendar)['school_period'])
                    sample = sample[periods.eq(target_day['school_period'])]
                if sample.empty or target_day['day_kind'] == 'unknown':
                    raise ValueError(f'no complete historical matching calendar regime for {station}/{target.hour:02}:00')
                exact = sample[sample.hour_start.dt.weekday.eq(target.weekday())]
                selection = 'same_weekday_calendar_kind' if len(exact) >= 2 else 'same_calendar_kind'
                sample = exact if len(exact) >= 2 else sample
                calendar_targets.append(dict(station_id=station, **target_day, selection=selection,
                                             historical_hours=len(sample)))
            else:
                exact = sample[sample.hour_start.dt.weekday.eq(target.weekday())]
                daytype = sample[sample.hour_start.dt.weekday.lt(5).eq(target.weekday() < 5)]
                sample = exact if len(exact) >= 2 else daytype if len(daytype) >= 2 else sample
            median = Decimal(str(sample['count'].median()))
            total = int(median.to_integral_value(rounding=ROUND_HALF_UP))
            keys = pd.MultiIndex.from_frame(sample[['station_id', 'hour_start']])
            selected = data.set_index(['station_id', 'hour_start']).loc[keys].reset_index()
            profile = selected.groupby(selected.interval_start.dt.minute).entry_count.sum().reindex([0,15,30,45], fill_value=0)
            weights = [int(x) for x in profile] if profile.sum() > 0 else [1]*4
            for q, count in enumerate(largest_remainders(total, weights)):
                start = target+timedelta(minutes=15*q)
                rows.append(dict(station_id=station, interval_start=start.isoformat(),
                    interval_end=(start+timedelta(minutes=15)).isoformat(), predicted_count=count,
                    coverage='complete', historical_hours=len(sample), hourly_median=str(median),
                    rounded_hour_count=total))
            for r in selected.itertuples():
                available = r.available_at if hasattr(r, 'available_at') else r.interval_end
                received = r.received_at if hasattr(r, 'received_at') else None
                feature_times.append(max(r.interval_end, available, received if received is not None else available))
                observation = dict(station_id=station, interval_start=r.interval_start.isoformat(),
                    interval_end=r.interval_end.isoformat(), count=int(r.entry_count),
                    source_id=r.source_id, available_at=available.isoformat())
                if received is not None:
                    observation['received_at'] = received.isoformat()
                selections.append(observation)
            included.update(event for state in states for event in state.get('event_ids', []))
    bundle = dict(schema_version='1.0', model_version='regime-calendar-station-hour-median-v1' if calendar is not None else 'regime-station-hour-median-v1',
        issued_at=as_of, as_of=as_of, available_at=as_of,
        features_available_at=as_of,
        target='turnstile_entries', unit='passengers_per_15m', coverage='complete',
        feature_version='causal-access-regime-quarter-profile-v1',
        source_versions=dict(source_ids=sorted(set(data.source_id)),
            history_digest=digest(selections), event_knowledge_mode=knowledge_mode,
            lookback_days=lookback_days, historical_hours_used=len(selections)//4,
            latest_observation_available_at=max(feature_times).isoformat(),
            weekday_calendar=calendar.manifest() if calendar is not None else 'weekday only; no holiday calendar inferred',
            calendar_targets=calendar_targets,
            latent_demand='not estimated from censored admissions'),
        availability_assumption='entry publication/receipt uses available_at/received_at if supplied; otherwise interval end; event ingestion enforced in strict mode',
        included_event_ids=sorted(included), rows=rows, refresh_seconds=3600,
        rounding='ROUND_HALF_UP hourly median; largest remainders with comparable historical quarter profile',
        evaluation_mode='causal_regime_median', claimed_ml_WAPE=None)
    validate_bundle(bundle, station_ids, as_of)
    return bundle
