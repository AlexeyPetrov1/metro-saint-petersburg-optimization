"""Available event, entry-quality and optional measured-motion checks for Line 1."""

from __future__ import annotations

import copy
import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd


STATION_IDS = {f'L1_{i:02d}' for i in range(1, 20)}


def _timestamp(value, label):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be an aware timestamp') from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f'{label} must be an aware timestamp')
    return stamp.tz_convert('UTC')


def _stations(station_ids):
    ids = list(station_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= STATION_IDS:
        raise ValueError('station_ids must be distinct project Line 1 stations')
    return ids


def _present(value):
    return value is not None and not pd.isna(value)


def validate_movement(frame):
    """Copy telemetry; reject naive times, invalid locations and invented headways."""
    data = pd.DataFrame(frame).copy(deep=True)
    required = {'line_id', 'observed_at', 'available_at', 'movement_state'}
    if not required <= set(data) or not {'station_id', 'track_id'} & set(data):
        raise ValueError('movement needs line_id, a station_id or track_id, aware times and movement_state')
    for field in ('station_id', 'track_id', 'measured_headway_s', 'reference_headway_s'):
        if field not in data:
            data[field] = None
    for field in ('observed_at', 'available_at'):
        data[field] = pd.to_datetime([_timestamp(value, field) for value in data[field]], utc=True)
    if (data.available_at < data.observed_at).any():
        raise ValueError('movement available_at must not precede observed_at')
    if not data.movement_state.isin(['normal', 'disrupted', 'unknown']).all():
        raise ValueError('movement_state must be normal, disrupted or unknown')
    for row in data.itertuples(index=False):
        if not isinstance(row.line_id, str) or not row.line_id:
            raise ValueError('line_id must be explicit')
        if not (_present(row.station_id) or _present(row.track_id)):
            raise ValueError('movement needs an explicit station_id or track_id')
        if _present(row.station_id) and (not isinstance(row.station_id, str)
                                        or not row.station_id.startswith(row.line_id+'_')):
            raise ValueError('station_id must belong to line_id')
        if _present(row.track_id) and (not isinstance(row.track_id, str) or not row.track_id):
            raise ValueError('track_id must be a nonempty string')
        for field in ('measured_headway_s', 'reference_headway_s'):
            value = getattr(row, field)
            if _present(value) and (isinstance(value, bool) or not isinstance(value, Real)
                                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{field} must be positive finite seconds or missing')
    return data


def load_movement(path):
    """Load externally supplied CSV or JSON telemetry, without synthetic defaults."""
    source = Path(path)
    if source.suffix.lower() == '.csv':
        frame = pd.read_csv(source)
    elif source.suffix.lower() == '.json':
        payload = json.loads(source.read_text(encoding='utf-8'))
        frame = pd.DataFrame(payload['rows'] if isinstance(payload, dict) else payload)
    else:
        raise ValueError('movement source must be CSV or JSON')
    return validate_movement(frame)


def movement_state_at(movement, as_of, station_ids, *, max_age_seconds=900, headway_ratio=1.5):
    """Select the latest fresh observation per sensor that was available at as_of."""
    now, ids = _timestamp(as_of, 'as_of'), _stations(station_ids)
    if (isinstance(max_age_seconds, bool) or not math.isfinite(max_age_seconds)
            or max_age_seconds <= 0 or not math.isfinite(headway_ratio) or headway_ratio <= 1):
        raise ValueError('freshness and headway ratio must be positive, with ratio > 1')
    result = dict(movement_state='unknown', motion_coverage='none', latest_movement_at=None,
                  station_states={sid: 'unknown' for sid in ids}, fresh_track_ids=[],
                  reasons=[], unknowns=[])
    if movement is None:
        result['unknowns'].append('movement_not_supplied')
        return result
    data = validate_movement(movement)
    data = data[data.line_id.eq('L1') & (data.station_id.isin(ids) | data.station_id.isna())].copy()
    known = data[data.observed_at.le(now) & data.available_at.le(now)].copy()
    if len(known) != len(data):
        result['unknowns'].append('movement_unavailable')
    fresh = known[known.observed_at.ge(now-pd.Timedelta(seconds=max_age_seconds))].copy()
    if len(fresh) != len(known):
        result['unknowns'].append('movement_stale')
    fresh['sensor'] = [f'station:{sid}:track:{track}' for sid, track in
                       zip(fresh.station_id.fillna(''), fresh.track_id.fillna(''))]
    latest = fresh.sort_values(['observed_at', 'available_at']).groupby('sensor', sort=False).tail(1)
    states = []
    for row in latest.itertuples(index=False):
        same = fresh[fresh.sensor.eq(row.sensor) & fresh.observed_at.eq(row.observed_at)
                     & fresh.available_at.eq(row.available_at)]
        conflict = (same.movement_state.nunique() > 1
                    or same[['measured_headway_s', 'reference_headway_s']].drop_duplicates().shape[0] > 1)
        inflated = any(_present(item.measured_headway_s) and _present(item.reference_headway_s)
                       and item.measured_headway_s/item.reference_headway_s >= headway_ratio
                       for item in same.itertuples(index=False))
        state = 'disrupted' if conflict or inflated else row.movement_state
        if conflict:
            result['reasons'].append('movement_conflict')
        if inflated and row.movement_state == 'normal':
            result['reasons'].append('movement_state_headway_conflict')
        if _present(row.station_id):
            previous = result['station_states'][row.station_id]
            result['station_states'][row.station_id] = 'disrupted' if 'disrupted' in (previous, state) else state
        if _present(row.track_id):
            result['fresh_track_ids'].append(row.track_id)
        states.append(state)
    if len(latest):
        result['latest_movement_at'] = latest.observed_at.max().isoformat()
        covered = sum(state != 'unknown' for state in result['station_states'].values())
        result['motion_coverage'] = 'complete' if covered == len(ids) else 'partial'
        result['movement_state'] = ('disrupted' if 'disrupted' in states else 'normal'
                                    if 'normal' in states else 'unknown')
    if result['motion_coverage'] != 'complete':
        result['unknowns'].append('movement_coverage_incomplete')
    result['reasons'] = sorted(set(result['reasons']))
    result['unknowns'] = sorted(set(result['unknowns']))
    result['fresh_track_ids'] = sorted(set(result['fresh_track_ids']))
    return result


def bounded_demand(admitted_count, entrance_state):
    """Censored admissions give a lower bound; the unmet demand remains unknown."""
    if (isinstance(admitted_count, bool) or not isinstance(admitted_count, Real)
            or not math.isfinite(admitted_count) or admitted_count < 0 or int(admitted_count) != admitted_count):
        raise ValueError('admitted_count must be a nonnegative integer')
    if entrance_state not in {'open', 'closed', 'restricted', 'unknown'}:
        raise ValueError('invalid entrance_state')
    censored = entrance_state != 'open'
    return dict(observed_admitted_count=admitted_count, true_demand_lower_bound=admitted_count,
                true_demand_upper_bound=None if censored else admitted_count, censored=censored)


def control_guard(frame, registry, as_of, station_ids, movement=None, knowledge_mode='strict', *,
                  require_movement_coverage=False, max_age_seconds=900, headway_ratio=1.5,
                  horizon_seconds=0):
    """Veto reductions on known disruption, censored admissions or source mismatch."""
    if knowledge_mode not in {'strict', 'publication_replay'}:
        raise ValueError('knowledge_mode must be strict or publication_replay')
    if (isinstance(horizon_seconds, bool) or not isinstance(horizon_seconds, Real)
            or not math.isfinite(horizon_seconds) or horizon_seconds < 0):
        raise ValueError('horizon_seconds must be finite and nonnegative')
    now, ids = _timestamp(as_of, 'as_of'), _stations(station_ids)
    result = dict(reduction_allowed=True, reasons=[], event_ids=[], motion_coverage='none',
                  latest_movement_at=None, demand_status='observed', unknowns=[],
                  knowledge_mode=knowledge_mode)
    data = pd.DataFrame(frame).copy(deep=True)
    if not {'station_id', 'interval_start', 'interval_end', 'entry_count'} <= set(data):
        raise ValueError('entry frame needs station_id, interval_start, interval_end and entry_count')
    data = data[data.station_id.isin(ids)].copy()
    for key in ('interval_start', 'interval_end'):
        data[key] = pd.to_datetime([_timestamp(value, key) for value in data[key]], utc=True)
    if (data.interval_end <= data.interval_start).any():
        raise ValueError('entry intervals must have a positive duration')
    known = data[data.interval_end.le(now)].copy()
    if 'available_at' not in known:
        result['unknowns'].append('entry_availability_assumed_interval_end')
    for field in ('available_at', 'received_at'):
        if field in known:
            stamps = [_timestamp(value, field) if _present(value) else pd.NaT for value in known[field]]
            known[field] = pd.to_datetime(stamps, utc=True)
            known = known[known[field].notna() & known[field].le(now)]
    latest = known.sort_values(['interval_end', 'interval_start']).groupby('station_id', sort=False).tail(1)
    missing = set(ids)-set(latest.station_id)
    for row in latest.itertuples(index=False):
        value = row.entry_count
        if (not _present(value) or isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0 or int(value) != value
                or now-row.interval_end > pd.Timedelta(hours=1)
                or getattr(row, 'is_complete', True) is not True):
            missing.add(row.station_id)
        if hasattr(row, 'quality_status'):
            quality_at = getattr(row, 'quality_available_at', None)
            if not _present(quality_at) or _timestamp(quality_at, 'quality_available_at') > now:
                result['unknowns'].append('source_quality_unavailable')
            elif row.quality_status == 'source_mismatch':
                result['reasons'].append('source_quality_unreliable')
            elif row.quality_status != 'ok':
                result['unknowns'].append('source_quality_unknown')
        else:
            result['unknowns'].append('source_quality_not_supplied')
    if missing:
        result['reasons'].append('entry_observation_missing')
        result['demand_status'] = 'unknown'
        result['unknowns'].append('entry_observation_missing')
    for sid in ids:
        if registry is None:
            result['unknowns'].append('event_registry_not_supplied')
            continue
        sample = latest[latest.station_id.eq(sid)]
        states = [registry.state_at(sid, now, now, knowledge_mode=knowledge_mode)]
        if horizon_seconds:
            if hasattr(registry, 'state_during'):
                states.append(registry.state_during(sid, now, now+pd.Timedelta(seconds=horizon_seconds),
                                                   now, knowledge_mode=knowledge_mode))
            else:
                result['unknowns'].append('event_horizon_not_supported')
        for row in sample.itertuples(index=False):
            states.append(registry.state_during(sid, row.interval_start, row.interval_end, now,
                                               knowledge_mode=knowledge_mode)
                          if hasattr(registry, 'state_during') else registry.state_at(
                              sid, row.interval_start, now, knowledge_mode=knowledge_mode))
        for state in states:
            result['event_ids'].extend(state.get('event_ids', []))
            if state.get('demand_censored') or state.get('entrance_state') in {'closed', 'restricted'}:
                result['reasons'].append('demand_censored')
                result['demand_status'] = 'censored'
            if state.get('movement_state') == 'disrupted':
                result['reasons'].append('movement_disrupted')
            elif state.get('movement_state') == 'unknown':
                result['unknowns'].append('event_movement_unknown')
                if set(state.get('event_ids', []))-set(state.get('candidate_event_ids', [])):
                    result['reasons'].append('movement_status_uncertain')
            if state.get('conflicts'):
                result['reasons'].append('event_state_conflict')
            if state.get('unknown_timing'):
                result['unknowns'].append('event_timing_unknown')
    motion = movement_state_at(movement, now, ids, max_age_seconds=max_age_seconds, headway_ratio=headway_ratio)
    result.update(motion_coverage=motion['motion_coverage'], latest_movement_at=motion['latest_movement_at'])
    result['reasons'].extend(motion['reasons'])
    result['unknowns'].extend(motion['unknowns'])
    if motion['movement_state'] == 'disrupted':
        result['reasons'].append('movement_disrupted')
    if require_movement_coverage and motion['motion_coverage'] != 'complete':
        result['reasons'].append('movement_coverage_incomplete')
    for key in ('reasons', 'event_ids', 'unknowns'):
        result[key] = sorted(set(result[key]))
    result['reduction_allowed'] = not result['reasons']
    return result


def _reduction(action):
    if not isinstance(action, dict):
        return False
    delta = action.get('count_delta')
    return ((isinstance(delta, Real) and delta < 0)
            or action.get('kind') in {'WITHDRAW', 'WITHDRAW_ONLY', 'REMOVE', 'REDUCE', 'REDUCE_SERVICE'}
            or bool(action.get('withdrawn_train_ids'))
            or any(_reduction(row) for row in action.get('actions', []))
            or _reduction(action.get('action')))


def guard_candidates(candidates, guard):
    """Copy candidate audits, rejecting reductions while retaining KEEP and releases."""
    result = copy.deepcopy(candidates)
    for row in result:
        row['control_guard'] = copy.deepcopy(guard)
        if not guard['reduction_allowed'] and _reduction(row):
            row.update(status='rejected', reason='control_guard: '+', '.join(guard['reasons']))
    return result
