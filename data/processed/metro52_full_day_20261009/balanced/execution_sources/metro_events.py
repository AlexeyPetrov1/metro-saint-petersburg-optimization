"""Versioned access/movement evidence, with explicit historical knowledge time.

``strict`` uses both publication and ingestion. ``publication_replay`` assumes
an archived publication was available immediately; it is an explicit replay
assumption, not evidence that this pipeline ingested the message then. Unknown
publication times remain unavailable. Retrospective labels may use later flow
evidence and must never be passed to forecasting as online features.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path
import json
import re
from zoneinfo import ZoneInfo


DEFAULT_REGISTRY = Path(__file__).parent / 'config/metro_events/events.json'
MOSCOW = ZoneInfo('Europe/Moscow')
STATION_NAMES = (
    'Проспект Ветеранов', 'Ленинский проспект', 'Автово', 'Кировский завод',
    'Нарвская', 'Балтийская', 'Технологический институт', 'Пушкинская',
    'Владимирская', 'Площадь Восстания', 'Чернышевская', 'Площадь Ленина',
    'Выборгская', 'Лесная', 'Площадь Мужества', 'Политехническая',
    'Академическая', 'Гражданский проспект', 'Девяткино',
)
STATION_ALIASES = {name: f'L1_{i:02}' for i, name in enumerate(STATION_NAMES, 1)}
STATION_ALIASES.update({'Технологический институт-1': 'L1_07', 'Пл. Восстания-1': 'L1_10',
                        'Пл. Восстания-2': 'L1_10', 'Гражданский пр.': 'L1_18'})
VESTIBULE_STATIONS = {
    'V01': 'L1_01', 'V02': 'L1_01', 'V03': 'L1_02', 'V04': 'L1_02',
    **{f'V{i:02}': f'L1_{i-2:02}' for i in range(5, 13)},
    'V13': 'L1_10', 'V14': 'L1_11', 'V15': 'L1_12', 'V16': 'L1_12',
    **{f'V{i:02}': f'L1_{i-4:02}' for i in range(17, 23)},
    'V23': 'L1_19', 'V24': 'L1_19',
}


def _timestamp(value, field='timestamp'):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'{field} must be a timezone-aware ISO timestamp') from exc


def _date(value):
    return date.fromisoformat(value)


def _clock(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{2}:\d{2}(:\d{2})?', value):
        raise ValueError('recurring windows require local HH:MM times')
    return time.fromisoformat(value)


def _validate(record):
    for field in ('event_id', 'version_id', 'kind'):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f'{field} must be nonempty')
    if type(record.get('version')) is not int or record['version'] < 1:
        raise ValueError('version must be a positive integer')
    if record.get('confidence') not in ('confirmed', 'candidate'):
        raise ValueError('confidence must be confirmed or candidate')
    if record.get('entrance_state', 'open') not in ('open', 'restricted', 'closed', 'unknown'):
        raise ValueError('invalid entrance_state')
    if record.get('movement_state', 'normal') not in ('normal', 'disrupted', 'unknown'):
        raise ValueError('invalid movement_state')
    if not isinstance(record.get('sources'), list) or not record['sources']:
        raise ValueError('sources must preserve at least one evidence source')
    _timestamp(record.get('ingested_at'), 'ingested_at')
    if record.get('published_at') is not None:
        _timestamp(record['published_at'], 'published_at')
    if record.get('conflicts_known_at') is not None:
        _timestamp(record['conflicts_known_at'], 'conflicts_known_at')
    scope = record.get('scope', {})
    if scope.get('level') not in ('network', 'line', 'station', 'vestibule', 'track'):
        raise ValueError('invalid event scope')
    line = record.get('line_id')
    if scope['level'] != 'network' and line not in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6'):
        raise ValueError('non-network scopes require an explicit line_id')
    stations = scope.get('station_ids', [])
    if scope['level'] in ('station', 'vestibule') and not stations:
        raise ValueError('station/vestibule scope requires station_ids')
    if any(not isinstance(s, str) or not s.startswith(line + '_') for s in stations):
        raise ValueError('station scope must belong to the declared line')
    if scope['level'] == 'vestibule' and not scope.get('vestibule_ids'):
        raise ValueError('vestibule scope requires vestibule_ids')
    if scope['level'] == 'track' and not scope.get('track_ids'):
        raise ValueError('track scope requires track_ids')
    for vestibule in scope.get('vestibule_ids', []):
        if line == 'L1' and VESTIBULE_STATIONS.get(vestibule) not in stations:
            raise ValueError('vestibule does not belong to scoped station')
    timing = record.get('timing', {})
    if not timing or any(key not in ('planned', 'observed') for key in timing):
        raise ValueError('timing must contain planned and/or observed evidence')
    for window in timing.values():
        if not any(window.get(key) for key in ('start', 'date_start')):
            raise ValueError('timing requires exact start or date_start')
        for field in ('start', 'end', 'knowledge_at', 'restoration_report_at'):
            if window.get(field) is not None:
                _timestamp(window[field], field)
        if window.get('start') and window.get('end') and _timestamp(window['end']) <= _timestamp(window['start']):
            raise ValueError('event end must follow its start')
        for field in ('date_start', 'date_end'):
            if window.get(field):
                _date(window[field])
        if window.get('date_start') and window.get('date_end') and _date(window['date_end']) < _date(window['date_start']):
            raise ValueError('date_end must not precede date_start')
        for recurring in window.get('windows', []) + window.get('comparison_windows', []):
            weekdays = recurring.get('weekdays', [])
            if not weekdays or any(type(w) is not int or not 0 <= w <= 6 for w in weekdays):
                raise ValueError('weekdays must be a nonempty subset of 0..6')
            if _clock(recurring['start_time']) == _clock(recurring['end_time']):
                raise ValueError('equal recurring bounds are ambiguous')
            if recurring.get('entrance_state', 'open') not in ('open', 'closed', 'restricted', 'unknown'):
                raise ValueError('invalid recurring entrance_state')


def _contains(timing, at):
    """Return inactive/active/uncertain and matched recurring window."""
    local = at.astimezone(MOSCOW)
    start = _timestamp(timing['start']) if timing.get('start') else None
    end = _timestamp(timing['end']) if timing.get('end') else None
    if start and at < start or end and at >= end:
        return 'inactive', None
    report = timing.get('restoration_report_at')
    if report and at >= _timestamp(report):
        return 'inactive', None  # Restored by the report, exact end still unknown.
    day_start = _date(timing['date_start']) if timing.get('date_start') else None
    day_end = _date(timing['date_end']) if timing.get('date_end') else None
    recurring = timing.get('windows', [])
    if recurring:
        for window in recurring:
            low, high = _clock(window['start_time']), _clock(window['end_time'])
            owner = local.date()
            clock = local.timetz().replace(tzinfo=None)
            if low < high:
                inside = low <= clock < high
            else:
                inside = clock >= low or clock < high
                if clock < high:
                    owner -= timedelta(days=1)
            if inside and owner.weekday() in window['weekdays'] and (not day_start or owner >= day_start) and (not day_end or owner <= day_end):
                return ('uncertain' if timing.get('uncertain') else 'active'), window
        return 'inactive', None
    if day_start and local.date() < day_start or day_end and local.date() > day_end:
        return 'inactive', None
    if timing.get('precision') == 'date':
        return 'uncertain', None
    if start and not end and not timing.get('open_ended'):
        return ('active' if at == start else 'uncertain'), None
    return 'active', None


def _observation_covers(timing, at):
    """A measured July morning does not describe other months or evenings."""
    support = {'date_start': timing.get('date_start'), 'date_end': timing.get('date_end'),
               'windows': timing.get('comparison_windows', [])}
    if timing.get('start'):
        support['date_start'] = _timestamp(timing['start']).astimezone(MOSCOW).date().isoformat()
    if timing.get('end'):
        support['date_end'] = _timestamp(timing['end']).astimezone(MOSCOW).date().isoformat()
    return _contains(support, at)[0] != 'inactive'


class EventRegistry:
    def __init__(self, records):
        self.records = deepcopy(list(records))
        versions, identifiers = set(), set()
        for record in self.records:
            _validate(record)
            pair = (record['event_id'], record['version'])
            if pair in versions or record['version_id'] in identifiers:
                raise ValueError('event versions and version_ids must be unique')
            versions.add(pair)
            identifiers.add(record['version_id'])

    @classmethod
    def from_records(cls, records):
        return cls(records)

    @classmethod
    def load(cls, path=DEFAULT_REGISTRY):
        document = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(document, dict) or document.get('schema_version') != 1:
            raise ValueError('unsupported event registry schema')
        return cls(document['events'])

    def visible(self, as_of, knowledge_mode='strict'):
        as_of = _timestamp(as_of, 'as_of')
        if knowledge_mode not in ('strict', 'publication_replay'):
            raise ValueError('knowledge_mode must be strict or publication_replay')
        return self._latest(r for r in self.records if r.get('published_at') and
            _timestamp(r['published_at']) <= as_of and
            (knowledge_mode == 'publication_replay' or _timestamp(r['ingested_at']) <= as_of))

    @staticmethod
    def _latest(records):
        chosen = {}
        for record in records:
            if record['event_id'] not in chosen or chosen[record['event_id']]['version'] < record['version']:
                chosen[record['event_id']] = record
        return [deepcopy(chosen[key]) for key in sorted(chosen)]

    def state_at(self, station_id, interval_start, as_of, vestibule_id=None, knowledge_mode='strict'):
        return self._state(station_id, _timestamp(interval_start, 'interval_start'),
                           self.visible(as_of, knowledge_mode), vestibule_id,
                           knowledge_mode, _timestamp(as_of))

    def label_at(self, station_id, interval_start, vestibule_id=None):
        return self._state(station_id, _timestamp(interval_start, 'interval_start'),
                           self._latest(self.records), vestibule_id, 'retrospective', None)

    def state_during(self, station_id, interval_start, interval_end, as_of,
                     vestibule_id=None, knowledge_mode='strict'):
        return self._during(station_id, interval_start, interval_end,
            self.visible(as_of, knowledge_mode), vestibule_id, knowledge_mode, _timestamp(as_of))

    def label_during(self, station_id, interval_start, interval_end, vestibule_id=None):
        return self._during(station_id, interval_start, interval_end,
                            self._latest(self.records), vestibule_id, 'retrospective', None)

    def _during(self, station, start, end, records, vestibule, mode, as_of):
        start, end = _timestamp(start, 'interval_start'), _timestamp(end, 'interval_end')
        if end <= start:
            raise ValueError('interval_end must follow interval_start')
        boundaries = {start, end}
        local_first = start.astimezone(MOSCOW).date() - timedelta(days=1)
        local_last = end.astimezone(MOSCOW).date()
        for record in records:
            for name, timing in record['timing'].items():
                if name == 'observed' and mode != 'retrospective' and _timestamp(
                        timing.get('knowledge_at') or record['ingested_at']) > as_of:
                    continue
                for key in ('start', 'end', 'restoration_report_at'):
                    if timing.get(key):
                        boundary = _timestamp(timing[key])
                        if start < boundary < end:
                            boundaries.add(boundary)
                for key in ('date_start', 'date_end'):
                    if timing.get(key):
                        day = _date(timing[key]) + (timedelta(days=1) if key == 'date_end' else timedelta())
                        boundary = datetime.combine(day, time(), MOSCOW)
                        if start < boundary < end:
                            boundaries.add(boundary)
                day = local_first
                while day <= local_last:
                    for window in timing.get('windows', []) + timing.get('comparison_windows', []):
                        for key in ('start_time', 'end_time'):
                            boundary = datetime.combine(day, _clock(window[key]), MOSCOW)
                            if start < boundary < end:
                                boundaries.add(boundary)
                    day += timedelta(days=1)
        points = sorted(boundaries)
        states = [self._state(station, left+(right-left)/2, records, vestibule, mode, as_of)
                  for left, right in zip(points, points[1:])]
        result = deepcopy(states[0])
        for key in ('event_ids', 'version_ids', 'event_kinds', 'candidate_event_ids', 'conflicts'):
            result[key] = sorted({item for state in states for item in state[key]})
        result['entrance_state'] = max((s['entrance_state'] for s in states),
            key={'open': 0, 'restricted': 1, 'closed': 2, 'unknown': 3}.get)
        result['movement_state'] = max((s['movement_state'] for s in states),
            key={'normal': 0, 'disrupted': 1, 'unknown': 2}.get)
        result['unknown_timing'] = any(s['unknown_timing'] for s in states)
        result['demand_censored'] = any(s['demand_censored'] for s in states)
        regimes = sorted({s['regime_id'] for s in states})
        result['regime_id'] = '+'.join(regimes)
        result['interval_mixed_regimes'] = len(regimes) > 1
        result['interval_seconds'] = (end-start).total_seconds()
        result['censored_seconds'] = sum((right-left).total_seconds() for left, right, state
            in zip(points, points[1:], states) if state['demand_censored'])
        result['closed_seconds'] = sum((right-left).total_seconds() for left, right, state
            in zip(points, points[1:], states) if state['entrance_state'] == 'closed')
        return result

    def _state(self, station_id, at, records, vestibule_id, mode, as_of):
        station = STATION_ALIASES.get(station_id, station_id)
        if not isinstance(station, str) or not re.fullmatch(r'L[1-6]_[A-Za-z0-9_]+', station):
            raise ValueError('station_id must be an explicit local id or configured alias')
        if vestibule_id in VESTIBULE_STATIONS and VESTIBULE_STATIONS[vestibule_id] != station:
            raise ValueError('vestibule_id does not belong to station_id')
        result = dict(entrance_state='open', movement_state='normal', event_ids=[], version_ids=[],
                      event_kinds=[], candidate_event_ids=[], conflicts=[], unknown_timing=False,
                      demand_censored=False, regime_id='ordinary', knowledge_mode=mode)
        entrance_rank = {'open': 0, 'restricted': 1, 'closed': 2, 'unknown': 3}
        movement_rank = {'normal': 0, 'disrupted': 1, 'unknown': 2}
        for record in records:
            scope = record['scope']
            if scope['level'] != 'network' and record['line_id'] != station.split('_')[0]:
                continue
            if scope.get('station_ids') and station not in scope['station_ids']:
                continue
            if scope['level'] == 'vestibule' and vestibule_id is not None and vestibule_id not in scope['vestibule_ids']:
                continue
            timings = record['timing']
            planned = timings.get('planned')
            observed = timings.get('observed')
            observed_known = observed and (mode == 'retrospective' or
                _timestamp(observed.get('knowledge_at') or record['ingested_at']) <= as_of)
            if observed_known and planned and not _observation_covers(observed, at):
                observed_known = False
            selected = observed if observed_known else planned
            if selected is None:
                continue
            activity, window = _contains(selected, at)
            conflict_here = False
            if observed_known and planned and record.get('conflicts'):
                planned_activity, _ = _contains(planned, at)
                conflict_here = activity != planned_activity
                if activity == 'inactive' and planned_activity != 'inactive':
                    activity = 'uncertain'
            if activity == 'inactive':
                continue
            result['event_ids'].append(record['event_id'])
            result['version_ids'].append(record['version_id'])
            result['event_kinds'].append(record['kind'])
            conflicts_known = mode == 'retrospective' or _timestamp(
                record.get('conflicts_known_at') or record['ingested_at']) <= as_of
            if conflicts_known:
                result['conflicts'].extend(record.get('conflicts', []))
            result['unknown_timing'] |= activity == 'uncertain' or conflict_here
            if record['confidence'] == 'candidate':
                result['candidate_event_ids'].append(record['event_id'])
                continue
            entrance = (window or {}).get('entrance_state', record.get('entrance_state', 'open'))
            movement = record.get('movement_state', 'normal')
            if activity == 'uncertain' or conflict_here:
                if entrance != 'open':
                    entrance = 'unknown'
                if movement != 'normal':
                    movement = 'unknown'
            if entrance == 'closed' and scope['level'] == 'vestibule' and vestibule_id is None:
                entrance = 'restricted'
            if entrance_rank[entrance] > entrance_rank[result['entrance_state']]:
                result['entrance_state'] = entrance
            if movement_rank[movement] > movement_rank[result['movement_state']]:
                result['movement_state'] = movement
        result['conflicts'] = sorted(set(result['conflicts']))
        result['event_kinds'] = sorted(set(result['event_kinds']))
        result['demand_censored'] = result['entrance_state'] in ('restricted', 'closed', 'unknown')
        regimes = [v for v, e in zip(result['version_ids'], result['event_ids']) if e not in result['candidate_event_ids']]
        if regimes:
            result['regime_id'] = '+'.join(regimes) + ':' + result['entrance_state'] + ':' + result['movement_state']
        return result
