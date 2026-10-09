"""Explicit withdrawal destinations and nonpassenger access mileage."""

from copy import deepcopy
from decimal import Decimal
from heapq import heappop, heappush
import re


SITES = [('L1_01', 'VETERANS', 'Тупики за станцией «Проспект Ветеранов»'),
         ('L1_17', 'AKADEMICHESKAYA', 'Тупики за станцией «Академическая»'),
         ('L1_12', 'LENIN_SQUARE', 'Путевой комплекс за станцией «Площадь Ленина»')]


def attach_stabling(data):
    out = deepcopy(data)
    stations = {s['station_id']: s for s in out['stations']}
    for track in out['tracks']:
        endpoints = {track['from_station'], track['to_station']}
        if 'DEPOT_AVTOVO' in endpoints:
            track['length_km'] = '1.35'
        elif 'DEPOT_NORTHERN' in endpoints:
            track['length_km'] = '1'
    sites = []
    track_ids = {t['track_id'] for t in out['tracks']}
    for station, suffix, label in SITES:
        if station not in stations:
            continue
        siding = 'SIDING_' + suffix
        sites.append(dict(station_id=station, siding_id=siding, label=label))
        if siding not in stations:
            out['stations'].append(dict(station_id=siding, kind='siding', stop_seconds=0, turnaround_seconds=180))
        for prefix, start, end in [('EXIT', station, siding), ('ENTRY', siding, station)]:
            identity = prefix + '_' + siding
            if identity not in track_ids:
                out['tracks'].append(dict(track_id=identity, from_station=start, to_station=end,
                                         length_km='0.35', travel_seconds=42))
                track_ids.add(identity)
    out['withdrawal_stabling'] = dict(siding_share=0.35, depot_share=0.65, sites=sites,
        source='User supplied red-line withdrawal shares and access distances',
        assumptions=dict(siding_travel_seconds=42, siding_turnaround_seconds=180,
                         timing_source='Demo 1km/120s speed; siding reversal assumed 180s',
                         siding_capacity=None, depot_access_travel_seconds=120),
        allocation='Rounded cumulative 35 percent of committed withdrawals; sorted selected train IDs')
    return out


def is_withdrawal_trip(identity):
    if identity.startswith('G47CLOSE:'):
        return False
    return bool(identity.startswith(('G47WITHDRAW:', 'G48TEMP:')) and identity.endswith(':EXIT')
                or re.search(r':EXIT[0-9]+\Z', identity))


def withdrawal_count(data):
    return sum(is_withdrawal_trip(t['trip_id']) for plan in data['plans'] for t in plan['trips'])


def _line_path(data, start, end):
    if start == end:
        return []
    stations = {s['station_id']: s for s in data['stations']}
    edges = {}
    for track in data['tracks']:
        if all(stations[track[k]]['kind'] == 'line' for k in ('from_station', 'to_station')):
            edges.setdefault(track['from_station'], []).append(track)
    queue = [(Decimal(0), (), start)]; visited = set()
    while queue:
        distance, path, station = heappop(queue)
        if station == end:
            return list(path)
        if station in visited:
            continue
        visited.add(station)
        for track in edges.get(station, []):
            heappush(queue, (distance + Decimal(str(track['length_km'])),
                            path + (track['track_id'],), track['to_station']))
    raise ValueError(f'no physical line path from {start} to {end}')


def _totals(route, tracks, stations):
    km = sum((Decimal(str(tracks[t]['length_km'])) for t in route), Decimal(0))
    seconds = sum(tracks[t]['travel_seconds'] for t in route)
    seconds += sum(stations[tracks[t]['to_station']]['stop_seconds']
                   + stations[tracks[t]['to_station']]['turnaround_seconds'] for t in route[:-1])
    return str(km.normalize()), seconds


def routing(data, train_id, origin, sequence=0):
    if type(sequence) is not int or sequence < 0:
        raise ValueError('withdrawal sequence must be a nonnegative integer')
    stations = {s['station_id']: s for s in data['stations']}
    tracks = {t['track_id']: t for t in data['tracks']}
    if origin not in stations or stations[origin]['kind'] != 'line':
        raise ValueError('withdrawal origin must be a known line station')
    config = data.get('withdrawal_stabling', {})
    # Integer arithmetic implements floor((n + 1)*.35 + .5) - floor(n*.35 + .5).
    siding_slot = ((sequence + 1) * 35 + 50) // 100 > (sequence * 35 + 50) // 100
    sites = config.get('sites', []) if siding_slot else []
    choices = []
    if sites:
        for site in sites:
            approach = _line_path(data, origin, site['station_id'])
            exit_route = approach + ['EXIT_' + site['siding_id']]
            choices.append((_totals(exit_route, tracks, stations)[0], site['siding_id'],
                            site['label'], exit_route,
                            ['ENTRY_' + site['siding_id']] + _line_path(data, site['station_id'], origin), 'siding'))
    else:
        for track in tracks.values():
            if not track['track_id'].startswith('EXIT_') or stations[track['to_station']]['kind'] != 'depot':
                continue
            terminal, depot = track['from_station'], track['to_station']
            if not config and terminal != origin:
                continue
            entry = next(t['track_id'] for t in tracks.values()
                         if t['from_station'] == depot and t['to_station'] == terminal and t['track_id'].startswith('ENTRY_'))
            exit_route = _line_path(data, origin, terminal) + [track['track_id']]
            label = 'Депо «Автово»' if depot == 'DEPOT_AVTOVO' else 'Депо «Северное»'
            choices.append((_totals(exit_route, tracks, stations)[0], depot, label, exit_route,
                            [entry] + _line_path(data, terminal, origin), 'depot'))
    if not choices:
        raise ValueError(f'no declared withdrawal destination from {origin}')
    _, destination, label, exit_route, entry_route, kind = min(choices, key=lambda r: (Decimal(r[0]), r[1]))
    exit_km, exit_seconds = _totals(exit_route, tracks, stations)
    entry_km, entry_seconds = _totals(entry_route, tracks, stations)
    release = data.get('headway_policy', {}).get('parameters', {}).get('depot_release_s', {}).get('value', {})
    release_seconds = 0 if kind == 'siding' else release.get('Avtovo' if destination == 'DEPOT_AVTOVO' else 'Northern',
                                                         720 if destination == 'DEPOT_AVTOVO' else 780)
    return dict(train_id=train_id, kind=kind, destination=destination, label=label, sequence=sequence,
                exit_route=exit_route, entry_route=entry_route, exit_track=exit_route[0], entry_track=entry_route[0],
                exit_km=exit_km, entry_km=entry_km, exit_seconds=exit_seconds, entry_seconds=entry_seconds,
                release_seconds=release_seconds)


def assign_routes(data, train_ids, origins):
    names = sorted(train_ids)
    if len(names) != len(set(names)) or set(names) - set(origins):
        raise ValueError('distinct selected trains and their origins are required')
    count = withdrawal_count(data)
    return {tid: routing(data, tid, origins[tid], count + i) for i, tid in enumerate(names)}


def report(data, motion):
    """Report planned destinations and measured arrivals without changing mileage."""
    stations = {s['station_id']: s for s in data['stations']}
    tracks = {t['track_id']: t for t in data['tracks']}
    config = data.get('withdrawal_stabling', {})
    labels = {s['siding_id']: s['label'] for s in config.get('sites', [])}
    labels.update(DEPOT_AVTOVO='Депо «Автово»', DEPOT_NORTHERN='Депо «Северное»')
    arrivals = {(e['train_id'], e['trip_id'], e['location']): e['second']
                for e in motion['events'] if e['event_type'] == 'arrival'}
    departures = {(r['train_id'], r['trip_id']): r['departure_second']
                  for r in data.get('leg_timing', []) if r['leg_index'] == 0}
    rows = []
    for plan in data['plans']:
        if motion.get('plan_id') is not None and plan.get('plan_id') != motion['plan_id']:
            continue
        for trip in plan['trips']:
            if not is_withdrawal_trip(trip['trip_id']) or trip.get('passenger', False):
                continue
            destination = tracks[trip['route'][-1]]['to_station']
            kind = stations[destination]['kind']
            if kind not in ('depot', 'siding'):
                continue
            arrival = arrivals.get((trip['train_id'], trip['trip_id'], destination))
            rows.append(dict(train_id=trip['train_id'], trip_id=trip['trip_id'], kind=kind,
                destination=destination, label=labels.get(destination, destination),
                departure_second=departures.get((trip['train_id'], trip['trip_id']), trip['departure_second']),
                arrival_second=arrival, pending=arrival is None,
                exit_route=list(trip['route']), exit_km=_totals(trip['route'], tracks, stations)[0]))
    rows.sort(key=lambda r: (r['departure_second'], r['train_id'], r['trip_id']))
    completed = [r for r in rows if not r['pending']]
    counts = {kind: sum(r['kind'] == kind for r in completed) for kind in ('siding', 'depot')}
    latest = {r['train_id']: r for r in sorted(completed, key=lambda r: r['arrival_second'])}
    final = {r['train_id']: r for r in motion['final_state']}
    parked = [dict(row, state=final[tid]['state']) for tid, row in latest.items()
              if tid in final and final[tid]['state'] != 'moving' and final[tid]['location'] == row['destination']]
    return dict(policy_targets=dict(siding=config.get('siding_share', 0.35), depot=config.get('depot_share', 0.65)),
        planned_withdrawals=rows, parked_trains=sorted(parked, key=lambda r: r['train_id']),
        planned_counts={kind: sum(r['kind'] == kind for r in rows) for kind in ('siding', 'depot')},
        completed_counts=counts, completed_shares={kind: counts[kind] / len(completed) if completed else None for kind in counts},
        pending_count=len(rows) - len(completed), assumptions=deepcopy(config.get('assumptions', {})),
        share_scope='Completed withdrawal arrivals; currently parked trains are listed separately')
