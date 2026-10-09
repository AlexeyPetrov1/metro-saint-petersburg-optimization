"""Withdraw at a known terminal without changing any retained movement clock."""

import copy
from decimal import Decimal

from dwell import timing_key
from goal43_policy import parameter
from metro44_assignment_plan import (_current_movements_preserved, _deadline,
                                     build_assignment_problem, terminal_resources)
from metro44_linked_gap_repair import release_constraints
from metro44_planner import trip_finish
from rolling import operational_headways
from simulation import digest, simulate
from withdrawal_stabling import assign_routes, routing, withdrawal_count


def _index(data):
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    tracks = {r['track_id']: r for r in data['tracks']}
    stations = {r['station_id']: r for r in data['stations']}
    first = lambda trip: timing[timing_key(trip['train_id'], trip['trip_id'], 0)]['departure_second']
    return timing, tracks, stations, first


def withdrawal_options(data, as_of_second, service_end_second, deadline=None, *, assigned_routes=None):
    """Concrete next-ready vehicles; committed depot preparation is retained."""
    _deadline(deadline)
    problem = build_assignment_problem(data, as_of_second)
    timing, tracks, stations, first = _index(data)
    retained = set(problem['retained_ids']); options = []
    for available in problem['availability']:
        _deadline(deadline)
        ready = available['ready_second']; tid = available['train_id']
        if not available['on_line'] or available.get('locked_exit') or not as_of_second <= ready <= as_of_second + 900:
            continue
        owned = [t for t in data['plans'][0]['trips'] if t['train_id'] == tid]
        cancelled = [t for t in owned if t['trip_id'] not in retained]
        services = [t for t in cancelled if t['passenger']]
        if not services or any(first(t) < ready for t in cancelled):
            continue
        route = (assigned_routes or {}).get(tid) or routing(data, tid, available['location'], withdrawal_count(data))
        km = sum((Decimal(str(tracks[r]['length_km'])) for t in cancelled for r in t['route']), Decimal(0)) - Decimal(route['exit_km'])
        price = parameter(data['headway_policy'], 'maintenance_prices_rub_by_type')[available['train_type']]['TO1']
        rate = Decimal(str(price)) / Decimal(str(parameter(data['headway_policy'], 'to1_interval_km')))
        options.append(dict(train_id=tid, train_type=available['train_type'], terminal=available['location'],
            ready_second=ready, exit_second=ready, exit_track=route['exit_route'][0],
            withdrawal_destination=route,
            begun_service_ids=[t['trip_id'] for t in owned if t['passenger'] and first(t) < as_of_second],
            preserved_trip_ids=[t['trip_id'] for t in owned if t['trip_id'] in retained],
            cancel_service_ids=[t['trip_id'] for t in services],
            cancel_exit_ids=[t['trip_id'] for t in cancelled if t['route'][-1].startswith('EXIT_')],
            cancel_entry_ids=[t['trip_id'] for t in cancelled if t['route'][0].startswith('ENTRY_')],
            saved_km=str(km), to1_resource_saving_rub=str(km * rate)))
    _deadline(deadline)
    return sorted(options, key=lambda row: (row['ready_second'], row['train_id']))


def append_offline_movement(data, train_id, identity, route, second, *, planned_second=None):
    tracks = {t['track_id']: t for t in data['tracks']}
    stations = {s['station_id']: s for s in data['stations']}
    data['plans'][0]['trips'].append(dict(trip_id=identity, train_id=train_id,
        departure_second=second if planned_second is None else planned_second, route=route, passenger=False))
    for index, track in enumerate(route):
        station = stations[tracks[track]['to_station']]
        service = station['stop_seconds'] + station['turnaround_seconds']
        data['leg_timing'].append(dict(train_id=train_id, trip_id=identity, leg_index=index,
            departure_second=second, service_seconds=service, holding_seconds=0, approach_holding_seconds=0))
        second += tracks[track]['travel_seconds'] + service


def _change(data, removed, exits, as_of, end, deadline, before=None):
    _deadline(deadline)
    timing, tracks, stations, _ = _index(data)
    changed = copy.deepcopy(data)
    changed['plans'][0]['trips'] = [t for t in changed['plans'][0]['trips'] if t['trip_id'] not in removed]
    changed['leg_timing'] = [r for r in changed['leg_timing'] if r['trip_id'] not in removed]
    added = []
    for exit in exits:
        tid, second, track, identity = exit['train_id'], exit['exit_second'], exit['exit_track'], exit['trip_id']
        if identity in {t['trip_id'] for t in changed['plans'][0]['trips']}:
            raise ValueError('new withdrawal EXIT identity collides')
        route = exit.get('withdrawal_destination', {}).get('exit_route', [track])
        append_offline_movement(changed, tid, identity, route, second)
        added.append(identity)
    _deadline(deadline)
    before = before or simulate(data, data['plans'][0]['plan_id'])
    motion = simulate(changed, changed['plans'][0]['plan_id'])
    prefix = [e for e in before['events'] if e['second'] < as_of] == [e for e in motion['events'] if e['second'] < as_of]
    current = _current_movements_preserved(data, changed, as_of)
    if not prefix or not current:
        raise ValueError('withdrawal changes executed prefix or a current arrival')
    resources = terminal_resources(changed, motion)
    gaps = operational_headways(changed, motion, as_of, end)
    releases = release_constraints(changed, 0)
    prices = parameter(data['headway_policy'], 'maintenance_prices_rub_by_type')
    interval = Decimal(str(parameter(data['headway_policy'], 'to1_interval_km')))
    def resource(work):
        return sum(((Decimal(str(r['passenger_train_km'])) + Decimal(str(r['nonpassenger_train_km']))) *
                    Decimal(str(prices[r['train_type']]['TO1'])) / interval for r in work), Decimal(0))
    audit = dict(movement_passed=gaps['passed'] and resources['passed'] and releases['passed'] and motion['obligations_closed'],
        executed_prefix_preserved=prefix, current_movements_preserved=current, kept_clocks_preserved=True,
        cancelled_trip_ids=sorted(removed), appended_exit_ids=added, terminal_resources=resources,
        headways={k: v for k, v in gaps.items() if k not in ('intervals', 'boundary_ages')},
        release_constraints=releases, obligations_closed=motion['obligations_closed'],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'], passenger_checked=False,
        estimated_to1_resource_saving_rub=str(resource(before['transport_work']) - resource(motion['transport_work'])),
        preparation_scope='unchanged assumed source commitments; no new release',
        withdrawal_destinations=[dict(r['withdrawal_destination'])
                                 for r in exits if 'withdrawal_destination' in r],
        release_authorized=False, cash_saving_rub=None, _data_digest=digest(changed),
        _motion_digest=digest(motion), _transport_work=motion['transport_work'])
    _deadline(deadline)
    return changed, audit


def apply_withdrawals(data, options, as_of_second, service_end_second, deadline=None):
    """Apply only canonical options. Failed coverage stays explicit in the audit."""
    _deadline(deadline)
    canonical = {r['train_id']: r for r in withdrawal_options(data, as_of_second, service_end_second, deadline)}
    names = [r['train_id'] for r in options]
    if len(names) != len(set(names)) or set(names) - set(canonical):
        raise ValueError('withdrawal must name distinct canonical source trains')
    for row in options:
        if any(row.get(key) != value for key, value in canonical[row['train_id']].items() if key in (
                'terminal', 'ready_second', 'exit_second', 'exit_track', 'cancel_service_ids', 'cancel_exit_ids', 'cancel_entry_ids')):
            raise ValueError('withdrawal option differs from canonical source readiness or duties')
    destinations = assign_routes(data, names, {tid: canonical[tid]['terminal'] for tid in names})
    canonical = {r['train_id']: r for r in withdrawal_options(data, as_of_second, service_end_second,
        deadline, assigned_routes=destinations)}
    removed = {identity for name in names for key in ('cancel_service_ids', 'cancel_exit_ids', 'cancel_entry_ids')
               for identity in canonical[name][key]}
    exits = [dict(canonical[name], trip_id=f'G47WITHDRAW:{as_of_second}:{name}:EXIT') for name in names]
    changed, audit = _change(data, removed, exits, as_of_second, service_end_second, deadline)
    audit.update(selected_train_ids=names, cancelled_service_ids=sorted(
        identity for name in names for identity in canonical[name]['cancel_service_ids']))
    return changed, audit


def build_withdrawal_candidate(data, second, planning_end, *, selected_train_ids, deadline=None, phase_seconds=0):
    if phase_seconds != 0:
        raise ValueError('withdrawal does not retime a phase')
    names = list(selected_train_ids)
    available = {r['train_id']: r for r in withdrawal_options(data, second, planning_end, deadline)}
    if set(names) - set(available):
        raise ValueError('withdrawal uses an unavailable source train')
    return apply_withdrawals(data, [available[name] for name in names], second, planning_end, deadline)


def close_service_at(data, as_of_second, service_end_second, deadline=None):
    """Normalize both arms to one service cutoff; preserve every retained clock."""
    _deadline(deadline)
    problem = build_assignment_problem(data, as_of_second)
    timing, tracks, stations, first = _index(data)
    retained = set(problem['retained_ids'])
    tail = {t['trip_id'] for t in data['plans'][0]['trips'] if t['passenger'] and
            first(t) >= service_end_second and t['trip_id'] not in retained}
    if not tail:
        return _change(data, set(), [], as_of_second, service_end_second, deadline, problem['source_motion'])
    removed = set(tail); exits = []
    for train in data['trains']:
        _deadline(deadline)
        tid = train['train_id']
        owned = sorted((t for t in data['plans'][0]['trips'] if t['train_id'] == tid), key=first)
        for i, trip in enumerate(owned):
            if trip['trip_id'] in retained or trip['passenger']:
                continue
            following = next((t for t in owned[i + 1:] if t['passenger']), None)
            if trip['route'][-1].startswith('EXIT_') or trip['route'][0].startswith('ENTRY_') and (following is None or following['trip_id'] in tail):
                removed.add(trip['trip_id'])
        kept = [t for t in owned if t['trip_id'] not in removed]
        location, ready = (trip_finish(kept[-1], timing, tracks, stations) if kept else
                           (train['location'], train['ready_after_seconds']))
        if location not in data['headway_policy']['demo']['terminal_parking_capacity_by_station']:
            continue
        exit_track = next(t for t in tracks.values() if t['from_station'] == location and t['track_id'].startswith('EXIT_'))
        exits.append(dict(train_id=tid, exit_second=max(as_of_second, ready), exit_track=exit_track['track_id'],
                          trip_id=f'G47CLOSE:{service_end_second}:{tid}:EXIT'))
    changed, audit = _change(data, removed, exits, as_of_second, service_end_second, deadline, problem['source_motion'])
    audit.update(service_cutoff_second=service_end_second, cancelled_service_ids=sorted(tail),
                 normalization_scope='same finite cohort and service cutoff; no passenger clock retiming')
    return changed, audit
