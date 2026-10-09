"""Build a daily vehicle plan directly from readiness and reserved track clocks."""

import bisect
import copy
import heapq
import math
from datetime import datetime, timedelta

from dwell import timing_key
from goal43_policy import minimum_headway, parameter
from metro_daily import _geometry, reference_counts
from metro44_assignment_plan import _current_movements_preserved, _deadline, terminal_resources
from metro44_linked_gap_repair import release_constraints
from metro44_planner import trip_finish
from metro47_reference import validate_readiness
from rolling import operational_headways
from simulation import digest, simulate


def route_minimum(data, terminal, second):
    tracks, stations, routes, _ = _geometry(data)
    origin = datetime.fromisoformat(data['period_start']); at = second; result = 0
    for track in routes[terminal]:
        result = max(result, minimum_headway(origin + timedelta(seconds=at), data['headway_policy']))
        section = tracks[track]; station = stations[section['to_station']]
        at += section['travel_seconds'] + station['stop_seconds'] + station['turnaround_seconds']
    return result


def build_direct_reference(data, roster, service_end, *, deadline=None, strengthen_peak=False):
    _deadline(deadline); validate_readiness(data, roster)
    tracks, stations, routes, half = _geometry(data)
    origin = datetime.fromisoformat(data['period_start']); policy = data['headway_policy']
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    first = lambda t: timing[timing_key(t['train_id'], t['trip_id'], 0)]['departure_second']
    kept = []
    for train in data['trains']:
        if train['location'].startswith('DEPOT_'):
            continue
        owned = sorted((t for t in data['plans'][0]['trips'] if t['train_id'] == train['train_id']), key=first)
        kept.append(next(t for t in owned if t['passenger']))
    identities = {t['trip_id'] for t in kept}
    out = copy.deepcopy(data); out['plans'][0]['trips'] = copy.deepcopy(kept)
    out['leg_timing'] = [copy.deepcopy(r) for r in data['leg_timing'] if r['trip_id'] == '@initial' or r['trip_id'] in identities]
    clocks = {key: [] for key in tracks}; visits = {terminal: [] for terminal in routes}; states = {}; last = {t: -1 for t in routes}
    for row in data.get('previous_departures', []):
        clocks[row['track_id']].append(row['departure_second'])
    owned = {t['train_id']: t for t in kept}
    for row in out['leg_timing']:
        trip = owned.get(row['train_id'])
        track = next(t['location'] for t in data['trains'] if t['train_id'] == row['train_id']) if row['trip_id'] == '@initial' else trip['route'][row['leg_index']]
        clocks[track].append(row['departure_second'])
    for key in clocks: clocks[key] = sorted(set(clocks[key]))
    for terminal, route in routes.items():
        starts = [first(t) for t in kept if t['route'][0] == route[0]]
        if starts: last[terminal] = max(starts)
    for train in data['trains']:
        tid = train['train_id']; trip = owned.get(tid)
        location, ready = trip_finish(trip, timing, tracks, stations) if trip else (train['location'], train['ready_after_seconds'])
        visit = [ready - 180, ready] if location in routes else None
        if visit is not None: visits[location].append(visit)
        states[tid] = dict(location=location, ready=ready, visit=visit, active=trip is not None,
                           spans=[[0, 10**9]] if trip else [], approach_row=None)
    def min_at(second): return minimum_headway(origin + timedelta(seconds=second), policy)
    def platform_ok(terminal, span, replaced=None):
        changes = [(span[0], 1), (span[1], -1)]
        for other in visits[terminal]:
            if other is not replaced and other[0] < span[1] and span[0] < other[1]:
                changes.extend([(max(span[0], other[0]), 1), (min(span[1], other[1]), -1)])
        count = 0
        for _, change in sorted(changes):
            count += change
            if count > 2: return False
        return True
    def capacity(start):
        changes = [(start, 1)]
        for state in states.values():
            for left, right in state['spans']: changes.extend([(left, 1), (right, -1)])
        count = 0
        for _, change in sorted(changes):
            count += change
            if count > 53: return False
        return True
    def entry_window(terminal, second):
        if platform_ok(terminal, [second - 180, second]): return True
        if not strengthen_peak or min_at(second) != 113: return False
        changed = []
        for state in sorted(states.values(), key=lambda s: s['ready']):
            row = state.get('approach_row'); visit = state['visit']
            if not state['active'] or state['location'] != terminal or row is None or state['ready'] <= second or visit[0] >= second:
                continue
            old = visit[:]; changed.append((state, old, row.get('approach_holding_seconds', 0)))
            arrival = second
            while not platform_ok(terminal, [arrival, arrival + 180], visit):
                edges = [v[1] for v in visits[terminal] if v is not visit and v[0] < arrival + 180 and v[1] > arrival]
                arrival = min(edges); _deadline(deadline)
            row['approach_holding_seconds'] += arrival - visit[0]
            visit[:] = [arrival, arrival + 180]; state['ready'] = arrival + 180
            if platform_ok(terminal, [second - 180, second]): return True
        for state, old, holding in changed:
            state['visit'][:] = old; state['ready'] = old[1]; state['approach_row']['approach_holding_seconds'] = holding
        return False
    def feasible_clock(route, departure):
        for _ in range(200):
            _deadline(deadline); offset = 0; updated = departure
            for track in route:
                at = departure + offset; values = clocks[track]; i = bisect.bisect_right(values, at)
                if i and at - values[i - 1] < min_at(at): updated = max(updated, values[i - 1] + min_at(at) - offset)
                if i < len(values) and values[i] - at < min_at(values[i]): updated = max(updated, values[i] + min_at(values[i] + 150) - offset)
                section = tracks[track]; station = stations[section['to_station']]
                offset += section['travel_seconds'] + station['stop_seconds'] + station['turnaround_seconds']
            if updated == departure: return departure
            departure = updated
        raise ValueError('direct track-clock propagation did not converge')
    serial = 0
    def append(tid, departure, route, passenger, arrival_hold=0):
        nonlocal serial
        serial += 1; identity = f'G47DIRECT:{tid}:{serial}'
        out['plans'][0]['trips'].append(dict(train_id=tid, trip_id=identity, departure_second=departure, route=route, passenger=passenger))
        at = departure
        for index, track in enumerate(route):
            section = tracks[track]; station = stations[section['to_station']]
            service = station['stop_seconds'] + station['turnaround_seconds']; hold = arrival_hold if index == len(route) - 1 else 0
            out['leg_timing'].append(dict(train_id=tid, trip_id=identity, leg_index=index,
                departure_second=at, service_seconds=service, holding_seconds=0, approach_holding_seconds=hold))
            bisect.insort(clocks[track], at); at += section['travel_seconds'] + hold + service
        return at
    def exit_train(tid):
        state = states[tid]; terminal = state['location']
        track = next(t for t in tracks.values() if t['from_station'] == terminal and t['track_id'].startswith('EXIT_'))
        dep = feasible_clock([track['track_id']], state['ready'])
        if not platform_ok(terminal, [state['visit'][0], dep], state['visit']):
            conflict = dict(train_id=tid, terminal=terminal, terminal_arrival_second=state['visit'][0],
                ready_second=state['ready'], earliest_exit_second=dep,
                other_terminal_visits=[v for v in visits[terminal] if v is not state['visit'] and v[0] < dep and v[1] > state['visit'][0]])
            error = ValueError(f'direct EXIT cannot clear a shared terminal: {conflict}')
            error.partial_plan = copy.deepcopy(out); error.conflict = conflict
            raise error
        state['visit'][1] = dep; ready = append(tid, dep, [track['track_id']], False)
        state['spans'][-1][1] = dep + track['travel_seconds']
        state.update(location=track['to_station'], ready=ready, active=False, visit=None, approach_row=None)
    def target(second, terminal):
        at = origin + timedelta(seconds=second)
        minimum = route_minimum(data, terminal, second)
        peak = min_at(second) == 113
        count = 53 if peak else reference_counts()[at.hour]
        return max(minimum, min(480, math.ceil(2 * half / count)))
    queue = [(max(0, last[t] + target(max(0, last[t]), t)), t) for t in routes]; heapq.heapify(queue)
    dispatch = []; failures = []
    while queue:
        _deadline(deadline); proposed, terminal = heapq.heappop(queue)
        second = feasible_clock(routes[terminal], max(proposed, last[terminal] + target(proposed, terminal)))
        if second >= service_end: continue
        if queue and second > queue[0][0]: heapq.heappush(queue, (second, terminal)); continue
        available = sorted((s['ready'], tid) for tid, s in states.items() if s['active'] and s['location'] == terminal and s['ready'] <= second)
        for _, tid in available[:-1]: exit_train(tid)
        if available:
            tid = available[-1][1]; state = states[tid]
        else:
            entry = next(t for t in tracks.values() if t['to_station'] == terminal and t['track_id'].startswith('ENTRY_'))
            depot = entry['from_station']; release = parameter(policy, 'depot_release_s')['Avtovo' if 'AVTOVO' in depot else 'Northern']
            lead = entry['travel_seconds'] + stations[terminal]['turnaround_seconds']
            reserves = sorted((s['ready'], tid) for tid, s in states.items() if not s['active'] and s['location'] == depot and s['ready'] + release <= second)
            if not reserves or not capacity(second - lead) or not entry_window(terminal, second):
                future = [s['ready'] for s in states.values() if s['active'] and s['location'] == terminal and s['ready'] > second]
                if not future: raise ValueError('direct constructor has no prepared reserve or known terminal return')
                heapq.heappush(queue, (min(future), terminal)); continue
            tid = reserves[0][1]; state = states[tid]
            entry_dep = feasible_clock([entry['track_id']], second - lead)
            if entry_dep != second - lead: heapq.heappush(queue, (entry_dep + lead, terminal)); continue
            append(tid, entry_dep, [entry['track_id']], False)
            state.update(active=True, visit=[second - 180, second]); state['spans'].append([entry_dep, 10**9]); visits[terminal].append(state['visit'])
        if not platform_ok(terminal, [state['visit'][0], second], state['visit']):
            exit_train(tid)
            future = [s['ready'] for s in states.values() if s['active'] and s['location'] == terminal and s['ready'] > second]
            if not future: raise ValueError(f'direct SERVICE cannot clear {terminal} at {second}; next ready missing')
            heapq.heappush(queue, (min(future), terminal)); continue
        state['visit'][1] = second
        route = routes[terminal]; destination = tracks[route[-1]]['to_station']; arrival = second + half - 180
        nominal = arrival
        while not platform_ok(destination, [arrival, arrival + 180]):
            edges = [span[1] for span in visits[destination] if span[0] < arrival + 180 and span[1] > arrival]
            if not edges: raise ValueError('direct approach has no terminal clearance edge')
            arrival = min(e for e in edges if e > arrival)
            _deadline(deadline)
        ready = append(tid, second, route, True, arrival - nominal)
        state.update(location=destination, ready=ready, visit=[arrival, ready], approach_row=out['leg_timing'][-1]); visits[destination].append(state['visit'])
        last[terminal] = second; dispatch.append(dict(second=second, terminal=terminal, train_id=tid))
        heapq.heappush(queue, (second + target(second, terminal), terminal))
    for tid, state in sorted(states.items(), key=lambda row: row[1]['ready']):
        if state['active']: exit_train(tid)
    _deadline(deadline); motion = simulate(out, out['plans'][0]['plan_id'])
    headways = operational_headways(out, motion, 0, service_end); resources = terminal_resources(out, motion); releases = release_constraints(out, 0)
    current = _current_movements_preserved(data, out, 0)
    audit = dict(movement_passed=headways['passed'] and resources['passed'] and releases['passed'] and motion['obligations_closed'] and current,
        headways={k: v for k, v in headways.items() if k not in ('intervals', 'boundary_ages')}, terminal_resources=resources,
        releases=releases, current_movements_preserved=current, obligations_closed=motion['obligations_closed'],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'], future_passenger_truth_used=False,
        strengthen_peak=strengthen_peak,
        approach_holding_seconds=sum(r.get('approach_holding_seconds', 0) for r in out['leg_timing']),
        departure_count=len(dispatch), dispatch=dispatch, _data_digest=digest(out), _motion_digest=digest(motion),
        scope='continuous initial33 same68 roster; direct future clocks, no global compiler', cash_saving_rub=None)
    _deadline(deadline); return out, audit
