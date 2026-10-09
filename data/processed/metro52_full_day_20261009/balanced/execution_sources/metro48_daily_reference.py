"""Daily FIFO circulation with track and terminal reservations before departure."""

import bisect
import copy
import heapq
from datetime import datetime, timedelta
from functools import lru_cache

from dwell import timing_key
from goal43_policy import minimum_headway, parameter
from metro_daily import _geometry
from metro44_assignment_plan import _current_movements_preserved, _deadline, terminal_resources
from metro44_linked_gap_repair import release_constraints
from metro44_planner import trip_finish
from metro47_reference import validate_readiness
from rolling import operational_headways
from simulation import digest, simulate


def terminal_slot(visits, arrival, *, duration=180, replaced=None):
    """Earliest arrival at two half-open terminal berths."""
    while True:
        overlapping = [v for v in visits if v is not replaced and v[0] < arrival + duration and v[1] > arrival]
        edges = [(arrival, 1), (arrival + duration, -1)]
        for left, right in overlapping:
            edges.extend([(max(left, arrival), 1), (min(right, arrival + duration), -1)])
        count = 0
        for second, change in sorted(edges):
            count += change
            if count > 2:
                arrival = min(right for left, right in overlapping if left <= second < right)
                break
        else:
            return arrival


def build_daily_reference(data, roster, service_end, *, deadline=None, pre_stage=True):
    _deadline(deadline)
    validate_readiness(data, roster)
    tracks, stations, routes, half = _geometry(data)
    origin = datetime.fromisoformat(data['period_start'])
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    first = lambda t: timing[timing_key(t['train_id'], t['trip_id'], 0)]['departure_second']
    kept = [next(t for t in sorted((t for t in data['plans'][0]['trips'] if t['train_id'] == train['train_id']), key=first)
                 if t['passenger']) for train in data['trains'] if not train['location'].startswith('DEPOT_')]
    identities = {t['trip_id'] for t in kept}
    out = copy.deepcopy(data)
    out['plans'][0]['trips'] = copy.deepcopy(kept)
    out['leg_timing'] = [copy.deepcopy(r) for r in data['leg_timing'] if r['trip_id'] == '@initial' or r['trip_id'] in identities]
    clocks = {key: [] for key in tracks}
    visits = {terminal: [] for terminal in routes}
    states, last = {}, dict.fromkeys(routes, -1)
    owned = {t['train_id']: t for t in kept}
    for row in data.get('previous_departures', []):
        clocks[row['track_id']].append(row['departure_second'])
    for row in out['leg_timing']:
        track = next(t['location'] for t in data['trains'] if t['train_id'] == row['train_id']) if row['trip_id'] == '@initial' else owned[row['train_id']]['route'][row['leg_index']]
        clocks[track].append(row['departure_second'])
    for key in clocks:
        clocks[key] = sorted(set(clocks[key]))
    for terminal, route in routes.items():
        starts = [first(t) for t in kept if t['route'][0] == route[0]]
        if starts:
            last[terminal] = max(starts)
    for train in data['trains']:
        tid = train['train_id']
        trip = owned.get(tid)
        location, ready = trip_finish(trip, timing, tracks, stations) if trip else (train['location'], train['ready_after_seconds'])
        visit = [ready - 180, ready] if location in routes else None
        if visit is not None:
            visits[location].append(visit)
        states[tid] = dict(location=location, ready=ready, visit=visit, active=trip is not None,
                           spans=[[0, 10**9]] if trip else [], approach_row=None)

    def min_at(second):
        return minimum_headway(origin + timedelta(seconds=second), data['headway_policy'])

    def target(second, terminal):
        return min_at(second)

    stage_hold = max(0, (53 * parameter(data['headway_policy'], 'offpeak_headway_min_s') - 2 * half)
        // (2 * (len(next(iter(routes.values()))) - 1))) if pre_stage else 0
    preparation_lead = 2 * half + max(parameter(data['headway_policy'], 'depot_release_s').values())

    @lru_cache(None)
    def preparing(minute):
        second = minute * 60
        return min_at(second) != 113 and any(min_at(second + offset) == 113
            for offset in range(0, preparation_lead + 900, 900))

    def track_slot(route, departure):
        for _ in range(500):
            _deadline(deadline)
            offset, updated = 0, departure
            for track in route:
                at = departure + offset
                values = clocks[track]
                index = bisect.bisect_right(values, at)
                if index and at - values[index - 1] < min_at(at):
                    updated = max(updated, values[index - 1] + min_at(at) - offset)
                if index < len(values) and values[index] - at < min_at(values[index]):
                    updated = max(updated, values[index] + min_at(values[index] + 150) - offset)
                section = tracks[track]
                station = stations[section['to_station']]
                offset += section['travel_seconds'] + station['stop_seconds'] + station['turnaround_seconds']
            if updated == departure:
                return departure
            departure = updated
        raise ValueError('track reservation propagation failed')

    def route_slots(route, departure):
        times = []
        for index, track in enumerate(route):
            departure = track_slot([track], departure)
            times.append(departure)
            section = tracks[track]
            station = stations[section['to_station']]
            departure += section['travel_seconds'] + station['stop_seconds'] + station['turnaround_seconds']
            if index + 1 < len(route) and preparing(departure // 60):
                remaining = route[index + 1:]
                def safe_delay(delay):
                    at = departure + delay
                    for later in remaining:
                        at = track_slot([later], at)
                        if clocks[later] and at > clocks[later][-1] + 480:
                            return False
                        section = tracks[later]
                        station = stations[section['to_station']]
                        at += section['travel_seconds'] + station['stop_seconds'] + station['turnaround_seconds']
                    return True
                low, high = 0, stage_hold
                while low < high:
                    middle = (low + high + 1) // 2
                    if safe_delay(middle):
                        low = middle
                    else:
                        high = middle - 1
                departure += low
        return times, departure

    def capacity(start):
        edges = [(start, 1)]
        for state in states.values():
            for left, right in state['spans']:
                edges.extend([(left, 1), (right, -1)])
        count = 0
        for _, change in sorted(edges):
            count += change
            if count > 53:
                return False
        return True

    def entry_window(terminal, second):
        desired = second - 180
        if terminal_slot(visits[terminal], desired) == desired:
            return True
        if min_at(second) != 113:
            return False
        changed = []
        for state in sorted(states.values(), key=lambda s: s['ready']):
            row, visit = state['approach_row'], state['visit']
            if row is None or not state['active'] or state['location'] != terminal or state['ready'] <= second or visit[0] >= second:
                continue
            arrival = terminal_slot(visits[terminal] + [[desired, second]], visit[0], replaced=visit)
            holding = row['approach_holding_seconds'] + arrival - visit[0]
            if holding > 150:
                continue
            changed.append((state, visit[:], row['approach_holding_seconds']))
            row['approach_holding_seconds'] = holding
            visit[:] = [arrival, arrival + 180]
            state['ready'] = arrival + 180
            if terminal_slot(visits[terminal], desired) == desired:
                return True
        for state, visit, holding in changed:
            state['visit'][:] = visit
            state['ready'] = visit[1]
            state['approach_row']['approach_holding_seconds'] = holding
        return False

    serial = 0
    def append(tid, departure, route, passenger, scheduled=None):
        nonlocal serial
        serial += 1
        identity = f'G48FIFO:{tid}:{serial}'
        out['plans'][0]['trips'].append(dict(train_id=tid, trip_id=identity, departure_second=departure, route=route, passenger=passenger))
        at = departure
        for index, track in enumerate(route):
            if scheduled is not None:
                at = scheduled[index]
            section = tracks[track]
            station = stations[section['to_station']]
            service = station['stop_seconds'] + station['turnaround_seconds']
            holding = scheduled[index + 1] - at - section['travel_seconds'] - service if scheduled is not None and index + 1 < len(route) else 0
            out['leg_timing'].append(dict(train_id=tid, trip_id=identity, leg_index=index,
                departure_second=at, service_seconds=service, holding_seconds=holding, approach_holding_seconds=0))
            bisect.insort(clocks[track], at)
            at += section['travel_seconds'] + service
        return at

    def exit_train(tid):
        state = states[tid]
        terminal = state['location']
        track = next(t for t in tracks.values() if t['from_station'] == terminal and t['track_id'].startswith('EXIT_'))
        dep = track_slot([track['track_id']], state['ready'])
        arrival = terminal_slot(visits[terminal], state['visit'][0], duration=dep - state['visit'][0], replaced=state['visit'])
        if arrival != state['visit'][0]:
            raise ValueError(f'EXIT berth cannot be cleared for {tid} at {dep}')
        state['visit'][1] = dep
        ready = append(tid, dep, [track['track_id']], False)
        state['spans'][-1][1] = dep + track['travel_seconds']
        state.update(location=track['to_station'], ready=ready, active=False, visit=None, approach_row=None)

    queue = [(max(0, last[t] + target(max(0, last[t]), t)), t) for t in routes]
    heapq.heapify(queue)
    dispatch = []
    while queue:
        _deadline(deadline)
        proposed, terminal = heapq.heappop(queue)
        route = routes[terminal]
        scheduled, ready = route_slots(route, max(proposed, last[terminal] + target(proposed, terminal)))
        second = scheduled[0]
        destination = tracks[route[-1]]['to_station']
        arrival = ready - 180
        berth = terminal_slot(visits[destination], arrival)
        if berth > arrival:
            heapq.heappush(queue, (second + berth - arrival, terminal))
            continue
        if second >= service_end:
            continue
        if queue and second > queue[0][0]:
            heapq.heappush(queue, (second, terminal))
            continue
        available = sorted((s['ready'], tid) for tid, s in states.items() if s['active'] and s['location'] == terminal and s['ready'] <= second)
        if available:
            tid = available[0][1]
            for _, other in available[1:]:
                exit_train(other)
            state = states[tid]
        else:
            entry = next(t for t in tracks.values() if t['to_station'] == terminal and t['track_id'].startswith('ENTRY_'))
            depot = entry['from_station']
            release = parameter(data['headway_policy'], 'depot_release_s')['Avtovo' if 'AVTOVO' in depot else 'Northern']
            lead = entry['travel_seconds'] + stations[terminal]['turnaround_seconds']
            reserves = sorted((s['ready'], tid) for tid, s in states.items() if not s['active'] and s['location'] == depot and s['ready'] + release <= second)
            if not reserves or not capacity(second - lead) or not entry_window(terminal, second):
                future = [s['ready'] for s in states.values() if s['active'] and s['location'] == terminal and s['ready'] > second]
                if not future:
                    raise ValueError(f'no available train at {terminal}, {second}')
                heapq.heappush(queue, (min(future), terminal))
                continue
            tid = reserves[0][1]
            state = states[tid]
            entry_dep = track_slot([entry['track_id']], second - lead)
            if entry_dep != second - lead:
                heapq.heappush(queue, (entry_dep + lead, terminal))
                continue
            append(tid, entry_dep, [entry['track_id']], False)
            state.update(active=True, visit=[second - 180, second])
            state['spans'].append([entry_dep, 10**9])
            visits[terminal].append(state['visit'])
        if terminal_slot(visits[terminal], state['visit'][0], duration=second - state['visit'][0], replaced=state['visit']) != state['visit'][0]:
            exit_train(tid)
            future = [s['ready'] for s in states.values() if s['active'] and s['location'] == terminal and s['ready'] > second]
            if not future:
                raise ValueError(f'SERVICE berth cannot be cleared at {terminal}, {second}')
            heapq.heappush(queue, (min(future), terminal))
            continue
        state['visit'][1] = second
        ready = append(tid, second, route, True, scheduled)
        state.update(location=destination, ready=ready, visit=[arrival, ready], approach_row=out['leg_timing'][-1])
        visits[destination].append(state['visit'])
        last[terminal] = second
        dispatch.append(dict(second=second, terminal=terminal, train_id=tid))
        heapq.heappush(queue, (second + target(second, terminal), terminal))
    for tid, state in sorted(states.items(), key=lambda row: row[1]['ready']):
        if state['active']:
            exit_train(tid)
    motion = simulate(out, out['plans'][0]['plan_id'])
    headways = operational_headways(out, motion, 0, service_end)
    resources = terminal_resources(out, motion)
    releases = release_constraints(out, 0)
    current = _current_movements_preserved(data, out, 0)
    audit = dict(movement_passed=headways['passed'] and resources['passed'] and releases['passed'] and motion['obligations_closed'] and current,
        headways={k: v for k, v in headways.items() if k not in ('intervals', 'boundary_ages')}, terminal_resources=resources,
        releases=releases, current_movements_preserved=current, obligations_closed=motion['obligations_closed'],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'], future_passenger_truth_used=False,
        new_approach_holding_seconds=sum(r['approach_holding_seconds'] for r in out['leg_timing'] if r['trip_id'].startswith('G48FIFO:')),
        maximum_new_approach_holding_seconds=max((r['approach_holding_seconds'] for r in out['leg_timing'] if r['trip_id'].startswith('G48FIFO:')), default=0),
        approach_holding_limit_seconds=150, departure_count=len(dispatch), dispatch=dispatch,
        intermediate_holding_seconds=sum(r['holding_seconds'] for r in out['leg_timing'] if r['trip_id'].startswith('G48FIFO:')),
        maximum_intermediate_holding_seconds=max((r['holding_seconds'] for r in out['leg_timing'] if r['trip_id'].startswith('G48FIFO:')), default=0),
        stage_hold_seconds_per_intermediate_station=stage_hold, preparation_lead_seconds=preparation_lead,
        line_counts=[dict(second=second, count=sum(left <= second < right for s in states.values() for left, right in s['spans']))
            for second in range(0, service_end, 900)],
        scope='same initial33 and fleet68; FIFO earliest feasible reservation; maximum calendar frequency',
        reference_optimality_claimed=False, cash_saving_rub=None, data_digest=digest(out), motion_digest=digest(motion))
    return out, audit
