"""Temporary depot or siding withdrawal with a preserved return duty."""

import copy
from decimal import Decimal

from dwell import timing_key
from goal43_policy import parameter
from metro44_assignment_plan import _deadline, build_assignment_problem
from metro44_catalog import physical_gate
from metro47_withdrawal import _index, append_offline_movement
from withdrawal_stabling import assign_routes, routing, withdrawal_count


def temporary_options(data, as_of_second, service_end_second, *, return_by_second, deadline=None,
                      assigned_routes=None):
    _deadline(deadline)
    if (type(return_by_second) is not int or type(as_of_second) is not int
            or type(service_end_second) is not int or as_of_second < 0
            or not as_of_second < return_by_second <= service_end_second):
        raise ValueError('invalid temporary return bound')
    problem = build_assignment_problem(data, as_of_second)
    timing, tracks, stations, first = _index(data)
    retained, options = set(problem['retained_ids']), []
    policy = data['headway_policy']
    for available in problem['availability']:
        _deadline(deadline)
        tid, ready, terminal = available['train_id'], available['ready_second'], available['location']
        if not available['on_line'] or available.get('locked_exit') or not as_of_second <= ready <= as_of_second+900:
            continue
        owned = sorted((t for t in data['plans'][0]['trips'] if t['train_id'] == tid), key=first)
        future = [t for t in owned if t['trip_id'] not in retained]
        route = (assigned_routes or {}).get(tid) or routing(data, tid, terminal, withdrawal_count(data))
        depot = route['destination']
        depot_ready = ready + route['exit_seconds'] + stations[depot]['stop_seconds'] + stations[depot]['turnaround_seconds']
        release = route['release_seconds']
        lead = route['entry_seconds'] + stations[terminal]['turnaround_seconds']
        candidates = []
        for index, resume in enumerate(future):
            cancelled = future[:index]
            if not cancelled or any(not t['passenger'] or first(t) < ready for t in cancelled):
                continue
            if (not resume['passenger'] or first(resume) > return_by_second
                    or first(resume) >= service_end_second or tracks[resume['route'][0]]['from_station'] != terminal):
                continue
            prior = cancelled[-1]
            if tracks[prior['route'][-1]]['to_station'] != terminal:
                continue
            last = timing[timing_key(tid, prior['trip_id'], len(prior['route'])-1)]
            arrival = last['departure_second'] + tracks[prior['route'][-1]]['travel_seconds'] + last.get('approach_holding_seconds', 0)
            entry_second = arrival - route['entry_seconds']
            entry_plan_second = min(entry_second, resume['departure_second']-1)
            command = max(as_of_second, depot_ready)
            if (first(resume) < command+release or entry_second < command+max(0, release-lead)
                    or arrival+stations[terminal]['turnaround_seconds'] > first(resume)
                    or entry_plan_second <= ready):
                continue
            km = sum((Decimal(str(tracks[r]['length_km'])) for t in cancelled for r in t['route']), Decimal(0))
            km -= Decimal(route['exit_km']) + Decimal(route['entry_km'])
            if km <= 0:
                continue
            price = parameter(policy, 'maintenance_prices_rub_by_type')[available['train_type']]['TO1']
            rate = Decimal(str(price)) / Decimal(str(parameter(policy, 'to1_interval_km')))
            candidates.append(dict(train_id=tid, train_type=available['train_type'], terminal=terminal,
                ready_second=ready, exit_second=ready, exit_track=route['exit_route'][0], depot=depot,
                withdrawal_destination=route,
                depot_ready_second=depot_ready, preparation_command_second=command,
                release_seconds=release, entry_track=route['entry_route'][0], entry_second=entry_second,
                entry_plan_second=entry_plan_second,
                entry_arrival_second=arrival, entry_lead_seconds=lead,
                resume_trip_id=resume['trip_id'], resume_second=first(resume), return_by_second=return_by_second,
                begun_service_ids=[t['trip_id'] for t in owned if t['passenger'] and first(t) < as_of_second],
                preserved_trip_ids=[t['trip_id'] for t in owned if t not in cancelled],
                cancel_service_ids=[t['trip_id'] for t in cancelled], cancel_entry_ids=[], cancel_exit_ids=[],
                saved_km=str(km), to1_resource_saving_rub=str(km*rate)))
        if candidates:
            options.append(max(candidates, key=lambda row: row['resume_second']))
    _deadline(deadline)
    return sorted(options, key=lambda row: (row['ready_second'], row['train_id']))


def apply_temporary_withdrawals(data, options, as_of_second, service_end_second, *, return_by_second, deadline=None):
    _deadline(deadline)
    canonical = {r['train_id']: r for r in temporary_options(data, as_of_second, service_end_second,
        return_by_second=return_by_second, deadline=deadline)}
    names = [r['train_id'] for r in options]
    if len(set(names)) != len(names) or set(names)-set(canonical):
        raise ValueError('temporary withdrawal requires distinct canonical trains')
    if any(r != canonical[r['train_id']] for r in options):
        raise ValueError('temporary withdrawal differs from canonical source duty')
    destinations = assign_routes(data, names, {tid: canonical[tid]['terminal'] for tid in names})
    assigned = {r['train_id']: r for r in temporary_options(data, as_of_second, service_end_second,
        return_by_second=return_by_second, deadline=deadline, assigned_routes=destinations)}
    if any(tid not in assigned or assigned[tid]['cancel_service_ids'] != canonical[tid]['cancel_service_ids']
           or assigned[tid]['resume_trip_id'] != canonical[tid]['resume_trip_id'] for tid in names):
        raise ValueError('allocated stabling route cannot preserve the selected return duty')
    canonical = assigned
    removed = {trip for tid in names for trip in canonical[tid]['cancel_service_ids']}
    changed = copy.deepcopy(data)
    changed['plans'][0]['trips'] = [t for t in changed['plans'][0]['trips'] if t['trip_id'] not in removed]
    changed['leg_timing'] = [r for r in changed['leg_timing'] if r['trip_id'] not in removed]
    added = {'EXIT': [], 'ENTRY': []}
    identities = {t['trip_id'] for t in changed['plans'][0]['trips']}
    for tid in names:
        row = canonical[tid]
        for kind in ('EXIT', 'ENTRY'):
            identity = f'G48TEMP:{as_of_second}:{tid}:{kind}'
            if identity in identities:
                raise ValueError('temporary movement identity collides')
            identities.add(identity)
            second = row[kind.lower()+'_second']
            append_offline_movement(changed, tid, identity, row['withdrawal_destination'][kind.lower()+'_route'],
                second, planned_second=row['entry_plan_second'] if kind == 'ENTRY' else second)
            added[kind].append(identity)
    _deadline(deadline)
    audit = physical_gate(changed, as_of_second, service_end_second, reference=data,
                          deadline=deadline if deadline is not None else float('inf'))
    if not audit['executed_prefix_preserved'] or not audit['current_movements_preserved']:
        raise ValueError('temporary withdrawal changes an executed or current movement')
    audit.update(selected_train_ids=names, cancelled_service_ids=sorted(removed),
        cancelled_trip_ids=sorted(removed), appended_exit_ids=added['EXIT'], appended_entry_ids=added['ENTRY'],
        kept_clocks_preserved=True, return_by_second=return_by_second,
        withdrawal_destinations=[canonical[t]['withdrawal_destination'] for t in names],
        resume_trip_ids={t: canonical[t]['resume_trip_id'] for t in names},
        resume_seconds={t: canonical[t]['resume_second'] for t in names},
        depot_preparation=dict(passed=True, cycles=[canonical[t] for t in names],
            scope='physical off-line arrival and readiness; depot inclusive 720/780s or siding reversal; terminal turnaround included'),
        estimated_to1_resource_saving_rub=str(sum((Decimal(canonical[t]['to1_resource_saving_rub']) for t in names), Decimal(0))),
        preparation_scope='exact per-cycle source geometry; actual depot readiness and operational authorization unknown',
        release_authorized=False, cash_saving_rub=None, passenger_checked=False)
    _deadline(deadline)
    return changed, audit


def build_temporary_candidate(data, second, planning_end, *, selected_train_ids, return_by_second,
                              deadline=None, phase_seconds=0):
    if phase_seconds != 0:
        raise ValueError('temporary withdrawal does not retime a phase')
    rows = {r['train_id']: r for r in temporary_options(data, second, planning_end,
        return_by_second=return_by_second, deadline=deadline)}
    if set(selected_train_ids)-set(rows):
        raise ValueError('temporary withdrawal names an unavailable train')
    return apply_temporary_withdrawals(data, [rows[t] for t in selected_train_ids], second, planning_end,
                                      return_by_second=return_by_second, deadline=deadline)
