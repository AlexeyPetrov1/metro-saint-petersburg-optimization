"""Retiming of unstarted legs within begun trips, followed by the exact movement gate."""

from copy import deepcopy
from time import monotonic

from dwell import compile_operational_timing, timing_key
from metro44_catalog import physical_gate
from metro44_terminal_timing import ENTRY_DIRECTIONS, rename_tracks


def _check(deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError('partial-trip retiming deadline')


def _index(data, as_of, train_ids):
    if type(as_of) is not int or as_of < 0:
        raise ValueError('nonnegative integer clock required')
    tracks = {r['track_id']: r for r in data['tracks']}
    trips = {r['trip_id']: r for p in data['plans'] for r in p['trips']}
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r
              for r in data['leg_timing']}
    if len(timing) != len(data['leg_timing']):
        raise ValueError('duplicate timing identity')
    known = {r['train_id'] for r in data['trains']}
    selected = known if train_ids is None else set(train_ids)
    if selected - known:
        raise ValueError('unknown train')
    return tracks, trips, timing, selected


def ready_bounds(data, as_of, *, train_ids=None):
    """Geometry lower bounds preserve started arrivals and every required service time.

    These are optimistic bounds, not a feasible dispatch or a passenger forecast.
    Platform clearance and neighboring trains may remove some or all of the slack.
    """
    tracks, trips, timing, selected = _index(data, as_of, train_ids)
    rows = []
    for tid in sorted(selected):
        begun = [t for t in trips.values() if t['train_id'] == tid and t['passenger']
                 and timing[timing_key(tid, t['trip_id'], 0)]['departure_second'] < as_of]
        if not begun:
            continue
        trip = max(begun, key=lambda t: timing[timing_key(tid, t['trip_id'], 0)]['departure_second'])
        identity = trip['trip_id']
        legs = [timing[timing_key(tid, identity, i)] for i in range(len(trip['route']))]
        future = [i for i, r in enumerate(legs) if r['departure_second'] >= as_of]
        last_started = max(i for i, r in enumerate(legs) if r['departure_second'] < as_of)
        frozen = legs[last_started]
        immutable_arrival = (frozen['departure_second'] + tracks[trip['route'][last_started]]['travel_seconds']
                             + frozen.get('approach_holding_seconds', 0))
        earliest = immutable_arrival + frozen['service_seconds']
        for i in future:
            earliest = max(as_of, earliest) + tracks[trip['route'][i]]['travel_seconds'] + legs[i]['service_seconds']
        last = legs[-1]
        ready = (last['departure_second'] + tracks[trip['route'][-1]]['travel_seconds']
                 + last.get('approach_holding_seconds', 0) + last['service_seconds'])
        if earliest > ready:
            raise ValueError('source route violates required remaining travel/service time')
        rows.append(dict(train_id=tid, trip_id=identity,
            terminal=tracks[trip['route'][-1]]['to_station'],
            source_ready_second=ready, earliest_ready_second=earliest,
            immutable_last_leg_index=last_started,
            immutable_last_arrival_second=immutable_arrival,
            immutable_next_departure_earliest_second=max(as_of, immutable_arrival + frozen['service_seconds']),
            future_leg_indices=future, maximum_geometry_advance_seconds=ready-earliest,
            scope='optimistic geometry only; started arrivals and source service durations retained'))
    return rows


def retime_partial_trips(data, as_of, service_end, *, train_ids=None,
                         deadline=None, profile='fixed', seed=7):
    """Release avoidable holds only in wholly unstarted legs of the latest begun trips.

    The shared compiler proposes timings. Restoring immutable rows precedes the
    authoritative full movement check, including current arrivals and resources.
    Future service jobs, assignments, routes and all other leg rows stay exact.
    """
    _check(deadline)
    bounds = ready_bounds(data, as_of, train_ids=train_ids)
    _, _, old, _ = _index(data, as_of, train_ids)
    eligible = {timing_key(r['train_id'], r['trip_id'], i) for r in bounds
                if r['maximum_geometry_advance_seconds'] > 0 for i in r['future_leg_indices']}
    out, compiler = deepcopy(data), None
    if eligible:
        commands = {k: r['departure_second'] for k, r in old.items()
                    if k not in eligible and r['departure_second'] >= 0}
        for row in bounds:
            if row['maximum_geometry_advance_seconds'] > 0:
                commands[timing_key(row['train_id'], row['trip_id'], row['future_leg_indices'][0])] = \
                    row['immutable_next_departure_earliest_second']
        ids = {r['track_id'] for r in data['tracks']}
        mapping = {a: b for a, b in ENTRY_DIRECTIONS.items() if a in ids}
        if set(mapping.values()) & ids:
            raise ValueError('terminal entry alias collision')
        out, compiler = compile_operational_timing(rename_tracks(out, mapping), out['plans'][0]['plan_id'],
            profile, seed, policy=data['headway_policy'], as_of=as_of,
            departure_not_before=commands)
        inverse = {b: a for a, b in mapping.items()}
        out, compiler = rename_tracks(out, inverse), rename_tracks(compiler, inverse)
        for i, row in enumerate(out['leg_timing']):
            key = timing_key(row['train_id'], row['trip_id'], row['leg_index'])
            if key not in eligible:
                out['leg_timing'][i] = deepcopy(old[key])
            else:
                row['service_seconds'] = old[key]['service_seconds']
    _check(deadline)
    identity = all(out[k] == data[k] for k in data if k != 'leg_timing')
    actual = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in out['leg_timing']}
    if not identity or set(actual) != set(old):
        raise ValueError('partial retiming changes route/train/trip identity')
    changes = []
    for key, row in actual.items():
        source = old[key]
        if row == source:
            continue
        if key not in eligible:
            raise ValueError('partial retiming changes an immutable leg')
        departure_advance = source['departure_second'] - row['departure_second']
        arrival_advance = departure_advance + source.get('approach_holding_seconds', 0) - row.get('approach_holding_seconds', 0)
        if departure_advance < 0 or arrival_advance < 0 or row['departure_second'] < as_of:
            raise ValueError('partial retiming delays or starts an undeclared leg')
        changes.append(dict(train_id=key[0], trip_id=key[1], leg_index=key[2],
                            departure_advance_seconds=departure_advance, arrival_advance_seconds=arrival_advance))
    gate = physical_gate(out, as_of, service_end, reference=data,
                         deadline=deadline if deadline is not None else float('inf'))
    if not gate['movement_passed']:
        raise ValueError('partial retiming fails exact movement/terminal/headway validation')
    return out, {**gate, 'identity_preserved': identity, 'ready_bounds': bounds,
        'changed_future_legs': changes, 'compiler': compiler,
        'passengers_checked': False, 'cash_saving_rub': None,
        'scope': 'future legs inside begun trips only; no skipped jobs or changed arrivals of started legs'}
