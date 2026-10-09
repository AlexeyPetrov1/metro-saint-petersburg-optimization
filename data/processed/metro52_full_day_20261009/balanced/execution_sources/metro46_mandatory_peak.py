"""Causal peak phase diagnosis and bounded retiming of existing service jobs."""

import copy
from time import monotonic

from dwell import timing_key
from metro44_assignment_plan import _current_movements_preserved, terminal_resources
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import simulate


def causal_peak_grid(last_departure, first_ready, as_of, end_second, target_headway, minimum_headway):
    """Join the actual prefix and readiness; target frequency is an experiment input."""
    if (isinstance(target_headway, bool) or not isinstance(target_headway, int)
            or not 113 <= minimum_headway <= target_headway <= 480):
        raise ValueError('invalid target headway')
    first = max(as_of, first_ready, last_departure + minimum_headway)
    return list(range(first, end_second, target_headway))


def advance_unstarted_services(data, as_of, clocks, service_end_second, *, deadline=None):
    """Retain every source job, train and route; advance only wholly unstarted trips.

    All source jobs remain mandatory, including jobs outside the chosen quarter.
    Exact movement validation is followed by a separate passenger comparison.
    """
    def budget():
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError('mandatory peak producer deadline')
    budget()
    trips = {t['trip_id']: t for t in data['plans'][0]['trips']}
    old = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    for identity, clock in clocks.items():
        if identity not in trips:
            raise ValueError('unknown service job')
        trip = trips[identity]
        first = old[timing_key(trip['train_id'], identity, 0)]['departure_second']
        if not trip['passenger'] or first < as_of:
            raise ValueError('only unstarted service jobs may be advanced')
        if isinstance(clock, bool) or not isinstance(clock, int) or not as_of <= clock <= first:
            raise ValueError('departure must advance within the causal future')
    before = simulate(data, data['plans'][0]['plan_id'])
    out = copy.deepcopy(data)
    bounds = {key: row['departure_second'] for key, row in old.items() if row['departure_second'] >= 0}
    for trip in out['plans'][0]['trips']:
        identity = trip['trip_id']
        if identity not in clocks:
            continue
        first = old[timing_key(trip['train_id'], identity, 0)]['departure_second']
        shift = clocks[identity] - first
        trip['departure_second'] = clocks[identity]
        for row in out['leg_timing']:
            if row['trip_id'] == identity:
                key = timing_key(row['train_id'], identity, row['leg_index'])
                row['departure_second'] += shift
                bounds[key] = row['departure_second']
    if clocks:
        budget()
        out, _ = compile_terminal_timing(out, out['plans'][0]['plan_id'], 'fixed', 7,
            policy=data['headway_policy'], as_of=as_of, departure_not_before=bounds)
    budget()
    after = simulate(out, out['plans'][0]['plan_id'])
    prefix = [e for e in before['events'] if e['second'] < as_of] == [e for e in after['events'] if e['second'] < as_of]
    current = _current_movements_preserved(data, out, as_of)
    if not prefix or not current:
        raise ValueError('producer rewrites an executed or current movement')
    out_trips = {t['trip_id']: t for t in out['plans'][0]['trips']}
    coverage = set(trips) == set(out_trips) and all(
        all(trips[k][field] == out_trips[k][field] for field in ('train_id', 'route', 'passenger')) for k in trips)
    if not coverage or out['trains'] != data['trains'] or out['tracks'] != data['tracks']:
        raise ValueError('producer changes mandatory source work or physical fleet')
    resources = terminal_resources(out, after)
    gaps = operational_headways(out, after, as_of, service_end_second)
    compiled = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in out['leg_timing']}
    realized = {identity: compiled[timing_key(trips[identity]['train_id'], identity, 0)]['departure_second'] for identity in clocks}
    return out, dict(movement_passed=resources['passed'] and gaps['passed'] and after['obligations_closed'],
        mandatory_service_jobs_preserved=coverage, executed_prefix_preserved=prefix,
        current_movements_preserved=current, requested_departures=clocks, realized_departures=realized,
        terminal_resources=resources, headways_passed=gaps['passed'],
        headway_violation_count=len(gaps['violations']), headway_violations=gaps['violations'][:10],
        peak_trains_on_line=after['checks']['peak_trains_on_line'], obligations_closed=after['obligations_closed'],
        passenger_validation_required=True, release_authorized=False, cash_saving_rub=None,
        scope='all source service jobs preserved; only wholly unstarted service trips advanced')
