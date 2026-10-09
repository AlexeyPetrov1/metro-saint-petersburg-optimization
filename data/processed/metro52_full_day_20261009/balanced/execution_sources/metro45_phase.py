"""Explore a small future dispatch phase while preserving executed train movements."""

from copy import deepcopy

from dwell import timing_key
from metro44_assignment_plan import _deadline
from metro44_catalog import physical_gate
from metro44_planner_bridge import rebuild_bridge_timetable
from metro44_terminal_timing import compile_terminal_timing


def rephase_bridge(data, as_of, service_end, *, shift_seconds=-20, deadline=None,
                   bridge_seconds=480, headway=160, **kwargs):
    _deadline(deadline)
    if type(shift_seconds) is not int or not -60 <= shift_seconds <= 60:
        raise ValueError('integer phase within60 seconds required')
    candidate, original = rebuild_bridge_timetable(data, as_of, service_end, deadline=deadline,
        bridge_seconds=bridge_seconds, headway=headway, **kwargs)
    out = deepcopy(candidate)
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in out['leg_timing']}
    trips = {t['trip_id']: t for t in out['plans'][0]['trips']}
    starts = {t['trip_id']: timing[timing_key(t['train_id'], t['trip_id'], 0)]['departure_second']
              for t in trips.values()}
    bounds = {}
    for row in out['leg_timing']:
        key = timing_key(row['train_id'], row['trip_id'], row['leg_index'])
        if row['departure_second'] >= 0 and (row['trip_id'] == '@initial' or starts[row['trip_id']] < as_of):
            bounds[key] = row['departure_second']
    for trip in trips.values():
        old = starts[trip['trip_id']]
        if old < as_of:
            continue
        at = max(as_of, old + shift_seconds) if old >= as_of + bridge_seconds else old
        trip['departure_second'] = at
        bounds[timing_key(trip['train_id'], trip['trip_id'], 0)] = at
    changed, timing_audit = compile_terminal_timing(out, out['plans'][0]['plan_id'], 'fixed', 7,
        policy=data['headway_policy'], as_of=as_of, departure_not_before=bounds)
    _deadline(deadline)
    gate = physical_gate(changed, as_of, service_end, reference=data,
                         deadline=deadline if deadline is not None else float('inf'))
    if not gate['movement_passed']:
        raise ValueError('future phase fails exact movement/terminal/clock validation')
    audit = {**original, **gate, 'future_timing_flex_enabled': True,
        'phase_shift_seconds': shift_seconds, 'phase_bound_scope':
        'past/current train trips fixed; near service starts retained; future internal timings may advance',
        'future_trip_coverage_preserved': set(trips) == {t['trip_id'] for t in changed['plans'][0]['trips']},
        'phase_compiler': timing_audit}
    return changed, audit
