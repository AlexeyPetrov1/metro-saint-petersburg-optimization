"""Validate stable dispatch candidates against shared terminal resources."""

from dwell import timing_key
from metro44_assignment_plan import terminal_resources, _current_movements_preserved, _deadline
from metro44_planner import rebuild_timetable
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import simulate,digest


def retime_terminal_plan(data,as_of,service_end,*,deadline=None,profile='fixed',seed=7):
    _deadline(deadline)
    bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']
            for r in data['leg_timing']if r['departure_second']>=0}
    changed,timing=compile_terminal_timing(data,data['plans'][0]['plan_id'],profile,seed,
        policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
    _deadline(deadline)
    before=simulate(data,data['plans'][0]['plan_id']);motion=simulate(changed,changed['plans'][0]['plan_id'])
    if [e for e in before['events']if e['second']<as_of]!=[e for e in motion['events']if e['second']<as_of]:
        raise ValueError('terminal adapter changes executed movement')
    current=_current_movements_preserved(data,changed,as_of)
    if not current:raise ValueError('terminal adapter changes a started arrival')
    resources=terminal_resources(changed,motion)
    gaps=operational_headways(changed,motion,as_of,service_end)
    audit=dict(movement_passed=gaps['passed']and motion['obligations_closed']and resources['passed'],
        current_movements_preserved=current,executed_prefix_preserved=True,terminal_resources=resources,
        obligations_closed=motion['obligations_closed'],headways_passed=gaps['passed'],
        minimum_headway=gaps['minimum_seconds'],maximum_headway=gaps['maximum_seconds'],
        headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],terminal_timing=timing,
        scope='shared terminal2/standby1 model; operator layout and signalling unverified',
        future_leg_policy='all previously compiled future departures are lower bounds',
        release_authorized=False,passenger_checked=False,
        _data_digest=digest(changed),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    _deadline(deadline);return changed,audit


def rebuild_terminal_timetable(data,as_of,service_end,*,deadline=None,**kwargs):
    _deadline(deadline)
    candidate,original_audit=rebuild_timetable(data,as_of,service_end,deadline=deadline,**kwargs)
    changed,audit=retime_terminal_plan(candidate,as_of,service_end,deadline=deadline)
    before=simulate(data,data['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    if [e for e in before['events']if e['second']<as_of]!=[e for e in after['events']if e['second']<as_of]:
        raise ValueError('terminal plan changes original executed movement')
    if not _current_movements_preserved(data,changed,as_of):raise ValueError('terminal plan changes original started arrival')
    merged={**original_audit,**audit}
    merged['constructor']='stable planner followed by coupled terminal compilation'
    _deadline(deadline);return changed,merged
