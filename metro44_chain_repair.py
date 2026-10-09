"""Bounded physical repairs of a supplied compiled vehicle chain."""

import copy

from dwell import timing_key
from metro44_assignment_plan import _current_movements_preserved,_deadline,terminal_resources
from metro44_bridge_reserves import standby_repair_bounds,max_gap_repair_bounds
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import digest,simulate


def repair_chain_timetable(data,as_of,service_end,*,standby_repairs=5,max_gap_repairs=5,deadline=None):
    _deadline(deadline)
    for limit in(standby_repairs,max_gap_repairs):
        if isinstance(limit,bool)or not isinstance(limit,int)or not 0<=limit<=5:raise ValueError('repair iterations must be between zero and five')
    changed=copy.deepcopy(data);parking=[];max_gaps=[]
    def recompile(extra):
        nonlocal changed
        bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in changed['leg_timing']if r['departure_second']>=0}
        delay=max((extra[k]-bounds[k]for k in extra),default=0)
        for key,value in extra.items():bounds[key]=max(bounds[key],value)
        _deadline(deadline)
        changed,_=compile_terminal_timing(changed,changed['plans'][0]['plan_id'],'fixed',7,
            policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
        _deadline(deadline);return delay
    def repair_parking():
        for _ in range(standby_repairs):
            _deadline(deadline);motion=simulate(changed,changed['plans'][0]['plan_id'])
            if not terminal_resources(changed,motion)['parking_violations']:break
            extra=standby_repair_bounds(changed,motion,as_of)
            if not extra:break
            parking.append(dict(inbound_bounds_count=len(extra),maximum_delay=recompile(extra)))
    repair_parking()
    for _ in range(max_gap_repairs):
        _deadline(deadline);motion=simulate(changed,changed['plans'][0]['plan_id'])
        gaps=operational_headways(changed,motion,as_of,service_end)
        if gaps['passed']:break
        extra=max_gap_repair_bounds(changed,motion,as_of,gaps['violations'])
        if not extra:break
        max_gaps.append(dict(service_bounds_count=len(extra),maximum_delay=recompile(extra),preceding_gaps_checked=True))
        repair_parking()
    before=simulate(data,data['plans'][0]['plan_id']);motion=simulate(changed,changed['plans'][0]['plan_id'])
    if [e for e in before['events']if e['second']<as_of]!=[e for e in motion['events']if e['second']<as_of]:
        raise ValueError('chain repair changes executed movements')
    if not _current_movements_preserved(data,changed,as_of):raise ValueError('chain repair changes a started arrival')
    resources=terminal_resources(changed,motion);gaps=operational_headways(changed,motion,as_of,service_end)
    audit=dict(movement_passed=resources['passed']and gaps['passed']and motion['obligations_closed'],
        terminal_resources=resources,headways_passed=gaps['passed'],minimum_headway=gaps['minimum_seconds'],
        maximum_headway=gaps['maximum_seconds'],headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],obligations_closed=motion['obligations_closed'],
        executed_prefix_preserved=True,current_movements_preserved=True,standby_repairs=parking,max_gap_repairs=max_gaps,
        source_vehicle_chains_preserved=True,passenger_checked=False,release_authorized=False,
        _data_digest=digest(changed),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    _deadline(deadline);return changed,audit
