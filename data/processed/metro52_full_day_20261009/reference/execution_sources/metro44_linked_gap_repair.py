"""Delay connected future departures, then check the complete vehicle chains."""

import copy
from collections import defaultdict

from dwell import timing_key
from goal43_policy import parameter
from metro44_assignment_plan import _current_movements_preserved,_deadline,terminal_resources
from metro44_bridge_reserves import standby_repair_bounds
from metro44_planner import trip_finish
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import digest,simulate


def linked_gap_bounds(data,motion,as_of,service_end,*,allow_future_legs=False,deadline=None):
    """Backward difference constraints; whole future trips or explicit future legs."""
    _deadline(deadline)
    trips={t['trip_id']:t for t in data['plans'][0]['trips']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    stations={s['station_id']:s for s in data['stations']}
    line={t['track_id']for t in data['tracks']if all(stations[t[k]]['kind']=='line'for k in('from_station','to_station'))}
    maximum=parameter(data['headway_policy'],'headway_max_s');rows=defaultdict(list);delays={}
    def node(event):
        trip=trips.get(event['trip_id'])
        if trip is None or not trip['passenger']:return None
        first=timing_key(trip['train_id'],trip['trip_id'],0)
        index=trip['route'].index(event['location']) if allow_future_legs and timing[first]['departure_second']<as_of else 0
        return timing_key(trip['train_id'],trip['trip_id'],index)
    for e in motion['events']:
        if e['event_type']=='departure'and e['location']in line and e['second']<=service_end:
            rows[e['location']].append((e['second'],node(e)))
    for events in rows.values():events.sort(key=lambda r:r[0])
    def require(key,delay):
        if delay<=delays.get(key,0):return False
        if key is None:raise ValueError('linked repair reaches an immutable departure')
        if timing[key]['departure_second']<as_of:
            raise ValueError('linked repair reaches a departed leg'if allow_future_legs else'linked repair reaches a started SERVICE')
        delays[key]=delay;return True
    for events in rows.values():
        if events:require(events[-1][1],service_end-maximum-events[-1][0])
    for _ in range(len(timing)+1):
        changed=False
        for events in rows.values():
            _deadline(deadline)
            for (left,lkey),(right,rkey)in reversed(list(zip(events,events[1:]))):
                if right<as_of:continue
                changed=require(lkey,right+delays.get(rkey,0)-left-maximum)or changed
        if not changed:break
    else:raise ValueError('linked departure constraints contain a positive cycle')
    return {key:timing[key]['departure_second']+delay for key,delay in delays.items()if delay>0}


def release_constraints(data,as_of):
    tracks={t['track_id']:t for t in data['tracks']};stations={s['station_id']:s for s in data['stations']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    actual=lambda t:timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second']
    errors=[];count=0
    for train in data['trains']:
        owned=sorted((t for t in data['plans'][0]['trips']if t['train_id']==train['train_id']),key=actual)
        for index,entry in enumerate(owned):
            if not entry['route'][0].startswith('ENTRY_')or actual(entry)<as_of:continue
            depot=tracks[entry['route'][0]]['from_station'];exits=[t for t in owned[:index]if t['route'][-1].startswith('EXIT_')]
            technical=trip_finish(exits[-1],timing,tracks,stations)[1]if exits else train.get('ready_after_seconds',0)
            first=next((t for t in owned[index+1:]if t['passenger']),None)
            lead=sum(tracks[r]['travel_seconds']for r in entry['route'])+stations[tracks[entry['route'][-1]]['to_station']]['turnaround_seconds']
            lead+=sum(stations[tracks[r]['to_station']]['stop_seconds']+stations[tracks[r]['to_station']]['turnaround_seconds']for r in entry['route'][:-1])
            release=(lead if stations[depot]['kind']=='siding' else
                     parameter(data['headway_policy'],'depot_release_s')['Avtovo'if'AVTOVO'in depot else'Northern'])
            start=max(as_of,technical);count+=1
            if first is None or actual(first)<start+release or actual(entry)<start+release-lead or exits and not stations[depot]['turnaround_seconds']:
                errors.append(dict(entry=entry['trip_id'],ready_second=technical,release_seconds=release))
    return dict(passed=not errors,count=count,errors=errors,readiness_scope='unchanged source readiness assumptions; no operational authorization')


def repair_linked_timetable(data,as_of,service_end,*,passes=5,allow_future_legs=False,deadline=None):
    _deadline(deadline)
    if isinstance(passes,bool)or not isinstance(passes,int)or not 0<=passes<=5:raise ValueError('linked repair passes must be between zero and five')
    changed=copy.deepcopy(data);before=simulate(data,data['plans'][0]['plan_id']);history=[];failure=None
    def check_prefix(motion):
        if [e for e in before['events']if e['second']<as_of]!=[e for e in motion['events']if e['second']<as_of]:
            raise ValueError('linked repair changes executed movements')
        if not _current_movements_preserved(data,changed,as_of):raise ValueError('linked repair changes a started arrival')
    for _ in range(passes):
        _deadline(deadline);motion=simulate(changed,changed['plans'][0]['plan_id']);check_prefix(motion)
        gaps=operational_headways(changed,motion,as_of,service_end);resources=terminal_resources(changed,motion)
        if gaps['passed']and resources['passed']:break
        try:
            extra=linked_gap_bounds(changed,motion,as_of,service_end,allow_future_legs=allow_future_legs,deadline=deadline)
            if resources['parking_violations']:
                for key,value in standby_repair_bounds(changed,motion,as_of).items():extra[key]=max(extra.get(key,value),value)
            if not extra:failure='no causal linked departure correction';break
            bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in changed['leg_timing']if r['departure_second']>=0}
            maximum_delay=max(extra[k]-bounds[k]for k in extra)
            for key,value in extra.items():bounds[key]=max(bounds[key],value)
            candidate,_=compile_terminal_timing(changed,changed['plans'][0]['plan_id'],'fixed',7,
                policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
            changed=candidate;motion=simulate(changed,changed['plans'][0]['plan_id']);check_prefix(motion)
            history.append(dict(bounds_count=len(extra),maximum_requested_delay=maximum_delay,
                max_gap_seconds=operational_headways(changed,motion,as_of,service_end)['maximum_seconds'],
                terminal_resources=terminal_resources(changed,motion),reserve_preparation=release_constraints(changed,as_of)))
        except ValueError as error:failure=str(error);break
    _deadline(deadline);motion=simulate(changed,changed['plans'][0]['plan_id']);check_prefix(motion)
    resources=terminal_resources(changed,motion);gaps=operational_headways(changed,motion,as_of,service_end);releases=release_constraints(changed,as_of)
    if changed['plans']!=data['plans']or changed['tracks']!=data['tracks']or changed['trains']!=data['trains']:
        raise ValueError('linked repair changes source jobs, assignments or geometry')
    audit=dict(movement_passed=not failure and resources['passed']and gaps['passed']and releases['passed']and motion['obligations_closed'],
        repair_failure=failure,repair_passes=history,allow_future_legs=allow_future_legs,source_vehicle_chains_preserved=True,
        executed_prefix_preserved=True,current_movements_preserved=True,terminal_resources=resources,reserve_preparation=releases,
        headways_passed=gaps['passed'],minimum_headway=gaps['minimum_seconds'],maximum_headway=gaps['maximum_seconds'],
        headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],obligations_closed=motion['obligations_closed'],
        release_authorized=False,passenger_checked=False,_data_digest=digest(changed),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    _deadline(deadline);return changed,audit
