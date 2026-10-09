"""Retype fixed service jobs, then validate their complete vehicle duties."""

import copy
from collections import Counter
from time import monotonic

from dwell import compile_operational_timing, timing_key
from goal43_policy import parameter
from metro44_assignment import solve_fixed_assignments
from metro44_planner import trip_finish
from rolling import operational_headways
from simulation import simulate, digest


def _deadline(deadline):
    if deadline is not None and monotonic()>=deadline:raise TimeoutError('assignment deadline')


def service_clocks(data):
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    return Counter((track,timing[timing_key(t['train_id'],t['trip_id'],i)]['departure_second'])
        for t in data['plans'][0]['trips']if t['passenger']for i,track in enumerate(t['route']))


def build_assignment_problem(data,as_of):
    tracks={r['track_id']:r for r in data['tracks']};stations={r['station_id']:r for r in data['stations']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    by_train={t['train_id']:sorted((r for r in data['plans'][0]['trips']if r['train_id']==t['train_id']),
              key=lambda r:timing[timing_key(r['train_id'],r['trip_id'],0)]['departure_second'])for t in data['trains']}
    retained=set();preparations=set()
    for owned in by_train.values():
        for i,trip in enumerate(owned):
            first=timing[timing_key(trip['train_id'],trip['trip_id'],0)]['departure_second']
            if first<as_of:retained.add(trip['trip_id'])
            if trip['route'][0].startswith('ENTRY_'):
                following=next((t for t in owned[i+1:]if t['passenger']),None)
                if following is None:continue
                departure=timing[timing_key(following['train_id'],following['trip_id'],0)]['departure_second']
                depot=tracks[trip['route'][0]]['from_station']
                release=(0 if stations[depot]['kind']=='siding' else
                         parameter(data['headway_policy'],'depot_release_s')['Avtovo'if'AVTOVO'in depot else'Northern'])
                if first<as_of or departure-release<as_of:
                    retained.update((trip['trip_id'],following['trip_id']))
                    if first>=as_of:preparations.add(trip['trip_id'])
    motion=simulate(data,data['plans'][0]['plan_id'])
    runtime={t['train_id']:(t['state'],t['location'])for t in data['trains']}
    for event in motion['events']:
        if event['second']>=as_of:break
        runtime[event['train_id']]=(event['next_state'],event['location'])
    def on_line(state,location):
        return any(stations[tracks[location][k]]['kind']=='line'for k in ('from_station','to_station'))if state=='moving'else stations[location]['kind']=='line'
    terminals=set(data['headway_policy']['demo']['terminal_parking_capacity_by_station'])
    availability=[]
    for train in data['trains']:
        tid=train['train_id'];owned=[t for t in by_train[tid]if t['trip_id']in retained]
        location=train['location'];ready=train.get('ready_after_seconds',0)
        if owned:location,ready=trip_finish(owned[-1],timing,tracks,stations)
        active=on_line(*runtime[tid])or bool(owned and location in terminals)
        row=dict(train_id=tid,train_type=train['train_type'],on_line=active,
                 location=location,ready_second=max(as_of,ready),entry_lead_seconds=300,exit_seconds=120,entry_km=1,exit_km=1)
        if active:
            if location not in terminals:
                if not owned or not owned[-1]['route'][-1].startswith('EXIT_'):raise ValueError('committed train has no known terminal availability')
                last=owned[-1];i=len(last['route'])-1;leg=timing[timing_key(tid,last['trip_id'],i)]
                row.update(ready_second=leg['departure_second']+tracks[last['route'][i]]['travel_seconds']+leg.get('approach_holding_seconds',0),
                           exit_seconds=0,locked_exit=True)
        elif stations[location]['kind']=='depot'and not owned and train['state']=='standby':
            entry=next(t for t in tracks.values()if t['from_station']==location and t['track_id'].startswith('ENTRY_'))
            release=parameter(data['headway_policy'],'depot_release_s')['Avtovo'if'AVTOVO'in location else'Northern']
            exit_track=next(t for t in tracks.values()if t['from_station']==entry['to_station']and t['track_id'].startswith('EXIT_'))
            row.update(location=entry['to_station'],depot=location,ready_second=max(as_of,ready)+release,
                release_seconds=release,entry_track=entry['track_id'],
                entry_lead_seconds=entry['travel_seconds']+stations[entry['to_station']]['turnaround_seconds'],
                exit_seconds=exit_track['travel_seconds'],entry_km=float(entry['length_km']),exit_km=float(exit_track['length_km']))
        else:continue
        availability.append(row)
    jobs=[]
    for trip in data['plans'][0]['trips']:
        if not trip['passenger']or trip['trip_id']in retained:continue
        destination,ready=trip_finish(trip,timing,tracks,stations)
        jobs.append(dict(job_id=trip['trip_id'],original_train_id=trip['train_id'],
            departure_second=timing[timing_key(trip['train_id'],trip['trip_id'],0)]['departure_second'],
            ready_second=ready,origin=tracks[trip['route'][0]]['from_station'],destination=destination,
            km=sum(float(tracks[t]['length_km'])for t in trip['route'])))
    return dict(jobs=jobs,availability=availability,retained_ids=sorted(retained),retained_preparation_obligations=sorted(preparations),
                source_motion=motion,scope='known fixed movement jobs; passenger demand excluded')


def terminal_resources(data,motion):
    demo=data['headway_policy']['demo'];terminals=demo['terminal_parking_capacity_by_station']
    visits={t:[]for t in terminals};parking={t:[]for t in terminals}
    open_visits={};open_parking={}
    for train in data['trains']:
        tid=train['train_id'];location=train['location']
        if location in terminals and train['state']!='moving':open_visits[tid]=(location,0)
        if location in terminals and train['state']=='standby':open_parking[tid]=(location,0)
    def close(opened,spans,tid,second):
        if tid in opened:
            terminal,start=opened.pop(tid)
            if start<second:spans[terminal].append((start,second))
    for event in motion['events']:
        tid=event['train_id'];second=event['second'];location=event['location']
        if event['event_type']=='departure':close(open_visits,visits,tid,second)
        if event['event_type']=='arrival'and location in terminals:open_visits[tid]=(location,second)
        close(open_parking,parking,tid,second)
        if event['next_state']=='standby'and location in terminals:open_parking[tid]=(location,second)
    for opened,spans in ((open_visits,visits),(open_parking,parking)):
        for tid in list(opened):close(opened,spans,tid,motion['duration_seconds'])
    def audit(spans,capacities):
        maximum={};violations=[]
        for terminal,rows in spans.items():
            count=peak=0
            for second,delta in sorted([(a,1)for a,b in rows]+[(b,-1)for a,b in rows]):
                count+=delta;peak=max(peak,count)
                if count>capacities[terminal]:violations.append(dict(terminal=terminal,second=second,count=count,capacity=capacities[terminal]))
            maximum[terminal]=peak
        return maximum,violations
    capacities={t:demo['platform_capacity_by_station'].get(t,demo.get('platform_capacity_default',1))for t in terminals}
    max_visits,visit_errors=audit(visits,capacities);max_parking,parking_errors=audit(parking,terminals)
    return dict(passed=not visit_errors and not parking_errors,maximum_visits_by_terminal=max_visits,
                maximum_standby_by_terminal=max_parking,visit_violations=visit_errors[:10],parking_violations=parking_errors[:10],
                scope='shared terminal visits include ENTRY and inbound trains; half-open intervals')


def _current_movements_preserved(before,after,as_of):
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in after['leg_timing']}
    tracks={r['track_id']:r for r in before['tracks']};trips={r['trip_id']:r for r in before['plans'][0]['trips']}
    trains={r['train_id']:r for r in before['trains']}
    for old in before['leg_timing']:
        track=trains[old['train_id']]['location']if old['trip_id']=='@initial'else trips[old['trip_id']]['route'][old['leg_index']]
        arrival=old['departure_second']+tracks[track]['travel_seconds']+old.get('approach_holding_seconds',0)
        if not old['departure_second']<as_of<=arrival:continue
        new=timing.get(timing_key(old['train_id'],old['trip_id'],old['leg_index']),{})
        if new.get('departure_second')!=old['departure_second']or new.get('approach_holding_seconds',0)!=old.get('approach_holding_seconds',0):return False
    return True


def apply_assignment_plan(data,as_of,assignments,*,deadline=None,profile='fixed',seed=7,service_end_second=None):
    _deadline(deadline);problem=build_assignment_problem(data,as_of)
    jobs={j['job_id']:j for j in problem['jobs']};available={t['train_id']:t for t in problem['availability']}
    if set(assignments)!=set(jobs):raise ValueError('assignment must cover every future service job')
    trips={t['trip_id']:t for t in data['plans'][0]['trips']};tracks={t['track_id']:t for t in data['tracks']}
    retained=set(problem['retained_ids']);out=copy.deepcopy(data)
    out['plans'][0]['trips']=[copy.deepcopy(t)for t in data['plans'][0]['trips']if t['trip_id']in retained]
    out['leg_timing']=[copy.deepcopy(r)for r in data['leg_timing']if r['trip_id']=='@initial'or r['trip_id']in retained]
    old_timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in out['leg_timing']if r['departure_second']>=0}
    if set(assignments.values())-set(available):raise ValueError('assignment uses an unavailable train')
    reserve_starts={}
    def append(tid,second,route,passenger,identity):
        if identity in trips:raise ValueError('new assignment trip identity collides')
        out['plans'][0]['trips'].append(dict(trip_id=identity,train_id=tid,departure_second=second,route=route,passenger=passenger))
        bounds[timing_key(tid,identity,0)]=second
    for tid,state in available.items():
        _deadline(deadline)
        owned=sorted((jobs[j]for j,train in assignments.items()if train==tid),key=lambda j:j['departure_second'])
        location,ready=state['location'],state['ready_second']
        if state.get('locked_exit'):
            if owned:raise ValueError('assignment uses a train with a committed depot return')
            continue
        if owned and not state['on_line']:
            departure=owned[0]['departure_second']
            if departure<ready:raise ValueError('reserve preparation is incomplete')
            entry=departure-state['entry_lead_seconds'];reserve_starts[tid]=entry
            append(tid,entry,[state['entry_track']],False,f'G44ASSIGN:{as_of}:{tid}:ENTRY')
        for job in owned:
            if location!=job['origin']or ready>job['departure_second']:raise ValueError('train is not ready at the service origin')
            old=trips[job['job_id']];identity=f'G44ASSIGN:{as_of}:{old["trip_id"]}:{tid}'
            append(tid,job['departure_second'],copy.deepcopy(old['route']),True,identity)
            for i in range(len(old['route'])):
                row=copy.deepcopy(old_timing[timing_key(old['train_id'],old['trip_id'],i)])
                row.update(train_id=tid,trip_id=identity);out['leg_timing'].append(row)
                bounds[timing_key(tid,identity,i)]=row['departure_second']
            location,ready=job['destination'],job['ready_second']
        if state['on_line']or owned:
            exit_track=next(t for t in tracks.values()if t['from_station']==location and t['track_id'].startswith('EXIT_'))
            append(tid,max(as_of,ready),[exit_track['track_id']],False,f'G44ASSIGN:{as_of}:{tid}:EXIT')
    _deadline(deadline)
    out,timing=compile_operational_timing(out,out['plans'][0]['plan_id'],profile,seed,policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
    _deadline(deadline);motion=simulate(out,out['plans'][0]['plan_id']);before=problem['source_motion']
    if [e for e in before['events']if e['second']<as_of]!=[e for e in motion['events']if e['second']<as_of]:raise ValueError('assignment changes executed movement')
    current=_current_movements_preserved(data,out,as_of)
    if not current:raise ValueError('assignment changes a current movement arrival')
    if service_clocks(data)!=service_clocks(out):raise ValueError('assignment changes fixed service clocks')
    resources=terminal_resources(out,motion)
    gaps=operational_headways(out,motion,as_of,service_end_second or motion['duration_seconds'])
    if service_end_second is None:
        gaps['violations']=[r for r in gaps['violations']if r['kind']!='boundary_end']
        gaps['passed']=not gaps['violations']and not gaps['unknown_tracks']
    gaps={k:v for k,v in gaps.items()if k not in ('intervals','boundary_ages')}
    gaps['service_end_boundary_checked']=service_end_second is not None
    audit=dict(movement_passed=gaps['passed']and motion['obligations_closed']and resources['passed'],
        service_clocks_preserved=True,current_movements_preserved=current,executed_prefix_preserved=True,
        terminal_resources=resources,peak_trains_on_line=motion['checks']['peak_trains_on_line'],headways=gaps,
        obligations_closed=motion['obligations_closed'],future_jobs=len(jobs),retained_ids=sorted(retained),
        retained_preparation_obligations=problem['retained_preparation_obligations'],
        preparation_scope='inferred from assumed prescheduled reference; conservatively committed',
        reserve_starts=reserve_starts,profile=profile,seed=seed,timing_holding_seconds=timing['holding_seconds'],
        scope='fixed service clocks; physical assignment check, passenger comparison pending',
        release_authorized=False,passenger_checked=False,cash_saving_rub=None,
        _data_digest=digest(out),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    _deadline(deadline);return out,audit


def optimize_fixed_timetable(data,as_of,*,time_limit_seconds=5,deadline=None,profile='fixed',seed=7,service_end_second=None):
    _deadline(deadline);problem=build_assignment_problem(data,as_of)
    result=solve_fixed_assignments(problem['jobs'],problem['availability'],time_limit_seconds=time_limit_seconds,
                                  line_train_limit=parameter(data['headway_policy'],'line_train_limit'))
    _deadline(deadline)
    if not result['assignments']:return None,dict(solver=result,movement_passed=False,release_authorized=False)
    changed,audit=apply_assignment_plan(data,as_of,result['assignments'],deadline=deadline,profile=profile,seed=seed,
                                       service_end_second=service_end_second)
    audit['solver']=result;return changed,audit
