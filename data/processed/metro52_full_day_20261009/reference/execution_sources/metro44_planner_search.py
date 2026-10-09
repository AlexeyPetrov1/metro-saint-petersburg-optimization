"""Finite physical dispatch experiments; stable GOAL 4.4 sources stay unchanged."""

import copy
import heapq
import statistics
from datetime import datetime, timedelta
from time import monotonic

from dwell import compile_operational_timing, timing_key
from goal43_policy import parameter
from metro_daily import _geometry
from metro44_planner import headway_seconds, _phase, trip_finish
from rolling import operational_headways
from simulation import simulate, digest


def search_specs():
    return [dict(candidate_id=f'REPAIR:{h}:{phase}:m{margin}',peak_headway=h,
                 phase_seconds={'L1_01':0,'L1_19':phase},occupancy_margin=margin)
            for h,phase,margin in ((120,40,0),(118,40,1),(115,40,2),
                                   (113,80,2),(125,0,0),(120,80,1))]


def _current_motion_preserved(before, after, as_of):
    old={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in before['leg_timing']}
    new={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in after['leg_timing']}
    tracks={t['track_id']:t for t in before['tracks']}
    trips={t['trip_id']:t for t in before['plans'][0]['trips']}
    trains={t['train_id']:t for t in before['trains']}
    for key,row in old.items():
        track_id=trains[row['train_id']]['location'] if row['trip_id']=='@initial' else trips[row['trip_id']]['route'][row['leg_index']]
        arrival=row['departure_second']+tracks[track_id]['travel_seconds']+row.get('approach_holding_seconds',0)
        if row['departure_second'] < as_of <= arrival:
            replacement=new.get(key,{})
            if replacement.get('departure_second')!=row['departure_second']: return False
            if replacement['departure_second']+tracks[track_id]['travel_seconds']+replacement.get('approach_holding_seconds',0)!=arrival: return False
    return True


def _release_deadlines(data, as_of, retained):
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    tracks={t['track_id']:t for t in data['tracks']}
    releases=parameter(data['headway_policy'],'depot_release_s')
    count=0
    for train in data['trains']:
        trips=sorted((t for t in data['plans'][0]['trips']if t['train_id']==train['train_id']),key=lambda t:t['departure_second'])
        for i,trip in enumerate(trips):
            if trip['trip_id']in retained or not trip['route'][0].startswith('ENTRY_'): continue
            service=next((t for t in trips[i+1:]if t['passenger']),None)
            if service is None: return False,count
            departure=timing[timing_key(train['train_id'],service['trip_id'],0)]['departure_second']
            depot='Avtovo'if 'AVTOVO'in tracks[trip['route'][0]]['from_station']else 'Northern'
            if departure-releases[depot] < as_of: return False,count
            count+=1
    return True,count


def capacity_repair_command(data, motion, as_of, retained):
    """Delay an uncommitted entry until an actual exit frees its line slot."""
    tracks={r['track_id']:r for r in data['tracks']}
    stations={r['station_id']:r for r in data['stations']}
    trips={r['trip_id']:r for r in data['plans'][0]['trips']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    runtime={r['train_id']:(r['state'],r['location']) for r in data['trains']}
    def on_line(state,location):
        if state=='moving': return any(stations[tracks[location][k]]['kind']=='line' for k in ('from_station','to_station'))
        return stations[location]['kind']=='line'
    count=sum(on_line(*r) for r in runtime.values())
    limit=data['max_trains_on_line']
    if count>limit: raise ValueError('initial line occupancy exceeds capacity; cannot repair future ENTRY')
    for i,event in enumerate(motion['events']):
        tid=event['train_id']; previous=runtime[tid]
        current=(event['next_state'],event['location']);runtime[tid]=current
        count+=int(on_line(*current))-int(on_line(*previous))
        if count<=limit: continue
        trip=trips[event['trip_id']]
        if event['event_type']!='departure' or not trip['route'][0].startswith('ENTRY_'):
            raise ValueError('capacity excess is not repairable by a future ENTRY')
        if event['second']<as_of or trip['trip_id']in retained:
            raise ValueError('capacity repair would move a committed ENTRY')
        exit_event=next((e for e in motion['events'][i+1:]
            if e['event_type']=='arrival' and stations[e['location']]['kind']=='depot'
            and trips[e['trip_id']]['route'][0].startswith('EXIT_')),None)
        if exit_event is None: raise ValueError('capacity repair has no following EXIT arrival')
        owned=sorted((t for t in trips.values()if t['train_id']==tid),key=lambda t:t['departure_second'])
        service=next((t for t in owned[owned.index(trip)+1:]if t['passenger']),None)
        if service is None: raise ValueError('capacity repair has no linked passenger service')
        delta=exit_event['second']+1-event['second']
        service_at=timing[timing_key(tid,service['trip_id'],0)]['departure_second']
        depot=tracks[trip['route'][0]]['from_station']
        release=parameter(data['headway_policy'],'depot_release_s')['Avtovo'if'AVTOVO'in depot else'Northern']
        service_bound=max(service_at+delta,as_of+release)
        return dict(first_violation_second=event['second'],entry_trip_id=trip['trip_id'],service_trip_id=service['trip_id'],
            train_id=tid,entry_not_before_second=exit_event['second']+1,service_not_before_second=service_bound,
            exit_arrival_second=exit_event['second'],commands={timing_key(tid,trip['trip_id'],0):exit_event['second']+1,
                                                            timing_key(tid,service['trip_id'],0):service_bound})
    return None


def terminal_parking_audit(data, motion):
    capacities=data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    runtime={r['train_id']:(r['state'],r['location']) for r in data['trains']}
    starts={tid:(location,0)for tid,(state,location)in runtime.items()if state=='standby'and location in capacities}
    spans={terminal:[]for terminal in capacities}
    for event in motion['events']:
        tid=event['train_id']
        if tid in starts:
            terminal,start=starts.pop(tid)
            if start<event['second']: spans[terminal].append((start,event['second']))
        state,location=event['next_state'],event['location'];runtime[tid]=(state,location)
        if state=='standby'and location in capacities: starts[tid]=(location,event['second'])
    for terminal,start in starts.values():
        if start<motion['duration_seconds']:spans[terminal].append((start,motion['duration_seconds']))
    maxima={};violations=[]
    for terminal,intervals in spans.items():
        changes=sorted([(a,1)for a,b in intervals]+[(b,-1)for a,b in intervals])
        count=peak=0
        for second,delta in changes:
            count+=delta;peak=max(peak,count)
            if count>capacities[terminal]:violations.append(dict(terminal=terminal,second=second,count=count,capacity=capacities[terminal]))
        maxima[terminal]=peak
    return dict(passed=not violations,maximum_by_terminal=maxima,violations=violations[:10],
                scope='actual terminal standby intervals; half-open occupancy')


def effective_peak_headways(data, motion):
    tracks={r['track_id']:r for r in data['tracks']}
    terminals=sorted(s['station_id']for s in data['stations']if s['kind']=='line')
    chosen=((terminals[0],next(t for t,r in tracks.items()if r['from_station']==terminals[0]and t.endswith('_up'))),
            (terminals[-1],next(t for t,r in tracks.items()if r['from_station']==terminals[-1]and t.endswith('_down'))))
    origin=datetime.fromisoformat(data['period_start'])
    windows=[];all_gaps=[]
    for terminal,track in chosen:
        ts=sorted(e['second']for e in motion['events']if e['event_type']=='departure'and e['location']==track)
        for left,right in ((8,10),(17,19)):
            start=int((origin.replace(hour=left,minute=0,second=0)-origin).total_seconds())
            end=start+(right-left)*3600
            gaps=[b-a for a,b in zip(ts,ts[1:])if start<=b<end]
            windows.append(dict(terminal=terminal,start_second=start,end_second=end,count=len(gaps),
                mean_seconds=statistics.mean(gaps)if gaps else None,median_seconds=statistics.median(gaps)if gaps else None,
                minimum_seconds=min(gaps)if gaps else None,maximum_seconds=max(gaps)if gaps else None))
            all_gaps.extend(gaps)
    return dict(scope='actual terminal departure intervals in weekday peak windows',windows=windows,
        mean_seconds=statistics.mean(all_gaps)if all_gaps else None,minimum_seconds=min(all_gaps)if all_gaps else None,
        under_150_count=sum(g<150 for g in all_gaps),count=len(all_gaps))


def evaluate_spec(data, as_of, service_end, spec, *, deadline=None):
    row=copy.deepcopy(spec)
    try:
        changed,audit=rebuild_timetable(data,as_of,service_end,headway=spec['peak_headway'],
            phase_seconds=spec['phase_seconds'],occupancy_margin=spec['occupancy_margin'],deadline=deadline)
        row.update(status='movement_checked'if audit['movement_passed']else 'rejected',
            audit={k:v for k,v in audit.items()if not k.startswith('_') and k!='departures'})
        row['effective_peak_headways']=effective_peak_headways(changed,simulate(changed,changed['plans'][0]['plan_id']))
        if not audit['movement_passed']: row['reason']='complete headway/return check failed'
        return row,changed if audit['movement_passed']else None
    except (TimeoutError,ValueError)as error:
        row.update(status='evaluation_timeout'if isinstance(error,TimeoutError)else 'rejected',reason=str(error))
        return row,None


def rebuild_timetable(data, as_of, service_end, *, headway=None, phase_seconds=0,
                      priority_rotation=0, initial=False, deadline=None, cancel_event=None,
                      wear_priority=False, occupancy_margin=0, repair_future_platforms=True,
                      allow_future_advance=True, repair_capacity=False, maximum_capacity_repairs=5):
    """Preserve the committed prefix; replace future service and its vehicle duties."""
    def check_deadline():
        if deadline is not None and monotonic() >= deadline or cancel_event is not None and cancel_event.is_set():
            raise TimeoutError('timetable generation deadline')
    check_deadline()
    if occupancy_margin not in (0,1,2): raise ValueError('occupancy margin must be 0, 1 or 2')
    if not isinstance(maximum_capacity_repairs,int) or not 0<=maximum_capacity_repairs<=5:
        raise ValueError('at most five capacity repairs')
    policy = data['headway_policy']
    tracks, stations, routes, half = _geometry(data)
    origin = datetime.fromisoformat(data['period_start'])
    lock = 0 if initial else as_of
    phases = {terminal: _phase(phase_seconds, terminal) for terminal in routes}
    timings = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data.get('leg_timing',[])}
    by_train = {}
    for trip in data['plans'][0]['trips']:
        by_train.setdefault(trip['train_id'],[]).append(trip)
    retained = []
    entry_obligations = set()
    for trips in by_train.values():
        owned = sorted(trips, key=lambda t: t['departure_second'])
        for i, trip in enumerate(owned):
            first_at = timings.get(timing_key(trip['train_id'], trip['trip_id'], 0), {}).get('departure_second', trip['departure_second'])
            if trip['route'][0].startswith('ENTRY_') and first_at < as_of:
                following = next((t for t in owned[i+1:] if t['passenger']), None)
                if following is not None: entry_obligations.add(following['trip_id'])
    for train in data['trains']:
        trips = sorted(by_train.get(train['train_id'],[]),key=lambda t:(t['departure_second'],t['trip_id']))
        for i, trip in enumerate(trips):
            first = timings.get(timing_key(trip['train_id'],trip['trip_id'],0),{})
            first_at = first.get('departure_second',trip['departure_second'])
            preserve_initial = i == 0 and train['location'] in routes or i == 0 and not train['location'].startswith('DEPOT_')
            if (preserve_initial and trip['passenger']) or not initial and (first_at < lock or trip['trip_id'] in entry_obligations):
                retained.append(copy.deepcopy(trip))
    assigned = {r['trip_id'] for r in retained}
    out = copy.deepcopy(data)
    out['plans'][0]['trips'] = retained
    out['leg_timing'] = [r for r in out.get('leg_timing',[]) if r['trip_id']=='@initial' or r['trip_id'] in assigned]
    roster = sorted(out['trains'],key=lambda t:t['train_id'])
    roster = roster[priority_rotation:]+roster[:priority_rotation]
    prices=parameter(policy,'maintenance_prices_rub_by_type')
    interval=parameter(policy,'to1_interval_km')
    wear={t['train_id']: prices[t['train_type']]['TO1']/interval if wear_priority else 0 for t in roster}
    states = {}
    intervals = {}
    platforms = {terminal:[] for terminal in routes}
    last = {terminal:-1 for terminal in routes}
    for r in data.get('previous_departures',[]):
        for terminal,route in routes.items():
            if r['track_id'] == route[0]: last[terminal] = max(last[terminal],r['departure_second'])
    for train in roster:
        tid = train['train_id']
        location, ready = train['location'],train.get('ready_after_seconds',0)
        owned = sorted((t for t in retained if t['train_id']==tid),key=lambda t:t['departure_second'])
        spans = []
        visit = None
        start = 0 if not location.startswith('DEPOT_') else None
        for trip in owned:
            first = timings.get(timing_key(tid,trip['trip_id'],0),{})
            dep = first.get('departure_second',trip['departure_second'])
            track = tracks[trip['route'][0]]
            if visit is not None: visit[1] = dep
            if trip['passenger'] and track['from_station'] in routes:
                last[track['from_station']] = max(last[track['from_station']],dep)
            if trip['route'][0].startswith('ENTRY_'): start = dep
            location,ready = trip_finish(trip,timings,tracks,stations)
            if location in routes:
                visit_ready=ready
                visit=[visit_ready-180,ready,visit_ready]
                platforms[location].append(visit)
            else: visit=None
            if trip['route'][0].startswith('EXIT_') and start is not None:
                spans.append([start,ready]); start=None
        states[tid] = dict(location=location,ready=ready,open_start=start,spans=spans,
                           visit=visit,
                           default_end=ready+120 if start is not None else None)
        intervals[tid] = spans

    capacity_cache = {}
    # Unused depot trains share the same occupancy sweep at a candidate instant.
    def capacity_with(tid, start, end):
        old=states[tid]
        begin=max(start,old['default_end']) if old['open_start']==start else start
        if begin >= end: return True
        key=(begin,end) if old['open_start'] is None and not old['spans'] else None
        if key is not None and key in capacity_cache: return capacity_cache[key]
        changes = [(begin,1),(end,-1)]
        for other, state in states.items():
            if other==tid: continue
            spans = list(state['spans'])
            if state['open_start'] is not None: spans.append((state['open_start'],state['default_end']))
            for a,b in spans:
                if a < end and b > begin: changes.extend([(max(a,begin),1),(min(b,end),-1)])
        count = peak = 0
        for _,delta in sorted(changes):
            count += delta; peak=max(peak,count)
        passed=peak <= parameter(policy,'line_train_limit')-occupancy_margin
        if key is not None: capacity_cache[key]=passed
        return passed

    def platform_available(terminal, interval, replaced=None):
        if repair_future_platforms and interval[0] >= as_of: return True
        changes=[]
        for span in platforms[terminal]:
            if span is not replaced and span[0] < interval[1] and span[1] > interval[0]:
                changes.extend([(max(span[0],interval[0]),1),(min(span[1],interval[1]),-1)])
        changes.extend([(interval[0],1),(interval[1],-1)])
        count=0
        for _,delta in sorted(changes):
            count+=delta
            if count > 2: return False
        return True

    def parking_available(terminal, interval, replaced=None):
        if repair_future_platforms and interval[0] >= as_of: return True
        if interval[0] >= interval[1]: return True
        for span in platforms[terminal]:
            if span is not replaced and span[2] < span[1] and span[2] < interval[1] and span[1] > interval[0]: return False
        return True

    serial = 0
    used_ids = {t['trip_id'] for t in data['plans'][0]['trips']}
    def append(tid, departure, route, passenger, tag):
        nonlocal serial
        while True:
            serial += 1
            trip_id=f'{tid}:G44:D{as_of}:{tag}:{serial}'
            if trip_id not in used_ids: break
        used_ids.add(trip_id)
        out['plans'][0]['trips'].append(dict(trip_id=trip_id,train_id=tid,
            departure_second=int(departure),route=route,passenger=passenger))

    def return_to_depot(tid):
        state=states[tid]
        if state['open_start'] is None: return
        terminal=state['location']
        exit_track=next(r for r in tracks.values() if r['from_station']==terminal and r['track_id'].startswith('EXIT_'))
        departure=max(as_of,state['ready'])
        if state['visit'] is not None: state['visit'][1]=departure
        append(tid,departure,[exit_track['track_id']],False,'EXIT')
        state['spans'].append([state['open_start'],departure+120])
        state.update(location=exit_track['to_station'],ready=departure+120+stations[exit_track['to_station']]['turnaround_seconds'],open_start=None,default_end=None,visit=None)

    queue=[]
    for terminal in routes:
        earliest=max(lock,last[terminal]+1)
        at=origin+timedelta(seconds=earliest)
        h=headway_seconds(headway,at,terminal,policy,origin,cycle=half*2)
        heapq.heappush(queue,(max(lock,last[terminal]+h)+phases[terminal],terminal))
    projected=[]
    while queue:
        check_deadline()
        second,terminal=heapq.heappop(queue)
        capacity_cache.clear()
        if second >= service_end: continue
        h=headway_seconds(headway,origin+timedelta(seconds=second),terminal,policy,origin,cycle=half*2)
        second=max(second,last[terminal]+h)
        if second >= service_end: continue
        depot_track=next(r for r in tracks.values() if r['to_station']==terminal and r['track_id'].startswith('ENTRY_'))
        depot=depot_track['from_station']
        release=parameter(policy,'depot_release_s')['Avtovo' if 'AVTOVO' in depot else 'Northern']
        max_idle=max(0,2*h-parameter(policy,'turnaround_s')-2)
        options=[]
        target=tracks[routes[terminal][-1]]['to_station']
        if not platform_available(target,(second+half-180,second+half)):
            heapq.heappush(queue,(second+5,terminal)); continue
        for rank, train in enumerate(roster):
            check_deadline()
            tid=train['train_id']; state=states[tid]
            if state['location']==terminal and state['ready'] <= second and second-state['ready'] > max_idle:
                return_to_depot(tid)
                capacity_cache.clear()
            if state['location']==terminal and state['ready'] <= second:
                idle=second-state['ready']
                if (idle <= max_idle and capacity_with(tid,state['open_start'],second+half+120)
                        and platform_available(terminal,(state['visit'][0],second),state['visit'])
                        and parking_available(terminal,(state['ready'],second),state['visit'])):
                    options.append((0,wear[tid],state['ready'],rank,tid,'reuse'))
                elif stations[depot]['turnaround_seconds']>0 and second >= state['ready']+120+stations[depot]['turnaround_seconds']+release and second-release >= as_of:
                    if capacity_with(tid,second-300,second+half+120) and platform_available(terminal,(second-180,second)):
                        options.append((1,wear[tid],state['ready'],rank,tid,'return'))
            elif state['location']==depot and state['ready'] <= second-release and second-release >= as_of:
                if state['spans'] and stations[depot]['turnaround_seconds']==0: continue
                if capacity_with(tid,second-300,second+half+120) and platform_available(terminal,(second-180,second)):
                    options.append((1,wear[tid],state['ready'],rank,tid,'depot'))
        if not options:
            heapq.heappush(queue,(second+5,terminal))
            continue
        _,_,_,_,tid,kind=min(options)
        state=states[tid]
        if kind=='return': return_to_depot(tid)
        if kind != 'reuse':
            append(tid,second-300,[depot_track['track_id']],False,'ENTRY')
            state['open_start']=second-300
            platforms[terminal].append([second-180,second,second])
        else: state['visit'][1]=second
        append(tid,second,routes[terminal],True,'SERVICE')
        visit=[second+half-180,second+half,second+half]
        platforms[target].append(visit)
        state.update(location=target,ready=second+half,default_end=second+half+120)
        state['visit']=visit
        last[terminal]=second
        projected.append(dict(second=second,terminal=terminal,target_headway=h))
        heapq.heappush(queue,(second+h,terminal))
    for train in roster: return_to_depot(train['train_id'])
    departures={timing_key(t['train_id'],t['trip_id'],0):
        (timings.get(timing_key(t['train_id'],t['trip_id'],0),{}).get('departure_second',t['departure_second'])
         if t['trip_id'] in assigned else t['departure_second'])
        for t in out['plans'][0]['trips']}
    for row in data.get('leg_timing', []):
        if row['trip_id'] in assigned and (not allow_future_advance or row['departure_second'] < as_of):
            departures[timing_key(row['train_id'],row['trip_id'],row['leg_index'])] = row['departure_second']
    capacity_repairs=[]
    for attempt in range(maximum_capacity_repairs+1):
        check_deadline()
        compile_input=out
        if repair_capacity:
            compile_input=copy.deepcopy(out)
            compile_input.pop('max_trains_on_line')
        out,timing=compile_operational_timing(compile_input,out['plans'][0]['plan_id'],'fixed',7,policy=policy,
            as_of=as_of,departure_not_before=departures)
        out['max_trains_on_line']=data['max_trains_on_line']
        check_deadline()
        try:
            motion=simulate(out,out['plans'][0]['plan_id'])
            break
        except ValueError as error:
            if not repair_capacity or not str(error).startswith('line capacity exceeded'):raise
            if attempt==maximum_capacity_repairs:raise ValueError(f'capacity repair exhausted after {attempt}: {error}')from error
            trace_data=copy.deepcopy(out);trace_data.pop('max_trains_on_line')
            # This unconstrained diagnostic is never returned as a feasible motion.
            trace=simulate(trace_data,trace_data['plans'][0]['plan_id'])
            check_deadline()
            repair=capacity_repair_command(out,trace,as_of,assigned)
            if repair is None:raise ValueError('capacity error disagrees with diagnostic trace')from error
            for key,bound in repair['commands'].items():departures[key]=max(departures.get(key,0),bound)
            capacity_repairs.append({k:v for k,v in repair.items()if k!='commands'})
    parking=terminal_parking_audit(out,motion)
    check_deadline()
    gaps=operational_headways(out,motion,as_of,service_end)
    before=simulate(data,data['plans'][0]['plan_id'])
    check_deadline()
    if [e for e in before['events'] if e['second'] < as_of] != [e for e in motion['events'] if e['second'] < as_of]:
        raise ValueError('replan changes executed movement')
    current_preserved=_current_motion_preserved(data,out,as_of)
    depot_passed,release_count=_release_deadlines(out,as_of,assigned)
    if not current_preserved: raise ValueError('repair changes an already started movement arrival')
    if not depot_passed: raise ValueError('repair violates a new depot preparation deadline')
    audit=dict(committed_until_second=lock,passenger_trips=sum(t['passenger'] for t in out['plans'][0]['trips']),
        planning_occupancy_margin=occupancy_margin,phase_seconds=phases,retained_entry_obligations=sorted(entry_obligations),
        repair_future_platforms=repair_future_platforms,allow_future_advance=allow_future_advance,
        current_motion_preserved=current_preserved,depot_preparation_passed=depot_passed,new_release_count=release_count,
        capacity_repairs=capacity_repairs,terminal_parking=parking,
        committed_leg_policy='past and started movement arrivals immutable; remaining future legs may be retimed',
        wear_priority=wear_priority,wear_scope='TO1 resource equivalent; not measured cash savings',
        priority_rotation=priority_rotation,headways_passed=gaps['passed'],
        movement_passed=gaps['passed'] and motion['obligations_closed'] and parking['passed'],
        minimum_headway=gaps['minimum_seconds'],maximum_headway=gaps['maximum_seconds'],
        headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],obligations_closed=motion['obligations_closed'],
        departures=projected,timing_holding_seconds=timing['holding_seconds'])
    audit.update(_data_digest=digest(out),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    check_deadline()
    return out,audit

