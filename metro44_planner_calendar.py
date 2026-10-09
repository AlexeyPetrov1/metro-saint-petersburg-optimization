"""Terminal calendar dispatch without the artificial elapsed-wait retirement rule."""

from __future__ import annotations

import copy
import heapq
import math
from datetime import datetime, timedelta
from time import monotonic

from dwell import timing_key
from metro44_terminal_timing import compile_terminal_timing as compile_operational_timing
from metro44_assignment_plan import terminal_resources, _current_movements_preserved
from goal43_policy import minimum_headway, parameter
from metro_daily import _geometry, reference_counts
from rolling import operational_headways
from simulation import simulate, digest


def headway_seconds(headway, at, terminal, policy, period_start=None, *, cycle=6080):
    """Resolve a clock-aware per-terminal profile; enforce only user headway limits."""
    minimum = minimum_headway(at, policy)
    if isinstance(headway, dict):
        headway = headway.get(terminal, headway.get('*'))
    if isinstance(headway, (list, tuple)):
        if period_start is None: raise ValueError('headway profile requires period_start')
        second = int((at-period_start).total_seconds())
        rows = [r for r in headway if r.get('terminal', '*') in (terminal, '*')
                and r['start_second'] <= second < r['end_second']]
        if len(rows) > 1: raise ValueError('overlapping headway profiles')
        headway = rows[0]['headway_seconds'] if rows else None
    if callable(headway): headway = headway(at, terminal)
    if headway is None: headway = min(parameter(policy,'headway_max_s'),math.ceil(cycle/reference_counts()[at.hour]))
    if isinstance(headway, bool) or not isinstance(headway, int) or not 1 <= headway <= parameter(policy,'headway_max_s'):
        raise ValueError('headway must be integer seconds within the user maximum')
    return max(minimum, headway)


def _phase(phase, terminal):
    value = phase.get(terminal, 0) if isinstance(phase, dict) else phase
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 480:
        raise ValueError('phase must be integer seconds between 0 and 480')
    return value


def trip_finish(trip, timings, tracks, stations):
    at=trip['departure_second']
    for i, track_id in enumerate(trip['route']):
        row=timings.get(timing_key(trip['train_id'],trip['trip_id'],i),{})
        track=tracks[track_id]; station=stations[track['to_station']]
        at=row.get('departure_second',at)+track['travel_seconds']+row.get('approach_holding_seconds',0)
        at+=row.get('service_seconds',station['stop_seconds']+station['turnaround_seconds'])
    return tracks[trip['route'][-1]]['to_station'],at


def reuse_available(ready, second, *, capacity, platform, parking):
    return ready <= second and capacity and platform and parking


def balanced_phase(half, headway):
    return (half % headway + ((-2*half) % headway)//2) % headway


def paired_anchor(first_a,first_b,headway,phase):
    candidates=[]
    for anchor in (first_a,first_b-phase):
        a=anchor+math.ceil((first_a-anchor)/headway)*headway
        b=anchor+phase+math.ceil((first_b-anchor-phase)/headway)*headway
        candidates.append((a-first_a+b-first_b,max(a-first_a,b-first_b),a))
    return min(candidates)[2]


def next_calendar_slot(second, calendar):
    for row in calendar:
        start,end,h,phase=(row[k] for k in ('start_second','end_second','headway_seconds','phase_seconds'))
        if not isinstance(h,int) or h<=0 or not isinstance(phase,int) or not 0<=phase<h or start>=end:
            raise ValueError('invalid terminal calendar')
        if start<=second<end:
            anchor=row.get('anchor_second',start)+phase
            slot=anchor+math.ceil((second-anchor)/h)*h
            return min(slot,end)
    return second


def segment_calendars(routes,origin,service_end,headway,policy,half):
    """Share one phase anchor across adjacent quarters with the same interval."""
    rows={terminal:[] for terminal in routes}
    bounds={0,service_end,*range(0,service_end,900)}
    profiles=headway.values() if isinstance(headway,dict) else (headway,)
    for profile in profiles:
        if isinstance(profile,(list,tuple)):
            bounds.update(int(r[k])for r in profile for k in ('start_second','end_second')if 0<int(r[k])<service_end)
    cuts=sorted(bounds)
    for start,end in zip(cuts,cuts[1:]):
        for index,terminal in enumerate(routes):
            h=headway_seconds(headway,origin+timedelta(seconds=start),terminal,policy,origin,cycle=half*2)
            final=headway_seconds(headway,origin+timedelta(seconds=end-1),terminal,policy,origin,cycle=half*2)
            if h!=final:raise ValueError('calendar interval changes inside a quarter')
            current=rows[terminal]
            if current and current[-1]['headway_seconds']==h and current[-1]['end_second']==start:
                current[-1]['end_second']=end
            else:current.append(dict(start_second=start,end_second=end,headway_seconds=h,
                phase_seconds=0 if index==0 else balanced_phase(half,h)))
    return rows


def rebuild_calendar_timetable(data, as_of, service_end, *, headway=None, phase_seconds=0,
                      priority_rotation=0, initial=False, deadline=None, cancel_event=None,
                      wear_priority=False, calendar=True):
    """Preserve the committed prefix; replace future service and its vehicle duties."""
    def check_deadline():
        if deadline is not None and monotonic() >= deadline or cancel_event is not None and cancel_event.is_set():
            raise TimeoutError('timetable generation deadline')
    check_deadline()
    policy = data['headway_policy']
    tracks, stations, routes, half = _geometry(data)
    origin = datetime.fromisoformat(data['period_start'])
    lock = 0 if initial else as_of
    phases = {terminal: _phase(phase_seconds, terminal) for terminal in routes}
    if calendar and any(phases.values()):raise ValueError('joint calendar supplies the terminal phase')
    calendars=segment_calendars(routes,origin,service_end,headway,policy,half) if calendar else {t:[] for t in routes}
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
        passed=peak <= parameter(policy,'line_train_limit')
        if key is not None: capacity_cache[key]=passed
        return passed

    def platform_available(terminal, interval, replaced=None):
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
    anchored=set()
    while queue:
        check_deadline()
        second,terminal=heapq.heappop(queue)
        capacity_cache.clear()
        if second >= service_end: continue
        while True:
            h=headway_seconds(headway,origin+timedelta(seconds=second),terminal,policy,origin,cycle=half*2)
            second=max(second,last[terminal]+h)
            row=next((r for r in calendars[terminal]if r['start_second']<=second<r['end_second']),None)
            if row is not None and row['start_second']not in anchored:
                start=row['start_second'];h=row['headway_seconds'];terms=list(routes)
                pair=[next(r for r in calendars[t]if r['start_second']<=start<r['end_second'])for t in terms]
                if len(terms)!=2 or any(r['headway_seconds']!=h for r in pair):
                    raise ValueError('paired calendar needs two terminals with the same interval')
                earliest=[max(lock,start,last[t]+h)for t in terms]
                if start<=lock:
                    for i,t in enumerate(terms):
                        ready=[s['ready']for s in states.values()if s['location']==t and s['open_start']is not None]
                        if ready:earliest[i]=max(earliest[i],min(ready))
                anchor=paired_anchor(*earliest,h,pair[1]['phase_seconds'])
                for r in pair:r['anchor_second']=anchor
                anchored.add(start)
            adjusted=next_calendar_slot(second,calendars[terminal])
            if adjusted==second:break
            second=adjusted
        if second >= service_end: continue
        if queue and second>queue[0][0]:
            heapq.heappush(queue,(second,terminal));continue
        depot_track=next(r for r in tracks.values() if r['to_station']==terminal and r['track_id'].startswith('ENTRY_'))
        depot=depot_track['from_station']
        release=parameter(policy,'depot_release_s')['Avtovo' if 'AVTOVO' in depot else 'Northern']
        options=[]
        target=tracks[routes[terminal][-1]]['to_station']
        if not platform_available(target,(second+half-180,second+half)):
            heapq.heappush(queue,(second+5,terminal)); continue
        for rank, train in enumerate(roster):
            check_deadline()
            tid=train['train_id']; state=states[tid]
            if state['location']==terminal and state['ready'] <= second:
                if reuse_available(state['ready'],second,
                        capacity=capacity_with(tid,state['open_start'],second+half+120),
                        platform=platform_available(terminal,(state['visit'][0],second),state['visit']),
                        parking=parking_available(terminal,(state['ready'],second),state['visit'])):
                    options.append((0,wear[tid],state['ready'],rank,tid,'reuse'))
                elif second >= state['ready']+120+stations[depot]['turnaround_seconds']+release and second-release >= as_of:
                    if capacity_with(tid,second-300,second+half+120) and platform_available(terminal,(second-180,second)):
                        options.append((1,wear[tid],state['ready'],rank,tid,'return'))
            elif state['location']==depot and state['ready'] <= second-release and second-release >= as_of:
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
        if row['trip_id'] in assigned:
            departures[timing_key(row['train_id'],row['trip_id'],row['leg_index'])] = row['departure_second']
    check_deadline()
    out,timing=compile_operational_timing(out,out['plans'][0]['plan_id'],'fixed',7,policy=policy,
        as_of=as_of,departure_not_before=departures)
    check_deadline()
    motion=simulate(out,out['plans'][0]['plan_id'])
    check_deadline()
    gaps=operational_headways(out,motion,as_of,service_end)
    resources=terminal_resources(out,motion)
    current=_current_movements_preserved(data,out,as_of)
    if not current:raise ValueError('calendar changes a started arrival')
    before=simulate(data,data['plans'][0]['plan_id'])
    check_deadline()
    if [e for e in before['events'] if e['second'] < as_of] != [e for e in motion['events'] if e['second'] < as_of]:
        raise ValueError('replan changes executed movement')
    audit=dict(committed_until_second=lock,passenger_trips=sum(t['passenger'] for t in out['plans'][0]['trips']),
        planning_occupancy_margin=0,phase_seconds=phases,retained_entry_obligations=sorted(entry_obligations),
        committed_leg_policy='started routes retained; future legs may be delayed, never advanced',
        wear_priority=wear_priority,wear_scope='TO1 resource equivalent; not measured cash savings',
        priority_rotation=priority_rotation,headways_passed=gaps['passed'],
        movement_passed=gaps['passed'] and motion['obligations_closed'] and resources['passed'],
        terminal_resources=resources,current_movements_preserved=current,
        artificial_idle_cutoff_removed=True,joint_calendar=calendars,
        minimum_headway=gaps['minimum_seconds'],maximum_headway=gaps['maximum_seconds'],
        headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],obligations_closed=motion['obligations_closed'],
        departures=projected,timing_holding_seconds=timing['holding_seconds'])
    audit.update(_data_digest=digest(out),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    check_deadline()
    return out,audit

