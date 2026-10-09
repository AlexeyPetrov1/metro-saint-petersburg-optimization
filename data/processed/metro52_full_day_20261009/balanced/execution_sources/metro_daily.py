"""Causal daily timetable experiments over the existing metro engines."""

from __future__ import annotations

import copy
import heapq
import math
from datetime import datetime, timedelta

import pandas as pd

from dwell import compile_operational_timing, timing_key
from forecasting import validate_bundle
from goal43_policy import load_policy, minimum_headway, parameter
from goal43_replay import build_operational_day
from rolling import operational_headways
from simulation import simulate


def reference_counts():
    # The photographed row, not counts reconstructed from a supposed 99-minute cycle.
    return {5: 12, 6: 33, 7: 50, 8: 53, 9: 43, 10: 36, 11: 33, 12: 30,
            13: 30, 14: 33, 15: 40, 16: 46, 17: 50, 18: 50, 19: 46,
            20: 33, 21: 25, 22: 23, 23: 18, 0: 12}


def desired_headway(at, cycle, policy, count_delta=0):
    requested = max(1, min(53, reference_counts()[at.hour]+count_delta))
    # Every departure along this trip must respect the current clock's minimum.
    route_minimum = max(minimum_headway(at+timedelta(seconds=offset), policy)
                        for offset in range(0, cycle//2, 60))
    operational_floor = math.ceil((cycle+240)/(parameter(policy,'line_train_limit')-4))
    return min(480, max(route_minimum, operational_floor, math.ceil(cycle/requested)))


def oracle_bundle(frame, station_ids, as_of, service_end):
    now = pd.Timestamp(as_of).floor('h')
    end = min(now+pd.Timedelta(hours=2), pd.Timestamp(service_end))
    selected = frame[frame.station_id.isin(station_ids) & frame.interval_start.ge(now)
                     & frame.interval_start.lt(end)].copy()
    expected = len(station_ids)*int((end-now).total_seconds()//900)
    if (len(selected) != expected or not selected.is_complete.all()
            or selected.duplicated(['station_id','interval_start']).any()
            or not selected.source_id.eq('uploaded-spb-line1-15min').all()):
        raise ValueError('oracle requires complete real quarter-hour coverage')
    rows = [dict(station_id=r.station_id, interval_start=r.interval_start.isoformat(),
        interval_end=r.interval_end.isoformat(), predicted_count=int(r.entry_count), coverage='complete')
        for r in selected.sort_values(['interval_start','station_id']).itertuples(index=False)]
    bundle = dict(schema_version='1.0', model_version='research-oracle-real-15m-v1',
        issued_at=now.isoformat(), as_of=now.isoformat(), available_at=now.isoformat(),
        features_available_at=now.isoformat(), target='turnstile_entries', unit='passengers_per_15m',
        coverage='complete', feature_version='intentionally-known-future-entry-counts',
        source_versions=dict(operationally_available=False, future_truth_intentionally_used=True),
        included_event_ids=[], rows=rows, refresh_seconds=3600,
        evaluation_mode='research_oracle_future_entries', claimed_ml_WAPE=None,
        availability_assumption='Research oracle: future truth supplied intentionally, unavailable in service')
    validate_bundle(bundle, station_ids, as_of)
    return bundle


def proposal_specs():
    return [dict(candidate_id=f'REPLAN:n{delta:+}:r{rotation}', count_delta=delta,
                 priority_rotation=rotation) for delta, rotation in
            [(0,0), (-1,0), (-2,0), (-2,7), (-4,0), (-8,0), (-12,0), (2,0), (12,0)]]


def exposure_safe(exposure):
    return exposure.get('admissible') is True


def _geometry(data):
    tracks = {r['track_id']:r for r in data['tracks']}
    stations = {r['station_id']:r for r in data['stations']}
    up = sorted((r for r in data['tracks'] if r['track_id'].endswith('_up')), key=lambda r:r['track_id'])
    down = sorted((r for r in data['tracks'] if r['track_id'].endswith('_down')), key=lambda r:r['track_id'], reverse=True)
    routes = {up[0]['from_station']:[r['track_id'] for r in up],
              down[0]['from_station']:[r['track_id'] for r in down]}
    half = sum(tracks[t]['travel_seconds'] + stations[tracks[t]['to_station']]['stop_seconds']
               + stations[tracks[t]['to_station']]['turnaround_seconds'] for t in routes[up[0]['from_station']])
    return tracks, stations, routes, half


def _trip_finish(trip, timings, tracks, stations):
    at = trip['departure_second']
    for i, track_id in enumerate(trip['route']):
        row = timings.get(timing_key(trip['train_id'], trip['trip_id'], i), {})
        track = tracks[track_id]; station = stations[track['to_station']]
        at = row.get('departure_second',at)+track['travel_seconds']+row.get('approach_holding_seconds',0)
        at += row.get('service_seconds',station['stop_seconds']+station['turnaround_seconds'])
        at += row.get('holding_seconds',0)
    return tracks[trip['route'][-1]]['to_station'],at


def rebuild_timetable(data, as_of, service_end, *, count_delta=0, change_end_second=None,
                      priority_rotation=0, initial=False):
    """Preserve the committed prefix; replace future service and its vehicle duties."""
    policy = data['headway_policy']
    tracks, stations, routes, half = _geometry(data)
    origin = datetime.fromisoformat(data['period_start'])
    lock = 0 if initial else min(service_end, as_of+max(parameter(policy,'depot_release_s').values()))
    change_end = service_end if change_end_second is None else change_end_second
    timings = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data.get('leg_timing',[])}
    by_train = {}
    for trip in data['plans'][0]['trips']:
        by_train.setdefault(trip['train_id'],[]).append(trip)
    retained = []
    for train in data['trains']:
        trips = sorted(by_train.get(train['train_id'],[]),key=lambda t:(t['departure_second'],t['trip_id']))
        for i, trip in enumerate(trips):
            first = timings.get(timing_key(trip['train_id'],trip['trip_id'],0),{})
            first_at = first.get('departure_second',trip['departure_second'])
            preserve_initial = i == 0 and train['location'] in routes or i == 0 and not train['location'].startswith('DEPOT_')
            if (preserve_initial and trip['passenger']) or not initial and first_at < lock:
                retained.append(copy.deepcopy(trip))
    assigned = {r['trip_id'] for r in retained}
    out = copy.deepcopy(data)
    out['plans'][0]['trips'] = retained
    out['leg_timing'] = [r for r in out.get('leg_timing',[]) if r['trip_id']=='@initial' or r['trip_id'] in assigned]
    roster = sorted(out['trains'],key=lambda t:t['train_id'])
    roster = roster[priority_rotation:]+roster[:priority_rotation]
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
            location,ready = _trip_finish(trip,timings,tracks,stations)
            if location in routes:
                last_row=timings.get(timing_key(tid,trip['trip_id'],len(trip['route'])-1),{})
                visit_ready=ready-last_row.get('holding_seconds',0)
                visit=[visit_ready-180,ready,visit_ready]
                platforms[location].append(visit)
            else: visit=None
            if trip['route'][0].startswith('EXIT_') and start is not None:
                spans.append([start,ready]); start=None
        states[tid] = dict(location=location,ready=ready,open_start=start,spans=spans,
                           visit=visit,
                           default_end=ready+120 if start is not None else None)
        intervals[tid] = spans

    # Interval sweep includes depot approach and return, not just passenger trains.
    def capacity_with(tid, start, end):
        old=states[tid]
        begin=max(start,old['default_end']) if old['open_start']==start else start
        if begin >= end: return True
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
        # Leave four positions for compiler-induced terminal/approach holds.
        # The movement engine still enforces the actual user limit of 53.
        return peak <= parameter(policy,'line_train_limit')-4

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
    def append(tid, departure, route, passenger, tag):
        nonlocal serial
        serial += 1
        out['plans'][0]['trips'].append(dict(trip_id=f'{tid}:D{as_of}:{tag}:{serial}',train_id=tid,
            departure_second=int(departure),route=route,passenger=passenger))

    def return_to_depot(tid):
        state=states[tid]
        if state['open_start'] is None: return
        terminal=state['location']
        exit_track=next(r for r in tracks.values() if r['from_station']==terminal and r['track_id'].startswith('EXIT_'))
        append(tid,state['ready'],[exit_track['track_id']],False,'EXIT')
        state['spans'].append([state['open_start'],state['ready']+120])
        state.update(location=exit_track['to_station'],ready=state['ready']+120+stations[exit_track['to_station']]['turnaround_seconds'],open_start=None,default_end=None,visit=None)

    queue=[]
    for terminal in routes:
        earliest=max(lock,last[terminal]+1)
        at=origin+timedelta(seconds=earliest)
        h=desired_headway(at,half*2,policy,count_delta if earliest < change_end else 0)
        heapq.heappush(queue,(max(lock,last[terminal]+h),terminal))
    projected=[]
    while queue:
        second,terminal=heapq.heappop(queue)
        if second >= service_end: continue
        h=desired_headway(origin+timedelta(seconds=second),half*2,policy,count_delta if second < change_end else 0)
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
            tid=train['train_id']; state=states[tid]
            if state['location']==terminal and state['ready'] <= second:
                idle=second-state['ready']
                if (idle <= max_idle and capacity_with(tid,state['open_start'],second+half+120)
                        and platform_available(terminal,(state['visit'][0],second),state['visit'])
                        and parking_available(terminal,(state['ready'],second),state['visit'])):
                    options.append((0,state['ready'],rank,tid,'reuse'))
                elif second >= state['ready']+120+stations[depot]['turnaround_seconds']+release and second-release >= as_of:
                    if capacity_with(tid,second-300,second+half+120) and platform_available(terminal,(second-180,second)):
                        options.append((1,state['ready'],rank,tid,'return'))
            elif state['location']==depot and state['ready'] <= second-release and second-release >= as_of:
                if capacity_with(tid,second-300,second+half+120) and platform_available(terminal,(second-180,second)):
                    options.append((1,state['ready'],rank,tid,'depot'))
        if not options:
            heapq.heappush(queue,(second+5,terminal))
            continue
        _,_,_,tid,kind=min(options)
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
        projected.append(dict(second=second,terminal=terminal,requested_count=reference_counts()[(origin+timedelta(seconds=second)).hour],
                              target_headway=h,count_delta=count_delta if second < change_end else 0))
        heapq.heappush(queue,(second+h,terminal))
    for train in roster: return_to_depot(train['train_id'])
    departures={timing_key(t['train_id'],t['trip_id'],0):
        timings.get(timing_key(t['train_id'],t['trip_id'],0),{}).get('departure_second',t['departure_second'])
        for t in out['plans'][0]['trips']}
    out,timing=compile_operational_timing(out,out['plans'][0]['plan_id'],'fixed',7,policy=policy,
        as_of=as_of,departure_not_before=departures)
    motion=simulate(out,out['plans'][0]['plan_id'])
    gaps=operational_headways(out,motion,as_of,service_end)
    audit=dict(committed_until_second=lock,passenger_trips=sum(t['passenger'] for t in out['plans'][0]['trips']),
        planning_occupancy_margin=4,
        count_delta=count_delta,priority_rotation=priority_rotation,headways_passed=gaps['passed'],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],obligations_closed=motion['obligations_closed'],
        departures=projected,timing_holding_seconds=timing['holding_seconds'])
    return out,audit


def prepare_day(date, *, service_hours=18):
    policy=load_policy(); policy=copy.deepcopy(policy); policy['demo']['reserves']=[]
    data,roster,_,profiles,meta=build_operational_day(policy,count=33,start_at=f'{date}T06:00:00+03:00',service_hours=service_hours)
    for station in data['stations']:
        if station['kind']=='depot': station['turnaround_seconds']=180
    inactive=[r for r in data['trains'] if r['location'].startswith('DEPOT_')]
    for i,train in enumerate(inactive):
        depot='Avtovo' if i%2==0 else 'Northern'
        train['location']=f'DEPOT_{depot.upper()}'
        row=next(r for r in roster if r['train_id']==train['train_id'])
        row.update(state='reserve',location=dict(kind='depot',depot=depot),ready_at=data['period_start'],readiness_status='assumed')
    data,audit=rebuild_timetable(data,0,meta['service_end_second'],initial=True)
    metadata=dict(service_end_second=meta['service_end_second'],initial_count=33,cycle_seconds=meta['cycle_seconds'],
        reference_date_verified=False,reference='data.zip: Парность и макс кол-во составов.docx, image3.jpeg',
        reference_counts=reference_counts(),projection=audit,readiness='controlled scenario: all 68 assigned trains assumed ready; crew and maintenance unknown',
        initial_passengers='empty at 06:00; 07:00 controller begins after common observed warmup',
        baseline_status='source-inspired feasible scenario, not verified operator timetable',actual_simulated_line_id=1)
    metadata['depot_reversal_seconds']=dict(value=180,status='assumed for repeated depot re-entry; no measured depot reversal supplied')
    return data,roster,profiles,metadata
