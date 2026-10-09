"""Retime known vehicle chains, including their explicit future depot releases."""

import copy
from collections import defaultdict
from datetime import datetime,timedelta

from dwell import timing_key
from goal43_policy import parameter
from metro44_assignment_plan import _current_movements_preserved,_deadline,terminal_resources
from metro44_planner import headway_seconds,trip_finish
from metro44_terminal_timing import compile_terminal_timing
from rolling import operational_headways
from simulation import digest,simulate


def _entry_obligations(data,as_of,readiness,preparation_started):
    tracks={t['track_id']:t for t in data['tracks']};stations={s['station_id']:s for s in data['stations']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    actual=lambda t:timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second']
    origin=datetime.fromisoformat(data['period_start']);registry={r['train_id']:r for r in readiness}
    source=data['plans'][0]['trips'];links={};committed=set();begun=[]
    for train in data['trains']:
        owned=sorted((t for t in source if t['train_id']==train['train_id']),key=lambda t:actual(t))
        for index,entry in enumerate(owned):
            if not entry['route'][0].startswith('ENTRY_'):continue
            first=next((t for t in owned[index+1:]if t['passenger']),None)
            if first is None:raise ValueError('entry has no linked passenger service')
            depot=tracks[entry['route'][0]]['from_station']
            if any(t['route'][0].startswith('EXIT_')for t in owned[:index]) and not stations[depot]['turnaround_seconds']:
                raise ValueError('zero turnaround depot reentry is forbidden')
            release=parameter(data['headway_policy'],'depot_release_s')['Avtovo'if'AVTOVO'in depot else'Northern']
            lead=sum(tracks[t]['travel_seconds']for t in entry['route'])+stations[tracks[entry['route'][-1]]['to_station']]['turnaround_seconds']
            row=dict(entry_trip_id=entry['trip_id'],first_service_trip_id=first['trip_id'],train_id=train['train_id'],
                release_seconds=release,entry_lead_seconds=lead,entry_original_second=actual(entry))
            if actual(entry)<as_of:
                committed.add(first['trip_id']);begun.append(entry['trip_id']);row['begun']=True
            else:
                ready=registry.get(train['train_id'],{})
                if not ready.get('ready_at') or ready.get('readiness_status') in (None,'unknown'):
                    raise ValueError(f'explicit reserve readiness required for {train["train_id"]}')
                technical=int((datetime.fromisoformat(ready['ready_at'])-origin).total_seconds())
                prior_returns=[t for t in owned[:index]if t['route'][0].startswith('EXIT_')]
                if prior_returns:
                    technical=max(technical,trip_finish(prior_returns[-1],timing,tracks,stations)[1])
                prepared=preparation_started.get(entry['trip_id'])
                if prepared is not None and (isinstance(prepared,bool)or not isinstance(prepared,int)or not max(0,technical)<=prepared<=as_of):
                    raise ValueError('documented preparation must begin after readiness and by as_of')
                start=max(as_of,technical)if prepared is None else prepared
                row.update(begun=False,ready_second=technical,readiness_status=ready['readiness_status'],
                    preparation_start_second=start,release_ready_second=start+release)
            links[first['trip_id']]=row
    if set(preparation_started)-{r['entry_trip_id']for r in links.values()if not r['begun']}:
        raise ValueError('documented preparation refers to no future entry')
    return links,committed,begun


def build_reserve_bridge(data,as_of,service_end,*,headway=None,bridge_seconds=480,
                         readiness=(),preparation_started=None,deadline=None):
    """Construct bounds from movement and readiness only; do not read passengers."""
    _deadline(deadline)
    if isinstance(bridge_seconds,bool)or not isinstance(bridge_seconds,int)or not 0<=bridge_seconds<=480:
        raise ValueError('bridge must be an integer between 0 and 480 seconds')
    if not 0<=as_of<service_end:raise ValueError('invalid service control interval')
    preparations=preparation_started or {}
    links,committed,begun=_entry_obligations(data,as_of,readiness,preparations)
    out=copy.deepcopy(data);origin=datetime.fromisoformat(data['period_start']);policy=data['headway_policy']
    tracks={t['track_id']:t for t in data['tracks']};stations={s['station_id']:s for s in data['stations']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    actual=lambda t:timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second']
    source=data['plans'][0]['trips'];bridge_end=min(as_of+bridge_seconds,service_end)
    kept=[];targets={};bridge=[];removed=[];last={};resolved=set();future=[]
    for trip in sorted((t for t in source if t['passenger']),key=lambda t:(actual(t),t['trip_id'])):
        _deadline(deadline);at=actual(trip);terminal=tracks[trip['route'][0]]['from_station'];target=at
        fixed=at<bridge_end or trip['trip_id']in committed
        row=links.get(trip['trip_id'])
        if not fixed:
            h=headway_seconds(headway,origin+timedelta(seconds=at),terminal,policy,origin);resolved.add(h)
            target=max(at,last.get(terminal,as_of-h)+h)
        if row and not row['begun']:
            target=max(target,row['release_ready_second'],row['entry_original_second']+row['entry_lead_seconds'])
            if fixed and target!=at:raise ValueError('retained bridge start precedes explicit depot release readiness')
        if not fixed and target>=service_end:removed.append(trip['trip_id']);continue
        copied=copy.deepcopy(trip)
        if not fixed:copied['departure_second']=target
        kept.append(copied);targets[trip['trip_id']]=target;last[terminal]=target
        if as_of<=at<bridge_end:bridge.append(trip['trip_id'])
        if row and not row['begun']:
            entry=next(t for t in source if t['trip_id']==row['entry_trip_id']);copy_entry=copy.deepcopy(entry)
            copy_entry['departure_second']=target-row['entry_lead_seconds']
            kept.append(copy_entry);targets[entry['trip_id']]=copy_entry['departure_second']
            future.append({**row,'first_service_not_before':target,'entry_not_before':copy_entry['departure_second']})
    kept += [copy.deepcopy(t)for t in source if not t['passenger']and actual(t)<as_of]
    kept_ids={t['trip_id']for t in kept}
    for entry in future:
        prior=[t for t in source if t['train_id']==entry['train_id']and t['route'][0].startswith('EXIT_')
               and as_of<=actual(t)<entry['entry_original_second']]
        if prior:
            raise ValueError('future depot reentry requires a separate verified return-and-readiness chain')
    out['plans'][0]['trips']=kept
    out['leg_timing']=[copy.deepcopy(r)for r in data['leg_timing']if r['trip_id']=='@initial'or r['trip_id']in kept_ids]
    bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in out['leg_timing']if r['departure_second']>=0}
    for trip in kept:
        if trip['trip_id']in targets:bounds[timing_key(trip['train_id'],trip['trip_id'],0)]=targets[trip['trip_id']]
    original_ids={t['trip_id']for t in source}
    for train in data['trains']:
        _deadline(deadline)
        owned=sorted((t for t in kept if t['train_id']==train['train_id']),key=lambda t:actual(t))
        if not owned:continue
        final=owned[-1];terminal,ready=trip_finish(final,timing,tracks,stations)
        if stations[terminal]['kind']=='depot':continue
        departure=max(as_of,ready+targets.get(final['trip_id'],actual(final))-actual(final))
        exit_track=next(t['track_id']for t in data['tracks']if t['from_station']==terminal and t['track_id'].startswith('EXIT_'))
        identity=f'G44BRIDGE_RESERVE:{as_of}:{train["train_id"]}:EXIT'
        if identity in original_ids:raise ValueError('reserve bridge exit identity collision')
        out['plans'][0]['trips'].append(dict(trip_id=identity,train_id=train['train_id'],departure_second=departure,route=[exit_track],passenger=False))
        bounds[timing_key(train['train_id'],identity,0)]=departure
    audit=dict(committed_until_second=as_of,bridge_seconds=bridge_seconds,bridge_end_second=bridge_end,
        retained_bridge_service_ids=sorted(bridge),retained_begun_entry_ids=sorted(begun),
        retained_begun_entry_first_service_ids=sorted(committed),removed_service_ids=sorted(removed),
        future_entry_obligations=future,documented_preparations=dict(preparations),
        first_departure_bounds=bounds,profile_resolved_headways=sorted(resolved),allocation_changed=False,
        unused_reserves_added=0,wear_priority=False,phase_seconds=0,
        constructor='known train service chains and existing future entries; short service bridge',
        readiness_scope='explicit input readiness may be assumed; crew and maintenance clearance are not inferred',
        timing_holding_scope='source future leg departures remain lower bounds; existing future ENTRY linked to first SERVICE')
    _deadline(deadline);return out,audit


def standby_repair_bounds(data,motion,as_of):
    """Delay unstarted approaches until the preceding ready train has left."""
    terminals=data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    spans={t:[]for t in terminals};opened={}
    for train in data.get('trains',[]):
        if train['state']=='standby'and train['location']in terminals:
            opened[train['train_id']]=(train['location'],0,None)
    for event in motion['events']:
        tid=event['train_id']
        if tid in opened:
            terminal,start,identity=opened.pop(tid)
            if start<event['second']:spans[terminal].append((start,event['second'],tid,identity))
        if event['next_state']=='standby'and event['location']in terminals:
            opened[tid]=(event['location'],event['second'],event.get('trip_id'))
    for tid,(terminal,start,identity)in opened.items():
        spans[terminal].append((start,motion['duration_seconds'],tid,identity))
    trips={t['trip_id']:t for t in data['plans'][0]['trips']}
    tracks={t['track_id']:t for t in data['tracks']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    bounds={}
    for terminal,rows in spans.items():
        preceding_end=-1
        for start,end,tid,identity in sorted(rows):
            if start<preceding_end:
                if identity not in trips:raise ValueError('standby repair cannot change an initial terminal train')
                trip=trips[identity];index=len(trip['route'])-1;key=timing_key(tid,identity,index);leg=timing[key]
                if leg['departure_second']<as_of:raise ValueError('standby repair would rewrite a started approach')
                bound=preceding_end-tracks[trip['route'][index]]['travel_seconds']-leg['service_seconds']
                if bound<=leg['departure_second']:raise ValueError('standby repair requires unsupported arrival holding')
                bounds[key]=max(bounds.get(key,0),bound)
            preceding_end=max(preceding_end,end)
    return bounds


def max_gap_repair_bounds(data,motion,as_of,violations):
    """Delay a previous unstarted SERVICE without breaking its preceding gap."""
    maximum=parameter(data['headway_policy'],'headway_max_s')
    trips={t['trip_id']:t for t in data['plans'][0]['trips']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    events={};by_track=defaultdict(list);shifts={}
    for event in motion['events']:
        if event['event_type']=='departure':
            events[event['location'],event['second']]=event;by_track[event['location']].append(event)
    for gap in violations:
        if gap['seconds']<=maximum:continue
        if gap['kind']not in('completed_interval','boundary_end'):
            raise ValueError('maximum gap repair has no unstarted previous SERVICE')
        event=events.get((gap['track_id'],gap['second']-gap['seconds']))
        trip=trips.get(event['trip_id'])if event else None
        if trip is None or not trip['passenger']:raise ValueError('maximum gap repair requires a known passenger SERVICE')
        first=timing[timing_key(trip['train_id'],trip['trip_id'],0)]['departure_second']
        if first<as_of:raise ValueError('maximum gap repair cannot change a started SERVICE')
        shifts[trip['trip_id']]=max(shifts.get(trip['trip_id'],0),gap['seconds']-maximum)
    for rows in by_track.values():
        rows.sort(key=lambda e:e['second'])
        for left,right in zip(rows,rows[1:]):
            if right['trip_id']not in shifts:continue
            gap=right['second']+shifts[right['trip_id']]-left['second']-shifts.get(left['trip_id'],0)
            if gap>maximum:raise ValueError('maximum gap repair would violate the preceding headway')
    return {timing_key(trips[t]['train_id'],t,0):timing[timing_key(trips[t]['train_id'],t,0)]['departure_second']+delta
            for t,delta in shifts.items()}


def rebuild_reserve_bridge_timetable(data,as_of,service_end,*,headway=None,bridge_seconds=480,
                                     readiness=(),preparation_started=None,deadline=None,standby_repairs=0,max_gap_repairs=0):
    for limit in(standby_repairs,max_gap_repairs):
        if isinstance(limit,bool)or not isinstance(limit,int)or not 0<=limit<=5:
            raise ValueError('repair count must be between zero and five')
    candidate,audit=build_reserve_bridge(data,as_of,service_end,headway=headway,bridge_seconds=bridge_seconds,
        readiness=readiness,preparation_started=preparation_started,deadline=deadline)
    changed,timing=compile_terminal_timing(candidate,candidate['plans'][0]['plan_id'],'fixed',7,
        policy=data['headway_policy'],as_of=as_of,departure_not_before=audit.pop('first_departure_bounds'))
    _deadline(deadline)
    repairs=[];gap_repairs=[]
    def recompile(extra):
        nonlocal changed,timing
        bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in changed['leg_timing']if r['departure_second']>=0}
        maximum_delay=max((extra[k]-bounds[k]for k in extra),default=0)
        for key,value in extra.items():bounds[key]=max(bounds[key],value)
        _deadline(deadline)
        changed,timing=compile_terminal_timing(changed,changed['plans'][0]['plan_id'],'fixed',7,
            policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
        _deadline(deadline)
        return maximum_delay
    def repair_parking():
        for _ in range(standby_repairs):
            motion=simulate(changed,changed['plans'][0]['plan_id']);resources=terminal_resources(changed,motion)
            if not resources['parking_violations']:break
            extra=standby_repair_bounds(changed,motion,as_of)
            if not extra:break
            repairs.append(dict(inbound_bounds_count=len(extra),maximum_delay=recompile(extra)))
    repair_parking()
    for _ in range(max_gap_repairs):
        motion=simulate(changed,changed['plans'][0]['plan_id']);gaps=operational_headways(changed,motion,as_of,service_end)
        if gaps['passed']:break
        extra=max_gap_repair_bounds(changed,motion,as_of,gaps['violations'])
        if not extra:break
        gap_repairs.append(dict(service_bounds_count=len(extra),maximum_delay=recompile(extra),preceding_gaps_checked=True))
        repair_parking()
    old={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    new={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in changed['leg_timing']}
    trips={t['trip_id']:t for t in data['plans'][0]['trips']}
    retained=set(audit['retained_bridge_service_ids']+audit['retained_begun_entry_first_service_ids'])
    for identity in retained:
        trip=trips[identity];key=timing_key(trip['train_id'],identity,0)
        if new[key]['departure_second']!=old[key]['departure_second']:
            raise ValueError('compiler changes a retained bridge or begun-entry SERVICE departure')
        if identity in audit['retained_begun_entry_first_service_ids']:
            for index in range(len(trip['route'])):
                leg=timing_key(trip['train_id'],identity,index)
                if new[leg]!=old[leg]:raise ValueError('compiler changes the first service of an already begun ENTRY')
    for row in audit['future_entry_obligations']:
        service=new[timing_key(row['train_id'],row['first_service_trip_id'],0)]['departure_second']
        entry=new[timing_key(row['train_id'],row['entry_trip_id'],0)]['departure_second']
        if service<row['release_ready_second'] or service<entry+row['entry_lead_seconds']:
            raise ValueError('compiled reserve release precedes explicit readiness')
        row.update(actual_first_service_second=service,actual_entry_second=entry)
    before=simulate(data,data['plans'][0]['plan_id']);motion=simulate(changed,changed['plans'][0]['plan_id'])
    if [e for e in before['events']if e['second']<as_of]!=[e for e in motion['events']if e['second']<as_of]:
        raise ValueError('reserve bridge changes executed movement')
    if not _current_movements_preserved(data,changed,as_of):raise ValueError('reserve bridge changes a started arrival')
    resources=terminal_resources(changed,motion);gaps=operational_headways(changed,motion,as_of,service_end)
    audit.update(movement_passed=gaps['passed']and motion['obligations_closed']and resources['passed'],
        headways_passed=gaps['passed'],minimum_headway=gaps['minimum_seconds'],maximum_headway=gaps['maximum_seconds'],
        headway_violation_count=len(gaps['violations']),headway_violations=gaps['violations'][:10],
        terminal_resources=resources,terminal_timing=timing,peak_trains_on_line=motion['checks']['peak_trains_on_line'],
        obligations_closed=motion['obligations_closed'],executed_prefix_preserved=True,current_movements_preserved=True,
        bridge_departures_preserved=True,reserve_release_readiness_passed=True,release_authorized=False,passenger_checked=False,
        standby_repairs=repairs,max_gap_repairs=gap_repairs,
        _data_digest=digest(changed),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    _deadline(deadline);return changed,audit
