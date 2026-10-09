"""Keep a short known service bridge when changing the dispatch regime."""

import copy
from datetime import datetime, timedelta
from time import monotonic

from dwell import timing_key
from metro44_assignment_plan import _current_movements_preserved, _deadline
from metro44_planner import headway_seconds, trip_finish
from metro44_planner_terminal import retime_terminal_plan
from metro44_terminal_timing import compile_terminal_timing
from simulation import simulate


SOURCE_PLANNER_SHA256='d8c53e8f248b47b71ec552b76be9e10676b0034a04d35698c6af48c990bdef18'




def _retime_known_jobs(data,as_of,service_end,bridge_seconds,headway,deadline,requested_wear=False):
    _deadline(deadline)
    out=copy.deepcopy(data);policy=data['headway_policy'];origin=datetime.fromisoformat(data['period_start'])
    tracks={t['track_id']:t for t in data['tracks']};stations={s['station_id']:s for s in data['stations']}
    timing={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    actual=lambda t:timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second']
    source=data['plans'][0]['trips'];bridge_end=min(as_of+bridge_seconds,service_end)
    if any(t['route'][0].startswith('ENTRY_') and actual(t)>=as_of for t in source):
        raise ValueError('fixed-chain bridge does not release future depot trains')
    last={};kept=[];targets={};bridge=[];removed=[]
    for t in sorted((t for t in source if t['passenger']),key=lambda t:(actual(t),t['trip_id'])):
        _deadline(deadline);at=actual(t);terminal=tracks[t['route'][0]]['from_station']
        if at>=bridge_end:
            h=headway_seconds(headway,origin+timedelta(seconds=at),terminal,policy,origin)
            target=max(at,last.get(terminal,as_of-h)+h)
        else:target=at
        if at>=as_of and target>=service_end:removed.append(t['trip_id']);continue
        copied=copy.deepcopy(t)
        if at>=bridge_end:copied['departure_second']=target
        kept.append(copied);targets[t['trip_id']]=target;last[terminal]=target
        if as_of<=at<bridge_end:bridge.append(t['trip_id'])
    kept += [copy.deepcopy(t)for t in source if not t['passenger']and actual(t)<as_of]
    ids={t['trip_id']for t in kept};out['plans'][0]['trips']=kept
    out['leg_timing']=[copy.deepcopy(r)for r in data['leg_timing']if r['trip_id']=='@initial' or r['trip_id']in ids]
    bounds={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']for r in out['leg_timing']if r['departure_second']>=0}
    for t in kept:
        if t['passenger']:bounds[timing_key(t['train_id'],t['trip_id'],0)]=targets[t['trip_id']]
    original_ids={t['trip_id']for t in source}
    for train in data['trains']:
        owned=sorted((t for t in kept if t['train_id']==train['train_id']),key=lambda t:actual(t))
        if not owned:continue
        final=owned[-1];terminal,ready=trip_finish(final,timing,tracks,stations)
        if stations[terminal]['kind']=='depot':continue
        departure=max(as_of,ready+targets.get(final['trip_id'],actual(final))-actual(final))
        exit_track=next(t['track_id']for t in data['tracks']if t['from_station']==terminal and t['track_id'].startswith('EXIT_'))
        identity=f'G44BRIDGE:{as_of}:{train["train_id"]}:EXIT'
        if identity in original_ids:raise ValueError('bridge exit identity collision')
        trip=dict(trip_id=identity,train_id=train['train_id'],departure_second=departure,route=[exit_track],passenger=False)
        out['plans'][0]['trips'].append(trip);bounds[timing_key(train['train_id'],identity,0)]=departure
    changed,_=compile_terminal_timing(out,out['plans'][0]['plan_id'],'fixed',7,policy=policy,
        as_of=as_of,departure_not_before=bounds)
    _deadline(deadline)
    audit=dict(committed_until_second=as_of,bridge_seconds=bridge_seconds,bridge_end_second=bridge_end,
        retained_bridge_service_ids=sorted(bridge),retained_bridge_entry_ids=[],removed_service_ids=sorted(removed),
        passenger_trips=sum(t['passenger']for t in changed['plans'][0]['trips']),
        future_assignment_policy='original train service chains retained; only starts outside explicit service end removed',
        timing_holding_scope='exact compiler preserves known future departure lower bounds; no new depot releases',
        requested_wear_priority=requested_wear,wear_priority=False,unused_reserves_added=0,
        phase_seconds=0,planning_occupancy_margin=0)
    return changed,audit


def rebuild_bridge_timetable(data,as_of,service_end,*,bridge_seconds=480,deadline=None,**kwargs):
    """Preserve nearby known starts, then check the complete coupled terminal plan."""
    _deadline(deadline)
    if isinstance(bridge_seconds,bool) or not isinstance(bridge_seconds,int) or not 0<=bridge_seconds<=480:
        raise ValueError('bridge must be an integer between 0 and 480 seconds')
    if kwargs.get('initial') and bridge_seconds:raise ValueError('service bridge requires an existing timetable')
    if kwargs.get('phase_seconds',0)!=0:raise ValueError('fixed-chain bridge preserves the source phase')
    candidate,original=_retime_known_jobs(data,as_of,service_end,bridge_seconds,kwargs.get('headway',160),deadline,
        kwargs.get('wear_priority',False))
    changed,audit=retime_terminal_plan(candidate,as_of,service_end,deadline=deadline)
    old={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    new={timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in changed['leg_timing']}
    trips={t['trip_id']:t for t in data['plans'][0]['trips']}
    for identity in original['retained_bridge_service_ids']:
        trip=trips[identity];key=timing_key(trip['train_id'],identity,0)
        if new[key]['departure_second']!=old[key]['departure_second']:
            raise ValueError('compiler changes a retained bridge departure')
    before=simulate(data,data['plans'][0]['plan_id']);after=simulate(changed,changed['plans'][0]['plan_id'])
    if [e for e in before['events'] if e['second']<as_of]!=[e for e in after['events'] if e['second']<as_of]:
        raise ValueError('bridge changes executed movement')
    if not _current_movements_preserved(data,changed,as_of):raise ValueError('bridge changes a started arrival')
    result={**original,**audit,'bridge_departures_preserved':True,
        'constructor':'known service chains retimed after a short bridge; coupled terminal compilation',
        'bridge_scope':'retained near-term SERVICE starts; reserve release remains anchored at actual as_of',
        'source_planner_sha256':SOURCE_PLANNER_SHA256}
    _deadline(deadline);return changed,result
