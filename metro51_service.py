"""Unique local boarding misses and paired waiting budgets from saved ledgers."""

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from math import isfinite

from metro50_peak_service import COUNTS, MOSCOW


def _limit(value, name, maximum=None):
    if type(value) not in (int,float) or not isfinite(value) or value<0 or maximum is not None and value>maximum:
        raise ValueError(f'{name} must be a finite nonnegative number'+(' <= 1' if maximum else ''))


def _window(runner,state,start,end):
    observed=state['second'];end=observed if end is None else end
    if any(type(v) is not int for v in (start,end,observed)) or not 0<=start<=end<=observed<=runner.duration:
        raise ValueError('invalid cohort or observation window')
    origin=datetime.fromisoformat(runner.motion['period_start'])
    if origin.tzinfo is None: raise ValueError('movement period requires a timezone')
    return origin.astimezone(MOSCOW),end


def _merge(rows):
    result={key:sum(r[key] for r in rows) for key in COUNTS}
    for key in ('max_missed_trains','max_boarded_wait_seconds','max_current_wait_seconds'):
        result[key]=max((r[key] for r in rows),default=0)
    count=result['passenger_count']
    result.update(missed_first_share=result['missed_first_count']/count if count else None,
                  missed_second_share=result['missed_second_count']/count if count else None,
                  denominator='all unique passengers in the selected platform-arrival cohort')
    return result


def local_boarding_service(runner,state,*,start_second=0,end_second=None,first_limit=.05,second_limit=.01):
    """Bind every station, direction and 15-minute platform-arrival cohort, all day.

    Both fractions use the entire unique cohort, including small groups. Unobserved
    first/second opportunities and partial quarters remain incomplete. Later trains
    up to state.second provide followup; repeated denied attempts are not people.
    """
    _limit(first_limit,'first limit',1);_limit(second_limit,'second limit',1)
    origin,end=_window(runner,state,start_second,end_second)
    selected={g['group_id'] for g in runner.groups if start_second<=g['platform_second']<end and g['count']}
    schedules,departures=defaultdict(list),{}
    for index,event in enumerate(runner.stream[:state['next_event_index']]):
        if event.get('event_type')!='departure' or not event.get('passenger_service'): continue
        track=runner.tracks[event['location']]
        direction=1 if runner.positions[track['to_station']]>runner.positions[track['from_station']] else -1
        destinations={runner.tracks[t]['to_station'] for t in event['remaining_route']}
        departures[event['event_id']]=(index,event,destinations)
        for destination in destinations:
            schedules[track['from_station'],direction,destination].append((event['second'],index,event['event_id']))
    original,remaining,opportunities={}, {}, {}
    for gid in selected:
        group=runner.by_id[gid]
        for dest in group['destinations']:
            key=gid,dest['station_id'];original[key]=remaining[key]=dest['count']
            direction=1 if runner.positions[dest['station_id']]>runner.positions[group['origin']] else -1
            schedule=schedules[group['origin'],direction,dest['station_id']]
            first=bisect_left(schedule,(group['platform_second'],-1,''))
            opportunities[key]=(schedule,first)
    ranks,max_wait,legacy_cursor=defaultdict(Counter),Counter(),defaultdict(int)
    for row in state['board_events']:
        gid=row['group_id']
        if gid not in selected: continue
        group=runner.by_id[gid];info=departures.get(row['motion_event_id'])
        if info is None: raise ValueError('boarding ledger references an unobserved passenger departure')
        position,event,destinations=info
        destination=row.get('destination')
        if destination is None:
            lower=legacy_cursor[gid,row['motion_event_id']]
            destination=next((d for d in sorted(destinations,key=runner.positions.get)
                if runner.positions[d]>=lower and remaining.get((gid,d),0)),None)
        key=gid,destination;count=row['count']
        if (destination not in destinations or group['origin']!=runner.tracks[event['location']]['from_station']
                or type(count) is not int or count<=0 or count>remaining.get(key,0)
                or row['second']!=event['second'] or row['second']<group['platform_second']):
            raise ValueError('boarding ledger contradicts OD balance or movement')
        legacy_cursor[gid,row['motion_event_id']]=runner.positions[destination]+1
        schedule,first=opportunities[key];rank=bisect_left(schedule,(row['second'],position,event['event_id']))-first
        if rank<0: raise ValueError('boarding precedes first eligible train')
        remaining[key]-=count;ranks[key][rank]+=count
        max_wait[key]=max(max_wait[key],row['second']-group['platform_second'])
    queued=Counter()
    for queue in state['queues'].values():
        for row in queue:
            if row['group_id'] in selected: queued[row['group_id'],row['destination']]+=row['count']
    if any(queued[k]!=n for k,n in remaining.items()) or any(k not in remaining for k in queued):
        raise ValueError('boarding/queue OD balance mismatch')
    details=defaultdict(list)
    for key,count in original.items():
        group=runner.by_id[key[0]];platform=group['platform_second'];at=origin+timedelta(seconds=platform)
        direction=1 if runner.positions[key[1]]>runner.positions[group['origin']] else -1
        schedule,first=opportunities[key];available=len(schedule)-first;boarded=ranks[key]
        missed_first=count-boarded[0] if available else 0
        missed_second=count-boarded[0]-boarded[1] if available>=2 else 0
        row=dict(passenger_count=count,first_opportunity_count=count if available else 0,
            first_opportunity_censored_count=count if not available else 0,missed_first_count=missed_first,
            boarded_second_count=boarded[1],second_opportunity_count=missed_first if available>=2 else 0,
            second_opportunity_censored_count=missed_first if available==1 else 0,
            missed_second_count=missed_second,missed_opportunity_count=sum(r*n for r,n in boarded.items())+remaining[key]*available,
            still_queued_count=remaining[key],max_missed_trains=max([available if remaining[key] else 0]+list(boarded)),
            max_boarded_wait_seconds=max_wait[key],max_current_wait_seconds=state['second']-platform if remaining[key] else 0)
        quarter=at.replace(minute=at.minute//15*15,second=0,microsecond=0)
        details[quarter,group['origin'],direction].append(row)
    cohorts=[]
    for (quarter,station,direction),rows in sorted(details.items()):
        summary=_merge(rows);count=summary['passenger_count']
        temporal=origin+timedelta(seconds=start_second)<=quarter and quarter+timedelta(minutes=15)<=origin+timedelta(seconds=end)
        complete=temporal and not (summary['first_opportunity_censored_count'] or summary['second_opportunity_censored_count'])
        checks=dict(first_limit_passed=Decimal(summary['missed_first_count'])<=Decimal(str(first_limit))*count,
                    second_limit_passed=Decimal(summary['missed_second_count'])<=Decimal(str(second_limit))*count,
                    observation_complete=complete)
        status='failed' if not checks['first_limit_passed'] or not checks['second_limit_passed'] else 'passed' if complete else 'incomplete'
        cohorts.append(dict(interval_start=quarter.isoformat(),interval_end=(quarter+timedelta(minutes=15)).isoformat(),
            station_id=station,direction=direction,**summary,checks=checks,status=status,
            temporal_coverage_complete=temporal,shares_are_lower_bounds_when_censored=not complete))
    metrics=_merge(cohorts)
    checks={key:all(r['checks'][key] for r in cohorts) for key in ('first_limit_passed','second_limit_passed','observation_complete')}
    statuses={r['status'] for r in cohorts}
    status='failed' if 'failed' in statuses else 'incomplete' if 'incomplete' in statuses else 'passed' if cohorts else 'not_applicable'
    return dict(version='unique-local-first-second-train-v1',status=status,
        passed=False if status=='failed' else None if status=='incomplete' else True,checks=checks,metrics=metrics,cohorts=cohorts,
        worst={name:max(cohorts,key=lambda r:r[field] or 0) if cohorts else None
               for name,field in [('first','missed_first_share'),('second','missed_second_share')]},
        first_limit=first_limit,second_limit=second_limit,cohort_start_second=start_second,cohort_end_second=end,
        observation_end_second=state['second'],failed_cohort_count=sum(r['status']=='failed' for r in cohorts),
        binding_scope='each station x direction x 15-minute platform-arrival cohort, all service periods',
        capacity_denied_attempts_used=False,voluntary_skip_modelled=False)


def _waiting(runner,state,start,end):
    selected={g['group_id']:g['count'] for g in runner.groups if start<=g['platform_second']<end}
    histogram=Counter();by_group=Counter()
    for row in state['board_events']:
        gid=row['group_id']
        if gid not in selected: continue
        platform=runner.by_id[gid]['platform_second'];second=row['second'];count=row['count']
        if (type(second) is not int or type(count) is not int or count<=0
                or not platform<=second<state['second'] or by_group[gid]+count>selected[gid]):
            raise ValueError('boarding ledger contradicts the waiting cohort')
        by_group[gid]+=count;histogram[second-platform]+=count
    count=sum(selected.values());boarded=sum(histogram.values())
    if boarded>count: raise ValueError('boarding ledger duplicates the waiting cohort')
    total=sum(w*n for w,n in histogram.items());complete=boarded==count and state['second']>=end
    def quantile(percent):
        rank=(percent*boarded+99)//100;seen=0
        for wait,n in sorted(histogram.items()):
            seen+=n
            if seen>=rank: return wait
    return dict(passenger_count=count,boarded=boarded,unboarded=count-boarded,complete=complete,
        full_boarded_wait_passenger_seconds=total,
        mean_wait_seconds=total/boarded if complete and boarded else None,
        p95_wait_seconds=quantile(95) if complete and boarded else None,
        p99_wait_seconds=quantile(99) if complete and boarded else None,
        maximum_boarded_wait_seconds=max(histogram,default=None))


def compare_waiting_budget(reference_runner,reference_state,candidate_runner,candidate_state,*,
                          start_second=0,end_second=None,mean_allowance=10,p95_allowance=40):
    """Compare equal platform-arrival cohorts, with saved later followup for both plans."""
    _limit(mean_allowance,'mean allowance');_limit(p95_allowance,'p95 allowance')
    end=min(reference_state['second'],candidate_state['second']) if end_second is None else end_second
    origin,end=_window(reference_runner,reference_state,start_second,end)
    other,_=_window(candidate_runner,candidate_state,start_second,end)
    identity=lambda runner:sorted((g['group_id'],g['origin'],g['platform_second'],g['count'],
        tuple(sorted((d['station_id'],d['count']) for d in g['destinations']))) for g in runner.groups
        if start_second<=g['platform_second']<end)
    if origin!=other or identity(reference_runner)!=identity(candidate_runner):
        raise ValueError('waiting comparison requires identical passenger cohorts and period origin')
    windows=[('whole_period',start_second,end)]
    day=origin.replace(hour=0,minute=0,second=0,microsecond=0)
    while day<origin+timedelta(seconds=end):
        if day.weekday()<5:
            for name,hours in [('morning_peak',(8,10)),('evening_peak',(17,19))]:
                a,b=(int((day+timedelta(hours=h)-origin).total_seconds()) for h in hours)
                left,right=max(start_second,a),min(end,b)
                if left<right: windows.append((name,left,right))
        day+=timedelta(days=1)
    rows=[]
    for name,left,right in windows:
        before,after=_waiting(reference_runner,reference_state,left,right),_waiting(candidate_runner,candidate_state,left,right)
        complete=before['complete'] and after['complete'];count=before['passenger_count']
        difference=after['full_boarded_wait_passenger_seconds']-before['full_boarded_wait_passenger_seconds']
        mean=difference/count if complete and count else None
        p95=after['p95_wait_seconds']-before['p95_wait_seconds'] if complete and count else None
        checks=dict(mean_budget_passed=mean is None if not count else complete and Decimal(difference)<=Decimal(str(mean_allowance))*count,
                    p95_budget_passed=p95 is None if not count else p95 is not None and p95<=p95_allowance)
        status='incomplete' if not complete else 'not_applicable' if not count else 'passed' if all(checks.values()) else 'failed'
        rows.append(dict(name=name,start_second=left,end_second=right,reference=before,candidate=after,
            mean_delta_seconds=mean,p95_delta_seconds=p95,checks=checks,status=status))
    statuses={r['status'] for r in rows}
    status='failed' if 'failed' in statuses else 'incomplete' if 'incomplete' in statuses else 'passed' if 'passed' in statuses else 'not_applicable'
    return dict(version='paired-platform-wait-budget-v1',status=status,
        passed=False if status=='failed' else None if status=='incomplete' else True,windows=rows,
        mean_allowance_seconds=mean_allowance,p95_allowance_seconds=p95_allowance,
        scope='identical platform-arrival cohorts for requested whole period and overlapping weekday peaks; later boarding observed')
