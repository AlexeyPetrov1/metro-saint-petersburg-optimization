"""Short synthetic running-train stops with saved real queues and honest gates."""

import argparse
import copy
import csv
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

from goal43_policy import minimum_headway
from simulation import integer, simulate


def running_stop(data,second,minutes,*,train_id=None,deadline=float('inf')):
    """Extend one committed leg; preserve order and propagate conservative holds.

    This section model has no signalling blocks or braking-distance geometry.
    Followers hold on approaches and cannot overtake the stopped train.
    """
    integer(second,'incident second');integer(minutes,'stop minutes',1)
    def check():
        if monotonic()>=deadline: raise TimeoutError('incident propagation deadline')
    check();out=copy.deepcopy(data);policy=out['headway_policy']
    rows=out['leg_timing'];tracks={t['track_id']:t for t in out['tracks']}
    trains={t['train_id']:t for t in out['trains']};stations={s['station_id']:s for s in out['stations']}
    trips={t['trip_id']:t for t in out['plans'][0]['trips']}
    origin=datetime.fromisoformat(out['period_start'])
    sections=[tracks[trains[r['train_id']]['location'] if r['trip_id']=='@initial'
        else trips[r['trip_id']]['route'][r['leg_index']]] for r in rows]
    old_d=[r['departure_second'] for r in rows]
    old_a=[d+t['travel_seconds']+r.get('approach_holding_seconds',0) for d,t,r in zip(old_d,sections,rows)]
    moving=[i for i,r in enumerate(rows) if old_d[i]<second<old_a[i]
        and (train_id is None or r['train_id']==train_id)
        and stations[sections[i]['from_station']]['kind']=='line' and stations[sections[i]['to_station']]['kind']=='line']
    if not moving: raise ValueError('incident must refer to a train actually moving between line stations')
    selected=min(moving,key=lambda i:(sections[i]['to_station']!='L1_17',not sections[i]['track_id'].endswith('_down'),old_a[i],rows[i]['train_id']))
    count=len(rows);lower={2*i:d for i,d in enumerate(old_d)}
    lower.update({2*i+1:a for i,a in enumerate(old_a)})
    frozen={node:value for node,value in lower.items() if value<second}
    lower[2*selected+1]+=minutes*60
    edges=defaultdict(dict);incoming={node:0 for node in lower}
    def edge(before,after,delay):
        if after not in edges[before]: incoming[after]+=1
        edges[before][after]=max(edges[before].get(after,0),delay)
    by_train,by_track,by_platform=defaultdict(list),defaultdict(list),defaultdict(list)
    for i,(row,track) in enumerate(zip(rows,sections)):
        edge(2*i,2*i+1,old_a[i]-old_d[i])
        by_train[row['train_id']].append(i);by_track[track['track_id']].append(i)
        if stations[track['to_station']]['kind']=='line':
            direction=track['track_id'].rsplit('_',1)[-1]
            by_platform[track['to_station'],direction].append(i)
    next_leg={}
    for indexes in by_train.values():
        indexes.sort(key=lambda i:(old_d[i],i))
        for a,b in zip(indexes,indexes[1:]):
            edge(2*a+1,2*b,rows[a]['service_seconds']);next_leg[a]=b
    pairs=[]
    for indexes in by_track.values():
        indexes.sort(key=lambda i:(old_d[i],i))
        for a,b in zip(indexes,indexes[1:]): edge(2*a,2*b,113);pairs.append((a,b))
    for (station,direction),indexes in by_platform.items():
        indexes.sort(key=lambda i:(old_a[i],old_d[i],i))
        demo=policy.get('demo',{});capacity=demo.get('platform_capacity_by_station',{}).get(station,demo.get('platform_capacity_default',1))
        slots=[]
        for j,i in enumerate(indexes):
            if j: edge(2*indexes[j-1]+1,2*i+1,0)
            if len(slots)<capacity: slots.append(i);continue
            available=[s for s,before in enumerate(slots) if before in next_leg]
            if not available: raise ValueError('incident propagation has no platform clearance')
            slot=min(available,key=lambda s:old_d[next_leg[slots[s]]])
            edge(2*next_leg[slots[slot]],2*i+1,1);slots[slot]=i
    def propagate():
        check();degrees=dict(incoming);ready=[n for n,d in degrees.items() if not d];heapq.heapify(ready)
        times=dict(lower);seen=0
        while ready:
            node=heapq.heappop(ready);seen+=1
            if node in frozen and times[node]!=frozen[node]: raise ValueError('incident rewrites executed movement')
            for after,delay in edges[node].items():
                times[after]=max(times[after],times[node]+delay);degrees[after]-=1
                if not degrees[after]: heapq.heappush(ready,after)
        if seen!=2*count: raise ValueError('incident platform precedence is cyclic')
        return times
    for _ in range(2*count+1):
        times=propagate();changed=False
        for a,b in pairs:
            need=minimum_headway(origin+timedelta(seconds=times[2*b]),policy)
            if edges[2*a][2*b]<need: edge(2*a,2*b,need);changed=True
        if not changed: break
    else: raise ValueError('incident headway propagation did not converge')
    previous={}
    for tid,indexes in by_train.items():
        for i in indexes:
            row=rows[i];departure=times[2*i]
            earliest=times[2*previous[tid]+1]+rows[previous[tid]]['service_seconds'] if tid in previous else trains[tid]['ready_after_seconds']
            row.update(departure_second=departure,holding_seconds=0 if row['trip_id']=='@initial' else max(0,departure-earliest),
                approach_holding_seconds=times[2*i+1]-departure-sections[i]['travel_seconds'])
            previous[tid]=i
    check()
    return out,dict(train_id=rows[selected]['train_id'],trip_id=rows[selected]['trip_id'],leg_index=rows[selected]['leg_index'],
        track_id=sections[selected]['track_id'],station_id=sections[selected]['to_station'],detected_second=second,
        duration_minutes=minutes,original_arrival_second=old_a[selected],forced_arrival_second=times[2*selected+1],
        stopped_until_second=second+minutes*60,external_committed_movement_changed=True,
        delayed_legs=sum(times[2*i]!=old_d[i] or times[2*i+1]!=old_a[i] for i in range(count)),
        followers_no_overtaking=True,platform_policy='saved directional platform slots, one-second clearance',
        signalling_verified=False,blocked_track_all_trains_verified=False,
        assumptions='approach holding and ordered arrivals; block occupancy, braking distances and dispatcher permissions unknown')


def _read(path):
    raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def _save(path,value):
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    with path.open('xb') as stream: stream.write(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)


def runner_from_checked_motion(data,groups,profiles,motion):
    """Reuse the exact checked motion; passenger compilation still validates its input."""
    import passengers
    from passenger_groups import GroupedPassengerSimulation
    from unittest.mock import patch
    from simulation import validate_result
    validate_result(motion)
    def cached(value,plan):
        if value!=data or plan!=motion['plan_id']: raise ValueError('checked motion input mismatch')
        return motion
    with patch.object(passengers,'simulate',cached):
        return GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles,
            boarding_seed=7,boarding_capacity_per_car=150)


def run(output,*,source='data/processed/metro50_full_day_20261009_v4/reference',total_budget_seconds=270,
        minutes=(3,8,15),resume_reference=None):
    from metro44_catalog import physical_gate
    from metro48_incidents import process_incident
    from metro51_service import local_boarding_service,compare_waiting_budget
    from passenger_groups import GroupedPassengerSimulation

    began=monotonic();deadline=began+total_budget_seconds
    output=Path(output);output.mkdir(parents=True,exist_ok=False);source=Path(source)
    result=_read(source/'result.json');folder=source/result['run']['folder']
    paths=[folder/name for name in ('movement_input.json','groups.json.gz','profiles.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data,groups,profiles=map(_read,paths);second=8100;cohort_end=11700;followup_end=15300
    groups=[g for g in groups if g['turnstile_second']<cohort_end]
    if resume_reference:
        previous=Path(resume_reference);manifest=_read(previous/'result.json')
        if any(manifest['source_hashes'].get(p)!=h for p,h in hashes.items()):
            raise ValueError('reference inputs changed since the saved probe')
        motion_path=folder/'motion.json';motion=_read(motion_path)
        base=runner_from_checked_motion(data,groups,profiles,motion)
        reference=_read(previous/'reference_checkpoint.json.gz')
        base._validate_checkpoint(reference,followup_end)
        hashes[str(motion_path)]=hashlib.sha256(motion_path.read_bytes()).hexdigest()
        hashes[str(previous/'reference_checkpoint.json.gz')]=hashlib.sha256((previous/'reference_checkpoint.json.gz').read_bytes()).hexdigest()
    else:
        base=GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles,boarding_seed=7,boarding_capacity_per_car=150)
        reference=None
    checkpoint=base.run(end_second=second,deadline=deadline)['checkpoint']
    if reference is None:
        reference=base.run(start_second=second,end_second=followup_end,checkpoint=checkpoint,deadline=deadline)['checkpoint']
    origin=datetime.fromisoformat(data['period_start']);at=lambda s:(origin+timedelta(seconds=s)).isoformat()
    print(json.dumps(dict(stage='incident_reference_ready',elapsed_seconds=monotonic()-began)),flush=True)
    _save(output/'reference_checkpoint.json.gz',reference)
    _save(output/'warmup_checkpoint.json.gz',checkpoint)
    cases=[]
    requested=tuple(minutes)
    for duration in requested:
        folder=output/f'stop_{duration}';folder.mkdir();captured=[]
        incident=dict(incident_id=f'synthetic-running-stop-{duration}',severity='stop',
            estimated_end_at=at(second+duration*60),source='explicit synthetic stop, not measured telemetry')
        def calculate(known,stamp,receipt_deadline,cancelled):
            limit=min(deadline,receipt_deadline)
            changed,audit=running_stop(data,second,duration,deadline=limit)
            import metro44_catalog
            from unittest.mock import patch
            recorded=[]
            def checked(value,plan):
                motion=simulate(value,plan);recorded.append(motion);return motion
            with patch.object(metro44_catalog,'simulate',checked):
                gate=physical_gate(changed,second,cohort_end,deadline=limit)
            motion=recorded[0]
            prefix=[e for e in motion['events'] if e['second']<second]==[e for e in base.motion['events'] if e['second']<second]
            gate.pop('_transport_work',None)
            runner=runner_from_checked_motion(changed,groups,profiles,motion)
            fork=runner.branch_checkpoint(checkpoint,base)
            state=runner.run(start_second=second,end_second=followup_end,checkpoint=fork,deadline=limit)['checkpoint']
            local=local_boarding_service(runner,state,start_second=second,end_second=cohort_end)
            waiting=compare_waiting_budget(base,reference,runner,state,start_second=second,end_second=cohort_end)
            samples=[r for r in state['queue_samples'] if second<=r['second']<followup_end]
            maxima={}
            for row in samples:
                key=f"{row['station_id']}:{row['direction']}";maxima[key]=max(maxima.get(key,0),row['queued'])
            count=state['entered'];balance=count==state['completed']+state['queued']+state['onboard_count']+state['approaching']
            diagnostic=dict(incident=audit,executed_prefix_preserved=prefix,movement=gate,local_service=local,
                waiting_budget=waiting,conservation_passed=balance,max_station_direction_queue=max(maxima.values(),default=0),
                queues_by_station_direction=maxima,remaining_queued=state['queued'],remaining_onboard=state['onboard_count'],
                hard_admission_cap=150,passenger_recovery_complete=state['queued']==0 and state['onboard_count']==0,
                action='retain fleet; protect affected direction and propagate departure/approach holds',
                economic_withdrawals_suppressed=True,economic_gain_proven=False,cash_saving_rub=None,
                future_actual_used_for_evaluation=True,causal_operational_recommendation=False,
                physical_comparison_scope='forced incident is the new committed external state; original executed prefix checked separately')
            captured.append((diagnostic,changed,state,samples))
            passed=prefix and gate['movement_passed'] and local['passed'] is True and waiting['passed'] is True and balance
            return dict(validated=passed,status='checked_recovery' if passed else 'no_admissible_recovery_verified',
                action=diagnostic['action'] if passed else None,protective_diagnostic_prepared=True,
                local_service_passed=local['passed'],movement_passed=gate['movement_passed'],passengers_checked=True,
                signalling_verified=False,cash_saving_rub=None)
        ledger,receipt=process_incident([],incident,at(second),calculate,budget_seconds=min(480,max(.01,deadline-monotonic())))
        if captured:
            diagnostic,changed,state,samples=captured[0]
            _save(folder/'movement_input.json.gz',changed);_save(folder/'passenger_checkpoint.json.gz',state)
            with (folder/'queue_by_station.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['second','station_id','direction','queued']);writer.writeheader();writer.writerows(samples)
        else: diagnostic=dict(status='evaluation_timeout',cash_saving_rub=None)
        case=dict(duration_minutes=duration,receipt=receipt,ledger=ledger,diagnostics=diagnostic,
            operational_budget_seconds=480,synthetic=True,external_delivery_measured=False)
        _save(folder/'result.json',case);cases.append(case)
        print(json.dumps(dict(stage='incident_probe_done',minutes=duration,elapsed_seconds=receipt.get('elapsed_seconds'),
            status=receipt['status'],total_elapsed_seconds=monotonic()-began)),flush=True)
        if monotonic()>=deadline: break
    preserved=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    if not preserved: raise ValueError('incident probe changed protected inputs')
    summary=dict(cases=cases,cohort_start_second=second,cohort_end_second=cohort_end,observation_end_second=followup_end,
        actual_admission_cutoff_second=cohort_end,source_hashes=hashes,sources_preserved=preserved,
        elapsed_seconds=monotonic()-began,synthetic=True,causal_forecast_comparison_performed=False,
        requested_durations_minutes=list(requested),resume_reference_verified=bool(resume_reference),
        all_requested_cases_attempted=len(cases)==len(requested),
        all_requested_cases_completed=len(cases)==len(requested) and all(c['receipt']['status']!='evaluation_timeout' for c in cases),
        economic_gain_proven=False,cash_saving_rub=None)
    _save(output/'result.json',summary)
    lines=['# Остановка движущегося состава: короткие проверки','',
        'Синтетическая остановка начинается в 08:15. Приходы взяты из общего датасета до 09:15, пассажиры прослежены до 10:15. Экономическое снятие подавлено. Это проверка защитной реакции, без сравнения прогнозных моделей.', '',
        '| Остановка, мин | Расчёт, с | Движение | Локальная посадка | Максимальная очередь | Рекомендация |',
        '|---:|---:|---|---|---:|---|']
    for case in cases:
        d=case['diagnostics'];r=case['receipt']
        lines.append(f"| {case['duration_minutes']} | {r.get('elapsed_seconds',0):.3f} | {d.get('movement',{}).get('movement_passed')} | {d.get('local_service',{}).get('status')} | {d.get('max_station_direction_queue')} | {r['status']} |")
    lines+=['','Проверка движения проводится относительно нового состояния, принудительно изменённого происшествием. История до 08:15 дополнительно сравнивается с исходной. Последующие поезда не обгоняют остановившийся: удержания распространяются через подходы, платформы и интервалы.', '',
        'Сигнализация, блок-участки и тормозные расстояния не представлены. Поэтому физическая безопасность реального блокирования пути не подтверждена. Отказ по интервалу, вместимости или локальной посадке сохраняется; готовность диагностики не означает допустимую диспетчерскую рекомендацию.', '',
        'Время расчёта измерено от вызова обработчика происшествия до готового проверенного результата. Доставка внешнего уведомления диспетчеру не измерена. Денежная экономия не заявляется.']
    with (output/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--total-budget-seconds',type=float,default=270)
    parser.add_argument('--minutes',type=int,nargs='+',default=[3,8,15]);parser.add_argument('--resume-reference')
    run(**vars(parser.parse_args()))
