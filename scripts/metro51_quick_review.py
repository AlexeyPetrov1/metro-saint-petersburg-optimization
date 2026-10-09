"""Read a saved 08:00-08:15 cohort while live pilots continue unchanged."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime,timedelta
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from metro51_service import _waiting


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def selected_boarding(groups,state,start,end):
    chosen={g['group_id']:g for g in groups if start<=g['platform_second']<end}
    original={(gid,d['station_id']):d['count'] for gid,g in chosen.items() for d in g['destinations']}
    boarded={key:0 for key in original};rows=[]
    for row in state['board_events']:
        gid=row['group_id']
        if gid not in chosen:continue
        key=gid,row['destination'];count=row['count'];second=row['second']
        if key not in original or type(count) is not int or count<=0 or not chosen[gid]['platform_second']<=second<state['second']:
            raise ValueError('invalid selected boarding ledger')
        boarded[key]+=count
        if boarded[key]>original[key]:raise ValueError('duplicate unique passengers in selected ledger')
        rows.append((gid,row['destination'],second,row['motion_event_id'],count))
    runner=SimpleNamespace(groups=groups,by_id={g['group_id']:g for g in groups})
    wait=_waiting(runner,state,start,end)
    identity=sorted((gid,g['origin'],g['platform_second'],g['count'],sorted((d['station_id'],d['count']) for d in g['destinations'])) for gid,g in chosen.items())
    count=sum(original.values());total=sum(boarded.values())
    return dict(passenger_count=count,boarded=total,unboarded=count-total,
        complete=state['second']>=end and boarded==original,observation_end_second=state['second'],
        late_boarding_count=sum(r[4] for r in rows if r[2]>=end),
        last_boarding_second=max((r[2] for r in rows),default=None),waiting=wait,
        group_sha256=digest(identity),board_ledger_sha256=digest(sorted(rows)),board_ledger_rows=len(rows))


def movement_signature(data,start,end):
    tracks={t['track_id']:t for t in data['tracks']};trains={t['train_id']:t for t in data['trains']}
    trips={t['trip_id']:t for t in data['plans'][0]['trips']};rows=[]
    for row in data['leg_timing']:
        track=tracks[trains[row['train_id']]['location'] if row['trip_id']=='@initial' else trips[row['trip_id']]['route'][row['leg_index']]]
        d=row['departure_second'];a=d+track['travel_seconds']+row.get('approach_holding_seconds',0)
        if start<=d<end or start<=a<end:
            rows.append((row['train_id'],row['trip_id'],row['leg_index'],track['track_id'],track['length_km'],d,a,row['service_seconds']))
    return digest(sorted(rows))


def load_signature(state,start,end):
    return digest(sorted((r['second'],r['train_id'],r['car_id'],r['load'],r['event_type'])
        for r in state['load_events'] if start<=r['second']<end))


def run(root):
    began=monotonic();root=Path(root);inputs={};start,end=7200,8100
    def read(path):
        raw=path.read_bytes();inputs[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
    reference=read(root/'reference/result.json');folder=root/'reference'/reference['run']['folder']
    data=read(folder/'movement_input.json');groups=read(folder/'groups.json.gz');state=read(folder/'passenger_checkpoint.json.gz')
    before=selected_boarding(groups,state,start,end);movement=movement_signature(data,start,end);loads=load_signature(state,start,end)
    at=(datetime.fromisoformat(data['period_start'])+timedelta(seconds=start)).isoformat()
    cohorts=[r for r in reference['metrics']['local_boarding_service']['cohorts'] if r['interval_start']==at]
    failed=[r for r in cohorts if r['status']=='failed'];samples=[r['load'] for r in state['load_events'] if start<=r['second']<end]
    rows=[dict(arm='reference',**before,board_ledger_identical=True,movement_identical=True,load_ledger_identical=True,
        local_status='failed' if failed else 'passed',local_failed_cohorts=len(failed),observed_resource_saving_percent=0)]
    for arm in ('old','ensemble_old','balanced'):
        paths=sorted((root/arm/'controller').glob('resume_*.json.gz'))
        if not paths:rows.append(dict(arm=arm,status='no_saved_checkpoint'));continue
        path=paths[-1];snapshot=read(path);common=read(root/arm/'common_scenario.json.gz')
        row=selected_boarding(common['groups'],snapshot['state'],start,end)
        group_equal=row['group_sha256']==before['group_sha256']
        boards=group_equal and row['board_ledger_sha256']==before['board_ledger_sha256']
        move=movement_signature(snapshot['data'],start,end)==movement
        load=load_signature(snapshot['state'],start,end)==loads
        selected=[]
        for p in sorted((root/arm/'controller').glob('decision_*.json')):
            if int(p.stem.split('_')[-1])>=9900:continue
            decision=read(p);selected.append(decision.get('applied_candidate_id'))
        rows.append(dict(arm=arm,**row,checkpoint=str(path),group_identity_equal=group_equal,
            board_ledger_identical=boards,movement_identical=move,load_ledger_identical=load,
            all_decisions_before_0845_keep=bool(selected) and all(s=='KEEP' for s in selected),
            local_status='incomplete' if not row['complete'] else ('failed' if failed else 'passed') if boards and move else 'not_recomputed_ledger_differs',
            local_failed_cohorts=len(failed) if boards and move and row['complete'] else None,
            observed_resource_saving_percent=0 if move else None))
    def quantile(percent):
        values=sorted(samples);return values[(percent*len(values)+99)//100-1] if values else None
    result=dict(interval_start=at,interval_end=(datetime.fromisoformat(at)+timedelta(minutes=15)).isoformat(),
        arms=rows,reference_local_cohorts=cohorts,reference_failed_local_cohorts=failed,
        reference_car_load_event_snapshots=dict(maximum=max(samples,default=None),p95=quantile(95),p99=quantile(99),
            observations=len(samples),weighting='one record per car arrival/departure, not time-weighted'),
        requirement_one_percent_saving_verified=False,full_pilot_completed=False,
        simulation_or_model_started=False,running_processes_modified=False,source_hashes=inputs,
        sources_preserved=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items()),
        elapsed_seconds=monotonic()-began,
        scope='unique platform arrivals in this quarter; all later recorded boardings through each saved checkpoint; historical comparison only')
    output=root/'quick_15m';output.mkdir(exist_ok=False)
    with (output/'report.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    fields=['arm','observation_end_second','passenger_count','boarded','unboarded','complete','late_boarding_count',
        'board_ledger_identical','movement_identical','load_ledger_identical','local_status','observed_resource_saving_percent']
    with (output/'arms.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fields=['station_id','direction','passenger_count','missed_first_count','missed_first_share','missed_second_count','missed_second_share','status']
    with (output/'local_cohorts.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(cohorts)
    lines=['# Сохранённый срез 08:00–08:15','',
        'Процессы продолжаются. Новая симуляция или модель не запускались. Пассажиры выбраны по времени прихода на платформу; поздние посадки включены до времени каждого checkpoint.', '',
        '| Вариант | Посадка наблюдена полностью | Журнал посадки совпал | Движение совпало | Локальная посадка | Экономия ресурсов в этом срезе |',
        '|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['arm']} | {r.get('complete')} | {r.get('board_ledger_identical')} | {r.get('movement_identical')} | {r.get('local_status',r.get('status'))} | {r.get('observed_resource_saving_percent')}% |")
    lines+=['','Нарушения исходного графика:', '', '| Станция | Направление | Людей | Первая непосадка | Повторная непосадка |','|---|---:|---:|---:|---:|']
    for r in failed:lines.append(f"| {r['station_id']} | {r['direction']} | {r['passenger_count']} | {r['missed_first_count']} / {100*r['missed_first_share']:.3f}% | {r['missed_second_count']} / {100*r['missed_second_share']:.3f}% |")
    lines+=['',f"Вагонная загрузка по событиям: максимум {max(samples,default=None)}, P95 {quantile(95)}, P99 {quantile(99)} человек. Это распределение записей прибытия/отправления, а не доля времени вагонов.", '',
        'Цель ≥1% экономии этим срезом не подтверждена. Совпадение движения подтверждает нулевое изменение пробега в выбранном окне; будущую экономию или годовой эффект из этого выводить нельзя. Локальные непосадки не исправлены фактом работы процессов. При неполной посадке ожидание остаётся неизвестным, а локальный статус — incomplete.', '',
        'Совпадение проверено по SHA-256 выбранных уникальных OD-групп, фактических строк посадки, движения и вагонных нагрузок. Показатели локальной посадки reference перенесены только при полном совпадении групп, журнала посадки и движения.']
    with (output/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'],cohort_people=before['passenger_count'],
        failed_local_cohorts=len(failed),arms=[dict(arm=r['arm'],complete=r.get('complete'),same=r.get('board_ledger_identical')) for r in rows]),ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    run(**vars(parser.parse_args()))
