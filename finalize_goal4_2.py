"""Attach independently validated decisions and consistent boundary snapshots."""

import copy
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from economics import write_json
from metro_replay import ROOT, check
from replay_metrics import movement_snapshot
from simulation import simulate


def exact_snapshot(data,second):
    extended=copy.deepcopy(data)
    duration=int((datetime.fromisoformat(data['period_end'])-datetime.fromisoformat(data['period_start'])).total_seconds())
    if second>=duration:
        extended['period_end']=(datetime.fromisoformat(data['period_end'])+timedelta(hours=2)).isoformat()
    motion=simulate(extended,extended['plans'][0]['plan_id'])
    snapshot=movement_snapshot(extended,motion,second)
    first={(r['train_id'],r['trip_id']):r['departure_second'] for r in data.get('leg_timing',[]) if r['leg_index']==0}
    snapshot['future_plan']=[t for p in data['plans'] for t in p['trips']
        if first.get((t['train_id'],t['trip_id']),t['departure_second'])>=second]
    for train in snapshot['trains']:
        event=next((e for e in motion['events'] if e['train_id']==train['train_id']
            and e['event_type']=='departure' and e['second']>=second),None)
        train['pending_departure']=dict(second=event['second'],trip_id=event['trip_id'],track_id=event['location']) if event else None
    snapshot['snapshot_version']='movement-snapshot-v2'
    snapshot['unfinished_obligations']=[dict(train_id=t['train_id'],**t['unfinished_obligation'])
        for t in snapshot['trains'] if t['unfinished_obligation']]
    snapshot['scope']='before boundary events, matching the passenger checkpoint; full raw plan is saved separately'
    return snapshot


def run():
    output=ROOT/'data/processed/goal4_2'; proof=ROOT/'data/processed/goal4_2_verification/selected_validation'
    check(output)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    selected_manifest=read(proof/'build_manifest.json')
    assert selected_manifest['status']=='success'
    for name,value in selected_manifest['inputs'].items(): assert sha(Path(name))==value,name
    for name,value in selected_manifest['outputs'].items(): assert sha(proof/name)==value,name
    summaries=read(proof/'summary.json'); scope=read(proof/'scope.json')
    assert all(row['seeds']==20 for row in summaries)
    for folder in (output/'episodes').iterdir():
        if not folder.is_dir(): continue
        data=read(folder/'plan.json')
        for second,name in [(14400,'movement_10.json'),(21600,'movement_12.json')]:
            write_json(folder/name,exact_snapshot(data,second))
    initial=read(output/'episodes/D_mild/decision_7200.json')
    late=read(output/'episodes/D_mild/decision_13500.json')
    selected=next(row for row in late['candidate_table'] if row['candidate_id']==late['selection']['selected_candidate_id'])
    comparison=read(output/'counterfactual_comparison.json')
    effects=[]
    for scenario in ('S1','S2','S3'):
        keep=next(r for r in comparison if r['scenario']==scenario and r['action']['kind']=='KEEP')
        add=next(r for r in comparison if r['scenario']==scenario and r['action']['kind']=='ADD')
        if keep['status']!='evaluated' or add['status']!='evaluated':
            effects.append(dict(scenario=scenario,status=add['status'],reason=add.get('reason'))); continue
        effects.append(dict(scenario=scenario,status='conditional_actual_replay',actual_effect_by_station=add['action']['actual_effect_by_station'],
            mean_main_wait_delta_seconds=add['cohort']['mean_wait_seconds']-keep['cohort']['mean_wait_seconds'],
            exceeded_car_seconds_delta=add['metrics']['exceeded_car_seconds']-keep['metrics']['exceeded_car_seconds'],
            train_km_delta=float(add['resources']['train_km'])-float(keep['resources']['train_km']),
            cash_delta_RUB=None,normal_load_passed=add['normal_load_passed'],release_authorized=False))
    card=read(output/'result_card.json')
    card.update(target='turnstile_entries',cycle_requirement=None,
        initial_decision=dict(second=7200,finding=initial.get('committed_violation'),
            proposed_action=None,status=initial['selection']['status'],simulation_fallback='KEEP',
            effect='current committed overload cannot be removed by an instantaneous fleet change',cash_delta_RUB=None),
        late_decision=dict(second=13500,selected_candidate_id=scope['selected_candidate_id'],
            forecast_horizon_seconds=late['available_horizon_seconds'],forecast_peak=selected['worst_checked_load'],
            preliminary_seeds=selected['tested_seeds'],validation_seeds=scope['seeds'],validation=summaries,
            actual_validation_status='violation_observed' if any(r['horizon_violation_fraction'] for r in summaries) else 'no_observed_violation',
            all_actual_checks_passed=all(r['horizon_violation_fraction']==0 for r in summaries),
            proposed_action='KEEP',effect_vs_baseline='identical continuation; no applied fleet change',cash_delta_RUB=None),
        conditional_add=effects,
        conditions=['scenario OD and uncalibrated travel/stop times','actual readiness, crew and infrastructure unknown',
            'finite seed sets provide no safety guarantee','variable rates unknown; monetary optimum is not established'],release_authorized=False)
    write_json(output/'result_card.json',card)
    lines=['','Проверка выбранного решения 09:45 на 20 новых seed от одного снимка:',
        '',
        '| OD | Нарушения до конца горизонта | Пик, диапазон | Ожидание вошедших 09:45–10:00, с, диапазон |',
        '|---|---:|---:|---:|']
    for row in summaries:
        peak,wait=row['horizon_peak'],row['late_mean_wait']
        interval=f"{wait['minimum']:.2f}–{wait['maximum']:.2f}" if wait['minimum'] is not None else 'не завершено'
        lines.append(f"| {row['scenario']} | {row['horizon_violation_fraction']:.0%} | {peak['minimum']}–{peak['maximum']} | {interval} |")
    failed=[r for r in summaries if r['horizon_violation_fraction']]
    if failed:
        lines.extend(['','Выбранный по прогнозу KEEP не прошёл все финальные проверки по фактическому спросу: '+
            '; '.join(f"{r['scenario']}, {round(r['horizon_violation_fraction']*r['seeds'])} из {r['seeds']}, "
                f"максимум {r['horizon_peak']['maximum']} человек" for r in failed)+'.'])
    lines.extend(['','Выбор KEEP сделан по прогнозу до этой проверки. Фактические будущие входы используются только для оценки. '
        'В каждом прогоне сохранены одинаковые люди, очереди, подходы и выполняемые перемещения в 09:45; меняются будущие случайные остановки. '
        'KEEP совпадает с продолжением базы. Выбор не дал изменения выпуска или доказанной денежной экономии.',
        '', 'Снимки движения на 10:00 и 12:00 теперь сохраняются перед граничными событиями, как пассажирские checkpoint. '
        'Незавершённые текущие движения и ожидания сохраняются; фактические начала будущих рейсов берутся из leg_timing. '
        'motion.json отдельно содержит результат движкового учёта с включёнными событиями конца периода.',
        '', 'Подробные результаты поздней проверки: data/processed/goal4_2_verification/selected_validation/.'])
    lines.extend(['','Условное добавление R01 в 08:00 относительно KEEP, оценка по факту на seed 7:',
        '', '| OD | Изменение ожидания, с | Изменение перегрузки, вагоно-с | Дополнительные поездо-км | Первый эффект | Соблюдение 120 |',
        '|---|---:|---:|---:|---|---|'])
    for row in effects:
        if row['status']!='conditional_actual_replay': continue
        second=min(row['actual_effect_by_station'].values())
        when=f"{6+second//3600:02d}:{second%3600//60:02d}:{second%60:02d}"
        lines.append(f"| {row['scenario']} | {row['mean_main_wait_delta_seconds']:.2f} | {row['exceeded_car_seconds_delta']} | "
            f"{row['train_km_delta']:.3f} | {when} | {'да' if row['normal_load_passed'] else 'нет'} |")
    lines.extend(['','Это отдельное сравнение условного действия. Оно не применялось в основном эпизоде '
        'и не проверялось на 20 финальных seed. Отрицательная разница ожидания или перегрузки означает снижение показателя. '
        'Денежный эффект неизвестен; дополнительный пробег не подменяет расчёт энергии или затрат.'])
    summary=output/'summary.md'; content=summary.read_text(encoding='utf-8')
    marker='\nПроверка выбранного решения 09:45 на 20 новых seed от одного снимка:'
    content=content.split(marker)[0]
    summary.write_text(content.rstrip()+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    manifest=read(output/'build_manifest.json')
    inputs=[Path(__file__),ROOT/'selected_validation.py',proof/'build_manifest.json']+[proof/name for name in selected_manifest['outputs']]
    manifest['inputs'].update({str(path):sha(path) for path in inputs})
    manifest['outputs']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='build_manifest.json'}
    manifest['derived_stages']=['selected_decision_validation','consistent_boundary_snapshots','calculated_decision_cards']
    write_json(output/'build_manifest.json',manifest)
    print(json.dumps(check(output)))


if __name__=='__main__': run()
