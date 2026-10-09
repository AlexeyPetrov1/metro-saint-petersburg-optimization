"""Reproducible balanced ensemble pilot with explicit journals and checkpoints."""

import argparse
import copy
import gzip
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

from metro50_daily import ROOT, SOURCE, read, closed_tail, admission_forecast


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode('utf-8')
    if path.suffix == '.gz':
        raw = gzip.compress(raw, mtime=0)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def control_times(end, *, start_second=3600):
    if type(start_second) is not int or not 900 <= start_second < end or start_second % 900:
        raise ValueError('control start must be a quarter boundary inside the service window')
    return list(range(start_second,end,900))


def filtered_groups(groups,end):
    return [copy.deepcopy(g) for g in groups if g['turnstile_second']<end]


def pilot_contract(arm,stage,seed=7,*,control_start_second=3600):
    if arm not in ('REFERENCE','OLD','ENSEMBLE_OLD','BALANCED') or stage not in ('pilot','full_day','evening'):
        raise ValueError('unsupported arm or stage')
    end = {'pilot':16200,'full_day':64800,'evening':48600}[stage]
    control_times(end,start_second=control_start_second)
    return dict(arm=arm,stage=stage,seed=seed,full_day=stage=='full_day',
        control_start_second=control_start_second,
        cohort_end_second=end, platform_cohort_end_second=end+900,
        observation_end_second=min(72000,end+7200),
        service_end_second=65400, line_id=1, fixed_fleet_size=68,
        first_miss_limit=.05,second_miss_limit=.01,mean_wait_allowance=10,p95_wait_allowance=40,
        boarding_capacity_per_car=150,comfort_per_car=120,
        cash_saving_rub=None,annual_repeat_days=250,annual_repeat_is_scenario=True,
        real_operating_schedule=False,control_clock_changes_allowed=True)


def resume_matches(saved,current):
    return all(saved.get(key)==current.get(key) for key in ('arm','stage','seed','source_hashes','control_start_second'))


def journal(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()


def actual_per_train(data,motion,end):
    from collections import defaultdict
    from decimal import Decimal
    tracks={t['track_id']:t for t in data['tracks']}
    trains={t['train_id']:t for t in data['trains']}
    current={tid:t['location'] for tid,t in trains.items() if t['state']=='moving'}
    distance=defaultdict(Decimal)
    for event in motion['events']:
        if event['second']>end:
            break
        tid=event['train_id']
        if event['event_type']=='departure':
            current[tid]=event['location']
        elif event['event_type']=='arrival' and event['second']>0:
            distance[tid]+=Decimal(str(tracks[current[tid]]['length_km']))
    return [dict(train_id=tid,train_type=t['train_type'],km=float(distance[tid]))
            for tid,t in trains.items()]


def explanation_packet(run_id,decision,raw,state):
    from metro51_explanations import build_fact_packet
    selected=decision.get('applied_candidate_id','KEEP')
    at=decision['decision_time']
    facts=[dict(id='state.queue',label='Сейчас ожидают на платформах',value=state['queued'],unit='чел.',
                source=f'controller/resume_{state["second"]:05}.json.gz',kind='observed'),
           dict(id='forecast.total',label='Прогноз входов на доступном горизонте',
                value=sum(r['predicted_count'] for r in raw['rows']),unit='чел.',
                source=f'controller/raw_forecast_{state["second"]:05}.json',kind='forecast'),
           dict(id='decision.status',label='Результат проверки вариантов',value=decision['selection']['status'],unit='',
                source=f'controller/decision_{state["second"]:05}.json',kind='rationale'),
           dict(id='decision.validated',label='Выбран допустимый по прогнозу вариант',
                value=decision['selection'].get('selected_candidate_id') is not None,unit='',
                source=f'controller/decision_{state["second"]:05}.json',kind='rationale'),
           dict(id='decision.revisit',label='Следующий плановый пересмотр',
                value=(datetime.fromisoformat(at)+timedelta(minutes=15)).isoformat(),unit='',source=at,kind='revisit'),
           dict(id='risk.forecast',label='Прогноз может недооценить локальный спрос',value=True,
                unit='',source='forecast contract',kind='risk'),
           dict(id='limit.attribution',label='Вклад отдельных признаков ансамбля ещё не измерен',value=True,
                unit='',source='model contract',kind='limitation')]
    for row in decision.get('candidate_table',[]):
        ident=row['candidate_id']
        trials=row.get('trials',[])
        details={k:row.get(k) for k in ('status','physical_passed','policy_passed','service_passed','reason')}
        details['waiting']=[dict(scenario=t.get('scenario'),**{k:t.get('mature_cohort',{}).get(k)
            for k in ('mean_wait_seconds','p95_wait_seconds','complete')}) for t in trials]
        details['local_checks']=[{k:t.get('peak_boarding_service',{}).get(k)
            for k in ('status','failed_cohort_count','binding_scope')} for t in trials]
        if row.get('committed_per_train'):
            from metro50_economics import maintenance_rates
            rates=maintenance_rates()
            details['known_closed_plan_cost']={side:sum(t['km']*rates[t['train_type']][side]
                for t in row['committed_per_train']) for side in ('without_sr_kr','with_sr_kr')}
        facts.append(dict(id='alternative.'+ident,label='Проверенный вариант '+ident,
            value=details,
            unit='',source=f'controller/decision_{state["second"]:05}.json',kind='alternative'))
    if raw.get('rows'):
        by_station={}
        for row in raw['rows']:
            by_station[row['station_id']]=by_station.get(row['station_id'],0)+row['predicted_count']
        facts.append(dict(id='forecast.stations',label='Прогноз входов по каждой станции',value=by_station,
            unit='чел. на горизонте',source=f'controller/raw_forecast_{state["second"]:05}.json',kind='forecast'))
    contribution=raw.get('source_versions',{}).get('feature_evidence')
    if contribution:
        facts.append(dict(id='attribution.b1',label='Измеренные вклады признаков LightGBM; не причинные эффекты',
            value=contribution['rows'],method=contribution['method'],unit='log-ratio z',
            source=f'controller/raw_forecast_{state["second"]:05}.json',kind='attribution'))
        facts=[f for f in facts if f['id']!='limit.attribution']
        facts.append(dict(id='limit.attribution',label='Вклад GRU в признаки не измерен; B1 contributions не равны вкладам всего стекинга',
            value=True,unit='',source='model contract',kind='limitation'))
    proposal=next((a.get('proposal',{}) for a in reversed(decision.get('joint_search',[]))),{})
    applied=proposal if selected=='COST_SOLVER' else {}
    for ident,value,label in [('headway.peak',113,'Минимальный интервал в пик'),
                             ('headway.maximum',480,'Максимальный интервал'),
                             ('load.maximum',150,'Предел пассажиров на вагон')]:
        facts.append(dict(id=ident,label=label,value=value,unit='с' if ident.startswith('headway') else 'чел.',
                          source='contract.json',kind='constraint'))
    return build_fact_packet(run_id=run_id,decision_id=str(state['second']),
        action=dict(candidate_id=selected,decision_time=at,
                    withdrawn_train_ids=applied.get('selected_train_ids',[]),
                    return_by_second=applied.get('return_by_second'),
                    trip_shifts={k:v for k,v in applied.get('trip_shifts',{}).items() if v},
                    forecast_admissible=decision['selection'].get('selected_candidate_id') is not None),facts=facts)


def run_arm(output,*,arm='BALANCED',stage='pilot',seed=7,resume=False,max_attempts=2,control_start_second=3600):
    from threading import Event
    from goal43_replay import save_run
    from metro44_catalog import physical_gate
    from metro44_focus import execution_prefix
    from metro44_run import public_decision,selected_plan
    from metro51_reliability import safe_recommendation
    from metro47_catalog import calculate_cost_catalog
    from metro47_cost_policy import choose_by_cost
    from metro48_forecasts import complete_tail
    from metro48_gru import FrozenQuarterGRU
    from metro48_temporary import build_temporary_candidate
    from metro48_temporary_solver import solve_temporary_withdrawals
    from metro50_economics import maintenance_rates
    from metro50_peak_service import peak_boarding_service,apply_peak_safety_margin
    from metro51_service import local_boarding_service
    from metro51_optimizer import joint_proposal,build_joint_candidate,choose_balanced
    from metro51_explanations import explain_decision,load_env
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest
    from withdrawal_stabling import attach_stabling
    import pandas as pd

    began=monotonic();output=Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError(output)
    load_env()
    contract=pilot_contract(arm,stage,seed,control_start_second=control_start_second)
    data=attach_stabling(read(SOURCE/'movement_input.json'))
    groups=filtered_groups(read(SOURCE/'groups.json.gz'),contract['cohort_end_second'])
    profiles=read(SOURCE/'profiles.json')
    roster=[dict(train_id=t['train_id'],type=t['train_type']) for t in data['trains']]
    frame_path=ROOT/'data/processed/metro51_inputs_20261009/source_frame.parquet'
    model=None
    if arm=='OLD':
        model=FrozenQuarterGRU()
    elif arm!='REFERENCE':
        from metro51_ensemble import FrozenQuarterEnsemble
        model=FrozenQuarterEnsemble()
    model_contract=model.contract() if model and hasattr(model,'contract') else None
    source_paths=[p for p in ROOT.glob('*.py') if p.name not in ('metro51_report.py','metro51_forecast_audit.py')]
    source_paths += [SOURCE/n for n in ('movement_input.json','groups.json.gz','profiles.json')]
    if arm!='REFERENCE':
        source_paths.append(frame_path)
    manifest=ROOT/'data/processed/metro51_inputs_20261009/input_manifest.json'
    if manifest.exists():
        source_paths.append(manifest)
    contract.update(period_start=data['period_start'],period_end=data['period_end'],
        model_contract=model_contract,groups_digest=digest(groups),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        data_identical_to_previous_upload=True,planning_horizon_seconds=14400,
        admission_reserve='frozen stack q90 once; no extra25percent' if arm=='BALANCED' else 'legacy120car engineering1.25',
        cost_scope='actual train arrivals through observation_end; pilot is not a daily/annual saving')
    if resume:
        saved=read(output/'contract.json')
        if not resume_matches(saved,contract):
            raise ValueError('resume source, model, rule or seed mismatch')
        if (output/'result.json').exists():
            return read(output/'result.json')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'contract.json',contract)
        atomic_json(output/'common_scenario.json.gz',dict(data=data,groups=groups,profiles=profiles,roster=roster))
        snapshot=output/'execution_sources';snapshot.mkdir()
        for p in ROOT.glob('*.py'):
            (snapshot/p.name).write_bytes(p.read_bytes())
    options=dict(boarding_seed=seed,boarding_capacity_per_car=150)
    runner=GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles,**options)
    reference_data=copy.deepcopy(data)
    reference_runner=GroupedPassengerSimulation(reference_data,reference_data['plans'][0]['plan_id'],groups,profiles,**options)
    reference_state=None;reference_previous=0
    previous=0;state=None;decisions=[]
    if resume:
        paths=sorted((output/'controller').glob('resume_*.json.gz')) if (output/'controller').exists() else []
        if paths:
            saved=read(paths[-1]);data,state,previous=saved['data'],saved['state'],saved['second']
            runner=GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles,**options)
            runner._validate_checkpoint(state,previous)
            decisions=[read(p) for p in sorted((output/'controller').glob('decision_*.json'))
                       if int(p.stem.split('_')[-1])<=previous]
    frame=pd.read_parquet(frame_path) if model else None
    ids=[s['station_id'] for s in data['stations'] if s['kind']=='line']
    origin=datetime.fromisoformat(data['period_start'])
    rates={k:v['without_sr_kr'] for k,v in maintenance_rates().items()}
    times=control_times(contract['cohort_end_second'],start_second=control_start_second) if model else []
    if stage=='evening':
        times=[s for s in times if s>=36000]
    for index,second in enumerate(times):
        if second<=previous:
            continue
        state=runner.run(start_second=previous,end_second=second,
                         **({'checkpoint':state} if state is not None else {}))['checkpoint']
        at=(origin+timedelta(seconds=second)).isoformat()
        prefix=execution_prefix(runner.motion,second)
        reference_state=reference_runner.run(start_second=reference_previous,end_second=second,
            **({'checkpoint':reference_state} if reference_state is not None else {}))['checkpoint']
        reference_previous=second
        journal(output/'events.jsonl',dict(stage='decision_started',arm=arm,second=second,at=at,
                                         queued=state['queued'],onboard=state['onboard_count']))
        def pipeline(deadline,cancelled):
            raw=model.forecast_bundle(frame,ids,at)
            if hasattr(model,'feature_evidence'):
                raw['source_versions']['feature_evidence']=model.feature_evidence(frame,at,max_features=5)
            boundary=(origin+timedelta(seconds=min(second+14400,64800))).isoformat()
            bounded=complete_tail(raw,frame,ids,at,boundary)
            bounded=closed_tail(bounded,(origin+timedelta(seconds=min(second+14400,runner.duration))).isoformat())
            bounded=admission_forecast(bounded,(origin+timedelta(seconds=64800)).isoformat())
            if arm=='BALANCED':
                quantiles={(r['station_id'],r['interval_start']):r['q90']
                           for r in raw.get('source_versions',{}).get('quantile_rows',[])}
                for row in bounded['rows']:
                    row['predicted_count']=max(row['predicted_count'],quantiles.get(
                        (row['station_id'],row['interval_start']),row['predicted_count']))
                bounded['source_versions']['balanced_reserve']=dict(method='frozen q90 once; no duplicated125percent',
                    native_quantile_row_count=len(quantiles),
                    historical_tail_quantile_calibrated=False,uncovered_station_quantile_calibrated=False)
            else:
                bounded=apply_peak_safety_margin(bounded,at,scenario='current5')
            back=min(second+13800,64800)
            def service(trial,out,start,end):
                if arm=='BALANCED':
                    return local_boarding_service(trial,out,start_second=start,end_second=end//900*900)
                return peak_boarding_service(trial,out,start_second=start,end_second=end//900*900,missed_share_limit=.05)
            history=[];cuts=[];limit=None;answer=None
            reference_trials=None
            if arm=='BALANCED' and digest(data)!=digest(reference_data):
                counterfactual=calculate_cost_catalog(reference_data,reference_runner,reference_state,profiles,
                    roster,bounded,second,65400,deadline,cancelled,passenger_service_check=service,
                    forecast_horizon_seconds=14400)
                reference_trials=next((r['trials'] for r in counterfactual['decision']['candidate_table']
                                       if r['candidate_id']=='KEEP' and r.get('trials')),None)
                if reference_trials is None:
                    raise ValueError('fixed reference forecast waiting budget unavailable')
            for attempt in range(max_attempts):
                if arm=='BALANCED':
                    solved=joint_proposal(data,bounded,state,second,65400,return_by_second=back,
                        maintenance_rates=rates,deadline=deadline,no_good_cuts=cuts,max_withdrawals=limit)
                    proposal=dict(selected_train_ids=solved['selected_train_ids'],return_by_second=back,
                                  trip_shifts=solved.get('trip_shifts',{}))
                    builder=build_joint_candidate
                    policy=lambda rows,**kw:choose_balanced(rows,maintenance_rates=rates,
                                                          reference_trials=reference_trials,**kw)
                else:
                    solved=solve_temporary_withdrawals(data,bounded,state,second,65400,return_by_second=back,
                        maintenance_rates=rates,deadline=deadline,time_limit_seconds=20,
                        peak_missed_share_limit=.05,no_good_cuts=cuts,max_withdrawals=limit)
                    proposal=dict(selected_train_ids=solved['selected_train_ids'],return_by_second=back)
                    builder=build_temporary_candidate
                    policy=lambda rows,**kw:choose_by_cost(rows,maintenance_rates=rates,**kw)
                changed=bool(proposal['selected_train_ids'] or any(proposal.get('trip_shifts',{}).values()))
                answer=calculate_cost_catalog(data,runner,state,profiles,roster,bounded,second,65400,deadline,cancelled,
                    proposal=proposal if changed else None,builder=builder,passenger_service_check=service,
                    forecast_horizon_seconds=14400,selection_policy=policy)
                history.append(dict(attempt=attempt+1,solver=solved,proposal=proposal,
                    candidate_statuses=[{k:r.get(k) for k in ('candidate_id','status','reason','service_passed')}
                                        for r in answer['decision']['candidate_table']]))
                if not changed or answer['decision']['selection'].get('selected_candidate_id')=='COST_SOLVER':
                    break
                selected=proposal['selected_train_ids']
                if selected:
                    cuts.append(tuple(sorted(selected)));limit=max(0,len(selected)//2)
                else:
                    break
            answer.update(raw_forecast=raw,bounded_forecast=bounded)
            answer['decision']['joint_search']=history
            answer['decision']['operational_global_optimum_proven']=False
            answer['decision']['fixed_reference_wait_budget_used']=arm=='BALANCED'
            if cuts:
                answer['decision']['search_complete']=False
                answer['decision']['selection']['search_complete']=False
            return answer
        receipt=safe_recommendation(pipeline,480)
        if not receipt['completed']:
            decision=dict(decision_second=second,decision_time=at,candidate_table=[],
                selection=receipt['selection'],applied_candidate_id='KEEP',pipeline_seconds=receipt['elapsed_seconds'],
                fallback_service_verified=False,timeout=receipt['selection']['status']=='evaluation_timeout',
                error_code=receipt.get('error_code'))
            raw={'rows':[]}
            journal(output/'events.jsonl',dict(stage=receipt['selection']['status'],second=second,
                                             error_code=receipt.get('error_code')))
        else:
            answer=receipt['result'];decision=answer['decision'];raw=answer['raw_forecast']
            # Only the new checked selection is applied; a descriptive mitigation is not dispatch authorization.
            safe_selection=dict(decision['selection'],mitigation_candidate_id=None)
            changed,applied=selected_plan(safe_selection,answer['candidates'],decision['candidate_table'])
            if changed is not data:
                replacement=GroupedPassengerSimulation(changed,changed['plans'][0]['plan_id'],groups,profiles,**options)
                if execution_prefix(replacement.motion,second)!=prefix:
                    raise ValueError('executed prefix changed')
                state=replacement.branch_checkpoint(state,runner);runner,data=replacement,changed
            decision.update(applied_candidate_id=applied,pipeline_seconds=receipt['elapsed_seconds'],
                executed_prefix_digest=prefix,local_recommendation_within_480=True,external_delivery_measured=False)
            atomic_json(output/'controller'/f'forecast_{second:05}.json',answer['bounded_forecast'])
            atomic_json(output/'controller'/f'raw_forecast_{second:05}.json',raw)
        public=public_decision(decision);decisions.append(public)
        atomic_json(output/'controller'/f'decision_{second:05}.json',public)
        atomic_json(output/'controller'/f'resume_{second:05}.json.gz',dict(second=second,data=data,state=state))
        packet=explanation_packet(output.name,public,raw,state)
        atomic_json(output/'explanations'/f'facts_{second:05}.json',packet)
        # Evidence templates are immediate; offline API worker enriches persisted facts independently.
        atomic_json(output/'explanations'/f'explanation_{second:05}.json',explain_decision(packet,use_api=False))
        journal(output/'events.jsonl',dict(stage='decision_completed',arm=arm,index=index+1,count=len(times),
            second=second,applied=decision['applied_candidate_id'],status=decision['selection']['status'],
            seconds=receipt['elapsed_seconds'],checkpoint=f'controller/resume_{second:05}.json.gz'))
        print(json.dumps(dict(stage='control',arm=arm,index=index+1,count=len(times),at=at,
                             applied=decision['applied_candidate_id'],seconds=receipt['elapsed_seconds']),ensure_ascii=False),flush=True)
        previous=second
    state=runner.run(start_second=previous,end_second=contract['observation_end_second'],
                     **({'checkpoint':state} if state is not None else {}))['checkpoint']
    runner._validate_checkpoint(state,state['second'])
    physical=physical_gate(data,0,contract['service_end_second'],reference_motion=runner.motion)
    local=local_boarding_service(runner,state,end_second=contract['platform_cohort_end_second'])
    from metro_ml_compare import _metrics
    metrics=_metrics(data,runner,state,roster,data['headway_policy'],0,contract['cohort_end_second'])
    per_train=actual_per_train(data,runner.motion,state['second'])
    metrics.update(local_boarding_service=local,physical_gate={k:v for k,v in physical.items() if not k.startswith('_')},
        actual_per_train=per_train,train_km=sum(r['km'] for r in per_train),
        mean_km_per_fixed_train=sum(r['km'] for r in per_train)/68,
        complete_passenger_accounting=state['queued']==state['approaching']==state['onboard_count']==0
            and state['entered']==state['completed'],cost_observation_end_second=state['second'])
    metrics['actual_service_admissible']=bool(physical['movement_passed'] and local['passed'] is True
        and metrics['complete_passenger_accounting'] and metrics['actual_service_admissible'])
    record=save_run(output,'RUN51_'+arm,data,runner,state,roster,data['headway_policy'],'GOAL51',arm,'S1',seed,
        full_day=contract['full_day'],cohort_start=0,cohort_end=contract['cohort_end_second'],
        service_end_second=contract['service_end_second'])
    result=dict(contract=contract,decisions=decisions,run=record,metrics=metrics,
        elapsed_seconds=monotonic()-began,full_day=contract['full_day'],
        sources_preserved=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                              for p,h in contract['source_hashes'].items()),
        cash_saving_rub=None,lost_revenue_rub=None)
    atomic_json(output/'result.json',result)
    journal(output/'events.jsonl',dict(stage='completed',arm=arm,seconds=result['elapsed_seconds'],
        actual_service_admissible=metrics['actual_service_admissible'],km=metrics['train_km']))
    print(json.dumps(dict(stage='completed',arm=arm,seconds=result['elapsed_seconds'],
        passed=metrics['actual_service_admissible'],km=metrics['train_km']),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=['REFERENCE','OLD','ENSEMBLE_OLD','BALANCED'],default='BALANCED')
    parser.add_argument('--stage',choices=['pilot','full_day','evening'],default='pilot')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-attempts',type=int,default=2)
    parser.add_argument('--control-start-second',type=int,default=3600)
    run_arm(**vars(parser.parse_args()))
