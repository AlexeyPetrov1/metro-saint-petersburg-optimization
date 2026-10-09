"""Run the shared daily controller with median, frozen ML, or a research oracle."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from economics import write_json
from forecasting import validate_bundle
from goal43_replay import actual_groups, save_run
from metro_daily import prepare_day, rebuild_timetable, proposal_specs, desired_headway
from metro_ml_compare import _metrics, metric_delta
from passenger_groups import GroupedPassengerSimulation
from rolling import decide_operational
from metro_pipeline import load_monitoring_context

PIPELINE_SOURCES = ['metro_events.py','metro_quality.py','metro_monitoring.py',
                    'metro_regime_forecast.py','metro_pipeline.py','forecasting.py','metro_calendar.py','metro_anomaly_monitor.py']
from simulation import digest


ROOT=Path(__file__).resolve().parent
DEFAULT_OUTPUT=ROOT/'data/processed/metro_daily_experiments_20261009'
DATE='2026-09-30'


def read_forecast(output, strategy, station_ids, decision_time):
    at=datetime.fromisoformat(decision_time)
    bundle=json.loads((Path(output)/'forecasts'/f'{strategy}_{at.hour:02}.json').read_text(encoding='utf-8'))
    validate_bundle(bundle,station_ids,decision_time)
    if strategy=='ORACLE' and (bundle.get('evaluation_mode')!='research_oracle_future_entries'
            or bundle['source_versions'].get('operationally_available') is not False):
        raise ValueError('oracle must be explicitly labelled as unavailable in service')
    if strategy!='ORACLE' and bundle['source_versions'].get('future_truth_intentionally_used'):
        raise ValueError('causal arm cannot use oracle future truth')
    closure=datetime.fromisoformat('2026-10-01T00:00:00+03:00')
    issued=datetime.fromisoformat(bundle['issued_at'])
    last=max(datetime.fromisoformat(r['interval_end']) for r in bundle['rows'])
    if last==closure and issued+timedelta(hours=2)>closure:
        second=closure
        while second < issued+timedelta(hours=2):
            bundle['rows'].extend(dict(station_id=s,interval_start=second.isoformat(),
                interval_end=(second+timedelta(minutes=15)).isoformat(),predicted_count=0,coverage='complete')
                for s in station_ids)
            second+=timedelta(minutes=15)
        bundle['source_versions']['service_closure']=dict(at=closure.isoformat(),
            status='known experiment calendar, not fabricated observed data')
        validate_bundle(bundle,station_ids,decision_time)
    return bundle


def _gzip(path, value):
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=3) as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(',',':'))


def bootstrap(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    target=output/'baseline.json.gz'
    if target.exists():
        with gzip.open(target,'rt',encoding='utf-8') as stream: return json.load(stream)
    frame=pd.read_parquet(output/'observations.parquet')
    data,roster,profiles,meta=prepare_day(DATE)
    if not meta['projection']['headways_passed'] or not meta['projection']['obligations_closed']:
        raise ValueError('daily baseline must satisfy movement constraints and return obligations')
    groups=actual_groups(frame,data,meta['service_end_second'])['S1']
    runner=GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles)
    checkpoint=runner.run(end_second=3600)['checkpoint']
    result=dict(data=data,roster=roster,profiles=profiles,meta=meta,groups=groups,checkpoint=checkpoint,
        groups_digest=digest(groups),checkpoint_digest=digest(checkpoint))
    _gzip(target,result)
    write_json(output/'baseline_reference.json',meta)
    contract=dict(date=DATE,line=1,initial_count=33,control_step_seconds=900,first_decision='07:00',
        last_decision='23:45',forecast_refresh_seconds=3600,od='S1',seed=7,
        baseline_status=meta['baseline_status'],reference_date_verified=False,
        groups_digest=result['groups_digest'],checkpoint_digest=result['checkpoint_digest'],
        candidate_specs=proposal_specs(),search_is_bounded=True,oracle_is_operational=False,
        reference_cycle_99_minutes_used=False,policy_digest=digest(data['headway_policy']),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['metro_daily.py','metro_daily_run.py','metro_ml.py','rolling.py','dwell.py','passenger_groups.py']+PIPELINE_SOURCES},
        forecast_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'forecasts').glob('*.json'))})
    write_json(output/'experiment_manifest.json',dict(contract=contract,identity=digest(contract)))
    print(json.dumps(dict(stage='daily_baseline',groups=len(groups),peak=meta['projection']['peak_trains_on_line'],
        passenger_trips=meta['projection']['passenger_trips'],headways_passed=True),ensure_ascii=False),flush=True)
    return result


def make_candidates(data, second, service_end, budget_seconds=480):
    began=time.perf_counter()
    rows=[dict(candidate_id='KEEP',strategy='KEEP',status='movement_checked',_data=data,
        action=dict(kind='KEEP',decision_second=second),timing=dict(search_complete=True))]
    origin=datetime.fromisoformat(data['period_start'])
    # Same hourly forecast expires at the same boundary for all three providers.
    horizon_end=min(service_end,(second//3600)*3600+7200)
    profiles=set(); seen={digest(data)}
    for spec in proposal_specs():
        if time.perf_counter()-began > budget_seconds-30:
            rows.append(dict(**spec,strategy='REPLAN',status='evaluation_timeout',reason='plan generation budget'))
            continue
        clocks=range(second+780,horizon_end,60)
        signature=(spec['priority_rotation'],tuple(desired_headway(origin+timedelta(seconds=t),6080,
            data['headway_policy'],spec['count_delta']) for t in clocks))
        if signature in profiles: continue
        profiles.add(signature)
        try:
            changed,audit=rebuild_timetable(data,second,service_end,count_delta=spec['count_delta'],
                priority_rotation=spec['priority_rotation'],change_end_second=horizon_end)
            identity=digest(changed)
            if identity in seen: continue
            seen.add(identity)
            if not audit['headways_passed'] or not audit['obligations_closed']:
                rows.append(dict(**spec,strategy='REPLAN',status='rejected',reason='full future movement check failed',
                    headways_passed=audit['headways_passed'],obligations_closed=audit['obligations_closed']))
                continue
            rows.append(dict(**spec,strategy='REPLAN',status='movement_checked',_data=changed,
                action=dict(kind='REPLAN',decision_second=second,**spec,passenger_trips=audit['passenger_trips']),
                timing=dict(search_complete=True,committed_until_second=audit['committed_until_second'],
                    passenger_trips=audit['passenger_trips'],peak_trains_on_line=audit['peak_trains_on_line'])))
        except ValueError as error:
            rows.append(dict(**spec,strategy='REPLAN',status='rejected',reason=str(error)))
    return rows,time.perf_counter()-began


def compact_decision(decision):
    compact={k:v for k,v in decision.items() if k!='candidate_table'}
    compact['candidate_table']=[]
    for row in decision['candidate_table']:
        saved={k:v for k,v in row.items() if k!='trials'}
        saved['trials']=[]
        for trial in row.get('trials',[]):
            e=trial['policy_exposure'];h=trial['headways']
            saved['trials'].append(dict(scenario=trial['scenario'],
                policy_admissible=e['admissible'],max_car_load=e['global']['max_load'],
                bad_windows=[w for w in e['windows'] if w['status']=='fail'],
                headways_passed=h['passed'],minimum_headway=h['minimum_seconds'],maximum_headway=h['maximum_seconds'],
                headway_violation_count=len(h['violations']),mature_complete=trial['mature_cohort']['complete'],
                initial_obligations_complete=trial['initial_obligations']['complete']))
        compact['candidate_table'].append(saved)
    return compact


def run_strategy(output, strategy, *, max_steps=None):
    output=Path(output)
    if strategy not in ('BASELINE','MEDIAN','ML','ORACLE'): raise ValueError('unknown daily strategy')
    final_path=output/f'{strategy.lower()}_result.json'
    if final_path.exists() and max_steps is None:
        return json.loads(final_path.read_text(encoding='utf-8'))
    shared=bootstrap(output)
    monitoring_context=load_monitoring_context(output)
    folder=output/'controller'/(strategy+'_pilot' if max_steps is not None else strategy);folder.mkdir(parents=True,exist_ok=True)
    data=shared['data'];profiles=shared['profiles'];groups=shared['groups'];meta=shared['meta']
    roster=shared['roster'];policy=data['headway_policy'];service_end=meta['service_end_second']
    runner=GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles)
    state=copy.deepcopy(shared['checkpoint']);previous=3600;decisions=[]
    ids=[s['station_id'] for s in data['stations'] if s['kind']=='line']
    source_names=['metro_daily.py','metro_daily_run.py','metro_ml.py','rolling.py','dwell.py','passenger_groups.py']+PIPELINE_SOURCES
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    versions=output/'source_versions';versions.mkdir(exist_ok=True)
    for name,sha in source_hashes.items():
        archived=versions/f'{sha}_{name}'
        if not archived.exists():archived.write_bytes((ROOT/name).read_bytes())
    began=time.perf_counter();times=list(range(3600,service_end,900))
    if max_steps is not None:times=times[:max_steps]
    if strategy!='BASELINE':
        for index,second in enumerate(times):
            if second > previous: state=runner.run(start_second=previous,end_second=second,checkpoint=state)['checkpoint']
            tick=time.perf_counter()
            at=(datetime.fromisoformat(data['period_start'])+timedelta(seconds=second)).isoformat()
            bundle=read_forecast(output,strategy,ids,at)
            candidates,generation_time=make_candidates(data,second,service_end)
            decision=decide_operational(data,{'S1':runner},{'S1':state},profiles,{},bundle,policy,
                candidates=candidates,budget_seconds=max(10,480-generation_time),include_recovery=False,
                monitoring_context=monitoring_context)
            decision['elapsed_seconds_including_plan_generation']=time.perf_counter()-tick
            decision['generation_seconds']=generation_time
            decision['source_mode']=strategy
            selection=decision['selection']
            selected=selection.get('selected_candidate_id') or selection.get('mitigation_candidate_id') or 'KEEP'
            candidate=next((r for r in candidates if r['candidate_id']==selected and r.get('_data')),candidates[0])
            changed=candidate['_data']
            if changed is not data:
                new=GroupedPassengerSimulation(changed,changed['plans'][0]['plan_id'],groups,profiles)
                state=new.branch_checkpoint(state,runner);runner=new;data=changed
            saved=compact_decision(decision);decisions.append(saved)
            write_json(folder/f'decision_{second:05}.json',saved)
            _gzip(folder/f'plan_{second:05}.json.gz',data)
            print(json.dumps(dict(stage='daily_control',strategy=strategy,index=index+1,decisions=len(times),
                at=at,selected=selected,candidates=len(decision['candidate_table']),
                seconds=round(time.perf_counter()-tick,2)),ensure_ascii=False),flush=True)
            previous=second
    if max_steps is not None:
        write_json(folder/'pilot.json',dict(decisions=decisions,elapsed_seconds=time.perf_counter()-began))
        return dict(decisions=decisions,elapsed_seconds=time.perf_counter()-began)
    state=runner.run(start_second=previous,checkpoint=state)['checkpoint']
    run_id=f'{DATE}_daily_{strategy}_S1_7'
    record=save_run(output,run_id,data,runner,state,roster,policy,'daily',strategy,'S1',7,
        full_day=True,cohort_start=3600,cohort_end=service_end,service_end_second=service_end)
    metrics=_metrics(data,runner,state,roster,policy,3600,service_end)
    metrics['peak_trains_on_line']=runner.motion['checks']['peak_trains_on_line']
    result=dict(strategy=strategy,date=DATE,metrics=metrics,decisions=decisions,run=record,
        groups_digest=shared['groups_digest'],checkpoint_digest=shared['checkpoint_digest'],
        elapsed_seconds=time.perf_counter()-began,research_oracle=strategy=='ORACLE',
        forecast_error='zero for future entry counts' if strategy=='ORACLE' else None,
        maximum_decision_seconds=max((d['elapsed_seconds_including_plan_generation'] for d in decisions),default=0))
    result['source_hashes']=source_hashes
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in source_hashes.items()):
        raise ValueError('experiment source changed during execution')
    write_json(final_path,result)
    print(json.dumps(dict(stage='daily_complete',strategy=strategy,train_km=metrics['train_km'],
        admissible=metrics['actual_service_admissible'],seconds=round(result['elapsed_seconds'],2)),ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--strategy',choices=['BASELINE','MEDIAN','ML','ORACLE'],required=True)
    parser.add_argument('--pilot-steps',type=int)
    args=parser.parse_args()
    run_strategy(args.output,args.strategy,max_steps=args.pilot_steps)


if __name__=='__main__':main()
