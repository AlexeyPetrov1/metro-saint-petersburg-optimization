"""Paired finite-cohort rolling catalogue; reuse the verified simulation kernels."""
import argparse
import copy
import gzip
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/processed/metro44_continuation_6h_closed_20261009/runs/CONT6H_CLOSED_KEEP_S1_7'
CACHE = ROOT / 'data/processed/metro_daily_experiments_20261009'
PILOT = ROOT / 'data/processed/metro44_catalog_pilot_20261009'
sys.path.insert(0, str(SNAPSHOT))

from forecasting import hourly_median, validate_bundle
from goal43_replay import save_run
from metro44_catalog import calculate_catalog
from metro44_focus import cohort_forecast, execution_prefix, load_resume
from metro44_run import (bounded_decision, keep_wait_budgets, public_decision, selected_plan,
                         write_gzip_once, write_json_once)
from metro44_selection import choose_candidate
from metro_ml import FrozenMetroForecaster
from metro_ml_compare import _metrics
from passenger_groups import GroupedPassengerSimulation
from simulation import digest


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_counterfactual(decision):
    for row in decision['candidate_table']:
        if row.get('passengers_checked') and any('mature_cohort' not in t for t in row.get('trials', [])):
            raise ValueError('strict diagnostic requires raw future cohort before public serialization')
    mean, p95 = keep_wait_budgets(decision['candidate_table'])
    return choose_candidate(decision['candidate_table'], mean_wait_budget_s=mean, p95_wait_budget_s=p95,
                            search_incomplete=not decision.get('search_complete', False))


def prepare():
    if (OUTPUT / 'shared_contract.json').exists():
        return read(OUTPUT / 'shared_scenario.json.gz'), read(OUTPUT / 'shared_contract.json')
    original = read(PILOT / 'contract.json')
    inputs = dict(original['input_hashes'])
    inputs[str(PILOT / 'observed_frame.parquet')] = sha(PILOT / 'observed_frame.parquet')
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    for h in range(9, 15):
        for method in ('ML', 'MEDIAN'):
            path = CACHE / 'forecasts' / f'{method}_{h:02}.json'
            inputs[str(path)] = sha(path)
    data, groups, profiles = (read(SOURCE / name) for name in ('movement_input.json','groups.json.gz','profiles.json'))
    roster = [dict(train_id=t['train_id'], type=t['train_type']) for t in data['trains']]
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles)
    state = runner.run(end_second=3600)['checkpoint']
    shared = dict(data=data, groups=groups, profiles=profiles, roster=roster, state=state)
    contract = dict(source=str(SOURCE), snapshot=str(SNAPSHOT), source_hashes=original['source_hashes'],
        input_hashes=inputs, groups_digest=digest(groups), checkpoint_digest=digest(state),
        cohort_start_second=3600, cohort_end_second=21600, passenger_departure_end_second=22200,
        clearance_seconds=600, closure_end_second=32400, decision_times=list(range(3600,21600,900)),
        first_decision_at='2026-09-30T09:15:00+03:00', last_decision_at='2026-09-30T14:00:00+03:00',
        initial_count=53, fixed_fleet_size=68, quality_policy='pareto', budgets_are_user_requirements=False,
        mean_budget_seconds=90, p95_budget_seconds=180, posthoc_development=True,
        finite_cohort_forecast_boundary='zero after14:15 means outside experimental cohort; not metro closing',
        boundary_end_is_operational_closure=False, actual_simulated_line_id=1, requested_line_id=2,
        independent_holdout=False, full_day=False, economic_improvement_claimed=False)
    write_gzip_once(OUTPUT / 'shared_scenario.json.gz', shared)
    write_json_once(OUTPUT / 'shared_contract.json', contract)
    return shared, contract


def run(arm, resume=False):
    import pandas as pd
    shared, common = prepare()
    hashes = {**common['source_hashes'], **common['input_hashes']}
    if any(sha(Path(p)) != h for p,h in hashes.items()): raise ValueError('source/input changed before run')
    folder = OUTPUT / 'controller' / arm
    contract = dict(arm=arm, shared_digest=digest(common), source_hashes=common['source_hashes'],
        budget_seconds=480, catalogue_generation_budget_seconds=470, policy='assumed development90/180')
    identity = digest(contract)
    if (folder / 'contract.json').exists():
        if not resume or read(folder / 'contract.json') != contract: raise ValueError('arm exists; explicit matching resume required')
    else: write_json_once(folder / 'contract.json', contract)
    final = OUTPUT / f'{arm.lower()}_result.json'
    if final.exists(): raise ValueError('refusing to overwrite completed arm')
    data, state = copy.deepcopy(shared['data']), copy.deepcopy(shared['state'])
    prior = load_resume(folder, identity, common['decision_times']) if resume else None
    previous, index = state['second'], 0
    if prior: data,state,previous,index = prior['data'],prior['state'],prior['second'],prior['next_index']
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], shared['groups'], shared['profiles'])
    warmup = execution_prefix(runner.motion,3600)
    ids = [s['station_id'] for s in data['stations'] if s['kind']=='line']
    origin = datetime.fromisoformat(data['period_start'])
    boundary = (origin+timedelta(seconds=21600)).isoformat()
    model = frame = None
    decisions = [read(folder / f'decision_{s:05}.json') for s in common['decision_times'][:index]]
    began = monotonic()
    for position, second in enumerate(common['decision_times'][index:],index):
        if second>previous: state = runner.run(start_second=previous,end_second=second,checkpoint=state)['checkpoint']
        at = (origin+timedelta(seconds=second)).isoformat()
        prefix = execution_prefix(runner.motion,second)
        def pipeline(deadline,cancelled):
            nonlocal model,frame
            inference_began = monotonic()
            cold = frame is None
            if frame is None: frame = pd.read_parquet(PILOT/'observed_frame.parquet')
            if arm=='MPC_ML':
                if model is None: model = FrozenMetroForecaster()
                fresh = model.forecast_bundle(frame,ids,at)
            else:
                issued = (origin+timedelta(seconds=second)).replace(minute=0,second=0,microsecond=0)
                fresh = hourly_median(frame,ids,issued.isoformat(),2)
            inference_seconds = monotonic()-inference_began
            cached = read(CACHE/'forecasts'/f'{"ML" if arm=="MPC_ML" else "MEDIAN"}_{datetime.fromisoformat(at).hour:02}.json')
            key = lambda r:(r['station_id'],r['interval_start'],r['interval_end'],r['predicted_count'])
            parity = sorted(map(key,fresh['rows']))==sorted(map(key,cached['rows']))
            bundle = cohort_forecast(fresh,boundary)
            validate_bundle(bundle,ids,at)
            result = calculate_catalog(data,runner,state,shared['profiles'],shared['roster'],CACHE,arm,ids,
                second,22200,3,deadline,cancelled,quality_policy='pareto',bundle_override=bundle)
            result.update(inference_seconds=inference_seconds,cold_load=cold,numeric_cache_parity=parity,
                          raw_forecast_digest=digest(fresh),bounded_forecast=bundle)
            return result
        receipt = bounded_decision(pipeline,480)
        if receipt['completed']:
            computed = receipt['result']; decision,candidates = computed['decision'],computed['candidates']
            strict = strict_counterfactual(decision)
            write_json_once(folder/f'forecast_{second:05}.json',computed['bounded_forecast'])
        else:
            decision = dict(selection=receipt['selection'],candidate_table=[],search_complete=False)
            candidates = [dict(candidate_id='KEEP',_data=data)];strict=None;computed={}
        changed,applied = selected_plan(decision['selection'],candidates,decision['candidate_table'])
        if changed is not data:
            new = GroupedPassengerSimulation(changed,changed['plans'][0]['plan_id'],shared['groups'],shared['profiles'])
            if execution_prefix(new.motion,second)!=prefix:raise ValueError('executed prefix rewritten')
            state = new.branch_checkpoint(state,runner);runner,data = new,changed
        if execution_prefix(runner.motion,3600)!=warmup:raise ValueError('common warmup rewritten')
        decision.update(applied_candidate_id=applied,decision_second=second,decision_time=at,
            strict_counterfactual_selection=strict,executed_prefix_digest=prefix,groups_digest=common['groups_digest'],
            initial_checkpoint_digest=common['checkpoint_digest'],pipeline_seconds=receipt['elapsed_seconds'],
            actual_inference_seconds=computed.get('inference_seconds'),cold_load=computed.get('cold_load'),
            numeric_cache_parity=computed.get('numeric_cache_parity'),raw_forecast_digest=computed.get('raw_forecast_digest'),
            pipeline_includes_actual_inference=True,local_recommendation_within_480=receipt['completed'],
            late_work_discarded=not receipt['completed'],external_delivery_measured=False)
        saved=public_decision(decision);decisions.append(saved)
        write_gzip_once(folder/f'resume_{second:05}.json.gz',dict(contract_identity=identity,next_index=position+1,
            second=second,state=state,data=data,decision=saved,state_digest=digest(state),data_digest=digest(data),
            executed_prefix_digest=prefix))
        write_json_once(folder/f'decision_{second:05}.json',saved)
        write_gzip_once(folder/f'plan_{second:05}.json.gz',data)
        print(json.dumps(dict(stage='catalog_control',arm=arm,index=position+1,at=at,applied=applied,
            strict_selected=None if strict is None else strict['selected_candidate_id'],
            pipeline_seconds=receipt['elapsed_seconds'],cache_parity=computed.get('numeric_cache_parity'))),flush=True)
        previous=second
    state=runner.run(start_second=previous,checkpoint=state)['checkpoint']
    record=save_run(OUTPUT,f'CONT6H_CATALOG_{arm}_S1_7',data,runner,state,shared['roster'],data['headway_policy'],
        'CONT6H_CATALOG',arm,'S1',7,full_day=False,cohort_start=3600,cohort_end=21600,service_end_second=21600)
    metrics=_metrics(data,runner,state,shared['roster'],data['headway_policy'],3600,21600)
    metrics['peak_trains_on_line']=runner.motion['checks']['peak_trains_on_line']
    result=dict(strategy=arm,contract_identity=identity,shared_contract=common,decisions=decisions,run=record,
        metrics=metrics,groups_digest=common['groups_digest'],checkpoint_digest=common['checkpoint_digest'],
        maximum_pipeline_seconds=max(d['pipeline_seconds']for d in decisions),full_day=False,
        forecast_inference_in_timing=True,independent_holdout=False,economic_improvement_claimed=False,
        cash_saving_rub=None,lost_revenue_rub=None,release_authorized=False,elapsed_seconds=monotonic()-began,
        boundary_end_is_operational_closure=False,source_preserved=all(sha(Path(p))==h for p,h in hashes.items()))
    if not result['source_preserved']:raise ValueError('source/input changed; journals retained')
    write_json_once(final,result)
    print(json.dumps(dict(stage='catalog_complete',arm=arm,decisions=len(decisions),
        train_km=metrics['train_km'],mean_wait=metrics['mean_wait_seconds'],p95=metrics['p95_wait_seconds'],
        source_preserved=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--arm',choices=['MPC_ML','MPC_BASELINE_FORECAST']);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:prepare()
    elif args.arm:run(args.arm,args.resume)
    else:parser.error('choose --prepare-only or --arm')
