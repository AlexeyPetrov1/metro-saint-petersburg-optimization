"""Paired quarter forecasts and concrete depot actions on one preserved episode."""

import argparse
import copy
import gzip
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data/processed/metro47_withdrawal_normalized_actual_20261009/runs/NORMALIZED_REFERENCE_S1_7'
FRAME = ROOT/'data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet'


def decision_times(start, end, *, event_times=()):
    if any(type(s) is not int or not start <= s < end for s in event_times):
        raise ValueError('incident clock outside control period')
    return sorted(set(range(start,end,900)) | set(event_times))


def calculate(data, runner, state, profiles, roster, raw, frame, second, end, deadline,
              cancelled, *, mode, extend, guard, solver, catalog, refine, builder):
    at = (datetime.fromisoformat(data['period_start'])+timedelta(seconds=second)).isoformat()
    boundary = (datetime.fromisoformat(data['period_start'])+timedelta(seconds=21600)).isoformat()
    bundle = guard(extend(raw,frame,[s['station_id'] for s in data['stations'] if s['kind']=='line'],at,boundary),at)
    extra = {'return_by_second':21600} if mode == 'temporary' else {}
    def solve(**constraints):
        return solver(data,bundle,state,second,end,deadline=deadline,**extra,**constraints)
    def evaluate(ids):
        return catalog(data,runner,state,profiles,roster,bundle,second,end,deadline,cancelled,
            proposal=dict(selected_train_ids=ids,**extra) if ids else None,builder=builder)
    answer = refine(solve,evaluate,deadline=deadline,cancelled=cancelled)
    answer.update(bounded_forecast=bundle,raw_forecast=raw)
    return answer


def run(output, *, arm='GRU', steps=20, mode='temporary'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if arm not in ('GRU','ML','MEDIAN','REFERENCE') or mode not in ('temporary','permanent'):
        raise ValueError('causal forecast and supported action required')
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('one to twenty consecutive decisions required')
    import pandas as pd
    from goal43_replay import save_run
    from metro44_focus import execution_prefix
    from metro44_run import bounded_decision, public_decision, selected_plan, write_json_once, write_gzip_once
    from metro_ml import FrozenMetroForecaster
    from metro_ml_compare import _metrics
    from metro47_catalog import calculate_cost_catalog
    from metro47_robust_forecast import apply_safety_margin
    from metro47_withdrawal_control import refine_withdrawals
    from metro48_forecasts import complete_tail, profiled_median, reprofile_hourly
    from metro48_gru import FrozenQuarterGRU, MODEL_INPUTS
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest
    if mode == 'temporary':
        from metro48_temporary import build_temporary_candidate as builder
        from metro48_temporary_solver import solve_temporary_withdrawals as solver
    else:
        from metro47_withdrawal import build_withdrawal_candidate as builder
        from metro47_withdrawal_solver import solve_withdrawals as solver
    def read(path):
        raw = Path(path).read_bytes()
        return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)
    data, groups, profiles = (read(SOURCE/name) for name in ('movement_input.json','groups.json.gz','profiles.json'))
    from withdrawal_stabling import attach_stabling
    data = attach_stabling(data)
    frame = pd.read_parquet(FRAME)
    roster = [dict(train_id=t['train_id'],type=t['train_type']) for t in data['trains']]
    runner = GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,profiles)
    state = runner.run(end_second=3600)['checkpoint']
    origin = datetime.fromisoformat(data['period_start']); ids = [s['station_id'] for s in data['stations'] if s['kind']=='line']
    paths = list(ROOT.glob('*.py')) + [SOURCE/name for name in ('movement_input.json','groups.json.gz','profiles.json')]
    paths += [FRAME,ROOT/'data/raw/metro/data.zip',ROOT/'data/raw/metro/lgbm_final.zip']
    paths += list((MODEL_INPUTS/'stack_gru_2026-09').glob('*'))
    paths = [p for p in paths if p.is_file() and not p.name.startswith('metro48_daily') and p.name != 'metro48_gru_evaluation.py']
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected = {str(p):sha(p) for p in paths}
    output.mkdir(parents=True)
    contract = dict(arm=arm,action_mode=mode,line_id=1,full_day=False,fixed_fleet_size=68,
        cohort_start_second=3600,cohort_end_second=21600,service_end_second=22200,
        closure_end_second=32400,decision_times=decision_times(3600,21600)[:steps],
        groups_digest=digest(groups),checkpoint_digest=digest(state),input_hashes=protected,
        objective='minimum closed-schedule TO1 resource cost',waiting_constraints=None,
        safety_margin=1.25,statistical_coverage_guarantee=False,parameters_tuned_on_this_run=False,
        historical_tail='same past-only station median beyond model horizon until14:15',
        action_commitment='explicit preserved return before14:15' if mode=='temporary' else 'withdrawal until episode closure',
        native_GRU_output_seconds=1800,GRU_refresh_seconds=900,shifted_GRU_updates_experimental=True,
        median_and_hourly_ML_profile='same causal GRU normalization station profile',
        development_day_reused=True,prospective_holdout=False,cash_saving_rub=None)
    write_json_once(output/'contract.json',contract)
    write_gzip_once(output/'common_scenario.json.gz',dict(data=data,groups=groups,profiles=profiles,state=state,roster=roster))
    folder = output/'controller'; model = old = None; decisions = []; previous = 3600; began = monotonic()
    warmup = execution_prefix(runner.motion,3600)
    for index,second in enumerate(contract['decision_times']):
        if second > previous:
            state = runner.run(start_second=previous,end_second=second,checkpoint=state)['checkpoint']
        at = (origin+timedelta(seconds=second)).isoformat(); prefix = execution_prefix(runner.motion,second)
        def pipeline(deadline,cancelled):
            nonlocal model,old
            inference = monotonic()
            if arm == 'REFERENCE':
                raw = profiled_median(frame,ids,at)
                answer = calculate_cost_catalog(data,runner,state,profiles,roster,raw,second,22200,deadline,cancelled)
                answer.update(bounded_forecast=raw,raw_forecast=raw)
            else:
                if model is None: model = FrozenQuarterGRU()
                if arm == 'GRU': raw = model.forecast_bundle(frame,ids,at)
                elif arm == 'MEDIAN': raw = profiled_median(frame,ids,at,profile_provider=model.station_profile)
                else:
                    if old is None: old = FrozenMetroForecaster(use_retrospective_history_flags=False)
                    raw = reprofile_hourly(old.forecast_bundle(frame,ids,at),model.station_profile)
                answer = calculate(data,runner,state,profiles,roster,raw,frame,second,22200,deadline,cancelled,
                    mode=mode,extend=complete_tail,guard=apply_safety_margin,solver=solver,
                    catalog=calculate_cost_catalog,refine=refine_withdrawals,builder=builder)
            answer['pipeline_compute_seconds'] = monotonic()-inference
            return answer
        receipt = bounded_decision(pipeline,480)
        if not receipt['completed']:
            raise TimeoutError('quarter decision exceeded480 seconds; prior journals preserved')
        answer = receipt['result']; decision = answer['decision']
        changed, applied = selected_plan(decision['selection'],answer['candidates'],decision['candidate_table'])
        if changed is not data:
            new = GroupedPassengerSimulation(changed,changed['plans'][0]['plan_id'],groups,profiles)
            if execution_prefix(new.motion,second) != prefix:
                raise ValueError('executed movement prefix changed')
            state = new.branch_checkpoint(state,runner); runner,data = new,changed
        if execution_prefix(runner.motion,3600) != warmup:
            raise ValueError('common warmup changed')
        decision.update(applied_candidate_id=applied,pipeline_seconds=receipt['elapsed_seconds'],
            executed_prefix_digest=prefix,local_recommendation_within_480=True,external_delivery_measured=False)
        saved = public_decision(decision); decisions.append(saved)
        write_json_once(folder/f'decision_{second:05}.json',saved)
        write_json_once(folder/f'raw_forecast_{second:05}.json',answer['raw_forecast'])
        write_json_once(folder/f'forecast_{second:05}.json',answer['bounded_forecast'])
        write_gzip_once(folder/f'resume_{second:05}.json.gz',dict(second=second,data=data,state=state,decision=saved))
        print(json.dumps(dict(stage='quarter_control',arm=arm,mode=mode,index=index+1,at=at,
            applied=applied,seconds=receipt['elapsed_seconds'])),flush=True)
        previous = second
    state = runner.run(start_second=previous,checkpoint=state)['checkpoint']
    record = save_run(output,f'Q48_{arm}_{mode}',data,runner,state,roster,data['headway_policy'],
        'QUARTER48',arm,'S1',7,full_day=False,cohort_start=3600,cohort_end=21600,service_end_second=22200)
    metrics = _metrics(data,runner,state,roster,data['headway_policy'],3600,21600)
    after = {str(p):sha(p) for p in paths}
    if after != protected:
        raise ValueError('protected inputs changed; journals retained')
    result = dict(contract=contract,decisions=decisions,run=record,metrics=metrics,
        elapsed_seconds=monotonic()-began,maximum_pipeline_seconds=max(d['pipeline_seconds'] for d in decisions),
        sources_preserved=True,full_day=False,cash_saving_rub=None,lost_revenue_rub=None)
    write_json_once(output/'result.json',result)
    print(json.dumps(dict(stage='quarter_complete',arm=arm,mode=mode,train_km=metrics['train_km'],
        wait=metrics['cohort']['mean_wait_seconds'],elapsed=result['elapsed_seconds'])),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=['GRU','ML','MEDIAN','REFERENCE'],default='GRU')
    parser.add_argument('--mode',choices=['temporary','permanent'],default='temporary')
    parser.add_argument('--steps',type=int,default=20)
    run(**vars(parser.parse_args()))
