"""Twenty new future environments from the selected 09:45 snapshot."""

import hashlib
import json
import time
from pathlib import Path
from statistics import median

import pandas as pd

from decision_engine import prepare_demand, headway_report
from dwell import compile_timing
from economics import write_json
from line1 import load_config, write_checkpoint
from metro_replay import ROOT, warmup, input_files
from passenger_groups import GroupedPassengerSimulation
from passengers import PassengerSimulation
from replay_metrics import cohort_waiting, overload_metrics, work_between


def observed_headways(data,motion):
    used={e['location'] for e in motion['events'] if e['event_type']=='departure'}
    used.update(r['track_id'] for r in data.get('previous_departures',[]))
    observed=dict(data,tracks=[t for t in data['tracks'] if t['track_id'] in used])
    return headway_report(observed,motion)


def future_checkpoint(target,source,snapshot):
    """Source is validated once; run() still fully validates every target fork."""
    start=snapshot['second']
    if target.profiles!=source.profiles or target.tracks!=source.tracks or target.motion['initial_state']!=source.motion['initial_state']:
        raise ValueError('future validation changes past fleet or network')
    past=[e for e in source.stream if e['second']<start]
    if past!=[e for e in target.stream if e['second']<start]: raise ValueError('future validation changes past events')
    if any(target.by_id.get(key)!=group for key,group in source.by_id.items() if group['turnstile_second']<start):
        raise ValueError('future validation changes past entries')
    state=json.loads(json.dumps(snapshot))
    state.update(context_id=target.context_id,next_event_index=len(past))
    state['sha256']=PassengerSimulation._seal(state)
    return state


def run():
    episode=ROOT/'data/processed/goal4_2/episodes/D_mild'
    decision_path=episode/'decision_13500.json'
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    selected=decision['selection']['selected_candidate_id']
    if selected!='KEEP:plan': raise ValueError('this validation requires the observed KEEP selection')
    for path in episode.glob('decision_*.json'):
        earlier=json.loads(path.read_text(encoding='utf-8'))
        if earlier['decision_second']<13500 and earlier['actual_applied_action']!='KEEP':
            raise ValueError('reconstruction requires the observed unchanged past plan')
    selected_row=next(r for r in decision['candidate_table'] if r['candidate_id']==selected)
    policy_path=ROOT/'config/decision_engine/goal4_2.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8'))
    assert selected_row['tested_seeds']==policy['preliminary_seeds']
    assert not set(policy['preliminary_seeds']) & set(policy['validation_seeds'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=input_files()+[Path(__file__),decision_path]
    inputs={str(p):sha(p) for p in paths}
    folder=ROOT/'data/processed/goal4_2_verification/selected_validation'; folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); config=load_config()
    groups,_,_=prepare_demand(pd.read_parquet(ROOT/'data/processed/metro/entries_station_15m.parquet'),config)
    data,registry,profiles,sources,states,_=warmup(config,groups,'mild',policy['actual_episode_seed'])
    for scenario,source in sources.items():
        states[scenario]=source.run(start_second=7200,end_second=13500,checkpoint=states[scenario])['checkpoint']
        source._validate_checkpoint(states[scenario],13500)
        write_checkpoint(folder/f'{scenario}_snapshot_0945.json.gz',states[scenario])
    horizon_end=13500+decision['available_horizon_seconds']; records=[]
    for seed in policy['validation_seeds']:
        began=time.perf_counter()
        changed,_=compile_timing(data,'line1-baseline','mild',seed,committed=data['leg_timing'],as_of=13500)
        for scenario,source in sources.items():
            runner=GroupedPassengerSimulation(changed,'line1-baseline',groups[scenario],profiles)
            checkpoint=future_checkpoint(runner,source,states[scenario])
            out=runner.run(start_second=13500,checkpoint=checkpoint)
            state=out['checkpoint']; horizon=overload_metrics(runner,state,13500,horizon_end)
            tail=overload_metrics(runner,state,13500,21600)
            row=dict(seed=seed,scenario=scenario,selected_candidate_id=selected,baseline_candidate_id='KEEP:plan',
                paired_actions_identical=True,forecast_horizon_end_second=horizon_end,
                horizon_overload=horizon,tail_overload=tail,
                main_cohort=cohort_waiting(runner,state,7200,14400),
                late_cohort=cohort_waiting(runner,state,13500,14400),
                queue_age_seconds=out['metrics']['max_queue_age_seconds'],resources=work_between(changed,runner.motion,13500,21600),
                minimum_headway_seconds=observed_headways(changed,runner.motion)['minimum_seconds'],cash_delta_RUB=None,
                scope='chosen decision fixed before future actual entries are replayed; same observed 09:45 stock',release_authorized=False)
            write_checkpoint(folder/f'{seed}_{scenario}.json.gz',state)
            write_json(folder/f'{seed}_{scenario}_summary.json',row)
            records.append(row)
        print(json.dumps(dict(stage='selected_validation',seed=seed,seconds=round(time.perf_counter()-began,2))),flush=True)
    aggregate=[]
    for scenario in sources:
        rows=[r for r in records if r['scenario']==scenario]
        result=dict(scenario=scenario,seeds=len(rows),
            horizon_violation_fraction=sum(r['horizon_overload']['exceeded_car_seconds']>0 for r in rows)/len(rows),
            tail_violation_fraction=sum(r['tail_overload']['exceeded_car_seconds']>0 for r in rows)/len(rows))
        for label,values in [('late_mean_wait',[r['late_cohort']['mean_wait_seconds'] for r in rows]),
                ('late_p95_wait',[r['late_cohort']['p95_wait_seconds'] for r in rows]),
                ('horizon_peak',[r['horizon_overload']['max_car_load'] for r in rows]),
                ('tail_peak',[r['tail_overload']['max_car_load'] for r in rows]),
                ('train_km',[float(r['resources']['train_km']) for r in rows])]:
            complete=sum(value is not None for value in values)
            result[label]=dict(median=median(values),minimum=min(values),maximum=max(values),complete_runs=complete) if complete==len(values) else dict(median=None,minimum=None,maximum=None,complete_runs=complete)
        aggregate.append(result)
    write_json(folder/'results.json',records); write_json(folder/'summary.json',aggregate)
    write_json(folder/'scope.json',dict(mode='D_mild',decision_second=13500,selected_candidate_id=selected,
        actual_entries='evaluation only; never used to select the saved decision',baseline='identical KEEP continuation',
        seeds=policy['validation_seeds'],seconds=time.perf_counter()-started,release_authorized=False))
    if any(sha(Path(p))!=value for p,value in inputs.items()): raise ValueError('validation inputs changed during the campaign')
    outputs={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='build_manifest.json'}
    write_json(folder/'build_manifest.json',dict(status='success',inputs=inputs,outputs=outputs))
    print(json.dumps(dict(status='success',runs=len(records),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__': run()
