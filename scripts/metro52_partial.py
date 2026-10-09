"""Materialize a paired observed window from a preserved BALANCED checkpoint."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import re
from time import monotonic

from metro51_run import actual_per_train, atomic_json
from metro50_daily import read
from passenger_groups import GroupedPassengerSimulation
from simulation import digest

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partial_contract(original, steps):
    start = original.get('control_start_second',900)
    if type(steps) is not int or steps < 1 or start != 900:
        raise ValueError('Partial scope requires positive steps and the original 900-second control start')
    end = start+(steps-1)*900
    contract = copy.deepcopy(original)
    contract.update(stage=f'partial_{steps}',full_day=False,stop_after_steps=steps,
        cohort_end_second=end,entry_end_second=end,platform_cohort_end_second=end,
        observation_end_second=end,service_end_second=end,control_end_second=end+1,
        decision_times=[] if original['arm']=='REFERENCE' else list(range(start,end+1,900)),
        observation_followup_seconds=0,passenger_observation_censored=True,
        checkpoint_boundary_events='pending',last_decision_future_execution_observed=False,
        mileage_boundary_scope='completed track arrivals <= cutoff, matching the existing independent report ledger',
        cash_saving_rub=None,annual_repeat_days=None,annual_repeat_is_scenario=False,
        cost_scope='actual completed track arrivals through the observed checkpoint; no day/year extrapolation',
        future_plan_scope='future movement plans retained for checkpoint provenance, not observed mileage',
        forecast_scope='saved origins through the cutoff; original future forecast horizons retained unchanged')
    if original.get('period_start'):
        contract['planned_period_end'] = original.get('period_end')
        contract['period_end'] = (datetime.fromisoformat(original['period_start'])+timedelta(seconds=end)).isoformat()
    return contract


def branch_observed(common, saved, *, seed, capacity):
    end = saved['second']
    if saved['state']['second'] != end:
        raise ValueError('Checkpoint boundary mismatch')
    options = dict(boarding_seed=seed,boarding_capacity_per_car=capacity)
    original = GroupedPassengerSimulation(saved['data'],saved['data']['plans'][0]['plan_id'],
        common['groups'],common['profiles'],**options)
    groups = [copy.deepcopy(g) for g in common['groups'] if g['turnstile_second']<end]
    runner = GroupedPassengerSimulation(saved['data'],saved['data']['plans'][0]['plan_id'],groups,common['profiles'],**options)
    state = runner.branch_checkpoint(saved['state'],original)
    return runner,state,groups


def metrics(data, runner, state, roster, contract):
    from metro44_catalog import physical_gate
    from metro51_service import local_boarding_service
    from replay_metrics import cohort_waiting,work_between
    from rolling import operational_exposure,operational_headways

    end,policy = state['second'],data['headway_policy']
    cohort = cohort_waiting(runner,state,0,end)
    mature = cohort_waiting(runner,state,0,max(0,end-600))
    resources = work_between(data,runner.motion,0,end)
    exposure = operational_exposure(state,runner.profiles,policy,0,end)
    headways = operational_headways(data,runner.motion,0,end)
    local = local_boarding_service(runner,state,end_second=end,
        first_limit=contract.get('first_miss_limit',.05),second_limit=contract.get('second_miss_limit',.01))
    physical = physical_gate(data,0,end,reference_motion=runner.motion)
    per_train = actual_per_train(data,runner.motion,end)
    km = sum(t['km'] for t in per_train)
    remaining = state['queued']+state['approaching']+state['onboard_count']
    balance = state['entered'] == state['completed']+remaining
    if not balance or abs(km-float(resources['train_km'])) > 1e-7:
        raise ValueError('Observed passenger balance or mileage ledgers disagree')
    closed = remaining == 0 and state['entered'] == state['completed']
    return dict(train_km=km,mean_km_per_fixed_train=km/len(roster),fixed_fleet_size=len(roster),
        actual_per_train=per_train,per_train=per_train,resources=resources,
        resources_scope=dict(start_second=0,end_second=end,includes_common_warmup=True,
            includes_depot_return='only arrivals observed before cutoff'),
        control_window_resources=resources,control_window_scope=dict(start_second=0,end_second=end),
        mean_wait_seconds=mature['mean_wait_seconds'],p95_wait_seconds=mature['p95_wait_seconds'],
        mean_wait_scope=dict(start_second=0,end_second=max(0,end-600),late_arrival_allowance_seconds=600,
            identical_across_strategies=True),cohort=cohort,mature_cohort=mature,
        policy_exposure=exposure,headways=headways,local_boarding_service=local,
        physical_gate={k:v for k,v in physical.items() if not k.startswith('_')},
        max_car_load=state['max_car_load'],max_queue=state['max_queue'],
        max_platform_direction_queue=state['max_platform_direction_queue'],
        entered=state['entered'],completed=state['completed'],queued_at_closure=state['queued'],
        approaching_at_closure=state['approaching'],onboard_at_closure=state['onboard_count'],
        passenger_balance_passed=balance,complete_passenger_accounting=closed,
        passenger_observation_censored=not closed,observation_followup_seconds=0,
        movement_obligations_closed=False,planned_movement_obligations_closed=runner.motion['obligations_closed'],
        all_service_arrivals_completed=cohort['complete'],cost_observation_end_second=end,
        actual_service_admissible=bool(closed and physical['movement_passed'] and local['passed'] is True
            and exposure['admissible'] is True and mature['complete']),
        cash_saving_rub=None,lost_revenue_rub=None,release_authorized=False)


def copy_recorded(source, target, receipt):
    target.parent.mkdir(parents=True,exist_ok=True)
    raw = source.read_bytes()
    target.write_bytes(raw)
    value = hashlib.sha256(raw).hexdigest()
    if sha(source) != value or sha(target) != value:
        raise ValueError('Persisted evidence changed while being copied')
    receipt[str(source)] = value


def materialize(source, output, *, steps=60, max_reference_seconds=300):
    from goal43_replay import save_run
    from rolling import operational_exposure

    source,output = Path(source).resolve(),Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Derived campaign must be separate from the original campaign')
    if output.exists():
        raise FileExistsError(output)
    if max_reference_seconds <= 0:
        raise ValueError('Reference replay deadline must be positive')
    began = monotonic()
    contracts = {a:read(source/a/'contract.json') for a in ['reference','balanced']}
    common = {a:read(source/a/'common_scenario.json.gz') for a in contracts}
    reference_result = read(source/'reference/result.json')
    if reference_result.get('contract') != contracts['reference'] or reference_result.get('full_day') is not True:
        raise ValueError('Original full-day REFERENCE is required and must remain untouched')
    if any(c.get('stage')!='full_day' or c.get('full_day') is not True or c.get('arm')!=a.upper()
           for a,c in contracts.items()):
        raise ValueError('Original two-arm full-day contracts are required')
    if digest(common['reference']) != digest(common['balanced']):
        raise ValueError('Original common physics, demand, profiles or roster differ')
    seed = contracts['balanced']['seed']
    capacity = contracts['balanced'].get('boarding_capacity_per_car',150)
    if seed != contracts['reference']['seed'] or capacity != contracts['reference'].get('boarding_capacity_per_car',150):
        raise ValueError('Paired seed or boarding capacity differs')
    derived = {a:partial_contract(c,steps) for a,c in contracts.items()}
    end = derived['balanced']['observation_end_second']
    if any(end >= c['cohort_end_second'] for c in contracts.values()):
        raise ValueError('Checkpoint must be strictly inside the original full-day cohort')
    checkpoint = source/'balanced/controller'/f'resume_{end:05}.json.gz'
    saved = read(checkpoint)
    if saved['second'] != end:
        raise ValueError('Requested checkpoint is not the selected decision boundary')
    decisions = [read(source/'balanced/controller'/f'decision_{s:05}.json')
                 for s in derived['balanced']['decision_times']]
    if [d.get('decision_second') for d in decisions] != derived['balanced']['decision_times']:
        raise ValueError('Decision journal does not match the selected checkpoint scope')
    for second in derived['balanced']['decision_times']:
        for prefix in ['facts','explanation']:
            if not (source/'balanced/explanations'/f'{prefix}_{second:05}.json').is_file():
                raise FileNotFoundError(f'Missing persisted {prefix} for selected decision')
    source_hashes = {p:h for c in contracts.values() for p,h in c.get('source_hashes',{}).items()}
    if not source_hashes or any(not Path(p).is_file() or sha(p)!=h for p,h in source_hashes.items()):
        raise ValueError('Original frozen sources no longer match their recorded hashes')
    for arm,c in contracts.items():
        for name,value in c['source_hashes'].items():
            if Path(name).suffix == '.py' and sha(source/arm/'execution_sources'/Path(name).name) != value:
                raise ValueError('Original execution snapshot does not match its source contract')
    preserved = {source/a/'contract.json':sha(source/a/'contract.json') for a in contracts}
    preserved.update({source/a/'common_scenario.json.gz':sha(source/a/'common_scenario.json.gz') for a in contracts})
    preserved.update({checkpoint:sha(checkpoint),source/'reference/result.json':sha(source/'reference/result.json')})
    balanced_runner,balanced_state,groups = branch_observed(common['balanced'],saved,seed=seed,capacity=capacity)
    data = common['reference']['data']
    reference_runner = GroupedPassengerSimulation(data,data['plans'][0]['plan_id'],groups,
        common['reference']['profiles'],boarding_seed=seed,boarding_capacity_per_car=capacity)
    reference_state = reference_runner.run(end_second=end,deadline=monotonic()+max_reference_seconds)['checkpoint']
    reference_runner._validate_checkpoint(reference_state,end)
    worlds = {'reference':(data,reference_runner,reference_state),
              'balanced':(saved['data'],balanced_runner,balanced_state)}
    calculated = {a:metrics(d,r,s,common[a]['roster'],derived[a]) for a,(d,r,s) in worlds.items()}
    output.mkdir(parents=True)
    atomic_json(output/'status.json',dict(stage='materializing_partial',full_day=False,stop_after_steps=steps))
    copied = {}
    try:
        for arm,(data,runner,state) in worlds.items():
            contract = derived[arm]
            contract.update(groups_digest=digest(groups),partial_provenance=dict(source_root=str(source),
                original_contract_sha256=preserved[source/arm/'contract.json'],
                original_reference_result_sha256=preserved[source/'reference/result.json'],
                checkpoint_sha256=preserved[checkpoint],checkpoint_relative_path=f'balanced/controller/{checkpoint.name}',
                original_full_day_completed=arm=='reference',cutoff_second=end,
                original_cohort_end_second=contracts[arm]['cohort_end_second'],
                boarding_seed_matched=True,physics_demand_profiles_roster_matched=True,
                future_groups_removed_using_audited_branch=True))
            folder = output/arm
            atomic_json(folder/'contract.json',contract)
            partial_common = copy.deepcopy(common[arm]);partial_common['groups']=groups
            atomic_json(folder/'common_scenario.json.gz',partial_common)
            for path in (source/arm/'execution_sources').glob('*.py'):
                copy_recorded(path,folder/'execution_sources'/path.name,copied)
            if arm=='balanced':
                for subfolder in ['controller','explanations']:
                    for path in (source/arm/subfolder).iterdir():
                        match = re.search(r'_(\d+)\.json(?:\.gz)?$',path.name)
                        if path.is_file() and match and int(match[1]) in contract['decision_times']:
                            copy_recorded(path,folder/subfolder/path.name,copied)
            record = save_run(folder,'RUN52_PARTIAL_'+arm.upper(),data,runner,state,common[arm]['roster'],
                data['headway_policy'],'GOAL52_PARTIAL',arm.upper(),'S1',seed,
                full_day=False,cohort_start=0,cohort_end=end,service_end_second=end)
            record['resources_scope'] = dict(start_second=0,end_second=end)
            record['metrics']['resources'] = calculated[arm]['resources']
            record['metrics']['policy_exposure'] = operational_exposure(state,runner.profiles,data['headway_policy'],0,end)
            atomic_json(folder/record['folder']/'summary.json',{k:v for k,v in record.items() if k!='folder'})
            result = dict(contract=contract,decisions=decisions if arm=='balanced' else [],run=record,
                metrics=calculated[arm],elapsed_seconds=monotonic()-began,full_day=False,
                sources_preserved=True,cash_saving_rub=None,lost_revenue_rub=None)
            atomic_json(folder/'result.json',result)
        for path in (source/'model_verification').rglob('*'):
            if path.is_file():
                copy_recorded(path,output/'model_verification'/path.relative_to(source/'model_verification'),copied)
        original_status = read(source/'status.json') if (source/'status.json').exists() else {}
        if original_status:
            copy_recorded(source/'status.json',output/'source_provenance/status.json',copied)
        if any(sha(p)!=h for p,h in preserved.items()) or any(sha(p)!=h for p,h in source_hashes.items()):
            raise ValueError('Original evidence or frozen sources changed during materialization')
        receipt = dict(stage='materialized_partial',full_day=False,stop_after_steps=steps,
            observation_end_second=end,expected_arms=['reference','balanced'],pending_arms=[],
            derived_from=str(source),original_full_day_completed=False,
            jobs={a:dict(exit_code=0,completion_kind='derived_reference_replay' if a=='reference' else 'checkpoint_materialization',
                source_exit_code=original_status.get('jobs',{}).get(a,{}).get('exit_code')) for a in contracts},
            censored_stocks={a:{k:state[k] for k in ['entered','completed','queued','approaching','onboard_count']}
                for a,(_,_,state) in worlds.items()},
            accepted_saving_rub=None,cash_saving_rub=None,annual_extrapolation_permitted=False,
            original_evidence_preserved={str(p):h for p,h in preserved.items()},copied_evidence_sha256=copied)
        atomic_json(output/'partial_manifest.json',receipt)
        atomic_json(output/'status.json',receipt)
        return receipt
    except Exception as error:
        atomic_json(output/'status.json',dict(stage='failed',full_day=False,error_type=type(error).__name__))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--max-reference-seconds',type=float,default=300)
    args = parser.parse_args()
    materialize(args.source,args.output,steps=args.steps,max_reference_seconds=args.max_reference_seconds)
