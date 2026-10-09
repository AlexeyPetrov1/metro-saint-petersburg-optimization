"""Audit saved paired dispatch journals; never run a passenger experiment."""

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import gzip
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(ledger):
    changed = [name for name, expected in ledger.items()
               if not Path(name).is_file() or sha(name) != expected]
    return dict(passed=not changed, checked_files=len(ledger), changed_files=changed)


def source_manifest(ledger):
    return dict(workspace_root=str(ROOT),files=ledger,
                scope='explicit immutable execution snapshot and saved inputs for this independent audit')


def guard_contract(point, guard, at):
    now=datetime.fromisoformat(at)
    meta=guard['source_versions']['residual_nowcast']
    if datetime.fromisoformat(meta['observed_latest_available_at'])>now:
        raise ValueError('observation arrived after guard cutoff')
    for row in meta['used_predictions']:
        start=datetime.fromisoformat(row['interval_start'])
        if any(datetime.fromisoformat(row[k])>start for k in
               ('forecast_issued_at','forecast_available_at','features_available_at')):
            raise ValueError('prior forecast was unavailable before observed quarter')
    keys=lambda r:(r['station_id'],r['interval_start'],r['interval_end'])
    future={keys(r):r for r in point['rows']if datetime.fromisoformat(r['interval_start'])>=now}
    actual={keys(r):r for r in guard['rows']}
    if future.keys()!=actual.keys():
        raise ValueError('guard future station/quarter keys changed')
    cap=Decimal(str(meta['parameters']['max_relative_change']))
    for key,row in actual.items():
        base=future[key]['predicted_count'];count=row['predicted_count']
        if count<base:raise ValueError('risk guard violates original point floor')
        upper=int((Decimal(base)*(1+cap)).to_integral_value(rounding=ROUND_FLOOR))
        if count>upper:raise ValueError('risk guard violates relative cap')
    return dict(passed=True,prior_prediction_count=len(meta['used_predictions']),future_row_count=len(actual),
                provenance='prior forecast publication precedes observed quarter; observations complete by decision',
                train_load_guarantee=False)


def selection_reasons(row, mean_budget, p95_budget):
    reasons=[]
    for field in ('physical_passed','passengers_checked','service_passed'):
        if row.get(field)is not True:reasons.append(field)
    trials=row.get('trials',[])
    if not trials:return reasons+['future_evidence_missing']
    for trial in trials:
        cohort=trial.get('future_cohort',{})
        exposure=trial.get('future_policy_summary',{})
        if cohort.get('complete')is not True:reasons.append('future_cohort_incomplete')
        if trial.get('headways_passed')is not True:reasons.append('future_headways')
        if trial.get('initial_obligations_complete')is not True or trial.get('initial_unboarded')not in(0,None):
            reasons.append('initial_passengers_unserved')
        if exposure.get('admissible')is not True:reasons.append('future_occupancy_policy')
        for summary in [exposure.get('global',{}),*exposure.get('windows',[])]:
            if summary.get('max_load',0)>160 or summary.get('hard_limit_car_seconds',0)>0:reasons.append('future_hard_car_limit')
            share=summary.get('exceptional_time_share')
            if share is not None and share>.01:reasons.append('future_overload_share')
        mean,p95=cohort.get('mean_wait_seconds'),cohort.get('p95_wait_seconds')
        if cohort.get('arrived')==0 and cohort.get('complete')is True:mean=p95=0
        if mean is None or mean_budget is None or mean>mean_budget:reasons.append('mean_wait_budget')
        if p95 is None or p95_budget is None or p95>p95_budget:reasons.append('p95_wait_budget')
    return sorted(set(reasons))


def residual_numeric_check(point, guard, observations, at):
    now=datetime.fromisoformat(at)
    meta=guard['source_versions']['residual_nowcast'];parameters=meta['parameters']
    cap=Decimal(str(parameters['max_relative_change']));gain=Decimal(str(parameters['gain']))
    observed={}
    for row in observations:
        if datetime.fromisoformat(row['interval_end'])>now:raise ValueError('future observation')
        key=(row['station_id'],row['interval_start'])
        if key in observed:raise ValueError('duplicate numeric observation')
        observed[key]=row
    predictions={(r['station_id'],r['interval_start']):r for r in meta['used_predictions']}
    if len(predictions)!=len(meta['used_predictions']):raise ValueError('duplicate prior forecast')
    multipliers={}
    for station,record in meta['station_adjustments'].items():
        rows=[r for (s,_),r in observed.items()if s==station]
        supported=[(r,predictions[station,r['interval_start']])for r in rows if(station,r['interval_start'])in predictions]
        actual=sum(r['entry_count']for r,_ in supported);expected=sum(p['predicted_count']for _,p in supported)
        status='no_recent_observations'if not rows else'no_prior_predictions'if not supported else'zero_predicted_volume'if expected==0 else'adjusted'
        multiplier=Decimal(1)if status!='adjusted'else max(1-cap,min(1+cap,1+gain*(Decimal(actual)/Decimal(expected)-1)))
        for key,value in dict(status=status,recent_intervals=len(rows),supported_intervals=len(supported),
            observed_recent_count=actual,predicted_recent_count=expected,multiplier=float(multiplier)).items():
            equal(record[key],value,f'numeric station {key}')
        multipliers[station]=multiplier
    future={(r['station_id'],r['interval_start']):r for r in point['rows']if datetime.fromisoformat(r['interval_start'])>=now}
    for row in guard['rows']:
        key=(row['station_id'],row['interval_start']);base=Decimal(future[key]['predicted_count'])
        lead=(datetime.fromisoformat(row['interval_start'])-now).total_seconds()
        decay=max(Decimal(0),1-Decimal(str(lead))/Decimal(parameters['decay_minutes']*60))
        corrected=int((base*(1+(multipliers[row['station_id']]-1)*decay)).to_integral_value(rounding=ROUND_HALF_UP))
        lower=int((base*(1-cap)).to_integral_value(rounding=ROUND_CEILING))
        upper=int((base*(1+cap)).to_integral_value(rounding=ROUND_FLOOR))
        expected=max(int(base),max(lower,min(upper,corrected)))
        equal(row['predicted_count'],expected,'guard numeric count')
    return dict(passed=True,station_count=len(multipliers),past_observation_count=len(observed),
                future_count=len(guard['rows']),method='independent integer ratio, clipping, decay and maximum reconstruction')


def reference_receipt(folder, ledger):
    folder = Path(folder)
    original = read(folder/'before.json')
    report = read(folder/'verification_report.json')
    if (report['status'] != 'passed' or any(original.get(name) != expected for name,expected in ledger.items())
            or not verify_hashes(ledger)['passed']):
        raise ValueError('reference journal proof or current bytes do not match')
    return dict(passed=True,proof=str(folder/'verification_report.json'),
                checked_files=len(ledger),prior_run_count=report['run_count'])


def trial_source_ledger(folder):
    folder = Path(folder)
    before, after = read(folder/'before.json'), read(folder/'after.json')
    if before != after:
        raise ValueError('original trial source ledger changed during execution')
    sources, extensions = dict(before), {}
    for path in sorted(folder.glob('*_extension.json')):
        extra = read(path)
        if extra['sources_before'] != extra['sources_after'] or extra['sources_preserved'] is not True:
            raise ValueError('extension sources changed during execution')
        for name, expected in extra['sources_before'].items():
            if name in sources and sources[name] != expected:
                raise ValueError('extension has contradictory source hashes')
            sources[name] = expected
        extensions[path.name] = extra
    return dict(sources=sources,extensions=extensions)


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'duplicate {key}')
    return result


def compare_groups(reference, candidate):
    if indexed(reference, 'group_id') != indexed(candidate, 'group_id'):
        raise ValueError('passenger group identity or fields changed')
    return dict(passed=True, group_count=len(reference),
                comparison='all fields by group_id; export order ignored')


def timing_pairs(data):
    tracks = indexed(data['tracks'], 'track_id')
    trains = indexed(data['trains'], 'train_id')
    jobs = {(r['train_id'], r['trip_id']): r for r in data['plans'][0]['trips']}
    result = {}
    for row in data['leg_timing']:
        tid, job, leg = row['train_id'], row['trip_id'], row['leg_index']
        key = (tid, job, leg)
        if key in result:
            raise ValueError('duplicate timing key')
        track = trains[tid]['location'] if job == '@initial' else jobs[tid, job]['route'][leg]
        dep = row['departure_second']
        arrival = dep + tracks[track]['travel_seconds'] + row.get('approach_holding_seconds', 0)
        result[key] = dict(track_id=track, departure_second=dep, arrival_second=arrival)
    return result


def compare_timing_obligations(reference, candidate, as_of):
    old, new = timing_pairs(reference), timing_pairs(candidate)
    committed = {key: row for key, row in old.items() if row['departure_second'] < as_of}
    for key, row in committed.items():
        if new.get(key) != row:
            raise ValueError(f'committed departure/arrival changed: {key}')
    extra_past = [key for key, row in new.items()
                  if row['departure_second'] < as_of and key not in committed]
    if extra_past:
        raise ValueError('new movement inserted into committed past')
    future = set(old) - set(committed)
    common = future & set(new)
    advances = lambda field: sum(new[k][field] < old[k][field] for k in common)
    return dict(passed=True, as_of_second=as_of, committed_pairs_checked=len(committed),
        current_pairs_checked=sum(r['arrival_second'] >= as_of for r in committed.values()),
        future_departure_advance_count=advances('departure_second'),
        future_arrival_advance_count=advances('arrival_second'),
        future_departure_change_count=sum(new[k]['departure_second'] != old[k]['departure_second'] for k in common),
        future_arrival_change_count=sum(new[k]['arrival_second'] != old[k]['arrival_second'] for k in common),
        future_removed_key_count=len(future-set(new)), future_added_key_count=len(set(new)-set(old)),
        future_changes_allowed=True, scope='strict dep/arrival/track preservation for departed legs; future planning can advance')


def exposure_summary(exposure):
    windows = exposure['windows']
    return dict(admissible=exposure['admissible'], global_exposure=exposure['global'],
        bad_window_count=sum(r['status'] == 'fail' for r in windows),
        bad_windows=[dict(start=r['start'], end=r['end']) for r in windows if r['status'] == 'fail'],
        empty_window_count=sum(r.get('passenger_car_seconds') == 0 for r in windows))


def waiting_cohort(groups, state, start, end):
    selected = {r['group_id']: r for r in groups if start <= r['turnstile_second'] < end}
    histogram, by_group = Counter(), Counter()
    for row in state['board_events']:
        group = selected.get(row['group_id'])
        if group is None:
            continue
        wait = row['second'] - group['platform_second']
        if wait < 0 or wait != row['wait_seconds']:
            raise ValueError('boarding waiting identity failed')
        by_group[row['group_id']] += row['count']
        histogram[wait] += row['count']
    if any(by_group[key] > row['count'] for key, row in selected.items()):
        raise ValueError('cohort boarded more passengers than arrived')
    arrived, boarded = sum(r['count'] for r in selected.values()), sum(histogram.values())
    complete = state['second'] >= end and arrived == boarded
    mean = p95 = None
    if complete and boarded:
        mean = sum(wait*count for wait, count in histogram.items()) / boarded
        rank, seen = (95*boarded+99)//100, 0
        for wait, count in sorted(histogram.items()):
            seen += count
            if seen >= rank:
                p95 = wait
                break
    return dict(start_second=start, end_second=end, arrived=arrived, boarded=boarded,
                unboarded=arrived-boarded, complete=complete,
                mean_wait_seconds=mean, p95_wait_seconds=p95)


def journey_context(data, motion, groups):
    jobs = {(r['train_id'],r['trip_id']):r for r in data['plans'][0]['trips']}
    next_leg, enriched = Counter(), []
    for event in motion['events']:
        row = dict(event)
        if row['event_type'] == 'departure':
            key = (row['train_id'],row['trip_id'])
            job, leg = jobs[key], next_leg[key]
            if job['route'][leg] != row['location']:
                raise ValueError('saved departure route mismatch')
            row.update(remaining_route=job['route'][leg:],passenger_service=job['passenger'])
            next_leg[key] += 1
        enriched.append(row)
    return SimpleNamespace(by_id=indexed(groups,'group_id'),
        tracks=indexed(data['tracks'],'track_id'),motion_by_id=indexed(enriched,'event_id'))


def strict_pareto(baseline, candidate, bounds, physical):
    if physical is not True or any(candidate.get(k) is not True for k in (
            'actual_service_admissible', 'all_service_arrivals_completed',
            'movement_obligations_closed', 'terminal_resources_passed')):
        return False
    values = [baseline.get('mean_wait_seconds'), baseline.get('p95_wait_seconds'),
              candidate.get('mean_wait_seconds'), candidate.get('p95_wait_seconds'),
              bounds.get('saving_train_km_lower_bound'), bounds.get('saving_to1_resource_rub_lower_bound')]
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        return False
    mean, p95, new_mean, new_p95, km, rub = values
    return new_mean <= mean and new_p95 <= p95 and km > 0 and rub > 0


def equal(actual, expected, label):
    if isinstance(expected, bool) or expected is None:
        good = type(actual) is type(expected) and actual == expected
    elif type(expected) in (int, float):
        good = type(actual) in (int, float) and isfinite(actual) and abs(actual-expected) <= 1e-8
    else:
        good = actual == expected
    if not good:
        raise ValueError(f'{label}: recorded {actual!r}, rebuilt {expected!r}')


def audit_feedback(experiment, output):
    experiment,output=Path(experiment).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    common=read(experiment/'shared_contract.json')
    result_path=next(experiment.glob('mpc_*_result.json'))
    result=read(result_path);arm=result['strategy'];folder=experiment/'controller'/arm
    shared=read(experiment/'shared_scenario.json.gz')
    snapshot=Path(common['snapshot']);sys.dont_write_bytecode=True;sys.path.insert(0,str(snapshot))
    from goal43_verify import verify,RUN_FILES
    from simulation import digest
    from rolling import operational_exposure,operational_headways
    from metro44_assignment_plan import terminal_resources
    from metro44_journey import cohort_journey
    from metro44_window import paired_window_bounds
    ledger={**common['source_hashes'],**common['input_hashes']}
    extension=read(experiment/'feedback_extension.json')
    equal(extension['sources_before'],extension['sources_after'],'feedback source before/after')
    ledger.update(extension['sources_before'])
    if not verify_hashes(ledger)['passed']:raise ValueError('feedback source/input provenance changed')
    protected=dict(ledger)
    for p in experiment.rglob('*'):
        if p.is_file():protected[str(p)]=sha(p)
    for p in (Path(__file__).resolve(),ROOT/'tests/test_metro45_independent_audit.py'):
        protected[str(p)]=sha(p)
    write(output/'before.json',protected);write(output/'sourcebaseline_manifest.json',source_manifest(protected))
    record=dict(result['run'],folder=str(experiment/result['run']['folder']))
    write(output/'runs.json',[record]);journal=verify(output)
    run=Path(record['folder']);data,motion,state,groups,profiles=[read(run/n)for n in RUN_FILES[:5]]
    compare_groups(shared['groups'],groups);equal(shared['profiles'],profiles,'profiles')
    equal(digest(groups),common['groups_digest'],'group digest')
    equal(digest(shared['state']),common['checkpoint_digest'],'initial checkpoint digest')
    times=common['decision_times'];decisions=result['decisions']
    equal([d['decision_second']for d in decisions],times,'consecutive decision clocks')
    equal(times,list(range(3600,3600+900*len(times),900)),'quarter-hour decision sequence')
    equal(len(times)==20,common['full_twenty_slots'],'full twenty slots flag')
    original_data=shared['data'];prior=original_data;checks=[]
    archive_root=ROOT/'data/processed/metro_daily_experiments_20261009/forecasts'
    archives={digest(read(p)):read(p)for p in archive_root.glob('ML_*.json')}
    histories={str(archive_root/f'ML_{h:02}.json'):sha(archive_root/f'ML_{h:02}.json')for h in (7,8)}
    for d in decisions:
        second=d['decision_second'];at=d['decision_time']
        resume=read(folder/f'resume_{second:05}.json.gz');plan=read(folder/f'plan_{second:05}.json.gz')
        equal(resume['data'],plan,'resume plan');equal(resume['state']['second'],second,'checkpoint second')
        equal(resume['data_digest'],digest(plan),'plan digest');equal(resume['state_digest'],digest(resume['state']),'state digest')
        compare_timing_obligations(prior,plan,second)
        equal(d['executed_prefix_digest'],digest([e for e in motion['events']if e['second']<second]),'executed final prefix')
        for key in ('board_events','load_events','queue_events','queue_samples'):
            equal(resume['state'][key],[e for e in state[key]if e['second']<second],f'{key} continuous prefix')
        prior=plan
        keep=next(r for r in d['candidate_table']if r['candidate_id']=='KEEP')
        cohorts=[t['future_cohort']for t in keep['trials']]
        mean=max(t['mean_wait_seconds']or 0 for t in cohorts);p95=max(t['p95_wait_seconds']or 0 for t in cohorts)
        reasons={r['candidate_id']:selection_reasons(r,mean,p95)for r in d['candidate_table']}
        selected=d['selection']['selected_candidate_id']
        if selected is not None and reasons[selected]:raise ValueError('selected candidate violated future gates or waiting budgets')
        equal(d['applied_candidate_id'],selected or 'KEEP','applied selected candidate')
        equal(d['feedback45']['feedback_checkpoint_second'],second,'feedback cutoff')
        equal(d['feedback45']['future_actual_demand_used'],False,'future actual demand')
        raw=read(folder/f'forecast_{second:05}.json')
        for key in ('issued_at','as_of','available_at','features_available_at'):
            if datetime.fromisoformat(raw[key])>datetime.fromisoformat(at):raise ValueError('future forecast publication')
        guard=None
        guard_path=folder/f'guard_{second:05}.json'
        if guard_path.exists():
            envelope=read(guard_path);guard=guard_contract(raw,envelope,at)
            meta=envelope['source_versions']['residual_nowcast']
            equal(meta['original_bundle_digest'],digest(raw),'residual original forecast')
            equal(envelope['source_versions']['risk_envelope']['original_bundle_digest'],digest(raw),'risk original forecast')
            for used in meta['used_predictions']:
                source=archives.get(used['forecast_bundle_digest'])
                if source is None:raise ValueError('prior forecast digest not found among preserved archives')
                prediction=next(r for r in source['rows']if r['station_id']==used['station_id']and r['interval_start']==used['interval_start'])
                equal(prediction['predicted_count'],used['predicted_count'],'prior forecast count')
            equal(d['forecast_digest'],digest(envelope),'decision guard digest')
        else:equal(d['forecast_digest'],digest(raw),'decision point digest')
        archives[digest(raw)]=raw
        if d['pipeline_seconds']>480 or d['local_recommendation_within_480']is not True:
            raise ValueError('complete pilot violated local computation deadline')
        checks.append(dict(second=second,selected=selected,applied=d['applied_candidate_id'],
            candidate_rejection_reasons=reasons,causal_guard=guard,pipeline_seconds=d['pipeline_seconds'],
            real_inference_seconds=d['actual_inference_seconds'],nested_catalog_inference_flag=d['forecast_inference_in_timing']))
    terminal=terminal_resources(data,motion);headways=operational_headways(data,motion,3600,22200)
    full=operational_exposure(state,profiles,data['headway_policy'],0,state['second'])
    control=operational_exposure(state,profiles,data['headway_policy'],3600,21600)
    waiting=waiting_cohort(groups,state,3600,21000)
    equal(waiting['mean_wait_seconds'],result['metrics']['mean_wait_seconds'],'mean wait')
    equal(waiting['p95_wait_seconds'],result['metrics']['p95_wait_seconds'],'p95 wait')
    journey=cohort_journey(journey_context(data,motion,groups),state,3600,21000)
    reference_motion=read(Path(common['source'])/'motion.json')
    bounds=paired_window_bounds(original_data,reference_motion,data,motion,3600,21600)
    preservation=verify_hashes(protected)
    if not preservation['passed']:raise ValueError('feedback audit inputs changed')
    receipt=dict(status='passed',journal_run_count=journal['run_count'],decision_count=len(checks),decisions=checks,
        source_guard=verify_hashes(ledger),inputs_preserved=preservation,full_episode=exposure_summary(full),
        controlled_window=exposure_summary(control),mature_waiting=waiting,mature_journey=journey,
        terminal_resources=terminal,planning_headways_passed=headways['passed'],movement_closed=motion['obligations_closed'],
        train_km=result['metrics']['train_km'],window_bounds=bounds,
        extra_history_current_hashes=histories,extra_history_not_in_startup_input_hashes=True,
        extra_history_semantic_digests_verified=True,
        timing_disclosure='nested catalog flag covers catalog-only time; outer pipeline includes actual inference',
        observation_availability='assumed complete intervals available at end; original publication timestamps absent',
        independent_holdout=False,full_day=False,reference='assumed constant53 finite cohort',release_authorized=False,
        cash_saving_rub=None,lost_revenue_rub=None,net_profit_effect_rub=None)
    write(output/'after.json',{p:sha(p)for p in protected});write(output/'independent_audit.json',receipt)
    print(json.dumps(dict(status='passed',decisions=len(checks),mean_wait=waiting['mean_wait_seconds'],
        p95_wait=waiting['p95_wait_seconds'],train_km=receipt['train_km'])),flush=True)
    return receipt


def audit(experiment, output, snapshot, reference_audit=None):
    experiment, output, snapshot = map(lambda p: Path(p).resolve(), (experiment, output, snapshot))
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot))
    from goal43_verify import verify, RUN_FILES
    from metro44_assignment_plan import terminal_resources
    from metro44_journey import cohort_journey
    from metro44_window import paired_window_bounds
    from rolling import operational_exposure, operational_headways

    candidate = read(experiment/'result.json')
    contract = read(experiment/'contract.json')
    trial_ledger = trial_source_ledger(experiment)
    equal(candidate['contract'], contract, 'contract')
    scenario = contract['scenario']
    closed = ROOT/'data/processed/metro44_continuation_6h_closed_20261009'
    robust = ROOT/'data/processed/metro44_profile_6h_robustness_20261009'
    ref_result = closed/'keep_result.json' if scenario == 'S1' else robust/f'keep_{scenario.lower()}_result.json'
    reference = read(ref_result)
    bridge_root = ROOT/'data/processed/metro44_bridge_off160_actual_6h_20261009_v2'
    bridge_path = bridge_root/f'{scenario.lower()}_result.json'
    bridge = read(bridge_path)
    rows = [('KEEP', reference, ref_result.parent), ('BRIDGE160', bridge, bridge_root),
            ('CANDIDATE', candidate, experiment)]
    folders = {name: root/item['run']['folder'] for name, item, root in rows}
    files = [p for folder in folders.values() for p in (folder/name for name in RUN_FILES)]
    files += [ref_result, bridge_path] + [experiment/name for name in ('result.json','contract.json','before.json','after.json')]
    snapshot_manifest = read(snapshot/'execution_manifest.json')
    files += [snapshot/name for name in snapshot_manifest['source_hashes']]
    files += [Path(name) for name in trial_ledger['sources']]
    files += [experiment/name for name in trial_ledger['extensions']]
    files += [Path(__file__).resolve(),ROOT/'tests/test_metro45_independent_audit.py']
    protected = {str(p): sha(p) for p in files}
    write(output/'before.json', protected)
    write(output/'sourcebaseline_manifest.json',source_manifest(protected))
    if not candidate['inputs_and_sources_preserved']:
        raise ValueError('original trial source ledger changed during execution')
    provenance = verify_hashes(trial_ledger['sources'])
    records = [dict(item['run'], run_id=name, folder=str(folders[name])) for name,item,_ in rows]
    reuse = None
    if reference_audit is not None:
        ref_files = [folders[arm]/name for arm in ('KEEP','BRIDGE160') for name in RUN_FILES]
        ref_files += [snapshot/name for name in snapshot_manifest['source_hashes']]
        reuse = reference_receipt(reference_audit,{str(p):protected[str(p)] for p in ref_files})
        records = [r for r in records if r['run_id']=='CANDIDATE']
    write(output/'runs.json', records)
    journal = verify(output)
    loaded = {name: {key: read(folder/key) for key in RUN_FILES} for name,folder in folders.items()}
    ref = loaded['KEEP']
    report = dict(status='passed', experiment=str(experiment), snapshot=str(snapshot),
                  journal_run_count=journal['run_count'], provenance=provenance,
                  reference_journal_reuse=reuse,
                  run_source_before_after_equal=True, arms={}, comparisons={},
                  independent_holdout=False, posthoc_development=True,
                  simulated_line_id=contract['simulated_line_id'], requested_line_id=contract['requested_line_id'],
                  blue_transfer_completed=contract['blue_transfer_completed'], release_authorized=False,
                  cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None)
    report['extensions'] = trial_ledger['extensions']
    start, end, now = contract['cohort_start_second'], contract['cohort_end_second'], contract['change_second']
    mature_end = end-600
    for name,item,_ in rows:
        saved = loaded[name]
        data, motion, state, groups, profiles = (saved[k] for k in RUN_FILES[:5])
        compare_groups(ref['groups.json.gz'], groups)
        equal(profiles, ref['profiles.json'], f'{name} car profiles')
        equal(data['period_start'], ref['movement_input.json']['period_start'], 'period start')
        equal(data['period_end'], ref['movement_input.json']['period_end'], 'closure period')
        equal(state['second'], contract['closure_end_second'], 'saved closure second')
        timing = compare_timing_obligations(ref['movement_input.json'], data, now)
        for key in ('board_events','load_events','queue_events','queue_samples'):
            equal([r for r in state[key] if r['second'] < now],
                  [r for r in ref['passenger_checkpoint.json.gz'][key] if r['second'] < now], f'{name} {key} prefix')
        equal([r for r in motion['events'] if r['second'] < now],
              [r for r in ref['motion.json']['events'] if r['second'] < now], f'{name} motion prefix')
        terminal = terminal_resources(data, motion)
        headways = operational_headways(data, motion, start, contract['passenger_departure_end_second'])
        full = operational_exposure(state, profiles, data['headway_policy'], 0, state['second'])
        control = operational_exposure(state, profiles, data['headway_policy'], start, end)
        wait = waiting_cohort(groups, state, start, mature_end)
        all_wait = waiting_cohort(groups, state, start, end)
        runner = journey_context(data,motion,groups)
        journey = cohort_journey(runner, state, start, mature_end)
        metrics = item['metrics']
        equal(wait['mean_wait_seconds'], metrics['mean_wait_seconds'], f'{name} mean wait')
        equal(wait['p95_wait_seconds'], metrics['p95_wait_seconds'], f'{name} p95 wait')
        equal(wait['arrived'], metrics['mature_cohort']['arrived'], f'{name} mature cohort')
        equal(all_wait['arrived'], metrics['cohort']['arrived'], f'{name} full controlled cohort')
        equal(control['admissible'], metrics['policy_exposure']['admissible'], f'{name} controlled exposure')
        physical = terminal['passed'] and headways['passed'] and motion['checks']['supported_constraints_status']=='passed' and motion['obligations_closed']
        if name == 'CANDIDATE':
            equal(physical, item['physical_passed'], 'candidate physics')
            equal(full['admissible'], item['whole_episode_service_admissible'], 'candidate full exposure')
            for key,value in journey.items():
                equal(item['journey']['CANDIDATE'][key], value, f'candidate journey {key}')
        report['arms'][name] = dict(physical_passed=physical, terminal_resources=terminal,
            timing_obligations=timing, full_episode=exposure_summary(full),
            controlled_window=exposure_summary(control), mature_waiting=wait,
            all_controlled_waiting=all_wait, mature_journey=journey,
            whole_episode_train_km=metrics['train_km'], fixed_roster_size=metrics['fixed_fleet_size'],
            whole_episode_km_per_fixed_train=metrics['mean_km_per_fixed_train'],
            headways=headways, all_groups_completed=state['entered']==state['completed'] and not any(state[k] for k in ('queued','approaching','onboard_count')))
    for ref_name in ('KEEP','BRIDGE160'):
        ref_saved, cand_saved = loaded[ref_name], loaded['CANDIDATE']
        bounds = paired_window_bounds(ref_saved['movement_input.json'],ref_saved['motion.json'],
            cand_saved['movement_input.json'],cand_saved['motion.json'],start,end)
        ref_metrics = next(item['metrics'] for name,item,_ in rows if name==ref_name)
        metrics = dict(candidate['metrics'], terminal_resources_passed=report['arms']['CANDIDATE']['terminal_resources']['passed'])
        strict = strict_pareto(ref_metrics,metrics,bounds,report['arms']['CANDIDATE']['physical_passed'])
        report['comparisons'][ref_name] = dict(bounds=bounds, strict_wait_km_improvement=strict,
            mean_wait_delta_seconds=metrics['mean_wait_seconds']-ref_metrics['mean_wait_seconds'],
            p95_wait_delta_seconds=metrics['p95_wait_seconds']-ref_metrics['p95_wait_seconds'])
        if ref_name=='KEEP':
            for key in ('saving_train_km_lower_bound','saving_train_km_upper_bound','saving_to1_resource_rub_lower_bound'):
                equal(candidate['window_bounds'][key], bounds[key], f'window bound {key}')
            equal(strict,candidate['strict_controlled_improvement'],'strict improvement')
    preservation = verify_hashes(protected)
    if not preservation['passed']:
        raise ValueError('saved experiment or frozen sources changed during audit')
    report['audit_inputs_preserved'] = preservation
    report['scope'] = dict(control_start=start,control_end=end,mature_end=mature_end,closure_end=contract['closure_end_second'],
        reference='assumed constant53 demo; varying operational day is a separate experiment',
        forecast=contract.get('decision'), observation='station entries; OD is assumed, not observed')
    write(output/'after.json', {name:sha(name) for name in protected})
    write(output/'independent_audit.json',report)
    print(json.dumps(dict(status=report['status'],run_count=journal['run_count'],
        provenance_passed=provenance['passed'], comparisons={k:{n:v for n,v in row.items() if n!='bounds'}
        for k,row in report['comparisons'].items()})),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--snapshot',default=str(ROOT/'data/processed/metro44_bridge_execution_v5'))
    parser.add_argument('--reference-audit')
    parser.add_argument('--feedback',action='store_true')
    args = parser.parse_args()
    if args.feedback:audit_feedback(args.experiment,args.output)
    else:audit(args.experiment,args.output,args.snapshot,args.reference_audit)
