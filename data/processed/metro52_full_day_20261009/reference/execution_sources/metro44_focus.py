"""Paired continuous replay of the saved peak53 case, without a new warmup."""

from __future__ import annotations

import argparse
import copy
import gzip
import inspect
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from goal43_replay import save_run
from metro_daily_run import read_forecast
from metro_ml_compare import _metrics
from metro44_run import (DEFAULT_SOURCE, ROOT, SOURCES, _calculate, _hash, archive_sources,
                         bounded_decision, forecast_source, public_decision, selected_plan,
                         write_gzip_once, write_json_once)
from passenger_groups import GroupedPassengerSimulation
from simulation import digest


DEFAULT_CASE = ROOT / 'data/processed/metro_ml_comparison/cases/2026-09-30_peak_S1'
DEFAULT_OUTPUT = ROOT / 'data/processed/metro44_20261009_focus'
ARMS = ('KEEP', 'STANDARD', 'MPC_BASELINE_FORECAST', 'MPC_ML', 'MPC_ORACLE')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_case(source_case):
    case = Path(source_case).resolve()
    manifest = _read(case / 'experiment_manifest.json')['contract']
    original = _read(case / 'keep_result.json')
    run = (case.parent.parent / original['run_folder']).resolve()
    if case.parent.parent not in run.parents or manifest['od'] != 'S1':
        raise ValueError('source must be the original paired S1 KEEP case')
    shared = dict(data=_read(run / 'movement_input.json'), profiles=_read(run / 'profiles.json'),
                  groups=_read(case / 'actual_groups.json'), roster=_read(case / 'fleet_registry.json'),
                  checkpoint=_read(case / 'warmup_checkpoint.json'), policy=_read(case / 'policy.json'))
    for key, expected in (('data', 'movement_digest'), ('groups', 'actual_groups_digest'),
                          ('checkpoint', 'initial_checkpoint_digest'), ('policy', 'policy_digest')):
        if digest(shared[key]) != manifest[expected]:
            raise ValueError(f'original {expected} changed')
    first, end = shared['checkpoint']['second'], manifest['service_seconds']
    if (type(first) is not int or type(end) is not int or not 0 < first < end
            or first % 900 or end % 900 or len(shared['roster']) != 68):
        raise ValueError('original checkpoint, service boundary, or fixed fleet is invalid')
    times = list(range(first, end, 900))
    clock = datetime.fromisoformat(shared['data']['period_start'])
    at = lambda second: (clock + timedelta(seconds=second)).isoformat()
    paths = [case / name for name in ('experiment_manifest.json', 'keep_result.json', 'actual_groups.json',
             'warmup_checkpoint.json', 'fleet_registry.json', 'policy.json',
             'median_forecasts.json', 'ml_forecasts.json')]
    paths += [run / 'movement_input.json', run / 'profiles.json']
    contract = dict(source_case=str(case), original_run_folder=str(run), case_id=manifest['case_id'],
        initial_count=manifest['initial_count'], fixed_fleet_size=68,
        groups_digest=digest(shared['groups']), checkpoint_digest=digest(shared['checkpoint']),
        input_hashes={str(path): _hash(path) for path in paths}, decision_times=times,
        cohort_start_second=first, cohort_end_second=end, service_end_second=end,
        first_decision_at=at(first), last_decision_at=at(times[-1]), service_end_at=at(end),
        boundary_end_is_operational_closure=False, economic_claim_requires_common_continuation=True,
        scope='original four-hour input cohort; unchanged common warmup and closed movement tail',
        actual_simulated_line_id=1, requested_line_id=2, blue_line_transfer_completed=False)
    return shared, contract


def cohort_forecast(bundle, end_at):
    """Limit this diagnostic cohort; midnight service closure is not inferred."""
    result = copy.deepcopy(bundle)
    end = datetime.fromisoformat(end_at)
    for row in result['rows']:
        if datetime.fromisoformat(row['interval_start']) >= end:
            row['predicted_count'] = 0
    result['source_versions']['finite_cohort_boundary'] = dict(at=end_at,
        boundary_end_is_operational_closure=False, observed_truth_extended=False,
        economic_claim_requires_common_continuation=True,
        scope='zero means outside this finite experimental cohort; real metro remains in service')
    return result


def prepare_forecasts(output, source, shared, contract):
    ids = [s['station_id'] for s in shared['data']['stations'] if s['kind'] == 'line']
    clock = datetime.fromisoformat(shared['data']['period_start'])
    issue_times = {}
    for second in contract['decision_times']:
        at = clock + timedelta(seconds=second)
        issue_times.setdefault(at.hour, at.isoformat())
    hashes, derived, issued = {}, {}, []
    for mode in ('MEDIAN', 'ML', 'ORACLE'):
        for hour, at in issue_times.items():
            name = f'{mode}_{hour:02}.json'
            path = Path(source) / 'forecasts' / name
            hashes[str(path.resolve())] = _hash(path)
            bundle = read_forecast(source, mode, ids, at)
            issued.append(bundle['issued_at'])
            if mode != 'ORACLE' and hour == min(issue_times):
                originals = _read(Path(contract['source_case']) / f'{mode.lower()}_forecasts.json')
                numeric = lambda b: sorted((r['station_id'], r['interval_start'], r['interval_end'],
                                            r['predicted_count']) for r in b['rows'])
                if not originals or numeric(originals[0]) != numeric(bundle):
                    raise ValueError('cached first forecast differs from original peak case')
            clipped = cohort_forecast(bundle, contract['service_end_at'])
            target = Path(output) / 'forecasts' / name
            if target.exists():
                if _read(target) != clipped:
                    raise ValueError('derived finite-cohort forecast changed')
            else:
                write_json_once(target, clipped)
            derived[name] = _hash(target)
    return dict(original_forecast_hashes=hashes, derived_forecast_hashes=derived,
                first_issue_at=min(issued), first_decision_at=min(issue_times.values()),
                actual_inference_in_timing=False,
                scope='cached frozen predictions with explicit finite-cohort boundary')


def source_hashes(calculate=None, extra_sources=()):
    paths = {path.name for path in ROOT.glob('*.py')} | set(SOURCES) | set(extra_sources)
    if calculate is not None:
        path = inspect.getsourcefile(calculate)
        if path is not None and ROOT in Path(path).resolve().parents:
            paths.add(str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/'))
    return {name: _hash(ROOT / name) for name in sorted(paths)}


def execution_prefix(motion, second):
    return digest([row for row in motion['events'] if row['second'] < second])


def load_resume(folder, identity, times):
    paths = sorted(Path(folder).glob('resume_*.json.gz'))
    if not paths:
        return None
    with gzip.open(paths[-1], 'rt', encoding='utf-8') as stream:
        record = json.load(stream)
    if record.get('contract_identity') != identity:
        raise ValueError('resume source/input/mode contract changed')
    index = record['next_index']
    if (type(index) is not int or not 1 <= index <= len(times)
            or record['second'] != times[index - 1] or record['state']['second'] != record['second']):
        raise ValueError('resume clock is inconsistent')
    expected = [Path(folder) / f'resume_{second:05}.json.gz' for second in times[:index]]
    if paths != expected:
        raise ValueError('resume executed prefix is incomplete')
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            saved = json.load(stream)
        if saved.get('contract_identity') != identity:
            raise ValueError('mixed resume contracts')
        for key in ('data', 'state'):
            if f'{key}_digest' in saved and digest(saved[key]) != saved[f'{key}_digest']:
                raise ValueError(f'resume {key} digest mismatch')
    return record


def run_focus(output, strategy, *, source_case=DEFAULT_CASE, forecast_cache=DEFAULT_SOURCE,
              pilot_steps=None, resume=False, calculate=None, extra_sources=(),
              budget_seconds=470, candidate_limit=3):
    if strategy not in ARMS and (calculate is None or not re.fullmatch('[A-Z][A-Z0-9_]*', strategy)):
        raise ValueError('unknown focused strategy')
    if not math.isfinite(budget_seconds) or not 1 <= budget_seconds <= 480:
        raise ValueError('decision budget must be from 1 to 480 seconds')
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 3:
        raise ValueError('candidate limit must be from 1 to 3')
    output, source_case = Path(output).resolve(), Path(source_case).resolve()
    original_root = source_case.parent.parent
    if original_root == output or original_root in output.parents:
        raise ValueError('output must be separate from source case')
    shared, paired = prepare_case(source_case)
    times = paired['decision_times']
    if pilot_steps is not None:
        if type(pilot_steps) is not int or not 1 <= pilot_steps <= len(times):
            raise ValueError('invalid focused pilot steps')
        times = times[:pilot_steps]
    suffix = '' if pilot_steps is None else f'_pilot_{pilot_steps}'
    folder = output / 'controller' / f'{strategy}{suffix}'
    frozen = source_hashes(calculate, extra_sources)
    archive_sources(output, frozen)
    shared_path = output / 'shared_scenario.json'
    if shared_path.exists():
        if _read(shared_path) != paired:
            raise ValueError('paired source case changed')
    else:
        write_json_once(shared_path, paired)
    forecast_contract = {}
    source = output
    if strategy != 'KEEP':
        if forecast_cache is None and calculate is None:
            raise ValueError('cached forecasts required for default calculator')
        if forecast_cache is not None:
            forecast_contract = prepare_forecasts(output, forecast_cache, shared, paired)
    contract = dict(strategy=strategy, pilot_steps=pilot_steps, source_hashes=frozen,
        shared_identity=digest(paired), forecasts=forecast_contract, decision_times=times,
        budget_seconds=budget_seconds, candidate_limit=candidate_limit,
        calculator='metro44_run._calculate' if calculate is None else f'{calculate.__module__}.{calculate.__qualname__}',
        boundary_end_is_operational_closure=False, economic_claim_requires_common_continuation=True,
        passenger_departure_end_second=paired['service_end_second'], clearance_seconds=0)
    identity = digest(contract)
    path = folder / 'contract.json'
    if path.exists():
        if not resume:
            raise ValueError('focused arm already exists; use resume or a separate output')
        if _read(path) != contract:
            raise ValueError('resume source/input/mode contract changed')
    else:
        write_json_once(path, contract)
    final_path = output / f'{strategy.lower()}_result.json' if pilot_steps is None else folder / 'pilot_result.json'
    if final_path.exists():
        result = _read(final_path)
        if not resume or result.get('contract_identity') != identity:
            raise ValueError('refusing to overwrite completed focused arm')
        return result
    data, state = copy.deepcopy(shared['data']), copy.deepcopy(shared['checkpoint'])
    previous, index, decisions = state['second'], 0, []
    prior = load_resume(folder, identity, times) if resume else None
    if prior is not None:
        data, state, previous, index = prior['data'], prior['state'], prior['second'], prior['next_index']
        for second in times[:index]:
            with gzip.open(folder / f'resume_{second:05}.json.gz', 'rt', encoding='utf-8') as stream:
                saved = json.load(stream)
            journal = folder / f'decision_{second:05}.json'
            if journal.exists():
                if _read(journal) != saved['decision']:
                    raise ValueError('resume decision journal differs from checkpoint')
            else:
                write_json_once(journal, saved['decision'])
            decisions.append(saved['decision'])
    roster, profiles = shared['roster'], shared['profiles']
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], shared['groups'], profiles)
    warmup_digest = execution_prefix(runner.motion, paired['cohort_start_second'])
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    clock = datetime.fromisoformat(data['period_start'])
    began = time.monotonic()
    compute = _calculate if calculate is None else calculate
    for position in range(index, len(times)):
        second = times[position]
        if second > previous:
            state = runner.run(start_second=previous, end_second=second, checkpoint=state)['checkpoint']
        at = (clock + timedelta(seconds=second)).isoformat()
        prefix = execution_prefix(runner.motion, second)
        if strategy == 'KEEP':
            receipt = dict(completed=True, elapsed_seconds=0)
            decision = dict(selection=dict(status='fixed_reference', selected_candidate_id=None,
                mitigation_candidate_id=None, fallback_candidate_id='KEEP'), candidate_table=[])
            candidates = [dict(candidate_id='KEEP', _data=data)]
        else:
            receipt = bounded_decision(lambda deadline, cancelled: compute(data, runner, state, profiles, roster,
                source, strategy, ids, second, paired['service_end_second'], candidate_limit, deadline, cancelled),
                budget_seconds)
            if receipt['completed']:
                decision, candidates = receipt['result']['decision'], receipt['result']['candidates']
            else:
                decision = dict(selection=receipt['selection'], candidate_table=[], search_complete=False)
                candidates = [dict(candidate_id='KEEP', _data=data)]
        changed, applied = selected_plan(decision['selection'], candidates, decision['candidate_table'])
        if changed is not data:
            new = GroupedPassengerSimulation(changed, changed['plans'][0]['plan_id'], shared['groups'], profiles)
            if execution_prefix(new.motion, second) != prefix:
                raise ValueError('candidate rewrites the executed movement prefix')
            state = new.branch_checkpoint(state, runner)
            runner, data = new, changed
        if execution_prefix(runner.motion, paired['cohort_start_second']) != warmup_digest:
            raise ValueError('shared warmup movement changed')
        decision.update(decision_time=at, decision_second=second, applied_candidate_id=applied,
            executed_prefix_digest=prefix, groups_digest=paired['groups_digest'],
            initial_checkpoint_digest=paired['checkpoint_digest'],
            elapsed_seconds_including_plan_generation=receipt['elapsed_seconds'],
            notification_deadline_passed=receipt['completed'], late_work_discarded=not receipt['completed'],
            forecast_inference_in_timing=False,
            mitigation_applied_as_research=decision['selection'].get('selected_candidate_id') is None and applied != 'KEEP')
        saved = public_decision(decision)
        decisions.append(saved)
        write_gzip_once(folder / f'resume_{second:05}.json.gz', dict(contract_identity=identity,
            next_index=position + 1, second=second, state=state, data=data, decision=saved,
            state_digest=digest(state), data_digest=digest(data), executed_prefix_digest=prefix))
        write_json_once(folder / f'decision_{second:05}.json', saved)
        write_gzip_once(folder / f'plan_{second:05}.json.gz', data)
        previous = second
        print(json.dumps(dict(stage='goal44_focus_control', strategy=strategy, index=position + 1,
            decisions=len(times), at=at, applied=applied, seconds=receipt['elapsed_seconds'])), flush=True)
    result = dict(strategy=strategy, contract_identity=identity, source_hashes=frozen,
        groups_digest=paired['groups_digest'], checkpoint_digest=paired['checkpoint_digest'],
        decisions=decisions, checkpoint_second=state['second'], warmup_movement_digest=warmup_digest,
        boundary_end_is_operational_closure=False, economic_claim_requires_common_continuation=True,
        economic_improvement_claimed=False, release_authorized=False, forecast_inference_in_timing=False,
        maximum_decision_seconds=max((d['elapsed_seconds_including_plan_generation'] for d in decisions), default=0),
        actual_simulated_line_id=1, requested_line_id=2, full_day=False,
        cohort_start_second=paired['cohort_start_second'], admission_end_second=paired['cohort_end_second'],
        passenger_departure_end_second=paired['service_end_second'])
    if pilot_steps is None:
        state = runner.run(start_second=previous, checkpoint=state)['checkpoint']
        run_id = f'{paired["case_id"]}_goal44_focus_{strategy}'
        result['run'] = save_run(output, run_id, data, runner, state, roster, shared['policy'],
            paired['case_id'], strategy, 'S1', 7, full_day=False,
            cohort_start=paired['cohort_start_second'], cohort_end=paired['cohort_end_second'],
            service_end_second=paired['service_end_second'])
        metrics = _metrics(data, runner, state, roster, shared['policy'],
            paired['cohort_start_second'], paired['cohort_end_second'])
        metrics['peak_trains_on_line'] = runner.motion['checks']['peak_trains_on_line']
        metrics['fully_service_admissible'] = (metrics['actual_service_admissible']
            and metrics['all_service_arrivals_completed'] and metrics['movement_obligations_closed'])
        result.update(metrics=metrics, checkpoint_second=state['second'], date=data['period_start'][:10])
    else:
        result['pilot'] = True
    result['elapsed_seconds'] = time.monotonic() - began
    if any(_hash(ROOT / name) != sha for name, sha in frozen.items()):
        raise ValueError('experiment source changed during execution; journals retained')
    input_hashes = dict(paired['input_hashes'], **forecast_contract.get('original_forecast_hashes', {}))
    if any(_hash(path) != sha for path, sha in input_hashes.items()):
        raise ValueError('experiment input changed during execution; journals retained')
    if any(_hash(output / 'forecasts' / name) != sha
            for name, sha in forecast_contract.get('derived_forecast_hashes', {}).items()):
        raise ValueError('derived finite-cohort forecast changed during execution')
    write_json_once(final_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--source-case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--forecast-cache', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--strategy', choices=ARMS, required=True)
    parser.add_argument('--pilot-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run_focus(args.output, args.strategy, source_case=args.source_case,
        forecast_cache=args.forecast_cache, pilot_steps=args.pilot_steps, resume=args.resume)
    print(json.dumps(dict(stage='goal44_focus_complete', strategy=args.strategy,
        decisions=len(result['decisions']), seconds=result['elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
