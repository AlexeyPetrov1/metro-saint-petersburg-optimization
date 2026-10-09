"""Paired development trial on preserved six-hour observations and a common tail."""

import argparse
import copy
import gzip
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'data/processed/metro44_bridge_execution_v5'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT))

from goal43_replay import save_run
from metro44_assignment_plan import terminal_resources
from metro44_journey import cohort_journey
from metro44_planner_bridge import rebuild_bridge_timetable
from metro44_window import paired_window_bounds
from metro45_adaptive import adaptive_profile, strict_improvement
from metro_ml_compare import _metrics
from passenger_groups import GroupedPassengerSimulation
from rolling import operational_exposure, operational_headways
from simulation import digest, simulate


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, *, scenario='S1', mode='ML', policy='adaptive', bridge_seconds=480):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    closed = ROOT / 'data/processed/metro44_continuation_6h_closed_20261009'
    robust = ROOT / 'data/processed/metro44_profile_6h_robustness_20261009'
    base_folder, result_path = ((closed, closed / 'keep_result.json') if scenario == 'S1' else
                               (robust, robust / f'keep_{scenario.lower()}_result.json'))
    base_result = read(result_path)
    folder = base_folder / base_result['run']['folder']
    inputs = [folder / name for name in ('movement_input.json', 'motion.json', 'groups.json.gz',
              'profiles.json', 'passenger_checkpoint.json.gz')] + [result_path]
    forecast_path = ROOT / f'data/processed/metro44_catalog_cont6h_20261009/controller/MPC_{mode}/forecast_06300.json'
    inputs.append(forecast_path)
    manifest = read(SNAPSHOT / 'execution_manifest.json')
    source_paths = [SNAPSHOT / name for name in manifest['source_hashes']]
    source_paths += [ROOT / name for name in ('metro45_adaptive.py', 'metro45_experiment.py')]
    before = {str(p): sha(p) for p in inputs + source_paths}
    write(output / 'before.json', before)
    data, groups, profiles = (read(folder / name) for name in
                             ('movement_input.json', 'groups.json.gz', 'profiles.json'))
    old = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles)
    warmup = old.run(end_second=6300)['checkpoint']
    baseline_state = read(folder / 'passenger_checkpoint.json.gz')
    old._validate_checkpoint(baseline_state, old.duration)
    baseline_metrics = base_result['metrics']
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=6300)).isoformat()
    model = read(forecast_path)
    if policy == 'adaptive':
        decision = adaptive_profile(model, ids, at, 6300, 22200, checkpoint=warmup, scenario=scenario)
        headway = decision['profile']
    else:
        headway = int(policy)
        decision = dict(profile=headway, future_actual_demand_used=False)
    contract = dict(scenario=scenario, source_mode=mode, policy=policy,
        change_second=6300, bridge_seconds=bridge_seconds, cohort_start_second=3600,
        cohort_end_second=21600, passenger_departure_end_second=22200, closure_end_second=32400,
        groups_digest=digest(groups), change_checkpoint_digest=digest(warmup),
        forecast_digest=digest(model), decision=decision, source_inputs=before,
        posthoc_development=True, independent_holdout=False, simulated_line_id=1,
        requested_line_id=2, blue_transfer_completed=False, release_authorized=False)
    write(output / 'contract.json', contract)
    began = monotonic()
    result = dict(status='pending', contract=contract, cash_saving_rub=None,
                  lost_revenue_rub=None, net_profit_effect_rub=None, release_authorized=False)
    try:
        changed, audit = rebuild_bridge_timetable(data, 6300, 22200, bridge_seconds=bridge_seconds,
            headway=headway, phase_seconds=0, deadline=began + 470)
        motion = simulate(changed, changed['plans'][0]['plan_id'])
        terminals = terminal_resources(changed, motion)
        gaps = operational_headways(changed, motion, 3600, 22200)
        physical = audit['movement_passed'] and terminals['passed'] and gaps['passed']
        result.update(physical_passed=physical, physical_seconds=monotonic() - began,
                      audit={k: v for k, v in audit.items() if not k.startswith('_')},
                      terminal_resources=terminals, planning_headways_passed=gaps['passed'])
        if not physical:
            raise ValueError('exact physical validation failed')
        runner = GroupedPassengerSimulation(changed, changed['plans'][0]['plan_id'], groups, profiles)
        state = runner.run(start_second=6300, checkpoint=runner.branch_checkpoint(copy.deepcopy(warmup), old),
                           deadline=began + 470)['checkpoint']
        roster = [dict(train_id=t['train_id'], type=t['train_type']) for t in data['trains']]
        metrics = _metrics(changed, runner, state, roster, data['headway_policy'], 3600, 21600)
        metrics.update(terminal_resources_passed=terminals['passed'],
                       peak_trains_on_line=motion['checks']['peak_trains_on_line'])
        bounds = paired_window_bounds(data, old.motion, changed, motion, 3600, 21600)
        journey = dict(KEEP=cohort_journey(old, baseline_state, 3600, 21000),
                       CANDIDATE=cohort_journey(runner, state, 3600, 21000))
        exposure = operational_exposure(state, runner.profiles, data['headway_policy'], 0, runner.duration)
        saved = save_run(output, f'ADAPT_{mode}_{policy}_{scenario}_7', changed, runner, state,
            roster, data['headway_policy'], 'six_hour_adaptive_dispatch', f'ADAPT_{mode}_{policy}', scenario, 7,
            full_day=False, cohort_start=3600, cohort_end=21600, service_end_second=22200)
        result.update(status='evaluated', metrics=metrics, baseline_metrics=baseline_metrics,
            window_bounds=bounds, journey=journey, complete_episode_exposure=exposure,
            controlled_window_service_admissible=metrics['actual_service_admissible'],
            strict_controlled_improvement=strict_improvement(baseline_metrics, metrics, bounds,
                                                           physical_passed=physical),
            whole_episode_service_admissible=exposure['admissible'],
            run=saved, movement_digest=digest(changed),
            mean_wait_delta_seconds=metrics['mean_wait_seconds'] - baseline_metrics['mean_wait_seconds'],
            p95_wait_delta_seconds=metrics['p95_wait_seconds'] - baseline_metrics['p95_wait_seconds'])
    except (ValueError, TimeoutError) as error:
        result.update(status='rejected', reason=str(error), strict_controlled_improvement=False)
    after = {str(p): sha(p) for p in inputs + source_paths}
    result.update(elapsed_seconds=monotonic() - began, inputs_and_sources_preserved=after == before)
    write(output / 'after.json', after)
    if after != before:
        raise ValueError('experiment sources or original inputs changed')
    write(output / 'result.json', result)
    keys = ('status', 'reason', 'physical_passed', 'strict_controlled_improvement',
            'controlled_window_service_admissible', 'whole_episode_service_admissible',
            'mean_wait_delta_seconds', 'p95_wait_delta_seconds', 'elapsed_seconds')
    brief = {k: result[k] for k in keys if k in result}
    brief['bounds'] = {k: v for k, v in result.get('window_bounds', {}).items()
                       if k.startswith('saving_')}
    print(json.dumps(brief), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenario', choices=['S1', 'S2', 'S3'], default='S1')
    parser.add_argument('--mode', choices=['ML', 'BASELINE_FORECAST'], default='ML')
    parser.add_argument('--policy', choices=['adaptive', '150', '160'], default='adaptive')
    parser.add_argument('--bridge-seconds', type=int, default=480)
    args = parser.parse_args()
    run(args.output, scenario=args.scenario, mode=args.mode, policy=args.policy,
        bridge_seconds=args.bridge_seconds)
