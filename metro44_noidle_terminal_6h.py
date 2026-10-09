"""Paired six-hour physical and passenger trial with a common closed tail."""

import copy
import argparse
import gzip
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic

from metro44_noidle_terminal import bind_terminal_adapter, sha
from metro44_planner_noidle import remove_idle_cutoff, terminal_total_audit


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def validate_scope(data, groups):
    start, end = map(datetime.fromisoformat, (data['period_start'], data['period_end']))
    if (end-start).total_seconds() < 32400:
        raise ValueError('a common three-hour continuation is required')
    if (data['period_start'] != '2026-09-30T08:15:00+03:00'
            or max(g['turnstile_second'] for g in groups) != 21540
            or any(g['source_id'] != 'uploaded-spb-line1-15min' for g in groups)):
        raise ValueError('observed six-hour entries are required')


def trial_scope(clearance):
    if type(clearance) is not int or not 0 <= clearance <= 1800:
        raise ValueError('invalid passenger clearance')
    return dict(admission_end_second=21600, passenger_departure_end_second=21600+clearance)


def headway_profile(peak, offpeak, end):
    return [dict(start_second=0, end_second=6300, headway_seconds=peak),
            dict(start_second=6300, end_second=end, headway_seconds=offpeak)]


def validate_decision(second):
    if type(second) is not int or not 3600 <= second < 21600 or second % 900:
        raise ValueError('decision must occur on a 15-minute control slot')
    return second


def run(headway=125, clearance=0, output_name='metro44_noidle_terminal_6h_20261009', offpeak=None, as_of=3600):
    scope = trial_scope(clearance)
    validate_decision(as_of)
    root = Path(__file__).resolve().parent
    frozen = root / 'data/processed/metro44_search_execution_v2'
    reference_output = root / 'data/processed/metro44_continuation_6h_closed_20261009'
    reference = reference_output / 'runs/CONT6H_CLOSED_KEEP_S1_7'
    data = read(reference / 'movement_input.json')
    groups, profiles = read(reference / 'groups.json.gz'), read(reference / 'profiles.json')
    validate_scope(data, groups)
    manifest = read(frozen / 'execution_manifest.json')
    if any(sha(frozen / name) != value for name,value in manifest['source_hashes'].items()):
        raise ValueError('frozen source snapshot changed')
    output = root / 'data/processed' / output_name
    output.mkdir(exist_ok=False)
    for name in ('metro44_terminal_timing.py', 'metro44_planner_noidle.py', 'metro44_window.py'):
        shutil.copy2(root / name, output / name)
    modified = output / 'experimental_planner.py'
    modified.write_text(bind_terminal_adapter(remove_idle_cutoff((frozen / 'metro44_planner_search.py').read_text('utf-8'))), 'utf-8')
    shutil.copy2(Path(__file__), output / 'runner_source.py')
    contract = dict(frozen_snapshot=str(frozen), frozen_source_hashes=manifest['source_hashes'],
        experimental_source_hashes={name: sha(output / name) for name in
            ('metro44_terminal_timing.py','metro44_planner_noidle.py','metro44_window.py','experimental_planner.py','runner_source.py')},
        reference_inputs={name: sha(reference / name) for name in ('movement_input.json','groups.json.gz','profiles.json')},
        reference_folder=str(reference), observed_start=data['period_start'], observed_end='2026-09-30T14:15:00+03:00',
        shared_end=data['period_end'], initial_count=53, decision_second=as_of,
        peak_headway=headway, offpeak_headway=offpeak, phase_seconds={'L1_01':0,'L1_19':40}, occupancy_margin=2,
        **scope,
        no_forecast_used=True, posthoc_development=True, original_sources_modified=False)
    write(output / 'contract.json', contract)
    sys.path.insert(0, str(frozen)); sys.path.insert(0, str(output))
    spec = importlib.util.spec_from_file_location('_terminal_6h_planner', modified)
    planner = importlib.util.module_from_spec(spec); spec.loader.exec_module(planner)
    from goal43_replay import save_run
    from goal43_verify import verify
    from metro44_window import paired_window_bounds
    from metro_ml_compare import _metrics
    from passenger_groups import GroupedPassengerSimulation
    from simulation import simulate, digest
    strategy = f'NOIDLE_TERMINAL{headway}'+('' if offpeak is None else f'_OFF{offpeak}')+f'_C{clearance}'
    result = dict(strategy=strategy, terminal_resource_assumption='existing incoming and entry tracks share two slots; signalling unverified', release_authorized=False)
    began = monotonic()
    try:
        profile = headway if offpeak is None else headway_profile(headway, offpeak, scope['passenger_departure_end_second'])
        changed, audit = planner.rebuild_timetable(data, as_of, scope['passenger_departure_end_second'], headway=profile,
            phase_seconds=contract['phase_seconds'], occupancy_margin=2,
            repair_future_platforms=True, deadline=began+455)
        motion = simulate(changed, changed['plans'][0]['plan_id'])
        terminal = terminal_total_audit(changed, motion)
        if not (audit['movement_passed'] and terminal['passed'] and audit['peak_trains_on_line'] <= 53 and motion['obligations_closed']):
            result.update(status='physical_rejected', audit=audit, terminal_total=terminal)
        else:
            source = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles)
            checkpoint = source.run(end_second=as_of)['checkpoint']
            runner = GroupedPassengerSimulation(changed, changed['plans'][0]['plan_id'], groups, profiles)
            state = runner.branch_checkpoint(copy.deepcopy(checkpoint), source)
            state = runner.run(start_second=as_of, checkpoint=state)['checkpoint']
            roster = read(root / 'data/processed/metro_ml_comparison/cases/2026-09-30_peak_S1/fleet_registry.json')
            metrics = _metrics(changed, runner, state, roster, changed['headway_policy'], 3600, 21600)
            metrics.update(peak_trains_on_line=motion['checks']['peak_trains_on_line'])
            record = save_run(output, strategy+'_6H_S1_7', changed, runner, state, roster,
                changed['headway_policy'], 'closed_six_hour', strategy, 'S1', 7,
                full_day=False, cohort_start=3600, cohort_end=21600, service_end_second=21600)
            write(output / 'runs.json', [record])
            checked = verify(output)
            bounds = paired_window_bounds(data, source.motion, changed, motion, 3600, 21600)
            reference_metrics = read(reference_output / 'keep_result.json')['metrics']
            result.update(status='passenger_evaluated', audit={k:v for k,v in audit.items() if not k.startswith('_') and k!='departures'},
                terminal_total=terminal, metrics=metrics, run=record, verification_status=checked['status'],
                groups_digest=digest(groups), checkpoint_digest=digest(checkpoint), paired_window_bounds=bounds,
                full_continuation_saving_train_km=reference_metrics['train_km']-metrics['train_km'],
                common_input_digest=digest(data), common_end_second=runner.duration,
                admissible=metrics['actual_service_admissible'] and metrics['all_service_arrivals_completed'] and metrics['movement_obligations_closed'],
                cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None)
    except (ValueError, TimeoutError) as error:
        result.update(status='timeout' if isinstance(error, TimeoutError) else 'rejected', reason=str(error))
    result['elapsed_seconds'] = monotonic()-began
    result['source_ledger_preserved'] = all(sha(frozen/name)==value for name,value in contract['frozen_source_hashes'].items())
    result['source_ledger_preserved'] &= all(sha(output/name)==value for name,value in contract['experimental_source_hashes'].items())
    result['reference_inputs_preserved'] = all(sha(reference/name)==value for name,value in contract['reference_inputs'].items())
    write(output / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'metrics','run','paired_window_bounds','audit'}}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headway', type=int, default=125)
    parser.add_argument('--clearance', type=int, default=0)
    parser.add_argument('--offpeak', type=int)
    parser.add_argument('--as-of', type=int, default=3600)
    parser.add_argument('--output-name', default='metro44_noidle_terminal_6h_20261009')
    args = parser.parse_args()
    run(args.headway, args.clearance, args.output_name, args.offpeak, args.as_of)
