"""Independent balanced-policy audit, paired economics, and reproducible reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from metro50_economics import economy_comparison

SIDES = ('without_sr_kr', 'with_sr_kr')
SIDE_NAMES = ('Без СР и КР', 'С СР и КР')
REQUIRED_MODEL_TESTS = {
    'test_actual_contract_uses_all_native_bases_and_true_cutoffs',
    'test_bundle_complete_quarter_refresh_and_future_mutation_invariance',
    'test_available_at_prevents_using_late_observations',
    'test_training_period_and_unsupported_horizon_are_rejected',
    'test_b3_output_matches_saved_native_networks_on_same_input',
    'test_source_frame_has_all_red_stations_and_same_day_truth',
    'test_feature_evidence_uses_measured_lightgbm_contributions',
}
MODEL_CHECKS = {'evidence_files_present', 'complete_native_components', 'model_weight_hashes',
                'input_manifest_weight_hashes', 'source_identity', 'b1_saved_equation_parity',
                'contract_feature_and_time_shape', 'test_proof_hashes', 'actual_model_tests',
                'run_model_contract_parity'}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root, *, excluded=None):
    root = Path(root).resolve()
    excluded = Path(excluded).resolve() if excluded is not None else None
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or excluded is not None and path.is_relative_to(excluded):
            continue
        if path.name.startswith('.env') or path.suffix.lower() in ('.key', '.pem'):
            continue
        rows.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                         sha256=sha256(path)))
    return rows


def verify_model_evidence(input_dir, *, test_evidence=None, run_contracts=()):
    """Check live artifacts, saved equation parity, and hashed actual-model JUnit."""
    inputs = Path(input_dir)
    names = ('contract.json', 'input_manifest.json', 'model_audit.json', 'source_identity_audit.json')
    checks = {key: False for key in MODEL_CHECKS}
    artifacts = {name: sha256(inputs/name) for name in names if (inputs/name).is_file()}
    checks['evidence_files_present'] = len(artifacts) == len(names)
    limits = ['Проверка адаптера не устанавливает свежую независимую оценку: сентябрь повторно используется для разработки.',
              'Даты публикации погодных признаков неизвестны; строгая доступность всех внешних данных не подтверждена.',
              'Сдвинутые 15-минутные обновления экспериментальны; исходный шаг стекинга 30 минут.']
    if not checks['evidence_files_present']:
        return dict(passed=False, checks=checks, failed_checks=sorted(k for k, v in checks.items() if not v),
                    artifacts=artifacts, input_dir=str(inputs), limitations=limits)
    contract, manifest, audit, identity = (read(inputs/name) for name in names)
    required_components = ['b1_lightgbm', 'b2_persistence', 'b3_gru_seeds012', 'frozen_meta_stack']
    checks['complete_native_components'] = contract.get('components') == required_components
    weights = contract.get('weights_sha256', {})
    files, expected = {}, {}
    for key, folder in [('b1', 'lgbm_holdout'), ('b3', 'stack_gru_2026-09')]:
        for name, value in weights.get(key, {}).items():
            relative = 'models/'+folder+'/'+name
            files[relative], expected[relative] = inputs/relative, value
    for key, name in [('b2', 'b2_quantiles_september.json'), ('stack_params', 'stack_params.json')]:
        files[name], expected[name] = inputs/name, weights.get(key)
    required_files = {'models/lgbm_holdout/'+name for name in ('meta.json', 'q10.txt', 'q50.txt', 'q90.txt', 'station_quantiles.csv')}
    required_files |= {'models/stack_gru_2026-09/'+name for name in ('meta.json', 'gru_seed0.pt', 'gru_seed1.pt', 'gru_seed2.pt')}
    actual = {name: sha256(path) for name, path in files.items() if path.is_file()}
    checks['model_weight_hashes'] = required_files <= set(actual) and set(actual) == set(expected) and actual == expected
    checks['input_manifest_weight_hashes'] = bool(required_files) and all(manifest.get('files', {}).get(name) == actual.get(name) for name in required_files)
    data_names = ('spb_line1_15min.parquet', 'spb_line1_hourly.parquet', 'spb_line1_15min_reconciliation.parquet')
    source_ok = True
    for name in data_names:
        record = identity.get('data_sha256', {}).get(name, {})
        path = inputs/'data/interim'/name
        source_ok &= path.is_file() and record.get('identical') is True and record.get('new') == record.get('old')
        if path.is_file():
            source_ok &= sha256(path) == record.get('new') == manifest.get('files', {}).get('data/interim/'+name)
    for name, value in weights.get('b3', {}).items():
        record = identity.get('GRU_sha256', {}).get(name, {})
        source_ok &= record.get('identical') is True and record.get('new') == record.get('old') == value
    checks['source_identity'] = bool(source_ok)
    error = audit.get('b1_native_slot_max_absolute_error')
    checks['b1_saved_equation_parity'] = (type(error) in (float, int) and math.isfinite(error)
        and 0 <= error < 1e-9 and audit.get('native_rows', 0) >= 90)
    features, vestibules = contract.get('feature_order', []), contract.get('vestibule_order', [])
    checks['contract_feature_and_time_shape'] = (
        len(features) == len(set(features)) == 26 and len(vestibules) == len(set(vestibules)) == 23
        and contract.get('native_update_seconds') == 1800 and contract.get('simulator_refresh_seconds') == 900
        and contract.get('future_truth_used') is False and contract.get('retrospective_history_flags') is False
        and contract.get('meta_fit_months') == [5, 7] and contract.get('b2_calibration_months') == [2, 5, 7])
    keys = ('components', 'weights_sha256', 'b1_train_end', 'b3_train_end', 'meta_fit_months',
            'b2_calibration_months', 'native_update_seconds', 'simulator_refresh_seconds', 'feature_order', 'vestibule_order')
    live_contracts = [c['model_contract'] for c in run_contracts if (c.get('model_contract') or {}).get('components') == required_components]
    checks['run_model_contract_parity'] = all(all(c.get(k) == contract.get(k) for k in keys) for c in live_contracts)
    passed_names = set()
    if test_evidence is not None and Path(test_evidence).is_file():
        try:
            proof = read(test_evidence)
            xml_path = Path(proof['junit_xml'])
            source_hashes = proof['source_sha256']
            source_names = {Path(p).name for p in source_hashes}
            checks['test_proof_hashes'] = (proof.get('exit_code') == 0 and xml_path.is_file()
                and sha256(xml_path) == proof['junit_sha256']
                and {'test_metro51_ensemble.py', 'metro51_ensemble.py'} <= source_names
                and all(Path(p).is_file() and sha256(p) == value for p, value in source_hashes.items())
                and proof.get('input_manifest_sha256') == artifacts['input_manifest.json']
                and 'pytest' in proof.get('command', []))
            if checks['test_proof_hashes']:
                tree = ET.parse(xml_path)
                for case in tree.iter('testcase'):
                    if not any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped')):
                        passed_names.add(case.get('name'))
                checks['actual_model_tests'] = REQUIRED_MODEL_TESTS <= passed_names
            artifacts['test_evidence.json'] = sha256(test_evidence)
            if xml_path.is_file():
                artifacts['model_junit.xml'] = sha256(xml_path)
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError):
            pass
    return dict(passed=all(checks.values()), checks=checks,
                failed_checks=sorted(k for k, v in checks.items() if not v),
                artifacts=artifacts, input_dir=str(inputs), verified_weight_hashes=actual,
                required_tests=sorted(REQUIRED_MODEL_TESTS), passed_tests=sorted(passed_names),
                contract=contract, saved_model_audit=audit, source_identity=identity, limitations=limits,
                method='live hashes + saved equation parity + actual-model JUnit; no model training or prediction during report')


def _quantiles(histogram):
    count = sum(histogram.values())
    def at(q):
        seen, target = 0, math.ceil(q*count)
        for value, n in sorted(histogram.items()):
            seen += n
            if seen >= target:
                return value
    return dict(count=count, p95=at(.95) if count else None, p99=at(.99) if count else None,
                maximum=max(histogram, default=None))


def comfort_metrics(runner, state, *, end_second=None):
    from metro50_peak_service import is_peak

    end = state['second'] if end_second is None else min(end_second, state['second'])
    origin = datetime.fromisoformat(runner.motion['period_start'])
    windows = [('whole_window', 0, end)]
    day = origin.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < origin+timedelta(seconds=end):
        if day.weekday() < 5:
            for name, hours in [('morning_peak', (8, 10)), ('evening_peak', (17, 19))]:
                a, b = (int((day+timedelta(hours=h)-origin).total_seconds()) for h in hours)
                left, right = max(0, a), min(end, b)
                if left < right:
                    windows.append((name, left, right))
        day += timedelta(days=1)
    by_car = defaultdict(list)
    for row in state['load_events']:
        by_car[row['train_id'], row['car_id']].append(row)
    result = {}
    for name, start, stop in windows:
        seconds = dict(observed_car_seconds=0, occupied_car_seconds=0,
                       above_120_car_seconds=0, above_140_car_seconds=0)
        departure_loads = Counter()
        for rows in by_car.values():
            since, load = 0, 0
            for row in rows+[dict(second=stop, load=0)]:
                length = max(0, min(row['second'], stop)-max(start, since))
                seconds['observed_car_seconds'] += length
                seconds['occupied_car_seconds'] += length*(load > 0)
                seconds['above_120_car_seconds'] += length*(load > 120)
                seconds['above_140_car_seconds'] += length*(load > 140)
                if row['second'] >= stop:
                    break
                event = runner.motion_by_id.get(row.get('motion_event_id'), {})
                if row['second'] >= start and row.get('event_type') == 'departure' and event.get('passenger_service'):
                    departure_loads[row['load']] += 1
                since, load = row['second'], row['load']
        seconds.update(_quantiles(departure_loads), start_second=start, end_second=stop,
            above_120_time_share=seconds['above_120_car_seconds']/seconds['occupied_car_seconds'] if seconds['occupied_car_seconds'] else None,
            above_140_time_share=seconds['above_140_car_seconds']/seconds['occupied_car_seconds'] if seconds['occupied_car_seconds'] else None,
            denominator='positive-load car-seconds; zero-load observed time reported separately',
            quantile_scope='every car at passenger departure in this time window; zero loads included')
        result[name] = seconds
    peak_loads = Counter(row['load'] for row in state['load_events']
        if 0 <= row['second'] < end and row['event_type'] == 'departure'
        and runner.motion_by_id.get(row['motion_event_id'], {}).get('passenger_service')
        and is_peak(origin+timedelta(seconds=row['second'])))
    result['peak'] = _quantiles(peak_loads)
    return result


def local_waiting_metrics(runner, state, *, cohort_end_second):
    histograms, population = defaultdict(Counter), Counter()
    for group in runner.groups:
        if not 0 <= group['platform_second'] < cohort_end_second:
            continue
        for dest in group['destinations']:
            population[group['origin'], dest['direction']] += dest['count']
    for row in state['board_events']:
        group = runner.by_id[row['group_id']]
        if not 0 <= group['platform_second'] < cohort_end_second:
            continue
        destination = row.get('destination')
        if destination is None:
            raise ValueError('local waiting metrics require destination-bearing boarding ledgers')
        direction = 1 if runner.positions[destination] > runner.positions[group['origin']] else -1
        histograms[group['origin'], direction][row['second']-group['platform_second']] += row['count']
    rows = []
    for (station, direction), passengers in sorted(population.items()):
        hist = histograms[station, direction]
        q = _quantiles(hist)
        current = [state['second']-r['platform_second'] for queue in state['queues'].values() for r in queue
                   if r['origin'] == station and r['direction'] == direction and r['count']]
        rows.append(dict(station_id=station, direction=direction, passenger_count=passengers,
            boarded=q['count'], complete=q['count'] == passengers,
            mean_wait_seconds=sum(v*n for v, n in hist.items())/passengers if passengers == q['count'] and passengers else None,
            p95_wait_seconds=q['p95'] if passengers == q['count'] else None,
            p99_wait_seconds=q['p99'] if passengers == q['count'] else None,
            maximum_wait_seconds=q['maximum'],
            maximum_queue_age_seconds=max([q['maximum'] or 0, *current]),
            max_queue=state.get('max_queue_by_direction', {}).get(f'{station}:{direction}', 0),
            queue_scope='exact maximum across observed prefix; age is longest observed individual wait, not density safety'))
    return rows


def _executed_ledger(data, motion, end):
    """Count completed track arrivals in the observed prefix, not full-day plans."""
    from decimal import Decimal

    tracks = {t['track_id']: t for t in data['tracks']}
    trains = {t['train_id']: t for t in data['trains']}
    road = {tid: row['location'] for tid, row in trains.items() if row['state'] == 'moving'}
    mileage = Counter()
    for event in motion['events']:
        if event['second'] > end:
            break
        tid = event['train_id']
        if event['event_type'] == 'departure':
            road[tid] = event['location']
        elif event['event_type'] == 'arrival' and event['second'] > 0:
            mileage[tid] += Decimal(str(tracks[road[tid]]['length_km']))
    return [dict(train_id=tid, train_type=record['train_type'], km=float(mileage[tid]))
            for tid, record in sorted(trains.items())]


def expected_control_times(contract):
    if contract.get('arm') == 'REFERENCE':
        return []
    if 'decision_times' in contract:
        return contract['decision_times']
    end = contract.get('cohort_end_second', contract.get('entry_end_second'))
    times = list(range(contract.get('control_start_second', 3600), end, 900))
    return [s for s in times if s >= 36000] if contract.get('stage') == 'evening' else times


def committed_prefix_preserved(folder, motion, decisions):
    from metro44_focus import execution_prefix
    from simulation import simulate

    for decision in decisions:
        second = decision['decision_second']
        saved_digest = decision.get('executed_prefix_digest')
        if saved_digest is not None:
            if saved_digest != execution_prefix(motion, second):
                return False
            continue
        selection = decision.get('selection', {})
        if (decision.get('applied_candidate_id') != 'KEEP'
                or selection.get('status') not in ('evaluation_error', 'evaluation_timeout')
                or selection.get('release_authorized') or decision.get('release_authorized')
                or selection.get('selected_candidate_id') is not None):
            return False
        path = Path(folder)/'controller'/f'resume_{second:05}.json.gz'
        if not path.is_file():
            return False
        saved = read(path)
        if saved.get('second') != second or saved.get('state', {}).get('second') != second:
            return False
        data = saved['data']
        prior = simulate(data, data['plans'][0]['plan_id'])
        if execution_prefix(prior, second) != execution_prefix(motion, second):
            return False
    return True


def audit_arm(folder):
    from rolling import operational_headways
    from metro44_catalog import physical_gate
    from metro44_focus import execution_prefix
    from metro50_review import peak_car_quantiles
    from metro51_service import local_boarding_service
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest, simulate

    folder = Path(folder)
    result, contract = read(folder/'result.json'), read(folder/'contract.json')
    common = read(folder/'common_scenario.json.gz')
    run = folder / result['run']['folder']
    data, motion, profiles = (read(run/name) for name in ('movement_input.json', 'motion.json', 'profiles.json'))
    groups, state = read(run/'groups.json.gz'), read(run/'passenger_checkpoint.json.gz')
    observed = contract.get('observation_end_second', state['second'])
    cohort_end = contract.get('platform_cohort_end_second', contract.get('entry_end_second', observed))
    entry_end = contract.get('cohort_end_second', contract.get('entry_end_second', cohort_end))
    service_end = contract['service_end_second']
    seed = contract.get('seed', contract.get('boarding_seed', 7))
    cap = contract.get('boarding_capacity_per_car', 150)
    checks = dict(result_contract_parity=result['contract'] == contract,
                  red_line_fixed_fleet=contract.get('line_id') == 1 and contract.get('fixed_fleet_size') == 68)
    hashes = contract.get('source_hashes', {})
    source_failures = [p for p, value in hashes.items() if not Path(p).is_file() or sha256(p) != value]
    checks['source_hashes_preserved'] = bool(hashes) and not source_failures and result.get('sources_preserved') is True
    snapshot = folder/'execution_sources'
    snapshot_failures = [p for p, value in hashes.items() if Path(p).suffix == '.py'
                         and (not (snapshot/Path(p).name).is_file() or sha256(snapshot/Path(p).name) != value)]
    checks['execution_snapshots_preserved'] = bool(hashes) and not snapshot_failures
    sort_groups = lambda rows: sorted(rows, key=lambda row: row['group_id'])
    checks['common_groups_preserved'] = digest(common['groups']) == contract['groups_digest'] and sort_groups(groups) == sort_groups(common['groups'])
    checks['geometry_and_profiles_preserved'] = all(data[k] == common['data'][k] for k in ('trains', 'stations', 'tracks', 'headway_policy')) and profiles == common['profiles']
    checks['fixed_roster'] = (len(common['roster']) == 68
        and len({r['train_id'] for r in common['roster']}) == 68
        and Counter(r['type'] for r in common['roster']) == Counter(Yubileyny=31, Baltiets=37)
        and all(t['cars'] == 8 for t in data['trains']))
    checks['saved_motion_parity'] = digest(simulate(data, data['plans'][0]['plan_id'])) == digest(motion)
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles,
                                        boarding_seed=seed, boarding_capacity_per_car=cap)
    checks['runner_motion_parity'] = digest(runner.motion) == digest(motion)
    try:
        runner._validate_checkpoint(state, state['second'])
        checks['checkpoint_integrity'] = True
    except (ValueError, KeyError, TypeError):
        checks['checkpoint_integrity'] = False
    checks['observation_complete'] = state['second'] == observed and observed >= cohort_end
    checks['passenger_accounting_complete'] = (state['entered'] == state['completed'] == sum(g['count'] for g in groups)
        and state['queued'] == state['approaching'] == state['onboard_count'] == 0)
    physical = physical_gate(data, 0, service_end, reference_motion=motion)
    checks['independent_physical_gate'] = physical['movement_passed'] is True
    headways = operational_headways(data, motion, 0, min(service_end, observed))
    checks['independent_headways'] = headways['passed'] is True
    local = local_boarding_service(runner, state, end_second=cohort_end,
                                  first_limit=contract.get('first_miss_limit', .05),
                                  second_limit=contract.get('second_miss_limit', .01))
    checks['local_boarding_service'] = local['passed'] is True
    peak = peak_car_quantiles(runner, state, end_second=min(service_end, observed))
    checks['peak_car_ledger_complete'] = peak['complete_car_coverage']
    checks['car_capacity_limit'] = max((r['load'] for r in state['load_events']), default=0) <= 150
    per_train = _executed_ledger(data, motion, observed)
    saved = result['metrics'].get('actual_per_train', result['metrics'].get('per_train', []))
    ordered = lambda rows: sorted(rows, key=lambda r: r['train_id'])
    checks['observed_mileage_ledger_parity'] = ordered(saved) == ordered(per_train)
    checks['observed_mileage_total_parity'] = math.isclose(sum(r['km'] for r in per_train), result['metrics']['train_km'], abs_tol=1e-7)
    decisions = result['decisions']
    expected = expected_control_times(contract)
    checks['control_schedule_complete'] = [d.get('decision_second') for d in decisions] == expected
    durations = [d.get('pipeline_seconds') for d in decisions]
    checks['recommendations_within_480_seconds'] = all(type(s) in (int, float) and math.isfinite(s) and 0 <= s <= 480 for s in durations)
    checks['journal_receipts_match'] = all((folder/'controller'/f"decision_{d['decision_second']:05}.json").is_file()
        and read(folder/'controller'/f"decision_{d['decision_second']:05}.json") == d for d in decisions)
    checks['committed_prefix_preserved'] = committed_prefix_preserved(folder, motion, decisions)
    metric = dict(result['metrics'], actual_per_train=per_train,
                  train_km=sum(r['km'] for r in per_train), headways=headways,
                  local_boarding_service=local, peak_car_quantiles=peak,
                  comfort=comfort_metrics(runner, state),
                  local_waiting=local_waiting_metrics(runner, state, cohort_end_second=cohort_end))
    accounting_checks = {k: v for k, v in checks.items() if k != 'local_boarding_service'}
    audit = dict(passed=all(checks.values()), checks=checks,
                 failed_checks=[k for k, v in checks.items() if not v],
                 baseline_valid_for_comparison=all(accounting_checks.values()),
                 source_files_changed=source_failures, snapshot_files_changed=snapshot_failures,
                 metrics=metric, local_service=local,
                 physical={k: v for k, v in physical.items() if not k.startswith('_')},
                 controller=dict(count=len(decisions), expected_count=len(expected),
                                 maximum_seconds=max(durations, default=None),
                                 timeouts=sum(bool(d.get('timeout')) for d in decisions),
                                 no_feasible_count=sum(d.get('selection', {}).get('status') == 'no_feasible_candidate' for d in decisions),
                                 external_delivery_measured=False),
                 common_scenario_digest=digest(common), groups_digest=digest(sort_groups(groups)),
                 roster_digest=digest(common['roster']), profiles_digest=digest(profiles),
                 geometry_digest=digest({k: common['data'][k] for k in ('trains', 'stations', 'tracks', 'headway_policy')}),
                 boarding_seed=seed,
                 entry_end_second=entry_end, cohort_end_second=cohort_end,
                 observation_end_second=observed,
                 elapsed_run_seconds=result.get('elapsed_seconds'),
                 checkpoint_verified_against_recompiled_runner=True,
                 full_passenger_replay_performed=False,
                 actual_mileage_scope='completed arrivals through observation end; initial crossing track convention shared across arms')
    return audit, runner, state


def make_comparison(results, audits, *, paired_inputs_passed, annual_days=250, model_evidence=None):
    if 'reference' not in results or set(results) != set(audits):
        raise ValueError('reference and the same arms in results and independent audits required')
    full_day = all(r.get('contract', {}).get('full_day') is True for r in results.values())
    period = 'daily' if full_day else 'observed_window'
    ledgers = {}
    for arm, result in results.items():
        metric = result['metrics']
        if not full_day and 'actual_per_train' not in metric:
            raise ValueError('pilot requires observed mileage, not full planned mileage')
        ledgers[arm] = {'per_train': metric.get('actual_per_train', metric.get('per_train'))}
    reference_valid = audits['reference'].get('baseline_valid_for_comparison') is True
    economics, accepted = {}, {}
    for arm, result in results.items():
        admitted = paired_inputs_passed is True and reference_valid and audits[arm].get('passed') is True
        econ = economy_comparison(ledgers['reference'], ledgers[arm], annual_days=annual_days, admissible=admitted)
        if not full_day:
            econ['observed_window'] = econ.pop('daily')
            econ.pop('annual')
            econ['planned'] = None
            for row in econ['per_train']:
                row['observed_window'] = row.pop('daily')
                row.pop('annual')
            econ['period_scope'] = 'observed pilot/evening window; daily and annual extrapolation prohibited'
        else:
            econ['period_scope'] = 'complete simulated day; annual repetition is an explicit scenario'
        economics[arm] = econ
        accepted[arm] = {s: econ[period][s]['saving_rub'] if admitted else None for s in SIDES}
    balanced = economics.get('balanced', {})
    model_evidence = model_evidence or {}
    model_verified = (model_evidence.get('passed') is True
        and MODEL_CHECKS <= set(model_evidence.get('checks', {}))
        and all(model_evidence['checks'].get(key) is True for key in MODEL_CHECKS)
        and {'contract.json', 'input_manifest.json', 'model_audit.json', 'source_identity_audit.json',
             'test_evidence.json', 'model_junit.xml'} <= set(model_evidence.get('artifacts', {})))
    return dict(schema_version=1, stage='full_day' if full_day else results['reference'].get('contract', {}).get('stage', 'partial'),
                full_day=full_day, period_key=period, annual_scenario_emitted=full_day,
                paired_inputs_passed=paired_inputs_passed is True,
                reference_valid_for_comparison=reference_valid,
                all_results_admissible=paired_inputs_passed is True and all(a.get('passed') is True for a in audits.values()),
                audits=audits, contracts={k: v.get('contract', {}) for k, v in results.items()},
                economics=economics, accepted_saving=accepted,
                model_evidence=model_evidence, ensemble_verified=model_verified,
                goal_achieved=bool(full_day and model_verified and balanced.get('meets_one_percent_target')),
                goal_status_scope='full-day model verification and admissible >=1% resource saving; transferability still requires separate days and anomaly cases',
                cash_saving_rub=None, lost_revenue_rub=None,
                results_summary={k: dict(elapsed_seconds=v.get('elapsed_seconds'), decisions=v.get('decisions', [])) for k, v in results.items()})


def compile_comparison(root, *, annual_days=250):
    from metro51_service import compare_waiting_budget

    root = Path(root)
    folders = {p.parent.name: p.parent for p in sorted(root.glob('*/result.json'))}
    if 'reference' not in folders:
        raise FileNotFoundError('finished reference/result.json required')
    results, audits, contexts = {}, {}, {}
    for arm, folder in folders.items():
        results[arm] = read(folder/'result.json')
        audits[arm], runner, state = audit_arm(folder)
        contexts[arm] = runner, state
    base = audits['reference']
    paired = all(all(a[key] == base[key] for key in ('groups_digest', 'profiles_digest', 'roster_digest', 'geometry_digest', 'boarding_seed',
                                                   'entry_end_second', 'cohort_end_second', 'observation_end_second'))
                 for a in audits.values())
    for arm, audit in audits.items():
        contract = results[arm]['contract']
        waiting = compare_waiting_budget(*contexts['reference'], *contexts[arm], end_second=base['cohort_end_second'],
            mean_allowance=contract.get('mean_wait_allowance', 10), p95_allowance=contract.get('p95_wait_allowance', 40))
        audit['waiting_budget'] = waiting
        audit['checks']['fixed_reference_waiting_budget'] = waiting['passed'] is True
        audit['passed'] = all(audit['checks'].values())
        audit['failed_checks'] = [k for k, v in audit['checks'].items() if not v]
        whole = next(w for w in waiting['windows'] if w['name'] == 'whole_period')['candidate']
        audit['metrics'].update(mean_wait_seconds=whole['mean_wait_seconds'], p95_wait_seconds=whole['p95_wait_seconds'],
                                wait_scope='all selected platform-arrival cohorts with later followup; independent paired audit')
    all_contracts = [read(path) for path in sorted(root.glob('*/contract.json'))]
    model_contract = next((c.get('model_contract') for c in all_contracts if (c.get('model_contract') or {}).get('components')), None)
    proof = root/'model_verification/evidence.json'
    model_evidence = verify_model_evidence(model_contract['input_dir'],
        test_evidence=proof, run_contracts=all_contracts) if model_contract else {}
    comparison = make_comparison(results, audits, paired_inputs_passed=paired, annual_days=annual_days,
                                 model_evidence=model_evidence)
    comparison['source_root'] = str(root.resolve())
    comparison['explanation_summary'], comparison['explanations'] = collect_explanations(root)
    comparison['post_execution_forecasts'] = collect_post_execution_forecasts(root)
    expected = [p.parent.name for p in sorted(root.glob('*/contract.json'))]
    complete = bool(expected) and set(expected) <= set(results)
    pilot_passed = complete and audits.get('balanced', {}).get('passed') is True and model_evidence.get('passed') is True
    full_status = ('completed' if comparison['full_day'] and complete else 'not_run_pending_pilot' if not complete else
                   'not_run_pending_full_day' if pilot_passed else 'not_run_due_to_failed_pilot')
    comparison['campaign_status'] = dict(expected_arms=expected, completed_arms=list(results),
        unfinished_arms=sorted(set(expected)-set(results)), all_arms_finished=complete,
        pilot_passed=pilot_passed, full_day_status=full_status,
        full_day_completed=comparison['full_day'] and complete,
        future_full_day_ran_in_another_folder_unknown=True)
    return comparison


def collect_post_execution_forecasts(root):
    path = Path(root)/'forecast_review/predictions.csv'
    if not path.is_file():
        return []
    groups = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            groups[row['arm'], row['origin']].append(row)
    result = []
    for (arm, origin), rows in sorted(groups.items()):
        observed = [r for r in rows if r.get('truth_status') == 'complete' and r.get('truth') not in ('', None)]
        true = sum(int(float(r['truth'])) for r in observed)
        predicted = sum(int(float(r['predicted'])) for r in observed)
        decision_predicted = sum(int(float(r.get('decision_predicted') or r['predicted'])) for r in observed)
        error = sum(abs(int(float(r['predicted']))-int(float(r['truth']))) for r in observed)
        result.append(dict(arm=arm, origin=origin, n=len(observed), target_rows=len(rows),
            actual_sum=true, predicted_sum=predicted, decision_predicted_sum=decision_predicted,
            absolute_error_sum=error, wape=error/true if true else None,
            mae=error/len(observed) if observed else None,
            forecast_vs_actual_delta=predicted-true,
            known_after_execution=True, used_for_dispatch=False,
            known_after=max((r['interval_end'] for r in observed), default=None),
            source_file='forecast_review/predictions.csv'))
    return result


def collect_incident_probes(folders):
    """Read synthetic probes separately from the dispatch campaign admission."""
    rows = {}
    for root in folders:
        for path in sorted(Path(root).glob('stop_*/result.json')):
            case = read(path)
            diagnostics = case.get('diagnostics')
            if not isinstance(diagnostics, dict) or 'movement' not in diagnostics:
                continue
            receipt = case.get('receipt', {})
            movement = diagnostics['movement']
            incident = diagnostics.get('incident', {})
            service = diagnostics.get('local_service', {})
            window = next((w for w in diagnostics.get('waiting_budget', {}).get('windows', [])
                           if w.get('name') == 'whole_period'), {})
            duration = case['duration_minutes']
            rows[duration] = dict(duration_minutes=duration,
                synthetic=case.get('synthetic') is True,
                receipt_status=receipt.get('status'), elapsed_seconds=receipt.get('elapsed_seconds'),
                process_within_eight_minutes=(isinstance(receipt.get('elapsed_seconds'), (int, float))
                    and receipt['elapsed_seconds'] <= 480 and receipt.get('within_budget') is True),
                external_delivery_measured=case.get('external_delivery_measured') is True,
                movement_passed=movement.get('movement_passed') is True,
                minimum_headway=movement.get('minimum_headway'), maximum_headway=movement.get('maximum_headway'),
                local_service_status=service.get('status'),
                worst_first_missed_share=max((c.get('missed_first_share') or 0 for c in service.get('cohorts', [])), default=None),
                worst_second_missed_share=max((c.get('missed_second_share') or 0 for c in service.get('cohorts', [])), default=None),
                mean_wait_delta_seconds=window.get('mean_delta_seconds'), p95_wait_delta_seconds=window.get('p95_delta_seconds'),
                maximum_queue=diagnostics.get('max_station_direction_queue'),
                remaining_queued=diagnostics.get('remaining_queued'), remaining_onboard=diagnostics.get('remaining_onboard'),
                recovery_complete=diagnostics.get('passenger_recovery_complete') is True,
                signalling_verified=incident.get('signalling_verified') is True,
                blocked_track_all_trains_verified=incident.get('blocked_track_all_trains_verified') is True,
                future_actual_used_for_evaluation=diagnostics.get('future_actual_used_for_evaluation'),
                causal_operational_recommendation=diagnostics.get('causal_operational_recommendation') is True,
                action=diagnostics.get('action'), accepted_resource_saving=False, cash_saving_rub=None,
                source_file=str(path.resolve()), source_sha256=sha256(path),
                validation_scope='persisted diagnostic result; not an independent signalling simulation')
    return [rows[k] for k in sorted(rows)]


def collect_explanations(root):
    from metro51_explanations import ExplanationError, validate_explanation

    rows = []
    summary = dict(verified_count=0, invalid_count=0, missing_count=0,
                   yandexgpt_count=0, template_count=0, timeout_count=0,
                   http_error_count=0, request_count=0, total_tokens=0,
                   token_usage_complete=True, api_cost_rub=None)
    for facts_path in sorted(Path(root).glob('*/explanations/facts_*.json')):
        arm = facts_path.parent.parent.name
        result_path = facts_path.with_name(facts_path.name.replace('facts_', 'explanation_', 1))
        if not result_path.is_file():
            summary['missing_count'] += 1
            continue
        try:
            packet, result = read(facts_path), read(result_path)
            validate_explanation(packet, result)
        except (ExplanationError, ValueError, KeyError, TypeError):
            summary['invalid_count'] += 1
            continue
        summary['verified_count'] += 1
        meta = result['metadata']
        source = meta.get('source')
        if source in ('template', 'yandexgpt'):
            summary[source+'_count'] += 1
        summary['timeout_count'] += meta.get('error_code') == 'timeout'
        summary['http_error_count'] += meta.get('http_status') is not None
        summary['request_count'] += meta.get('request_count', 0)
        usage = meta.get('usage')
        if (source == 'yandexgpt' or meta.get('request_count', 0) > 0) and (not isinstance(usage, dict) or usage.get('total_tokens') is None):
            summary['token_usage_complete'] = False
        if isinstance(usage, dict) and usage.get('total_tokens') is not None:
            summary['total_tokens'] += usage['total_tokens']
        rows.append(dict(arm=arm, run_id=result['run_id'], decision_id=result['decision_id'],
                         action=packet['action'],
                         context_hash=meta['context_hash'], source=source, model=meta.get('model'),
                         elapsed_seconds=meta.get('elapsed_seconds'), error_code=meta.get('error_code'),
                         sections=result['sections'], packet_file=str(facts_path),
                         explanation_file=str(result_path),
                         attribution_groups=summarize_attribution_facts(packet['facts'])))
    if not summary['token_usage_complete']:
        summary['total_tokens'] = None
    return summary, rows


def summarize_attribution_facts(facts, *, max_features=3):
    groups = defaultdict(list)
    for fact in facts:
        if fact.get('kind') != 'attribution' or not isinstance(fact.get('value'), list):
            continue
        for row in fact['value']:
            if not isinstance(row, dict) or not {'station_id', 'vestibule_id', 'target_hour', 'horizon_hours', 'features'} <= row.keys():
                continue
            key = row['station_id'], row['target_hour'], row['horizon_hours']
            for feature in row['features']:
                groups[key].append(dict(vestibule_id=row['vestibule_id'],
                    name=feature['name'], value=feature['value'], contribution_z=feature['contribution_z'],
                    evidence_id=fact['id'], source=fact['source'] if 'source' in fact else None))
    result = []
    for (station, hour, horizon), features in sorted(groups.items()):
        strongest = sorted(features, key=lambda r: abs(r['contribution_z']), reverse=True)[:max_features]
        result.append(dict(station_id=station, target_hour=hour, horizon_hours=horizon,
            features=strongest, unit='raw z = log((y+1)/(hourly_reference+1))',
            interpretation='model contribution, not causal effect; no summation across vestibules',
            omitted_feature_count=len(features)-len(strongest)))
    return result


def render_dispatcher(comparison, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status = comparison.get('campaign_status', {})
    full = status.get('full_day_status')
    banner = ('Полный суточный прогон не выполнен: пилот не прошёл приёмку.' if full == 'not_run_due_to_failed_pilot' else
              'Полный суточный прогон не выполнен; результаты пилота ещё проверяются.' if not comparison.get('full_day') else
              'Суточный опыт завершён; допустимость каждого варианта показана отдельно.')
    cards = []
    sections = {'action': 'Действие', 'prediction': 'Прогноз', 'rationale': 'Основание выбора',
                'effects': 'Последствия и альтернативы', 'risks': 'Риски', 'revisit': 'Когда пересмотреть'}
    for explanation in comparison.get('explanations', []):
        arm = explanation['arm']
        card = '<article class="decision" data-arm="'+html.escape(arm, quote=True)+'"><h3>'+html.escape(arm)+' / решение '+html.escape(explanation['decision_id'])+'</h3>'
        card += '<p class="meta">'+html.escape('Объяснение: '+str(explanation.get('source'))+'; модель: '+str(explanation.get('model'))+'; сбой API: '+str(explanation.get('error_code') or 'нет'))+' </p>'
        action = explanation.get('action')
        if isinstance(action, dict):
            label = 'Сохранить текущий график' if action.get('candidate_id') == 'KEEP' else 'Изменить график: '+str(action.get('candidate_id'))
            card += '<h4>Действие</h4><p>'+html.escape(label)+'; '+html.escape(str(action.get('decision_time') or ''))+'</p>'
            if 'forecast_admissible' in action:
                card += '<p>Прогнозная допустимость: '+('подтверждена' if action['forecast_admissible'] is True else 'не подтверждена')+'.</p>'
            card += '<details><summary>Данные решения</summary><pre>'+html.escape(json.dumps(action, ensure_ascii=False, indent=2))+'</pre></details>'
        for section, items in explanation['sections'].items():
            if not items:
                continue
            regular = [r for r in items if r.get('kind') != 'attribution'
                       and not (isinstance(action, dict) and r.get('kind') == 'action')]
            attribution = [r for r in items if r.get('kind') == 'attribution']
            if regular:
                card += '<h4>'+html.escape(sections.get(section, section))+'</h4><ul>'
                card += ''.join('<li>'+html.escape(r['text'])+'<small>'+html.escape(', '.join(r.get('evidence_ids', [])))+'</small></li>' for r in regular)+'</ul>'
            if attribution:
                card += '<details><summary>Измеренные вклады модели по станции и горизонту</summary><p>Модельные вклады не доказывают причинность. Единица B1: raw z = log((y+1)/(hourly_reference+1)); перевод в пассажиров и деньги не выполняется. Вклады разных вестибюлей не складываются. Показаны три наибольших по абсолютной величине вклада в каждой группе.</p>'
                contribution_rows = [[group['station_id'], group['target_hour'], group['horizon_hours'],
                                      feature['vestibule_id'], feature['name'], feature['value'],
                                      f"{feature['contribution_z']:+.6f}"]
                    for group in explanation.get('attribution_groups', []) for feature in group['features']]
                table, _ = _table(['Станция', 'Целевой час', 'Горизонт, ч', 'Вестибюль', 'Признак', 'Значение', 'Вклад, z'], contribution_rows)
                card += table+'</details>'
        card += '</article>'
        cards.append(card)
    audit_rows = [[a, au.get('passed'), au.get('metrics', {}).get('mean_wait_seconds'),
                   au.get('metrics', {}).get('p95_wait_seconds'), ', '.join(au.get('failed_checks', [])) or 'нет']
                  for a, au in comparison.get('audits', {}).items()]
    audit_table, _ = _table(['Вариант', 'Допустим', 'Среднее ожидание, с', 'P95 ожидания, с', 'Нарушения'], audit_rows)
    future_rows = [[r['arm'], r['origin'], r['actual_sum'], r['predicted_sum'], 100*r['wape'] if r.get('wape') is not None else None]
                   for r in comparison.get('post_execution_forecasts', [])]
    future_table, _ = _table(['Вариант', 'Момент прогноза', 'Факт входов', 'Прогноз q50', 'WAPE, %'], future_rows)
    arms = sorted({r['arm'] for r in comparison.get('explanations', [])})
    options = '<option value="all">Все варианты</option>'+''.join('<option>'+html.escape(a)+'</option>' for a in arms)
    doc = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Журнал диспетчера</title>
<style>body{margin:0;background:#edf2f7;color:#183047;font:16px/1.5 system-ui}main{max-width:1100px;margin:30px auto;padding:20px}h1,h2,h3{line-height:1.2}.notice{padding:20px;background:#fff0d9;border-left:5px solid #bf752b}.decision{background:#fff;border:1px solid #cfdae4;border-radius:12px;margin:20px 0;padding:24px}h4{margin:15px 0 4px}ul{margin:0}li{overflow-wrap:anywhere}small{display:block;color:#5b7185;font-size:12px}.meta{color:#5b7185}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border:1px solid #cfdae4;text-align:left}th{background:#dce8f2}select,input{padding:10px;border:1px solid #afbfce;border-radius:6px}details{margin-top:14px}a{color:#155f99}</style>
<main><h1>Красная линия: журнал решений и рисков</h1>'''
    doc += '<p class="notice">'+html.escape(banner)+'</p><p>Это просмотр сохранённых решений и проверенных объяснений. Отображение не отправляет команды поездам.</p>'
    doc += '<p>Пилот: входы пассажиров 06:00–10:30; учёт готовности к посадке до 10:45; наблюдение и пробег до 12:30. Обязательства последнего окна прослеживаются после 10:30.</p>' if not comparison.get('full_day') else ''
    doc += '<h2>Проверка перевозки</h2>'+audit_table
    doc += '<h2>Объяснения решений</h2><label>Вариант <select id="arm">'+options+'</select></label> <input id="search" placeholder="Поиск по объяснению">'+(''.join(cards) or '<p>Подтверждённые объяснения пока отсутствуют.</p>')
    doc += '<h2>После исполнения: прогноз против факта</h2><p>Эти результаты стали известны после завершения целевых интервалов. Они используются для оценки ошибки и не были данными оптимизатора при выборе действия. Суммы относятся к целевым станциям и интервалам данного прогноза; разные моменты выпуска перекрываются.</p>'+future_table
    doc += '<p><a href="report.html">Полный отчёт</a> · <a href="report.pdf">PDF</a> · <a href="explanations.csv">Все объяснения CSV</a> · <a href="post_execution_forecasts.csv">Проверка прогнозов CSV</a></p></main>'
    doc += '''<script>function apply(){const a=document.getElementById('arm').value,q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('.decision').forEach(x=>x.hidden=(a!=='all'&&x.dataset.arm!==a)||!x.textContent.toLowerCase().includes(q));}document.getElementById('arm').addEventListener('change',apply);document.getElementById('search').addEventListener('input',apply);</script></html>'''
    path = output/'dispatcher.html'
    path.write_text(doc, encoding='utf-8')
    return path


def _csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def _format(value):
    if value is None:
        return 'не подтверждено'
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, (int, float)):
        return f'{value:,.2f}'.replace(',', '\u00a0')
    return str(value)


def _table(headers, rows):
    rows = [[_format(v) for v in row] for row in rows]
    markup = '<table><thead><tr>' + ''.join('<th>'+html.escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>'
    markup += ''.join('<tr>' + ''.join('<td>'+html.escape(v)+'</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'
    markdown = '\n| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n'
    markdown += '\n'.join('| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in rows) + '\n'
    return markup, markdown


def _figures(comparison, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    arms, period = list(comparison['economics']), comparison['period_key']
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, side in enumerate(SIDES):
        x = [i + (index-.5)*.35 for i in range(len(arms))]
        values = [comparison['economics'][a][period][side]['candidate_cost_rub']/1e6 for a in arms]
        ax.bar(x, values, width=.35, label=SIDE_NAMES[index])
    ax.set_xticks(list(range(len(arms))), arms)
    ax.set_ylabel('Расчётная стоимость ресурса, млн ₽')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'cost.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    values = [100*max((c.get('missed_first_share') or 0 for c in comparison['audits'][a].get('local_service', {}).get('cohorts', [])), default=0) for a in arms]
    ax.bar(arms, values, color=['#b54735' if v > 5 else '#26755e' for v in values])
    ax.axhline(5, color='#b54735', linestyle='--', label='Локальный предел 5%')
    ax.set_ylabel('Худшая локальная непосадка, %')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'local_misses.png', dpi=150); plt.close(fig)
    return ['cost.png', 'local_misses.png']


def export_pdf(html_path, pdf_path):
    candidates = [Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'),
                  Path('C:/Program Files/Microsoft/Edge/Application/msedge.exe')]
    executable = next((p for p in candidates if p.is_file()), None)
    if executable is None:
        found = shutil.which('msedge') or shutil.which('chromium') or shutil.which('google-chrome')
        executable = Path(found) if found else None
    if executable is None:
        raise RuntimeError('PDF renderer unavailable; report.html and report.md were saved')
    command = [str(executable), '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
               '--no-pdf-header-footer', f'--user-data-dir={(pdf_path.parent/"pdf-browser-profile").resolve()}',
               f'--print-to-pdf={pdf_path.resolve()}', html_path.resolve().as_uri()]
    process = subprocess.run(command, capture_output=True, timeout=55,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (pdf_path.parent/'pdf_renderer.log').write_bytes(process.stderr)
    if process.returncode or not pdf_path.is_file():
        raise RuntimeError('PDF export failed; HTML and Markdown were preserved')
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
    if not reader.pages or not all(term in extracted for term in ('Без СР и КР', 'С СР и КР', '±5%', '±20%', 'Безопасность')):
        raise RuntimeError('PDF text verification failed')
    (pdf_path.parent/'pdf_text.txt').write_text(extracted, encoding='utf-8')
    return dict(pages=len(reader.pages), sha256=sha256(pdf_path), verified_text=True)


def render_reports(comparison, output, *, create_pdf=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    period, full = comparison['period_key'], comparison['full_day']
    tables, blocks, markdown = [], [], []

    def section(title, paragraphs=(), headers=None, rows=None):
        document = '<section><h2>'+html.escape(title)+'</h2>'
        text = '\n## '+title+'\n\n'
        for paragraph in paragraphs:
            document += '<p>'+html.escape(paragraph)+'</p>'
            text += paragraph+'\n\n'
        if headers:
            html_table, md_table = _table(headers, rows)
            document += html_table
            text += md_table
        blocks.append(document+'</section>')
        markdown.append(text)

    title = 'Красная линия: баланс перевозки, ремонтного ресурса и объяснений'
    blocks.append('<h1>'+title+'</h1>')
    markdown.append('# '+title+'\n')
    scope = ('Полный суточный опыт.' if full else 'Пробный опыт: экономика рассчитана только за наблюдаемое окно; пересчёт на день и год не выполняется.')
    warning = ('Все результаты прошли независимые проверки.' if comparison['all_results_admissible'] else
               'Есть нарушения или неполные проверки. Расчётное снижение стоимости таких вариантов не считается допустимой экономией.')
    campaign = comparison.get('campaign_status', {})
    stage_note = ('Полный суточный прогон не выполнен: пилот не прошёл приёмку.'
                  if campaign.get('full_day_status') == 'not_run_due_to_failed_pilot' else
                  'Полный суточный прогон не выполнен; пилот ещё не завершён или ожидает проверки.' if not full else
                  'Показан результат суточного опыта, а не только утреннего окна.')
    section('Область исследования', [scope, warning,
        stage_note,
        'Пилот: входы 06:00–10:30, готовность к посадке до 10:45, наблюдение и экономический пробег до 12:30.' if not full else
        'Полный день: входы 06:00–00:00, учёт конца платформенного потока после полуночи, закрытие наблюдения до 02:00.',
        'Стоимость описывает расход ремонтного ресурса по пробегу. Денежная экономия бюджета, энергия, работа персонала и упущенная выручка не подтверждены.',
        'Исходный график является модельным. Результат одного дня не доказывает переносимость на реальную линию, другие дни или аварии.'])
    section('Безопасность и качество перевозки', headers=['Вариант', 'Допустим', 'Пробег, км', 'Среднее ожидание, с', 'P95 ожидания, с', 'Худшая первая непосадка, %', 'Худшая повторная, %'],
        rows=[[a, au['passed'], au['metrics'].get('train_km'), au['metrics'].get('mean_wait_seconds'), au['metrics'].get('p95_wait_seconds'),
               100*max((r.get('missed_first_share') or 0 for r in au.get('local_service', {}).get('cohorts', [])), default=0),
               100*max((r.get('missed_second_share') or 0 for r in au.get('local_service', {}).get('cohorts', [])), default=0)] for a, au in comparison['audits'].items()])
    section('Проверка ограничений', [
        'Локальные доли относятся к уникальным людям в группе «станция × направление × 15 минут готовности к посадке». Пределы первой и повторной непосадки: 5% и 1%. Незавершённые наблюдения не считаются прошедшими проверку.',
        'Физические ограничения: минимум 113 секунд в будний пик 08:00–10:00 и 17:00–19:00, 150 секунд вне пика, максимум 480 секунд; не более 53 составов на линии; предел посадки 150 человек на вагон. 120 человек является мягким ориентиром комфорта.',
        'Ожидание проверяется на одинаковых пассажирских группах относительно фиксированного исходного графика: среднее +10 секунд и P95 +40 секунд за всё окно и отдельно в каждом наблюдаемом пике.'],
        headers=['Вариант', 'Непройденные проверки', 'Предел нагрузки, чел/вагон', 'Макс. время решения, с'],
        rows=[[a, ', '.join(au['failed_checks']) or 'нет', au['metrics'].get('max_car_load'), au.get('controller', {}).get('maximum_seconds')] for a, au in comparison['audits'].items()])
    worst = []
    for arm, audit in comparison['audits'].items():
        for row in sorted(audit.get('local_service', {}).get('cohorts', []), key=lambda r: r.get('missed_first_share') or 0, reverse=True)[:10]:
            worst.append([arm, row['station_id'], row['direction'], row['interval_start'], row['passenger_count'], row['missed_first_count'], 100*(row.get('missed_first_share') or 0), row.get('status')])
    section('Самые трудные локальные группы', headers=['Вариант', 'Станция', 'Напр.', 'Начало', 'Людей', 'Не сели', '%', 'Статус'], rows=worst)
    comfort_rows, local_wait_rows = [], []
    for arm, audit in comparison['audits'].items():
        for window, row in audit['metrics'].get('comfort', {}).items():
            if window == 'peak':
                continue
            comfort_rows.append([arm, window, row['p95'], row['p99'], row['maximum'],
                row['above_120_car_seconds'], row['above_140_car_seconds'],
                100*row['above_120_time_share'] if row['above_120_time_share'] is not None else None,
                100*row['above_140_time_share'] if row['above_140_time_share'] is not None else None])
        for row in audit['metrics'].get('local_waiting', []):
            local_wait_rows.append([arm, row['station_id'], row['direction'], row['passenger_count'],
                row['mean_wait_seconds'], row['p95_wait_seconds'], row['p99_wait_seconds'],
                row['max_queue'], row['maximum_queue_age_seconds']])
    section('Комфорт в вагонах', [
        'P95/P99 относятся к каждому вагону после посадки при пассажирском отправлении, включая пустые вагоны. Пики представлены отдельно, если попали в наблюдение.',
        'Вагоно-секунды выше 120 и 140 учитывают длительность состояния, включая движение и ожидание с пассажирами. Доли делятся на время положительной загрузки, чтобы простой пустых вагонов не улучшал показатель. Полное наблюдаемое время также сохранено в comparison.json.'],
        ['Вариант', 'Окно', 'P95', 'P99', 'Макс.', '>120, ваг·с', '>140, ваг·с', '>120, % времени', '>140, % времени'], comfort_rows)
    section('Ожидание, очереди и их возраст по платформам', [
        'Среднее и P95/P99 считаются по выбранным группам готовности к посадке с последующим наблюдением. Максимальная очередь относится ко всему наблюдаемому окну; возраст соответствует самому длинному наблюдавшемуся ожиданию человека. Геометрия платформ неизвестна, оценка плотности не выдаётся.'],
        ['Вариант', 'Станция', 'Напр.', 'Людей', 'Среднее, с', 'P95, с', 'P99, с', 'Макс. очередь', 'Макс. возраст, с'], local_wait_rows)
    for reporting_period in ([period, 'annual'] if full else [period]):
        label = 'Сценарная оценка: годовое повторение сопоставимых дней' if reporting_period == 'annual' else 'Стоимость и дельта за день' if full else 'Стоимость и дельта за наблюдаемое окно'
        rows = []
        fields = [('reference_cost_rub', 'Исходная стоимость, ₽'), ('candidate_cost_rub', 'Стоимость варианта, ₽'),
                  ('saving_rub', 'Расчётная дельта стоимости, ₽'), ('saving_percent', 'Расчётная дельта, %'),
                  ('mean_candidate_cost_per_fixed_train_rub', 'Стоимость на один из 68 составов, ₽'),
                  ('mean_saving_per_fixed_train_rub', 'Расчётная дельта на один из 68 составов, ₽')]
        for arm, economy in comparison['economics'].items():
            e = economy[reporting_period]
            for field, name in fields:
                rows.append([arm+' / '+name]+[e[s][field] for s in SIDES])
                for side in SIDES:
                    tables.append(dict(arm=arm, period=reporting_period, side=side, metric=field, value=e[side][field],
                                       admitted=economy['admissible'], accepted_saving_rub=e[side]['saving_rub'] if economy['admissible'] else None))
            rows.append([arm+' / Подтверждённая допустимая экономия, ₽']+[e[s]['saving_rub'] if economy['admissible'] else None for s in SIDES])
        note = [] if reporting_period != 'annual' else [f"Это сценарий {next(iter(comparison['economics'].values()))['annual']['days']} повторений этого дня, а не доказанная экономия годового бюджета."]
        section(label, note, ['Показатель', *SIDE_NAMES], rows)
    rows = []
    for arm, economy in comparison['economics'].items():
        for key, name in [('reference_rub_per_km', 'Исходная стоимость, ₽/км'), ('candidate_rub_per_km', 'Стоимость варианта, ₽/км'),
                          ('saving_rub_per_saved_km', 'Расчётная дельта, ₽/сокращённый км'), ('saving_rub_per_reference_km', 'Расчётная дельта, ₽/исходный км')]:
            rows.append([arm+' / '+name]+[economy[period][s][key] for s in SIDES])
    section('Расчёт на километр', ['Пробег включает пассажирские и непассажирские движения. Для пилота учитываются завершённые прибытия до конца наблюдения, полный будущий график в экономику не включён. Общая начальная поездная граница учитывается одинаково.'], ['Показатель', *SIDE_NAMES], rows)
    for bounded_period in ([period, 'annual'] if full else [period]):
        rows = []
        for arm, economy in comparison['economics'].items():
            for key, name in [('reference_cost_rub', 'Исходная стоимость, ₽'), ('candidate_cost_rub', 'Стоимость варианта, ₽'), ('saving_rub', 'Расчётная дельта, ₽'), ('saving_percent', 'Расчётная дельта, %')]:
                rows.append([arm+' / '+name]+[' … '.join(_format(economy[bounded_period][s]['uncertainty'][key+'_'+bound]) for bound in ('min', 'max')) for s in SIDES])
        section('Погрешности расчёта'+(' / годовой сценарий' if bounded_period == 'annual' else ''), [
            '±5% относится только к рассчитанному пробегу каждого сравниваемого плана. Пробеги двух планов допускают противоположные ошибки. Межремонтные интервалы неизменны.',
            '±20% относится только к ценам СР и КР. Одна цена ремонта применяется к обоим планам; цены СР и КР меняются независимо. Диапазоны являются чувствительностью расчёта, а не доверительными интервалами.'], ['Показатель', *SIDE_NAMES], rows)
    section('Модель и границы переносимости', [
        'Частота обновления прогноза 15 минут не равна исходному шагу обучения. Стандартный стекинг имеет 30-минутные выходы, сдвинутые четвертьчасовые обновления требуют отдельной проверки.',
        'Даты обучения, состав ансамбля, хеши и неизвестные даты публикации внешних признаков сохранены в контрактах. Причинные объяснения признаков допустимы только при измеренных вкладах.'],
        ['Вариант', 'Состав модели', 'Конец обучения', 'Сдвинутые обновления'],
        [[a, ', '.join((c.get('model_contract') or {}).get('components', [])) or 'исходный график / см. контракт',
          str({k: v for k, v in (c.get('model_contract') or {}).items() if 'train_end' in k}),
          (c.get('model_contract') or {}).get('shifted_updates_are_experimental')] for a, c in comparison.get('contracts', {}).items()])
    model_evidence = comparison.get('model_evidence', {})
    section('Независимое подтверждение ансамбля', [
        'Статус не берётся из флага результата. Перепроверяются хеши B1, трёх GRU, B2, стекинга, комплект, сверка исходного уравнения B1 и JUnit реальных модельных тестов. Хеши тестового кода и входного manifest должны совпасть.',
        'Данные станционных потоков и веса GRU в новой поставке совпадают с прежними; подключение полного ансамбля добавляет B1/B2 и сохранённый стекинг, а не новую версию этих данных.'],
        ['Проверка', 'Пройдена'], [[k, v] for k, v in sorted(model_evidence.get('checks', {}).items())] or [['Подтверждение', 'не выполнено']])
    if comparison.get('post_execution_forecasts'):
        section('После исполнения: прогноз против факта', [
            'Истина из будущих целевых интервалов здесь используется только для оценки. Эти значения не содержались в исходных пакетах решения и не выдаются за известные заранее. Перекрывающиеся прогнозы не являются независимыми наблюдениями.'],
            ['Вариант', 'Выпуск прогноза', 'Целей', 'Факт', 'Прогноз q50', 'Прогноз решения', 'WAPE, %'],
            [[r['arm'], r['origin'], r['n'], r['actual_sum'], r['predicted_sum'], r.get('decision_predicted_sum'),
              100*r['wape'] if r.get('wape') is not None else None] for r in comparison['post_execution_forecasts']])
    explanation = comparison.get('explanation_summary', {})
    section('Объяснения YandexGPT и отказоустойчивость', [
        'YandexGPT выбирает и группирует ID проверенных фактов. Текст и числа формируются из этих фактов; свободные новые причины и изменение плана запрещены. Сбой API заменяет ответ шаблоном и не меняет поездное управление.',
        'Реальная проверка API и объяснения всех решений являются разными проверками. Наличие одного успешного smoke не доказывает качество всех объяснений.',
        'Тайм-аут, отсутствие допустимого варианта, резервное KEEP и неподтверждённая доставка внешнего уведомления отмечаются отдельно. Автоматическое восстановление считается проверенным только при сохранённом испытании.'],
        ['Показатель', 'Значение'], [[k, v] for k, v in explanation.items()] or [['Покрытие объяснений', 'см. журналы и отдельный smoke; сводная проверка не выполнена']])
    probes = comparison.get('incident_probes', [])
    if probes:
        section('Диагностические остановки составов', [
            'Синтетическая остановка одного состава на 3, 8 и 15 минут проверяет распространение задержек. Заданная длительность известна эксперименту; будущие фактические входы используются только для оценки. Это не причинный прогон оптимизатора с неизвестной длительностью происшествия.',
            'Сохранены составы и удержания отправлений в затронутом направлении; экономические снятия подавлены. Сигнализация, блок-участки, тормозные расстояния и диспетчерские разрешения не проверены. Поэтому прохождение модельных интервалов не подтверждает реальную безопасность.',
            'Восстановление пассажирских перевозок не завершено в случаях с оставшимися пассажирами в поездах. Время расчёта сравнивается с 480 секундами; доставка внешнего уведомления неизвестна. Экономия в этих случаях не подтверждена.'],
            ['Остановка, мин', 'Движение', 'Интервал мин/макс, с', 'Непосадка 1/2, %', 'Δ ожидания mean/P95, с', 'Макс. очередь', 'Остались в поездах', 'Восстановление', 'Расчёт, с'],
            [[r['duration_minutes'], r['movement_passed'],
              str(r['minimum_headway'])+' / '+str(r['maximum_headway']),
              _format(100*r['worst_first_missed_share'])+' / '+_format(100*r['worst_second_missed_share']) if r['worst_first_missed_share'] is not None and r['worst_second_missed_share'] is not None else None,
              _format(r['mean_wait_delta_seconds'])+' / '+_format(r['p95_wait_delta_seconds']),
              r['maximum_queue'], r['remaining_onboard'], 'завершено' if r['recovery_complete'] else 'не завершено', r['elapsed_seconds']]
             for r in probes])
        section('Реакция на происшествие и уведомления', [
            'Ни один диагностический случай не считается допуском к управлению. Подтверждён только сохранённый результат процесса; отдельный независимый аудит сигнализации не выполнен. Полные журналы доступны по путям в incident_probes.csv.'],
            ['Остановка, мин', 'Статус расчёта', '≤8 минут', 'Доставка измерена', 'Будущий факт для оценки', 'Причинная рекомендация', 'Сигнализация проверена'],
            [[r['duration_minutes'], r['receipt_status'], r['process_within_eight_minutes'], r['external_delivery_measured'],
              r['future_actual_used_for_evaluation'], r['causal_operational_recommendation'], r['signalling_verified']] for r in probes])
    images = _figures(comparison, output)
    section('Графики', ['Расчётная стоимость показана и для недопустимых вариантов. Статус проверки определяет право назвать дельту допустимой экономией.'])
    for image in images:
        blocks.append('<figure><img src="'+image+'"></figure>')
        markdown.append('\n!['+image+']('+image+')\n')
    per_train_rows = []
    for arm, economy in comparison['economics'].items():
        detail = []
        for train in economy['per_train']:
            for p in ([period, 'annual'] if full else [period]):
                row = train[p]
                for side in SIDES:
                    per_train_rows.append(dict(arm=arm, train_id=train['train_id'], train_type=train['train_type'], period=p, side=side,
                        reference_km=row['reference_km'], candidate_km=row['candidate_km'], saved_km=row['saved_km'],
                        reference_cost_rub=row[side]['reference_cost_rub'], candidate_cost_rub=row[side]['candidate_cost_rub'],
                        diagnostic_saving_rub=row[side]['saving_rub'], saving_percent=row[side]['saving_percent'],
                        accepted_saving_rub=row[side]['saving_rub'] if economy['admissible'] else None,
                        saving_rub_min=row[side]['uncertainty']['saving_rub_min'], saving_rub_max=row[side]['uncertainty']['saving_rub_max']))
                for field, label in [('candidate_cost_rub', 'Стоимость'), ('saving_rub', 'Расчётная дельта')]:
                    detail.append([train['train_id']+' / '+train['train_type'], ('Годовой сценарий' if p == 'annual' else 'День' if full else 'Окно')+' / '+label]+[row[s][field] for s in SIDES])
        section('Каждый состав: '+arm, ['Все 68 составов включены, в том числе не работавшие. Среднее на состав не подменяет индивидуальный расчёт. Дельты недопустимого варианта остаются диагностическими.'], ['Состав', 'Показатель', *SIDE_NAMES], detail)
    for row in comparison.get('explanations', []):
        if row['arm'] != 'balanced':
            continue
        section('Объяснение решения '+row['decision_id'], [
            'Источник: '+str(row['source'])+'; модель: '+str(row.get('model'))+'; ID контекста: '+row['context_hash'],
            'Подтверждены соответствие пакету фактов и неизменность действия.'],
            ['Секция', 'Объяснение', 'ID факта / источник'],
            [[name, 'Подробные измеренные вклады сгруппированы ниже; полный список сохранён в исходном пакете фактов.' if item.get('kind') == 'attribution' else item['text'], ', '.join(item['evidence_ids'])+' / '+item['source']]
             for name, items in row['sections'].items() for item in items])
        groups = sorted(row.get('attribution_groups', []), key=lambda g: max((abs(f['contribution_z']) for f in g['features']), default=0), reverse=True)[:5]
        if groups:
            section('Измеренные вклады: решение '+row['decision_id'], [
                'Показаны пять групп с крупнейшими по модулю вкладами raw z и до трёх признаков в каждой. Это вклад часового B1 в log((y+1)/(hourly_reference+1)), а не изменение числа пассажиров, стоимости или причинный эффект. Вестибюли не суммируются. Все группы доступны в dispatcher.html и исходных фактах.'],
                ['Станция', 'Час цели', 'Горизонт, ч', 'Вестибюль', 'Признак', 'Вклад, z'],
                [[g['station_id'], g['target_hour'], g['horizon_hours'], f['vestibule_id'], f['name'], f"{f['contribution_z']:+.6f}"] for g in groups for f in g['features']])
    files = comparison.get('input_inventory', [])
    section('Журналы и воспроизводимость', [
        f'Зафиксировано входных файлов: {len(files)}. Полный список, размеры и SHA-256 находятся в input_inventory.csv и comparison.json.',
        'Сохранены результаты, движение, пассажирские контрольные точки, прогнозы, альтернативы, время поиска, решения и пакеты фактов. Ключи API и файлы .env в инвентаризацию не включены.',
        'Независимый аудит перепроверяет физику, сохранённые журналы и контрольную точку. Полное повторное пассажирское моделирование в отчётном аудите не выполняется.'])
    css = '''@page{size:A4;margin:13mm}body{font:9pt/1.35 "Segoe UI",Arial,sans-serif;color:#192b40}h1{font-size:19pt}h2{font-size:14pt;break-after:avoid}section{break-before:page}section:first-of-type{break-before:auto}table{width:100%;border-collapse:collapse;font-size:7.7pt}th,td{border:1px solid #ccd5df;padding:4px;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f7;text-align:left}thead{display:table-header-group}tr{break-inside:avoid}p{margin:8px 0}figure{margin:15px 0;break-inside:avoid}img{width:100%}'''
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>'+title+'</title><style>'+css+'</style><body>'+''.join(blocks)+'</body></html>'
    (output/'report.html').write_text(document, encoding='utf-8')
    (output/'report.md').write_text('\n'.join(markdown), encoding='utf-8')
    (output/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    _csv(output/'economics.csv', tables)
    _csv(output/'per_train.csv', per_train_rows)
    _csv(output/'input_inventory.csv', files)
    _csv(output/'local_cohorts.csv', [dict(arm=a, **row) for a, au in comparison['audits'].items() for row in au.get('local_service', {}).get('cohorts', [])])
    _csv(output/'waiting_budgets.csv', [dict(arm=a, **row) for a, au in comparison['audits'].items() for row in au.get('waiting_budget', {}).get('windows', [])])
    _csv(output/'decisions.csv', [dict(arm=a, **row) for a, result in comparison.get('results_summary', {}).items() for row in result['decisions']])
    _csv(output/'explanations.csv', comparison.get('explanations', []))
    _csv(output/'local_waiting.csv', [dict(arm=a, **row) for a, au in comparison['audits'].items() for row in au['metrics'].get('local_waiting', [])])
    _csv(output/'comfort.csv', [dict(arm=a, window=name, **row) for a, au in comparison['audits'].items() for name, row in au['metrics'].get('comfort', {}).items()])
    _csv(output/'post_execution_forecasts.csv', comparison.get('post_execution_forecasts', []))
    _csv(output/'incident_probes.csv', probes)
    render_dispatcher(comparison, output)
    manifest = dict(pdf_created=False, annual_scenario_emitted=full, source_results_overwritten=False,
                    full_fleet_size=68, side_columns=list(SIDE_NAMES), per_train_csv_rows=len(per_train_rows),
                    input_file_count=len(files), all_results_admissible=comparison['all_results_admissible'])
    manifest.update(ensemble_verified=comparison.get('ensemble_verified') is True,
                    campaign_status=comparison.get('campaign_status', {}),
                    decision_explanations_in_dispatcher=True)
    if create_pdf:
        manifest['pdf'] = export_pdf(output/'report.html', output/'report.pdf')
        manifest['pdf_created'] = True
    manifest['artifacts'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'verification.json'}
    (output/'verification.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def write_status_report(root, output, *, incidents=(), create_pdf=False):
    """Summarize saved evidence only, without a simulation or engine audit."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {p.parent.name: read(p) for p in root.glob('*/result.json')}
    contracts = {p.parent.name: read(p) for p in root.glob('*/contract.json')}
    pending = sorted(set(contracts)-set(results))
    summary, explanations = collect_explanations(root)
    example = next((e for e in explanations if e['source'] == 'yandexgpt' and e['arm'] == 'balanced' and e['decision_id'] == '6300'),
                   next((e for e in explanations if e['source'] == 'yandexgpt'), None))
    html_blocks, md_blocks = [], ['# Состояние GOAL: исправления и сохранённые результаты\n']

    def section(title, text, headers=(), rows=()):
        markup, markdown = _table(headers, rows) if headers else ('', '')
        html_blocks.append('<h2>'+html.escape(title)+'</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in text)+markup)
        md_blocks.append('\n## '+title+'\n\n'+'\n\n'.join(text)+markdown)

    section('Статус и границы результата', [
        'Длинные прогоны остановлены по просьбе пользователя. Полный суточный прогон не выполнен; годовую экономию по этим данным не заявляем.',
        'Это чтение сохранённых файлов. Повторный engineaudit, моделирование и API-запросы при подготовке этого документа не выполнялись.',
        'Завершённые варианты: '+(', '.join(sorted(results)) or 'нет')+'. Незавершённые: '+(', '.join(pending) or 'нет')+'.',
        'Модель интегрирована, но цель ≥1% допустимой экономии за полный день не подтверждена.'])
    section('Исправления и проверки', [
        'В CP добавлена проверка обязательного продолжения движения: разрыв графика без допустимого следующего отправления не может молча пройти. Сохраняются 113 секунд в пик и 150 вне пика.',
        'Отчёт требует независимых hashes весов, данных, исходников и модельных тестов; одиночный ensemble_contract_verified не даёт выполнить GOAL.',
        '14 реальных тестов полного ансамбля прошли. Сверка исходного уравнения B1, три GRU, B2 и сохранённый стекинг подтверждены отдельными артефактами model_verification.',
        'Первый общий прогон: 1999 passed, 17 failed. Для 14 report failures установлен импорт старого snapshot; тесты привязаны к текущему корневому файлу. После полной сборки тестов report-regressions: 27 passed. Весь suite после исправлений повторно не проверялся.'])
    preview = root/'preview/comparison.json'
    prior = read(preview) if preview.is_file() else {}
    reference = prior.get('audits', {}).get('reference', {})
    section('Безопасность и очереди', [
        'Сохранённый независимый аудит reference v2 прошёл физику, hashes и пассажирский учёт, но провалил local_boarding_service.',
        'На «Академической», направление −1, 08:00–08:15: первая непосадка 40,05%, повторная 6,97%. Требования 5% и 1% не выполнены. Исправления кода не доказывают устранение этой очереди.',
        'KEEP с forecast_admissible=false означает резервное сохранение графика, а не подтверждённое допустимое решение.',
        'В сохранённом reference пик08–10: P95/P99 нагрузки124/150 человек — каждый вагон после посадки при пассажирском отправлении; 18024 наблюдения. Вагоно-время>120:5,782%, >140:1,926% занятого времени. Это другая методика, чем усреднение поездов или событий высадки; старые110/141 не объявляются эффектом улучшения.'] if reference else [
        'Повторный аудит не выполнен. Допустимость незавершённых вариантов не подтверждена; KEEP при forecast_admissible=false не является подтверждённым решением.'])
    econ_rows = []
    if 'reference' in results:
        ledger = lambda r: {'per_train': r['metrics'].get('actual_per_train', [])}
        if ledger(results['reference'])['per_train']:
            for arm, result in sorted(results.items()):
                if not ledger(result)['per_train']:
                    continue
                economy = economy_comparison(ledger(results['reference']), ledger(result), admissible=False)
                for key, label in [('candidate_cost_rub', 'Стоимость ресурса, ₽'), ('saving_rub', 'Диагностическая дельта, ₽'), ('saving_percent', 'Диагностическая дельта, %')]:
                    econ_rows.append([arm+' / '+label]+[economy['daily'][s][key] for s in SIDES])
    section('Экономика наблюдаемого окна', [
        'Только фактический расчётный пробег до12:30; будущий график не включён. Ни одна показанная дельта не принимается как допустимая экономия. ТО, ТР и СР/КР описывают износ ремонтного ресурса; денежные выплаты и упущенная выручка неизвестны.',
        'Погрешность рассчитанного пробега ±5%; цены СР и КР ±20%. Межремонтные интервалы неизменны. Полные таблицы на километр и по68 составам доступны в сохранённых пилотных отчётах; переноса на год нет.'],
        ['Показатель', *SIDE_NAMES], econ_rows)
    section('YandexGPT: настоящий ответ и резервный шаблон', [
        f"Сохранено и проверено {summary['verified_count']} объяснений: API {summary['yandexgpt_count']}, шаблон {summary['template_count']}. Изменение команд LLM запрещено; ошибки ответа заменяются шаблоном.",
        f"Токены по сохранённым ответам: {summary['total_tokens']}; стоимость API в рублях неизвестна. Предыдущий smoke прошёл на точной модели yandexgpt-5-lite; модель не подменялась.",
        ('Настоящий API-пример: '+example['arm']+' / '+example['decision_id']+'; модель '+str(example['model'])+'; '+_format(example['elapsed_seconds'])+' с. Действие KEEP; прогнозная допустимость не подтверждена. Факты решения не изменены.' if example else 'Настоящий API-пример в этих файлах отсутствует.')])
    if example:
        sample_rows = [[name, item['text']] for name in ('rationale', 'risks', 'revisit')
                       for item in example['sections'].get(name, []) if item.get('kind') != 'attribution']
        section('Пример проверенного объяснения', ['Исходные файлы: '+example['packet_file']+'; '+example['explanation_file']+'.'], ['Секция', 'Проверенные факты'], sample_rows[:8])
    probes = collect_incident_probes(incidents)
    if probes:
        section('Диагностические остановки', [
            'Это синтетические случаи с заданной длительностью. Будущие данные используются для оценки, не для причинного управления. Сигнализация и блок-участки не проверены, доставка уведомлений неизвестна, экономия не подтверждена.',
            'Процесс расчёта укладывается в480 секунд; при8/15 мин восстановление перевозки к10:15 не завершено.'],
            ['Стоп, мин', 'Движение', 'Мин/макс, с', 'Макс. очередь', 'Δ mean/P95, с', 'Остались в поездах'],
            [[r['duration_minutes'], r['movement_passed'], f"{r['minimum_headway']} / {r['maximum_headway']}", r['maximum_queue'],
              _format(r['mean_wait_delta_seconds'])+' / '+_format(r['p95_wait_delta_seconds']), r['remaining_onboard']] for r in probes])
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Статус GOAL</title><style>@page{size:A4;margin:14mm}body{font:10pt/1.4 "Segoe UI",Arial;color:#183047;max-width:1050px;margin:auto}h2{font-size:14pt;break-after:avoid}table{border-collapse:collapse;width:100%;font-size:9pt}th,td{border:1px solid #ccd5df;padding:6px;overflow-wrap:anywhere}thead{display:table-header-group}tr{break-inside:avoid}th{background:#eaf1f7}h1{font-size:20pt}</style><h1>Состояние GOAL: исправления и сохранённые результаты</h1>'+''.join(html_blocks)+'</html>'
    (output/'status.md').write_text('\n'.join(md_blocks), encoding='utf-8')
    (output/'status.html').write_text(document, encoding='utf-8')
    comparison = dict(prior, explanations=explanations, explanation_summary=summary,
                     post_execution_forecasts=collect_post_execution_forecasts(root),
                     campaign_status={'full_day_status': 'not_run_due_to_failed_pilot'}, full_day=False)
    dispatcher = render_dispatcher(comparison, output)
    dispatcher.write_text(dispatcher.read_text(encoding='utf-8').replace('href="report.html"', 'href="status.html"').replace('href="report.pdf"', 'href="status.pdf"'), encoding='utf-8')
    _csv(output/'explanations.csv', explanations)
    _csv(output/'post_execution_forecasts.csv', comparison['post_execution_forecasts'])
    payload = dict(completed_arms=sorted(results), unfinished_arms=pending, engine_audit_performed=False,
                   goal_achieved=False, full_day_completed=False, explanation_summary=summary,
                   actual_api_example=example, incident_probes=probes)
    if create_pdf:
        payload['pdf'] = export_pdf(output/'status.html', output/'status.pdf')
    (output/'status.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--annual-days', type=int, default=250)
    parser.add_argument('--no-pdf', action='store_true')
    parser.add_argument('--incidents', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    output = args.output or args.root/'review'
    comparison = compile_comparison(args.root, annual_days=args.annual_days)
    comparison['incident_probes'] = collect_incident_probes(args.incidents)
    comparison['input_inventory'] = inventory(args.root, excluded=output)
    result = render_reports(comparison, output, create_pdf=not args.no_pdf)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
