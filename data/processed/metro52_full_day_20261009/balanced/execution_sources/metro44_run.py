"""Continuous GOAL 4.4 arms using one frozen scenario and cached entry forecasts."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from goal43_policy import minimum_headway, parameter
from goal43_replay import save_run
from metro_daily_run import compact_decision, read_forecast
from metro_ml_compare import _metrics
from metro44_mpc import forecast_section_rates, solve_joint_profile
from metro44_planner import rebuild_timetable
from metro44_selection import choose_candidate
from passenger_groups import GroupedPassengerSimulation
from rolling import decide_operational
from simulation import digest, simulate


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'data/processed/metro_daily_experiments_20261009'
DEFAULT_OUTPUT = ROOT / 'data/processed/metro44_20261009'
ARMS = ('FIXED_SCHEDULE', 'STANDARD', 'MPC_BASELINE_FORECAST', 'MPC_ML', 'MPC_ORACLE')
SOURCES = ('metro44_run.py', 'metro44_planner.py', 'metro44_mpc.py', 'metro44_selection.py',
           'metro_daily.py', 'metro_daily_run.py', 'metro_ml.py', 'rolling.py', 'dwell.py',
           'passenger_groups.py', 'goal43_replay.py', 'metro_ml_compare.py')


def forecast_source(strategy):
    if strategy not in ARMS:
        raise ValueError('unknown GOAL 4.4 strategy')
    return {'FIXED_SCHEDULE': None, 'STANDARD': 'MEDIAN', 'MPC_BASELINE_FORECAST': 'MEDIAN',
            'MPC_ML': 'ML', 'MPC_ORACLE': 'ORACLE'}[strategy]


def control_times(service_end):
    return list(range(3600, service_end, 900))


def departure_end(admission_end, strategy):
    forecast_source(strategy)
    return admission_end + (0 if strategy == 'FIXED_SCHEDULE' else 600)


def station_offsets(data):
    stations = [s for s in data['stations'] if s['kind'] == 'line']
    tracks = {(r['from_station'], r['to_station']): r for r in data['tracks']}
    offsets = {}
    for direction, order in (('up', list(range(len(stations)))), ('down', list(reversed(range(len(stations)))))):
        values, elapsed = [0] * len(stations), 0
        for left, right in zip(order, order[1:]):
            elapsed += tracks[stations[left]['station_id'], stations[right]['station_id']]['travel_seconds']
            elapsed += stations[right]['stop_seconds']
            values[right] = elapsed
        offsets[direction] = values
    return offsets


def arm_folder(output, strategy, pilot_steps):
    forecast_source(strategy)
    if pilot_steps is not None and (isinstance(pilot_steps, bool) or not isinstance(pilot_steps, int)
                                    or not 1 <= pilot_steps <= 68):
        raise ValueError('pilot steps must be an integer from 1 to 68')
    suffix = f'_pilot_{pilot_steps}' if pilot_steps is not None else ''
    return Path(output) / 'controller' / f'{strategy}{suffix}'


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _publish_once(path, encoded, *, compressed=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'refusing to overwrite {path}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if compressed:
                with gzip.GzipFile(fileobj=stream, mode='wb', compresslevel=3, mtime=0) as packed:
                    packed.write(encoded)
            else:
                stream.write(encoded)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f'refusing to overwrite {path}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_gzip_once(path, value):
    _publish_once(path, json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                  allow_nan=False).encode('utf-8'), compressed=True)


def write_json_once(path, value):
    _publish_once(path, json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False).encode('utf-8'))


def archive_sources(output, hashes):
    for name, sha in hashes.items():
        encoded = (ROOT / name).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != sha:
            raise ValueError('source changed before archival')
        target = Path(output) / 'source_versions' / f'{sha}_{name}'
        if not target.exists():
            try:
                _publish_once(target, encoded)
            except ValueError:
                if not target.exists() or _hash(target) != sha:
                    raise
        if _hash(target) != sha:
            raise ValueError('archived source digest mismatch')


def load_resume(folder, contract_identity):
    paths = sorted(Path(folder).glob('resume_*.json.gz'))
    if not paths:
        return None
    with gzip.open(paths[-1], 'rt', encoding='utf-8') as stream:
        record = json.load(stream)
    if record.get('contract_identity') != contract_identity:
        raise ValueError('resume source/input/mode contract changed')
    if record['state']['second'] != record['second'] or record['second'] != 3600 + (record['next_index'] - 1) * 900:
        raise ValueError('resume clock is inconsistent')
    if type(record['next_index']) is not int or not 1 <= record['next_index'] <= 68:
        raise ValueError('resume clock index is invalid')
    expected = [Path(folder) / f'resume_{second:05}.json.gz'
                for second in range(3600, 3600 + record['next_index'] * 900, 900)]
    if paths != expected:
        raise ValueError('resume executed prefix is incomplete')
    for key in ('data', 'state'):
        if f'{key}_digest' in record and digest(record[key]) != record[f'{key}_digest']:
            raise ValueError(f'resume {key} digest mismatch')
    return record


def prepare_shared(source, output):
    source, output = Path(source), Path(output)
    with gzip.open(source / 'baseline.json.gz', 'rt', encoding='utf-8') as stream:
        shared = json.load(stream)
    if shared['checkpoint']['second'] != 3600 or shared['meta']['initial_count'] != 33:
        raise ValueError('scenario must retain the shared 33-train initial state and 07:00 checkpoint')
    if digest(shared['groups']) != shared['groups_digest'] or digest(shared['checkpoint']) != shared['checkpoint_digest']:
        raise ValueError('shared passenger scenario digest changed')
    paths = [source / 'baseline.json.gz', source / 'observations.parquet', *sorted((source / 'forecasts').glob('*.json'))]
    contract = dict(groups_digest=shared['groups_digest'], checkpoint_digest=shared['checkpoint_digest'],
                    source_folder=str(source.resolve()), initial_count=33, actual_line=1,
                    input_hashes={str(p.relative_to(source)).replace('\\', '/'): _hash(p) for p in paths},
                    baseline_forecast='one historical median predictor; legacy cache filenames retained',
                    oracle_operationally_available=False, fixed_fleet_size=68)
    path = output / 'shared_scenario.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != contract:
            raise ValueError('shared scenario/input contract changed')
    else:
        write_json_once(path, contract)
    return shared, contract


def closed_per_train(data, roster, *, transport_work=None):
    if transport_work is None:
        motion = simulate(data, data['plans'][0]['plan_id'])
        if not motion['obligations_closed']:
            raise ValueError('resource comparison requires closed movement obligations')
        transport_work = motion['transport_work']
    types = {row['train_id']: row['type'] for row in roster}
    work = {row['train_id']: row for row in transport_work}
    if types.keys() != work.keys() or len(types) != 68:
        raise ValueError('closed mileage must retain all 68 fixed-fleet train identities')
    return [dict(train_id=t, train_type=types[t], km=float(work[t]['passenger_train_km'])
                 + float(work[t]['nonpassenger_train_km'])) for t in types]


def directional_profile(data, result):
    terminals = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    return [dict(terminal=terminal, start_second=row['start_second'], end_second=row['end_second'],
                 headway_seconds=row[f'{direction}_headway_seconds'])
            for row in result['slots'] for terminal, direction in ((terminals[0], 'up'), (terminals[-1], 'down'))]


def public_decision(decision):
    saved = compact_decision(decision)
    for row, original in zip(saved['candidate_table'], decision['candidate_table']):
        if 'audit' in row:
            row['audit'] = {k: v for k, v in row['audit'].items() if not k.startswith('_')}
        for trial, source in zip(row['trials'], original.get('trials', [])):
            trial['future_cohort'] = {k: v for k, v in source.get('mature_cohort', {}).items()
                                     if k in ('complete', 'arrived', 'boarded', 'unboarded',
                                              'mean_wait_seconds', 'p95_wait_seconds', 'scope')}
            exposure = source.get('future_exposure', source['policy_exposure'])
            trial['future_policy_summary'] = {k: exposure.get(k) for k in ('admissible', 'global', 'windows')}
            trial['initial_unboarded'] = source.get('initial_obligations', {}).get('unboarded')
            for key in ('peak_boarding_service', 'verification_scope'):
                if key in source: trial[key] = copy.deepcopy(source[key])
    return saved


def estimated_keep_wait_budget(data, bundle, second, *, fallback_seconds=90, platform_delay=120):
    """Same forecast/S1 minute arrivals against known KEEP station departures.

    This waiting surrogate excludes capacity and queue effects. It constrains the
    aggregate search only; the exact selector still uses verified KEEP mean/p95.
    """
    origin = datetime.fromisoformat(data['period_start'])
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    positions = {s: i for i, s in enumerate(ids)}
    trips = {t['trip_id']: t for p in data['plans'] for t in p['trips']}
    tracks = {t['track_id']: t for t in data['tracks']}
    departures = defaultdict(list)
    for row in data.get('leg_timing', []):
        trip = trips.get(row['trip_id'])
        if trip is None or not trip['passenger']:
            continue
        track = tracks[trip['route'][row['leg_index']]]
        left, right = track['from_station'], track['to_station']
        if left in positions and right in positions:
            direction = 'up' if positions[right] > positions[left] else 'down'
            departures[left, direction].append(row['departure_second'])
    for key in departures:
        departures[key] = sorted(set(departures[key]))
    end = max(int((datetime.fromisoformat(row['interval_end']) - origin).total_seconds()) for row in bundle['rows'])
    mature_end = max(second, end - 600)
    weighted_wait = weighted_half_gap = known = unknown = headway_known = missing_previous = 0
    for row in bundle['rows']:
        quarter = int((datetime.fromisoformat(row['interval_start']) - origin).total_seconds())
        i, destinations = positions[row['station_id']], len(ids) - 1
        quotient, remainder = divmod(row['predicted_count'], 15)
        for minute in range(15):
            entry_second = quarter + minute * 60
            if not second <= entry_second < mature_end:
                continue
            count = quotient + (minute < remainder)
            base, extra = divmod(count, destinations)
            down = base * i + min(i, extra)
            for direction, amount in (('down', down), ('up', count - down)):
                if not amount:
                    continue
                at = entry_second + platform_delay
                times = departures[row['station_id'], direction]
                index = bisect_left(times, at)
                if index == len(times):
                    unknown += amount
                else:
                    known += amount
                    weighted_wait += amount * (times[index] - at)
                    if index == 0:
                        missing_previous += amount
                    else:
                        headway_known += amount
                        weighted_half_gap += amount * (times[index] - times[index - 1]) / 2
    mean = weighted_wait / known if known else None
    half_mean = weighted_half_gap / headway_known if headway_known else None
    fallback = bool(unknown or missing_previous or not known)
    correction = None if fallback else max(0, half_mean - mean)
    reasons = (['next_departure_missing'] if unknown else []) + (
        ['previous_departure_missing'] if missing_previous else []) + (['no_future_entries'] if not known and not unknown else [])
    return dict(mean_wait_budget_seconds=float(fallback_seconds) if fallback else mean + correction,
        freeflow_keep_mean_seconds=mean, known_headway_half_mean_seconds=half_mean,
        surrogate_phase_correction_seconds=correction, headway_passenger_weight=headway_known,
        unknown_headway_passenger_weight=unknown + missing_previous, fallback_reasons=reasons,
        passenger_weight=known + unknown, unknown_passenger_weight=unknown, fallback_assumption_used=fallback,
        fallback_seconds=fallback_seconds if fallback else None, mature_end_second=mature_end,
        platform_delay_seconds=platform_delay, capacity_checked=False,
        scope='known KEEP station clocks; passenger-weighted local half-gaps calibrate search only; exact KEEP mean/p95 unchanged; capacity and queues excluded')


def keep_wait_budgets(rows):
    keep = next((r for r in rows if r['candidate_id'] == 'KEEP'), None)
    if not keep or keep.get('status') != 'evaluated' or not keep.get('passengers_checked') or not keep.get('physical_passed'):
        return None, None
    trials = keep.get('trials', [])
    if not keep.get('service_passed') or not trials or not all(
            t.get('future_exposure', t.get('policy_exposure', {})).get('admissible') is True
            and t.get('headways', {}).get('passed') is True
            and t.get('mature_cohort', {}).get('complete') is True
            and t.get('initial_obligations', {}).get('complete') is True for t in trials):
        return None, None
    cohorts = [t['mature_cohort'] for t in trials]
    means = [t['mean_wait_seconds'] for t in cohorts]
    p95s = [t['p95_wait_seconds'] for t in cohorts]
    if any(v is None for v in means + p95s):
        return None, None
    return max(means), max(p95s)


def selected_plan(selection, candidates, table):
    keep = next(r for r in candidates if r['candidate_id'] == 'KEEP')
    name = selection.get('selected_candidate_id') or selection.get('mitigation_candidate_id') or 'KEEP'
    candidate = next((r for r in candidates if r['candidate_id'] == name), None)
    row = next((r for r in table if r['candidate_id'] == name), None)
    validated = (row is not None and row.get('status') == 'evaluated' and row.get('passengers_checked') is True
                 and row.get('physical_passed') is True and bool(row.get('trials'))
                 and all(t.get('headways', {}).get('passed') is True for t in row['trials']))
    return (candidate['_data'], name) if candidate and candidate.get('_data') and validated else (keep['_data'], 'KEEP')


def bounded_decision(callback, budget_seconds, *, cancel_event=None):
    if isinstance(budget_seconds, bool) or not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError('decision budget must be finite and nonnegative')
    began = time.monotonic(); deadline = began + budget_seconds
    cancelled = cancel_event if cancel_event is not None else Event()
    answers = Queue(maxsize=1)

    def timeout():
        cancelled.set()
        return dict(completed=False, result=None, elapsed_seconds=time.monotonic() - began,
                    selection=dict(status='evaluation_timeout', selected_candidate_id=None,
                                   mitigation_candidate_id=None, fallback_candidate_id='KEEP',
                                   release_authorized=False, economic_optimality=False,
                                   economic_improvement_claimed=False))

    def calculate():
        try:
            result = callback(deadline, cancelled)
            if not cancelled.is_set():
                answers.put((True, result))
        except Exception as exc:
            if not cancelled.is_set():
                answers.put((False, exc))

    if time.monotonic() >= deadline or cancelled.is_set():
        return timeout()
    Thread(target=calculate, name='goal44-decision', daemon=True).start()
    try:
        successful, result = answers.get(timeout=max(0, deadline - time.monotonic()))
    except Empty:
        return timeout()
    if time.monotonic() >= deadline or isinstance(result, TimeoutError):
        return timeout()
    if not successful:
        raise result
    return dict(completed=True, result=result, elapsed_seconds=time.monotonic() - began)


def _slots(data, bundle, at, state):
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    rows = forecast_section_rates(bundle, ids, at, period_start=data['period_start'], checkpoint=state,
                                  station_offsets_seconds=station_offsets(data))
    unshifted = forecast_section_rates(bundle, ids, at, period_start=data['period_start'])
    waiting = {(r['start_second'], r['end_second']): r for r in unshifted}
    origin = datetime.fromisoformat(data['period_start'])
    now = int((datetime.fromisoformat(at) - origin).total_seconds())
    mature_end = max(now, max((r['end_second'] for r in rows), default=now) - 600)
    for row in rows:
        source = waiting[row['start_second'], row['end_second']]
        duration = max(0, min(row['end_second'], mature_end) - max(row['start_second'], now))
        fraction = duration / (row['end_second'] - row['start_second'])
        row.update(up_wait_entries=source['up_entries'] * fraction,
                   down_wait_entries=source['down_entries'] * fraction,
                   waiting_cohort_seconds=duration, waiting_cohort_end_second=mature_end,
                   waiting_platform_delay_seconds=120, waiting_terminal_clock_mapping_exact=False,
                   waiting_weight_scope='unshifted entry cohort; partial quarter proportional; origin gate quarter approximates terminal clock')
        row['minimum_headway_seconds'] = max(minimum_headway(origin + timedelta(seconds=second), data['headway_policy'])
                                             for second in (row['start_second'], row['end_second'] - 1))
        row['maximum_headway_seconds'] = 480
        row['ready_capacity'] = 53  # Aggregate bound; planner enforces individual readiness/preparation.
        row['readiness_scope'] = 'all reserve trains assumed ready in shared scenario; exact depot lead-time validation follows'
    return rows


def make_candidates(data, second, service_end, bundle, strategy, state, *, budget_seconds=480, candidate_limit=3,
                    deadline=None, cancel_event=None):
    began = time.monotonic(); deadline = min(deadline if deadline is not None else float('inf'),
                                            began + min(150, budget_seconds * 0.4))
    if time.monotonic() >= deadline or cancel_event is not None and cancel_event.is_set():
        raise TimeoutError('candidate generation deadline')
    rows = [dict(candidate_id='KEEP', strategy='KEEP', status='movement_checked', _data=data,
                 timing=dict(search_complete=True), action=dict(kind='KEEP', decision_second=second))]
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    slots = _slots(data, bundle, at, state)
    dense = [dict(start_second=r['start_second'], end_second=r['end_second'], terminal='*',
                  headway_seconds=max(120, r['minimum_headway_seconds'])) for r in slots]
    profiles = [('DENSE', dense, 0, False)]
    search = None; surrogate = None
    if strategy == 'STANDARD':
        demand = []
        ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
        for row in slots:
            for terminal, direction in ((ids[0], 'up'), (ids[-1], 'down')):
                rate = row[f'{direction}_rate']
                h = min(480, max(row['minimum_headway_seconds'], int(1200 / rate / 10) * 10)) if rate else 480
                demand.append(dict(terminal=terminal, start_second=row['start_second'], end_second=row['end_second'],
                                   headway_seconds=h))
        profiles.extend([('DEMAND', demand, 0, False), ('DENSE_WEAR', dense, 0, True)])
    else:
        surrogate = estimated_keep_wait_budget(data, bundle, second)
        search = solve_joint_profile(slots, mean_wait_budget_seconds=surrogate['mean_wait_budget_seconds'],
            time_limit_seconds=min(3, max(0.1, deadline - time.monotonic())), deadline=deadline)
        if search.get('slots'):
            profile = directional_profile(data, search)
            profiles.insert(0, ('MPC_BUDGETED', profile, 0, False))
            profiles.append(('MPC_BUDGETED_WEAR', profile, 0, True))
    seen = {digest(data)}
    for name, profile, phase, wear in profiles[:candidate_limit]:
        if time.monotonic() >= deadline:
            rows.append(dict(candidate_id=name, strategy='REPLAN', status='evaluation_timeout', reason='generation deadline'))
            continue
        try:
            changed, audit = rebuild_timetable(data, second, service_end, headway=profile, phase_seconds=phase,
                                                wear_priority=wear, deadline=deadline, cancel_event=cancel_event)
            if not audit['movement_passed']:
                rows.append(dict(candidate_id=name, strategy='REPLAN', status='rejected', reason='movement validation failed', audit=audit))
                continue
            identity = digest(changed)
            audit['_data_digest'] = identity
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(dict(candidate_id=name, strategy='REPLAN', status='movement_checked', _data=changed,
                             timing=dict(search_complete=True), audit=audit,
                             action=dict(kind='REPLAN', decision_second=second, phase_seconds=phase,
                                         wear_priority=wear, profile=profile)))
        except (ValueError, TimeoutError) as exc:
            rows.append(dict(candidate_id=name, strategy='REPLAN', status='evaluation_timeout' if isinstance(exc, TimeoutError)
                             else 'rejected', reason=str(exc)))
    return rows, dict(generation_seconds=time.monotonic() - began, cp_search=search, cp_waiting_surrogate=surrogate,
                      bounded_catalog=True, candidate_limit=candidate_limit,
                      exact_validator_required=True)


def _calculate(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
               candidate_limit, deadline, cancelled):
    began = time.monotonic()

    def check():
        if cancelled.is_set() or time.monotonic() >= deadline:
            raise TimeoutError('GOAL 4.4 decision deadline')

    check()
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    bundle = read_forecast(source, forecast_source(strategy), ids, at)
    read_seconds = time.monotonic() - began
    candidates, search = make_candidates(data, second, planning_end, bundle, strategy, state,
        budget_seconds=max(0, deadline - time.monotonic()), candidate_limit=candidate_limit,
        deadline=deadline, cancel_event=cancelled)
    check(); validation_began = time.monotonic()
    decision = decide_operational(data, {'S1': runner}, {'S1': state}, profiles, {}, bundle, data['headway_policy'],
        candidates=candidates, budget_seconds=max(0, deadline - time.monotonic() - 10), include_recovery=False)
    check(); validation_seconds = time.monotonic() - validation_began
    candidate_by_id = {c['candidate_id']: c for c in candidates}
    accounting_began = time.monotonic()
    for row in decision['candidate_table']:
        if row.get('status') == 'evaluated' and row.get('passengers_checked'):
            check()
            candidate = candidate_by_id[row['candidate_id']]
            audit = candidate.get('audit', {})
            cached = None
            if row['candidate_id'] == 'KEEP':
                motion = getattr(runner, 'motion', {})
                if motion.get('obligations_closed') is True:
                    cached = motion.get('transport_work')
            elif audit.get('movement_passed') is True and audit.get('_data_digest') == digest(candidate['_data']):
                cached = audit.get('_transport_work')
            row['committed_per_train'] = (closed_per_train(candidate['_data'], roster, transport_work=cached)
                                           if cached is not None else closed_per_train(candidate['_data'], roster))
    check()
    mean, p95 = keep_wait_budgets(decision['candidate_table'])
    selection = choose_candidate(decision['candidate_table'], mean_wait_budget_s=mean, p95_wait_budget_s=p95,
                                  search_incomplete=not decision.get('search_complete', False))
    check()
    decision.update(selection=selection, source_mode=strategy, search44=search,
        wait_budget_reference='same forecast KEEP' if mean is not None and p95 is not None else 'establish feasible service reference',
        forecast_inference_in_timing=False, latency_breakdown=dict(cached_read_seconds=read_seconds,
            generation_seconds=search.get('generation_seconds'), validation_seconds=validation_seconds,
            accounting_and_selection_seconds=time.monotonic() - accounting_began))
    return dict(decision=decision, candidates=candidates)


def run_strategy(output, strategy, *, source=DEFAULT_SOURCE, pilot_steps=None, resume=False,
                 budget_seconds=470, candidate_limit=3):
    forecast_source(strategy)
    if not math.isfinite(budget_seconds) or not 1 <= budget_seconds <= 480:
        raise ValueError('decision budget must be from 1 to 480 seconds')
    if not isinstance(candidate_limit, int) or not 1 <= candidate_limit <= 3:
        raise ValueError('candidate limit must be from 1 to 3 new plans')
    output, source = Path(output), Path(source)
    folder = arm_folder(output, strategy, pilot_steps)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('experiment output must be separate from source results')
    shared, shared_contract = prepare_shared(source, output)
    source_hashes = {name: _hash(ROOT / name) for name in SOURCES}
    archive_sources(output, source_hashes)
    contract = dict(strategy=strategy, pilot_steps=pilot_steps, source_hashes=source_hashes,
                    shared_identity=digest(shared_contract), budget_seconds=budget_seconds,
                    candidate_limit=candidate_limit, control_step_seconds=900,
                    admission_end_second=shared['meta']['service_end_second'],
                    passenger_departure_end_second=departure_end(shared['meta']['service_end_second'], strategy),
                    clearance_seconds=0 if strategy == 'FIXED_SCHEDULE' else 600,
                    forecast_source_mode='BASELINE_FORECAST' if forecast_source(strategy) == 'MEDIAN'
                    else forecast_source(strategy), forecast_inference_in_timing=False,
                    latency_scope='cached forecast read, candidate generation, validation, resource accounting, selection')
    identity = digest(contract); contract_path = folder / 'contract.json'
    if contract_path.exists():
        if not resume:
            raise ValueError('arm already exists; use --resume or a separate output')
        if json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('resume source/input/mode contract changed')
    else:
        write_json_once(contract_path, contract)
    final_path = output / f'{strategy.lower()}_result.json' if pilot_steps is None else folder / 'pilot_result.json'
    if final_path.exists():
        if not resume:
            raise ValueError('refusing to overwrite completed arm')
        result = json.loads(final_path.read_text(encoding='utf-8'))
        if result.get('contract_identity') != identity:
            raise ValueError('completed result contract changed')
        return result
    data, state = copy.deepcopy(shared['data']), copy.deepcopy(shared['checkpoint'])
    roster, profiles, meta = shared['roster'], shared['profiles'], shared['meta']
    service_end = meta['service_end_second']; planning_end = departure_end(service_end, strategy)
    times = control_times(service_end)
    if pilot_steps is not None:
        times = times[:pilot_steps]
    prior = load_resume(folder, identity) if resume else None
    index, previous, decisions = 0, 3600, []
    if prior is not None:
        data, state, index, previous = prior['data'], prior['state'], prior['next_index'], prior['second']
        for path in sorted(folder.glob('resume_*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            if record['contract_identity'] != identity:
                raise ValueError('mixed resume contracts')
            for key in ('data', 'state'):
                if f'{key}_digest' in record and digest(record[key]) != record[f'{key}_digest']:
                    raise ValueError(f'resume {key} digest mismatch')
            journal = folder / f"decision_{record['second']:05}.json"
            if not journal.exists():
                write_json_once(journal, record['decision'])
            elif json.loads(journal.read_text(encoding='utf-8')) != record['decision']:
                raise ValueError('resume decision journal differs from committed checkpoint')
            plan = folder / f"plan_{record['second']:05}.json.gz"
            if not plan.exists():
                write_gzip_once(plan, record['data'])
            decisions.append(record['decision'])
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], shared['groups'], profiles)
    policy = data['headway_policy']; ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    began = time.monotonic()
    if strategy != 'FIXED_SCHEDULE':
        for position in range(index, len(times)):
            second = times[position]
            if second > previous:
                state = runner.run(start_second=previous, end_second=second, checkpoint=state)['checkpoint']
            tick = time.monotonic()
            at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
            receipt = bounded_decision(lambda deadline, cancelled: _calculate(data, runner, state, profiles, roster,
                source, strategy, ids, second, planning_end, candidate_limit, deadline, cancelled), budget_seconds)
            if receipt['completed']:
                decision, candidates = receipt['result']['decision'], receipt['result']['candidates']
            else:
                decision = dict(decision_time=at, decision_second=second, selection=receipt['selection'],
                    candidate_table=[], source_mode=strategy, search_complete=False, forecast_inference_in_timing=False)
                candidates = [dict(candidate_id='KEEP', _data=data)]
            selection = decision['selection']
            changed, applied = selected_plan(selection, candidates, decision['candidate_table'])
            decision.update(selection=selection, source_mode=strategy, applied_candidate_id=applied,
                mitigation_applied_as_research=selection.get('selected_candidate_id') is None and applied != 'KEEP',
                elapsed_seconds_including_plan_generation=receipt['elapsed_seconds'],
                notification_deadline_passed=receipt['completed'], late_work_discarded=not receipt['completed'],
                forecast_inference_in_timing=False)
            if changed is not data:
                new = GroupedPassengerSimulation(changed, changed['plans'][0]['plan_id'], shared['groups'], profiles)
                state = new.branch_checkpoint(state, runner); runner, data = new, changed
            saved = public_decision(decision); decisions.append(saved)
            write_gzip_once(folder / f'resume_{second:05}.json.gz', dict(contract_identity=identity,
                next_index=position + 1, second=second, state=state, data=data, decision=saved,
                state_digest=digest(state), data_digest=digest(data)))
            write_json_once(folder / f'decision_{second:05}.json', saved)
            write_gzip_once(folder / f'plan_{second:05}.json.gz', data)
            print(json.dumps(dict(stage='goal44_control', strategy=strategy, index=position + 1,
                decisions=len(times), at=at, applied=applied, status=selection['status'],
                seconds=round(time.monotonic() - tick, 2)), ensure_ascii=False), flush=True)
            previous = second
    if pilot_steps is not None:
        result = dict(strategy=strategy, pilot=True, full_day=False, contract_identity=identity,
                      decisions=decisions, checkpoint_second=state['second'], source_hashes=source_hashes,
                      groups_digest=shared['groups_digest'], checkpoint_digest=shared['checkpoint_digest'],
                      elapsed_seconds=time.monotonic() - began, release_authorized=False)
    else:
        state = runner.run(start_second=previous, checkpoint=state)['checkpoint']
        date = data['period_start'][:10]; run_id = f'{date}_goal44_{strategy}_S1_7'
        run = save_run(output, run_id, data, runner, state, roster, policy, 'goal44', strategy, 'S1', 7,
                       full_day=True, cohort_start=3600, cohort_end=service_end, service_end_second=service_end)
        metrics = _metrics(data, runner, state, roster, policy, 3600, service_end)
        metrics['peak_trains_on_line'] = runner.motion['checks']['peak_trains_on_line']
        metrics['fully_service_admissible'] = (metrics['actual_service_admissible']
            and metrics['all_service_arrivals_completed'] and metrics['movement_obligations_closed'])
        result = dict(strategy=strategy, date=date, metrics=metrics, decisions=decisions, run=run,
                      contract_identity=identity, source_hashes=source_hashes,
                      groups_digest=shared['groups_digest'], checkpoint_digest=shared['checkpoint_digest'],
                      research_oracle=strategy == 'MPC_ORACLE', forecast_inference_in_timing=False,
                      admission_end_second=service_end, passenger_departure_end_second=planning_end,
                      maximum_decision_seconds=max((d['elapsed_seconds_including_plan_generation'] for d in decisions), default=0),
                      elapsed_seconds=time.monotonic() - began, release_authorized=False)
    if any(_hash(ROOT / name) != value for name, value in source_hashes.items()):
        raise ValueError('experiment source changed during execution; intermediate journals retained')
    if any(_hash(source / name) != value for name, value in shared_contract['input_hashes'].items()):
        raise ValueError('experiment source input changed during execution')
    write_json_once(final_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--strategy', choices=ARMS, required=True)
    parser.add_argument('--pilot-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget-seconds', type=float, default=470)
    parser.add_argument('--candidate-limit', type=int, default=3)
    args = parser.parse_args()
    result = run_strategy(args.output, args.strategy, source=args.source, pilot_steps=args.pilot_steps,
                          resume=args.resume, budget_seconds=args.budget_seconds, candidate_limit=args.candidate_limit)
    print(json.dumps(dict(stage='goal44_complete', strategy=args.strategy, pilot=result.get('pilot', False),
                         decisions=len(result['decisions']), seconds=result['elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
