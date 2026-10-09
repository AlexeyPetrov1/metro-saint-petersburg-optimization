"""Independent arithmetic audit and adapter for saved Line 1 comparisons."""

import argparse
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

from metro47_acceptance import assess_pair


ROOT = Path(__file__).resolve().parent
RUN_FILES = ('movement_input.json', 'motion.json', 'passenger_checkpoint.json.gz',
             'groups.json.gz', 'profiles.json', 'summary.json')


def read(path):
    path = Path(path); raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def build_case(summary, checkpoint, motion, movement_input, *, groups, profiles,
               physical=None, exposure=None, service_scope=None):
    """Adapt audited journals without substituting a horizon or passenger cohort."""
    duration = motion['duration_seconds']
    if checkpoint['second'] != duration:
        raise ValueError('checkpoint is not at movement closure')
    if physical is None or exposure is None:
        raise ValueError('verified physical and exposure evidence is required')
    types = {t['train_id']: t['train_type'] for t in movement_input['trains']}
    work = [dict(r, train_type=types[r['train_id']]) for r in motion['transport_work']]
    scope = deepcopy(summary['resources_scope'])
    scope['includes_entry_exit'] = scope['start_second'] == 0 and scope['end_second'] == duration
    stocks = [checkpoint[k] for k in ('queued', 'approaching', 'onboard_count')]
    metrics = summary.get('metrics', {})
    waiting = metrics.get('cohort', metrics)
    return dict(simulated_line=1, period_start=movement_input['period_start'],
        cohort_digest=digest(sorted(groups, key=lambda r: r['group_id'])),
        group_order_digest=digest(groups), profiles_digest=digest(profiles),
        common_start_digest=digest({k: movement_input[k] for k in
            ('period_start', 'stations', 'tracks', 'trains', 'headway_policy')}),
        cohort_scope=deepcopy(summary['cohort_scope']),
        service_scope=deepcopy(service_scope if service_scope is not None else summary['service_scope']),
        resource_scope=scope, closure_second=duration,
        movement_obligations_closed=motion.get('obligations_closed') is True
            and not motion.get('unfinished_movements') and not motion.get('unfinished_obligations'),
        physical=deepcopy(physical), exposure=deepcopy(exposure), transport_work=work,
        entered=checkpoint['entered'], completed=checkpoint['completed'],
        queued=stocks[0], approaching=stocks[1], onboard=stocks[2],
        all_service_arrivals_completed=checkpoint['entered'] == checkpoint['completed'] and not any(stocks),
        mean_wait_seconds=waiting.get('mean_wait_seconds'),
        p95_wait_seconds=waiting.get('p95_wait_seconds'),
        waiting_scope=deepcopy(summary['cohort_scope'] if 'cohort' in metrics
                              else metrics.get('mean_wait_scope')),
        waiting_cohort_arrived=waiting.get('arrived'),
        full_day=summary.get('full_day') is True)


def terminal_gate(data, motion):
    caps = data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    state = {t['train_id']: (t['state'], t['location']) for t in data['trains']}
    total, parked, violations = dict.fromkeys(caps, 0), dict.fromkeys(caps, 0), []
    def inspect(second):
        for terminal, cap in caps.items():
            count = sum(s != 'moving' and loc == terminal for s, loc in state.values())
            idle = sum(s == 'standby' and loc == terminal for s, loc in state.values())
            total[terminal] = max(total[terminal], count)
            parked[terminal] = max(parked[terminal], idle)
            if count > 2 or idle > min(cap, 1):
                violations.append(dict(second=second, terminal=terminal, total=count, standby=idle))
    inspect(0)
    for second, events in itertools.groupby(motion['events'], lambda r: r['second']):
        for event in events: state[event['train_id']] = event['next_state'], event['location']
        inspect(second)
    return dict(passed=bool(caps) and not violations, maximum_total=total,
                maximum_standby=parked, violations=violations[:10], half_open=True)


def timing_pairs(data):
    tracks = {r['track_id']: r for r in data['tracks']}
    trains = {r['train_id']: r for r in data['trains']}
    trips = {r['trip_id']: r for r in data['plans'][0]['trips']}
    pairs = {}
    for row in data['leg_timing']:
        key = row['train_id'], row['trip_id'], row['leg_index']
        track = trains[key[0]]['location'] if key[1] == '@initial' else trips[key[1]]['route'][key[2]]
        if key in pairs: raise ValueError('duplicate physical timing identity')
        pairs[key] = (track, row['departure_second'], row['departure_second']
                      + tracks[track]['travel_seconds'] + row.get('approach_holding_seconds', 0))
    return pairs


def preparation_gate(reference, candidate, as_of, parameter):
    """Audit new ENTRY against the decision and physical depot readiness."""
    old, new = timing_pairs(reference), timing_pairs(candidate)
    tracks = {r['track_id']: r for r in candidate['tracks']}
    stations = {r['station_id']: r for r in candidate['stations']}
    trains = {r['train_id']: r for r in candidate['trains']}
    timing = {(r['train_id'], r['trip_id'], r['leg_index']): r for r in candidate['leg_timing']}
    by_train = defaultdict(list)
    for trip in candidate['plans'][0]['trips']: by_train[trip['train_id']].append(trip)
    release = parameter(candidate['headway_policy'], 'depot_release_s')
    errors = []
    for tid, owned in by_train.items():
        owned.sort(key=lambda t: new[tid, t['trip_id'], 0][1])
        for index, entry in enumerate(owned):
            key = tid, entry['trip_id'], 0
            departure = new[key][1]
            if not entry['route'][0].startswith('ENTRY_') or departure < as_of or old.get(key) == new[key]: continue
            depot = tracks[entry['route'][0]]['from_station']
            technical = trains[tid]['ready_after_seconds']
            if index:
                prior = owned[index-1]; last = timing[tid, prior['trip_id'], len(prior['route'])-1]
                if tracks[prior['route'][-1]]['to_station'] != depot:
                    errors.append(dict(train_id=tid, reason='no prior physical depot return')); continue
                technical = new[tid, prior['trip_id'], len(prior['route'])-1][2] + last['service_seconds']
            elif trains[tid]['location'] != depot:
                errors.append(dict(train_id=tid, reason='no initial depot readiness')); continue
            first = next((t for t in owned[index+1:] if t['passenger']), None)
            ready = max(as_of, technical)
            lead = sum(tracks[k]['travel_seconds'] for k in entry['route']) + stations[tracks[entry['route'][-1]]['to_station']]['turnaround_seconds']
            lead += sum(stations[tracks[k]['to_station']]['stop_seconds'] + stations[tracks[k]['to_station']]['turnaround_seconds']
                        for k in entry['route'][:-1])
            seconds = lead if stations[depot]['kind'] == 'siding' else release['Avtovo' if 'AVTOVO' in depot else 'Northern']
            if first is None or new[tid, first['trip_id'], 0][1] < ready + seconds or departure < ready + seconds - lead:
                errors.append(dict(train_id=tid, entry_second=departure, earliest_service_second=ready+seconds))
    return dict(passed=not errors, violations=errors, readiness_scope='source assumed crew and maintenance eligibility; actual dispatch authorization unknown')


def audit_pair(reference_run, candidate_run, output, snapshot, *, change_second, service_end):
    reference_run, candidate_run, output, snapshot = (Path(p).resolve() for p in
        (reference_run, candidate_run, output, snapshot))
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot))
    from goal43_verify import verify, _movement
    from goal43_policy import load_policy, parameter
    from rolling import operational_exposure

    manifest = read(snapshot/'execution_manifest.json')
    files = [snapshot/name for name in manifest['source_hashes']]
    files += [folder/name for folder in (reference_run, candidate_run) for name in RUN_FILES]
    files += [snapshot/'execution_manifest.json', Path(__file__), ROOT/'metro47_acceptance.py',
              ROOT/'tests/test_metro47_acceptance.py', ROOT/'tests/test_metro47_audit_run.py']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protected = {str(p): sha(p) for p in files}
    if {name: protected[str(snapshot/name)] for name in manifest['source_hashes']} != manifest['source_hashes']:
        raise ValueError('frozen verifier source bytes changed')
    write(output/'before.json', protected)
    write(output/'sourcebaseline_manifest.json', dict(workspace_root=str(ROOT), files=protected))
    loaded = {name: {f: read(folder/f) for f in RUN_FILES} for name, folder in
              (('REFERENCE', reference_run), ('CANDIDATE', candidate_run))}
    write(output/'runs.json', [dict(item['summary.json'], run_id=name,
          folder=str(reference_run if name == 'REFERENCE' else candidate_run)) for name, item in loaded.items()])
    verified = verify(output)
    policy = load_policy()
    common_service = dict(start_second=3600, end_second=service_end)
    cases, audits = {}, {}
    original = loaded['REFERENCE']['movement_input.json']
    original_motion = loaded['REFERENCE']['motion.json']
    for name, item in loaded.items():
        data, motion, state = (item[k] for k in ('movement_input.json', 'motion.json', 'passenger_checkpoint.json.gz'))
        reconstructed = _movement(data, motion, policy, common_service)
        terminal = terminal_gate(data, motion)
        same_geometry = all(data[k] == original[k] for k in ('stations', 'tracks', 'trains', 'headway_policy'))
        prefix = [e for e in motion['events'] if e['second'] < change_second] == [e for e in original_motion['events'] if e['second'] < change_second]
        old_pairs, new_pairs = timing_pairs(original), timing_pairs(data)
        committed = {k: r for k, r in old_pairs.items() if r[1] < change_second}
        current = all(new_pairs.get(k) == r for k, r in committed.items()) and not any(
            r[1] < change_second and k not in committed for k, r in new_pairs.items())
        prep = preparation_gate(original, data, change_second, parameter)
        gate = dict(movement_passed=not reconstructed['violations'] and same_geometry,
            headways_passed=reconstructed['service_headways']['passed'], terminal_passed=terminal['passed'],
            depot_preparation_passed=prep['passed'], executed_prefix_preserved=prefix,
            current_movements_preserved=current, peak_trains_on_line=reconstructed['peak'])
        exposure = operational_exposure(state, item['profiles.json'], data['headway_policy'], 0, motion['duration_seconds'])
        cases[name] = build_case(item['summary.json'], state, motion, data,
            groups=item['groups.json.gz'], profiles=item['profiles.json'], physical=gate,
            exposure=exposure, service_scope=common_service)
        audits[name] = dict(physical=gate, terminal=terminal, preparation=prep,
            service_intervals=reconstructed['service_headways']['checked_intervals'],
            minimum_headway=reconstructed['service_headways']['minimum_seconds'],
            maximum_headway=reconstructed['service_headways']['maximum_seconds'],
            physical_violations=reconstructed['violations'], committed_pairs=len(committed),
            entered=state['entered'], completed=state['completed'], full_day=cases[name]['full_day'])
    acceptance = assess_pair(cases['REFERENCE'], cases['CANDIDATE'])
    after = {str(p): sha(p) for p in files}
    if after != protected: raise ValueError('audit changed protected sources or journals')
    write(output/'after.json', after)
    write(output/'reference_case.json', cases['REFERENCE']); write(output/'candidate_case.json', cases['CANDIDATE'])
    report = dict(status='integrity_passed', accepted=acceptance['accepted'],
        full_day=all(c['full_day'] for c in cases.values()), snapshot=str(snapshot),
        arithmetic_verifier_status=verified['status'], audited_run_count=verified['run_count'],
        protected_file_count=len(protected), sources_and_inputs_preserved=True,
        common_service_scope_explicitly_rechecked=common_service, arms=audits,
        acceptance=acceptance, simulation_rerun=False, prospective_generalization_proven=False,
        release_authorized=False, cash_saving_rub=None)
    write(output/'acceptance.json', report)
    print(json.dumps(dict(output=str(output),accepted=report['accepted'],full_day=report['full_day'],
        saving_fraction=acceptance['saving_fraction_lower_bound'],reasons=acceptance['reasons']), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-run', 'candidate-run', 'output', 'snapshot'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--change-second', type=int, required=True)
    parser.add_argument('--service-end', type=int, required=True)
    args = parser.parse_args()
    audit_pair(args.reference_run, args.candidate_run, args.output, args.snapshot,
               change_second=args.change_second, service_end=args.service_end)
