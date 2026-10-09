"""Independent closed-episode ledger audit plus every committed control transition."""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys

from goal43_policy import parameter
from metro47_audit_run import RUN_FILES, digest, preparation_gate, read, timing_pairs, write
from simulation import simulate


ROOT = Path(__file__).resolve().parent


def control_receipts(decisions, expected_times):
    clocks = [r.get('decision_second') for r in decisions]
    durations = [r.get('pipeline_seconds') for r in decisions]
    timed = all(type(s) in (int, float) and isfinite(s) and 0 <= s <= 480 for s in durations)
    marked = all(r.get('local_recommendation_within_480') is True for r in decisions)
    return dict(passed=bool(decisions) and clocks == list(expected_times) and timed and marked,
        decision_count=len(decisions), expected_decision_count=len(expected_times),
        decision_times=clocks, complete_quarter_sequence=clocks == list(expected_times),
        pipeline_within_480=timed and marked,
        maximum_pipeline_seconds=max(durations) if timed and durations else None,
        external_delivery_measured=bool(decisions) and all(r.get('external_delivery_measured') is True for r in decisions))


def transition_gate(before, after, second, *, before_motion=None, after_motion=None):
    old_motion = before_motion if before_motion is not None else simulate(before, before['plans'][0]['plan_id'])
    new_motion = after_motion if after_motion is not None else simulate(after, after['plans'][0]['plan_id'])
    old, new = timing_pairs(before), timing_pairs(after)
    prefix = [e for e in old_motion['events'] if e['second'] < second]
    same_prefix = prefix == [e for e in new_motion['events'] if e['second'] < second]
    committed = {k: r for k, r in old.items() if r[1] < second}
    current = all(new.get(k) == r for k, r in committed.items()) and not any(
        r[1] < second and k not in committed for k, r in new.items())
    begun = {t['trip_id']: t for t in before['plans'][0]['trips'] if t['passenger']
             and old[t['train_id'], t['trip_id'], 0][1] < second}
    trips = {t['trip_id']: t for t in after['plans'][0]['trips']}
    whole = all(trips.get(tid) == t for tid, t in begun.items()) and all(
        new.get(k) == value for k, value in old.items() if k[1] in begun)
    committed_prep = {}
    tracks = {t['track_id']: t for t in before['tracks']}
    releases = parameter(before['headway_policy'], 'depot_release_s')
    for train in before['trains']:
        owned = sorted((t for t in before['plans'][0]['trips'] if t['train_id'] == train['train_id']),
                       key=lambda t: old[t['train_id'], t['trip_id'], 0][1])
        for index, entry in enumerate(owned):
            if not entry['route'][0].startswith('ENTRY_'):
                continue
            service = next((t for t in owned[index+1:] if t['passenger']), None)
            depot = tracks[entry['route'][0]]['from_station']
            release = 0 if depot.startswith('SIDING_') else releases['Avtovo' if 'AVTOVO' in depot else 'Northern']
            if service and (old[train['train_id'], entry['trip_id'], 0][1] < second
                    or old[train['train_id'], service['trip_id'], 0][1] - release < second):
                committed_prep.update({t['trip_id']: t for t in (entry, service)})
    prep_kept = all(trips.get(tid) == t for tid, t in committed_prep.items()) and all(
        new.get(k) == value for k, value in old.items() if k[1] in committed_prep)
    geometry = all(before.get(k) == after.get(k) for k in ('stations', 'tracks', 'trains', 'headway_policy', 'withdrawal_stabling'))
    preparation = preparation_gate(before, after, second, parameter)
    passed = same_prefix and current and whole and prep_kept and geometry and preparation['passed']
    return dict(passed=passed, decision_second=second, executed_prefix_preserved=same_prefix,
        current_movements_preserved=current, begun_service_clocks_preserved=whole,
        preparation_commitments_preserved=prep_kept, committed_preparation_trip_count=len(committed_prep),
        same_geometry=geometry, depot_preparation=preparation,
        executed_prefix_digest=digest(prefix), committed_leg_count=len(committed), begun_service_count=len(begun))


def audit48(reference, candidate, output, snapshot):
    reference, candidate, output, snapshot = (Path(p).resolve() for p in (reference, candidate, output, snapshot))
    if output.exists():
        raise FileExistsError(output)
    if not all((folder/'result.json').is_file() for folder in (reference, candidate)):
        raise ValueError('both completed experiment result files are required')
    output.mkdir(parents=True)
    manifest = read(snapshot/'execution_manifest.json')
    protected = [snapshot/name for name in manifest['source_hashes']]
    protected += [snapshot/'execution_manifest.json', Path(__file__), ROOT/'tests/test_metro48_audit.py']
    runtime = [ROOT/name for name in ('simulation.py', 'dwell.py', 'goal43_policy.py',
        'rolling.py', 'metro47_audit_run.py', 'metro47_acceptance.py', 'metro48_temporary.py',
        'metro48_temporary_solver.py', 'metro48_experiment.py', 'withdrawal_stabling.py')]
    protected += runtime
    loaded = {}
    for name, folder in (('REFERENCE', reference), ('CANDIDATE', candidate)):
        contract, result = read(folder/'contract.json'), read(folder/'result.json')
        runs = list((folder/'runs').iterdir())
        if len(runs) != 1 or not runs[0].is_dir():
            raise ValueError('one complete saved run is required per arm')
        loaded[name] = dict(folder=folder, contract=contract, result=result, run=runs[0])
        protected += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json', '.gz')]
        protected += [Path(p) for p in contract['input_hashes']]
    protected = sorted(set(p.resolve() for p in protected))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    before = {str(p): sha(p) for p in protected}
    write(output/'before.json', before)
    source_bytes = {name: before[str((snapshot/name).resolve())] for name in manifest['source_hashes']}
    if source_bytes != manifest['source_hashes']:
        raise ValueError('frozen independent verifier source bytes changed')
    input_integrity = all(all(before[str(Path(p).resolve())] == value for p, value in item['contract']['input_hashes'].items())
                          for item in loaded.values())
    if not input_integrity:
        raise ValueError('experiment-declared input bytes changed')
    subprocess.run([sys.executable, str(ROOT/'metro47_audit_run.py'),
        '--reference-run', str(loaded['REFERENCE']['run']), '--candidate-run', str(loaded['CANDIDATE']['run']),
        '--output', str(output/'ledger'), '--snapshot', str(snapshot), '--change-second', '3600', '--service-end', '22200'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    ledger = read(output/'ledger/acceptance.json')
    common = read(candidate/'common_scenario.json.gz')
    old_data = common['data']
    old_motion = simulate(old_data, old_data['plans'][0]['plan_id'])
    initial_context = digest({k: old_data[k] for k in ('stations', 'tracks', 'trains', 'headway_policy')})
    quarters = []
    expected = list(range(3600, 21600, 900))
    decisions = loaded['CANDIDATE']['result']['decisions']
    receipts = control_receipts(decisions, expected)
    for decision in decisions:
        second = decision['decision_second']
        resume = read(candidate/'controller'/f'resume_{second:05}.json.gz')
        new_data = resume['data']; new_motion = simulate(new_data, new_data['plans'][0]['plan_id'])
        gate = transition_gate(old_data, new_data, second, before_motion=old_motion, after_motion=new_motion)
        gate['checkpoint_clock_passed'] = resume['second'] == second == resume['state']['second']
        gate['saved_prefix_digest_passed'] = gate['executed_prefix_digest'] == decision['executed_prefix_digest']
        gate['passed'] &= gate['checkpoint_clock_passed'] and gate['saved_prefix_digest_passed']
        quarters.append(gate)
        old_data, old_motion = new_data, new_motion
    write(output/'control_transitions.json', quarters)
    write(output/'control_receipts.json', receipts)
    journal_parity = {}
    for name, item in loaded.items():
        data, saved = (read(item['run']/file) for file in ('movement_input.json', 'motion.json'))
        replayed = simulate(data, data['plans'][0]['plan_id'])
        journal_parity[name] = dict(passed=digest(saved) == digest(replayed),
            saved_digest=digest(saved), replayed_digest=digest(replayed),
            runtime_source=str(ROOT/'simulation.py'), runtime_sha256=before[str(ROOT/'simulation.py')])
    final = read(loaded['CANDIDATE']['run']/'movement_input.json')
    final_plan_preserved = digest(final) == digest(old_data)
    write(output/'journal_parity.json', journal_parity)
    cases = {name: read(output/f'ledger/{name.lower()}_case.json') for name in loaded}
    total = sum(g['count'] for g in common['groups'])
    controlled = sum(g['count'] for g in common['groups'] if 3600 <= g['turnstile_second'] < 21600)
    cohort = dict(expected_total=total, expected_controlled=controlled,
        total_is_247117=total == 247117, controlled_is_177248=controlled == 177248,
        passed=total == 247117 and controlled == 177248 and all(
            c['entered'] == total == c['completed'] and c['waiting_cohort_arrived'] == controlled for c in cases.values()))
    extra = dict(control_receipts=receipts['passed'], all_quarter_transitions=len(quarters) == 20 and all(r['passed'] for r in quarters),
        reconstructed_journals=all(r['passed'] for r in journal_parity.values()),
        final_controlled_plan=final_plan_preserved, complete_original_cohort=cohort['passed'],
        experiment_sources_preserved=all(r['result'].get('sources_preserved') is True for r in loaded.values()),
        shared_initial_context=all(digest({k: read(r['run']/'movement_input.json')[k]
            for k in ('stations', 'tracks', 'trains', 'headway_policy')}) == initial_context for r in loaded.values()))
    reasons = ledger['acceptance']['reasons'] + ['quarter48_'+name for name, passed in extra.items() if not passed]
    after = {str(p): sha(p) for p in protected}
    if after != before:
        raise ValueError('audit changed protected sources or saved experiment journals')
    write(output/'after.json', after)
    report = dict(status='integrity_passed', accepted=ledger['accepted'] and all(extra.values()),
        full_day=False, reasons=sorted(set(reasons)), acceptance=ledger['acceptance'],
        enhanced_checks=extra, cohort=cohort, arms=ledger['arms'],
        independent_ledger_snapshot=str(snapshot), frozen_verifier_file_count=len(manifest['source_hashes']),
        protected_file_count=len(protected), sources_and_inputs_preserved=True,
        simulation_rerun=True, controller_decision_count=len(quarters),
        exact_current_kernel_journal_parity=all(r['passed'] for r in journal_parity.values()),
        all_20_quarter_prefixes_current_movements_and_begun_services_preserved=extra['all_quarter_transitions'],
        local_recommendation_within_480=receipts['passed'], external_delivery_measured=False,
        prospective_generalization_proven=False, release_authorized=False, cash_saving_rub=None,
        lost_revenue_rub=None, net_profit_effect_rub=None)
    write(output/'acceptance.json', report)
    print(json.dumps(dict(output=str(output), accepted=report['accepted'],
        saving_fraction=ledger['acceptance']['saving_fraction_lower_bound'], reasons=report['reasons'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('reference', 'candidate', 'output', 'snapshot'):
        parser.add_argument('--'+argument, required=True)
    audit48(**vars(parser.parse_args()))
