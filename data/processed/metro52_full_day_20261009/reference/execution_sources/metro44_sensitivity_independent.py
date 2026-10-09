"""Verify paired saved arrival diagnostics without replaying passengers."""

import hashlib
import json
import sys
from pathlib import Path

from metro44_bridge_independent import prefix_equal, read, sha


def check_jitter(original, changed, seed=None):
    old = {r['group_id']: r for r in original}
    new = {r['group_id']: r for r in changed}
    assert len(old) == len(original) == len(new) == len(changed)
    assert old.keys() == new.keys()
    times = {'turnstile_second', 'platform_second'}
    for key, first in old.items():
        second = new[key]
        assert {k:v for k,v in first.items() if k not in times} == {k:v for k,v in second.items() if k not in times}
        assert first['turnstile_second']//60 == second['turnstile_second']//60
        assert first['platform_second']-first['turnstile_second'] == second['platform_second']-second['turnstile_second']
        if seed is not None:
            offset = int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8], 'big') % 60
            assert second['turnstile_second'] % 60 == offset
    return True


def original_order(original, saved):
    by_id = {r['group_id']: r for r in saved}
    return [by_id[r['group_id']] for r in original]


def main():
    root = Path(__file__).resolve().parent
    snapshot = root/'data/processed/metro44_bridge_execution_v5'
    source = root/'data/processed/metro44_arrival_sensitivity_20261009'
    output = root/'data/processed/metro44_arrival_sensitivity_independent_audit_20261009_v2'
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(snapshot))
    from goal43_verify import verify
    from metro44_planner_noidle import terminal_total_audit
    from rolling import operational_exposure
    from simulation import digest
    def write(name, value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    frozen = read(snapshot/'execution_manifest.json')['source_hashes']
    assert {name:sha(snapshot/name) for name in frozen} == frozen
    protected = {str(p):sha(p) for p in source.rglob('*') if p.is_file()}
    before = read(source/'before.json')
    assert before == read(source/'after.json')
    assert {name:sha(Path(name)) for name in before} == before
    cases = [read(source/f'{od.lower()}_seed{seed}.json') for od in ('S1','S2','S3') for seed in (7,11)]
    records = [dict(arm['run'],folder=str(source/arm['run']['folder'])) for c in cases for arm in c['arms'].values()]
    write('runs.json',records)
    write('sourcebaseline_manifest.json',dict(workspace_root=str(snapshot),files=frozen))
    verified = verify(output)
    assert verified['status']=='passed' and verified['run_count']==12
    closed = root/'data/processed/metro44_continuation_6h_closed_20261009'
    robust = root/'data/processed/metro44_profile_6h_robustness_20261009'
    bases = {'S1':(closed,'keep_result.json'),'S2':(robust,'keep_s2_result.json'),'S3':(robust,'keep_s3_result.json')}
    rows=[]
    for case in cases:
        od,seed=case['od'],case['seed']
        parent,name=bases[od];base_folder=parent/read(parent/name)['run']['folder']
        original=read(base_folder/'groups.json.gz')
        folders={arm:source/result['run']['folder'] for arm,result in case['arms'].items()}
        groups={arm:read(folder/'groups.json.gz') for arm,folder in folders.items()}
        assert groups['KEEP']==groups['BRIDGE160']
        assert check_jitter(original,groups['KEEP'],seed)
        assert digest(original_order(original,groups['KEEP']))==case['groups_digest']
        checkpoints={arm:read(folder/'passenger_checkpoint.json.gz') for arm,folder in folders.items()}
        prefixes={k:prefix_equal(checkpoints['KEEP'][k],checkpoints['BRIDGE160'][k],6300)
                  for k in ('load_events','queue_events','board_events')}
        assert all(prefixes.values())
        arms={}
        for arm,result in case['arms'].items():
            folder=folders[arm];data=read(folder/'movement_input.json');motion=read(folder/'motion.json')
            profiles=read(folder/'profiles.json');state=checkpoints[arm]
            assert profiles==read(base_folder/'profiles.json')
            assert state['entered']==state['completed']==247117 and not(state['queued'] or state['approaching'] or state['onboard_count'])
            exposure=operational_exposure(state,profiles,data['headway_policy'],3600,21600)
            assert exposure==result['metrics']['policy_exposure'] and exposure['admissible']
            terminal=terminal_total_audit(data,motion)
            assert terminal['passed'] and motion['obligations_closed']
            journey=result['journey']
            assert journey['arrived']==journey['boarded']==journey['completed']==171708 and journey['complete']
            assert abs(journey['mean_gate_to_exit_seconds']-journey['mean_platform_to_exit_seconds']-120)<1e-9
            assert abs(journey['mean_platform_to_exit_seconds']-journey['mean_ride_seconds']-result['metrics']['mean_wait_seconds'])<1e-9
            arms[arm]=dict(controlled_admissible=True,terminal_total_passed=True,all_complete=True,
                mean_wait=result['metrics']['mean_wait_seconds'],p95_wait=result['metrics']['p95_wait_seconds'],
                mean_gate_to_exit=journey['mean_gate_to_exit_seconds'],p95_gate_to_exit=journey['p95_gate_to_exit_seconds'])
        delta=arms['BRIDGE160']['mean_wait']-arms['KEEP']['mean_wait']
        assert delta==case['mean_wait_delta_seconds']
        rows.append(dict(od=od,seed=seed,exact_paired_groups=True,minute_and_quarter_counts_destinations_preserved=True,
            unchanged_access_delay=True,deterministic_seed_offsets_verified=True,immutable_prefix_checks=prefixes,
            arms=arms,mean_wait_delta_seconds=delta,
            mean_journey_delta_seconds=arms['BRIDGE160']['mean_gate_to_exit']-arms['KEEP']['mean_gate_to_exit'],
            p95_journey_delta_seconds=arms['BRIDGE160']['p95_gate_to_exit']-arms['KEEP']['p95_gate_to_exit']))
    assert {name:sha(snapshot/name) for name in frozen}==frozen
    assert {str(p):sha(p) for p in source.rglob('*') if p.is_file()}==protected
    write('independent_audit.json',dict(status='passed',run_count=12,source_files_checked=len(frozen),
        protected_files_checked=len(protected),sources_inputs_results_preserved=True,rows=rows,
        scope='separate synthetic within-minute sensitivity; unchanged observed minute totals and assumed OD; constant-53 reference',
        journey_verification='saved summary identities checked; destination FIFO reconstruction covered by original helper tests, not independently reimplemented',
        cash_saving_rub=None,release_authorized=False))
    print(json.dumps(dict(status='passed',run_count=12,output=str(output),rows=rows)))


if __name__=='__main__':
    main()
