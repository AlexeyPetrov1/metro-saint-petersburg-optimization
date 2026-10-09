from copy import deepcopy

import pytest

from metro47_acceptance import assess_pair
from metro47_audit_run import build_case, terminal_gate
from test_metro47_acceptance import case


def saved():
    source = case()
    summary = dict(cohort_scope=source['cohort_scope'], service_scope=source['service_scope'],
        resources_scope={'start_second': 0, 'end_second': 32400}, full_day=False,
        metrics=dict(mean_wait_seconds=70, p95_wait_seconds=140))
    checkpoint = dict(second=32400, entered=2, completed=2, queued=0,
                      approaching=0, onboard_count=0)
    motion = dict(duration_seconds=32400, obligations_closed=True,
                  transport_work=[{k: v for k, v in r.items() if k != 'train_type'}
                                  for r in source['transport_work']])
    data = dict(period_start=source['period_start'],
        trains=[dict(train_id=r['train_id'], train_type=r['train_type'], state='standby', location='D')
                for r in source['transport_work']], tracks=[], stations=[], headway_policy={})
    groups = [dict(group_id='late', count=1, turnstile_second=4000),
              dict(group_id='early', count=1, turnstile_second=3600)]
    return (summary, checkpoint, motion, data), dict(groups=groups, profiles={'formation': []},
               physical=source['physical'], exposure=source['exposure'])


def test_case_uses_full_movement_ledger_including_empty_work():
    args, kwargs = saved(); built = build_case(*args, **kwargs)
    assert len(built['transport_work']) == 68
    assert built['transport_work'][0]['nonpassenger_train_km'] == '10'
    assert built['resource_scope']['includes_entry_exit']
    assert built['full_day'] is False and built['closure_second'] == 32400
    assert built['all_service_arrivals_completed']


def test_waiting_uses_the_full_declared_cohort_instead_of_mature_or_flat_metrics():
    args, kwargs = saved()
    args[0]['metrics'].update(cohort=dict(mean_wait_seconds=136, p95_wait_seconds=288,
        arrived=177248, complete=True), mature_cohort=dict(mean_wait_seconds=135,
        p95_wait_seconds=235))
    built = build_case(*args, **kwargs)
    assert built['mean_wait_seconds'] == 136 and built['p95_wait_seconds'] == 288
    assert built['waiting_scope'] == args[0]['cohort_scope']
    assert built['waiting_cohort_arrived'] == 177248


def test_normalized_group_order_does_not_change_cohort_identity():
    args, kwargs = saved(); first = build_case(*args, **kwargs)
    kwargs['groups'].reverse(); second = build_case(*args, **kwargs)
    assert first['cohort_digest'] == second['cohort_digest']
    assert first['group_order_digest'] != second['group_order_digest']


def test_same_causal_initial_context_is_independent_of_future_jobs():
    args, kwargs = saved(); first = build_case(*args, **kwargs)
    args[-1]['plans'] = [{'trips': ['new future schedule']}]
    assert build_case(*args, **kwargs)['common_start_digest'] == first['common_start_digest']


def test_service_override_is_explicit_and_preserves_the_old_summary():
    args, kwargs = saved(); old = deepcopy(args)
    built = build_case(*args, **kwargs, service_scope={'start_second': 3600, 'end_second': 22200})
    assert built['service_scope'] == dict(start_second=3600, end_second=22200)
    assert args == old


def test_missing_verified_physical_or_exposure_proof_cannot_be_assumed():
    args, kwargs = saved(); kwargs.pop('physical')
    with pytest.raises(ValueError): build_case(*args, **kwargs)


def test_claimed_partial_scope_is_not_promoted_to_the_full_horizon():
    args, kwargs = saved(); args[0]['resources_scope']['start_second'] = 3600
    built = build_case(*args, **kwargs)
    assert not built['resource_scope']['includes_entry_exit']
    assert not assess_pair(built, built)['accepted']


def test_checkpoint_must_be_at_the_saved_movement_closure():
    args, kwargs = saved(); args[1]['second'] = 30000
    with pytest.raises(ValueError, match='closure'): build_case(*args, **kwargs)


def test_terminal_audit_counts_entry_arrivals_and_same_time_departures():
    data = dict(headway_policy={'demo': {'terminal_parking_capacity_by_station': {'A': 1}}},
        trains=[dict(train_id='T1', state='turnaround', location='A'),
                dict(train_id='T2', state='turnaround', location='A'),
                dict(train_id='T3', state='moving', location='ENTRY')])
    motion = dict(events=[dict(second=10, train_id='T3', next_state='turnaround', location='A'),
                         dict(second=10, train_id='T1', next_state='moving', location='AB')])
    assert terminal_gate(data, motion)['passed']
    motion['events'][1]['second'] = 11
    assert not terminal_gate(data, motion)['passed']
