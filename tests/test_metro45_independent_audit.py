from copy import deepcopy
import hashlib
import json

import pytest

from metro45_independent_audit import (
    compare_groups, compare_timing_obligations, exposure_summary,
    strict_pareto, verify_hashes, waiting_cohort, journey_context, reference_receipt,
    trial_source_ledger,
    source_manifest,
    guard_contract, selection_reasons,
)


def timetable():
    return dict(tracks=[dict(track_id='T', travel_seconds=20)],
        trains=[dict(train_id='B', location='T', state='moving')],
        plans=[dict(trips=[dict(train_id='B', trip_id='J', route=['T', 'T'])])],
        leg_timing=[dict(train_id='B', trip_id='J', leg_index=0,
                        departure_second=90, approach_holding_seconds=10),
                    dict(train_id='B', trip_id='J', leg_index=1,
                         departure_second=150, approach_holding_seconds=5)])


def test_groups_are_compared_by_identity_without_list_order():
    groups = [dict(group_id='a', count=2, destinations=[dict(station_id='x', count=2)]),
              dict(group_id='b', count=1, destinations=[])]
    assert compare_groups(groups, groups[::-1])['group_count'] == 2
    changed = deepcopy(groups); changed[0]['destinations'][0]['station_id'] = 'y'
    with pytest.raises(ValueError, match='group'):
        compare_groups(groups, changed)
    with pytest.raises(ValueError, match='duplicate'):
        compare_groups(groups, [groups[0], groups[0]])


def test_future_departures_and_arrivals_can_advance():
    old = timetable(); new = deepcopy(old)
    new['leg_timing'][1]['departure_second'] = 140
    new['leg_timing'][1]['approach_holding_seconds'] = 0
    receipt = compare_timing_obligations(old, new, 100)
    assert receipt['future_departure_advance_count'] == 1
    assert receipt['future_arrival_advance_count'] == 1
    assert receipt['current_pairs_checked'] == 1


@pytest.mark.parametrize('field', ['departure_second', 'approach_holding_seconds'])
def test_current_departure_and_arrival_are_immutable(field):
    old = timetable(); new = deepcopy(old)
    new['leg_timing'][0][field] += 1
    with pytest.raises(ValueError, match='committed'):
        compare_timing_obligations(old, new, 100)


def test_future_reassignment_does_not_remove_committed_keys():
    old = timetable(); new = deepcopy(old)
    new['leg_timing'][1]['train_id'] = 'Y'
    new['trains'].append(dict(train_id='Y', location='T', state='standby'))
    new['plans'][0]['trips'].append(dict(train_id='Y', trip_id='J', route=['T', 'T']))
    receipt = compare_timing_obligations(old, new, 100)
    assert receipt['future_removed_key_count'] == receipt['future_added_key_count'] == 1
    new['leg_timing'].pop(0)
    with pytest.raises(ValueError, match='committed'):
        compare_timing_obligations(old, new, 100)


def test_historical_failures_and_empty_tail_keep_their_scope():
    exposure = dict(admissible=False, global_=None,
        windows=[dict(start=0, end=900, status='fail', passenger_car_seconds=20),
                 dict(start=900, end=1800, status='pass', passenger_car_seconds=30),
                 dict(start=1800, end=2700, status='unknown', passenger_car_seconds=0)])
    exposure['global'] = dict(status='fail', max_load=160)
    report = exposure_summary(exposure)
    assert report['bad_window_count'] == 1
    assert report['empty_window_count'] == 1
    assert report['admissible'] is False
    assert exposure_summary(dict(admissible=True, windows=exposure['windows'][1:],
                                 **{'global': dict(status='pass', max_load=120)}))['bad_window_count'] == 0


def test_waiting_uses_mature_turnstile_cohort_and_refuses_partial_claim():
    groups = [dict(group_id='a', turnstile_second=10, platform_second=12, count=2),
              dict(group_id='late', turnstile_second=90, platform_second=92, count=7)]
    state = dict(second=200, board_events=[dict(group_id='a', second=20, count=2, wait_seconds=8)])
    result = waiting_cohort(groups, state, 0, 80)
    assert result['arrived'] == result['boarded'] == 2
    assert result['complete'] and result['mean_wait_seconds'] == result['p95_wait_seconds'] == 8
    result = waiting_cohort(groups, state, 0, 100)
    assert not result['complete'] and result['mean_wait_seconds'] is None


def test_p95_regression_rejects_strict_gain_despite_km_and_mean_gain():
    base = dict(mean_wait_seconds=71., p95_wait_seconds=138.)
    candidate = dict(mean_wait_seconds=70., p95_wait_seconds=143.,
        actual_service_admissible=True, all_service_arrivals_completed=True,
        movement_obligations_closed=True, terminal_resources_passed=True)
    bounds = dict(saving_train_km_lower_bound=10., saving_to1_resource_rub_lower_bound=74.)
    assert not strict_pareto(base, candidate, bounds, True)
    candidate['p95_wait_seconds'] = 138
    assert strict_pareto(base, candidate, bounds, True)
    candidate['mean_wait_seconds'] = float('nan')
    assert not strict_pareto(base, candidate, bounds, True)


def test_provenance_hash_checks_file_bytes(tmp_path):
    path = tmp_path / 'weights.txt'; path.write_bytes(b'original')
    ledger = {str(path): hashlib.sha256(b'original').hexdigest()}
    assert verify_hashes(ledger)['passed']
    path.write_bytes(b'changed')
    assert not verify_hashes(ledger)['passed']


def test_journey_context_uses_departure_order_on_repeated_route_tracks():
    data = timetable(); data['plans'][0]['trips'][0]['passenger'] = True
    motion = dict(events=[dict(event_id='first',event_type='departure',train_id='B',trip_id='J',location='T'),
                          dict(event_id='second',event_type='departure',train_id='B',trip_id='J',location='T')])
    before = deepcopy(motion)
    runner = journey_context(data,motion,[])
    assert runner.motion_by_id['first']['remaining_route'] == ['T','T']
    assert runner.motion_by_id['second']['remaining_route'] == ['T']
    assert motion == before


def test_reference_reuse_requires_prior_proof_and_exact_current_bytes(tmp_path):
    source = tmp_path/'source.json'; source.write_bytes(b'original')
    proof = tmp_path/'proof'; proof.mkdir()
    ledger = {str(source):hashlib.sha256(b'original').hexdigest()}
    (proof/'before.json').write_text(json.dumps(ledger))
    (proof/'verification_report.json').write_text(json.dumps(dict(status='passed',run_count=3)))
    assert reference_receipt(proof,ledger)['passed']
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='reference'):
        reference_receipt(proof,ledger)


def test_extension_sources_are_part_of_trial_provenance(tmp_path):
    base = {str(tmp_path/'base.py'):'aaa'}
    extension = {str(tmp_path/'joint.py'):'bbb'}
    for name in ('before.json','after.json'):
        (tmp_path/name).write_text(json.dumps(base))
    path = tmp_path/'joint_extension.json'
    path.write_text(json.dumps(dict(sources_before=extension,sources_after=extension,sources_preserved=True)))
    assert trial_source_ledger(tmp_path)['sources'] == {**base,**extension}
    path.write_text(json.dumps(dict(sources_before=extension,sources_after={},sources_preserved=False)))
    with pytest.raises(ValueError,match='extension'):
        trial_source_ledger(tmp_path)


def test_explicit_manifest_checks_snapshot_bytes_without_unrelated_fallback(tmp_path,monkeypatch):
    import goal43_verify
    out=tmp_path/'audit';out.mkdir()
    source=tmp_path/'source.py';source.write_bytes(b'fixed snapshot')
    ledger={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    fallback=tmp_path/'goal4_3_verification';fallback.mkdir()
    (fallback/'baseline_manifest.json').write_text(json.dumps(dict(files={'missing_old_input.json':'bad'})))
    monkeypatch.setattr(goal43_verify,'ROOT',tmp_path/'different_snapshot_root')
    (out/'sourcebaseline_manifest.json').write_text(json.dumps(source_manifest(ledger)))
    assert goal43_verify._manifest(out)['checked_files']==1
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        goal43_verify._manifest(out)


def guard_fixture():
    point=dict(rows=[dict(station_id='S',interval_start='2026-09-30T10:00:00+03:00',
        interval_end='2026-09-30T10:15:00+03:00',predicted_count=10)])
    guard=deepcopy(point)
    guard['source_versions']=dict(residual_nowcast=dict(observed_latest_available_at='2026-09-30T10:00:00+03:00',
        parameters=dict(max_relative_change=.15),used_predictions=[dict(interval_start='2026-09-30T09:45:00+03:00',
        forecast_issued_at='2026-09-30T09:00:00+03:00',forecast_available_at='2026-09-30T09:00:00+03:00',
        features_available_at='2026-09-30T09:00:00+03:00')]))
    return point,guard


def test_guard_only_uses_forecasts_published_before_observed_quarter():
    point,guard=guard_fixture()
    assert guard_contract(point,guard,'2026-09-30T10:00:00+03:00')['passed']
    guard['source_versions']['residual_nowcast']['used_predictions'][0]['forecast_issued_at']='2026-09-30T09:50:00+03:00'
    with pytest.raises(ValueError,match='prior forecast'):
        guard_contract(point,guard,'2026-09-30T10:00:00+03:00')


def test_risk_guard_preserves_point_floor_and_fixed_relative_cap():
    point,guard=guard_fixture();guard['rows'][0]['predicted_count']=9
    with pytest.raises(ValueError,match='point floor'):
        guard_contract(point,guard,'2026-09-30T10:00:00+03:00')
    guard['rows'][0]['predicted_count']=12
    with pytest.raises(ValueError,match='relative cap'):
        guard_contract(point,guard,'2026-09-30T10:00:00+03:00')


def test_selection_explains_p95_regression_despite_lower_mean():
    cohort=dict(complete=True,mean_wait_seconds=70.,p95_wait_seconds=143.)
    exposure=dict(admissible=True,global_={});exposure['global']=dict(max_load=87,exceptional_time_share=0.,hard_limit_car_seconds=0.)
    trial=dict(future_cohort=cohort,future_policy_summary=exposure,headways_passed=True,initial_obligations_complete=True,initial_unboarded=0)
    row=dict(candidate_id='LOWER_KM',status='evaluated',physical_passed=True,passengers_checked=True,service_passed=True,trials=[trial])
    reasons=selection_reasons(row,71.,138.)
    assert reasons == ['p95_wait_budget']


def test_completed_empty_future_cohort_has_zero_wait_for_selection():
    trial=dict(future_cohort=dict(complete=True,arrived=0,mean_wait_seconds=None,p95_wait_seconds=None),
        future_policy_summary=dict(admissible=True,**{'global':dict(passenger_car_seconds=0)}),
        headways_passed=True,initial_obligations_complete=True,initial_unboarded=0)
    row=dict(status='evaluated',physical_passed=True,passengers_checked=True,service_passed=True,trials=[trial])
    assert selection_reasons(row,0.,0.)==[]


def numeric_guard_fixture():
    point,guard=guard_fixture()
    meta=guard['source_versions']['residual_nowcast']
    meta['parameters']=dict(recent_minutes=30,gain=.25,max_relative_change=.15,decay_minutes=60)
    meta['used_predictions'][0].update(station_id='S',predicted_count=10)
    meta['station_adjustments']={'S':dict(status='adjusted',recent_intervals=1,supported_intervals=1,
        observed_recent_count=20,predicted_recent_count=10,multiplier=1.15)}
    guard['rows'][0]['predicted_count']=11
    observations=[dict(station_id='S',interval_start='2026-09-30T09:45:00+03:00',
        interval_end='2026-09-30T10:00:00+03:00',entry_count=20)]
    return point,guard,observations


def test_numeric_guard_reconstructs_clipped_ratio_and_integer_rounding():
    from metro45_independent_audit import residual_numeric_check
    point,guard,observations=numeric_guard_fixture()
    assert residual_numeric_check(point,guard,observations,'2026-09-30T10:00:00+03:00')['passed']
    guard['rows'][0]['predicted_count']=10
    with pytest.raises(ValueError,match='guard numeric count'):
        residual_numeric_check(point,guard,observations,'2026-09-30T10:00:00+03:00')


def test_numeric_guard_rejects_wrong_past_count_and_future_observation():
    from metro45_independent_audit import residual_numeric_check
    point,guard,observations=numeric_guard_fixture()
    observations[0]['entry_count']=19
    with pytest.raises(ValueError,match='observed_recent_count'):
        residual_numeric_check(point,guard,observations,'2026-09-30T10:00:00+03:00')
    observations[0]['interval_end']='2026-09-30T10:15:00+03:00'
    with pytest.raises(ValueError,match='future observation'):
        residual_numeric_check(point,guard,observations,'2026-09-30T10:00:00+03:00')
