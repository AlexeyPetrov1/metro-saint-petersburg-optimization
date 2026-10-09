import copy
import time

import pytest

from metro45_joint_dispatch import solve_joint_dispatch, build_joint_problem, apply_joint_dispatch, repair_joint_holds, propose_joint_dispatch, departure_scope


def job(identity='J', origin='A', destination='B', first=0, last=0, duration=280, km=10):
    return dict(job_id=identity, origin=origin, destination=destination,
                earliest_second=first, latest_second=last, run_seconds=duration,
                km=km, minimum_headway_seconds=150)


def train(identity='B', kind='baltiets', origin='A', ready=0, **extra):
    return dict(train_id=identity, train_type=kind, location=origin,
                ready_second=ready, on_line=True, **extra)


def test_time_and_train_are_chosen_jointly_without_dropping_jobs():
    fleet = [train(), train('Y', 'yubileyny', ready=5)]
    fixed = solve_joint_dispatch([job()], fleet)
    flexible = solve_joint_dispatch([job(last=5)], fleet)
    assert fixed['assignments'] == {'J': 'B'}
    assert flexible['assignments'] == {'J': 'Y'}
    assert flexible['departures'] == {'J': 5}
    assert flexible['to1_resource_equivalent_rub'] == 74
    assert flexible['mandatory_jobs_covered'] == 1


def test_mandatory_unreachable_slot_is_explicitly_infeasible():
    answer = solve_joint_dispatch([job(origin='C')], [train()])
    assert answer['status'] == 'infeasible'
    assert answer['assignments'] == {}
    assert answer['mandatory_jobs_covered'] == 0


@pytest.mark.parametrize('release', [720, 780])
def test_full_depot_preparation_precedes_service(release):
    reserve = dict(
        train_id='R', train_type='yubileyny', location='A', on_line=False,
        depot_ready_second=0, release_seconds=release, entry_lead_seconds=300,
        entry_km=1, exit_km=1)
    early = solve_joint_dispatch([job(first=release-1, last=release-1)], [reserve])
    valid = solve_joint_dispatch([job(first=release, last=release)], [reserve])
    assert early['status'] == 'infeasible'
    assert valid['reserve_entries'] == {'R': release-300}


def test_unknown_depot_readiness_is_rejected():
    reserve = dict(train_id='R', train_type='yubileyny', location='A',
                   on_line=False, release_seconds=720)
    with pytest.raises(ValueError, match='readiness'):
        solve_joint_dispatch([job(first=720, last=720)], [reserve])


def test_whole_line_capacity_includes_reserve_entry():
    reserve = dict(train_id='R', train_type='yubileyny', location='A',
                   on_line=False, depot_ready_second=0, release_seconds=720,
                   entry_lead_seconds=300)
    answer = solve_joint_dispatch([job(first=720, last=720)], [reserve],
                                  fixed_line_intervals=[(0, 2000, 53)])
    assert answer['status'] == 'infeasible'


def test_destination_turnaround_uses_two_shared_terminal_places():
    jobs = [job('J0', 'B', 'A', 0, 0, 500),
            job('J1', 'B', 'A', 150, 150, 380),
            job('J2', 'B', 'A', 300, 300, 260)]
    fleet = [train('T0', origin='B'), train('T1', origin='B', ready=150),
             train('T2', origin='B', ready=300)]
    answer = solve_joint_dispatch(jobs, fleet)
    assert answer['status'] == 'infeasible'


def test_two_ready_trains_cannot_wait_in_one_standby_place():
    jobs = [job('J0', first=200, last=200), job('J1', first=400, last=400)]
    answer = solve_joint_dispatch(jobs, [train('T0'), train('T1')])
    assert answer['status'] == 'infeasible'


def test_same_vehicle_chain_obeys_terminal_turn_and_route():
    first = job('J0')
    early = job('J1', 'B', 'A', 279, 279)
    valid = job('J1', 'B', 'A', 280, 280)
    assert solve_joint_dispatch([first, early], [train()])['status'] == 'infeasible'
    answer = solve_joint_dispatch([first, valid], [train()])
    assert answer['assignments'] == {'J0': 'B', 'J1': 'B'}


def test_unstarted_route_padding_is_optional_not_an_inherited_obligation():
    first = dict(job('J0'), max_run_seconds=400)
    second = job('J1', 'B', 'A', 350, 350)
    answer = solve_joint_dispatch([first, second], [train()])
    assert answer['mandatory_jobs_covered'] == 2
    assert answer['ready_seconds']['J0'] == 280


def test_fixed_neighbor_clocks_and_actual_headway_are_respected():
    answer = solve_joint_dispatch([job(first=160, last=180)], [train(ready=160)],
                                  fixed_departures=[('A', 0, 150), ('A', 320, 150)])
    assert 160 <= answer['departures']['J'] <= 170


def test_solver_does_not_modify_input_and_expired_budget_stops():
    jobs, fleet = [job(last=10)], [train()]
    before = copy.deepcopy((jobs, fleet))
    solve_joint_dispatch(jobs, fleet)
    assert (jobs, fleet) == before
    with pytest.raises(TimeoutError):
        solve_joint_dispatch(jobs, fleet, deadline=time.monotonic()-1)


@pytest.fixture(scope='module')
def movement_case():
    import json
    from pathlib import Path
    path = Path('data/processed/metro44_bridge_off160_actual_6h_20261009_v2/runs/OFF160_BRIDGE_S1_7')
    return json.loads((path/'movement_input.json').read_text()), json.loads((path/'motion.json').read_text())


def test_real_problem_frees_unstarted_clocks_without_rewriting_source(movement_case):
    data, motion = movement_case
    before = copy.deepcopy(data)
    problem = build_joint_problem(data, 6300, ['B010', 'B011', 'Y010', 'Y011'],
                                  window_seconds=30, source_motion=motion)
    assert data == before
    assert len(problem['availability']) == 4
    assert all(j['earliest_second'] < j['original_departure_second'] for j in problem['jobs'])
    assert all(j['earliest_second'] >= 6300 for j in problem['jobs'])
    assert len(problem['jobs']) == len(problem['job_ids'])
    assert problem['fixed_terminal_intervals']
    assert problem['fixed_line_intervals']
    assert any(j['run_seconds'] < j['source_elapsed_seconds'] for j in problem['jobs'])
    tracks = {r['track_id']:r for r in data['tracks']}
    stations = {r['station_id']:r for r in data['stations']}
    trips = {r['trip_id']:r for r in data['plans'][0]['trips']}
    for j in problem['jobs']:
        assert j['run_seconds'] == sum(tracks[t]['travel_seconds'] +
            (stations[tracks[t]['to_station']]['turnaround_seconds'] or
             stations[tracks[t]['to_station']]['stop_seconds']) for t in trips[j['job_id']]['route'])


def test_only_explicit_declared_depot_reserve_is_available(movement_case):
    data, motion = movement_case
    problem = build_joint_problem(data, 6300, ['B010', 'Y010', 'Y025'], source_motion=motion)
    reserve = next(t for t in problem['availability'] if t['train_id'] == 'Y025')
    assert reserve['on_line'] is False and reserve['release_seconds'] == 780
    assert reserve['readiness_status'] == 'assumed'
    with pytest.raises(ValueError, match='declared reserve'):
        build_joint_problem(data, 6300, ['B010', 'Y010', 'Y026'], source_motion=motion)


def test_real_identity_assignment_keeps_committed_motion_and_all_future_slots(movement_case):
    data, motion = movement_case
    problem = build_joint_problem(data, 6300, ['B010', 'B011', 'Y010', 'Y011'],
                                  window_seconds=30, source_motion=motion)
    solution = dict(assignments={j['job_id']:j['original_train_id'] for j in problem['jobs']},
                    departures={j['job_id']:j['original_departure_second'] for j in problem['jobs']})
    changed, audit = apply_joint_dispatch(data, problem, solution, service_end_second=22200)
    assert audit['movement_passed']
    assert audit['executed_prefix_preserved'] and audit['current_movements_preserved']
    assert audit['mandatory_jobs_preserved'] and audit['untouched_departures_preserved']
    assert audit['terminal_resources']['passed']
    assert audit['peak_trains_on_line'] <= 53
    assert audit['obligations_closed']
    assert {t['trip_id'] for t in changed['plans'][0]['trips'] if t['passenger']} == {
        t['trip_id'] for t in data['plans'][0]['trips'] if t['passenger']}


def test_missing_job_cannot_reach_the_physical_adapter(movement_case):
    data, motion = movement_case
    problem = build_joint_problem(data, 6300, ['B010', 'Y010'], source_motion=motion)
    with pytest.raises(ValueError, match='every mandatory'):
        apply_joint_dispatch(data, problem, dict(assignments={}, departures={}))


def test_free_future_terminal_order_uses_joint_ready_calendar(movement_case, monkeypatch):
    import metro44_terminal_timing
    data, motion = movement_case
    problem = build_joint_problem(data, 6300, ['B010', 'Y010'], source_motion=motion)
    solution = dict(assignments={j['job_id']:j['original_train_id'] for j in problem['jobs']},
                    departures={j['job_id']:j['original_departure_second'] for j in problem['jobs']},
                    ready_seconds={j['job_id']:j['source_ready_second']-10 for j in problem['jobs']})
    tracks = {t['track_id']:t for t in data['tracks']}
    def captured(value, *args, **kwargs):
        trips = {t['trip_id']:t for t in value['plans'][0]['trips']}
        for identity,ready in solution['ready_seconds'].items():
            last = len(trips[identity]['route'])-1
            leg = next(r for r in value['leg_timing'] if r['trip_id']==identity and r['leg_index']==last)
            track = tracks[trips[identity]['route'][last]]
            assert leg['departure_second']+track['travel_seconds']+leg.get('approach_holding_seconds',0) == ready-180
        assert metro44_terminal_timing._started_movements(value, 6300) == metro44_terminal_timing._started_movements(data, 6300)
        raise RuntimeError('captured joint calendar')
    monkeypatch.setattr(metro44_terminal_timing, 'compile_terminal_timing', captured)
    with pytest.raises(RuntimeError, match='captured joint calendar'):
        apply_joint_dispatch(data, problem, solution)


def test_bounded_terminal_repair_only_holds_unstarted_arrival(movement_case):
    from metro44_assignment_plan import terminal_resources, _current_movements_preserved
    from simulation import simulate
    from rolling import operational_headways
    data, _ = movement_case
    bad = copy.deepcopy(data)
    row = next(r for r in bad['leg_timing'] if r['trip_id']=='B010:H0003' and r['leg_index']==17)
    row['approach_holding_seconds'] -= 30
    original = copy.deepcopy(bad)
    assert not terminal_resources(bad, simulate(bad,bad['plans'][0]['plan_id']))['passed']
    fixed, audit, motion = repair_joint_holds(bad, 6300)
    assert bad == original and audit['passed']
    assert audit['repairs']
    assert _current_movements_preserved(bad,fixed,6300)
    assert [r['departure_second'] for r in fixed['leg_timing']] == [r['departure_second'] for r in bad['leg_timing']]
    assert operational_headways(fixed,motion,6300,22200)['passed']
    assert motion['obligations_closed'] and motion['checks']['peak_trains_on_line']<=53


def test_terminal_repair_refuses_current_approach_movement(movement_case):
    data, _ = movement_case
    bad = copy.deepcopy(data)
    row = next(r for r in bad['leg_timing'] if r['trip_id']=='B010:H0003' and r['leg_index']==17)
    row['approach_holding_seconds'] -= 30
    with pytest.raises(ValueError, match='committed'):
        repair_joint_holds(bad, 12250)
    with pytest.raises(TimeoutError):
        repair_joint_holds(bad,6300,deadline=time.monotonic()-1)


def test_controller_producer_returns_explicit_bounded_failure(movement_case):
    data, _ = movement_case
    candidate, audit = propose_joint_dispatch(data,6300,['unknown'])
    assert candidate is None and audit['movement_passed'] is False
    assert audit['reason'] and audit['cash_saving_rub'] is None
    candidate, audit = propose_joint_dispatch(data,6300,['B010'],deadline=time.monotonic()-1)
    assert candidate is None and audit['status']=='timeout'


def test_optional_clock_stability_keeps_source_clock_and_bounds_all_secondary_terms():
    flexible = dict(job(last=10),original_departure_second=5,max_run_seconds=400)
    assert solve_joint_dispatch([flexible],[train()])['departures']['J']==0
    stable = solve_joint_dispatch([flexible],[train()],clock_stability=True)
    assert stable['departures']['J']==5 and stable['ready_seconds']['J']==285
    assert stable['secondary_upper_bound']==125
    assert stable['resource_lex_multiplier']==126


def test_resource_priority_dominates_clock_and_route_padding_bound():
    flexible = dict(job(last=40000,km=0.001),original_departure_second=0,max_run_seconds=70280)
    answer = solve_joint_dispatch([flexible],[train(),train('Y','yubileyny',ready=40000)],clock_stability=True)
    assert answer['assignments']=={'J':'Y'}
    assert answer['secondary_upper_bound']==110000
    assert answer['resource_lex_multiplier']==110001


def test_optional_scope_allows_only_wholly_unstarted_empty_exit(movement_case):
    data, _ = movement_case
    changed = copy.deepcopy(data)
    row = next(r for r in changed['leg_timing'] if r['trip_id']=='G44BRIDGE:6300:B005:EXIT')
    row['departure_second'] += 1
    assert not departure_scope(data,changed,6300)['preserved']
    relaxed = departure_scope(data,changed,6300,allow_future_exit_retiming=True)
    assert relaxed['preserved'] and relaxed['allowed_exit_changes'][0]['delay_seconds']==1
    assert not departure_scope(data,changed,row['departure_second'],allow_future_exit_retiming=True)['preserved']


def test_optional_exit_scope_still_rejects_unselected_service_and_entry(movement_case):
    data, _ = movement_case
    changed = copy.deepcopy(data)
    row = next(r for r in changed['leg_timing'] if r['trip_id']=='B005:H0003' and r['leg_index']==0)
    row['departure_second'] += 1
    assert not departure_scope(data,changed,6300,allow_future_exit_retiming=True)['preserved']
    source = dict(plans=[dict(trips=[dict(trip_id='E',train_id='T',passenger=False,route=['ENTRY_AVTOVO'])])],
                  leg_timing=[dict(trip_id='E',train_id='T',leg_index=0,departure_second=100)])
    changed = copy.deepcopy(source);changed['leg_timing'][0]['departure_second'] += 1
    assert not departure_scope(source,changed,0,allow_future_exit_retiming=True)['preserved']
