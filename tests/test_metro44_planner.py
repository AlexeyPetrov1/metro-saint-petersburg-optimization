import copy
from datetime import datetime, timedelta
from time import monotonic

import pytest

from dwell import compile_operational_timing, timing_key
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro_daily import prepare_day
from metro44_planner import headway_seconds, rebuild_timetable, dense_reference, trip_finish
from rolling import operational_headways
from simulation import simulate, digest


def test_headway_has_only_user_limits_and_independent_quarter_profiles():
    policy = load_policy()
    origin = datetime.fromisoformat('2026-09-30T06:00:00+03:00')
    profile = [dict(terminal='L1_01', start_second=7200, end_second=8100, headway_seconds=113),
               dict(terminal='L1_01', start_second=8100, end_second=9000, headway_seconds=135),
               dict(terminal='L1_19', start_second=7200, end_second=9000, headway_seconds=170)]
    assert headway_seconds(profile, origin+timedelta(seconds=7200), 'L1_01', policy, origin) == 113
    assert headway_seconds(profile, origin+timedelta(seconds=8100), 'L1_01', policy, origin) == 135
    assert headway_seconds(profile, origin+timedelta(seconds=7200), 'L1_19', policy, origin) == 170
    assert headway_seconds(113, origin, 'L1_01', policy) == 150
    assert headway_seconds(None, origin+timedelta(hours=18), 'L1_01', policy) == 480
    with pytest.raises(ValueError, match='headway'):
        headway_seconds(481, origin, 'L1_01', policy)


def test_readiness_does_not_count_departure_holding_twice():
    trip = dict(train_id='T', trip_id='P', departure_second=50, route=['R'])
    timings = {timing_key('T', 'P', 0): dict(departure_second=100, holding_seconds=50, service_seconds=20)}
    tracks = {'R': dict(to_station='B', travel_seconds=140)}
    stations = {'B': dict(stop_seconds=20, turnaround_seconds=0)}
    assert trip_finish(trip, timings, tracks, stations) == ('B', 260)


def test_readiness_matches_actual_turnaround_complete_with_intermediate_holds():
    policy = load_policy()
    data, _, _, _, _ = build_operational_day(policy, count=1, service_hours=1, small_line=['A', 'B', 'C'])
    trip = next(t for t in data['plans'][0]['trips'] if t['passenger'] and len(t['route']) == 2)
    old = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    commands = {timing_key(trip['train_id'], trip['trip_id'], i): old[timing_key(trip['train_id'],trip['trip_id'],i)]['departure_second']+20+40*i
                for i in range(2)}
    changed, _ = compile_operational_timing(data, data['plans'][0]['plan_id'], policy=policy,
                                           departure_not_before=commands)
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in changed['leg_timing']}
    assert all(timing[timing_key(trip['train_id'],trip['trip_id'],i)]['holding_seconds'] > 0 for i in range(2))
    tracks = {r['track_id']: r for r in changed['tracks']}
    stations = {r['station_id']: r for r in changed['stations']}
    station, ready = trip_finish(trip, timing, tracks, stations)
    event = next(e for e in simulate(changed, changed['plans'][0]['plan_id'])['events']
                 if e['trip_id'] == trip['trip_id'] and e['event_type'] == 'turnaround_complete')
    assert (station, ready) == (event['location'], event['second'])


@pytest.fixture(scope='module')
def day():
    return prepare_day('2026-09-30', service_hours=6)


def test_replan_preserves_past_but_does_not_lock_every_train_for_thirteen_minutes(day):
    data, _, _, meta = day
    original = copy.deepcopy(data)
    changed, audit = rebuild_timetable(data, 3600, meta['service_end_second'], headway=210)
    before = simulate(data, data['plans'][0]['plan_id'])
    after = simulate(changed, changed['plans'][0]['plan_id'])
    assert data == original
    assert audit['committed_until_second'] == 3600
    assert audit['planning_occupancy_margin'] == 0
    assert [e for e in before['events'] if e['second'] < 3600] == [e for e in after['events'] if e['second'] < 3600]
    assert after['obligations_closed']
    assert audit['headways_passed']
    assert any(t['passenger'] and 3600 <= t['departure_second'] < 4380
               and t['trip_id'] not in {r['trip_id'] for r in data['plans'][0]['trips']}
               for t in changed['plans'][0]['trips'])


def test_new_depot_releases_keep_their_own_preparation_deadlines(day):
    data, _, _, meta = day
    second = 3600
    changed, _ = rebuild_timetable(data, second, meta['service_end_second'], headway=150)
    previous = {t['trip_id'] for t in data['plans'][0]['trips']}
    tracks = {r['track_id']: r for r in changed['tracks']}
    for train in changed['trains']:
        trips = sorted((t for t in changed['plans'][0]['trips'] if t['train_id'] == train['train_id']),
                       key=lambda t: t['departure_second'])
        for i, trip in enumerate(trips):
            if trip['trip_id'] in previous or not trip['route'][0].startswith('ENTRY_'):
                continue
            service = next(t for t in trips[i+1:] if t['passenger'])
            depot = tracks[trip['route'][0]]['from_station']
            release = 720 if 'AVTOVO' in depot else 780
            assert service['departure_second']-release >= second


def test_expired_generation_deadline_aborts_before_work(day):
    data, _, _, meta = day
    with pytest.raises(TimeoutError):
        rebuild_timetable(data, 3600, meta['service_end_second'], deadline=monotonic()-1)


def test_started_entry_and_its_next_passenger_trip_are_retained(day):
    data, _, _, meta = day
    timing = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r for r in data['leg_timing']}
    entry = next(t for t in data['plans'][0]['trips'] if t['route'][0].startswith('ENTRY_'))
    second = timing[timing_key(entry['train_id'], entry['trip_id'], 0)]['departure_second']+1
    future = sorted((t for t in data['plans'][0]['trips'] if t['train_id'] == entry['train_id']
                     and t['passenger'] and t['departure_second'] > entry['departure_second']),
                    key=lambda t: t['departure_second'])[0]
    changed, audit = rebuild_timetable(data, second, meta['service_end_second'], headway=200)
    retained = {t['trip_id'] for t in changed['plans'][0]['trips']}
    assert {entry['trip_id'], future['trip_id']} <= retained
    assert future['trip_id'] in audit['retained_entry_obligations']


def test_phase_and_direction_profiles_change_departures_without_teleporting(day):
    data, _, _, meta = day
    profile = {'L1_01': 190, 'L1_19': lambda at, terminal: 230 if at.minute < 30 else 210}
    changed, audit = rebuild_timetable(data, 3600, meta['service_end_second'], headway=profile,
                                      phase_seconds={'L1_01': 15, 'L1_19': 0})
    assert audit['phase_seconds'] == {'L1_01': 15, 'L1_19': 0}
    projected = audit['departures']
    assert {r['target_headway'] for r in projected if r['terminal'] == 'L1_01'} == {190}
    assert {r['target_headway'] for r in projected if r['terminal'] == 'L1_19'} <= {210, 230}
    assert audit['movement_passed']
    assert len(changed['trains']) == 68


def test_initial_rebuild_never_reuses_old_future_trip_identifiers(day):
    data, _, _, meta = day
    changed, audit = rebuild_timetable(data, 0, meta['service_end_second'], headway=150, initial=True)
    old = {t['trip_id'] for t in data['plans'][0]['trips']}
    retained = {t['trip_id'] for t in data['plans'][0]['trips']
                if ':H' in t['trip_id']}
    assert all(t['trip_id'] not in old or t['trip_id'] in retained for t in changed['plans'][0]['trips'])
    assert audit['movement_passed']


def test_cached_work_is_bound_to_the_exact_checked_plan(day):
    data, _, _, meta = day
    changed, audit = rebuild_timetable(data, 3600, meta['service_end_second'], headway=210)
    motion = simulate(changed, changed['plans'][0]['plan_id'])
    assert audit['_data_digest'] == digest(changed)
    assert audit['_transport_work'] == motion['transport_work']


def test_wear_priority_changes_only_future_vehicle_assignment(day):
    data, _, _, meta = day
    regular, ra = rebuild_timetable(data, 3600, meta['service_end_second'], headway=150)
    cheaper, ca = rebuild_timetable(data, 3600, meta['service_end_second'], headway=150, wear_priority=True)
    assert ra['movement_passed'] and ca['movement_passed']
    old = simulate(data, data['plans'][0]['plan_id'])
    new = simulate(cheaper, cheaper['plans'][0]['plan_id'])
    assert [e for e in old['events'] if e['second'] < 3600] == [e for e in new['events'] if e['second'] < 3600]
    assert regular['trains'] == cheaper['trains']
    rates = {'Baltiets': 9.2, 'Yubileyny': 7.4}
    types = {t['train_id']: t['train_type'] for t in data['trains']}
    cost = lambda work: sum(float(t['train_km'])*rates[types[t['train_id']]] for t in work)
    assert cost(ca['_transport_work']) < cost(ra['_transport_work'])


def test_dense_reference_is_checked_for_the_whole_service_day():
    data, _, _, meta = prepare_day('2026-09-30')
    changed, audit = dense_reference(data, 0, meta['service_end_second'], initial=True)
    motion = simulate(changed, changed['plans'][0]['plan_id'])
    assert motion['checks']['peak_trains_on_line'] <= 53
    assert motion['obligations_closed']
    assert operational_headways(changed, motion, 0, meta['service_end_second'])['passed']
    assert audit['movement_passed']
    assert audit['planning_occupancy_margin'] == 0
    assert audit['peak_target_headway'] >= 113
    assert audit['dense_search_complete'] is False
