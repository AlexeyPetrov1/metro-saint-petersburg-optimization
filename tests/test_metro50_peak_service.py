import copy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from metro50_peak_service import (SCENARIOS, apply_peak_safety_margin,
                                  peak_boarding_service)


def fixture(*, count=100, boards=((60, 95), (180, 5)), departure_times=(60, 180),
            platform=0, end=900, day='2026-09-30', hour='08:00:00', stations=('A', 'B')):
    tracks = {f'{a}{b}': dict(from_station=a, to_station=b)
              for a, b in zip(stations, stations[1:])}
    route = list(tracks)
    motion = [dict(kind='motion', event_type='departure', second=second,
                   event_id=f'D{second}', train_id=f'T{second}', location=route[0],
                   passenger_service=True, remaining_route=route)
              for second in departure_times]
    group = dict(group_id='G', turnstile_second=0, platform_second=platform,
                 origin='A', count=count,
                 destinations=[dict(station_id=stations[-1], count=count, direction=1)])
    runner = SimpleNamespace(groups=[group], by_id={'G': group}, positions={s: i for i, s in enumerate(stations)},
                             tracks=tracks, stream=motion, duration=3600,
                             motion={'period_start': f'{day}T{hour}+03:00'})
    boarded = [dict(group_id='G', motion_event_id=f'D{second}', second=second,
                    wait_seconds=second-platform, count=n) for second, n in boards if second < end]
    residual = count-sum(r['count'] for r in boarded)
    queued = [dict(group_id='G', origin='A', destination=stations[-1], direction=1,
                   platform_second=platform, count=residual)] if residual and platform < end else []
    state = dict(second=end, next_event_index=sum(e['second'] < end for e in motion),
                 board_events=boarded, queues={'A:1': queued}, approaching=count if platform >= end else 0,
                 queue_events=[], queue_samples=[])
    return runner, state


def test_unique_first_misses_are_not_repeated_denied_attempts():
    runner, state = fixture(boards=((60, 80), (180, 10), (300, 10)), departure_times=(60, 180, 300))
    result = peak_boarding_service(runner, state, missed_share_limit=.2)
    a = result['aggregate']
    assert a['passenger_count'] == a['first_opportunity_count'] == 100
    assert a['missed_first_count'] == 20
    assert a['boarded_second_count'] == 10
    assert a['missed_second_count'] == 10
    assert a['missed_opportunity_count'] == 30
    assert a['missed_first_share'] == .2
    assert result['status'] == 'passed'
    assert a['second_train_missed_count'] == 10


def test_limit_is_inclusive_and_applies_to_each_peak_quarter():
    runner, state = fixture()
    result = peak_boarding_service(runner, state)
    assert result['passed'] is True
    assert result['intervals'][0]['missed_first_share'] == .05
    assert peak_boarding_service(runner, state, missed_share_limit=.049)['status'] == 'failed'


def test_still_queued_after_departures_counts_as_known_missed():
    runner, state = fixture(boards=((60, 90),), departure_times=(60, 180))
    a = peak_boarding_service(runner, state, missed_share_limit=.1)['aggregate']
    assert a['missed_first_count'] == a['missed_second_count'] == a['still_queued_count'] == 10
    assert a['missed_opportunity_count'] == 20
    assert a['max_current_wait_seconds'] == 900


def test_no_eligible_departure_is_censored_never_passed():
    runner, state = fixture(boards=(), departure_times=())
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['first_opportunity_count'] == 0
    assert result['aggregate']['first_opportunity_censored_count'] == 100
    assert result['aggregate']['missed_first_share'] is None
    assert result['status'] == 'incomplete' and result['passed'] is None


def test_unobserved_second_departure_is_descriptive_unless_explicitly_required():
    runner, state = fixture(boards=((60, 95),), departure_times=(60,))
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['second_opportunity_censored_count'] == 5
    assert result['status'] == 'passed'
    strict = peak_boarding_service(runner, state, require_next_train=True)
    assert strict['status'] == 'incomplete' and strict['passed'] is None


def test_second_misses_do_not_add_an_unrequested_zero_tolerance_gate():
    runner, state = fixture(count=1000, boards=((60, 970), (180, 26), (300, 4)),
                            departure_times=(60, 180, 300))
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['missed_first_share'] == .03
    assert result['aggregate']['second_train_missed_count'] == 4
    assert result['aggregate']['still_queued_count'] == 0
    assert result['passed'] is True and result['next_train_required'] is False
    assert peak_boarding_service(runner, state, require_next_train=True)['passed'] is False


@pytest.mark.parametrize('required', [0, 1, None, 'false'])
def test_next_train_switch_requires_an_explicit_boolean(required):
    runner, state = fixture()
    with pytest.raises(ValueError, match='boolean'):
        peak_boarding_service(runner, state, require_next_train=required)


def test_unobserved_first_departure_and_partial_cohort_are_incomplete():
    runner, state = fixture(end=30)
    result = peak_boarding_service(runner, state)
    assert result['intervals'][0]['temporal_coverage_complete'] is False
    assert result['status'] == 'incomplete'
    assert result['aggregate']['first_opportunity_censored_count'] == 100


def test_arrival_at_departure_can_board_first_train():
    runner, state = fixture(platform=60, boards=((60, 100),))
    assert peak_boarding_service(runner, state)['aggregate']['missed_first_count'] == 0


def test_approaching_people_are_excluded_from_platform_cohort():
    runner, state = fixture(platform=1000, end=900, boards=())
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['passenger_count'] == 0
    assert result['approaching_count'] == 100
    assert result['status'] == 'not_applicable'


@pytest.mark.parametrize('day,hour', [('2026-10-03', '08:00:00'), ('2026-09-30', '10:00:00')])
def test_peak_uses_weekdays_and_exclusive_end(day, hour):
    runner, state = fixture(day=day, hour=hour)
    assert peak_boarding_service(runner, state)['status'] == 'not_applicable'


def test_peak_cohort_is_selected_by_platform_arrival_not_turnstile():
    runner, state = fixture(hour='07:59:00', platform=120, end=960,
                            boards=((180, 100),), departure_times=(180, 300))
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['passenger_count'] == 100
    assert result['intervals'][0]['interval_start'] == '2026-09-30T08:00:00+03:00'
    assert result['status'] == 'passed'


def test_short_turn_not_supporting_destination_is_not_eligible():
    runner, state = fixture(stations=('A', 'B', 'C'), boards=((180, 100),))
    runner.stream[0]['remaining_route'] = ['AB']
    assert peak_boarding_service(runner, state)['aggregate']['missed_first_count'] == 0


def test_same_group_destinations_are_reconstructed_from_fifo_ledger():
    runner, state = fixture(stations=('A', 'B', 'C'), count=5, boards=())
    runner.groups[0]['destinations'] = [dict(station_id='B', count=3, direction=1),
                                         dict(station_id='C', count=2, direction=1)]
    runner.stream[0]['remaining_route'] = ['AB']
    state['board_events'] = [dict(group_id='G', motion_event_id='D60', second=60, count=3, wait_seconds=60),
                             dict(group_id='G', motion_event_id='D180', second=180, count=2, wait_seconds=180)]
    state['queues'] = {}
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['missed_first_count'] == 0
    assert result['aggregate']['passenger_count'] == 5


def test_small_station_cohort_is_descriptive_not_separate_hard_gate():
    runner, state = fixture(count=1000, boards=((60, 1000),), stations=('A', 'B', 'C'))
    extra = copy.deepcopy(runner.groups[0]); extra.update(group_id='X', count=1, origin='B')
    extra['destinations'] = [dict(station_id='C', count=1, direction=1)]
    runner.groups.append(extra); runner.by_id['X'] = extra
    for second in (60, 180):
        event = copy.deepcopy(runner.stream[0])
        event.update(event_id=f'B{second}', second=second, location='BC', remaining_route=['BC'])
        runner.stream.append(event)
    runner.stream.sort(key=lambda e: e['second'])
    state['next_event_index'] = len(runner.stream)
    state['board_events'].append(dict(group_id='X', motion_event_id='B180', second=180, count=1))
    state['queues'] = {}
    result = peak_boarding_service(runner, state)
    assert result['passed'] is True
    assert result['aggregate']['missed_first_count'] == 1
    assert next(c for c in result['cohorts'] if c['station_id'] == 'B')['status'] == 'failed'


def test_every_quarter_is_binding_even_when_daily_average_passes():
    runner, state = fixture(count=1000, boards=((60, 1000),), end=1800)
    extra = copy.deepcopy(runner.groups[0]); extra.update(group_id='X', count=10, platform_second=900)
    extra['destinations'][0]['count'] = 10
    runner.groups.append(extra); runner.by_id['X'] = extra
    for second in (960, 1080):
        event = copy.deepcopy(runner.stream[0]); event.update(event_id=f'D{second}', second=second)
        runner.stream.append(event)
    state['next_event_index'] = len(runner.stream)
    state['board_events'].extend([dict(group_id='X', motion_event_id='D960', second=960, count=9),
                                  dict(group_id='X', motion_event_id='D1080', second=1080, count=1)])
    state['queues'] = {}
    result = peak_boarding_service(runner, state)
    assert result['aggregate']['status'] == 'passed'
    assert result['intervals'][1]['status'] == result['status'] == 'failed'


def test_later_departures_observed_after_cohort_end_complete_followup():
    runner, state = fixture(platform=899, boards=((960, 95), (1080, 5)),
                            departure_times=(960, 1080), end=1800)
    result = peak_boarding_service(runner, state, end_second=900)
    assert result['passed'] is True
    assert result['aggregate']['missed_first_count'] == 5


@pytest.mark.parametrize('seed', [None, 20261009])
def test_real_grouped_runner_uses_the_same_ledger_contract(seed):
    from passenger_groups import GroupedPassengerSimulation
    movement = dict(schema_version='1.0', source='peak fixture', previous_departures=[], max_trains_on_line=2,
                    period_start='2026-09-30T08:00:00+03:00', period_end='2026-09-30T08:15:00+03:00',
                    stations=[dict(station_id=s, kind='line', stop_seconds=0, turnaround_seconds=0) for s in ('A', 'B')],
                    tracks=[dict(track_id='AB', from_station='A', to_station='B', length_km='1', travel_seconds=120)],
                    trains=[dict(train_id=t, train_type='fixture', cars=1, location='A', state='standby', ready_after_seconds=0)
                            for t in ('T1', 'T2')],
                    plans=[dict(plan_id='P', trips=[dict(trip_id=t, train_id=t, departure_second=second,
                                                        route=['AB'], passenger=True) for t, second in [('T1', 60), ('T2', 180)]])])
    groups = fixture()[0].groups
    profiles = {t: [dict(car_id='C', car_type='fixture', max_capacity=95,
                         normal_threshold=80, source='fixture')] for t in ('T1', 'T2')}
    runner = GroupedPassengerSimulation(movement, 'P', groups, profiles, boarding_seed=seed)
    result = peak_boarding_service(runner, runner.run()['checkpoint'])
    assert result['aggregate']['missed_first_count'] == 5 and result['passed'] is True


def test_destination_ledger_does_not_require_legacy_fifo_destination_order():
    runner, state = fixture(stations=('A', 'B', 'C'), count=5, boards=())
    runner.groups[0]['destinations'] = [dict(station_id='B', count=3, direction=1),
                                         dict(station_id='C', count=2, direction=1)]
    state['board_events'] = [dict(group_id='G', destination='C', motion_event_id='D60', second=60, count=2),
                             dict(group_id='G', destination='B', motion_event_id='D60', second=60, count=3)]
    state['queues'] = {}
    assert peak_boarding_service(runner, state)['passed'] is True


def test_window_excludes_earlier_cohorts_and_does_not_mutate_ledger():
    runner, state = fixture()
    before = copy.deepcopy(state)
    result = peak_boarding_service(runner, state, start_second=900)
    assert result['aggregate']['passenger_count'] == 0
    assert state == before


def test_broken_board_queue_balance_is_rejected():
    runner, state = fixture()
    state['queues'] = {'A:1': [dict(group_id='G', destination='B', count=1)]}
    with pytest.raises(ValueError, match='balance'):
        peak_boarding_service(runner, state)


def forecast(start='2026-09-30T07:45:00+03:00'):
    dt = datetime.fromisoformat(start)
    return dict(schema_version='1.0', model_version='frozen', feature_version='causal',
                issued_at=start, as_of=start, available_at=start, features_available_at=start,
                target='turnstile_entries', unit='passengers_per_15m', coverage='complete',
                included_event_ids=[], refresh_seconds=900, source_versions={'causal': True},
                rows=[dict(station_id='A', interval_start=(dt+timedelta(minutes=15*q)).isoformat(),
                           interval_end=(dt+timedelta(minutes=15*(q+1))).isoformat(),
                           predicted_count=101, coverage='complete') for q in range(12)])


def test_named_scenarios_and_dense_margin_only_relax_weekday_peaks():
    assert SCENARIOS['current5']['missed_share_limit'] == .05
    assert SCENARIOS['dense10']['missed_share_limit'] == .1
    source = forecast(); before = copy.deepcopy(source)
    result = apply_peak_safety_margin(source, source['as_of'], scenario='dense10')
    assert [r['predicted_count'] for r in result['rows']] == [127]+[101]*8+[127]*3
    assert source == before
    assert result['source_versions']['engineering_safety_margin']['statistical_coverage_guarantee'] is False


def test_current_margin_retains_25_percent_everywhere_and_dense_weekend_does_too():
    source = forecast()
    assert all(r['predicted_count'] == 127 for r in apply_peak_safety_margin(source, source['as_of'])['rows'])
    source = forecast('2026-10-03T07:45:00+03:00')
    assert all(r['predicted_count'] == 127 for r in apply_peak_safety_margin(source, source['as_of'], scenario='dense10')['rows'])


def test_peak_margin_rejects_double_application_and_unknown_scenario():
    source = forecast()
    with pytest.raises(ValueError, match='already'):
        apply_peak_safety_margin(apply_peak_safety_margin(source, source['as_of']), source['as_of'])
    with pytest.raises(ValueError):
        apply_peak_safety_margin(source, source['as_of'], scenario='unknown')


@pytest.mark.parametrize('limit', [-.1, 1.1, True, float('nan')])
def test_invalid_limits_rejected(limit):
    runner, state = fixture()
    with pytest.raises(ValueError):
        peak_boarding_service(runner, state, missed_share_limit=limit)
