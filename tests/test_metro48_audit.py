from copy import deepcopy

import pytest

from metro48_audit import control_receipts, transition_gate


def source():
    return dict(stations=[dict(station_id=s, kind='line', turnaround_seconds=0) for s in ('A', 'B', 'C')],
        tracks=[dict(track_id='P', from_station='A', to_station='B', travel_seconds=60),
                dict(track_id='Q', from_station='B', to_station='C', travel_seconds=60)],
        trains=[dict(train_id='T', state='standby', location='A', ready_after_seconds=0)],
        headway_policy={'parameters': {'depot_release_s': {'value': {'Avtovo': 720, 'Northern': 780}}}},
        plans=[dict(trips=[dict(trip_id='S', train_id='T', route=['P', 'Q'], passenger=True)])],
        leg_timing=[dict(train_id='T', trip_id='S', leg_index=i, departure_second=t,
                        service_seconds=0, approach_holding_seconds=0)
                    for i, t in enumerate((100, 200))])


def motion():
    return dict(events=[dict(second=100, event_type='departure', train_id='T', location='P')])


def check(before, after, old=None, new=None):
    return transition_gate(before, after, 150, before_motion=old or motion(), after_motion=new or motion())


def test_whole_begun_service_remains_exact_even_for_a_future_not_started_leg():
    before = source(); after = deepcopy(before)
    assert check(before, after)['passed']
    after['leg_timing'][1]['departure_second'] += 1
    result = check(before, after)
    assert result['executed_prefix_preserved'] and result['current_movements_preserved']
    assert not result['begun_service_clocks_preserved'] and not result['passed']


def test_current_arrival_and_executed_prefix_are_independently_required():
    before = source(); after = deepcopy(before)
    after['leg_timing'][0]['approach_holding_seconds'] = 1
    assert not check(before, after)['current_movements_preserved']
    new = motion(); new['events'][0]['location'] = 'Q'
    assert not check(before, before, new=new)['executed_prefix_preserved']


def test_new_past_movement_and_changed_geometry_are_never_allowed():
    before = source(); after = deepcopy(before)
    after['leg_timing'].append(dict(train_id='T', trip_id='S', leg_index=2, departure_second=140, service_seconds=0))
    after['plans'][0]['trips'][0]['route'].append('P')
    assert not check(before, after)['current_movements_preserved']
    after = deepcopy(before); after['tracks'][0]['travel_seconds'] += 1
    assert not check(before, after)['same_geometry']


def test_started_depot_preparation_preserves_its_not_yet_started_service():
    before = source()
    before['stations'].append(dict(station_id='DEPOT_AVTOVO', kind='depot', turnaround_seconds=0))
    before['tracks'].append(dict(track_id='ENTRY_AVTOVO', from_station='DEPOT_AVTOVO', to_station='A', travel_seconds=60))
    before['trains'][0]['location'] = 'DEPOT_AVTOVO'
    before['plans'][0]['trips'].append(dict(trip_id='E', train_id='T', route=['ENTRY_AVTOVO'], passenger=False))
    before['leg_timing'][0]['departure_second'] = 500
    before['leg_timing'][1]['departure_second'] = 700
    before['leg_timing'].append(dict(train_id='T', trip_id='E', leg_index=0, departure_second=100, service_seconds=0))
    after = deepcopy(before); after['leg_timing'][0]['departure_second'] += 1
    result = check(before, after)
    assert result['executed_prefix_preserved'] and result['current_movements_preserved']
    assert not result['preparation_commitments_preserved'] and not result['passed']


def decisions(times):
    return [dict(decision_second=s, pipeline_seconds=2.5,
                 local_recommendation_within_480=True, external_delivery_measured=False) for s in times]


def test_every_required_quarter_and_local_480_second_receipt_are_checked():
    times = list(range(3600, 21600, 900))
    result = control_receipts(decisions(times), times)
    assert result['passed'] and result['decision_count'] == 20
    assert result['external_delivery_measured'] is False
    assert not control_receipts(decisions(times[:-1]), times)['passed']


@pytest.mark.parametrize('seconds', [480.0001, -1, True, float('nan')])
def test_invalid_or_excess_pipeline_duration_is_rejected(seconds):
    rows = decisions([3600]); rows[0]['pipeline_seconds'] = seconds
    assert not control_receipts(rows, [3600])['passed']


def test_unknown_receipt_cannot_be_promoted_to_within_budget():
    rows = decisions([3600]); rows[0].pop('local_recommendation_within_480')
    assert not control_receipts(rows, [3600])['passed']
