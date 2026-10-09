import pytest

from metro44_window import paired_window_bounds, window_ledger


def case(second_departure=12, train_type='Baltiets'):
    data = dict(trains=[dict(train_id='T1', train_type=train_type, state='standby', location='A')],
                tracks=[dict(track_id='AB', length_km='10')])
    events = [dict(train_id='T1', event_type='departure', second=0, location='AB', next_state='moving', trip_id='first'),
              dict(train_id='T1', event_type='arrival', second=10, location='B', next_state='standby', trip_id='first'),
              dict(train_id='T1', event_type='departure', second=second_departure, location='AB', next_state='moving', trip_id='next'),
              dict(train_id='T1', event_type='arrival', second=second_departure+10, location='A', next_state='standby', trip_id='next')]
    return data, dict(events=events)


def test_arrival_boundary_is_counted_once():
    data, motion = case()
    result = window_ledger(data, motion, 5, 10)
    assert result['arrival_train_km'] == 10
    assert result['end_movement_km_upper_bound'] == 0


def test_departure_at_end_cannot_accrue_distance_inside_window():
    data, motion = case(15)
    assert window_ledger(data, motion, 5, 15)['end_movement_km_upper_bound'] == 0


def test_shared_start_cancels_unknown_initial_position():
    reference, ref_motion = case()
    candidate, cand_motion = case(18)
    result = paired_window_bounds(reference, ref_motion, candidate, cand_motion, 5, 15)
    assert result['saving_train_km_lower_bound'] == 0
    assert result['saving_train_km_upper_bound'] == 10
    assert result['saving_to1_resource_rub_upper_bound'] == 92
    assert result['cash_saving_rub'] is None


def test_positive_lower_bound_and_fixed_fleet_denominator():
    data, motion = case()
    candidate, cand_motion = case(30)
    result = paired_window_bounds(data, motion, candidate, cand_motion, 0, 25)
    assert result['saving_train_km_lower_bound'] == 10
    assert result['saving_km_per_fixed_train_lower_bound'] == 10
    assert result['saving_to1_resource_rub_lower_bound'] == 92


def test_type_specific_resource_rate():
    data, motion = case(train_type='Yubileyny')
    candidate, cand_motion = case(30, 'Yubileyny')
    assert paired_window_bounds(data, motion, candidate, cand_motion, 0, 25)['saving_to1_resource_rub_lower_bound'] == 74


def test_different_start_obligation_rejected():
    data, motion = case()
    candidate, cand_motion = case()
    cand_motion['events'][1]['second'] = 11
    with pytest.raises(ValueError, match='common start'):
        paired_window_bounds(data, motion, candidate, cand_motion, 5, 15)


def test_changed_type_and_invalid_window_rejected():
    data, motion = case()
    candidate, cand_motion = case(train_type='Yubileyny')
    with pytest.raises(ValueError, match='fixed fleet'):
        paired_window_bounds(data, motion, candidate, cand_motion, 5, 15)
    with pytest.raises(ValueError, match='window'):
        window_ledger(data, motion, 15, 5)
