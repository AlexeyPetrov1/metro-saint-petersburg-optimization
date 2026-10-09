import pytest

from metro44_noidle_terminal_6h import headway_profile, trial_scope, validate_decision, validate_scope


def test_real_entries_cover_the_extended_six_hour_window():
    data = {'period_start': '2026-09-30T08:15:00+03:00', 'period_end': '2026-09-30T17:15:00+03:00'}
    groups = [{'turnstile_second': 21540, 'source_id': 'uploaded-spb-line1-15min'}]
    validate_scope(data, groups)
    with pytest.raises(ValueError, match='observed six-hour'):
        validate_scope(data, [dict(groups[0], turnstile_second=14340)])


def test_continuation_is_not_cut_at_original_four_hour_boundary():
    data = {'period_start': '2026-09-30T08:15:00+03:00', 'period_end': '2026-09-30T16:15:00+03:00'}
    with pytest.raises(ValueError, match='three-hour'):
        validate_scope(data, [{'turnstile_second': 21540, 'source_id': 'uploaded-spb-line1-15min'}])


def test_clearance_extends_service_without_extending_admission():
    scope = trial_scope(600)
    assert scope == {'admission_end_second': 21600, 'passenger_departure_end_second': 22200}
    assert scope['passenger_departure_end_second'] - scope['admission_end_second'] >= 120 + 480


def test_negative_clearance_is_rejected():
    with pytest.raises(ValueError, match='clearance'):
        trial_scope(-1)


def test_lower_offpeak_frequency_starts_at_actual_ten_oclock():
    rows = headway_profile(120, 160, 22200)
    assert rows == [{'start_second': 0, 'end_second': 6300, 'headway_seconds': 120},
                    {'start_second': 6300, 'end_second': 22200, 'headway_seconds': 160}]


def test_peak_can_remain_committed_until_ten_oclock():
    assert validate_decision(6300) == 6300
    with pytest.raises(ValueError, match='15-minute'):
        validate_decision(6301)
