import pytest

from metro44_sensitivity_independent import check_jitter, original_order


def test_jitter_preserves_minute_destinations_count_and_access_delay():
    original = [{'group_id': 'a', 'turnstile_second': 60, 'platform_second': 180,
                 'count': 3, 'destinations': [{'station_id': 'b', 'count': 3}]}]
    changed = [dict(original[0], turnstile_second=87, platform_second=207)]
    assert check_jitter(original, changed)
    changed[0]['count'] = 4
    with pytest.raises(AssertionError):
        check_jitter(original, changed)


def test_crossing_minute_or_altering_access_time_is_rejected():
    original = [{'group_id': 'a', 'turnstile_second': 60, 'platform_second': 180}]
    for changed in [{'group_id': 'a', 'turnstile_second': 120, 'platform_second': 240},
                    {'group_id': 'a', 'turnstile_second': 87, 'platform_second': 208}]:
        with pytest.raises(AssertionError):
            check_jitter(original, [changed])


def test_saved_time_order_is_restored_for_the_original_input_digest():
    first = [{'group_id': 'a'}, {'group_id': 'b'}]
    saved = [{'group_id': 'b', 'x': 3}, {'group_id': 'a', 'x': 2}]
    assert original_order(first, saved) == [saved[1], saved[0]]
