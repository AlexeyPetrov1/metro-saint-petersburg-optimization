from collections import Counter
from copy import deepcopy

import pytest

from metro44_arrivals import jitter_arrivals


def groups():
    return [dict(group_id=f'g{i}', turnstile_second=840+i*60, platform_second=960+i*60,
                 station_id='A', count=i+1, destinations=[dict(station_id='B', count=i+1)])
            for i in range(4)]


def test_jitter_preserves_counts_minutes_quarters_walk_and_destinations():
    before = groups(); after = jitter_arrivals(before, 7)
    census = lambda rows: Counter({(g['group_id'], g['turnstile_second']//60,
        g['turnstile_second']//900): g['count'] for g in rows})
    assert census(before) == census(after)
    for old, new in zip(before, after):
        assert new['platform_second']-new['turnstile_second'] == 120
        assert {k:v for k,v in old.items() if k not in ('turnstile_second','platform_second')} == {
            k:v for k,v in new.items() if k not in ('turnstile_second','platform_second')}


def test_input_is_immutable_and_nested_rows_are_independent():
    before = groups(); original = deepcopy(before); after = jitter_arrivals(before, 7)
    after[0]['destinations'][0]['count'] = 100
    assert before == original


def test_seed_is_reproducible_and_independent_of_group_order():
    before = groups()
    assert jitter_arrivals(before, 7) == jitter_arrivals(before, 7)
    assert jitter_arrivals(before, 7) != jitter_arrivals(before, 11)
    assert jitter_arrivals(before, 7) == list(reversed(jitter_arrivals(list(reversed(before)), 7)))


@pytest.mark.parametrize('field,value', [('turnstile_second',-1), ('turnstile_second',1.5),
                                       ('platform_second',1)])
def test_invalid_times_rejected(field, value):
    before = groups(); before[0][field] = value
    with pytest.raises(ValueError):
        jitter_arrivals(before, 7)


def test_empty_input():
    assert jitter_arrivals([], 7) == []
