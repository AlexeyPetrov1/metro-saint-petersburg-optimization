import copy

import pytest

from decision_engine import build_regular_plan, headway_report
from dwell import compile_timing, keyed_service, regular_plan_at_limit
from line1 import load_config
from simulation import simulate


def three_stations(travel=120):
    config = dict(station_ids=["A", "B", "C"], date="2026-09-01", timezone="Europe/Moscow",
        length_m=2000, travel_seconds=travel, stop_seconds=20, turnaround_seconds=10,
        train_type="fixture", car_types=["fixture"])
    return build_regular_plan(config, 4, end_second=1100)[0]


def departures(motion):
    return {(e["train_id"], e["trip_id"], e["location"]): e["second"]
        for e in motion["events"] if e["event_type"] == "departure"}


def test_fixed_reproduces_original_motion_exactly():
    data = three_stations()
    compiled, report = compile_timing(data, "line1-baseline", "fixed", 7)
    assert compiled == data
    assert simulate(compiled, "line1-baseline") == simulate(data, "line1-baseline")
    assert report["holding_seconds"] == 0


@pytest.mark.parametrize("profile,low,high", [("mild", 15, 25), ("wider", 10, 30)])
def test_stable_event_keys_keep_noise_independent_of_call_order(profile,low,high):
    keys = [("T1", "H1", i) for i in range(50)]
    first = {key: keyed_service(profile, 7, key, 20) for key in keys}
    assert first == {key: keyed_service(profile, 7, key, 20) for key in reversed(keys)}
    assert min(first.values()) >= low and max(first.values()) <= high
    assert len(set(first.values())) > 3


def test_short_and_long_service_change_the_same_train_downstream():
    data = three_stations(); data["trains"] = data["trains"][:1]
    data["plans"][0]["trips"] = [t for t in data["plans"][0]["trips"] if t["train_id"] == "T01"]
    data.pop("previous_departures")
    baseline = departures(simulate(data, "line1-baseline"))
    for seconds in (10, 30):
        changed,_ = compile_timing(data,"line1-baseline",dict(profile="constant",seconds=seconds),3)
        actual = departures(simulate(changed,"line1-baseline"))
        key = ("T01", "T01:H0000", "P02_up")
        assert actual[key]-baseline[key] == seconds-20
        next_trip = ('T01', 'T01:H0001', 'P02_down')
        assert actual[next_trip]-baseline[next_trip] == seconds-20


def test_noise_causes_real_holding_and_preserves_headway():
    data = three_stations(103)
    changed, report = compile_timing(data,"line1-baseline","wider",0)
    assert report["holding_seconds"] > 0
    assert headway_report(changed,simulate(changed,"line1-baseline"))["minimum_seconds"] >= 113
    assert report["infrastructure_status"] == "unknown"


def test_candidate_permutation_and_input_mutation_do_not_change_timing():
    data = three_stations(); before=copy.deepcopy(data)
    a,_=compile_timing(data,"line1-baseline","mild",5)
    compile_timing(data,"line1-baseline","wider",9)
    b,_=compile_timing(data,"line1-baseline","mild",5)
    assert a == b and data == before


def test_earlier_departure_still_requires_the_previous_leg_to_finish():
    data = three_stations(); data['trains'] = data['trains'][:1]
    data['plans'][0]['trips'] = [t for t in data['plans'][0]['trips'] if t['train_id'] == 'T01']
    data.pop('previous_departures')
    changed, _ = compile_timing(data, 'line1-baseline', dict(profile='constant', seconds=10), 3)
    row = next(r for r in changed['leg_timing'] if r['trip_id'] == 'T01:H0001' and r['leg_index'] == 0)
    row['departure_second'] -= 1
    with pytest.raises(ValueError, match='busy'):
        simulate(changed, 'line1-baseline')


def test_53_uses_explicit_terminal_hold_and_cycle_5989():
    data,meta=regular_plan_at_limit(load_config(),53,end_second=21600)
    assert meta["cycle_seconds"] == 5989
    assert meta["extra_terminal_hold_seconds"] == [24,25]
    assert headway_report(data,simulate(data,"line1-baseline"))["minimum_seconds"] == 113


@pytest.mark.parametrize('travel,cycle,holds', [(150,6300,[0,0]), (130,5989,[204,205])])
def test_cycle_is_derived_from_section_times_and_required_headway(travel,cycle,holds):
    config = load_config(); config['travel_seconds'] = travel
    data, meta = regular_plan_at_limit(config,53,end_second=21600)
    assert meta['cycle_seconds'] == cycle
    assert meta.get('extra_terminal_hold_seconds', [0,0]) == holds
    assert headway_report(data,simulate(data,'line1-baseline'))['minimum_seconds'] >= 113


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError): compile_timing(three_stations(),"line1-baseline","mild",seed)

