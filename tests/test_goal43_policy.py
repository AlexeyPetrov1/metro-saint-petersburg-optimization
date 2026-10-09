from copy import deepcopy
from datetime import datetime
import random

import pytest

from goal43_policy import (
    build_train_registry,
    depot_release_seconds,
    load_policy,
    minimum_headway,
    occupancy_exposure,
    parameter,
    validate_policy,
    validate_train_registry,
)


def interval(start, end, load, train_id="B001", car_index=0):
    return dict(train_id=train_id, car_index=car_index, start=start, end=end, load=load)


def test_profile_keeps_provenance_and_withdrawn_roundtrip():
    policy = load_policy()
    assert policy["profile_id"] == "goal4_3"
    assert policy["schema_version"] == 1
    for entry in policy["parameters"].values():
        assert set(("value", "unit", "source", "status")) <= entry.keys()
        assert entry["source"] and entry["unit"]
    assert parameter(policy, "reference_roundtrip_s") is None
    assert policy["parameters"]["reference_roundtrip_s"]["status"] == "withdrawn"
    assert parameter(policy, "turnaround_s") == 180
    assert parameter(policy, "line_train_limit") == 53
    assert parameter(policy, "fleet_by_type") == {"Baltiets": 37, "Yubileyny": 31}
    assert parameter(policy, "cars_per_train") == 8


def test_demo_platform_and_parking_resources_are_finite_explicit_assumptions():
    demo = load_policy()["demo"]
    assert demo["platform_capacity_by_station"] == {"L1_01": 2, "L1_19": 2}
    assert demo["platform_capacity_default"] == 1
    assert demo["terminal_holding_max_s"] == 480
    assert demo["terminal_parking_capacity_by_station"] == {"L1_01": 1, "L1_19": 1}
    assert demo["infrastructure_status"] == "assumed"
    assert demo["infrastructure_source"]


@pytest.mark.parametrize("at,expected", [
    ("2026-09-01T07:59:59+03:00", 150),
    ("2026-09-01T08:00:00+03:00", 113),
    ("2026-09-01T09:59:59+03:00", 113),
    ("2026-09-01T10:00:00+03:00", 150),
    ("2026-09-01T16:59:59+03:00", 150),
    ("2026-09-01T17:00:00+03:00", 113),
    ("2026-09-01T18:59:59+03:00", 113),
    ("2026-09-01T19:00:00+03:00", 150),
    ("2026-09-05T08:30:00+03:00", 150),
    ("2026-09-06T17:30:00+03:00", 150),
    ("2026-09-01T05:00:00Z", 113),
])
def test_headway_uses_moscow_weekdays_and_half_open_windows(at, expected):
    assert minimum_headway(at) == expected
    assert minimum_headway(datetime.fromisoformat(at)) == expected


def test_headway_rejects_ambiguous_naive_datetime():
    with pytest.raises(ValueError, match="timezone"):
        minimum_headway("2026-09-01T08:00:00")


@pytest.mark.parametrize("depot,expected", [("Avtovo", 720), ("Northern", 780)])
def test_depot_time_includes_all_preparation(depot, expected):
    assert depot_release_seconds(depot) == expected


@pytest.mark.parametrize("name,value", [
    ("peak_headway_min_s", 112),
    ("offpeak_headway_min_s", 112),
    ("headway_max_s", 100),
    ("exceptional_time_share_max", 1.01),
    ("hard_limit", 149),
])
def test_invalid_policy_cannot_be_used(name, value):
    policy = load_policy()
    policy["parameters"][name]["value"] = value
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_fleet_registry_has_explicit_reserves_only():
    registry = build_train_registry()
    assert len(registry) == 68
    assert len({row["train_id"] for row in registry}) == 68
    assert sum(row["type"] == "Baltiets" for row in registry) == 37
    assert sum(row["type"] == "Yubileyny" for row in registry) == 31
    assert sum(row["state"] == "on_line" for row in registry) == 53
    assert sum(row["state"] == "reserve" and row["ready_at"] is not None
               for row in registry) == 2
    assert sum(row["state"] == "unavailable" for row in registry) == 13
    assert all(row["km_since_to1"] is None for row in registry)
    assert all(row["status"] == "assumed" for row in registry)
    result = validate_train_registry(registry)
    assert result["on_line_count"] == 53


def test_stopped_train_still_occupies_line_and_54th_is_rejected():
    registry = build_train_registry()
    registry[0]["state"] = "stopped"
    assert validate_train_registry(registry)["on_line_count"] == 53
    reserve = next(row for row in registry if row["state"] == "reserve")
    reserve["state"] = "on_line"
    reserve["location"] = {"kind": "line", "line_id": "L1"}
    with pytest.raises(ValueError, match="53"):
        validate_train_registry(registry)


def test_empty_explicit_reserve_list_does_not_infer_readiness():
    registry = build_train_registry(initial_on_line={"Baltiets": 20, "Yubileyny": 20}, reserves=[])
    assert sum(row["state"] == "on_line" for row in registry) == 40
    assert not any(row["ready_at"] is not None for row in registry)


def test_active_train_cannot_also_be_registered_as_depot_reserve():
    with pytest.raises(ValueError, match="already"):
        build_train_registry(reserves=[dict(train_id="B001", depot="Avtovo",
                                           ready_at="2026-09-01T06:00:00+03:00")])


def test_registry_rejects_duplicate_ids_and_unconfirmed_ready_timestamp():
    registry = build_train_registry()
    duplicate = deepcopy(registry)
    duplicate[-1]["train_id"] = duplicate[0]["train_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_train_registry(duplicate)
    reserve = next(row for row in registry if row["state"] == "reserve")
    reserve["ready_at"] = "2026-09-01T06:00:00"
    with pytest.raises(ValueError, match="timezone"):
        validate_train_registry(registry)


@pytest.mark.parametrize("load,comfort,exceptional,hard", [
    (120, 0, 0, True), (150, 900, 0, True),
    (151, 900, 900, True), (160, 900, 900, True), (161, 900, 900, False),
])
def test_car_thresholds_are_separate(load, comfort, exceptional, hard):
    result = occupancy_exposure([interval(0, 900, load)])
    summary = result["global"]
    assert summary["comfort_car_seconds"] == comfort
    assert summary["exceptional_car_seconds"] == exceptional
    assert summary["hard_limit_ok"] is hard
    assert result["status"] == ("pass" if load <= 150 else "fail")


@pytest.mark.parametrize("exceptional_seconds,expected", [(8.91, "pass"), (9, "pass"), (9.09, "fail")])
def test_one_percent_boundary_is_time_not_departure_count(exceptional_seconds, expected):
    result = occupancy_exposure([
        interval(0, exceptional_seconds, 151),
        interval(exceptional_seconds, 900, 150),
    ])
    assert result["global"]["exceptional_time_share"] == pytest.approx(exceptional_seconds / 900)
    assert result["status"] == expected


def test_empty_idle_cannot_dilute_exceptional_share():
    result = occupancy_exposure([
        interval(0, 9, 151), interval(9, 900, 150), interval(900, 90_000, 0),
        interval(0, 90_000, 0, "Y001"),
    ])
    assert result["global"]["passenger_car_seconds"] == 900
    assert result["global"]["exceptional_time_share"] == 0.01


def test_no_passengers_is_unknown_not_success():
    for intervals in ([], [interval(0, 900, 0)]):
        result = occupancy_exposure(intervals)
        assert result["status"] == "unknown"
        assert result["admissible"] is None
        assert result["global"]["exceptional_time_share"] is None


def test_per_step_limit_cannot_be_hidden_by_whole_episode_average():
    result = occupancy_exposure([
        interval(0, 15, 151), interval(15, 1800, 150),
    ])
    assert result["global"]["exceptional_time_share"] < 0.01
    assert result["windows"][0]["exceptional_time_share"] > 0.01
    assert result["status"] == "fail"


def test_exposure_reports_local_car_and_train_overload():
    result = occupancy_exposure([
        interval(0, 9, 160), interval(9, 900, 150),
        interval(0, 900, 120, car_index=1),
        interval(0, 900, 120, "Y001"),
    ])
    assert result["global"]["passenger_car_seconds"] == 2700
    assert len(result["per_train"]) == 2
    assert len(result["per_car"]) == 3
    assert result["worst_car"]["train_id"] == "B001"
    assert result["worst_car"]["car_index"] == 0
    assert result["worst_car"]["max_load"] == 160


def test_explicit_coverage_clips_intervals_and_retains_empty_windows():
    result = occupancy_exposure([interval(0, 1000, 150)], window_start_s=900, window_end_s=2700)
    assert result["global"]["passenger_car_seconds"] == 100
    assert [(row["start"], row["end"]) for row in result["windows"]] == [(900, 1800), (1800, 2700)]
    assert result["windows"][1]["status"] == "unknown"


def test_simulator_seconds_aliases_are_supported_without_mutating_input():
    rows = [dict(train_id="B001", car_index=0, start_s=0, end_s=900, load=150)]
    original = deepcopy(rows)
    assert occupancy_exposure(rows)["global"]["passenger_car_seconds"] == 900
    assert rows == original


def test_conflicting_second_aliases_cannot_hide_exposure():
    row = dict(interval(0, 900, 151), start_s=100, end_s=900)
    with pytest.raises(ValueError, match="Conflicting"):
        occupancy_exposure([row])


def test_overlapping_intervals_of_same_car_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        occupancy_exposure([interval(0, 900, 120), interval(100, 800, 151)])


@pytest.mark.parametrize("row", [
    interval(10, 0, 120), interval(0, 10, -1), interval(0, 10, 12.5),
    interval(0, 10, 120, car_index=8), interval(0, float("inf"), 120),
])
def test_invalid_occupancy_interval_is_rejected(row):
    with pytest.raises(ValueError):
        occupancy_exposure([row])


def test_exposure_matches_an_independent_second_by_second_oracle():
    rng = random.Random(4301)
    duration = 2100
    rows = []
    for train in ("B001", "Y001"):
        for car in range(3):
            cursor = rng.randrange(0, 90)
            while cursor < duration:
                stop = min(duration, cursor + rng.randrange(35, 180))
                rows.append(interval(cursor, stop, rng.choice([0, 120, 121, 150, 151, 160]), train, car))
                cursor = stop + rng.randrange(0, 30)
    result = occupancy_exposure(rows, window_start_s=0, window_end_s=duration)

    def oracle(selected, first, last):
        counts = dict(passenger_car_seconds=0, exceptional_car_seconds=0,
                      comfort_car_seconds=0, hard_limit_car_seconds=0, max_load=0)
        for row in selected:
            for second in range(first, last):
                if row["start"] <= second < row["end"]:
                    counts["passenger_car_seconds"] += row["load"] > 0
                    counts["exceptional_car_seconds"] += row["load"] > 150
                    counts["comfort_car_seconds"] += row["load"] > 120
                    counts["hard_limit_car_seconds"] += row["load"] > 160
                    counts["max_load"] = max(counts["max_load"], row["load"])
        return counts

    def compare(actual, expected):
        for name, value in expected.items():
            assert actual[name] == value
        numerator = expected["exceptional_car_seconds"]
        denominator = expected["passenger_car_seconds"]
        assert actual["exceptional_time_share"] == pytest.approx(numerator / denominator if denominator else None)

    compare(result["global"], oracle(rows, 0, duration))
    for window in result["windows"]:
        compare(window, oracle(rows, int(window["start"]), int(window["end"])))
    for car in result["per_car"]:
        selected = [row for row in rows if row["train_id"] == car["train_id"] and row["car_index"] == car["car_index"]]
        compare(car, oracle(selected, 0, duration))
    for train in result["per_train"]:
        compare(train, oracle([row for row in rows if row["train_id"] == train["train_id"]], 0, duration))
