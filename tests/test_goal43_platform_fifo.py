import copy

from dwell import compile_operational_timing
from goal43_controller import observed_planning_data
from goal43_policy import load_policy
from simulation import simulate


def held_followers():
    data = dict(schema_version="1.0", source="observed approach FIFO fixture",
        period_start="2026-09-01T08:00:00+03:00", period_end="2026-09-01T09:00:00+03:00",
        stations=[dict(station_id=s, kind="line", stop_seconds=20 if s == "B" else 0,
            turnaround_seconds=180 if s != "B" else 0) for s in "ABC"] +
            [dict(station_id="D", kind="depot", stop_seconds=0, turnaround_seconds=0)],
        tracks=[dict(track_id=f"{a}{b}_up", from_station=a, to_station=b, length_km="1", travel_seconds=100)
            for a,b in ("AB", "BC")] +
            [dict(track_id="EXIT_C", from_station="C", to_station="D", length_km="1", travel_seconds=20)],
        trains=[dict(train_id=t, train_type="fixture", cars=8, state="standby", location="A", ready_after_seconds=0)
            for t in ("L", "Z", "A")], previous_departures=[], max_trains_on_line=53)
    trips = []
    for index, train in enumerate(("L", "Z", "A")):
        trips += [dict(trip_id=f"{train}:SERVICE", train_id=train, departure_second=113*index,
            route=["AB_up", "BC_up"], passenger=True),
            dict(trip_id=f"{train}:EXIT", train_id=train, departure_second=500+113*index,
                route=["EXIT_C"], passenger=False)]
    data["plans"] = [dict(plan_id="P", trips=trips)]
    return data


def test_observed_approach_arrival_ties_preserve_same_track_departure_fifo():
    policy = load_policy()
    baseline, _ = compile_operational_timing(held_followers(), "P", policy=policy)
    incident = dict(incident_id="I", train_id="L", station_id="B", direction="up",
        detected_second=100, expected_end_second=1000)
    stopped, _ = compile_operational_timing(baseline, "P", policy=policy, incidents=[incident], as_of=100)
    observed = observed_planning_data(stopped, stopped, 500)
    approach = {r["train_id"]: r for r in observed["leg_timing"] if r["trip_id"].endswith(":SERVICE") and r["leg_index"] == 0}
    assert [approach[t]["departure_second"] for t in ("Z", "A")] == [113, 226]
    assert {r["departure_second"] + 100 + r["approach_holding_seconds"] for t,r in approach.items() if t != "L"} == {500}
    original = copy.deepcopy(observed)
    rebuilt, audit = compile_operational_timing(observed, "P", policy=policy, incidents=[incident], as_of=500)
    before, after = simulate(stopped, "P"), simulate(rebuilt, "P")
    assert [e for e in after["events"] if e["second"] < 500] == [e for e in before["events"] if e["second"] < 500]
    visits = sorted((r for r in audit["platform_occupancy"] if r["platform"] == ["B", "up"]), key=lambda r:r["arrival_second"])
    assert [r["train_id"] for r in visits] == ["L", "Z", "A"]
    assert observed == original


def test_two_platform_slots_keep_observed_incoming_track_fifo():
    policy, data = load_policy(), held_followers()
    policy["demo"]["platform_capacity_by_station"]["B"] = 2
    names = ("L", "M", "Z", "A")
    data["trains"] = [dict(data["trains"][0], train_id=train) for train in names]
    data["plans"][0]["trips"] = [dict(trip_id=f"{train}:{kind}", train_id=train,
        departure_second=(113*index if kind == "SERVICE" else 500+113*index),
        route=(["AB_up", "BC_up"] if kind == "SERVICE" else ["EXIT_C"]), passenger=kind == "SERVICE")
        for index,train in enumerate(names) for kind in ("SERVICE", "EXIT")]
    incidents = [dict(incident_id=f"I{index}", train_id=train, station_id="B", direction="up",
        detected_second=100+113*index, expected_end_second=1000) for index,train in enumerate(("L", "M"))]
    baseline, _ = compile_operational_timing(data, "P", policy=policy, incidents=incidents)
    observed = observed_planning_data(baseline, baseline, 500)
    rebuilt, audit = compile_operational_timing(observed, "P", policy=policy, incidents=incidents, as_of=500)
    arrivals = sorted((r for r in audit["platform_occupancy"] if r["platform"] == ["B", "up"]), key=lambda r:r["arrival_second"])
    assert [r["train_id"] for r in arrivals] == list(names)
    assert [e for e in simulate(rebuilt, "P")["events"] if e["second"] < 500] == [e for e in simulate(baseline, "P")["events"] if e["second"] < 500]
