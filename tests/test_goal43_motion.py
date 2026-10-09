import copy

import pytest

from dwell import compile_operational_timing
from goal43_policy import load_policy
from passenger_groups import GroupedPassengerSimulation
from simulation import simulate


def line(start="2026-09-01T08:00:00+03:00", spacing=113, *, clearance=True):
    data = {
        "schema_version": "1.0", "source": "explicit platform fixture",
        "period_start": start, "period_end": start[:11] + "09:00:00+03:00",
        "stations": [{"station_id": s, "kind": "line", "stop_seconds": 20 if s == "B" else 0,
                      "turnaround_seconds": 180 if s != "B" else 0} for s in "ABC"],
        "tracks": [{"track_id": f"{a}{b}_up", "from_station": a, "to_station": b,
                    "length_km": "1", "travel_seconds": 140} for a, b in ("AB", "BC")],
        "trains": [{"train_id": f"T{i}", "train_type": "Юбилейный", "cars": 8,
                    "state": "standby", "location": "A", "ready_after_seconds": 0} for i in (1, 2)],
        "plans": [{"plan_id": "P", "trips": [{"trip_id": f"H{i}", "train_id": f"T{i}",
                   "departure_second": (i - 1) * spacing, "route": ["AB_up", "BC_up"],
                   "passenger": True} for i in (1, 2)]}],
        "previous_departures": [], "max_trains_on_line": 53,
    }
    return add_exit(data) if clearance else data


def incident():
    return dict(incident_id="I", train_id="T1", station_id="B", direction="up",
                detected_second=140, expected_end_second=440)


def test_stop_blocks_platform_and_preserves_departed_approach():
    original = line()
    baseline, _ = compile_operational_timing(original, "P", policy=load_policy())
    stopped, audit = compile_operational_timing(baseline, "P", policy=load_policy(),
                                               incidents=[incident()], as_of=140)
    events = simulate(stopped, "P")["events"]
    first_exit = next(e["second"] for e in events if e["train_id"] == "T1" and e["event_type"] == "departure" and e["location"] == "BC_up")
    follower_arrival = next(e["second"] for e in events if e["train_id"] == "T2" and e["event_type"] == "arrival" and e["location"] == "B")
    assert first_exit >= 440
    assert follower_arrival > first_exit
    assert next(e["second"] for e in events if e["train_id"] == "T2" and e["event_type"] == "departure") == 113
    assert [e for e in events if e["second"] < 140] == [e for e in simulate(baseline, "P")["events"] if e["second"] < 140]
    assert audit["approach_holding_seconds"] > 0
    assert original == line()


def test_current_passenger_state_branches_without_reset():
    baseline, _ = compile_operational_timing(line(), "P", policy=load_policy())
    cars = [{"car_id": f"C{i}", "car_type": "demo", "max_capacity": 160,
             "normal_threshold": 120, "source": "user GOAL4.3"} for i in range(8)]
    profiles = {t["train_id"]: copy.deepcopy(cars) for t in baseline["trains"]}
    groups = [dict(group_id="G", turnstile_second=0, platform_second=0, origin="A", count=16,
                   destinations=[dict(station_id="C", count=16)])]
    source = GroupedPassengerSimulation(baseline, "P", groups, profiles)
    checkpoint = source.run(end_second=140)["checkpoint"]
    stopped, _ = compile_operational_timing(baseline, "P", policy=load_policy(), incidents=[incident()], as_of=140)
    branch = GroupedPassengerSimulation(stopped, "P", groups, profiles)
    state = branch.branch_checkpoint(checkpoint, source)
    assert state["onboard"] == checkpoint["onboard"]
    assert state["queues"] == checkpoint["queues"]
    continuous = branch.run()["checkpoint"]
    resumed = branch.run(start_second=140, checkpoint=state)["checkpoint"]
    assert continuous == resumed


def test_offpeak_headway_is_enforced_in_shared_movement_validator():
    data = line("2026-09-01T06:00:00+03:00")
    data["headway_policy"] = load_policy()
    with pytest.raises(ValueError, match="headway"):
        simulate(data, "P")
    compiled, _ = compile_operational_timing(data, "P", policy=load_policy())
    events = simulate(compiled, "P")["events"]
    assert next(e["second"] for e in events if e["train_id"] == "T2" and e["event_type"] == "departure") >= 150


def test_directional_stop_does_not_block_opposite_platform():
    data = line()
    data["tracks"].append(dict(track_id="CB_down", from_station="C", to_station="B", length_km="1", travel_seconds=140))
    data["trains"].append(dict(train_id="T3", train_type="Юбилейный", cars=8, state="standby", location="C", ready_after_seconds=0))
    data["plans"][0]["trips"].append(dict(trip_id="H3", train_id="T3", departure_second=100, route=["CB_down"], passenger=True))
    changed, _ = compile_operational_timing(data, "P", policy=load_policy(), incidents=[incident()])
    events = simulate(changed, "P")["events"]
    assert next(e["second"] for e in events if e["train_id"] == "T3" and e["event_type"] == "arrival") == 240


def test_arrival_approach_override_never_changes_track_length():
    data = line()
    data["leg_timing"] = [dict(train_id="T1", trip_id="H1", leg_index=0,
        departure_second=0, service_seconds=20, approach_holding_seconds=200)]
    data["plans"][0]["trips"] = [data["plans"][0]["trips"][0]]
    result = simulate(data, "P")
    assert next(e["second"] for e in result["events"] if e["event_type"] == "arrival") == 340
    assert result["transport_work"][0]["train_km"] == "2"
    data["leg_timing"][0]["approach_holding_seconds"] = -1
    with pytest.raises(ValueError):
        simulate(data, "P")


def test_boundary_history_uses_headway_after_hold_crosses_peak_end():
    data = line("2026-09-01T09:59:00+03:00")
    data["period_end"] = "2026-09-01T10:59:00+03:00"
    data["trains"] = data["trains"][:1]
    data["plans"][0]["trips"] = [dict(data["plans"][0]["trips"][0], departure_second=50)]
    data["previous_departures"] = [dict(track_id="AB_up", departure_second=-63)]
    changed, _ = compile_operational_timing(data, "P", policy=load_policy(),
        departure_not_before={("T1", "H1", 0): 60})
    assert next(e["second"] for e in simulate(changed, "P")["events"] if e["event_type"] == "departure") == 87


def add_exit(data, train="T1", departure=480):
    if not any(s["station_id"] == "DEPOT_C" for s in data["stations"]):
        data["stations"].append(dict(station_id="DEPOT_C", kind="depot", stop_seconds=0, turnaround_seconds=0))
        data["tracks"].append(dict(track_id="EXIT_C", from_station="C", to_station="DEPOT_C",
                                   length_km="1", travel_seconds=20))
    data["plans"][0]["trips"].append(dict(trip_id=train+":DAYEXIT", train_id=train,
        departure_second=departure, route=["EXIT_C"], passenger=False))
    return data


def initially_stopped(state):
    data = line(clearance=False)
    data["tracks"][0]["travel_seconds"] = 10
    if state == "stop":
        data["trains"][0].update(state=state, location="B", ready_after_seconds=20)
        data["plans"][0]["trips"][0].update(departure_second=20, route=["BC_up"])
        add_exit(data)
    else:
        data["tracks"][1]["travel_seconds"] = 10
        data["trains"][0].update(state=state, location="C", ready_after_seconds=180)
        data["plans"][0]["trips"] = [data["plans"][0]["trips"][1]]
        add_exit(data, departure=180)
    data["plans"][0]["trips"][1 if state == "stop" else 0]["departure_second"] = 0
    return data


@pytest.mark.parametrize("state,station,release", [("stop", "B", 20), ("turnaround", "C", 180)])
def test_initial_platform_occupancy_blocks_follower_until_actual_departure(state, station, release):
    data = initially_stopped(state)
    changed, audit = compile_operational_timing(data, "P", policy=load_policy())
    events = simulate(changed, "P")["events"]
    arrival = next(e["second"] for e in events if e["train_id"] == "T2" and
                   e["event_type"] == "arrival" and e["location"] == station)
    assert arrival >= release+1
    assert audit["initial_platform_occupancy"][0]["train_id"] == "T1"
    assert changed["trains"] == data["trains"]


@pytest.mark.parametrize("state,station", [("stop", "B"), ("turnaround", "C")])
def test_initial_platform_incident_preserves_departed_history_and_delays_release(state, station):
    data, policy = initially_stopped(state), load_policy()
    baseline, _ = compile_operational_timing(data, "P", policy=policy)
    event = dict(incident_id="INITIAL", train_id="T1", station_id=station, direction="up",
                 detected_second=10, expected_end_second=600)
    changed, audit = compile_operational_timing(baseline, "P", policy=policy, incidents=[event], as_of=10)
    before, after = simulate(baseline, "P"), simulate(changed, "P")
    assert [e for e in after["events"] if e["second"] < 10] == [e for e in before["events"] if e["second"] < 10]
    departure = next(e["second"] for e in after["events"] if e["train_id"] == "T1" and e["event_type"] == "departure")
    assert departure >= 600
    assert audit["incidents"][0]["used_end_second"] == 600
    assert changed["trains"] == baseline["trains"]


def test_final_platform_slot_without_actual_departure_is_not_cleared_by_turnaround():
    with pytest.raises(ValueError, match="platform.*clearance|platform.*release"):
        compile_operational_timing(line(clearance=False), "P", policy=load_policy())


def test_final_platform_slot_is_cleared_by_real_depot_exit_only():
    changed, audit = compile_operational_timing(line(), "P", policy=load_policy())
    events = simulate(changed, "P")["events"]
    release = next(e["second"] for e in events if e["train_id"] == "T1" and e["event_type"] == "departure" and e["location"] == "EXIT_C")
    follower = next(e["second"] for e in events if e["train_id"] == "T2" and e["event_type"] == "arrival" and e["location"] == "C")
    assert follower > release
    assert audit["terminal_clearance_assumptions"] == []


def test_initial_platform_without_outgoing_movement_keeps_its_slot():
    data = initially_stopped("stop")
    data["plans"][0]["trips"] = [trip for trip in data["plans"][0]["trips"] if trip["train_id"] != "T1"]
    with pytest.raises(ValueError, match="platform.*clearance|platform.*release"):
        compile_operational_timing(data, "P", policy=load_policy())


def test_permanent_platform_occupant_does_not_block_another_released_slot():
    data, policy = line(clearance=False), load_policy()
    data["trains"].append(dict(data["trains"][1], train_id="T3"))
    data["plans"][0]["trips"].append(dict(data["plans"][0]["trips"][1], train_id="T3", trip_id="H3", departure_second=226))
    add_exit(data, train="T2", departure=593)
    policy["demo"]["platform_capacity_by_station"]["C"] = 2
    changed, audit = compile_operational_timing(data, "P", policy=policy)
    events = simulate(changed, "P")["events"]
    release = next(e["second"] for e in events if e["train_id"] == "T2" and e["event_type"] == "departure" and e["location"] == "EXIT_C")
    arrival = next(e["second"] for e in events if e["train_id"] == "T3" and e["event_type"] == "arrival" and e["location"] == "C")
    assert arrival > release
    assert audit["terminal_clearance_assumptions"] == []


def test_unknown_incident_protection_moves_forward_at_each_decision():
    data, policy = initially_stopped("stop"), load_policy()
    data["plans"][0]["trips"][0]["departure_second"] = 1000
    data["plans"][0]["trips"][1]["departure_second"] = 1000
    data["plans"][0]["trips"][-1]["departure_second"] = 1320
    baseline, _ = compile_operational_timing(data, "P", policy=policy)
    event = dict(incident_id="UNKNOWN", train_id="T1", station_id="B", direction="up",
                 detected_second=10, expected_end_second=None)
    changed, audit = compile_operational_timing(baseline, "P", policy=policy, incidents=[event], as_of=900)
    release = next(e["second"] for e in simulate(changed, "P")["events"]
                   if e["train_id"] == "T1" and e["event_type"] == "departure")
    assert release >= 900+480
    assert audit["incidents"][0]["used_end_second"] == 1380
    assert audit["incidents"][0]["expected_end_second"] is None
