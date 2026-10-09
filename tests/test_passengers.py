"""Passenger oracles: hand calculations and event/window invariants."""

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from passengers import PassengerSimulation, run_example, check_output
from simulation import simulate


ROOT = Path(__file__).resolve().parents[1]


def movement():
    return {"schema_version": "1.0", "source": "educational fixture",
        "period_start": "2026-05-01T03:00:00+03:00",
        "period_end": "2026-05-01T03:10:00+03:00", "max_trains_on_line": 2,
        "previous_departures": [],
        "stations": [{"station_id": s, "kind": "line", "stop_seconds": 0,
                      "turnaround_seconds": 0} for s in ("A", "B")],
        "tracks": [{"track_id": "AB", "from_station": "A", "to_station": "B",
                    "length_km": "1", "travel_seconds": 120}],
        "trains": [{"train_id": t, "train_type": "educational", "cars": 1,
                    "location": "A", "state": "standby", "ready_after_seconds": 0}
                   for t in ("T1", "T2")],
        "plans": [{"plan_id": "service", "trips": [
            {"trip_id": "first", "train_id": "T1", "departure_second": 60,
             "route": ["AB"], "passenger": True},
            {"trip_id": "second", "train_id": "T2", "departure_second": 180,
             "route": ["AB"], "passenger": True}]},
            {"plan_id": "no_trains", "trips": []}]}


def demand():
    return {"schema_version": "1.0", "source": "educational demand, not turnstile OD",
        "arrivals": [{"passenger_id": f"P{i}", "arrival_second": 0, "route": ["AB"]}
                     for i in range(1, 6)],
        "capacities": [{"train_id": t, "max_capacity": 3, "normal_threshold": 2,
                        "source": "educational capacity and threshold"} for t in ("T1", "T2")]}


def engine(data=None, passengers=None, plan="service"):
    return PassengerSimulation(data or movement(), plan, passengers or demand())


def test_manual_five_passengers_and_overload():
    out = engine().run()
    m = out["metrics"]
    assert m["waiting_passenger_seconds"] == 540
    assert m["boarded_wait_passenger_seconds"] == 540
    assert m["mean_wait_seconds"] == "108"
    assert m["p95_wait_seconds"] == 180
    assert m["boarded_waits_seconds"] == [60, 60, 60, 180, 180]
    assert (m["entered"], m["boarded"], m["completed"], m["queued"], m["onboard"]) == (5, 5, 5, 0, 0)
    assert m["balance"]["passed"] and m["waiting_identity_passed"]
    assert m["max_queue"] == 5
    assert m["denied_boarding_attempts"] == m["unique_denied_passengers"] == 2
    assert [(r["train_id"], r["load"], r["normal_threshold_exceeded"])
            for r in out["load_events"]] == [("T1", 3, True), ("T2", 2, False)]
    assert out["release_authorized"] is False


def test_no_trains_keeps_queue_and_censored_wait():
    out = engine(plan="no_trains").run()
    m = out["metrics"]
    assert (m["queued"], m["waiting_passenger_seconds"], m["residual_wait_passenger_seconds"]) == (5, 3000, 3000)
    assert m["mean_wait_seconds"] is None and m["p95_wait_seconds"] is None
    assert m["completed"] == m["boarded"] == m["onboard"] == 0
    assert m["balance"]["passed"] and not out["passenger_obligations_closed"]
    assert [r["waiting_age_seconds"] for r in out["queues"]] == [600] * 5


@pytest.mark.parametrize("cuts", [[60, 180, 600], [1, 59, 60, 61, 179, 180, 181, 300, 600], list(range(1, 601))])
def test_continuous_equals_json_checkpoint_windows(cuts):
    runner = engine()
    expected, state, start, deltas = runner.run(), None, 0, []
    for end in cuts:
        part = runner.run(start_second=start, end_second=end, checkpoint=state)
        deltas.append(part["window_waiting_passenger_seconds"])
        state = json.loads(json.dumps(part["checkpoint"]))
        start = end
    assert part["metrics"] == expected["metrics"]
    assert part["checkpoint"] == expected["checkpoint"]
    assert part["passenger_events"] == expected["passenger_events"]
    assert part["load_events"] == expected["load_events"]
    assert sum(deltas) == 540


def test_queue_age_survives_idle_windows():
    runner = engine(plan="no_trains")
    first = runner.run(end_second=240)
    last = runner.run(start_second=240, checkpoint=first["checkpoint"])
    assert first["metrics"]["waiting_passenger_seconds"] == 1200
    assert last["window_waiting_passenger_seconds"] == 1800
    assert last["metrics"]["waiting_passenger_seconds"] == 3000
    assert last["queues"][0]["arrival_second"] == 0
    assert last["queues"][0]["waiting_age_seconds"] == 600
    assert last["checkpoint"] == runner.run()["checkpoint"]


def test_arrival_at_departure_boards_once_and_window_border_is_deferred():
    d = demand()
    d["arrivals"] = [{"passenger_id": "P", "arrival_second": 60, "route": ["AB"]}]
    runner = engine(passengers=d)
    left = runner.run(end_second=60)
    assert left["metrics"]["entered"] == left["metrics"]["boarded"] == 0
    right = runner.run(start_second=60, checkpoint=left["checkpoint"])
    assert right["metrics"]["boarded_waits_seconds"] == [0]
    assert len([r for r in right["passenger_events"] if r["event_type"] == "board"]) == 1
    assert right["checkpoint"] == runner.run()["checkpoint"]


def test_closing_arrival_processed_once_but_demand_at_end_is_outside():
    data, d = movement(), demand()
    data["period_end"] = "2026-05-01T03:05:00+03:00"
    d["arrivals"].append({"passenger_id": "late", "arrival_second": 300, "route": ["AB"]})
    out = engine(data, d).run()
    assert out["metrics"]["completed"] == 5
    assert out["metrics"]["entered"] == 5
    assert out["metrics"]["future_arrivals"] == 1
    assert out["passenger_events"][-1]["second"] == 300


def test_mid_journey_preserves_onboard_passengers():
    runner = engine()
    left = runner.run(end_second=100)
    assert (left["metrics"]["queued"], left["metrics"]["onboard"], left["metrics"]["completed"]) == (2, 3, 0)
    assert left["metrics"]["waiting_passenger_seconds"] == 380
    right = runner.run(start_second=100, checkpoint=left["checkpoint"])
    assert right["checkpoint"] == runner.run()["checkpoint"]


def test_passenger_load_does_not_change_motion_or_transport_work():
    data, d = movement(), demand()
    before_data, before_d = copy.deepcopy(data), copy.deepcopy(d)
    loaded = engine(data, d)
    empty = copy.deepcopy(d)
    empty["arrivals"] = []
    unloaded = engine(data, empty)
    assert loaded.motion == unloaded.motion == simulate(data, "service")
    loaded.run()
    unloaded.run()
    assert loaded.motion["transport_work"] == unloaded.motion["transport_work"]
    assert data == before_data and d == before_d


def test_nonpassenger_trip_does_not_board():
    data = movement()
    for trip in data["plans"][0]["trips"]:
        trip["passenger"] = False
    out = engine(data).run()
    assert out["metrics"]["queued"] == 5
    assert out["metrics"]["waiting_passenger_seconds"] == 3000


def test_fifo_is_stable_and_wrong_destination_does_not_block_eligible_queue():
    data, d = movement(), demand()
    data["stations"].append({"station_id": "C", "kind": "line", "stop_seconds": 0, "turnaround_seconds": 0})
    data["tracks"].append({"track_id": "BC", "from_station": "B", "to_station": "C", "length_km": "2", "travel_seconds": 120})
    d["arrivals"][0]["route"] = ["AB", "BC"]
    out = engine(data, d).run()
    assert [r["passenger_id"] for r in out["passenger_events"] if r["event_type"] == "board"] == ["P2", "P3", "P4", "P5"]
    assert out["queues"][0]["passenger_id"] == "P1"
    assert out["metrics"]["residual_wait_passenger_seconds"] == 600
    d["arrivals"].reverse()
    d["capacities"].reverse()
    assert engine(data, d).run()["checkpoint"] == out["checkpoint"]


def test_through_passenger_and_intermediate_boarding_use_motion_events():
    data, d = movement(), demand()
    data["stations"][1]["stop_seconds"] = 30
    data["stations"].append({"station_id": "C", "kind": "line", "stop_seconds": 0, "turnaround_seconds": 0})
    data["tracks"].append({"track_id": "BC", "from_station": "B", "to_station": "C", "length_km": "2", "travel_seconds": 120})
    data["plans"][0]["trips"][0]["route"] = ["AB", "BC"]
    d["arrivals"] = [{"passenger_id": "through", "arrival_second": 0, "route": ["AB", "BC"]},
                     {"passenger_id": "local", "arrival_second": 180, "route": ["BC"]}]
    out = engine(data, d).run()
    assert out["metrics"]["boarded_waits_seconds"] == [30, 60]
    assert out["metrics"]["completed"] == 2
    assert [r["second"] for r in out["passenger_events"] if r["event_type"] == "alight"] == [330, 330]
    motion_ids = {e["event_id"] for e in engine(data, d).motion["events"]}
    assert all(r["motion_event_id"] in motion_ids for r in out["passenger_events"] if r["event_type"] != "enter")


def test_alighting_releases_capacity_before_same_time_boarding():
    data, d = movement(), demand()
    data["stations"].append({"station_id": "C", "kind": "line", "stop_seconds": 0, "turnaround_seconds": 0})
    data["tracks"].append({"track_id": "BC", "from_station": "B", "to_station": "C", "length_km": "2", "travel_seconds": 120})
    data["plans"][0]["trips"][0]["route"] = ["AB", "BC"]
    d["arrivals"] = [{"passenger_id": "first", "arrival_second": 0, "route": ["AB"]},
                     {"passenger_id": "next", "arrival_second": 180, "route": ["BC"]}]
    d["capacities"][0].update(max_capacity=1, normal_threshold=1)
    out = engine(data, d).run()
    assert out["metrics"]["completed"] == 2
    assert out["metrics"]["denied_boarding_attempts"] == 0
    assert all(r["load"] <= r["max_capacity"] for r in out["load_events"])


@pytest.mark.parametrize("change", ["negative_time", "bool_time", "fraction_time", "duplicate_person", "unknown_track", "broken_route", "empty_route", "missing_capacity", "duplicate_capacity", "bool_capacity", "zero_capacity", "threshold_above_capacity", "unknown_train", "missing_source", "unknown_field"])
def test_bad_passenger_input_rejected(change):
    data, d = movement(), demand()
    if change == "negative_time": d["arrivals"][0]["arrival_second"] = -1
    if change == "bool_time": d["arrivals"][0]["arrival_second"] = True
    if change == "fraction_time": d["arrivals"][0]["arrival_second"] = 0.5
    if change == "duplicate_person": d["arrivals"].append(copy.deepcopy(d["arrivals"][0]))
    if change == "unknown_track": d["arrivals"][0]["route"] = ["missing"]
    if change == "broken_route": d["arrivals"][0]["route"] = ["AB", "AB"]
    if change == "empty_route": d["arrivals"][0]["route"] = []
    if change == "missing_capacity": d["capacities"].pop()
    if change == "duplicate_capacity": d["capacities"].append(copy.deepcopy(d["capacities"][0]))
    if change == "bool_capacity": d["capacities"][0]["max_capacity"] = True
    if change == "zero_capacity": d["capacities"][0]["max_capacity"] = 0
    if change == "threshold_above_capacity": d["capacities"][0]["normal_threshold"] = 4
    if change == "unknown_train": d["capacities"][0]["train_id"] = "missing"
    if change == "missing_source": d.pop("source")
    if change == "unknown_field": d["invented"] = 1
    with pytest.raises(ValueError): engine(data, d)


@pytest.mark.parametrize("change", ["gap", "overlap", "replay", "wrong_demand", "wrong_plan", "tamper", "empty_checkpoint", "bool_end", "past_end", "zero_window"])
def test_invalid_window_or_checkpoint_rejected(change):
    runner = engine()
    cp = runner.run(end_second=60)["checkpoint"]
    start, end = 60, 600
    if change == "gap": start = 61
    if change == "overlap": start = 59
    if change == "replay": start, end = 0, 60
    if change == "wrong_demand":
        d = demand(); d["arrivals"][0]["arrival_second"] = 1; runner = engine(passengers=d)
    if change == "wrong_plan": runner = engine(plan="no_trains")
    if change == "tamper": cp["waiting_passenger_seconds"] += 1
    if change == "empty_checkpoint": cp = {}
    if change == "bool_end": end = True
    if change == "past_end": end = 601
    if change == "zero_window": end = 60
    with pytest.raises(ValueError): runner.run(start_second=start, end_second=end, checkpoint=cp)


def test_nonzero_start_requires_checkpoint():
    with pytest.raises(ValueError): engine().run(start_second=60)


def test_wait_quantile_nearest_rank_and_pending_queue_excluded_from_boarded_mean():
    d = demand()
    d["arrivals"][4]["arrival_second"] = 200
    out = engine(passengers=d).run()
    assert out["metrics"]["boarded_waits_seconds"] == [60, 60, 60, 180]
    assert out["metrics"]["mean_wait_seconds"] == "90"
    assert out["metrics"]["p95_wait_seconds"] == 180
    assert out["metrics"]["residual_wait_passenger_seconds"] == 400
    assert out["metrics"]["waiting_passenger_seconds"] == 760


def test_seeded_arrivals_match_independent_fifo_oracle():
    rng = random.Random(3208)
    for _ in range(30):
        d = demand()
        d["arrivals"] = [{"passenger_id": f"P{i:03}", "arrival_second": rng.randrange(601), "route": ["AB"]}
                         for i in range(rng.randrange(21))]
        remaining = sorted((r["arrival_second"], r["passenger_id"]) for r in d["arrivals"] if r["arrival_second"] < 600)
        waits = []
        for departure in (60, 180):
            eligible = [r for r in remaining if r[0] <= departure][:3]
            waits += [departure - r[0] for r in eligible]
            remaining = [r for r in remaining if r not in eligible]
        out = engine(passengers=d).run()
        assert out["metrics"]["boarded_waits_seconds"] == sorted(waits)
        assert out["metrics"]["queued"] == len(remaining)
        assert out["metrics"]["waiting_passenger_seconds"] == sum(waits) + sum(600 - r[0] for r in remaining)
        assert out["metrics"]["balance"]["passed"]


def test_fixture_artifacts_reproduce_and_check(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for path in (a, b): run_example(ROOT / "examples/passengers/movement.json", ROOT / "examples/passengers/demand.json", path)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {p.name: p.read_bytes() for p in b.iterdir()}
    assert check_output(a)["status"] == "passed"
    (a / "control_check.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): check_output(a)


def test_cli_unknown_option_returns_clean_error(tmp_path):
    done = subprocess.run([sys.executable, "-m", "passengers", "run", "--movement", str(tmp_path / "missing.json"),
                           "--output", str(tmp_path / "out")], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 2
    assert "Traceback" not in done.stderr


def test_foreign_output_not_overwritten(tmp_path):
    (tmp_path / "keep.txt").write_text("untouched", encoding="utf-8")
    with pytest.raises(ValueError): run_example(ROOT / "examples/passengers/movement.json", ROOT / "examples/passengers/demand.json", tmp_path)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("kind", ["array", "missing_fields", "invalid_cursor", "invalid_person", "inconsistent_wait"])
def test_resealed_malformed_checkpoint_is_rejected(kind):
    from simulation import digest
    runner = engine()
    cp = runner.run(end_second=60)["checkpoint"]
    if kind == "array": cp["passengers"] = {}
    if kind == "missing_fields": cp.pop("load_events")
    if kind == "invalid_cursor": cp["next_event_index"] = 999
    if kind == "invalid_person": cp["passengers"][0]["state"] = "vanished"
    if kind == "inconsistent_wait": cp["waiting_passenger_seconds"] = 0
    cp["sha256"] = digest({k: v for k, v in cp.items() if k != "sha256"})
    with pytest.raises(ValueError): runner.run(start_second=60, checkpoint=cp)


def test_checkpoint_can_resume_in_new_process_instance_and_is_not_mutated():
    cp = engine().run(end_second=100)["checkpoint"]
    before = copy.deepcopy(cp)
    last = engine().run(start_second=100, checkpoint=cp)
    assert cp == before
    assert last["checkpoint"] == engine().run()["checkpoint"]


def test_checkpoint_bound_to_train_capacity():
    cp = engine().run(end_second=60)["checkpoint"]
    d = demand()
    d["capacities"][0]["max_capacity"] = 4
    with pytest.raises(ValueError): engine(passengers=d).run(start_second=60, checkpoint=cp)


def test_repeated_boarding_denials_count_attempts_without_losing_people():
    d = demand()
    d["arrivals"] += [{"passenger_id": "P6", "arrival_second": 0, "route": ["AB"]},
                      {"passenger_id": "P7", "arrival_second": 0, "route": ["AB"]}]
    out = engine(passengers=d).run()
    assert out["metrics"]["denied_boarding_attempts"] == 5  # 4 + 1
    assert out["metrics"]["unique_denied_passengers"] == 4
    assert out["metrics"]["queued"] == 1 and out["queues"][0]["passenger_id"] == "P7"
    assert out["metrics"]["waiting_passenger_seconds"] == 1320  # 3*60 + 3*180 + 1*600


def test_separate_directions_do_not_board_wrong_train():
    data, d = movement(), demand()
    data["tracks"].append({"track_id": "BA", "from_station": "B", "to_station": "A", "length_km": "1", "travel_seconds": 120})
    data["trains"][1]["location"] = "B"
    data["plans"][0]["trips"][1]["route"] = ["BA"]
    d["arrivals"][4]["route"] = ["BA"]
    out = engine(data, d).run()
    assert (out["metrics"]["completed"], out["metrics"]["queued"]) == (4, 1)
    assert out["metrics"]["max_queue_by_station"] == {"A": 4, "B": 1}
    assert out["metrics"]["waiting_passenger_seconds"] == 960  # 3*60 + 1*180 + 1*600


def test_train_at_closing_departure_does_not_board():
    data = movement()
    data["plans"][0]["trips"][1]["departure_second"] = 600
    out = engine(data).run()
    assert out["metrics"]["queued"] == 2
    assert out["metrics"]["waiting_passenger_seconds"] == 1380  # 3*60 + 2*600
    assert len(out["load_events"]) == 1


def test_nearest_rank_95_does_not_use_max_for_large_sample():
    data, d = movement(), demand()
    d["capacities"][0]["max_capacity"] = 100
    d["arrivals"] = [{"passenger_id": str(i), "arrival_second": i, "route": ["AB"]} for i in range(60)]
    out = engine(data, d).run()
    assert out["metrics"]["boarded_waits_seconds"] == list(range(1, 61))
    assert out["metrics"]["p95_wait_seconds"] == 57


@pytest.mark.parametrize("invalid", [[], {}, {"inputs": {}, "outputs": {}}])
def test_bad_output_manifest_is_cleanly_rejected(tmp_path, invalid):
    (tmp_path / "build_manifest.json").write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError): check_output(tmp_path)


def test_failed_rerun_invalidates_old_success(tmp_path):
    network, passengers = tmp_path / "motion.json", tmp_path / "demand.json"
    network.write_text(json.dumps(movement()), encoding="utf-8")
    passengers.write_text(json.dumps(demand()), encoding="utf-8")
    output = tmp_path / "output"
    run_example(network, passengers, output)
    passengers.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError): run_example(network, passengers, output)
    assert json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))["status"] == "error"
    with pytest.raises(ValueError): check_output(output)


def test_output_cannot_overwrite_movement_artifacts():
    with pytest.raises(ValueError): run_example(ROOT / "examples/passengers/movement.json", ROOT / "examples/passengers/demand.json", ROOT / "data/processed/simulation")


def test_existing_audited_cycle_supports_direct_rides_without_motion_changes():
    data = json.loads((ROOT / "examples/simulation/cycle.json").read_text(encoding="utf-8"))
    d = {"schema_version": "1.0", "source": "educational rides on GOAL 3.1 cycle",
         "capacities": [{"train_id": "T1", "max_capacity": 3, "normal_threshold": 2, "source": "educational"}],
         "arrivals": [{"passenger_id": "out", "arrival_second": 0, "route": ["AB", "BC"]},
                      {"passenger_id": "back", "arrival_second": 400, "route": ["CB", "BA"]}]}
    runner = PassengerSimulation(data, "baseline", d)
    out = runner.run()
    assert out["metrics"]["boarded_waits_seconds"] == [0, 80]
    assert out["metrics"]["completed"] == 2
    assert runner.motion == simulate(data, "baseline")
    assert out["transport_work"][0]["passenger_train_km"] == "6"
    assert out["transport_work"][0]["passenger_car_km"] == "48"
