"""Black-box simulator audit: independent clock oracle and corrupt boundary inputs."""

import copy
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from economics import compare
from simulation import check_output, run_example, simulate, to_evaluation


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/simulation/cycle.json"
COSTS = ROOT / "config/costs/default.json"


def fixture():
    return json.loads(INPUT.read_text(encoding="utf-8"))


def initial_pair(gap):
    data = fixture()
    data["max_trains_on_line"] = 2
    data["plans"][0]["trips"] = []
    data["trains"] = [{**data["trains"][0], "train_id": f"T{i}", "state": "moving",
                       "location": "AB", "initial_movement": {"arrival_second": arrival, "passenger": True}}
                      for i, arrival in enumerate((5, 5 + gap), 1)]
    return data


@pytest.mark.parametrize("gap", [0, 112, 113])
def test_initial_moving_trains_must_respect_headway(gap):
    data = initial_pair(gap)
    if gap < 113:
        with pytest.raises(ValueError, match="headway"):
            simulate(data, "baseline")
    else:
        result = simulate(data, "baseline")
        assert result["checks"]["peak_trains_on_line"] == 2
        assert result["mileage_complete"] is False


def test_initial_movement_and_boundary_history_are_one_event():
    data = initial_pair(113)
    data["trains"] = data["trains"][:1]
    data["max_trains_on_line"] = 1
    data["previous_departures"] = [{"track_id": "AB", "departure_second": -115}]
    assert simulate(data, "baseline")["obligations_closed"] is True
    data["previous_departures"].append({"track_id": "AB", "departure_second": -114})
    with pytest.raises(ValueError, match="headway"):
        simulate(data, "baseline")


@pytest.mark.parametrize("field,value", [
    ("model_version", "old-simulator"), ("schema_version", "unknown"),
    ("evaluation_mode", "evaluated"), ("mileage_complete", "false"),
    ("obligations_closed", "true"), ("duration_seconds", 961),
])
def test_adapter_rejects_incompatible_or_inconsistent_result(field, value):
    result = simulate(fixture(), "baseline")
    result[field] = value
    with pytest.raises(ValueError):
        to_evaluation(result)


def test_adapter_does_not_hide_unfinished_movement_or_invent_car_km():
    data = fixture()
    data["period_end"] = "2026-05-01T03:00:59+03:00"
    result = simulate(data, "baseline")
    result["obligations_closed"] = True
    with pytest.raises(ValueError):
        to_evaluation(result)
    result = simulate(fixture(), "baseline")
    result["transport_work"][0]["passenger_car_km"] = "384"
    with pytest.raises(ValueError):
        to_evaluation(result)


def test_check_requires_input_hashes_and_manifest_object(tmp_path):
    output = tmp_path / "run"
    run_example(INPUT, output, COSTS)
    path = output / "build_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["inputs"] = {}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        check_output(output)
    path.write_text("[]", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "simulation", "check", "--output", str(output)],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


def test_run_does_not_overwrite_another_build_manifest(tmp_path):
    folder = tmp_path / "existing-data"
    folder.mkdir()
    original = b'{"model_version":"another-module","status":"success"}\n'
    manifest = folder / "build_manifest.json"
    manifest.write_bytes(original)
    with pytest.raises(ValueError, match="output"):
        run_example(INPUT, folder, COSTS)
    assert manifest.read_bytes() == original
    assert list(folder.iterdir()) == [manifest]


def test_submicrosecond_time_is_not_silently_truncated():
    data = fixture()
    data["period_start"] = "2026-05-01T03:00:00.0000001+03:00"
    with pytest.raises(ValueError, match="seconds"):
        simulate(data, "baseline")


def test_single_route_and_separate_trips_have_same_work():
    split = simulate(fixture(), "baseline")
    data = fixture()
    data["plans"][0]["trips"] = [{"trip_id": "cycle", "train_id": "T1", "departure_second": 0,
                                      "route": ["AB", "BC", "CB", "BA"], "passenger": True}]
    joined = simulate(data, "baseline")
    assert joined["transport_work"] == split["transport_work"]
    assert joined["final_state"] == split["final_state"]
    assert [(r["second"], r["event_type"], r["location"]) for r in joined["events"]] == [
            (r["second"], r["event_type"], r["location"]) for r in split["events"]]


def test_mixed_car_counts_and_passenger_flags_are_aggregated_once():
    data = fixture()
    data["max_trains_on_line"] = 2
    data["trains"][0]["cars"] = 3
    data["trains"].append({**data["trains"][0], "train_id": "T2", "cars": 5, "location": "C"})
    data["plans"][0]["trips"] = [
        {"trip_id": "p", "train_id": "T1", "departure_second": 0, "route": ["AB"], "passenger": True},
        {"trip_id": "n", "train_id": "T2", "departure_second": 0, "route": ["CB"], "passenger": False}]
    result = simulate(data, "baseline")
    ev = to_evaluation(result)
    assert {key: ev.quantities[key].amount for key in ("passenger_train_km", "nonpassenger_train_km",
          "passenger_car_km", "nonpassenger_car_km", "train_km", "car_km")} == {
              "passenger_train_km": Decimal(1), "nonpassenger_train_km": Decimal(2),
              "passenger_car_km": Decimal(3), "nonpassenger_car_km": Decimal(10),
              "train_km": Decimal(3), "car_km": Decimal(13)}
    assert sum(sum(row["state_seconds"].values()) for row in result["transport_work"]) == 1920


def test_zero_waits_and_closing_departure_remain_consistent():
    data = fixture()
    for station in data["stations"]:
        station.update(stop_seconds=0, turnaround_seconds=0)
    data["plans"][0]["trips"] = [{"trip_id": "to-C", "train_id": "T1", "departure_second": 0,
                                      "route": ["AB", "BC"], "passenger": False}]
    result = simulate(data, "baseline")
    assert result["transport_work"][0]["state_seconds"] == {"moving": 300, "stop": 0, "turnaround": 0, "standby": 660}
    assert [(e["second"], e["event_type"]) for e in result["events"]] == [
        (0, "departure"), (120, "arrival"), (120, "departure"), (300, "arrival")]
    data["plans"][0]["trips"][0]["departure_second"] = 960
    result = simulate(data, "baseline")
    assert result["events"] == []
    assert result["transport_work"][0]["nonpassenger_train_km"] == "0"
    assert result["final_state"][0]["ready_to_depart"] is True
    assert result["obligations_closed"] is False
    with pytest.raises(ValueError, match="closed"):
        compare(to_evaluation(result), to_evaluation(result), [])


def test_missing_context_and_capacity_are_explicit_not_passed():
    data = fixture()
    del data["previous_departures"], data["max_trains_on_line"]
    result = simulate(data, "baseline")
    checks = {r["name"]: r["status"] for r in result["checks"]["performed"]}
    assert checks["initial departure boundary context"] == "unknown"
    assert checks["line capacity"] == "not_configured"
    assert result["checks"]["supported_constraints_status"] == "partial"
    assert to_evaluation(result).constraint_status == "unknown"


def test_peak_fleet_with_simultaneous_exit_and_entry_is_order_independent():
    data = fixture()
    data["stations"].append({"station_id": "D", "kind": "depot", "stop_seconds": 0, "turnaround_seconds": 0})
    data["tracks"] += [
        {"track_id": "AD", "from_station": "A", "to_station": "D", "length_km": "1", "travel_seconds": 120},
        {"track_id": "DA", "from_station": "D", "to_station": "A", "length_km": "1", "travel_seconds": 120}]
    data["trains"].append({**data["trains"][0], "train_id": "T2", "location": "D"})
    data["plans"][0]["trips"] = [
        {"trip_id": "exit", "train_id": "T1", "departure_second": 0, "route": ["AD"], "passenger": False},
        {"trip_id": "enter", "train_id": "T2", "departure_second": 120, "route": ["DA"], "passenger": False}]
    result = simulate(data, "baseline")
    assert result["checks"]["peak_trains_on_line"] == 1
    data["trains"].reverse()
    assert simulate(data, "baseline") == result
    data["plans"][0]["trips"][1]["departure_second"] = 119
    with pytest.raises(ValueError, match="capacity"):
        simulate(data, "baseline")


def tick_oracle(departure, end, loops):
    """Independent per-second occupancy, using only the task's literal cycle."""
    counts = {"moving": 0, "stop": 0, "turnaround": 0, "standby": 0}
    spans = [(0, 120, "moving"), (120, 180, "stop"), (180, 360, "moving"),
             (360, 480, "turnaround"), (480, 660, "moving"), (660, 720, "stop"),
             (720, 840, "moving"), (840, 960, "turnaround")]
    for second in range(end):
        elapsed = second - departure
        state = "standby"
        if 0 <= elapsed < 960 * loops:
            local = elapsed % 960
            state = next(state for a, b, state in spans if a <= local < b)
        counts[state] += 1
    km = sum(distance for lap in range(loops) for at, distance in [(120, 1), (360, 2), (660, 2), (840, 1)]
             if departure + 960 * lap + at <= end)
    return counts, str(km), str(km * 8)


def test_differential_clock_oracle_over_edges_and_seeded_windows():
    rng = random.Random(310)
    cases = [(0, e, 1) for e in [1,119,120,121,179,180,181,359,360,361,479,480,481,659,660,661,719,720,721,839,840,841,959,960,961]]
    cases += [(rng.randrange(0,60), rng.randrange(1,3000), rng.randrange(1,4)) for _ in range(150)]
    for departure, end, loops in cases:
        data = fixture()
        start = datetime.fromisoformat(data["period_start"])
        data["period_end"] = (start + timedelta(seconds=end)).isoformat()
        data["plans"][0]["trips"] = [
            {"trip_id": f"lap-{i}", "train_id": "T1", "departure_second": departure + 960*i,
             "route": ["AB","BC","CB","BA"], "passenger": True} for i in range(loops)]
        expected_seconds, expected_km, expected_car_km = tick_oracle(departure, end, loops)
        out = simulate(data, "baseline")
        row = out["transport_work"][0]
        assert row["state_seconds"] == expected_seconds, (departure, end, loops)
        assert row["passenger_train_km"] == expected_km
        assert row["passenger_car_km"] == expected_car_km
        assert out["obligations_closed"] is (end >= departure + 960*loops)
        assert all(e["second"] <= end for e in out["events"])
        shuffled = copy.deepcopy(data)
        for key in ("trains","tracks","stations","plans"):
            rng.shuffle(shuffled[key])
        for plan in shuffled["plans"]:
            rng.shuffle(plan["trips"])
        with localcontext() as ctx:
            ctx.prec = 6
            assert simulate(shuffled, "baseline") == out


def test_variable_network_against_fraction_and_second_oracle():
    rng = random.Random(311)
    for case in range(85):
        data = fixture()
        cars = rng.randrange(1, 13)
        data["trains"][0]["cars"] = cars
        for station in data["stations"]:
            station["stop_seconds"] = rng.randrange(0, 31) if station["station_id"] == "B" else 0
            station["turnaround_seconds"] = 0 if station["station_id"] == "B" else rng.randrange(1, 36)
        for track in data["tracks"]:
            track["travel_seconds"] = rng.randrange(1, 51)
            track["length_km"] = f"{rng.randrange(1, 100000)}e-4"
        tracks = {t["track_id"]: t for t in data["tracks"]}
        stations = {s["station_id"]: s for s in data["stations"]}
        spans, arrivals, trips = [], [], []
        ready = 0
        for i, key in enumerate(("AB", "BC", "CB", "BA")):
            track = tracks[key]
            departure = ready + rng.randrange(0, 16)
            arrival = departure + track["travel_seconds"]
            stop = stations[track["to_station"]]
            wait_kind = "turnaround" if stop["turnaround_seconds"] else "stop"
            wait = stop["turnaround_seconds"] or stop["stop_seconds"]
            spans += [(departure, arrival, "moving"), (arrival, arrival + wait, wait_kind)]
            passenger = bool(rng.randrange(2))
            arrivals.append((arrival, Fraction(track["length_km"]), passenger))
            trips.append({"trip_id": f"leg-{i}", "train_id": "T1", "departure_second": departure,
                          "route": [key], "passenger": passenger})
            ready = arrival + wait
        end = rng.randrange(1, ready + 40)
        data["period_end"] = (datetime.fromisoformat(data["period_start"]) + timedelta(seconds=end)).isoformat()
        data["plans"][0]["trips"] = trips
        counts = dict.fromkeys(("moving","stop","turnaround","standby"), 0)
        for second in range(end):
            state = next((s for a, b, s in spans if a <= second < b), "standby")
            counts[state] += 1
        with localcontext() as ctx:
            ctx.prec = 6
            result = simulate(data, "baseline")
            evaluation = to_evaluation(result)
        row = result["transport_work"][0]
        assert row["state_seconds"] == counts, case
        assert result["obligations_closed"] is (end >= ready)
        for flag, prefix in ((True,"passenger"), (False,"nonpassenger")):
            km = sum((d for at, d, p in arrivals if p == flag and at <= end), Fraction())
            with localcontext() as ctx:
                ctx.prec = 100
                expected = Decimal(km.numerator) / Decimal(km.denominator)
                expected_car = Decimal((km*cars).numerator) / Decimal((km*cars).denominator)
            assert Decimal(row[prefix+"_train_km"]) == expected
            assert Decimal(row[prefix+"_car_km"]) == expected_car
            assert evaluation.quantities[prefix+"_car_km"].amount == expected_car


def test_corrupted_results_and_failed_publication_are_detected(tmp_path, monkeypatch):
    output = tmp_path / "run"
    run_example(INPUT, output, COSTS)
    log = output / "baseline.events.csv"
    log.write_bytes(log.read_bytes() + b'corrupt\n')
    with pytest.raises(ValueError, match="changed"):
        check_output(output)
    run_example(INPUT, output, COSTS)
    real_replace = Path.replace
    def fail_one_publication(path, target):
        if Path(target) == output / "candidate.simulation.json":
            raise OSError("injected publication failure")
        return real_replace(path, target)
    monkeypatch.setattr(Path, "replace", fail_one_publication)
    with pytest.raises(OSError, match="injected"):
        run_example(INPUT, output, COSTS)
    assert json.loads((output / "build_manifest.json").read_text())["status"] == "error"
    with pytest.raises(ValueError):
        check_output(output)


def test_arithmetic_failure_invalidates_previous_success(tmp_path):
    output = tmp_path / "run"
    run_example(INPUT, output, COSTS)
    data = fixture()
    data["tracks"][0]["length_km"] = "1e1000000"
    bad_input = tmp_path / "overflow.json"
    bad_input.write_text(json.dumps(data), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "simulation", "run", "--input", str(bad_input),
                           "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert json.loads((output / "build_manifest.json").read_text())["status"] == "error"
    with pytest.raises(ValueError):
        check_output(output)
