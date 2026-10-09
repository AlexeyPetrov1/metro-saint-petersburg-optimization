"""Manual oracles from GOAL 3.1, independent of the simulator's calculations."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from economics import Article, compare, evaluation_from_dict, load_costs
from simulation import simulate, to_evaluation, run_example


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/simulation/cycle.json"
EXPECTED_EVENTS = [
    (0, "departure", "AB", "standby", "moving"),
    (120, "arrival", "B", "moving", "stop"),
    (180, "stop_complete", "B", "stop", "standby"),
    (180, "departure", "BC", "standby", "moving"),
    (360, "arrival", "C", "moving", "turnaround"),
    (480, "turnaround_complete", "C", "turnaround", "standby"),
    (480, "departure", "CB", "standby", "moving"),
    (660, "arrival", "B", "moving", "stop"),
    (720, "stop_complete", "B", "stop", "standby"),
    (720, "departure", "BA", "standby", "moving"),
    (840, "arrival", "A", "moving", "turnaround"),
    (960, "turnaround_complete", "A", "turnaround", "standby"),
]


def example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def one_leg(data, passenger=True):
    data["plans"][0]["trips"] = [{"trip_id": "one", "train_id": "T1",
        "departure_second": 0, "route": ["AB"], "passenger": passenger}]
    return data


def test_manual_cycle_and_ledger():
    out = simulate(example(), "baseline")
    assert [(e["second"], e["event_type"], e["location"], e["previous_state"], e["next_state"])
            for e in out["events"]] == EXPECTED_EVENTS
    row = out["transport_work"][0]
    assert (row["passenger_train_km"], row["passenger_car_km"], row["nonpassenger_train_km"], row["car_km"]) == ("6", "48", "0", "48")
    assert row["state_seconds"] == {"moving": 600, "stop": 120, "turnaround": 240, "standby": 0}
    assert sum(row["state_seconds"].values()) == 960
    assert out["final_state"][0]["location"] == "A"
    assert out["final_state"][0]["ready_to_depart"] is True
    assert out["unfinished_movements"] == out["unfinished_obligations"] == []
    assert out["status"] == "complete"
    assert out["checks"]["supported_constraints_status"] == "passed"
    assert out["release_authorized"] is False


def test_input_order_and_decimal_context_do_not_change_result():
    data = example()
    original = copy.deepcopy(data)
    expected = simulate(data, "baseline")
    for key in ("stations", "tracks", "trains", "plans"):
        data[key].reverse()
    data["plans"][1]["trips"].reverse()
    with localcontext() as ctx:
        ctx.prec = 3
        assert simulate(data, "baseline") == expected
    assert simulate(original, "baseline") == expected
    snapshot = copy.deepcopy(expected)
    data["trains"][0]["cars"] = 4
    assert expected == snapshot
    data["trains"][0]["cars"] = 8
    expected["checks"]["absent"].append("caller-only annotation")
    assert simulate(data, "baseline") == snapshot


@pytest.mark.parametrize("change, message", [
    ("overlap", "busy"), ("turnaround", "busy"), ("stop", "busy"),
    ("missing_track", "track"), ("broken_route", "route"),
    ("wrong_location", "location"), ("unknown_train", "train"),
])
def test_invalid_plans_are_rejected(change, message):
    data = example()
    trips = data["plans"][0]["trips"]
    if change == "overlap":
        trips[1]["departure_second"] = 60
    elif change == "turnaround":
        trips[1]["departure_second"] = 479
    elif change == "stop":
        trips[0]["route"] = ["AB"]
        trips[1].update(departure_second=179, route=["BC"])
    elif change == "missing_track":
        trips[0]["route"] = ["missing"]
    elif change == "broken_route":
        trips[0]["route"] = ["AB", "CB"]
    elif change == "wrong_location":
        trips[0]["route"] = ["CB", "BA"]
    elif change == "unknown_train":
        trips[0]["train_id"] = "missing"
    with pytest.raises(ValueError, match=message):
        simulate(data, "baseline")


@pytest.mark.parametrize("gap, accepted", [(112, False), (113, True), (114, True)])
def test_directed_track_headway(gap, accepted):
    data = one_leg(example())
    data["max_trains_on_line"] = 2
    data["trains"].append({**data["trains"][0], "train_id": "T2"})
    data["plans"][0]["trips"].append({**data["plans"][0]["trips"][0],
        "trip_id": "two", "train_id": "T2", "departure_second": gap})
    if accepted:
        assert simulate(data, "baseline")["checks"]["headway_seconds"] == 113
    else:
        with pytest.raises(ValueError, match="headway"):
            simulate(data, "baseline")


def test_previous_departure_across_window_boundary():
    data = one_leg(example())
    data["previous_departures"] = [{"track_id": "AB", "departure_second": -112}]
    with pytest.raises(ValueError, match="headway"):
        simulate(data, "baseline")
    data["previous_departures"][0]["departure_second"] = -113
    assert simulate(data, "baseline")["status"] == "complete"


def test_nonpassenger_distance_is_separate():
    out = simulate(one_leg(example(), False), "baseline")
    row = out["transport_work"][0]
    assert (row["passenger_train_km"], row["nonpassenger_train_km"],
            row["nonpassenger_car_km"], row["car_km"]) == ("0", "1", "8", "8")


def test_mid_segment_and_pending_turnaround_are_incomplete():
    data = example()
    data["period_end"] = "2026-05-01T03:00:59+03:00"
    out = simulate(data, "baseline")
    assert out["transport_work"][0]["passenger_train_km"] == "0"
    assert out["transport_work"][0]["state_seconds"]["moving"] == 59
    assert out["unfinished_movements"][0]["arrival_second"] == 120
    assert out["status"] == "partial"
    ev = to_evaluation(out)
    assert ev.obligations_closed is False
    assert ev.quantities["passenger_train_km"].status == "partial"
    with pytest.raises(ValueError, match="closed"):
        compare(ev, ev, [])
    data["period_end"] = "2026-05-01T03:15:00+03:00"
    out = simulate(data, "baseline")
    assert not out["unfinished_movements"]
    assert any(r["kind"] == "turnaround" for r in out["unfinished_obligations"])
    assert out["status"] == "partial"


def test_capacity_counts_standby_and_depot_entries():
    data = one_leg(example())
    data["trains"].append({**data["trains"][0], "train_id": "T2"})
    with pytest.raises(ValueError, match="capacity"):
        simulate(data, "baseline")
    data["stations"].append({"station_id": "D", "kind": "depot", "stop_seconds": 0, "turnaround_seconds": 0})
    data["trains"][1]["location"] = "D"
    data["tracks"].append({"track_id": "DA", "from_station": "D", "to_station": "A", "length_km": "0.5", "travel_seconds": 60})
    data["plans"][0]["trips"].append({"trip_id": "entry", "train_id": "T2", "departure_second": 0, "route": ["DA"], "passenger": False})
    with pytest.raises(ValueError, match="capacity"):
        simulate(data, "baseline")


def test_initial_wait_and_initial_movement_context():
    data = one_leg(example())
    data["trains"][0].update(state="turnaround", ready_after_seconds=1)
    with pytest.raises(ValueError, match="busy"):
        simulate(data, "baseline")
    data["trains"][0].update(state="moving", location="BA", ready_after_seconds=0,
        initial_movement={"arrival_second": 20, "passenger": False})
    data["plans"][0]["trips"][0]["departure_second"] = 140
    out = simulate(data, "baseline")
    assert out["transport_work"][0]["nonpassenger_train_km"] == "1"
    assert out["transport_work"][0]["state_seconds"]["moving"] == 140
    assert out["boundary_crossing_movements"]
    assert out["status"] == "partial"


def test_reversal_needs_a_turnaround_station():
    data = one_leg(example())
    data["plans"][0]["trips"][0]["route"] = ["AB", "BA"]
    with pytest.raises(ValueError, match="turnaround"):
        simulate(data, "baseline")


def test_simultaneous_events_are_ordered_and_closing_wait_is_completed():
    data = one_leg(example())
    data["max_trains_on_line"] = 2
    data["trains"].append({**data["trains"][0], "train_id": "T2", "location": "C"})
    data["plans"][0]["trips"].append({"trip_id": "second", "train_id": "T2", "departure_second": 0,
                                      "route": ["CB"], "passenger": True})
    expected = simulate(data, "baseline")
    data["trains"].reverse()
    data["plans"][0]["trips"].reverse()
    assert simulate(data, "baseline") == expected
    assert [e["train_id"] for e in expected["events"] if e["second"] == 0] == ["T1", "T2"]
    assert [e["event_type"] for e in expected["events"] if e["second"] == 180] == ["arrival", "stop_complete"]
    data = one_leg(example())
    data["period_end"] = "2026-05-01T03:02:00+03:00"
    out = simulate(data, "baseline")
    assert out["transport_work"][0]["passenger_train_km"] == "1"
    assert out["unfinished_obligations"][0]["kind"] == "stop"
    data["period_end"] = "2026-05-01T03:03:00+03:00"
    assert simulate(data, "baseline")["obligations_closed"] is True


def test_economics_adapter_and_same_initial_context():
    base, cand = [to_evaluation(simulate(example(), p)) for p in ("baseline", "candidate")]
    assert base.initial_state_id == cand.initial_state_id
    assert base.quantities["passenger_car_km"].amount == Decimal(48)
    assert base.quantities["moving_hours"].amount == Decimal(1) / Decimal(6)
    assert cand.quantities["standby_hours"].amount == Decimal(4) / Decimal(15)
    for ev in (base, cand):
        assert ev.quantities["paid_person_hours"].amount is None
        assert ev.quantities["purchased_energy"].amount is None
        assert ev.constraint_status == "unknown"
        assert evaluation_from_dict(json.loads(json.dumps(asdict(ev), default=str))) == ev
    result = compare(base, cand, load_costs(ROOT / "config/costs/default.json"))
    assert result["natural_deltas"]["passenger_car_km"]["delta"] == "48"
    assert result["groups"]["cash"]["savings"] is None
    assert result["release_authorized"] is False
    fixture_price = Article("manual", "passenger_car_km", "2", "RUB/car_km", ("manual",), "cash", "test fixture", "assumed")
    assert compare(base, cand, [fixture_price])["articles"][0]["savings"] == "96"


def test_invalid_units_time_and_unknown_fields():
    for field, bad in [("travel_seconds", 0), ("travel_seconds", 120.5), ("length_km", "NaN"), ("length_km", "-1")]:
        data = example()
        data["tracks"][0][field] = bad
        with pytest.raises(ValueError):
            simulate(data, "baseline")
    data = example()
    data["period_start"] = "2026-05-01T03:00:00"
    with pytest.raises(ValueError, match="timezone"):
        simulate(data, "baseline")
    data = example()
    data["tracks"][0]["arrival_second"] = 500
    with pytest.raises(ValueError, match="unsupported"):
        simulate(data, "baseline")


def test_example_outputs_repeat_and_failed_run_is_explicit(tmp_path):
    first, repeat = tmp_path / "first", tmp_path / "repeat"
    run_example(EXAMPLE, first, ROOT / "config/costs/default.json")
    run_example(EXAMPLE, repeat, ROOT / "config/costs/default.json")
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in repeat.iterdir()}
    expected = json.loads((first / "control_check.json").read_text(encoding="utf-8"))
    assert expected["status"] == "passed"
    assert all(r["expected"] == r["actual"] for r in expected["checks"])
    assert (first / "baseline.events.csv").exists()
    bad = example()
    bad["plans"][0]["trips"][0]["route"] = ["missing"]
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad), encoding="utf-8")
    command = [sys.executable, "-m", "simulation", "run", "--input", str(bad_file), "--output", str(first)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr
    assert json.loads((first / "build_manifest.json").read_text())["status"] == "error"


def test_output_cannot_overwrite_input_or_preserved_data():
    with pytest.raises(ValueError, match="output"):
        run_example(EXAMPLE, EXAMPLE.parent, ROOT / "config/costs/default.json")
    with pytest.raises(ValueError, match="output"):
        run_example(EXAMPLE, ROOT / "data/processed/metro", ROOT / "config/costs/default.json")
