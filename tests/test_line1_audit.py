"""Adversarial inputs and an independent, individual passenger oracle."""

import copy
import json
import random
import os
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from line1 import _safe_output, build_movement, load_config, minute_groups, run_experiment, select_period
from passenger_groups import GroupedPassengerSimulation
from passengers import PassengerSimulation
from simulation import simulate
from test_line1 import tiny_groups, tiny_movement, tiny_profiles, tiny_runner


@pytest.mark.parametrize("change", [
    "missing", "extra", "version", "bool_second", "float_cursor", "negative_peak",
    "boarded", "entered", "platform", "completed", "approaching", "queue_counter",
    "onboard_counter", "wait_area", "histogram", "queue_counts", "loads", "fleet",
    "car_count", "cohort_count", "cohort_identity", "cohort_time", "cohort_direction",
    "queue_count", "queue_identity", "queue_time", "queue_key", "queue_duplicate",
    "histogram_key", "history_type", "history_future", "board_history", "load_history",
])
def test_resealed_inconsistent_checkpoint_rejected(change):
    runner = tiny_runner()
    cp = copy.deepcopy(runner.run(end_second=70)["checkpoint"])
    if change == "missing": del cp["queued"]
    if change == "extra": cp["surprise"] = 1
    if change == "version": cp["model_version"] = "other"
    if change == "bool_second":
        cp = tiny_runner().run(end_second=1)["checkpoint"]
        cp["second"] = True
    if change == "float_cursor": cp["next_event_index"] = float(cp["next_event_index"])
    if change == "negative_peak": cp["max_queue"] = -1
    if change in ("boarded", "entered", "completed", "approaching"):
        cp[change] += 1
    if change == "platform": cp["platform_arrived"] += 1
    if change == "queue_counter": cp["queued"] += 1
    if change == "onboard_counter": cp["onboard_count"] += 1
    if change == "wait_area": cp["waiting_passenger_seconds"] += 1
    if change == "histogram": cp["wait_histogram"]["60"] += 1
    if change == "histogram_key": cp["wait_histogram"]["060"] = cp["wait_histogram"].pop("60")
    if change == "queue_counts": cp["queue_counts"]["A:1"] += 1
    if change == "loads": cp["loads"]["T1"][0] += 1
    if change == "fleet": del cp["loads"]["T2"]
    if change == "car_count": cp["onboard"]["T1"].append([])
    cohort = cp["onboard"]["T1"][0][0] if change != "bool_second" else None
    if change == "cohort_count": cohort["count"] -= 1
    if change == "cohort_identity": cohort["destination"] = "A"
    if change == "cohort_time": cohort["boarded_second"] = 69
    if change == "cohort_direction": cohort["direction"] = -1
    queue = cp["queues"]["A:1"][0]
    if change == "queue_count": queue["count"] += 1
    if change == "queue_identity": queue["group_id"] = "unknown"
    if change == "queue_time": queue["platform_second"] += 1
    if change == "queue_key": cp["queues"]["B:-1"] = cp["queues"].pop("A:1")
    if change == "queue_duplicate": cp["queues"]["A:1"].append(copy.deepcopy(queue))
    if change == "history_type": cp["load_events"] = {}
    if change == "history_future": cp["queue_events"][0]["second"] = 70
    if change == "board_history": cp["board_events"][0]["count"] += 1
    if change == "load_history": cp["load_events"][-1]["load"] += 1
    cp["sha256"] = PassengerSimulation._seal(cp)
    with pytest.raises(ValueError):
        runner.run(start_second=1 if change == "bool_second" else 70, checkpoint=cp)


@pytest.mark.parametrize("change", ["groups_object", "group_object", "missing_field", "destinations_object",
    "destination_object", "bool_direction", "float_direction", "profiles_list", "missing_car_field"])
def test_malformed_group_input_has_clear_validation_error(change):
    groups, profiles = tiny_groups(), tiny_profiles()
    if change == "groups_object": groups = {}
    if change == "group_object": groups = [None]
    if change == "missing_field": del groups[0]["turnstile_second"]
    if change == "destinations_object": groups[0]["destinations"] = {}
    if change == "destination_object": groups[0]["destinations"] = [None]
    if change == "bool_direction": groups[0]["destinations"][0]["direction"] = True
    if change == "float_direction": groups[0]["destinations"][0]["direction"] = 1.0
    if change == "profiles_list": profiles = []
    if change == "missing_car_field": del profiles["T1"][0]["source"]
    with pytest.raises(ValueError):
        GroupedPassengerSimulation(tiny_movement(), "service", groups, profiles)


@pytest.mark.parametrize("change", ["object", "missing", "date", "train_type", "capacity", "car_types",
    "formation", "zero_segment", "provenance"])
def test_configuration_cannot_misattribute_source_values(tmp_path, change):
    config = load_config()
    if change == "object": config = []
    if change == "missing": del config["date"]
    if change == "date": config["date"] = "invalid"
    if change == "train_type": config["train_type"] = "another fleet"
    if change == "capacity": config["capacity_by_type"]["81-722.1"] = 999
    if change == "car_types": config["car_types"] = "12345678"
    if change == "formation": config["car_types"] = ["81-722.1"] * 8
    if change == "zero_segment": config["length_m"] = 1
    if change == "provenance": config["provenance"]["travel_stop_turnaround"]["status"] = "confirmed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError): load_config(path)


def test_output_must_not_replace_its_own_configuration(tmp_path):
    from line1 import MODEL_VERSION
    config_path = tmp_path / "assumptions.json"
    config_path.write_text(json.dumps(load_config()), encoding="utf-8")
    (tmp_path / "build_manifest.json").write_text(json.dumps({"model_version": MODEL_VERSION}), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="configuration.*output"):
        run_experiment(tmp_path, config_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("directory", [False, True])
def test_owned_output_links_cannot_write_outside_output(tmp_path, directory):
    from line1 import MODEL_VERSION
    output = tmp_path / "output"; output.mkdir()
    victim = tmp_path / "victim"
    if directory: victim.mkdir(); (victim / "metrics.json").write_text("keep")
    else: victim.write_text("keep")
    link = output / ("S1" if directory else "summary.md")
    if directory and os.name == "nt":
        script = tmp_path / "junction.ps1"
        script.write_text("param($linkPath,$targetPath)\nNew-Item -ItemType Junction -Path $linkPath -Target $targetPath -ErrorAction Stop | Out-Null", encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-File", str(script), str(link), str(victim)], check=True, capture_output=True)
    elif directory: link.symlink_to(victim, target_is_directory=True)
    else: os.link(victim, link)
    (output / "build_manifest.json").write_text(json.dumps({"model_version": MODEL_VERSION}), encoding="utf-8")
    with pytest.raises(ValueError): _safe_output(output)
    assert (victim / "metrics.json" if directory else victim).read_text() == "keep"


def test_changed_plan_summary_reports_actual_parameters():
    from line1 import build_movement, scenario_parameters_text
    config = load_config(); config["train_count"] = 22; config["platform_delay_seconds"] = 30
    _, meta = build_movement(config)
    text = scenario_parameters_text(config, meta)
    assert "22" in text and "270" in text and "подход 30 с" in text
    assert "44 состава" not in text and "135 с" not in text and "подход 120 с" not in text


@pytest.mark.parametrize("count", [True, 32.0, -1, None])
def test_minute_source_counts_strict_integer_contract(count):
    start = pd.Timestamp("2026-09-01T06:00:00+03:00")
    frame = pd.DataFrame([{"station_id": "A", "interval_start": start,
        "interval_end": start + pd.Timedelta(minutes=15), "entry_count": count, "source_id": "fixture"}])
    with pytest.raises(ValueError): minute_groups(frame, ["A"], start, start + pd.Timedelta(minutes=15))


@pytest.mark.parametrize("flag", [None, pd.NA, "False", 1])
def test_unknown_or_non_boolean_coverage_is_not_complete(flag):
    start = pd.Timestamp("2026-09-01T06:00:00+03:00")
    rows = [{"station_id": "A", "interval_start": start+pd.Timedelta(minutes=15*i),
        "interval_end": start+pd.Timedelta(minutes=15*(i+1)), "entry_count": 3, "is_complete": True} for i in range(16)]
    rows[0]["is_complete"] = flag
    with pytest.raises(ValueError): select_period(pd.DataFrame(rows), ["A"])


@pytest.mark.parametrize("offset", [1, 139, 140, 159, 160, 2859, 2860, 2969, 2970, 5939, 5940, 7199, 7200, 7201, 14399]
                         + random.Random(3311).sample(range(14400), 15))
def test_line_plan_matches_independent_periodic_arithmetic(offset):
    config = load_config(); data, _ = build_movement(config, phase_offset=offset)
    result = simulate(data, "line1-baseline"); duration = 14400-offset
    legs = [(160*i, f"P{i+1:02}_up", 20 if i < 17 else 110) for i in range(18)]
    legs += [(2970+160*i, f"P{18-i:02}_down", 20 if i < 17 else 110) for i in range(18)]
    actual = Counter((e["train_id"],e["event_type"],e["second"],e["location"]) for e in result["events"]
                     if e["event_type"] in ("departure", "arrival"))
    expected = Counter(); distances = Counter(); seconds = {}
    tracks = {t["track_id"]:t for t in data["tracks"]}
    for i in range(44):
        train = f"T{i+1:02}"; phase = (i*135+offset) % 5940
        state_seconds = Counter()
        for dep, track, wait in legs:
            for cycle in range(-1, 4):
                second = dep+cycle*5940-phase
                if 0 <= second < duration: expected[train,"departure",second,track] += 1
                if 0 < second+140 <= duration:
                    expected[train,"arrival",second+140,tracks[track]["to_station"]] += 1
                    distances[train] += Decimal(tracks[track]["length_km"])
                for left, right, state in ((second,second+140,"moving"),
                    (second+140,second+140+wait,"turnaround" if wait==110 else "stop")):
                    state_seconds[state] += max(0,min(duration,right)-max(0,left))
        seconds[train] = state_seconds
    assert actual == expected
    for work in result["transport_work"]:
        train = work["train_id"]
        assert Decimal(work["passenger_train_km"]) == distances[train]
        assert work["state_seconds"] == {k:seconds[train][k] for k in ("moving","stop","turnaround","standby")}


def oracle_case(seed):
    rng = random.Random(seed)
    data = tiny_movement()
    data["period_end"] = "2026-09-01T06:15:00+03:00"
    data["max_trains_on_line"] = 3
    data["stations"] = [{"station_id": s, "kind": "line", "stop_seconds": 20 if s == "B" else 0,
                         "turnaround_seconds": 0 if s == "B" else 110} for s in "ABC"]
    data["tracks"] = [{"track_id": a+b, "from_station": a, "to_station": b, "length_km": "1", "travel_seconds": 120}
                      for a, b in ("AB", "BC", "CB", "BA")]
    data["trains"] = [{"train_id": t, "train_type": "fixture", "cars": 2, "location": loc,
                       "state": "standby", "ready_after_seconds": 0} for t, loc in (("T1", "A"), ("T2", "A"), ("R1", "C"))]
    trips = []
    for t, dep, route in (("T1", 60, ["AB", "BC"]), ("T2", 180, ["AB", "BC"]), ("R1", 0, ["CB", "BA"])):
        trips.extend([{"trip_id": t+"out", "train_id": t, "departure_second": dep, "route": route, "passenger": True},
                      {"trip_id": t+"back", "train_id": t, "departure_second": dep+370,
                       "route": [x[::-1] for x in reversed(route)], "passenger": seed % 3 != 0}])
    data["plans"] = [{"plan_id": "service", "trips": trips}]
    groups = []
    for i in range(25):
        origin = rng.choice("ABC"); gate = rng.choice([0, 60, 120, 180, 320, 430, 550, 899, rng.randrange(900)])
        dests = [{"station_id": s, "count": rng.randrange(8)} for s in "ABC" if s != origin]
        dests = [d for d in dests if d["count"]]
        groups.append({"group_id": f"G{i:03}", "turnstile_second": gate,
            "platform_second": gate+rng.choice([0, 10, 120, 250]), "origin": origin,
            "count": sum(d["count"] for d in dests), "destinations": dests})
    profiles = {t["train_id"]: [{"car_id": f"C{i}", "car_type": "fixture", "max_capacity": cap,
                 "normal_threshold": 2, "source": "individual oracle"} for i, cap in enumerate((3, 5))] for t in data["trains"]}
    return data, groups, profiles


def individual_oracle(data, groups, profiles, motion):
    """Person-by-person boarding. Does not call the group allocator or processors."""
    tracks = {t["track_id"]: t for t in data["tracks"]}
    trips = {t["trip_id"]: t for t in data["plans"][0]["trips"]}
    people = []
    for g in sorted(groups, key=lambda g: (g["platform_second"], g["group_id"])):
        for d in sorted(g["destinations"], key=lambda d: d["station_id"]):
            people.extend({"group": g["group_id"], "origin": g["origin"], "dest": d["station_id"],
                "platform": g["platform_second"], "gate": g["turnstile_second"], "state": "outside", "train": None}
                for _ in range(d["count"]))
    stream = [(g["turnstile_second"], 0, g["group_id"], "gate", g) for g in groups]
    stream += [(g["platform_second"], 1, g["group_id"], "platform", g) for g in groups]
    stream += [(e["second"], 2 if e["event_type"] == "arrival" else 4, f"{i:012}", "motion", e)
               for i, e in enumerate(motion["events"]) if e["event_type"] in ("arrival", "departure")]
    loads = {t: [0]*len(cars) for t, cars in profiles.items()}
    area = 0; now = 0; max_queue = 0; max_direction = 0; waits = []; peak = 0; exceeded = 0
    for second, _, _, kind, event in sorted(stream):
        if second >= 900: continue
        area += sum(p["state"] == "queue" for p in people) * (second-now); now = second
        if kind in ("gate", "platform"):
            for p in people:
                if p["group"] == event["group_id"]: p["state"] = "approach" if kind == "gate" else "queue"
        else:
            t = event["train_id"]
            if event["event_type"] == "arrival":
                for p in people:
                    if p["state"] == "onboard" and p["train"] == t and p["dest"] == event["location"]:
                        loads[t][p["car"]] -= 1; p["state"] = "completed"
            else:
                trip = trips[event["trip_id"]]
                if not trip["passenger"]: continue
                track = tracks[event["location"]]; route = trip["route"]
                dests = {tracks[r]["to_station"] for r in route[route.index(event["location"]):]}
                forward = track["to_station"] > track["from_station"]
                for p in people:
                    if (p["state"] != "queue" or p["origin"] != track["from_station"] or p["dest"] not in dests
                            or (p["dest"] > p["origin"]) != forward): continue
                    free = [i for i, car in enumerate(profiles[t]) if loads[t][i] < car["max_capacity"]]
                    if not free: break
                    car = min(free, key=lambda i: (loads[t][i], i))
                    loads[t][car] += 1; p.update(state="onboard", train=t, car=car)
                    waits.append(second-p["platform"])
            peak = max(peak, *loads[t])
            exceeded += sum(n > car["normal_threshold"] for n, car in zip(loads[t], profiles[t]))
        counts = Counter((p["origin"], p["dest"] > p["origin"]) for p in people if p["state"] == "queue")
        max_queue = max(max_queue, sum(counts.values())); max_direction = max(max_direction, max(counts.values(), default=0))
    area += sum(p["state"] == "queue" for p in people)*(900-now)
    census = Counter(p["state"] for p in people)
    return {"entered": sum(p["gate"] < 900 for p in people), "platform_arrived": sum(p["platform"] < 900 for p in people),
        "boarded": len(waits), "completed": census["completed"], "queued": census["queue"], "onboard": census["onboard"],
        "approaching": census["approach"], "waiting_passenger_seconds": area,
        "full_boarded_wait_passenger_seconds": sum(waits), "p95_full_wait_seconds": sorted(waits)[(95*len(waits)+99)//100-1] if waits else None,
        "max_queue": max_queue, "max_platform_direction_queue": max_direction, "max_car_load": peak,
        "normal_threshold_exceedances": exceeded}, loads


@pytest.mark.parametrize("seed", range(80))
def test_group_engine_matches_independent_individual_census(seed):
    data, groups, profiles = oracle_case(seed)
    runner = GroupedPassengerSimulation(data, "service", groups, profiles)
    whole = runner.run()
    expected, loads = individual_oracle(data, groups, profiles, runner.motion)
    assert {k: whole["metrics"][k] for k in expected} == expected
    assert whole["checkpoint"]["loads"] == loads
    cp = None; start = 0; area = 0
    for end in (1, 60, 61, 180, 320, 430, 899, 900):
        window = runner.run(start_second=start, end_second=end, checkpoint=cp)
        area += window["metrics"]["waiting_passenger_seconds"]
        cp = json.loads(json.dumps(window["checkpoint"])); start = end
    assert cp == whole["checkpoint"] and area == expected["waiting_passenger_seconds"]
    empty = GroupedPassengerSimulation(data, "service", [], profiles)
    assert empty.motion["transport_work"] == runner.motion["transport_work"]


def test_closing_movement_arrival_alights_once():
    data = tiny_movement(); data["period_end"] = "2026-09-01T06:03:00+03:00"
    data["plans"][0]["trips"] = data["plans"][0]["trips"][:1]
    runner = GroupedPassengerSimulation(data, "service", tiny_groups(count=3), tiny_profiles())
    whole = runner.run(); warm = runner.run(end_second=179)
    split = runner.run(start_second=179, checkpoint=warm["checkpoint"])
    assert whole["metrics"]["completed"] == 3 and whole["metrics"]["onboard"] == 0
    assert split["checkpoint"] == whole["checkpoint"]
    assert sum(e["event_type"] == "arrival" for e in whole["load_events"]) == 1
