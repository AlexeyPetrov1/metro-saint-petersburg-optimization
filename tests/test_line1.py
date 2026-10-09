"""Exact source aggregation, phased circulation and grouped passenger oracles."""

import copy
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from line1 import (STATION_IDS, largest_remainders, minute_groups, assign_destinations,
                   select_period, build_movement, load_config, car_profiles, main_economics)
from passenger_groups import GroupedPassengerSimulation, allocate_cars
from simulation import simulate


ROOT = Path(__file__).resolve().parents[1]


def tiny_movement():
    return {"schema_version": "1.0", "source": "group accounting fixture",
        "period_start": "2026-09-01T06:00:00+03:00", "period_end": "2026-09-01T06:10:00+03:00",
        "previous_departures": [], "max_trains_on_line": 2,
        "stations": [{"station_id": s, "kind": "line", "stop_seconds": 0, "turnaround_seconds": 0} for s in ("A", "B")],
        "tracks": [{"track_id": "AB", "from_station": "A", "to_station": "B", "length_km": "1", "travel_seconds": 120}],
        "trains": [{"train_id": t, "train_type": "fixture", "cars": 1, "location": "A", "state": "standby", "ready_after_seconds": 0} for t in ("T1", "T2")],
        "plans": [{"plan_id": "service", "trips": [
            {"trip_id": "one", "train_id": "T1", "departure_second": 60, "route": ["AB"], "passenger": True},
            {"trip_id": "two", "train_id": "T2", "departure_second": 180, "route": ["AB"], "passenger": True}]},
            {"plan_id": "none", "trips": []}]}


def tiny_groups(gate=0, platform=0, count=5):
    return [{"group_id": "G", "turnstile_second": gate, "platform_second": platform,
             "origin": "A", "count": count, "destinations": [{"station_id": "B", "count": count}]}]


def tiny_profiles():
    return {t: [{"car_id": "C1", "car_type": "fixture", "max_capacity": 3,
                "normal_threshold": 2, "source": "fixture"}] for t in ("T1", "T2")}


def tiny_runner(plan="service", groups=None):
    return GroupedPassengerSimulation(tiny_movement(), plan, groups if groups is not None else tiny_groups(), tiny_profiles())


@pytest.mark.parametrize("total,weights,expected", [(5, [1, 1, 1], [2, 2, 1]), (0, [1, 4], [0, 0]), (7, [1, 4, 1], [1, 5, 1])])
def test_largest_remainders_oracles(total, weights, expected):
    assert largest_remainders(total, weights) == expected


@pytest.mark.parametrize("total,weights", [(-1, [1]), (True, [1]), (1, []), (1, [0, 0]), (1, [1, -1]), (1, [0.5, 1]), (1, [True])])
def test_bad_integer_allocations_rejected(total, weights):
    with pytest.raises(ValueError): largest_remainders(total, weights)


def test_exact_allocation_large_numbers_and_stable_ties():
    for total in (1, 18, 10**30 + 7):
        amounts = largest_remainders(total, list(range(1, 19)))
        assert sum(amounts) == total
        assert all(type(n) is int and n >= 0 for n in amounts)


def test_minute_distribution_conserves_each_source_interval_and_delay():
    start = pd.Timestamp("2026-09-01T06:00:00+03:00")
    frame = pd.DataFrame([{"station_id": "A", "interval_start": start,
                          "interval_end": start + pd.Timedelta(minutes=15), "entry_count": 32, "source_id": "D02"}])
    groups = minute_groups(frame, ["A"], start, start + pd.Timedelta(minutes=15), 120)
    assert [g["count"] for g in groups] == [3, 3] + [2] * 13
    assert sum(g["count"] for g in groups) == 32
    assert all(g["platform_second"] - g["turnstile_second"] == 120 for g in groups)
    assert groups[-1]["platform_second"] == 960
    assert len(groups) == 15  # not 32 passenger objects


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3"])
def test_destinations_preserve_groups_and_direction(scenario):
    source = [{"group_id": "G", "turnstile_second": 0, "platform_second": 120, "origin": "L1_07", "count": 401}]
    out = assign_destinations(source, STATION_IDS, scenario)
    assert sum(d["count"] for d in out[0]["destinations"]) == 401
    assert all(d["station_id"] != "L1_07" for d in out[0]["destinations"])
    for d in out[0]["destinations"]:
        assert d["direction"] == (1 if STATION_IDS.index(d["station_id"]) > 6 else -1)
    assert source[0]["count"] == 401 and "destinations" not in source[0]
    assert out == assign_destinations(source, STATION_IDS, scenario)


def test_od_weights_match_manual_counts():
    groups = [{"group_id": "G", "turnstile_second": 0, "platform_second": 120, "origin": "L1_01", "count": 30}]
    rows = assign_destinations(groups, STATION_IDS, "S2")[0]["destinations"]
    assert {r["station_id"]: r["count"] for r in rows} == {s: (4 if s in STATION_IDS[6:10] else 1) for s in STATION_IDS[1:]}
    groups[0]["count"] = 171
    rows = assign_destinations(groups, STATION_IDS, "S3")[0]["destinations"]
    assert [r["count"] for r in rows] == list(range(1, 19))


def test_source_station_order_is_explicit_and_complete():
    config = load_config()
    mapping = pd.read_csv(ROOT / "data/processed/metro/vestibule_map.csv").drop_duplicates("station_id").set_index("station_id")
    assert config["station_ids"] == STATION_IDS
    assert mapping.loc[STATION_IDS[0], "station_name"] == "Проспект Ветеранов"
    assert mapping.loc[STATION_IDS[-1], "station_name"] == "Девяткино"
    assert set(mapping.index) == set(STATION_IDS)


def test_real_date_has_complete_coverage_and_exact_counts():
    frame = pd.read_parquet(ROOT / "data/processed/metro/entries_station_15m.parquet")
    selected, meta = select_period(frame, STATION_IDS)
    assert meta["date"] == "2026-09-01" and meta["date_replaced"] is False
    assert len(selected) == 19 * 16 and selected.is_complete.all()
    groups = minute_groups(selected, STATION_IDS, pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"]), 120)
    assert sum(g["count"] for g in groups) == int(selected.entry_count.sum())
    expected = selected.groupby(["station_id", "interval_start"]).entry_count.sum().to_dict()
    actual = Counter()
    for g in groups: actual[g["origin"], pd.Timestamp(g["source_interval_start"])] += g["count"]
    assert dict(actual) == expected


def test_date_fallback_only_if_date_absent_and_uses_complete_weekday():
    frame = pd.read_parquet(ROOT / "data/processed/metro/entries_station_15m.parquet")
    frame = frame[frame.interval_start.dt.date.astype(str) != "2026-09-01"]
    _, meta = select_period(frame, STATION_IDS)
    assert meta["date"] == "2026-09-02" and meta["date_replaced"]


def test_incomplete_requested_date_rejected_not_silently_replaced():
    frame = pd.read_parquet(ROOT / "data/processed/metro/entries_station_15m.parquet")
    row = frame[(frame.station_id == "L1_01") & (frame.interval_start == pd.Timestamp("2026-09-01T06:00:00+03:00"))].index[0]
    frame = frame.drop(row)
    with pytest.raises(ValueError): select_period(frame, STATION_IDS)


@pytest.mark.parametrize("phase_offset", [0, 7200])
def test_phased_train_plan_headways_continuity_and_full_cycles(phase_offset):
    data, meta = build_movement(load_config(), phase_offset=phase_offset)
    out = simulate(data, "line1-baseline")
    assert len(data["stations"]) == 19 and len(data["tracks"]) == 36
    assert len(data["trains"]) == 44 and meta["cycle_seconds"] == 5940 and meta["headway_seconds"] == 135
    assert out["checks"]["supported_constraints_status"] == "passed"
    assert out["checks"]["peak_trains_on_line"] == 44
    assert len({t["location"] for t in data["trains"]}) > 19
    for track in data["tracks"]:
        departures = [e["second"] for e in out["events"] if e["event_type"] == "departure" and e["location"] == track["track_id"]]
        assert all(b-a == 135 for a,b in zip(departures, departures[1:]))
    lengths = {t["track_id"]: int(round(float(t["length_km"]) * 1000)) for t in data["tracks"]}
    assert sum(v for k,v in lengths.items() if k.endswith("_up")) == 29600
    assert sum(v for k,v in lengths.items() if k.endswith("_down")) == 29600


def test_main_motion_continues_same_periodic_plan():
    full, _ = build_movement(load_config())
    main, _ = build_movement(load_config(), phase_offset=7200)
    a, b = simulate(full, "line1-baseline"), simulate(main, "line1-baseline")
    key = lambda e, offset: (e["second"]+offset, e["train_id"], e["event_type"], e["location"])
    assert [key(e, 0) for e in a["events"] if e["second"] > 7200] == [key(e,7200) for e in b["events"]]


def test_car_profile_retains_two_heads_four_motor_two_trailer_and_source_limits():
    profiles = car_profiles(load_config(), ["T01"])
    cars = profiles["T01"]
    assert [r["max_capacity"] for r in cars] == [168, 187, 187, 187, 187, 187, 187, 168]
    assert Counter(r["car_type"] for r in cars) == {"81-722.1": 2, "81-723.1": 4, "81-724.1": 2}
    assert all(r["normal_threshold"] == 120 for r in cars)
    assert sum(r["max_capacity"] for r in cars) == 1458
    assert all("D14" in r["source"] for r in cars)


def test_least_loaded_group_allocation_matches_individual_greedy_oracle():
    rng = random.Random(3308)
    for _ in range(100):
        caps = [rng.randrange(1, 12) for _ in range(8)]
        loads = [rng.randrange(c+1) for c in caps]
        count = rng.randrange(50)
        expected, additions = loads[:], [0]*8
        for _ in range(count):
            eligible = [i for i,c in enumerate(caps) if expected[i]<c]
            if not eligible: break
            i = min(eligible, key=lambda i:(expected[i], i))
            expected[i] += 1; additions[i] += 1
        assert allocate_cars(count, loads, caps) == additions
        assert loads != expected or not any(additions)


def test_grouped_manual_passenger_controls_match_legacy():
    out = tiny_runner().run()
    m = out["metrics"]
    assert m["waiting_passenger_seconds"] == 540
    assert m["mean_full_wait_seconds"] == "108" and m["p95_full_wait_seconds"] == 180
    assert (m["entered"],m["platform_arrived"],m["boarded"],m["completed"],m["queued"],m["onboard"],m["approaching"]) == (5,5,5,5,0,0,0)
    assert m["max_car_load"] == 3 and m["normal_threshold_exceedances"] == 1
    assert m["balance"]["passed"] and m["waiting_identity_passed"]
    assert tiny_runner().motion == simulate(tiny_movement(), "service")


def test_grouped_no_trains_retains_all_people_and_age():
    out = tiny_runner("none").run()
    assert out["metrics"]["queued"] == 5 and out["metrics"]["waiting_passenger_seconds"] == 3000
    assert out["metrics"]["mean_full_wait_seconds"] is None
    assert out["queues"][0]["count"] == 5 and out["queues"][0]["waiting_age_seconds"] == 600
    assert len(out["queues"]) == 1


def test_delay_crosses_warmup_boundary_exactly_once():
    runner = tiny_runner("none", tiny_groups(gate=230, platform=350))
    warm = runner.run(end_second=300)
    assert warm["metrics"]["entered"] == warm["metrics"]["approaching"] == 5
    assert warm["metrics"]["platform_arrived"] == warm["metrics"]["queued"] == 0
    main = runner.run(start_second=300, checkpoint=json.loads(json.dumps(warm["checkpoint"])))
    assert main["metrics"]["entered"] == 0 and main["metrics"]["platform_arrived"] == 5
    assert main["metrics"]["waiting_passenger_seconds"] == 1250
    assert main["metrics"]["initial_passengers"] == 5 and main["metrics"]["balance"]["passed"]
    assert main["checkpoint"] == runner.run()["checkpoint"]


def test_wait_inside_main_distinct_from_full_boarded_wait():
    data = tiny_movement()
    data["plans"][0]["trips"][0]["departure_second"] = 300
    data["plans"][0]["trips"] = data["plans"][0]["trips"][:1]
    runner = GroupedPassengerSimulation(data, "service", tiny_groups(count=1), tiny_profiles())
    warm = runner.run(end_second=240)
    main = runner.run(start_second=240, checkpoint=warm["checkpoint"])
    assert main["metrics"]["waiting_passenger_seconds"] == 60
    assert main["metrics"]["mean_full_wait_seconds"] == "300"
    assert main["metrics"]["initial_queue_age_passenger_seconds"] == 240


@pytest.mark.parametrize("cuts", [[60,180,600], [1,59,60,61,179,180,181,599,600]])
def test_grouped_split_equals_whole_state_and_event_traces(cuts):
    runner=tiny_runner(); whole=runner.run(); cp=None; start=0; area=0
    for end in cuts:
        out=runner.run(start_second=start,end_second=end,checkpoint=cp)
        area += out["metrics"]["waiting_passenger_seconds"]
        cp=json.loads(json.dumps(out["checkpoint"])); start=end
    assert cp==whole["checkpoint"] and area==540


def test_arrival_at_horizon_end_stays_on_approach():
    out=tiny_runner("none",tiny_groups(gate=480,platform=600)).run()
    assert out["metrics"]["entered"]==out["metrics"]["approaching"]==5
    assert out["metrics"]["platform_arrived"]==out["metrics"]["queued"]==0
    assert out["approaching_groups"][0]["platform_second"]==600


def test_groups_never_expand_people_into_individual_objects():
    out=tiny_runner("none",tiny_groups(count=10**9)).run()
    assert len(out["queues"])==1
    assert out["metrics"]["queued"]==10**9
    assert out["metrics"]["waiting_passenger_seconds"]==600*10**9


def test_main_economics_uses_existing_adapter_and_keeps_money_unknown():
    data,_=build_movement(load_config(),phase_offset=7200)
    out=main_economics(simulate(data,"line1-baseline"))
    assert out["money_status"]=="unknown" and out["savings"] is None
    assert out["release_authorized"] is False and out["all_constraints_verified"] is False
    q=out["evaluation"]["quantities"]
    assert q["purchased_energy"]["amount"] is None and q["paid_person_hours"]["amount"] is None
    assert out["comparison_performed"] is False


def test_exact_queue_peak_includes_arrival_before_same_instant_boarding():
    groups=tiny_groups(gate=60,platform=60,count=3)
    out=tiny_runner(groups=groups).run()
    assert out["metrics"]["max_queue"]==3
    assert out["metrics"]["max_platform_direction_queue"]==3


def test_main_queue_peak_does_not_inherit_a_larger_warmup_peak():
    runner=tiny_runner()
    warm=runner.run(end_second=240)
    main=runner.run(start_second=240,checkpoint=warm["checkpoint"])
    assert warm["metrics"]["max_queue"]==5
    assert main["metrics"]["max_queue"]==0


@pytest.mark.parametrize("change", ["wrong_direction","duplicate_group","count_mismatch","self_destination","unknown_destination","negative_gate","platform_before_gate","negative_count","bool_count","wrong_car_count","car_overflow_threshold","wrong_checkpoint","gap","overlap"])
def test_bad_grouped_inputs_or_windows_rejected(change):
    data,groups,profiles=tiny_movement(),tiny_groups(),tiny_profiles()
    if change=="wrong_direction": groups[0]["destinations"][0]["direction"]=-1
    if change=="duplicate_group": groups.append(copy.deepcopy(groups[0]))
    if change=="count_mismatch": groups[0]["count"]=6
    if change=="self_destination": groups[0]["destinations"][0]["station_id"]="A"
    if change=="unknown_destination": groups[0]["destinations"][0]["station_id"]="C"
    if change=="negative_gate": groups[0]["turnstile_second"]=-1
    if change=="platform_before_gate": groups[0].update(turnstile_second=120,platform_second=100)
    if change=="negative_count": groups[0]["count"]=-1
    if change=="bool_count": groups[0]["count"]=True
    if change=="wrong_car_count": profiles["T1"].append(copy.deepcopy(profiles["T1"][0]))
    if change=="car_overflow_threshold": profiles["T1"][0]["normal_threshold"]=4
    with pytest.raises(ValueError):
        runner=GroupedPassengerSimulation(data,"service",groups,profiles)
        cp=runner.run(end_second=60)["checkpoint"]
        if change=="wrong_checkpoint": cp["entered"]+=1
        runner.run(start_second=61 if change=="gap" else 59 if change=="overlap" else 60,checkpoint=cp)


def test_checkpoint_compression_preserves_content_and_bytes(tmp_path):
    import gzip
    from line1 import write_checkpoint
    cp=tiny_runner().run()["checkpoint"]
    for name in ("a.json.gz","b.json.gz"): write_checkpoint(tmp_path/name,cp)
    a,b=tmp_path/"a.json.gz",tmp_path/"b.json.gz"
    assert a.read_bytes()==b.read_bytes()
    assert json.loads(gzip.decompress(a.read_bytes()))==cp


def test_s1_real_groups_whole_run_equals_warmup_and_three_main_windows():
    config=load_config()
    frame=pd.read_parquet(ROOT/"data/processed/metro/entries_station_15m.parquet")
    selected,period=select_period(frame,config["station_ids"])
    groups=assign_destinations(minute_groups(selected,config["station_ids"],pd.Timestamp(period["start"]),pd.Timestamp(period["end"])) ,config["station_ids"],"S1")
    data,_=build_movement(config)
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,car_profiles(config,[t["train_id"] for t in data["trains"]]))
    whole=runner.run(); start=0; cp=None; total_wait=0
    for end in (7200,7201,10800,14400):
        out=runner.run(start_second=start,end_second=end,checkpoint=cp)
        total_wait+=out["metrics"]["waiting_passenger_seconds"]
        cp=json.loads(json.dumps(out["checkpoint"])); start=end
    assert cp==whole["checkpoint"]
    assert total_wait==whole["metrics"]["waiting_passenger_seconds"]
    assert whole["metrics"]["cumulative"]["entered"]==199850
    assert whole["metrics"]["queued"]==555


@pytest.mark.parametrize("count,loads,caps,expected", [
    (2,[0,1,1,1],[5]*4,[2,0,0,0]),
    (9,[2,2,3,3],[10]*4,[3,3,2,1]),
    (10,[2,4],[2,100],[0,10]),
    (1000,[1,3,2],[2,5,4],[1,2,2])])
def test_balanced_allocator_fast_path_and_saturation(count,loads,caps,expected):
    assert allocate_cars(count,loads,caps)==expected


def test_experiment_artifact_manifest_rejects_empty_and_foreign_output(tmp_path):
    from line1 import check_output,run_experiment
    (tmp_path/"build_manifest.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError): check_output(tmp_path)
    (tmp_path/"keep.txt").write_text("keep",encoding="utf-8")
    with pytest.raises(ValueError): run_experiment(tmp_path)
    assert (tmp_path/"keep.txt").read_text(encoding="utf-8")=="keep"


def test_preserved_motion_outputs_cannot_be_line_experiment_output():
    from line1 import run_experiment
    with pytest.raises(ValueError): run_experiment(ROOT/"data/processed/simulation")


def test_fixed_experiment_time_fields_cannot_be_ignored(tmp_path):
    config=load_config(); config["warmup_start"]="05:00"
    path=tmp_path/"config.json"; path.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError): load_config(path)


def test_line_manifest_cannot_certify_outputs_without_program_and_source_inputs(tmp_path):
    import hashlib
    from line1 import OUTPUTS,check_output,MODEL_VERSION
    outputs={}
    for name in OUTPUTS:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b"")
        outputs[name]=hashlib.sha256(b"").hexdigest()
    arbitrary=tmp_path.parent/"arbitrary.txt"; arbitrary.write_bytes(b"data")
    manifest={"model_version":MODEL_VERSION,"status":"success",
        "configuration_input":str(ROOT/"config/line1/experiment.json"),
        "inputs":{str(arbitrary):hashlib.sha256(b"data").hexdigest()},"outputs":outputs}
    (tmp_path/"build_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError): check_output(tmp_path)
