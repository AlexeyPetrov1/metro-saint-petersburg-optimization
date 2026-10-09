"""Regular-release comparisons, with independent two-station controls."""

import copy
import json
from decimal import Decimal

import pytest
import pandas as pd

from decision_engine import (build_regular_plan, headway_report, load_policy, pareto_frontier,
    run_variant, resource_report, check_service, prepare_demand, safe_output)
from line1 import load_config
from simulation import simulate, to_evaluation


def toy_config():
    return {"station_ids": ["A", "B"], "date": "2026-09-01", "timezone": "Europe/Moscow",
        "length_m": 2000, "travel_seconds": 120, "stop_seconds": 0, "turnaround_seconds": 10,
        "train_type": "manual fixture", "car_types": ["fixture"]}


def toy_inputs(count=2):
    data, meta = build_regular_plan(toy_config(), count, end_second=520)
    profiles = {t["train_id"]: [{"car_id": "C1", "car_type": "fixture", "max_capacity": 5,
        "normal_threshold": 5, "source": "manual two-station fixture"}] for t in data["trains"]}
    return data, meta, profiles


def demand():
    return [{"group_id": name, "turnstile_second": 0, "platform_second": platform, "origin": "A",
        "count": count, "destinations": [{"station_id": "B", "count": count}]}
        for name, platform, count in (("G0", 0, 4), ("G1", 1, 1))]


@pytest.mark.parametrize("count,low,high", [(42,141,142),(43,138,139),(44,135,135),(45,132,132),(46,129,130)])
def test_real_fleet_phases_are_exact_and_wrap_without_rounding_drift(count,low,high):
    data, meta = build_regular_plan(load_config(), count)
    assert len(data["trains"]) == count and meta["cycle_seconds"] == 5940
    phases = meta["phase_seconds"]
    assert phases == [i*5940//count for i in range(count)]
    gaps = [b-a for a,b in zip(phases,phases[1:]+[phases[0]+5940])]
    assert sum(gaps) == 5940 and min(gaps) == low and max(gaps) == high
    result = simulate(data,"line1-baseline")
    assert result["checks"]["peak_trains_on_line"] == count
    report = headway_report(data,result)
    assert report["minimum_seconds"] == low and report["maximum_seconds"] == high
    for track in report["by_track"].values(): assert track["minimum_seconds"] >= 113
    for train in data["trains"]:
        arrivals = [e for e in result["events"] if e["train_id"]==train["train_id"] and e["event_type"]=="arrival"]
        assert all(a["second"] < b["second"] for a,b in zip(arrivals,arrivals[1:]))


def test_baseline_plan_reproduces_existing_generator():
    from line1 import build_movement
    config=load_config()
    old,_=build_movement(config)
    new,_=build_regular_plan(config,44,end_second=14400)
    assert new == old
    assert simulate(new,"line1-baseline")==simulate(old,"line1-baseline")


@pytest.mark.parametrize("offset", [1,139,140,160,2970,7200,14399])
def test_nonintegral_headways_preserve_physical_main_window_initial_state(offset):
    config=load_config(); whole,_=build_regular_plan(config,43)
    cut,_=build_regular_plan(config,43,phase_offset=offset)
    a,b=simulate(whole,"line1-baseline"),simulate(cut,"line1-baseline")
    key=lambda e,t:(e["second"]+t,e["train_id"],e["event_type"],e["location"])
    expected=[key(e,0) for e in a["events"] if e["second"]>offset or e["second"]==offset and e["event_type"]=="departure"]
    assert expected==[key(e,offset) for e in b["events"]]


@pytest.mark.parametrize("count", [0,-1,True,44.0,54,1000])
def test_invalid_release_has_specific_reason(count):
    with pytest.raises(ValueError): build_regular_plan(load_config(),count)


def test_short_cycle_rejects_headway_below_113_including_wrap():
    with pytest.raises(ValueError,match="113"):
        build_regular_plan(toy_config(),3,end_second=520)


@pytest.mark.parametrize("count,wait,p95,km,queue_at_260", [(1,259,259,"8",1),(2,129,129,"16",0)])
def test_manual_two_station_control(count,wait,p95,km,queue_at_260):
    data, _, profiles=toy_inputs(count)
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=0,main_end_second=260)
    assert out["full_metrics"]["waiting_passenger_seconds"] == wait
    assert out["full_metrics"]["mean_full_wait_seconds"] == str(Decimal(wait)/5)
    assert out["full_metrics"]["p95_full_wait_seconds"] == p95
    assert out["main"]["metrics"]["queued"] == queue_at_260
    assert out["tail"]["metrics"]["queued"] == out["tail"]["metrics"]["onboard"] == 0
    assert out["tail"]["metrics"]["cumulative"]["completed"] == 5
    assert resource_report(out["runner"].motion)["passenger_train_km"] == km
    assert out["checks"]["physical_capacity"]["status"] == "passed"


def test_variant_does_not_mutate_inputs_or_neighbors():
    data,_,profiles=toy_inputs(); groups=demand()
    before=copy.deepcopy((data,groups,profiles))
    first=run_variant(data,groups,profiles,load_policy(),decision_second=0,main_end_second=260)
    second=run_variant(data,groups,profiles,load_policy(),decision_second=0,main_end_second=260)
    assert (data,groups,profiles)==before
    assert first["tail"]["checkpoint"]==second["tail"]["checkpoint"]


def test_600_seconds_and_queue_growth_are_disabled_by_default():
    policy=load_policy()
    assert policy["max_boarded_p95_wait_seconds"] is None
    assert policy["max_active_wait_seconds"] is None
    assert policy["prevent_queue_growth"] is False


def test_normal_load_is_separate_from_physical_capacity():
    data,_,profiles=toy_inputs()
    for cars in profiles.values(): cars[0]["normal_threshold"]=2
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=0,main_end_second=260)
    assert out["checks"]["physical_capacity"]["status"]=="passed"
    assert out["checks"]["normal_load"]["status"]=="failed"


def test_absent_boardings_are_not_zero_wait_and_queues_survive_continuation():
    data,_,profiles=toy_inputs(); data["plans"][0]["trips"]=[]
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=0,main_end_second=260)
    assert out["full_metrics"]["p95_full_wait_seconds"] is None
    assert out["full_metrics"]["queued"]==5
    assert out["full_metrics"]["max_queue_age_seconds"]==520
    assert out["checks"]["waiting_policy"]["status"]=="not_configured"


def test_economics_receives_simulated_work_without_invented_money():
    data,_,profiles=toy_inputs()
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=0,main_end_second=260)
    eval=to_evaluation(out["runner"].motion)
    assert eval.quantities["passenger_train_km"].amount==Decimal("16")
    assert eval.quantities["purchased_energy"].amount is None
    assert out["economics"]["cash_total_RUB"] is None
    assert out["economics"]["savings"] is None


def rows():
    return [{"variant_id":v,"scenario":s,"train_km":km,"waiting_passenger_seconds":w,"queued_main":q}
        for v,km,w,q in (("a","10",30,2),("b","12",20,1),("c","14",40,3)) for s in ("S1","S2","S3")]


def test_pareto_tradeoffs_and_iteration_order():
    criteria=["train_km","waiting_passenger_seconds","queued_main"]
    expected=pareto_frontier(rows(),criteria)
    assert expected["non_dominated"]==["a","b"]
    assert pareto_frontier(list(reversed(rows())),criteria)==expected


def test_one_bad_od_prevents_dominance_without_averaging():
    data=rows()[:6]
    for r in data:
        if r["variant_id"]=="b": r["train_km"]="9"
        if r["variant_id"]=="b" and r["scenario"]=="S3": r["waiting_passenger_seconds"]=100
    assert pareto_frontier(data,["train_km","waiting_passenger_seconds"])["non_dominated"]==["a","b"]


def test_null_percentile_cannot_dominate_a_known_wait():
    data=[{"variant_id":v,"scenario":s,"p95":p} for v,p in (("a",None),("b",60)) for s in ("S1","S2","S3")]
    assert pareto_frontier(data,["p95"])["non_dominated"]==["a","b"]


def test_existing_artifacts_and_foreign_directory_protected(tmp_path):
    from pathlib import Path
    with pytest.raises(ValueError): safe_output(Path(__file__).resolve().parents[1]/"data/processed/line1")
    (tmp_path/"keep.txt").write_text("keep")
    with pytest.raises(ValueError): safe_output(tmp_path)
    assert (tmp_path/"keep.txt").read_text()=="keep"


def source_frame():
    start=pd.Timestamp("2026-09-01T06:00:00+03:00")
    return pd.DataFrame([{"station_id":s,"interval_start":start+pd.Timedelta(minutes=15*i),
        "interval_end":start+pd.Timedelta(minutes=15*(i+1)),"entry_count":18,"is_complete":True,"source_id":"fixture"}
        for i in range(24) for s in ("A","B")])


def test_demand_is_created_once_and_includes_observed_continuation():
    config=toy_config(); config["platform_delay_seconds"]=120
    by_s,minutes,period=prepare_demand(source_frame(),config)
    assert period["end"].endswith("12:00:00+03:00") and period["source_rows"]==48
    assert sum(g["count"] for g in minutes)==864
    assert sum(g["count"] for g in minutes if g["turnstile_second"]>=14400)==288
    for groups in by_s.values():
        assert sum(g["count"] for g in groups)==864
        assert sum(d["count"] for g in groups for d in g["destinations"])==864
    assert by_s["S1"]==by_s["S2"]==by_s["S3"]  # only one other station


@pytest.mark.parametrize("change", ["missing","duplicate","incomplete","unknown","negative","float"])
def test_missing_or_invalid_continuation_is_not_silently_zero_demand(change):
    config=toy_config(); config["platform_delay_seconds"]=120; frame=source_frame()
    if change=="missing": frame=frame.drop(frame.index[-1])
    if change=="duplicate": frame=pd.concat([frame,frame.iloc[[-1]]])
    if change=="incomplete": frame.loc[frame.index[-1],"is_complete"]=False
    if change=="unknown": frame["is_complete"]=frame.is_complete.astype("boolean"); frame.loc[frame.index[-1],"is_complete"]=pd.NA
    if change=="negative": frame.loc[frame.index[-1],"entry_count"]=-1
    if change=="float": frame["entry_count"]=frame.entry_count.astype(float)
    with pytest.raises(ValueError): prepare_demand(frame,config)


@pytest.mark.parametrize("change", ["missing_scenario","duplicate","negative","nan"])
def test_pareto_rejects_incomplete_or_invalid_matrix(change):
    data=rows()
    if change=="missing_scenario": data=[r for r in data if r["scenario"]!="S3"]
    if change=="duplicate": data.append(copy.deepcopy(data[0]))
    if change=="negative": data[0]["train_km"]="-1"
    if change=="nan": data[0]["train_km"]="NaN"
    with pytest.raises(ValueError): pareto_frontier(data,["train_km"])


def test_configured_waiting_policy_cannot_accept_null_as_passed():
    data,_,profiles=toy_inputs(); data["plans"][0]["trips"]=[]
    policy=load_policy(); policy["max_boarded_p95_wait_seconds"]=600
    out=run_variant(data,demand(),profiles,policy,decision_second=0,main_end_second=260)
    assert out["checks"]["waiting_policy"]["status"]=="unknown"


def test_full_waiting_policy_is_not_applicable_without_any_demand():
    data,_,profiles=toy_inputs(); policy=load_policy(); policy["max_boarded_p95_wait_seconds"]=600
    out=run_variant(data,[],profiles,policy,decision_second=0,main_end_second=260)
    assert out["checks"]["waiting_policy"]["status"]=="not_applicable"


def test_full_period_passenger_balance_and_waiting_identity():
    data,_,profiles=toy_inputs()
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=0,main_end_second=260)
    assert out["full_metrics"]["balance"]["passed"] and out["full_metrics"]["waiting_identity_passed"]
    assert out["full_metrics"]["balance"]["entered"]==5 and out["full_metrics"]["balance"]["completed"]==5


def test_cli_failure_preserves_old_results_and_invalidates_owned_success(tmp_path):
    from decision_engine import main,MODEL_VERSION
    bad=tmp_path.parent/"bad_policy.json"; bad.write_text("{}")
    (tmp_path/"build_manifest.json").write_text(json.dumps({"model_version":MODEL_VERSION,"status":"success","outputs":{}}))
    assert main(["run","--output",str(tmp_path),"--policy",str(bad)])==2
    assert json.loads((tmp_path/"build_manifest.json").read_text())["status"]=="error"


def test_each_regular_release_keeps_its_own_warmed_state_and_adapter_boundaries():
    data,_,profiles=toy_inputs()
    window,_=build_regular_plan(toy_config(),2,phase_offset=100,end_second=520)
    window_motion=simulate(window,"line1-baseline")
    out=run_variant(data,demand(),profiles,load_policy(),decision_second=100,main_end_second=260,window_motion=window_motion)
    full=out["full_metrics"]
    assert full["initial_passengers"]==5 and full["initial_queued"]==1 and full["initial_onboard"]==4
    assert full["entered"]==0 and full["completed"]==5
    assert full["boarded"]==1 and full["mean_full_wait_seconds"]=="129"
    assert full["waiting_passenger_seconds"]==30 and full["full_boarded_wait_passenger_seconds"]==129
    assert full["initial_queue_age_passenger_seconds"]==99
    assert full["balance"]["passed"] and full["waiting_identity_passed"]
    resources=resource_report(window_motion)
    assert resources["distance_status"]=="partial" and resources["mileage_complete"] is False
    assert resources["state_seconds"]["moving"]==760


def test_processing_release_order_does_not_change_metrics_or_end_states():
    original=demand(); before=copy.deepcopy(original)
    results=[]
    for order in ((1,2),(2,1)):
        snapshot={}
        for count in order:
            data,_,profiles=toy_inputs(count)
            out=run_variant(data,original,profiles,load_policy(),decision_second=0,main_end_second=260)
            snapshot[count]=(out["main"]["metrics"],out["tail"]["checkpoint"],out["economics"])
        results.append(snapshot)
    assert results[0]==results[1] and original==before


@pytest.mark.parametrize("count", [1,2])
def test_combined_windows_match_continuous_run_including_queue_peaks(count):
    data,_,profiles=toy_inputs(count)
    groups=demand()+[{"group_id":"late", "turnstile_second":261,"platform_second":261,
        "origin":"A","count":1,"destinations":[{"station_id":"B","count":1}]}]
    out=run_variant(data,groups,profiles,load_policy(),decision_second=0,main_end_second=260)
    continuous=out["runner"].run()
    assert continuous["checkpoint"]==out["tail"]["checkpoint"]
    for key in ("entered","boarded","completed","waiting_passenger_seconds","queued","onboard",
                "approaching","max_queue","max_platform_direction_queue","max_car_load",
                "normal_threshold_exceedances","mean_full_wait_seconds","p95_full_wait_seconds"):
        assert out["full_metrics"][key]==continuous["metrics"][key],key
