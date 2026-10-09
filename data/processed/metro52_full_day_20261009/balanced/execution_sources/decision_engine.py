"""GOAL 4.1: compare regular fleet releases, without an operational transition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pandas as pd

from economics import compare, difference, exact_sum, load_costs, number, priced, ratio, write_json, write_text
from line1 import (ROOT, assign_destinations, car_profiles, check_output as check_line1, largest_remainders,
                   load_config, minute_groups, select_period, write_checkpoint, write_csv)
from passenger_groups import GroupedPassengerSimulation
from simulation import digest, fields_only, integer, simulate, to_evaluation


MODEL_VERSION = "regular-release-comparison-v1"
PLAN_ID = "line1-baseline"
POLICY_PATH = ROOT / "config/decision_engine/policy.json"
SCENARIOS = ("S1", "S2", "S3")
CRITERIA = {"train_km", "waiting_passenger_seconds", "p95", "queued_main", "queue_age_main",
            "queued_end", "queue_age_end", "threshold_observations"}
COMMON_FILES={"variants.json","period.json","assumptions.json","policy.json","minute_entries.csv","baseline_original.json",
    "table.csv","table.json","economic_comparison.json","pareto.json","summary.md","result_card.json","checks.json"}
VARIANT_FILES={"plan.json","movement_full.json","movement_evaluation.json","schedule.csv","headways.json","resources.json"}
OD_FILES={"metrics.json","warmup_checkpoint.json.gz","final_checkpoint.json.gz","queues_main.csv","queues_end.csv",
          "approaching_end.csv","queue_cells.json","car_load_events.csv","economics.json"}


def load_policy(path=None):
    policy=json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))
    fields_only(policy,("schema_version","policy_version","comparison_mode","warmup_start","main_start","main_end",
        "continuation_end","scenario_ids","fleet_offsets","max_boarded_p95_wait_seconds","max_active_wait_seconds",
        "prevent_queue_growth","optional_waiting_policy_source","dominance_criteria","dominance_scope","normal_load_source",
        "operational_transition_modeled"))
    if (policy["schema_version"]!="1.0" or policy["comparison_mode"]!="retrospective_planned_release"
            or policy["scenario_ids"]!=list(SCENARIOS) or policy["fleet_offsets"]!=[-2,-1,0,1,2]
            or policy["operational_transition_modeled"] is not False):
        raise ValueError("policy must compare five regular releases across S1/S2/S3")
    if any(policy[k]!=v for k,v in {"warmup_start":"06:00","main_start":"08:00","main_end":"10:00","continuation_end":"12:00"}.items()):
        raise ValueError("comparison uses common warmup 06–08, main 08–10 and continuation 10–12")
    for key in ("max_boarded_p95_wait_seconds","max_active_wait_seconds"):
        if policy[key] is not None: integer(policy[key],key,1)
    if type(policy["prevent_queue_growth"]) is not bool: raise ValueError("queue growth policy must be boolean")
    criteria=policy["dominance_criteria"]
    if not isinstance(criteria,list) or not criteria or len(criteria)!=len(set(criteria)) or set(criteria)-CRITERIA:
        raise ValueError("unsupported dominance criteria")
    return policy


def build_regular_plan(config,count,*,phase_offset=0,end_second=21600,terminal_holds=(0,0)):
    """Quantize absolute rational phases once; every train keeps its exact cycle."""
    integer(count,"train count",1); integer(phase_offset,"phase offset"); integer(end_second,"end second",1)
    if len(terminal_holds)!=2: raise ValueError("two terminal holds are required")
    for hold in terminal_holds: integer(hold,"terminal hold")
    if count>53: raise ValueError(f"{count} exceeds organizer maximum 53; readiness is unknown")
    if phase_offset>=end_second: raise ValueError("initial phase is outside comparison period")
    ids=config["station_ids"]; sections=len(ids)-1
    if sections<1 or len(set(ids))!=len(ids): raise ValueError("ordered line requires distinct stations")
    for key in ("travel_seconds","turnaround_seconds","length_m"): integer(config[key],key,1)
    integer(config["stop_seconds"],"stop seconds")
    lengths=largest_remainders(config["length_m"],[1]*sections)
    if not all(lengths): raise ValueError("segment lengths must remain positive")
    stations=[{"station_id":s,"kind":"line","stop_seconds":config["stop_seconds"] if 0<i<sections else 0,
               "turnaround_seconds":config["turnaround_seconds"]+terminal_holds[int(i==sections)] if i in (0,sections) else 0} for i,s in enumerate(ids)]
    tracks=[]
    for i,length in enumerate(lengths):
        for direction,a,b in (("up",ids[i],ids[i+1]),("down",ids[i+1],ids[i])):
            tracks.append({"track_id":f"P{i+1:02}_{direction}","from_station":a,"to_station":b,
                "length_km":number(Decimal(length)/1000),"travel_seconds":config["travel_seconds"]})
    lookup={t["track_id"]:t for t in tracks}
    routes=[[f"P{i:02}_up" for i in range(1,sections+1)],[f"P{i:02}_down" for i in range(sections,0,-1)]]
    half=sections*config["travel_seconds"]+(sections-1)*config["stop_seconds"]+config["turnaround_seconds"]
    halves=[half+terminal_holds[1],half+terminal_holds[0]]
    cycle=sum(halves)
    if cycle//count<113: raise ValueError(f"minimum quantized cyclic headway {cycle//count} < 113 seconds")
    phases=[i*cycle//count for i in range(count)]
    legs=[]
    for direction,route in enumerate(routes):
        at=halves[0] if direction else 0
        for j,track in enumerate(route):
            arrival=at+config["travel_seconds"]
            wait=config["turnaround_seconds"]+terminal_holds[1-direction] if j==sections-1 else config["stop_seconds"]
            legs.append({"track_id":track,"departure":at,"arrival":arrival,"ready":arrival+wait,"direction":direction,"index":j})
            at=arrival+wait
    trains,trips,history=[],[],[]
    duration=end_second-phase_offset
    for i,original_phase in enumerate(phases):
        train_id=f"T{i+1:02}"; absolute=original_phase+phase_offset; phase=absolute%cycle
        leg=next(l for l in legs if l["departure"]<=phase<l["ready"])
        route=routes[leg["direction"]]
        train={"train_id":train_id,"train_type":config["train_type"],"cars":len(config["car_types"]),"ready_after_seconds":0}
        if phase==leg["departure"]:
            train.update(state="standby",location=lookup[leg["track_id"]]["from_station"])
            departure,remaining=0,route[leg["index"]:]
        elif phase<leg["arrival"]:
            train.update(state="moving",location=leg["track_id"],initial_movement={"arrival_second":leg["arrival"]-phase,"passenger":True})
            departure,remaining=leg["ready"]-phase,route[leg["index"]+1:]
        else:
            train.update(state="turnaround" if leg["index"]==sections-1 else "stop",
                location=lookup[leg["track_id"]]["to_station"],ready_after_seconds=leg["ready"]-phase)
            departure,remaining=leg["ready"]-phase,route[leg["index"]+1:]
        trains.append(train); half_id=2*(absolute//cycle)+int(phase>=halves[0]); direction=leg["direction"]
        if not remaining: direction=1-direction; half_id+=1; remaining=routes[direction]
        while departure<duration:
            trips.append({"trip_id":f"{train_id}:H{half_id:04}","train_id":train_id,"departure_second":departure,"route":remaining,"passenger":True})
            departure+=len(remaining)*config["travel_seconds"]+(len(remaining)-1)*config["stop_seconds"]+config["turnaround_seconds"]+terminal_holds[1-direction]
            direction=1-direction; half_id+=1; remaining=routes[direction]
    for leg in legs:
        times=[leg["departure"]-((p+phase_offset)%cycle) for p in phases]
        history.append({"track_id":leg["track_id"],"departure_second":max(t if t<0 else t-cycle for t in times)})
    start=datetime.fromisoformat(config["date"]+"T06:00:00").replace(tzinfo=ZoneInfo(config["timezone"]))+timedelta(seconds=phase_offset)
    data={"schema_version":"1.0","source":"GOAL 3.3 assumed periodic plan, historical entries are separate",
        "period_start":start.isoformat(),"period_end":(start+timedelta(seconds=duration)).isoformat(),
        "stations":stations,"tracks":tracks,"trains":trains,"plans":[{"plan_id":PLAN_ID,"trips":trips}],
        "previous_departures":history,"max_trains_on_line":53}
    gaps=[b-a for a,b in zip(phases,phases[1:]+[cycle])]
    metadata={"train_count":count,"cycle_seconds":cycle,"phase_seconds":phases,
        "phase_precision_seconds":1,"phase_rule":"floor(i*cycle/N), then exact cycle repeats; error less than one second",
        "theoretical_mean_headway_seconds":number(ratio(Decimal(cycle),Decimal(count))),
        "cyclic_minimum_seconds":min(gaps),"cyclic_maximum_seconds":max(gaps),"timing_status":"assumed"}
    if any(terminal_holds): metadata["extra_terminal_hold_seconds"]=list(terminal_holds)
    return data,metadata


def headway_report(data,motion):
    by_track={t["track_id"]:[] for t in data["tracks"]}
    initial={r["track_id"]:r["departure_second"] for r in data.get("previous_departures",[])}
    for event in motion["events"]:
        if event["event_type"]=="departure": by_track[event["location"]].append(event["second"])
    result={}; all_gaps=[]
    for track,departures in by_track.items():
        times=sorted(set(([initial[track]] if track in initial else [])+departures))
        gaps=[b-a for a,b in zip(times,times[1:])]
        if not gaps: raise ValueError(f"no observed headway on {track}")
        if min(gaps)<113: raise ValueError(f"headway on {track}: {min(gaps)} < 113")
        result[track]={"minimum_seconds":min(gaps),"maximum_seconds":max(gaps),
            "mean_seconds":number(ratio(Decimal(sum(gaps)),Decimal(len(gaps)))),"intervals":len(gaps)}
        all_gaps.extend(gaps)
    return {"minimum_seconds":min(all_gaps),"maximum_seconds":max(all_gaps),
        "mean_seconds":number(ratio(Decimal(sum(all_gaps)),Decimal(len(all_gaps)))),"by_track":result,
        "scope":"observed departures across all directed tracks, including preceding boundary departure"}


def prepare_demand(frame,config):
    _,period=select_period(frame,config["station_ids"],config["date"])
    start=pd.Timestamp(period["start"]); end=start+pd.Timedelta(hours=6)
    selected=frame[(frame.interval_start>=start)&(frame.interval_start<end)].copy()
    expected={(s,start+pd.Timedelta(minutes=15*i)) for s in config["station_ids"] for i in range(24)}
    if (set(zip(selected.station_id,selected.interval_start))!=expected or len(selected)!=len(expected)
            or not pd.api.types.is_bool_dtype(selected.is_complete.dtype) or not selected.is_complete.notna().all()
            or not selected.is_complete.all() or not pd.api.types.is_integer_dtype(selected.entry_count.dtype)
            or pd.api.types.is_bool_dtype(selected.entry_count.dtype) or not selected.entry_count.notna().all()
            or not (selected.entry_count>=0).all()
            or not (selected.interval_end-selected.interval_start==pd.Timedelta(minutes=15)).all()):
        raise ValueError("historical continuation to 12:00 lacks complete, unique source coverage")
    groups=minute_groups(selected,config["station_ids"],start,end,config["platform_delay_seconds"])
    if sum(g["count"] for g in groups)!=int(selected.entry_count.sum()): raise ValueError("source allocation changed total entries")
    period.update(main_end=period["end"],end=end.isoformat(),source_rows=len(selected),future_demand_mode="retrospective observations; no forecasting")
    return {s:assign_destinations(groups,config["station_ids"],s) for s in SCENARIOS},groups,period


def resource_report(motion):
    evaluation=to_evaluation(motion)
    out={k:number(q.amount) for k,q in evaluation.quantities.items() if k.endswith("_km")}
    out["state_seconds"]={s:sum(t["state_seconds"][s] for t in motion["transport_work"]) for s in ("moving","stop","turnaround","standby")}
    out.update(mileage_complete=motion["mileage_complete"],obligations_closed=motion["obligations_closed"],
        distance_status=evaluation.quantities["train_km"].status,distance_scope="arrival ledger (start,end]; boundary completeness is preserved",
        energy_kWh=None,paid_person_hours=None,maneuver_cost_RUB=None,release_authorized=False)
    return out


def _economic_report(motion,articles):
    evaluation=to_evaluation(motion)
    period={"start":evaluation.period_start,"end":evaluation.closure_end}
    rows=[{"name":a.name,"quantity_key":a.quantity_key,"accounting_kind":a.accounting_kind,**priced(evaluation.quantities[a.quantity_key],a,period)} for a in articles]
    cash=[r for r in rows if r["accounting_kind"]=="cash"]
    missing=sorted(set(evaluation.required_cash_articles)-{r["name"] for r in cash})
    full=bool(cash) and not missing and all(r["amount"] is not None for r in cash)
    return {"cash_total_RUB":number(exact_sum(Decimal(r["amount"]) for r in cash)) if full else None,
        "known_cash_subtotal_RUB":number(exact_sum(Decimal(r["known_subtotal"]) for r in cash if r["known_subtotal"] is not None)),
        "money_status":"complete_configured_articles" if full else "unknown_or_partial","savings":None,
        "articles":rows,"unknown_articles":[r["name"] for r in rows if r["amount"] is None],"missing_required_articles":missing,
        "movement_obligations_closed":evaluation.obligations_closed,"unmodeled_transition_costs":"unknown; no transition is simulated",
        "release_authorized":False}


def _full_metrics(main,tail,initial):
    state=tail["checkpoint"]
    histogram={int(w):n-initial["wait_histogram"].get(w,0) for w,n in state["wait_histogram"].items()
               if n-initial["wait_histogram"].get(w,0)}
    boarded=sum(histogram.values()); full_wait=sum(w*n for w,n in histogram.items())
    rank=(95*boarded+99)//100; cumulative=0; p95=None
    for wait,count in sorted(histogram.items()):
        cumulative+=count
        if cumulative>=rank: p95=wait; break
    result=copy.deepcopy(tail["metrics"])
    for key in ("entered","platform_arrived","boarded","completed","waiting_passenger_seconds"):
        result[key]=main["metrics"][key]+tail["metrics"][key]
    result.update(full_boarded_wait_passenger_seconds=full_wait,
        mean_full_wait_seconds=number(ratio(Decimal(full_wait),Decimal(boarded))) if boarded else None,
        p95_full_wait_seconds=p95,max_car_load=max(main["metrics"]["max_car_load"],tail["metrics"]["max_car_load"]),
        normal_threshold_exceedances=main["metrics"]["normal_threshold_exceedances"]+tail["metrics"]["normal_threshold_exceedances"],
        max_queue=max(main["metrics"]["max_queue"],tail["metrics"]["max_queue"]),
        max_platform_direction_queue=max(main["metrics"]["max_platform_direction_queue"],tail["metrics"]["max_platform_direction_queue"]),
        max_active_wait_seconds=max([result["max_queue_age_seconds"]]+list(histogram)),
        initial_passengers=main["metrics"]["initial_passengers"],wait_scope="full platform waiting age of boardings in [main_start,continuation_end)")
    for key in ("initial_approaching","initial_queued","initial_onboard","initial_queue_age_passenger_seconds"):
        result[key]=main["metrics"][key]
    result["balance"]={"initial":result["initial_passengers"],"entered":result["entered"],"completed":result["completed"],
        "queued":result["queued"],"onboard":result["onboard"],"approaching":result["approaching"]}
    result["balance"]["passed"]=(result["initial_passengers"]+result["entered"]==
        result["completed"]+result["queued"]+result["onboard"]+result["approaching"])
    result["waiting_identity_passed"]=(result["waiting_passenger_seconds"]==full_wait+
        result["residual_queue_age_passenger_seconds"]-result["initial_queue_age_passenger_seconds"])
    return result


def queue_cells(result):
    counts=Counter()
    for row in result["queues"]: counts[f"{row['origin']}:{row['direction']}"]+=row["count"]
    return dict(sorted(counts.items()))


def check_service(main,tail,full,profiles,policy,baseline_queues=None):
    initial=main["initial_car_loads"]
    initial_exceeded=sum(n>car["normal_threshold"] for t,loads in initial.items() for n,car in zip(loads,profiles[t]))
    physical=all(n<=car["max_capacity"] for t,loads in initial.items() for n,car in zip(loads,profiles[t]))
    physical=physical and all(r["load"]<=r["max_capacity"] for out in (main,tail) for r in out["load_events"])
    normal=initial_exceeded==0 and full["normal_threshold_exceedances"]==0
    checks={"physical_capacity":{"status":"passed" if physical else "failed"},
        "normal_load":{"status":"passed" if normal else "failed","initial_exceeded_cars":initial_exceeded,
            "exceedance_observations":full["normal_threshold_exceedances"],"scope":"per car; initial state and observations after motion events"}}
    limits=[policy["max_boarded_p95_wait_seconds"],policy["max_active_wait_seconds"]]
    wait_status="not_configured"; reasons=[]
    if any(v is not None for v in limits):
        wait_status="passed"
        if limits[0] is not None:
            value=full["p95_full_wait_seconds"]
            if value is None:
                wait_status="not_applicable" if full["entered"]==full["initial_passengers"]==0 else "unknown"
                reasons.append("no boardings; queued and approaching people remain explicitly reported")
            elif value>limits[0]: wait_status="failed"; reasons.append("configured P95 limit exceeded")
        if limits[1] is not None and full["max_active_wait_seconds"]>limits[1]:
            wait_status="failed"; reasons.append("configured active waiting age exceeded")
    checks["waiting_policy"]={"status":wait_status,"reasons":reasons,"source":"optional demonstration policy; not organizer requirements"}
    growth=[]
    if policy["prevent_queue_growth"] and baseline_queues is not None:
        for label,out in (("main",main),("end",tail)):
            cells=queue_cells(out); base=baseline_queues[label]
            growth.extend({"boundary":label,"station_direction":k,"candidate":n,"baseline":base.get(k,0)} for k,n in cells.items() if n>base.get(k,0))
    checks["queue_policy"]={"status":"not_configured" if not policy["prevent_queue_growth"] else
        "unknown" if baseline_queues is None else "failed" if growth else "passed","increases":growth}
    checks.update(platform_capacity={"status":"unknown"},infrastructure_and_readiness={"status":"unknown"},release_authorized=False)
    return checks


def run_variant(data,groups,profiles,policy,*,decision_second=7200,main_end_second=14400,window_motion=None,baseline_queues=None,articles=None):
    runner=GroupedPassengerSimulation(data,PLAN_ID,groups,profiles)
    if not 0<=decision_second<main_end_second<runner.duration: raise ValueError("comparison requires warmup/main/continuation boundaries")
    warm=runner.run(end_second=decision_second) if decision_second else None
    initial=warm["checkpoint"] if warm else runner._initial()
    main=runner.run(start_second=decision_second,end_second=main_end_second,checkpoint=warm["checkpoint"] if warm else None)
    tail=runner.run(start_second=main_end_second,checkpoint=main["checkpoint"])
    full=_full_metrics(main,tail,initial)
    if window_motion is None:
        if decision_second: raise ValueError("resource evaluation needs the actual movement window from main_start")
        window_motion=runner.motion
    articles=list(articles) if articles is not None else load_costs(ROOT/"config/costs/default.json")
    return {"runner":runner,"warm":warm,"main":main,"tail":tail,"full_metrics":full,
        "checks":check_service(main,tail,full,profiles,policy,baseline_queues),
        "economics":_economic_report(window_motion,articles),"window_motion":window_motion}


def pareto_frontier(rows,criteria):
    scenarios=list(SCENARIOS); ids=sorted({r["variant_id"] for r in rows})
    if {r["scenario"] for r in rows}!=set(SCENARIOS): raise ValueError("all three OD scenarios are required")
    indexed={(r["variant_id"],r["scenario"]):r for r in rows}
    if len(indexed)!=len(rows) or len(rows)!=len(ids)*len(scenarios): raise ValueError("incomplete or duplicate variant/scenario matrix")
    vectors={v:[indexed[v,s][k] for s in scenarios for k in criteria] for v in ids}
    for vector in vectors.values():
        for value in vector:
            if value is not None and (not Decimal(str(value)).is_finite() or Decimal(str(value))<0):
                raise ValueError("dominance criteria require finite nonnegative values or null")
    def dominates(a,b):
        if any(x is None or y is None for x,y in zip(a,b)): return False
        a=[Decimal(str(x)) for x in a]; b=[Decimal(str(x)) for x in b]
        return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))
    excluded={v:[other for other in ids if other!=v and dominates(vectors[other],vectors[v])] for v in ids}
    return {"non_dominated":[v for v in ids if not excluded[v]],"dominated_by":excluded,"criteria":criteria,
        "OD_scope":"every criterion in every scenario, without averaging","unknown_values_prevent_dominance":True,
        "economic_optimum_claimed":False,"operational_recommendation":False}


def safe_output(output):
    output=Path(output).resolve()
    protected=[ROOT/p for p in ("data/raw","docs","tests","config","examples","metro_data",".venv")]
    protected.extend(ROOT/"data/processed"/name for name in ("metro","metro_repeat","economics","economics_repeat",
        "economics_audit","simulation","simulation_repeat","simulation_audit","passengers","passengers_repeat",
        "passengers_verification","line1","line1_repeat","line1_verification","line1_audit","decision_verification"))
    protected.extend(p for p in (ROOT/"data/processed").iterdir() if p.is_dir()
        and p.name not in ("decision","decision_repeat") and p.resolve()!=output)
    if ROOT.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("regular comparison output overlaps protected results or sources")
    if output.exists() and any(output.iterdir()):
        path=output/"build_manifest.json"
        if not path.is_file(): raise ValueError("foreign output directory")
        manifest=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest,dict) or manifest.get("model_version")!=MODEL_VERSION: raise ValueError("foreign output manifest")
        paths=list(output.rglob("*"))
        if any(not p.resolve().is_relative_to(output) or p.is_file() and p.stat().st_nlink>1 for p in paths): raise ValueError("output contains foreign links")
        allowed=set(manifest.get("outputs",{}))|{"build_manifest.json"}
        known=COMMON_FILES|{"build_manifest.json"}|{f"demand/{s}.json.gz" for s in SCENARIOS}
        known|={f"N{n:02}/{name}" for n in range(1,54) for name in VARIANT_FILES}
        known|={f"N{n:02}/{s}/{name}" for n in range(1,54) for s in SCENARIOS for name in OD_FILES}
        if allowed-known: raise ValueError("output manifest declares unsupported files")
        if {p.relative_to(output).as_posix() for p in paths if p.is_file()}-allowed: raise ValueError("foreign files in comparison output")
    return output


def input_files(policy_path):
    original=json.loads((ROOT/"data/processed/line1/build_manifest.json").read_text(encoding="utf-8"))
    files={Path(p).resolve() for p in original["inputs"]}
    files.update((ROOT/p).resolve() for p in ("decision_engine.py","data/processed/line1/build_manifest.json",
        "data/processed/line1/scenario_comparison.json","data/processed/line1/movement_input.json"))
    files.update((ROOT/f"data/processed/line1/{s}/{name}").resolve() for s in SCENARIOS
        for name in ("metrics.json","warmup_checkpoint.json.gz","final_checkpoint.json.gz"))
    files.add(Path(policy_path).resolve())
    return sorted(files)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluation(out,variant,groups,main_end):
    evaluation=to_evaluation(out["window_motion"])
    warm=out["warm"]["checkpoint"] if out["warm"] else out["runner"]._initial()
    state={k:v for k,v in warm.items() if k not in ("context_id","sha256")}
    final=out["full_metrics"]
    closed=evaluation.obligations_closed and not any(final[k] for k in ("queued","onboard","approaching"))
    start=datetime.fromisoformat(out["runner"].motion["period_start"])
    return replace(evaluation,plan_id=variant,demand_id=digest(groups),
        initial_state_id=digest({"movement":evaluation.initial_state_id,"passengers":state}),
        period_end=(start+timedelta(seconds=main_end)).isoformat(),obligations_closed=closed)


def _evaluation_json(evaluation):
    return {k:getattr(evaluation,k) for k in ("plan_id","demand_id","initial_state_id","period_start","period_end","closure_end",
        "model_version","evaluation_mode","obligations_closed","constraint_status","required_cash_articles")}|{
        "quantities":{k:q.as_dict() for k,q in evaluation.quantities.items()}}


def _publish(stage,output):
    from tempfile import NamedTemporaryFile
    for path in sorted(stage.rglob("*")):
        if not path.is_file(): continue
        target=output/path.relative_to(stage); target.parent.mkdir(parents=True,exist_ok=True)
        if not target.resolve().is_relative_to(output): raise ValueError("published file escapes output")
        with NamedTemporaryFile(dir=target.parent,delete=False) as handle:
            temporary=Path(handle.name); handle.write(path.read_bytes())
        try: temporary.replace(target)
        finally: temporary.unlink(missing_ok=True)


def run_experiment(output=None,policy_path=None):
    from metro_data.build import check as check_data
    output=safe_output(output or ROOT/"data/processed/decision")
    policy_path=Path(policy_path or POLICY_PATH).resolve()
    if policy_path.is_relative_to(output): raise ValueError("policy input overlaps comparison output")
    output.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter(); performance={}
    try:
        policy=load_policy(policy_path); config=load_config()
        check_line1(ROOT/"data/processed/line1"); check_data(ROOT,ROOT/"data/processed/metro")
        consumed=input_files(policy_path); before={str(p):_sha(p) for p in consumed}
        original=json.loads((ROOT/"data/processed/line1/scenario_comparison.json").read_text(encoding="utf-8"))
        frame=pd.read_parquet(ROOT/"data/processed/metro/entries_station_15m.parquet")
        by_s,minutes,period=prepare_demand(frame,config); config=copy.deepcopy(config); config["date"]=period["date"]
        articles=load_costs(ROOT/"config/costs/default.json")
        counts=[config["train_count"]+d for d in policy["fleet_offsets"]]
        order=[config["train_count"]]+sorted(n for n in counts if n!=config["train_count"])
        rows,variants,base_queues,base_evaluations=[],[],{},{}
        economic_comparisons={}; baseline_regression={}; demand_hashes={s:digest(by_s[s]) for s in SCENARIOS}
        performance["prepare_seconds"]=time.perf_counter()-began
        with TemporaryDirectory(prefix="regular-release-") as temporary:
            stage=Path(temporary)
            write_json(stage/"baseline_original.json",original)
            for scenario in SCENARIOS: write_checkpoint(stage/f"demand/{scenario}.json.gz",by_s[scenario])
            write_csv(stage/"minute_entries.csv",minutes)
            for count in order:
                variant=f"N{count:02}"; started=time.perf_counter()
                try:
                    movement,parameters=build_regular_plan(config,count)
                    window_data,_=build_regular_plan(config,count,phase_offset=7200)
                    window_motion=simulate(window_data,PLAN_ID)
                    main_data,_=build_regular_plan(config,count,phase_offset=7200,end_second=14400)
                    main_motion=simulate(main_data,PLAN_ID)
                except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
                    variants.append({"variant_id":variant,"train_count":count,"status":"rejected","reason":str(exc)})
                    if count==config["train_count"]: raise ValueError(f"baseline release rejected: {exc}") from exc
                    continue
                folder=stage/variant; folder.mkdir()
                profiles=car_profiles(config,[t["train_id"] for t in movement["trains"]])
                resources=resource_report(window_motion); main_resources=resource_report(main_motion)
                full_motion=None; variant_checks=[]
                for scenario in SCENARIOS:
                    od_started=time.perf_counter()
                    out=run_variant(movement,by_s[scenario],profiles,policy,window_motion=window_motion,
                        baseline_queues=base_queues.get(scenario),articles=articles)
                    if digest(by_s[scenario])!=demand_hashes[scenario]: raise ValueError("comparison mutated common demand")
                    full_motion=out["runner"].motion
                    if full_motion["checks"]["peak_trains_on_line"]!=count: raise ValueError("actual fleet differs from declared release")
                    if count==config["train_count"]:
                        base_queues[scenario]={"main":queue_cells(out["main"]),"end":queue_cells(out["tail"])}
                        out["checks"]=check_service(out["main"],out["tail"],out["full_metrics"],profiles,policy,base_queues[scenario])
                        old=next(r for r in original if r["scenario"]==scenario); m=out["main"]["metrics"]
                        baseline_regression[scenario]={"queue":m["queued"]==old["queue_end"],
                            "waiting":m["waiting_passenger_seconds"]==old["waiting_passenger_seconds"],
                            "mean":m["mean_full_wait_seconds"]==old["mean_full_wait_seconds"],
                            "p95":m["p95_full_wait_seconds"]==old["p95_full_wait_seconds"],
                            "peak_car":m["max_car_load"]==old["max_car_load"]}
                        if not all(baseline_regression[scenario].values()): raise ValueError("baseline passenger indicators changed")
                    evaluation=_evaluation(out,variant,by_s[scenario],14400)
                    if count==config["train_count"]: base_evaluations[scenario]=evaluation
                    try:
                        economic=compare(base_evaluations[scenario],evaluation,articles)
                    except ValueError as exc:
                        economic={"status":"not_comparable","reason":str(exc),"cash_savings_RUB":None,
                            "boundary_completeness_preserved":True,"initial_state_ids_are_not_overridden":True,
                            "release_authorized":False}
                    economic_comparisons[f"{variant}/{scenario}"]=economic
                    od_folder=folder/scenario; od_folder.mkdir()
                    write_json(od_folder/"metrics.json",{"scenario":scenario,"variant_id":variant,"warmup":out["warm"]["metrics"],
                        "main":out["main"]["metrics"],"continuation":out["tail"]["metrics"],"common_period":out["full_metrics"],
                        "checks":out["checks"],"demand_sha256":demand_hashes[scenario],"initial_state_scope":"own warmup for this regular release"})
                    write_checkpoint(od_folder/"warmup_checkpoint.json.gz",out["warm"]["checkpoint"])
                    write_checkpoint(od_folder/"final_checkpoint.json.gz",out["tail"]["checkpoint"])
                    columns=("group_id","origin","destination","direction","turnstile_second","platform_second","count","waiting_age_seconds")
                    write_csv(od_folder/"queues_main.csv",out["main"]["queues"],columns)
                    write_csv(od_folder/"queues_end.csv",out["tail"]["queues"],columns)
                    write_csv(od_folder/"approaching_end.csv",({k:g[k] for k in ("group_id","origin","turnstile_second","platform_second","count")} for g in out["tail"]["approaching_groups"]),
                        ("group_id","origin","turnstile_second","platform_second","count"))
                    write_json(od_folder/"queue_cells.json",{"main":queue_cells(out["main"]),"end":queue_cells(out["tail"])})
                    write_csv(od_folder/"car_load_events.csv",out["main"]["load_events"]+out["tail"]["load_events"])
                    write_json(od_folder/"economics.json",{**out["economics"],"evaluation":_evaluation_json(evaluation),"comparison":economic})
                    main=out["main"]["metrics"]; tail=out["tail"]["metrics"]; full=out["full_metrics"]
                    row={"variant_id":variant,"train_count":count,"scenario":scenario,"waiting_passenger_seconds":main["waiting_passenger_seconds"],
                        "mean_wait":main["mean_full_wait_seconds"],"p95":main["p95_full_wait_seconds"],"queued_main":main["queued"],
                        "queue_age_main":main["max_queue_age_seconds"],"onboard_main":main["onboard"],"approaching_main":main["approaching"],
                        "queued_end":tail["queued"],"queue_age_end":tail["max_queue_age_seconds"],"onboard_end":tail["onboard"],"approaching_end":tail["approaching"],
                        "max_car_load_main":main["max_car_load"],"max_car_load":full["max_car_load"],"threshold_observations":full["normal_threshold_exceedances"],
                        "initial_exceeded_cars":out["checks"]["normal_load"]["initial_exceeded_cars"],"normal_load_status":out["checks"]["normal_load"]["status"],
                        "physical_capacity_status":out["checks"]["physical_capacity"]["status"],"train_km":resources["train_km"],
                        "train_km_main":main_resources["train_km"],"passenger_train_km":resources["passenger_train_km"],
                        "nonpassenger_train_km":resources["nonpassenger_train_km"],"passenger_car_km":resources["passenger_car_km"],
                        "nonpassenger_car_km":resources["nonpassenger_car_km"],"money_status":out["economics"]["money_status"],
                        "cash_total_RUB":out["economics"]["cash_total_RUB"],"savings_RUB":None,"demand_sha256":demand_hashes[scenario],
                        "balance_passed":main["balance"]["passed"] and tail["balance"]["passed"] and full["balance"]["passed"]}
                    row.update({s+"_seconds":n for s,n in resources["state_seconds"].items()})
                    rows.append(row); variant_checks.append(out["checks"]["normal_load"]["status"]=="passed")
                    performance[f"{variant}_{scenario}_seconds"]=time.perf_counter()-od_started
                    del out
                headways=headway_report(movement,full_motion)
                for row in rows:
                    if row["variant_id"]==variant:
                        row.update(headway_min=headways["minimum_seconds"],headway_mean=headways["mean_seconds"],headway_max=headways["maximum_seconds"])
                write_json(folder/"plan.json",movement); write_json(folder/"movement_full.json",full_motion)
                write_json(folder/"movement_evaluation.json",window_motion); write_json(folder/"resources.json",{"main":main_resources,"common_period":resources})
                write_csv(folder/"schedule.csv",full_motion["events"]); write_json(folder/"headways.json",headways)
                variants.append({"variant_id":variant,"status":"evaluated","parameters":parameters,"train_count":count,
                    "headways":{k:v for k,v in headways.items() if k!="by_track"},"normal_load_passed_all_OD":all(variant_checks)})
                performance[f"{variant}_seconds"]=time.perf_counter()-started
                print(json.dumps({"variant":variant,"scenarios":3,"seconds":performance[f"{variant}_seconds"],
                    "normal_load_passed_all_OD":all(variant_checks)}),file=sys.stderr,flush=True)
            rows.sort(key=lambda r:(r["train_count"],r["scenario"])); variants.sort(key=lambda v:v["train_count"])
            baseline={r["scenario"]:r for r in rows if r["train_count"]==config["train_count"]}
            delta_keys=("train_km","passenger_train_km","nonpassenger_train_km","passenger_car_km","nonpassenger_car_km",
                "moving_seconds","stop_seconds","turnaround_seconds","standby_seconds","waiting_passenger_seconds",
                "queued_main","queued_end","max_car_load","threshold_observations")
            for row in rows:
                base=baseline[row["scenario"]]
                row.update({"delta_"+k:number(difference(Decimal(str(row[k])),Decimal(str(base[k])))) for k in delta_keys})
                row["delta_scope"]="candidate minus baseline; modeled ledger, not complete cost savings"
            frontier=pareto_frontier(rows,policy["dominance_criteria"])
            compliant=[v["variant_id"] for v in variants if v["status"]=="evaluated" and v["normal_load_passed_all_OD"]]
            checks={"baseline_regression":baseline_regression,"demand_identical_within_OD":all(len({r["demand_sha256"] for r in rows if r["scenario"]==s})==1 for s in SCENARIOS),
                "all_balances_passed":all(r["balance_passed"] for r in rows),"normal_compliant_all_OD":compliant,
                "future_demand_after_main_window_present":sum(g["count"] for g in minutes if g["turnstile_second"]>=14400)>0,
                "source_entries_06_12":sum(g["count"] for g in minutes),"minute_groups":len(minutes),
                "release_authorized":False,"all_constraints_verified":False}
            card={"status":"scenario_planned_release_comparison","normal_load_result":"passed_variants_exist" if compliant else "all_evaluated_variants_violate_normal_load",
                "normal_compliant_variants":compliant,"selected_operational_action":None,"economic_optimum":None,
                "money_statement":"Денежная экономия не определена","period":period,"pareto_non_dominated":frontier["non_dominated"],
                "reason":"regular releases have their own warmed states; no feasible operational transition is modeled",
                "transition_inputs_required":["ready and available fleet with maintenance and release history","reserve locations and readiness",
                    "depot and terminal tracks, switching and signaling constraints","entry/withdrawal routes and maneuver durations",
                    "actual current train and passenger state","transition schedule and staffing","energy, labor and maneuver cost models",
                    "passenger forecast available at decision time and its uncertainty"],"release_authorized":False}
            write_json(stage/"result_card.json",card); write_json(stage/"checks.json",checks); write_json(stage/"pareto.json",frontier)
            write_json(stage/"table.json",rows); write_csv(stage/"table.csv",rows)
            write_json(stage/"variants.json",variants); write_json(stage/"period.json",period); write_json(stage/"policy.json",policy)
            write_json(stage/"economic_comparison.json",economic_comparisons)
            write_json(stage/"assumptions.json",{"line_config":config,"comparison_policy":policy,
                "fleet_availability":"unknown; 53 is an infrastructure scenario limit, not available ready fleet",
                "initial_states":"own regular phases, empty passenger stocks at 06:00, same warmup duration",
                "historical_future_entries":"observations, not an available dispatch forecast","operational_transition_modeled":False,
                "transfer_inflow":"пересадочный приток не учтён","phase_quantization":"integer seconds; rational phases rounded once, no cumulative drift",
                "end_state_semantics":"nonfinal windows [a,b); final closing arrivals/waits included; passengers at future platform arrival remain on approach",
                "baseline_comparison_scope":"same queue/wait/load metrics; legacy 10:00 closes arrivals, continuation defers them to next window",
                "unknown_constraints":["platform capacity","signaling and switches","fleet readiness","actual applicability of car profile","maneuvers"]})
            lines=["# Сравнение планового выпуска","",f"{period['date']}, Europe/Moscow: разогрев 06:00–08:00, основное окно 08:00–10:00, продолжение до 12:00.",
                "","Исторический спрос продолжается после 10:00. Каждый регулярный график имеет своё состояние после одинакового разогрева. Оперативный переход не моделируется.",
                "","| Составов | OD | Интервал min–max, с | Очередь 10:00 | Очередь 12:00 | Ожидание 08–10, пасс·с | Среднее, с | P95, с | Пик вагона 08–12 | Пробег 08–12, поездо-км | Δ пробега |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                mean=f"{Decimal(r['mean_wait']):.2f}" if r["mean_wait"] is not None else "неизвестно"
                lines.append(f"| {r['train_count']} | {r['scenario']} | {r['headway_min']}–{r['headway_max']} | {r['queued_main']} | {r['queued_end']} | {r['waiting_passenger_seconds']} | {mean} | {r['p95']} | {r['max_car_load']} | {r['train_km']} | {r['delta_train_km']} |")
            lines.extend(["","Норма 120 проверена отдельно по каждому вагону во всех OD. "+("Прошли: "+", ".join(compliant) if compliant else "Все проверенные варианты нарушают норму."),
                "","Денежная экономия не определена. Статусы неполноты и незавершённых обязательств адаптера сохранены; разные исходные состояния не подменены одним идентификатором.",
                "","Недоминируемые варианты: "+", ".join(frontier["non_dominated"])+". Каждый критерий сравнивается отдельно во всех трёх OD; среднего по сценариям и общего балла нет.",
                "","Полное ожидание относится к севшим в основном окне, в том числе начавшим ждать до 08:00. Не севшие люди, возраст очереди, подходы и загрузка на границах сохранены отдельно.",
                "","Времена движения и OD условны. Физическая вместимость взята из выбранного профиля; применимость к реальному парку неизвестна. 53 состава не подтверждают наличие готового парка. Пересадочный приток не учтён."])
            write_text(stage/"summary.md","\n".join(lines)+"\n")
            if before!={str(p):_sha(p) for p in consumed}: raise ValueError("comparison inputs changed during calculation")
            manifest={"model_version":MODEL_VERSION,"status":"success","policy_input":str(policy_path),"inputs":before,
                "outputs":{p.relative_to(stage).as_posix():_sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
            _publish(stage,output); write_json(output/"build_manifest.json",manifest)
        performance.update(total_seconds=time.perf_counter()-began,output_files=sum(p.is_file() for p in output.rglob("*")),
            output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
        write_json(output.parent/f"{output.name}_performance.json",performance)
        return {"status":card["status"],"normal_compliant_variants":compliant,"non_dominated":frontier["non_dominated"],
            "cash_savings_RUB":None,"output":str(output),"performance":performance,"release_authorized":False}
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as exc:
        write_json(output/"build_manifest.json",{"model_version":MODEL_VERSION,"status":"error","error":str(exc)})
        raise


def check_output(output):
    output=Path(output).resolve(); manifest=json.loads((output/"build_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest,dict) or manifest.get("model_version")!=MODEL_VERSION or manifest.get("status")!="success":
        raise ValueError("no successful current comparison manifest")
    for name in ("inputs","outputs"):
        if not isinstance(manifest.get(name),dict) or not manifest[name]: raise ValueError("incomplete comparison manifest")
    policy=Path(manifest.get("policy_input",""))
    if not policy.is_absolute() or set(manifest["inputs"])!={str(p) for p in input_files(policy)}: raise ValueError("incomplete comparison input set")
    load_policy(policy)
    variants=json.loads((output/"variants.json").read_text(encoding="utf-8"))
    expected=COMMON_FILES|{f"demand/{s}.json.gz" for s in SCENARIOS}
    for v in variants:
        if v["status"]!="evaluated": continue
        expected|={f"{v['variant_id']}/{n}" for n in VARIANT_FILES}
        expected|={f"{v['variant_id']}/{s}/{n}" for s in SCENARIOS for n in OD_FILES}
    actual={p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if set(manifest["outputs"])!=expected or actual!=expected|{"build_manifest.json"}: raise ValueError("comparison artifact set changed")
    for name,h in manifest["inputs"].items():
        if not Path(name).is_absolute() or _sha(name)!=h: raise ValueError(f"input changed: {name}")
    for name,h in manifest["outputs"].items():
        if not (output/name).resolve().is_relative_to(output) or _sha(output/name)!=h: raise ValueError(f"output changed: {name}")
    return {"status":"passed","files":len(actual),"release_authorized":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","check")); parser.add_argument("--output",type=Path,default=ROOT/"data/processed/decision")
    parser.add_argument("--policy",type=Path,default=POLICY_PATH); args=parser.parse_args(argv)
    try:
        result=run_experiment(args.output,args.policy) if args.command=="run" else check_output(args.output)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True)); return 0
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as exc:
        print(f"decision_engine: {exc}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())
