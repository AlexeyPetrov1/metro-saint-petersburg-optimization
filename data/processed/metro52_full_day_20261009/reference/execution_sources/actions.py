"""Single-train changes over the existing movement and passenger interfaces."""

import copy
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from decision_engine import _economic_report
from dwell import compile_timing
from economics import load_costs
from passenger_groups import GroupedPassengerSimulation
from replay_metrics import cohort_waiting, overload_metrics, work_between
from simulation import simulate


def demo_depots(data,config):
    out=copy.deepcopy(data); registry={}
    for suffix,terminal in (("A",config["station_ids"][0]),("B",config["station_ids"][-1])):
        depot=f"DEPOT_{suffix}"
        out["stations"].append(dict(station_id=depot,kind="depot",stop_seconds=0,turnaround_seconds=10))
        for prefix,a,b in (("ENTRY",depot,terminal),("EXIT",terminal,depot)):
            out["tracks"].append(dict(track_id=f"{prefix}_{suffix}",from_station=a,to_station=b,length_km="1",travel_seconds=120))
    out["trains"].append(dict(train_id="R01",train_type=config["train_type"],cars=len(config["car_types"]),
        state="standby",location="DEPOT_A",ready_after_seconds=0))
    registry["R01"]=dict(ready=True,preparation_seconds=900,source="GOAL 4.2 explicit demonstration assumptions",
        readiness_status="assumed",infrastructure_status="unknown",maintenance_status="unknown",crew_status="unknown")
    return out,registry


def _append_service(data,train,terminal,second):
    plan=data["plans"][0]; stations={s["station_id"]:s for s in data["stations"]}
    up=sorted((t for t in data["tracks"] if t["track_id"].endswith("_up")),key=lambda t:t["track_id"])
    down=sorted((t for t in data["tracks"] if t["track_id"].endswith("_down")),key=lambda t:t["track_id"],reverse=True)
    routes={up[0]["from_station"]:up,down[0]["from_station"]:down}
    duration=int((datetime.fromisoformat(data["period_end"])-datetime.fromisoformat(data["period_start"])).total_seconds())
    index=0
    while second<duration:
        route=routes[terminal]
        plan["trips"].append(dict(trip_id=f"{train}:ACTION{second:06}:{index}",train_id=train,
            departure_second=second,route=[t["track_id"] for t in route],passenger=True))
        for track in route:
            station=stations[track["to_station"]]
            second+=track["travel_seconds"]+station["stop_seconds"]+station["turnaround_seconds"]
        terminal=route[-1]["to_station"]; index+=1


def expand_action(data,kind,train_id,as_of,registry):
    out=copy.deepcopy(data); action=dict(kind=kind,train_id=train_id,decision_second=as_of,
        planned_effect_second=as_of,operational_status="conditional_scenario",release_authorized=False)
    if kind=="KEEP": return out,action
    plan=out["plans"][0]; trains={t["train_id"]:t for t in out["trains"]}; tracks={t["track_id"]:t for t in out["tracks"]}
    stations={s["station_id"]:s for s in out["stations"]}
    if kind=="ADD":
        info=registry.get(train_id,{})
        if info.get("ready") is not True: raise ValueError("reserve is not explicitly ready in the scenario snapshot")
        train=trains[train_id]
        if train["state"]!="standby" or stations[train["location"]]["kind"]!="depot" or any(t["train_id"]==train_id for t in plan["trips"]):
            raise ValueError("reserve is unavailable or already assigned")
        entry=next((t for t in tracks.values() if t["from_station"]==train["location"]),None)
        if not entry: raise ValueError("entry route is missing")
        departure=as_of+info["preparation_seconds"]
        plan["trips"].append(dict(trip_id=f"{train_id}:ENTRY{as_of}",train_id=train_id,
            departure_second=departure,route=[entry["track_id"]],passenger=False))
        effect=departure+entry["travel_seconds"]+stations[entry["to_station"]]["turnaround_seconds"]
        _append_service(out,train_id,entry["to_station"],effect)
        action.update(planned_effect_second=effect,conditions=info)
    elif kind=="WITHDRAW":
        motion=simulate(data,plan["plan_id"])
        arrival=next((e for e in motion["events"] if e["train_id"]==train_id and e["second"]>=as_of
            and e["event_type"]=="arrival" and stations[e["location"]]["kind"]=="line"
            and stations[e["location"]]["turnaround_seconds"]),None)
        if not arrival: raise ValueError("no completed terminal trip within the horizon")
        from withdrawal_stabling import assign_routes
        destination=assign_routes(out,[train_id],{train_id:arrival["location"]})[train_id]
        ordered=sorted((t for t in plan["trips"] if t["train_id"]==train_id),key=lambda t:t["departure_second"])
        current=next((i for i,t in enumerate(ordered) if t["trip_id"]==arrival["trip_id"]),-1)
        removed={t["trip_id"] for t in ordered[current+1:]}
        plan["trips"]=[t for t in plan["trips"] if t["trip_id"] not in removed]
        ready=arrival["second"]+stations[arrival["location"]]["turnaround_seconds"]
        plan["trips"].append(dict(trip_id=f"{train_id}:EXIT{as_of}",train_id=train_id,
            departure_second=ready,route=destination['exit_route'],passenger=False))
        action.update(planned_effect_second=ready+destination['exit_seconds'], withdrawal_destination=destination,
            conditions={"terminal_trip_complete":True,"exit_route_status":"assumed","crew_status":"unknown","maintenance_status":"unknown"})
    elif kind=="DEFER_WITHDRAWAL":
        pending=next((t for t in plan["trips"] if t["train_id"]==train_id and t["departure_second"]>=as_of
            and not t["passenger"] and t["route"][-1].startswith("EXIT_")),None)
        if not pending: raise ValueError("no pending withdrawal to defer")
        plan["trips"].remove(pending)
        _append_service(out,train_id,tracks[pending["route"][0]]["from_station"],pending["departure_second"])
        action["planned_effect_second"]=pending["departure_second"]
    else: raise ValueError("unsupported single-train action")
    return out,action


def evaluate_candidates(data,groups,profiles,registry,snapshot,source,candidates,profile,seed):
    result=[]; start=snapshot["second"]
    for kind,train in candidates:
        candidate_id=f"{kind}:{train or 'plan'}"
        try:
            changed,action=expand_action(data,kind,train,start,registry)
            changed,timing=compile_timing(changed,data["plans"][0]["plan_id"],profile,seed,
                committed=data.get("leg_timing",[]),as_of=start)
            runner=GroupedPassengerSimulation(changed,data["plans"][0]["plan_id"],groups,profiles)
            state=runner.branch_checkpoint(snapshot,source)
            out=runner.run(start_second=start,checkpoint=state)
            effects={}
            trips={t["trip_id"]:t for p in changed["plans"] for t in p["trips"]}
            if kind=="ADD":
                for event in runner.motion["events"]:
                    if event["train_id"]==train and event["event_type"]=="departure" and trips[event["trip_id"]]["passenger"]:
                        origin=runner.tracks[event["location"]]["from_station"]
                        effects.setdefault(origin,event["second"])
            action.update(actual_effect_by_station=effects,
                actual_effect_second=min(effects.values()) if effects else action["planned_effect_second"] if kind=="KEEP" else None)
            overload=overload_metrics(runner,out["checkpoint"],start,runner.duration)
            physical=all(r["load"]<=r["max_capacity"] for r in out["load_events"])
            result.append(dict(candidate_id=candidate_id,status="evaluated",action=action,
                metrics={**out["metrics"],**overload},cohort=cohort_waiting(runner,out["checkpoint"],start,min(start+7200,runner.duration)),
                normal_load_passed=overload["exceeded_car_seconds"]==0,
                physical_passed=physical,resources=work_between(changed,runner.motion,start,runner.duration),
                economics=_economic_report(runner.motion,load_costs(Path(__file__).parent/"config/costs/default.json")),
                cash_delta_RUB=None,money_status="unknown; rates and closure are incomplete",timing=timing,
                _runner=runner,_output=out,_data=changed))
        except ValueError as exc:
            result.append(dict(candidate_id=candidate_id,status="rejected",action=dict(kind=kind,train_id=train),
                reason=str(exc),metrics=None,cash_delta_RUB=None))
    return result


def select_candidate(rows):
    feasible=[r for r in rows if r["status"]=="evaluated" and r["physical_passed"] and r["normal_load_passed"]]
    if not feasible:
        evaluated=[r for r in rows if r["status"]=="evaluated" and r["physical_passed"]]
        mitigation=min(evaluated,key=lambda r:(r["metrics"]["max_car_load"],r["metrics"]["waiting_passenger_seconds"],r["candidate_id"])) if evaluated else None
        return dict(status="no_feasible_candidate",selected_candidate_id=None,
            mitigation_candidate_id=mitigation["candidate_id"] if mitigation else None,release_authorized=False,
            scope="checked single-train candidates; no line-wide capacity claim")
    selected=min(feasible,key=lambda r:(Decimal(str(r["resources"]["train_km"])),r["metrics"]["waiting_passenger_seconds"],
        r["action"]["kind"]!="KEEP",r["candidate_id"]))
    return dict(status="conditional_resource_policy",selected_candidate_id=selected["candidate_id"],
        policy_version="minimum-arrival-ledger-train-km-v1",economic_optimality=False,
        cash_delta_RUB=None,release_authorized=False)
