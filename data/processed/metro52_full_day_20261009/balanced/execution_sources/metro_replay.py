"""GOAL 4.2: hourly medians, keyed dwell, snapshot choices and honest replay."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
from statistics import median

import pandas as pd

from actions import demo_depots,expand_action,evaluate_candidates
from decision_engine import SCENARIOS,build_regular_plan,headway_report,prepare_demand
from dwell import compile_timing,regular_plan_at_limit
from economics import write_json,write_text
from forecasting import hourly_median
from line1 import ROOT,car_profiles,load_config,write_checkpoint,write_csv
from metro_data.build import check as check_data
from passenger_groups import GroupedPassengerSimulation
from passengers import PassengerSimulation
from replay_metrics import cohort_waiting,overload_metrics,peak_exchange,work_between,movement_snapshot,queue_area
from rolling import decide_snapshot


VERSION="metro-replay-v1"
CONFIG=ROOT/"config/decision_engine/goal4_2.json"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_output(path):
    path=Path(path).resolve(); parent=(ROOT/"data/processed").resolve()
    if path.parent!=parent or not path.name.startswith("goal4_2") or path.name=="goal4_2_verification":
        raise ValueError("output must be a dedicated data/processed/goal4_2* directory")
    marker=path/"owner.json"
    if path.exists() and any(path.iterdir()) and (not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("owner")!=VERSION):
        raise ValueError("output belongs to another report")
    path.mkdir(parents=True,exist_ok=True); write_json(marker,dict(owner=VERSION))
    return path


def public(value): return {k:v for k,v in value.items() if not k.startswith("_")}


def save_passengers(folder,data,runner,out,start=7200,main_end=14400):
    folder.mkdir(parents=True,exist_ok=True); state=out["checkpoint"]
    summary=dict(metrics=out["metrics"],cohort=cohort_waiting(runner,state,start,main_end),
        overload=overload_metrics(runner,state,start,runner.duration),peak=peak_exchange(runner,state,start,runner.duration),
        resources=work_between(data,runner.motion,start,runner.duration),
        denied_boarding_attempts=state["denied_boarding_attempts"],release_authorized=False)
    histogram=defaultdict(int)
    for row in state["board_events"]:
        if start<=row["second"]<main_end: histogram[row["wait_seconds"]]+=row["count"]
    boarded=sum(histogram.values()); total=0; rank=(95*boarded+99)//100; p95=None
    for wait,count in sorted(histogram.items()):
        total+=count
        if total>=rank: p95=wait; break
    summary.update(waiting_area_8_12=queue_area(state,start,runner.duration),
        old_boarded_metric=dict(mean=sum(w*n for w,n in histogram.items())/boarded if boarded else None,p95=p95),
        original_metrics_scope=dict(start_second=out["window_start_second"],end_second=out["window_end_second"]),
        queue_age_seconds=out["metrics"]["max_queue_age_seconds"])
    write_json(folder/"summary.json",summary); write_checkpoint(folder/"final_checkpoint.json.gz",state)
    return summary


def audit(output,config,policy):
    parameters=json.loads((ROOT/"data/processed/metro/parameters.json").read_text(encoding="utf-8"))["parameters"]
    rows=[]
    def add(name,source,unit,status,consumer,note):
        rows.append(dict(parameter=name,source=source,unit=unit,status=status,consumer=consumer,note=note))
    add("station entries","data/processed/metro/entries_station_15m.parquet; original XLSX D01-D04","persons/15min","confirmed","forecasting.hourly_median; line1.minute_groups","19 stations; separate hourly source D05 is not mixed")
    add("44 trains","config/line1/experiment.json","active trains","assumed","decision_engine.build_regular_plan","teaching base; no actual inventory")
    add("travel/dwell/turnaround","config/line1/experiment.json","seconds","assumed","simulation.simulate","140/20/110 are uncalibrated scenario times; cycle is derived from movement and regulation")
    add("reference roundtrip","user clarification; parameters.json#reference_roundtrip","seconds","withdrawn","none","99 minutes withdrawn as untrue; no cycle target, calibration or feasibility requirement")
    add("OD S1/S2/S3","line1.assign_destinations","destination probabilities","assumed","passenger_groups.GroupedPassengerSimulation","S1 uniform; S2 weight 4 to L1_07-L1_10; S3 distance in station gaps; transfers excluded")
    add("car capacity","D14 image1.png; config/line1/source_review.json","persons/car","confirmed","GroupedPassengerSimulation; allocate_cars","168 for two heads, 187 for six intermediate cars at 5 persons/m2; actual fleet applicability unknown")
    add("aggregate capacity headings","D14; config/line1/source_review.json","persons/train","conflicting","per-car profile is used","swapped maximum/nominal aggregate labels are preserved as a source conflict")
    add("normal load","parameters.json#regular_load_limit_per_car","persons/car","confirmed","replay_metrics.overload_metrics; actions.select_candidate","120 differs from physical boarding capacity")
    add("headway and line fleet","parameters.json#headway_min/max_trains_on_line","seconds/trains","confirmed","simulation.simulate","113 seconds; 53 active trains; blocks/switches remain unknown")
    add("hourly forecast","user override; forecasting.py","station entries/hour","confirmed","rolling.decide_snapshot","hourly median of complete historical hourly sums; forecast refresh hourly; decisions every 15 minutes")
    add("historical availability","parameters.json#historical_publication_time","timestamp","assumed","forecasting.hourly_median","publication times missing; assume complete intervals available at their end; no future intervals")
    add("external event parser","not present in this workspace","features","missing","none","XLSX parser is reused; no weather/event multipliers are invented")
    add("variable dwell calibration","no measured per-train stop times","seconds","assumed","dwell.keyed_service","fixed 20, triangular 15/20/25 and 10/20/30; instantaneous passenger exchange; door throughput unknown")
    add("reserve timing","D13; prior plan","seconds","conflicting","actions.demo_depots","7/30 minutes and 15-20 minutes have different unresolved semantics; demo explicitly assumes 900-second preparation plus 120-second entry route")
    add("operational readiness/crew/maintenance","parameters.json; maintenance_rules.csv","train readiness","missing","conditional action checks","14 copied maintenance rules do not establish actual train clearance")
    add("variable monetary rates","config/costs/default.json","RUB/unit","missing","decision_engine._economic_report; economics.priced","rates remain null; no zero-ruble savings or monetary optimum")
    add("supported actions","actions.expand_action; existing simulation trip interface","single train","assumed","simulation.simulate","KEEP, ADD, WITHDRAW, DEFER_WITHDRAWAL; explicit demonstration depot routes")
    write_csv(output/"source_usage.csv",rows)
    write_json(output/"assumptions.json",dict(rows=rows,policy=policy,publication=parameters["historical_publication_time"],
        original_ml_request="replaced by hourly median at the user's instruction; no WAPE=6% claim",
        motion_feedback="random service changes movement events; passenger exchange is instantaneous",release_authorized=False))


def forecasts(output,frame,config):
    result={}; folder=output/"forecasts"; folder.mkdir(exist_ok=True)
    for hour in (8,9,10,11):
        as_of=f"{config['date']}T{hour:02}:00:00+03:00"
        bundle=hourly_median(frame,config["station_ids"],as_of)
        result[hour]=bundle; write_json(folder/f"{hour:02}.json",bundle)
    rows=[]
    for hour in (8,9):
        for row in result[hour]["rows"]:
            if datetime.fromisoformat(row["interval_start"]).hour==hour:
                actual=frame[(frame.station_id==row["station_id"]) & (frame.interval_start==pd.Timestamp(row["interval_start"]))].iloc[0]
                rows.append(dict(station_id=row["station_id"],interval_start=row["interval_start"],
                    predicted=row["predicted_count"],actual=int(actual.entry_count),issued_at=result[hour]["issued_at"]))
    write_csv(output/"forecast_vs_actual.csv",rows)
    wape=sum(abs(r["actual"]-r["predicted"]) for r in rows)/sum(r["actual"] for r in rows)
    write_json(output/"forecast_evaluation.json",dict(WAPE=wape,rows=len(rows),period="08:00-10:00",
        training="complete hours available before each issue; all weekdays pooled",target="station turnstile entries",
        source_availability="assumed; original publication times unavailable",ML_WAPE_6_percent_verified=False))
    return result


def diagnose(output,config,groups,policy):
    folder=output/"diagnostic"; folder.mkdir(exist_ok=True); rows=[]; baseline=ROOT/"data/processed/decision_goal4_2_baseline_resumed/N44"
    data=json.loads((baseline/"plan.json").read_text(encoding="utf-8")); profiles=car_profiles(config,[t["train_id"] for t in data["trains"]])
    old_peaks=[]
    for scenario in SCENARIOS:
        runner=GroupedPassengerSimulation(data,"line1-baseline",groups[scenario],profiles)
        state=json.loads(gzip.decompress((baseline/scenario/"final_checkpoint.json.gz").read_bytes()))
        if state["sha256"]!=PassengerSimulation._seal(state) or state["context_id"]!=runner.context_id:
            raise ValueError("old peak checkpoint differs from the exact baseline context")
        old_peaks.append(dict(scenario=scenario,**peak_exchange(runner,state,7200,21600)))
    write_json(output/"baseline_peak_exchange.json",old_peaks)
    cases=[(n,0,None) for n in range(47,54)]+[(44,p,None) for p in policy["phase_offsets_seconds"][1:]]+[(44,0,policy["car_preference_sensitivity"])]
    for count,phase,preference in cases:
        began=time.perf_counter(); name=f"N{count}_phase{phase}"+("_preferred" if preference else "")
        data,meta=regular_plan_at_limit(config,count,phase_offset=phase,end_second=21600+phase)
        data["period_start"]=f"{config['date']}T06:00:00+03:00"; data["period_end"]=f"{config['date']}T12:00:00+03:00"
        case=folder/name; case.mkdir(exist_ok=True); write_json(case/"plan.json",data); write_json(case/"parameters.json",meta)
        profiles=car_profiles(config,[t["train_id"] for t in data["trains"]])
        motion=None
        for scenario in SCENARIOS:
            runner=GroupedPassengerSimulation(data,"line1-baseline",groups[scenario],profiles,car_preference=preference)
            out=runner.run(); motion=runner.motion
            summary=save_passengers(case/scenario,data,runner,out)
            rows.append(dict(case=name,train_count=count,phase_seconds=phase,scenario=scenario,profile="fixed",
                car_preference="weighted sensitivity" if preference else "least loaded",
                max_car_load=summary["overload"]["max_car_load"],exceeded_car_seconds=summary["overload"]["exceeded_car_seconds"],
                normal_passed=summary["overload"]["exceeded_car_seconds"]==0,
                mean_wait=summary["cohort"]["mean_wait_seconds"],p95_wait=summary["cohort"]["p95_wait_seconds"],
                waiting=summary["cohort"]["waiting_passenger_seconds_lower_bound"],queued=out["metrics"]["queued"],
                train_km=summary["resources"]["train_km"],cycle_seconds=meta["cycle_seconds"]))
        write_json(case/"motion.json",motion); write_json(case/"headways.json",headway_report(data,motion))
        print(json.dumps(dict(stage="diagnostic",case=name,seconds=round(time.perf_counter()-began,2)),ensure_ascii=False),flush=True)
    write_csv(output/"static_diagnostic.csv",rows); write_json(output/"static_diagnostic.json",rows)
    return rows


def warmup(config,groups,profile,seed):
    data,_=build_regular_plan(config,44); data,registry=demo_depots(data,config)
    data,timing=compile_timing(data,"line1-baseline",profile,seed,committed=[])
    profiles=car_profiles(config,[t["train_id"] for t in data["trains"]])
    sources={s:GroupedPassengerSimulation(data,"line1-baseline",groups[s],profiles) for s in SCENARIOS}
    states={s:r.run(end_second=7200)["checkpoint"] for s,r in sources.items()}
    return data,registry,profiles,sources,states,timing


def fixture_examples(output):
    config=dict(station_ids=["A","B"],date="2026-09-01",timezone="Europe/Moscow",length_m=2000,
        travel_seconds=120,stop_seconds=0,turnaround_seconds=10,train_type="fixture",car_types=["fixture"])
    data,_=build_regular_plan(config,2,end_second=7200); data,registry=demo_depots(data,config)
    data,_=compile_timing(data,"line1-baseline","fixed",7,committed=[])
    profiles={t["train_id"]:[dict(car_id="C1",car_type="fixture",max_capacity=5,
        normal_threshold=4,source="manual fixture")] for t in data["trains"]}
    source=GroupedPassengerSimulation(data,"line1-baseline",[],profiles)
    state=source.run(end_second=900)["checkpoint"]; result={}
    for name,total in (("steady",60),("growth",120),("low",0),("saturated",600)):
        records=[]
        for day in ("2026-08-29","2026-08-30","2026-08-31"):
            for station in ("A","B"):
                for hour,count in ((6,0),(7,total)):
                    for q in range(4):
                        start=pd.Timestamp(f"{day}T{hour:02}:{q*15:02}:00+03:00")
                        records.append(dict(station_id=station,interval_start=start,interval_end=start+pd.Timedelta(minutes=15),
                            entry_count=count//4,is_complete=True,source_id="fixture"))
        bundle=hourly_median(pd.DataFrame(records),["A","B"],"2026-09-01T06:00:00+03:00")
        decision=decide_snapshot(data,{"S1":source},{"S1":state},profiles,registry,bundle,
            [("KEEP",None),("ADD","R01"),("WITHDRAW","T01")],"fixed",[7])
        decision["evaluation_mode"]="fixture"; result[name]=decision
    folder=Path(output)/"fixtures"; folder.mkdir(exist_ok=True,parents=True)
    write_json(folder/"decisions.json",result)
    return result


def replay_mode(output,config,groups,bundles,policy,mode,actual_profile,planning_profile,seed,initial):
    folder=output/"episodes"/mode; folder.mkdir(parents=True,exist_ok=True)
    data,registry,profiles,sources,states,timing=initial
    states=copy.deepcopy(states); decisions=[]; start=7200
    for scenario,state in states.items():
        (folder/scenario).mkdir(exist_ok=True); write_checkpoint(folder/scenario/"checkpoint_08.json.gz",state)
    candidates=[("KEEP",None),("ADD","R01"),("WITHDRAW","T01")]
    for second in range(7200,14400,900):
        began=time.perf_counter(); choice=("KEEP",None)
        if planning_profile:
            hour=6+second//3600
            decision=decide_snapshot(data,sources,states,profiles,registry,bundles[hour],candidates,planning_profile,
                policy["preliminary_seeds"] if planning_profile!="fixed" else [seed],early_rejection=True)
            selected=decision["selection"]["selected_candidate_id"]
            if selected:
                row=next(r for r in decision["candidate_table"] if r["candidate_id"]==selected)
                choice=(row["action"]["kind"],row["action"]["train_id"])
        else:
            decision=dict(decision_second=second,selection=dict(status="baseline_continuation",selected_candidate_id="KEEP:plan"),candidate_table=[])
        decision.update(actual_applied_action=choice[0],actual_profile=actual_profile,
            current_warning={s:max(n for v in state["loads"].values() for n in v)>120 for s,state in states.items()})
        write_json(folder/f"decision_{second}.json",decision); decisions.append(decision)
        changed=data
        if choice[0]!='KEEP':
            changed,action=expand_action(data,*choice,second,registry)
            changed,_=compile_timing(changed,"line1-baseline",actual_profile,seed,committed=data["leg_timing"],as_of=second)
        next_sources={}; next_states={}
        for scenario,source in sources.items():
            runner=source if choice[0]=='KEEP' else GroupedPassengerSimulation(changed,"line1-baseline",groups[scenario],profiles)
            snapshot=states[scenario] if choice[0]=='KEEP' else runner.branch_checkpoint(states[scenario],source)
            out=runner.run(start_second=second,end_second=second+900,checkpoint=snapshot)
            next_sources[scenario]=runner; next_states[scenario]=out["checkpoint"]
        sources,states,data=next_sources,next_states,changed
        print(json.dumps(dict(stage="replay",mode=mode,second=second,choice=choice[0],status=decision["selection"]["status"],seconds=round(time.perf_counter()-began,2))),flush=True)
    summaries=[]
    for scenario,source in sources.items():
        (folder/scenario).mkdir(exist_ok=True); write_checkpoint(folder/scenario/"checkpoint_10.json.gz",states[scenario])
        out=source.run(start_second=14400,checkpoint=states[scenario]); summary=save_passengers(folder/scenario,data,source,out)
        # Legacy boardings are separate from the all-arrivals cohort.
        board=[r for r in out["checkpoint"]["board_events"] if 7200<=r["second"]<14400]
        count=sum(r["count"] for r in board)
        summary["legacy_boarded_mean_wait_seconds"]=sum(r["count"]*r["wait_seconds"] for r in board)/count if count else None
        summary.update(mode=mode,scenario=scenario,actual_profile=actual_profile,
            applied_actions=[d["actual_applied_action"] for d in decisions])
        write_json(folder/scenario/"summary.json",summary); summaries.append(summary)
    motion=next(iter(sources.values())).motion
    write_json(folder/"plan.json",data); write_json(folder/"motion.json",motion)
    write_json(folder/"movement_10.json",movement_snapshot(data,motion,14400))
    write_json(folder/"movement_12.json",dict(second=21600,trains=motion["final_state"],unfinished_obligations=motion["unfinished_obligations"]))
    return summaries


def replay(output,config,groups,bundles,policy):
    summaries=[]; seed=policy["actual_episode_seed"]
    for actual,modes in (("fixed",[("A_fixed",None),("B_fixed","fixed")]),
                         ("mild",[("A_mild",None),("B_mild","fixed"),("D_mild","mild")])):
        initial=warmup(config,groups,actual,seed)
        for mode,planning in modes:
            summaries+=replay_mode(output,config,groups,bundles,policy,mode,actual,planning,seed,initial)
    write_json(output/"mode_comparison.json",dict(modes=summaries,
        original_C_mode="ML model replaced by the user's requested median; C would duplicate B and is not presented as an independent ML result"))
    return summaries


def seed_campaign(output,config,groups,policy):
    folder=output/"seed_campaign"; folder.mkdir(exist_ok=True); records=[]
    initial=warmup(config,groups,"mild",policy["actual_episode_seed"])
    data,registry,profiles,sources,states,_=initial
    campaigns=[("preliminary","mild",policy["preliminary_seeds"]),("preliminary","wider",policy["preliminary_seeds"]),
               ("validation","mild",policy["validation_seeds"])]
    for stage,profile,seeds in campaigns:
        for seed in seeds:
            began=time.perf_counter(); case=folder/f"{stage}_{profile}_{seed}"; case.mkdir(exist_ok=True)
            for scenario,source in sources.items():
                changed,timing=compile_timing(data,"line1-baseline",profile,seed,committed=data["leg_timing"],as_of=7200)
                runner=GroupedPassengerSimulation(changed,"line1-baseline",groups[scenario],profiles)
                snapshot=runner.branch_checkpoint(states[scenario],source)
                out=runner.run(start_second=7200,checkpoint=snapshot)
                summary=save_passengers(case/scenario,changed,runner,out)
                records.append(dict(stage=stage,profile=profile,seed=seed,scenario=scenario,
                    max_car_load=summary["overload"]["max_car_load"],exceeded_car_seconds=summary["overload"]["exceeded_car_seconds"],
                    mean_wait=summary["cohort"]["mean_wait_seconds"],p95_wait=summary["cohort"]["p95_wait_seconds"],
                    queued=out["metrics"]["queued"],queue_age=out["metrics"]["max_queue_age_seconds"],
                    train_km=summary["resources"]["train_km"],holding_seconds=summary["resources"]["state_seconds"].get("holding",0),
                    service_mean_seconds=timing["service_mean_seconds"]))
            write_csv(output/"seed_results.csv",records)
            print(json.dumps(dict(stage=stage,profile=profile,seed=seed,seconds=round(time.perf_counter()-began,2))),flush=True)
    # Compare the conditional ADD against KEEP in the exact same actual environment.
    rows=[]
    for scenario,source in sources.items():
        evaluated=evaluate_candidates(data,groups[scenario],profiles,registry,states[scenario],source,
            [("KEEP",None),("ADD","R01")],"mild",policy["actual_episode_seed"])
        for row in evaluated:
            if row["status"]=="evaluated":
                row["cohort"]=cohort_waiting(row["_runner"],row["_output"]["checkpoint"],7200,14400)
                save_passengers(output/"counterfactual"/row["action"]["kind"]/scenario,row["_data"],row["_runner"],row["_output"])
                write_json(output/"counterfactual"/row["action"]["kind"]/"plan.json",row["_data"])
                write_json(output/"counterfactual"/row["action"]["kind"]/"motion.json",row["_runner"].motion)
            rows.append(dict(scenario=scenario,**public(row)))
    write_json(output/"counterfactual_comparison.json",rows)
    write_json(output/"seed_results.json",records)
    aggregate=[]
    for stage,profile,seeds in campaigns:
        for scenario in SCENARIOS:
            subset=[r for r in records if r["stage"]==stage and r["profile"]==profile and r["scenario"]==scenario]
            result=dict(stage=stage,profile=profile,scenario=scenario,seeds=len(subset),
                violation_fraction=sum(r["exceeded_car_seconds"]>0 for r in subset)/len(subset))
            for key in ("max_car_load","exceeded_car_seconds","mean_wait","p95_wait","queued","train_km","holding_seconds"):
                values=[float(r[key]) for r in subset if r[key] is not None]
                result[key]=dict(median=median(values),minimum=min(values),maximum=max(values)) if values else None
            aggregate.append(result)
    write_json(output/"seed_summary.json",aggregate)
    return records


def plots(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    tz=ZoneInfo("Europe/Moscow"); plt.rcParams["font.family"]="DejaVu Sans"
    table=pd.read_csv(output/"forecast_vs_actual.csv"); station_ids=["L1_07","L1_09","L1_10"]
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for station,ax in zip(station_ids,axes):
        rows=table[table.station_id==station]; x=pd.to_datetime(rows.interval_start)
        ax.plot(x,rows.actual,label="Факт входов"); ax.plot(x,rows.predicted,label="Почасовая медиана")
        ax.set_ylabel(station); ax.grid(alpha=.2); ax.legend()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",tz=tz)); axes[0].set_title("Входы за 15 минут, Москва; факт используется для оценки")
    fig.tight_layout(); fig.savefig(output/"forecast_vs_actual.png",dpi=160); plt.close(fig)
    keep=output/"counterfactual/KEEP"; add=output/"counterfactual/ADD"
    states={name:json.loads(gzip.decompress((path/"S3/final_checkpoint.json.gz").read_bytes())) for name,path in (("KEEP",keep),("ADD",add))}
    peak=json.loads((keep/"S3/summary.json").read_text(encoding="utf-8"))["peak"]
    motion=json.loads((keep/"motion.json").read_text(encoding="utf-8")); events={e["event_id"]:e for e in motion["events"]}
    event=events[next(r["motion_event_id"] for r in states["KEEP"]["load_events"] if r["train_id"]==peak["train_id"] and r["second"]==peak["second"])]
    data=json.loads((keep/"plan.json").read_text(encoding="utf-8")); tracks={t["track_id"]:t for t in data["tracks"]}
    origin=tracks[event["location"]]["from_station"] if event["event_type"]=="departure" else event["location"]
    direction=1 if peak["direction"]=="up" else -1
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for name,state in states.items():
        rows=[r for r in state["load_events"] if r["train_id"]==peak["train_id"] and r["car_id"]==peak["car_id"] and r["second"]>=7200]
        axes[0].step([6+r["second"]/3600 for r in rows],[r["load"] for r in rows],where="post",label=name)
        queue=[r for r in state["queue_samples"] if r["station_id"]==origin and r["direction"]==direction and r["second"]>=7200]
        axes[1].plot([6+r["second"]/3600 for r in queue],[r["queued"] for r in queue],label=name)
    axes[0].axhline(120,color="red",linestyle="--",label="120 пассажиров")
    add_motion=json.loads((add/'motion.json').read_text(encoding='utf-8'))
    effect=next(e['second'] for e in add_motion['events'] if e['train_id']=='R01'
        and e['event_type']=='departure' and e['location'].endswith(('_up','_down')))
    for ax in axes: ax.axvline(6+effect/3600,color="gray",linestyle=":"); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel("Пассажиров в вагоне"); axes[1].set_ylabel("Очередь, человек"); axes[1].set_xlabel("Час, Москва")
    axes[0].set_title(f"S3, {peak['train_id']}/{peak['car_id']}; ADD показан как условный сценарий")
    fig.tight_layout(); fig.savefig(output/"load_and_queue.png",dpi=160); plt.close(fig)
    data=json.loads((add/"plan.json").read_text(encoding="utf-8")); motion=json.loads((add/"motion.json").read_text(encoding="utf-8"))
    grouped=defaultdict(list)
    for event in motion["events"]:
        if event["event_type"]=="departure" and event["location"].endswith(("_up","_down")): grouped[event["location"]].append(event["second"])
    headways=[(b,b-a) for times in grouped.values() for a,b in zip(times,times[1:]) if b>=7200]
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    axes[0].scatter([6+t/3600 for t,g in headways],[g for t,g in headways],s=4); axes[0].axhline(113,color="red",linestyle="--")
    service=[r for r in data["leg_timing"] if r["service_seconds"]<=30 and r["departure_second"]>=7200]
    axes[1].scatter([6+r["departure_second"]/3600 for r in service],[r["service_seconds"] for r in service],s=4)
    for ax,label in zip(axes,("Интервал, с","Обслуживание, с","Составов на линии")): ax.set_ylabel(label); ax.grid(alpha=.2)
    stations={s["station_id"]:s for s in data["stations"]}; tracks={t["track_id"]:t for t in data["tracks"]}
    positions={t["train_id"]:(t["state"],t["location"]) for t in data["trains"]}
    fleet=[]
    def active():
        return sum(any(stations[tracks[loc][k]]["kind"]=="line" for k in ("from_station","to_station"))
            if state=="moving" else stations[loc]["kind"]=="line" for state,loc in positions.values())
    for event in motion["events"]:
        positions[event["train_id"]]=(event["next_state"],event["location"])
        if event["second"]>=7200: fleet.append((6+event["second"]/3600,active()))
    axes[2].step([t for t,n in fleet],[n for t,n in fleet],where="post",label="Условный ADD, выпуск по событиям"); axes[2].legend()
    actual_effect=next(e["second"] for e in motion["events"] if e["train_id"]=="R01" and e["event_type"]=="departure" and e["location"].endswith(("_up","_down")))
    for ax in axes: ax.axvline(6+actual_effect/3600,color="gray",linestyle=":")
    axes[-1].set_xlabel("Час, Москва"); axes[0].set_title("Фактические события движения, mild; инфраструктурные разрешения неизвестны")
    fig.tight_layout(); fig.savefig(output/"movement.png",dpi=160); plt.close(fig)
    seeds=pd.read_csv(output/"seed_results.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for scenario in SCENARIOS:
        rows=seeds[(seeds.stage=="validation")&(seeds.scenario==scenario)]
        axes[0].plot(rows.seed,rows.mean_wait,"o-",label=scenario); axes[1].plot(rows.seed,rows.max_car_load,"o-",label=scenario)
    axes[0].set_ylabel("Среднее ожидание, с"); axes[1].set_ylabel("Пик вагона, человек"); axes[1].axhline(120,color="red",linestyle="--")
    for ax in axes: ax.set_xlabel("Независимый seed"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"seed_variation.png",dpi=160); plt.close(fig)


def report(output):
    modes=json.loads((output/"mode_comparison.json").read_text(encoding="utf-8"))["modes"]
    static=json.loads((output/"static_diagnostic.json").read_text(encoding="utf-8"))
    forecast=json.loads((output/"forecast_evaluation.json").read_text(encoding="utf-8"))
    lines=["# GOAL 4.2: почасовая медиана и изменчивые остановки","",
        f"На 1 сентября 2026 года WAPE медианы за 08:00–10:00 равен {forecast['WAPE']:.2%}. Это входы на станции, не загрузка вагонов.","",
        "Модель с заявленным WAPE 6% заменена почасовой медианой по указанию пользователя. Все дни недели объединены. Историческое время публикации исходных таблиц неизвестно; доступность полного интервала к его окончанию принята как допущение.","",
        "99 минут отозваны пользователем как ненастоящее значение. Ограничения на длительность оборота нет. Времена перегонов 140 с, остановок 20 с и разворотов 110 с остаются некалиброванными допущениями прежнего контрольного сценария. Переменные остановки и удержания изменяют фактический оборот.","",
        "| Режим | OD | Ожидание всех вошедших, с | P95, с | Пик вагона | Вагоно-с превышения 120 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in modes:
        c,o=row["cohort"],row["overload"]
        lines.append(f"| {row['mode']} | {row['scenario']} | {c['mean_wait_seconds']} | {c['p95_wait_seconds']} | {o['max_car_load']} | {o['exceeded_car_seconds']} |")
    lines += ["","A продолжает план. B выбирает по медиане с фиксированными будущими остановками. D выбирает по медиане с проверкой mild. A/B_fixed имеют одну фиксированную среду, A/B/D_mild имеют один снимок в 08:00 и одинаковые внешние случайные условия. Отдельного результата ML нет.","",
        "Проверенные статические выпуски 47–53:","","| Выпуск | OD | Пик | Превышение, вагоно-с | Соблюдение 120 |","|---:|---|---:|---:|---|"]
    for row in static:
        if row["phase_seconds"]==0 and row["car_preference"]=="least loaded":
            lines.append(f"| {row['train_count']} | {row['scenario']} | {row['max_car_load']} | {row['exceeded_car_seconds']} | {row['normal_passed']} |")
    lines += ["","В текущем контрольном сценарии для 53 составов минимальный интервал 113 с требует расчётного цикла 5989 с. Дополнительные 49 с распределены на конечные: 24 и 25 с. Эти числа вычислены из времён участков и требуемого интервала. При других временах участков они пересчитываются. Выпуск не меняется мгновенно. Сценарные пути депо заданы отдельно; фактические готовность, обслуживание, персонал и инфраструктурные разрешения неизвестны.","",
        "В 08:00 уже накопленная загрузка выше 120 сохраняется у всех кандидатов. Такой кандидат не может задним числом устранить текущее нарушение. no_feasible_candidate означает отсутствие штатного решения в проверенном наборе действий. Условный ADD оценивается отдельно; он не выдан как диспетчерская команда.","",
        "Пики прежнего расчёта расшифрованы в baseline_peak_exchange.json. 187 совпадает с физическим пределом промежуточного вагона выбранного профиля. S2/S3 удерживают пассажиров дольше S1; пересадки не добавлены. Проверены фазы 0/37/73 секунды и отдельное предпочтение третьего вагона. Это чувствительность предполагаемой OD и посадки, не измеренный предел линии.","",
        "На каждом шаге используется прогноз, выпущенный к этому моменту. Будущие наблюдавшиеся входы используются только при оценке принятого плана. Медиана обновляется в начале часа; между обновлениями горизонт сокращается. Прогноз после его покрытия не дополняется будущим фактом.","",
        "Строгое требование соблюдения 120 во всех проверяемых условиях позволяет остановить проверку кандидата после первого нарушения. В decision_*.json сохранены запрошенные и фактически проверенные seed. Разброс продолжения плана проверен на 10 предварительных seed для mild/wider и 20 других seed для mild. Эти конечные выборки не дают гарантии безопасности.","",
        "Денежные ставки остаются null. Использованы прежние economics.priced и экономический адаптер; натуральная работа вычислена из фактических событий. Пробег имеет частичный статус журнала прибытий, незавершённые перемещения и обязательства сохраняются. Уникальное число пассажиров, затронутых перегрузкой, не восстановлено; вместо него точно считаются пассажиро-секунды в перегруженных вагонах.","",
        "Изменчивость обслуживания сдвигает дальнейшее движение того же поезда. Удержания проходят общий валидатор 113 секунд и сохраняются отдельно от обслуживания и отстоя. Загрузка не меняет время обслуживания; обмен пассажиров мгновенный, дверная пропускная способность не проверена. Параметры mild/wider имеют инженерный статус.","",
        "Команды и устройство кода описаны в docs/goal4_2.md. source_usage.csv связывает источники с реально использующими их функциями. Состояния на 10:00/12:00, все рассмотренные кандидаты, причины отказа, прогнозы и случайные ключи сохранены рядом с отчётом."]
    write_text(output/"summary.md","\n".join(lines)+"\n")
    card=dict(status="conditional_scenario_analysis",forecast_model="station-hour-median-v1",
        finding="normal load violated in the checked snapshot/candidate combinations",cash_savings_RUB=None,
        explanation_source="calculated decision packages only",release_authorized=False)
    write_json(output/"result_card.json",card)


def input_files():
    names=["simulation.py","passengers.py","passenger_groups.py","decision_engine.py","line1.py","economics.py",
        "forecasting.py","dwell.py","actions.py","replay_metrics.py","rolling.py","metro_replay.py",
        "config/decision_engine/goal4_2.json","config/line1/experiment.json","config/line1/source_review.json",
        "config/costs/default.json","data/processed/metro/entries_station_15m.parquet","data/processed/metro/parameters.json",
        "data/processed/metro/build_manifest.json","data/processed/decision_goal4_2_baseline_resumed/N44/plan.json"]
    names += [f'data/processed/decision_goal4_2_baseline_resumed/N44/{s}/final_checkpoint.json.gz' for s in SCENARIOS]
    names += [name for name in ('data/processed/goal4_2_verification/diagnostic_provenance.json',
        'data/processed/goal4_2_verification/metro_replay_before_keep_reuse.py') if (ROOT/name).exists()]
    return [ROOT/name for name in names]


def inherited_diagnostics(output):
    path=ROOT/'data/processed/goal4_2_verification/diagnostic_provenance.json'
    if not path.exists(): return None
    receipt=json.loads(path.read_text(encoding='utf-8'))
    for name,value in receipt['inputs'].items():
        if sha(receipt['archived_changed_inputs'].get(name,name))!=value:
            raise ValueError(f'inherited diagnostic input changed: {name}')
    for name,value in receipt['outputs'].items():
        if sha(output/name)!=value: raise ValueError(f'inherited diagnostic result changed: {name}')
    return str(path)


def run(output,stages):
    output=safe_output(output); start=time.perf_counter(); inputs={str(p):sha(p) for p in input_files()}
    inherited=inherited_diagnostics(output) if 'diagnose' not in stages else None
    check_data(ROOT,ROOT/"data/processed/metro")
    config=load_config(); policy=json.loads(CONFIG.read_text(encoding="utf-8"))
    if set(policy["preliminary_seeds"]) & set(policy["validation_seeds"]): raise ValueError("seed sets overlap")
    frame=pd.read_parquet(ROOT/"data/processed/metro/entries_station_15m.parquet"); groups,_,_=prepare_demand(frame,config)
    audit(output,config,policy); bundles=forecasts(output,frame,config); fixture_examples(output)
    for stage in stages:
        if stage=="diagnose": diagnose(output,config,groups,policy)
        elif stage=="replay": replay(output,config,groups,bundles,policy)
        elif stage=="seeds": seed_campaign(output,config,groups,policy)
        elif stage=="report": plots(output); report(output)
    if any(sha(p)!=h for p,h in inputs.items()): raise ValueError("consumed inputs changed during the campaign")
    outputs={p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file() and p.name!="build_manifest.json"}
    write_json(output/"build_manifest.json",dict(model_version=VERSION,status="success",inputs=inputs,outputs=outputs,stages=stages,
        inherited_stages=['diagnose'] if inherited else [],diagnostic_provenance=inherited))
    return dict(output=str(output),files=len(outputs)+1,stages=stages,seconds=time.perf_counter()-start,release_authorized=False)


def check(output):
    output=Path(output).resolve(); manifest=json.loads((output/"build_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_version")!=VERSION or manifest.get("status")!="success": raise ValueError("no successful GOAL 4.2 manifest")
    if manifest.get('inherited_stages'): inherited_diagnostics(output)
    for name,value in manifest["inputs"].items():
        if sha(name)!=value: raise ValueError(f"input changed: {name}")
    for name,value in manifest["outputs"].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or sha(path)!=value: raise ValueError(f"output changed: {name}")
    observed={p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if observed!=set(manifest["outputs"])|{"build_manifest.json"}: raise ValueError("artifact set changed")
    return dict(status="passed",files=len(observed),release_authorized=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=("run","check"))
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--profile",choices=("goal4_2","goal4_3"),default="goal4_2")
    parser.add_argument("--stages",nargs="+",choices=("diagnose","replay","seeds","report"),default=["diagnose","replay","seeds","report"])
    args=parser.parse_args(argv)
    try:
        if args.profile == "goal4_3":
            from goal43_replay import run as operational_run
            from goal43_verify import verify
            value = operational_run(args.output or ROOT/"data/processed/goal4_3") if args.command == "run" else verify(args.output or ROOT/"data/processed/goal4_3")
            print(json.dumps(value,ensure_ascii=False),flush=True); return 0
        args.output = args.output or ROOT/"data/processed/goal4_2"
        value=run(args.output,args.stages) if args.command=="run" else check(args.output)
        print(json.dumps(value,ensure_ascii=False),flush=True); return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f"metro_replay: {exc}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())
