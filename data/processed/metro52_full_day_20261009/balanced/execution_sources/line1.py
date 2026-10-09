"""GOAL 3.3: historical entries with an explicitly assumed periodic line plan."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pandas as pd

from economics import load_costs, priced, total_quantities, write_json, write_text
from simulation import digest, fields_only, integer, number, simulate, to_evaluation


ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "line1-experiment-v1"
STATION_IDS = ["L1_01", "L1_02", "L1_03", "L1_04", "L1_05", "L1_06", "L1_07",
               "L1_08", "L1_09", "L1_10", "L1_11", "L1_12", "L1_13", "L1_14",
               "L1_15", "L1_16", "L1_17", "L1_18", "L1_19"]
COMMON_OUTPUTS={"assumptions.json","checks.json","minute_entries.csv","movement_full.simulation.json",
    "movement_input.json","movement_main.simulation.json","period.json","transport_work.json",
    "scenario_comparison.csv","scenario_comparison.json","plot_metadata.json","summary.md",
    "S1_movement.png","S1_queues.png","S1_car_loads.png"}
SCENARIO_OUTPUTS={"metrics.json","warmup_checkpoint.json.gz","final_checkpoint.json.gz","initial_car_loads.json",
    "economics.json","queues_end.csv","approaching_end.csv","queue_samples.csv","car_load_events.csv","OD_groups.csv"}
OUTPUTS=COMMON_OUTPUTS|{f"{s}/{n}" for s in ("S1","S2","S3") for n in SCENARIO_OUTPUTS}


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "config/line1/experiment.json").read_text(encoding="utf-8"))
    fields_only(config,("schema_version","station_ids","date","timezone","warmup_start","main_start","end",
        "train_count","travel_seconds","stop_seconds","turnaround_seconds","platform_delay_seconds","length_m",
        "train_type","car_types","capacity_by_type","normal_threshold","provenance"))
    if not isinstance(config["date"],str) or datetime.strptime(config["date"],"%Y-%m-%d").strftime("%Y-%m-%d")!=config["date"]:
        raise ValueError("scenario date must be YYYY-MM-DD")
    if config["schema_version"] != "1.0" or config["station_ids"] != STATION_IDS:
        raise ValueError("line configuration needs the explicit project station traversal")
    if any(config[k]!=v for k,v in {"warmup_start":"06:00","main_start":"08:00","end":"10:00","timezone":"Europe/Moscow"}.items()):
        raise ValueError("GOAL 3.3 fixes warmup 06–08 and evaluation 08–10 Europe/Moscow")
    for key in ("train_count", "travel_seconds", "stop_seconds", "turnaround_seconds", "length_m", "normal_threshold"):
        integer(config[key], key, 1)
    integer(config["platform_delay_seconds"], "platform delay")
    cars=config["car_types"]
    if (config["train_type"]!="Юбилейный" or not isinstance(cars,list) or len(cars)!=8
            or cars[0]!="81-722.1" or cars[-1]!="81-722.1"
            or sorted(cars)!=sorted(["81-722.1"]*2+["81-723.1"]*4+["81-724.1"]*2)):
        raise ValueError("D14 profile requires two heads, four motor and two nonmotor intermediate cars")
    capacities=config["capacity_by_type"]
    if not isinstance(capacities,dict) or set(capacities)!={"81-722.1","81-723.1","81-724.1"}:
        raise ValueError("D14 capacity profile is incomplete")
    for kind,expected_capacity in {"81-722.1":168,"81-723.1":187,"81-724.1":187}.items():
        if integer(capacities[kind],"sourced car capacity",1)!=expected_capacity:
            raise ValueError("capacity contradicts the selected D14 per-car source row")
    if config["length_m"]!=29600:
        raise ValueError("this experiment uses the explicit 29600-metre reference assumption")
    statuses={"station_order":"explicit_project_dictionary","train_count":"assumed","travel_stop_turnaround":"assumed",
        "length_m":"assumed","capacity_by_type":"source_observed","car_types":"partial","normal_threshold":"organizer_confirmed",
        "platform_delay_seconds":"assumed","minute_distribution":"assumed","OD":"assumed","initial_passengers":"assumed",
        "boarding_policy":"assumed","transfer_inflow":"not_modeled","historical_entries":"source_observed"}
    provenance=config["provenance"]
    if (not isinstance(provenance,dict) or set(provenance)!=set(statuses)
            or any(not isinstance(provenance[k],dict) or provenance[k].get("status")!=v for k,v in statuses.items())):
        raise ValueError("scenario provenance must preserve observed, assumed and unmodeled statuses")
    parameters=json.loads((ROOT/"data/processed/metro/parameters.json").read_text(encoding="utf-8"))["parameters"]
    expected={"cars_per_train":8,"headway_min":113,"max_trains_on_line":53,
              "regular_load_limit_per_car":config["normal_threshold"]}
    if any(parameters[k]["value"]!=value or parameters[k]["status"]!="organizer_confirmed" for k,value in expected.items()):
        raise ValueError("scenario conflicts with sourced organizer parameters")
    cycle=2*(18*config["travel_seconds"]+17*config["stop_seconds"]+config["turnaround_seconds"])
    if config["train_count"]>53 or cycle%config["train_count"] or cycle//config["train_count"]<113:
        raise ValueError("scenario fleet or periodic headway violates supported movement constraints")
    return config


def largest_remainders(total, weights):
    integer(total, "allocation total")
    if not isinstance(weights, list) or not weights:
        raise ValueError("allocation requires weights")
    for w in weights:
        integer(w, "allocation weight")
    denominator = sum(weights)
    if not denominator:
        raise ValueError("allocation weights sum to zero")
    amounts = [total * w // denominator for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denominator), i))
    for i in order[:total - sum(amounts)]:
        amounts[i] += 1
    return amounts


def select_period(frame, station_ids, requested_date="2026-09-01"):
    def extract(day):
        start = pd.Timestamp(day + "T06:00:00", tz="Europe/Moscow")
        end = start + pd.Timedelta(hours=4)
        selected = frame[(frame.interval_start >= start) & (frame.interval_start < end)].copy()
        expected = {(s, start + pd.Timedelta(minutes=15 * i)) for s in station_ids for i in range(16)}
        actual = set(zip(selected.station_id, selected.interval_start))
        valid = (actual == expected and len(selected) == len(expected)
                 and pd.api.types.is_bool_dtype(selected.is_complete.dtype) and selected.is_complete.notna().all()
                 and selected.is_complete.all()
                 and selected.entry_count.notna().all() and pd.api.types.is_integer_dtype(selected.entry_count.dtype)
                 and not pd.api.types.is_bool_dtype(selected.entry_count.dtype) and (selected.entry_count >= 0).all()
                 and (selected.interval_end - selected.interval_start == pd.Timedelta(minutes=15)).all())
        return selected, start, end, bool(valid)
    days = sorted({str(day) for day in frame.interval_start.dt.date})
    chosen = requested_date
    if chosen not in days:
        chosen = next((d for d in days if d.startswith("2026-09-") and datetime.fromisoformat(d).weekday() < 5 and extract(d)[3]), None)
        if chosen is None:
            raise ValueError("no full September weekday found")
    selected, start, end, valid = extract(chosen)
    if not valid:
        raise ValueError("chosen day lacks complete, unique 19-station coverage or valid counts")
    return selected, {"date": chosen, "requested_date": requested_date, "date_replaced": chosen != requested_date,
                      "start": start.isoformat(), "main_start": (start + pd.Timedelta(hours=2)).isoformat(),
                      "end": end.isoformat(), "timezone": "Europe/Moscow", "source_rows": len(selected)}


def minute_groups(frame, station_ids, start, end, delay=120):
    integer(delay, "platform delay")
    positions = {s: i for i, s in enumerate(station_ids)}
    records = frame.to_dict("records")
    groups = []
    for row in sorted(records, key=lambda r: (r["interval_start"], positions[r["station_id"]])):
        if not start <= row["interval_start"] < end:
            continue
        total = integer(row["entry_count"],"source count")
        quotient, remainder = divmod(total, 15)
        for minute in range(15):
            second = int((row["interval_start"] - start).total_seconds()) + 60 * minute
            groups.append({"group_id": f"{row['station_id']}:M{second // 60:04}",
                "turnstile_second": second, "platform_second": second + delay,
                "origin": row["station_id"], "count": quotient + (minute < remainder),
                "source_id": row["source_id"], "source_interval_start": row["interval_start"].isoformat(),
                "source_interval_count": total})
    return groups


def assign_destinations(groups, station_ids, scenario):
    if scenario not in {"S1", "S2", "S3"}:
        raise ValueError("unknown OD scenario")
    positions = {s: i for i, s in enumerate(station_ids)}
    result = []
    for group in groups:
        origin = positions[group["origin"]]
        destinations = [s for s in station_ids if s != group["origin"]]
        weights = [4 if scenario == "S2" and s in STATION_IDS[6:10] else
                   abs(positions[s] - origin) if scenario == "S3" else 1 for s in destinations]
        amounts = largest_remainders(group["count"], weights)
        result.append({**copy.deepcopy(group), "destinations": [
            {"station_id": s, "count": count, "direction": 1 if positions[s] > origin else -1}
            for s, count in zip(destinations, amounts) if count]})
    return result


def build_movement(config, date=None, phase_offset=0):
    integer(phase_offset, "phase offset")
    ids = config["station_ids"]
    lengths = largest_remainders(config["length_m"], [1] * 18)
    stations = [{"station_id": s, "kind": "line", "stop_seconds": config["stop_seconds"] if 0 < i < 18 else 0,
                 "turnaround_seconds": config["turnaround_seconds"] if i in (0, 18) else 0} for i, s in enumerate(ids)]
    tracks = []
    for i, length in enumerate(lengths):
        for direction, a, b in (("up", ids[i], ids[i+1]), ("down", ids[i+1], ids[i])):
            tracks.append({"track_id": f"P{i+1:02}_{direction}", "from_station": a, "to_station": b,
                           "length_km": number(Decimal(length) / 1000), "travel_seconds": config["travel_seconds"]})
    lookup = {r["track_id"]: r for r in tracks}
    routes = [[f"P{i:02}_up" for i in range(1, 19)], [f"P{i:02}_down" for i in range(18, 0, -1)]]
    half = 18 * config["travel_seconds"] + 17 * config["stop_seconds"] + config["turnaround_seconds"]
    cycle = half * 2
    if cycle % config["train_count"]:
        raise ValueError("periodic phases must have an integer headway")
    headway = cycle // config["train_count"]
    if headway < 113:
        raise ValueError("periodic headway below audited minimum")
    duration = 14400 - phase_offset
    if duration <= 0:
        raise ValueError("phase offset outside experiment")
    start = datetime.fromisoformat((date or config["date"]) + "T06:00:00").replace(tzinfo=ZoneInfo(config["timezone"])) + timedelta(seconds=phase_offset)
    legs = []
    for direction, route in enumerate(routes):
        at = direction * half
        for j, track_id in enumerate(route):
            arrival = at + config["travel_seconds"]
            wait = config["turnaround_seconds"] if j == 17 else config["stop_seconds"]
            legs.append({"track_id": track_id, "departure": at, "arrival": arrival, "ready": arrival + wait,
                         "direction": direction, "index": j})
            at = arrival + wait
    trains, trips, history = [], [], []
    for i in range(config["train_count"]):
        train_id = f"T{i+1:02}"
        absolute_phase = i * headway + phase_offset
        phase = absolute_phase % cycle
        leg = next(leg for leg in legs if leg["departure"] <= phase < leg["ready"])
        route = routes[leg["direction"]]
        train = {"train_id": train_id, "train_type": config["train_type"], "cars": 8,
                 "ready_after_seconds": 0}
        if phase == leg["departure"]:
            train.update(state="standby", location=lookup[leg["track_id"]]["from_station"])
            departure, remaining = 0, route[leg["index"]:]
        elif phase < leg["arrival"]:
            train.update(state="moving", location=leg["track_id"], initial_movement={"arrival_second": leg["arrival"]-phase, "passenger": True})
            departure, remaining = leg["ready"] - phase, route[leg["index"]+1:]
        else:
            train.update(state="turnaround" if leg["index"] == 17 else "stop",
                         location=lookup[leg["track_id"]]["to_station"], ready_after_seconds=leg["ready"]-phase)
            departure, remaining = leg["ready"] - phase, route[leg["index"]+1:]
        trains.append(train)
        half_id = absolute_phase // half
        direction = leg["direction"]
        if not remaining:
            direction = 1-direction
            half_id += 1
            remaining = routes[direction]
        while departure < duration:
            trips.append({"trip_id": f"{train_id}:H{half_id:04}", "train_id": train_id,
                          "departure_second": departure, "route": remaining, "passenger": True})
            departure += len(remaining)*config["travel_seconds"] + (len(remaining)-1)*config["stop_seconds"] + config["turnaround_seconds"]
            direction = 1-direction
            half_id += 1
            remaining = routes[direction]
    # Last periodic departure on every track supplies the negative boundary context.
    for leg in legs:
        times = [leg["departure"] - ((i*headway+phase_offset) % cycle) for i in range(config["train_count"])]
        negatives = [t if t < 0 else t-cycle for t in times]
        history.append({"track_id": leg["track_id"], "departure_second": max(negatives)})
    data = {"schema_version": "1.0", "source": "GOAL 3.3 assumed periodic plan, historical entries are separate",
            "period_start": start.isoformat(), "period_end": (start+timedelta(seconds=duration)).isoformat(),
            "stations": stations, "tracks": tracks, "trains": trains,
            "plans": [{"plan_id": "line1-baseline", "trips": trips}], "previous_departures": history,
            "max_trains_on_line": 53}
    return data, {"cycle_seconds": cycle, "half_cycle_seconds": half, "headway_seconds": headway,
                  "length_m": sum(lengths), "segment_lengths_m": lengths, "train_count": len(trains),
                  "timing_status": "assumed", "distance_status": "assumed_reference_length"}


def car_profiles(config, train_ids):
    return {train: [{"car_id": f"C{i+1:02}", "car_type": kind,
                     "max_capacity": config["capacity_by_type"][kind], "normal_threshold": config["normal_threshold"],
                     "source": "D14 word/media/image1.png, per-car 5 persons/m2; actual applicability unknown"}
                    for i, kind in enumerate(config["car_types"])] for train in train_ids}


def scenario_parameters_text(config,meta):
    return (f"Принято: {config['train_count']} состава, цикл {meta['cycle_seconds']} с, интервал {meta['headway_seconds']} с, "
        f"перегон {config['travel_seconds']} с, промежуточная остановка {config['stop_seconds']} с, "
        f"оборот {config['turnaround_seconds']} с, подход {config['platform_delay_seconds']} с. "
        "Длина 29,6 км взята из справочного учебного расчёта и распределена по перегонам условно; геометрия оператора не подтверждена.")


def main_economics(motion):
    evaluation = to_evaluation(motion)
    quantities = total_quantities(evaluation)
    articles = load_costs(ROOT / "config/costs/default.json")
    period = {"start": evaluation.period_start, "end": evaluation.period_end}
    rows = [{"name": a.name, "accounting_kind": a.accounting_kind,
             **priced(quantities[a.quantity_key], a, period)} for a in articles]
    dto=asdict(evaluation)
    dto["quantities"]={k:q.as_dict() for k,q in evaluation.quantities.items()}
    return {"model_version": "economics-v1.1", "evaluation": dto,
            "articles": rows, "money_status": "unknown", "cash_total_RUB": None, "savings": None,
            "comparison_performed": False, "all_constraints_verified": False, "release_authorized": False,
            "distance_scope": "arrival ledger (start,end], partial at moving boundaries; assumed segment geometry",
            "reason": "one movement plan; energy and paid work models and money rates are missing"}


def write_csv(path, rows, columns=None):
    rows=list(rows)
    if columns is None:
        columns=list(rows[0]) if rows else []
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_checkpoint(path,state):
    data=json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    buffer=io.BytesIO()
    with gzip.GzipFile(filename="",fileobj=buffer,mode="wb",compresslevel=6,mtime=0) as compressed:
        compressed.write(data)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(buffer.getvalue())


def plot_results(output,movement,motion,main,station_names,config,period):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    tz=ZoneInfo("Europe/Moscow")
    origin=datetime.fromisoformat(period["start"])
    at=lambda seconds:origin+timedelta(seconds=seconds)
    positions={s:i for i,s in enumerate(config["station_ids"])}
    tracks={t["track_id"]:t for t in movement["tracks"]}
    initial=main["initial_car_loads"]
    candidates=[(n,t) for t,loads in initial.items() for n in loads]+[(r["load"],r["train_id"]) for r in main["load_events"]]
    peak,selected=max(candidates,key=lambda pair:(pair[0],pair[1]))
    label=f"S1 · {period['date']} · 08:00–10:00 Europe/Moscow"
    def clock(ax):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,20,40],tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",tz=tz))
        ax.set_xlim(at(7200),at(14400)); ax.set_xlabel("Время, Europe/Moscow")
    def save(fig,name):
        fig.savefig(output/name,dpi=150,metadata={"Software":"HackathonSPB line1-experiment-v1"})
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,8),layout="constrained")
    for train in movement["trains"]:
        events=[e for e in motion["events"] if e["train_id"]==train["train_id"]]
        if train["state"]=="moving":
            track=tracks[train["location"]]; fraction=1-train["initial_movement"]["arrival_second"]/track["travel_seconds"]
            position=positions[track["from_station"]]+fraction*(positions[track["to_station"]]-positions[track["from_station"]])
        else: position=positions[train["location"]]
        xs,ys=[at(0)],[position]
        for e in events:
            station=tracks[e["location"]]["from_station"] if e["event_type"]=="departure" else e["location"]
            position=positions[station]; xs.append(at(e["second"])); ys.append(position)
        # A leg crossing 10:00 is drawn to its known arrival beyond the axes.
        for leg in motion["unfinished_movements"]:
            if leg["train_id"]==train["train_id"]:
                xs.append(at(leg["arrival_second"])); ys.append(positions[tracks[leg["track_id"]]["to_station"]])
        highlighted=train["train_id"]==selected
        ax.plot(xs,ys,color="#c64632" if highlighted else "#386b89",alpha=1 if highlighted else .38,
                linewidth=1.6 if highlighted else .55,label=selected if highlighted else None)
    ax.set_yticks(range(19),[station_names[s] for s in config["station_ids"]],fontsize=8)
    ax.set_ylabel("Положение: порядковый номер станции")
    ax.set_title(f"Заданный сценарный график движения · {label}\n{len(movement['trains'])} состава; времена перегонов и остановок assumed")
    ax.grid(alpha=.2); ax.legend(loc="upper left"); clock(ax); save(fig,"S1_movement.png")
    fig,axes=plt.subplots(1,2,figsize=(17,8),layout="constrained",sharey=True)
    records={(r["second"],r["station_id"],r["direction"]):r["queued"] for r in main["queue_samples"]}
    times=list(range(7200,14400,60))
    matrices=[np.array([[records[t,s,d] for t in times] for s in config["station_ids"]]) for d in (1,-1)]
    limit=max(int(m.max()) for m in matrices)
    for ax,d,matrix in zip(axes,(1,-1),matrices):
        picture=ax.imshow(matrix,origin="lower",aspect="auto",vmin=0,vmax=max(1,limit),cmap="YlOrRd",
            extent=(mdates.date2num(at(7200)),mdates.date2num(at(14400)),-.5,18.5),interpolation="nearest")
        ax.set_title("К Девяткино" if d==1 else "К Проспекту Ветеранов")
        ax.set_yticks(range(19),[station_names[s] for s in config["station_ids"]],fontsize=8); clock(ax)
    axes[0].set_ylabel("Станция")
    fig.colorbar(picture,ax=axes,label="Очередь, человек; снимки раз в минуту")
    fig.suptitle(f"Очереди по направлениям · {label}\nПересадочный приток не учтён; пики между снимками приведены в таблице")
    save(fig,"S1_queues.png")
    fig,ax=plt.subplots(figsize=(13,5),layout="constrained")
    for i,kind in enumerate(config["car_types"]):
        car_id=f"C{i+1:02}"
        rows=[r for r in main["load_events"] if r["train_id"]==selected and r["car_id"]==car_id]
        xs=[at(7200)]+[at(r["second"]) for r in rows]+[at(14400)]
        ys=[initial[selected][i]]+[r["load"] for r in rows]
        ys.append(ys[-1])
        ax.step(xs,ys,where="post",linewidth=1.1,label=f"{car_id}, {kind}")
    ax.axhline(120,color="#ac2525",linestyle="--",linewidth=1.8,label="Штатный порог 120")
    ax.set_ylabel("Загрузка вагона, человек")
    ax.set_title(f"{selected}, максимум {peak} человек в вагоне · {label}\nСостояние сохраняется между событиями; посадка в наименее загруженный вагон")
    ax.grid(alpha=.2); ax.legend(ncol=3,fontsize=8,loc="upper right"); clock(ax); save(fig,"S1_car_loads.png")
    return {"selected_train_id":selected,"selected_train_max_car_load":peak,
            "clock_timezone":"Europe/Moscow","queue_sample_seconds":60,
            "load_interpolation":"piecewise constant known state; missing events never replaced by zero"}


def _safe_output(output):
    output=Path(output).resolve()
    protected=[ROOT/p for p in ("data/raw","docs","tests","examples","config","metro_data",".venv",
        "data/processed/metro","data/processed/metro_repeat","data/processed/economics","data/processed/economics_repeat",
        "data/processed/economics_audit","data/processed/simulation","data/processed/simulation_repeat",
        "data/processed/simulation_audit","data/processed/passengers","data/processed/passengers_repeat",
        "data/processed/passengers_verification","data/processed/line1_verification","data/processed/line1_audit")]
    if ROOT.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("line experiment output overlaps protected files")
    if output.exists() and any(output.iterdir()):
        path=output/"build_manifest.json"
        if not path.is_file(): raise ValueError("foreign output directory")
        owner=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(owner,dict) or owner.get("model_version")!=MODEL_VERSION:
            raise ValueError("foreign output manifest")
        legacy={f"{s}/{n}" for s in ("S1","S2","S3") for n in ("warmup_checkpoint.json","final_checkpoint.json")}
        names=OUTPUTS|legacy|{"build_manifest.json"}
        paths=list(output.rglob("*"))
        if any(not p.resolve().is_relative_to(output) or p.is_file() and p.stat().st_nlink>1 for p in paths):
            raise ValueError("output contains a link to files outside its ownership")
        actual={p.relative_to(output).as_posix() for p in paths if p.is_file()}
        if actual-names: raise ValueError("foreign files in output directory")
    return output


def _source_review():
    review=json.loads((ROOT/"config/line1/source_review.json").read_text(encoding="utf-8"))
    registry=pd.read_csv(ROOT/"docs/reference/manifest_sources.csv").set_index("google_drive_id")
    new=[r for r in review["files"] if r["id"] not in registry.index]
    changed=[r["id"] for r in review["files"] if r["id"] in registry.index and
             (str(registry.loc[r["id"],"modified_time"])!=r["modified_time"] or int(registry.loc[r["id"],"size_bytes"])!=r["size_bytes"])]
    return {**review,"new_files_in_listed_folders":new,"changed_metadata_since_registry":changed,
            "local_raw_file_count":sum(p.is_file() for p in (ROOT/"data/raw/metro").rglob("*"))}


def _input_files(config_path):
    fixed=("config/line1/source_review.json","requirements-experiment.txt","line1.py","passenger_groups.py",
        "passengers.py","simulation.py","economics.py","config/costs/default.json",
        "data/processed/metro/entries_station_15m.parquet","data/processed/metro/parameters.json",
        "data/processed/metro/vestibule_map.csv","data/processed/metro/sources.csv",
        "data/processed/metro/source_text.json","data/processed/metro/build_manifest.json",
        "docs/reference/manifest_sources.csv")
    source_manifest=json.loads((ROOT/"data/processed/metro/build_manifest.json").read_text(encoding="utf-8"))
    return [Path(config_path).resolve()]+[ROOT/p for p in fixed]+[ROOT/p for p in source_manifest["inputs"] if p.startswith("data/raw/")]


def run_experiment(output=None,config_path=None):
    from metro_data.build import check as check_data
    from passenger_groups import GroupedPassengerSimulation
    began=time.perf_counter(); timing={}
    output=_safe_output(output or ROOT/"data/processed/line1")
    config_path=Path(config_path or ROOT/"config/line1/experiment.json").resolve()
    if config_path.is_relative_to(output): raise ValueError("configuration input overlaps experiment output")
    output.mkdir(parents=True,exist_ok=True)
    try:
        config=load_config(config_path)
        check_data(ROOT,ROOT/"data/processed/metro")
        frame=pd.read_parquet(ROOT/"data/processed/metro/entries_station_15m.parquet")
        selected,period=select_period(frame,config["station_ids"],config["date"])
        groups=minute_groups(selected,config["station_ids"],pd.Timestamp(period["start"]),pd.Timestamp(period["end"]),config["platform_delay_seconds"])
        mapping=pd.read_csv(ROOT/"data/processed/metro/vestibule_map.csv").drop_duplicates("station_id").set_index("station_id")
        names={s:mapping.loc[s,"station_name"] for s in config["station_ids"]}
        movement,plan_meta=build_movement(config,date=period["date"])
        profiles=car_profiles(config,[t["train_id"] for t in movement["trains"]])
        main_plan,_=build_movement(config,date=period["date"],phase_offset=7200)
        main_motion=simulate(main_plan,"line1-baseline")
        economy=main_economics(main_motion)
        timing["prepare_seconds"]=time.perf_counter()-began
        results,comparison,work_hashes={},[],[]
        with TemporaryDirectory(prefix="line1-") as directory:
            stage=Path(directory)
            for scenario in ("S1","S2","S3"):
                scenario_started=time.perf_counter()
                assigned=assign_destinations(groups,config["station_ids"],scenario)
                runner=GroupedPassengerSimulation(movement,"line1-baseline",assigned,profiles)
                warm=runner.run(end_second=7200)
                main=runner.run(start_second=7200,checkpoint=warm["checkpoint"])
                folder=stage/scenario; folder.mkdir()
                write_json(folder/"metrics.json",{"scenario":scenario,"period":period,"main":main["metrics"],
                    "warmup":warm["metrics"],"transfer_inflow":"пересадочный приток не учтён",
                    "OD_status":"assumed engineering scenario","release_authorized":False})
                write_checkpoint(folder/"warmup_checkpoint.json.gz",warm["checkpoint"])
                write_checkpoint(folder/"final_checkpoint.json.gz",main["checkpoint"])
                write_json(folder/"initial_car_loads.json",main["initial_car_loads"])
                write_json(folder/"economics.json",economy)
                write_csv(folder/"queues_end.csv",main["queues"],("group_id","origin","destination","direction","turnstile_second","platform_second","count","waiting_age_seconds"))
                write_csv(folder/"approaching_end.csv",({k:g[k] for k in ("group_id","origin","turnstile_second","platform_second","count")} for g in main["approaching_groups"]),
                    ("group_id","origin","turnstile_second","platform_second","count"))
                write_csv(folder/"queue_samples.csv",main["queue_samples"])
                write_csv(folder/"car_load_events.csv",main["load_events"])
                write_csv(folder/"OD_groups.csv",({"group_id":g["group_id"],"turnstile_second":g["turnstile_second"],
                    "platform_second":g["platform_second"],"origin":g["origin"],"destination":d["station_id"],
                    "direction":d["direction"],"count":d["count"]} for g in assigned for d in g["destinations"]))
                m=main["metrics"]
                row={"scenario":scenario,"turnstile_entries":m["entered"],"platform_arrivals":m["platform_arrived"],
                    "approaching_end":m["approaching"],"boarded":m["boarded"],"completed":m["completed"],
                    "queue_end":m["queued"],"onboard_end":m["onboard"],"waiting_passenger_seconds":m["waiting_passenger_seconds"],
                    "mean_full_wait_seconds":m["mean_full_wait_seconds"],"p95_full_wait_seconds":m["p95_full_wait_seconds"],
                    "max_total_queue":m["max_queue"],"max_station_direction_queue":m["max_platform_direction_queue"],
                    "max_car_load":m["max_car_load"],"car_threshold_exceedance_observations":m["normal_threshold_exceedances"],
                    "balance_passed":m["balance"]["passed"],"money_status":economy["money_status"]}
                comparison.append(row); work_hashes.append(digest(runner.motion["transport_work"]))
                if scenario=="S1":
                    results["S1"]={"runner":runner,"main":main}
                    write_json(stage/"movement_full.simulation.json",runner.motion)
                timing[f"{scenario}_seconds"]=time.perf_counter()-scenario_started
            full_motion=results["S1"]["runner"].motion
            key=lambda e,offset:(e["second"]+offset,e["train_id"],e["event_type"],e["location"])
            continuity=[key(e,0) for e in full_motion["events"] if e["second"]>7200]==[key(e,7200) for e in main_motion["events"]]
            if len(set(work_hashes))!=1 or not continuity:
                raise ValueError("movement differs across OD scenarios or at main boundary")
            write_json(stage/"movement_input.json",movement)
            write_json(stage/"movement_main.simulation.json",main_motion)
            write_json(stage/"transport_work.json",{"period":period,"main":main_motion["transport_work"],
                "full":full_motion["transport_work"],"main_distance_status":"partial_assumed_geometry",
                "identical_all_scenarios":True,"full_transport_work_sha256_by_scenario":dict(zip(("S1","S2","S3"),work_hashes))})
            write_json(stage/"period.json",period)
            write_json(stage/"assumptions.json",{**config,"plan_checks":plan_meta,"source_review":_source_review(),
                "initial_state_influence_removed":False,"all_constraints_verified":False,"release_authorized":False})
            write_csv(stage/"minute_entries.csv",groups)
            write_csv(stage/"scenario_comparison.csv",comparison)
            write_json(stage/"scenario_comparison.json",comparison)
            checks={"source_coverage_passed":True,"source_interval_totals_preserved":sum(g["count"] for g in groups)==int(selected.entry_count.sum()),
                "external_entries_identical_all_scenarios":len({r["turnstile_entries"] for r in comparison})==1,
                "balances_passed":all(r["balance_passed"] for r in comparison),"movement_continuity_passed":continuity,
                "transport_work_identical_all_scenarios":True,"release_authorized":False,"all_constraints_verified":False,
                "source_entries_total_06_10":int(selected.entry_count.sum()),"minute_group_count":len(groups),
                "individual_passenger_records_created":0,"cycle_seconds":plan_meta["cycle_seconds"],"headway_seconds":plan_meta["headway_seconds"]}
            write_json(stage/"checks.json",checks)
            plot_started=time.perf_counter()
            chart_meta=plot_results(stage,movement,full_motion,results["S1"]["main"],names,config,period)
            write_json(stage/"plot_metadata.json",chart_meta)
            timing["plot_seconds"]=time.perf_counter()-plot_started
            summary=["# Эксперимент GOAL 3.3","",f"{period['date']}, Europe/Moscow: предварительный расчёт 06:00–08:00, оценка 08:00–10:00.",
                "","Исторические входы наблюдались через турникеты; назначения и движение заданы как инженерные сценарии. Пересадочный приток не учтён.",
                "","| Сценарий | Очередь в 10:00 | Ожидание в окне, пассажиро-секунды | Среднее полного ожидания, с | P95, с | Пик очереди станции/направления | Пик вагона, человек |",
                "|---|---:|---:|---:|---:|---:|---:|"]
            for r in comparison:
                average=f"{Decimal(r['mean_full_wait_seconds']):.2f}" if r["mean_full_wait_seconds"] is not None else "неизвестно"
                summary.append(f"| {r['scenario']} | {r['queue_end']} | {r['waiting_passenger_seconds']} | {average} | {r['p95_full_wait_seconds']} | {r['max_station_direction_queue']} | {r['max_car_load']} |")
            summary += ["","Источник D14: Юбилейный, 168 человек в головном вагоне, 187 в моторном и безмоторном промежуточном при 5 чел/м². Штатный порог 120 задан отдельно. Позиции промежуточных вагонов приняты условно.",
                "",scenario_parameters_text(config,plan_meta),
                "","Все пассажирские балансы выполнены с учётом начальных очередей, людей в вагонах и на подходе. Полное ожидание севших включает возраст очереди до 08:00; интеграл очереди относится только к основному окну. Предварительный расчёт не доказывает устранения влияния неизвестного состояния в 06:00.",
                "","Транспортная работа всех трёх вариантов одинакова. Пробег учитывается прежним журналом прибытия, с частичным статусом на границах. Денежный статус неизвестен: энергия, оплаченный труд и ставки не заданы. Экономия не рассчитывается, сравнивается один план при разных OD.",
                "","Неизвестны реальные OD, пересадочный поток, полное расписание, точные перегоны, распределение людей по вагонам и исходная загрузка. Посадка в наименее загруженный вагон может недооценивать реальную неравномерность. Допуск к выпуску не подтверждён.",
                "","![Движение S1](S1_movement.png)","","![Очереди S1](S1_queues.png)","","![Вагоны S1](S1_car_loads.png)"]
            write_text(stage/"summary.md","\n".join(summary)+"\n")
            input_files=_input_files(config_path)
            manifest={"model_version":MODEL_VERSION,"status":"success",
                "configuration_input":str(config_path),
                "inputs":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
                "outputs":{p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob("*")) if p.is_file()}}
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    target=output/path.relative_to(stage); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
            for scenario in ("S1","S2","S3"):
                for name in ("warmup_checkpoint.json","final_checkpoint.json"):
                    old=(output/scenario/name).resolve()
                    if not old.is_relative_to(output): raise ValueError("old checkpoint escapes output")
                    old.unlink(missing_ok=True)
            write_json(output/"build_manifest.json",manifest)
        timing["total_seconds"]=time.perf_counter()-began
        timing["output_bytes"]=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        timing["output_files"]=sum(p.is_file() for p in output.rglob("*"))
        # Runtime varies; keep measurements outside the deterministic artifact set.
        write_json(output.parent/f"{output.name}_performance.json",timing)
        return {"output":str(output),"period":period,"comparison":comparison,"performance":timing,"checks":checks}
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as exc:
        write_json(output/"build_manifest.json",{"model_version":MODEL_VERSION,"status":"error","error":str(exc)})
        raise


def check_output(output):
    output=Path(output).resolve()
    manifest=json.loads((output/"build_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest,dict) or manifest.get("model_version")!=MODEL_VERSION or manifest.get("status")!="success":
        raise ValueError("no current successful line1 manifest")
    if (not isinstance(manifest.get("inputs"),dict) or not manifest["inputs"]
            or not isinstance(manifest.get("outputs"),dict) or set(manifest["outputs"])!=OUTPUTS):
        raise ValueError("incomplete line1 manifest")
    configuration=Path(manifest.get("configuration_input",""))
    if not configuration.is_absolute(): raise ValueError("missing explicit configuration input")
    if set(manifest["inputs"])!={str(p.resolve()) for p in _input_files(configuration)}:
        raise ValueError("manifest lacks the complete program and source input set")
    load_config(configuration)
    actual={p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual!=set(manifest["outputs"])|{"build_manifest.json"}: raise ValueError("line1 artifact set changed")
    for raw,expected in manifest["inputs"].items():
        p=Path(raw)
        if not p.is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError(f"input changed: {raw}")
    for raw,expected in manifest["outputs"].items():
        p=output/raw
        if not p.resolve().is_relative_to(output) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"output changed: {raw}")
    return {"status":"passed","files":len(actual),"release_authorized":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","check"))
    parser.add_argument("--output",type=Path,default=ROOT/"data/processed/line1")
    parser.add_argument("--config",type=Path,default=ROOT/"config/line1/experiment.json")
    args=parser.parse_args(argv)
    try:
        result=run_experiment(args.output,args.config) if args.command=="run" else check_output(args.output)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as exc:
        print(f"line1: {exc}",file=sys.stderr); return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
