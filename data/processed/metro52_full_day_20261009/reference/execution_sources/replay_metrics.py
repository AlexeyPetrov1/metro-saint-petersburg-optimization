"""Passenger cohorts, time-weighted overload and movement-ledger diagnostics."""

from collections import Counter, defaultdict
from decimal import Decimal

from economics import number


def movement_snapshot(data,motion,second):
    """Snapshot just before boundary events, matching passenger checkpoints."""
    rows=[]; by_train=defaultdict(list)
    for event in motion["events"]: by_train[event["train_id"]].append(event)
    for initial in data["trains"]:
        events=by_train[initial["train_id"]]
        past=[e for e in events if e["second"]<second]
        state=initial["state"] if not past else past[-1]["next_state"]
        location=initial["location"] if not past else past[-1]["location"]
        obligation=None
        if state=="moving":
            arrival=next(e for e in events if e["event_type"]=="arrival" and e["second"]>=second)
            obligation=dict(kind="movement",track_id=location,arrival_second=arrival["second"],trip_id=arrival["trip_id"])
        elif state in ("stop","turnaround"):
            ready=next(e for e in events if e["event_type"]==state+"_complete" and e["second"]>=second)
            obligation=dict(kind=state,location=location,ready_second=ready["second"])
        rows.append(dict(train_id=initial["train_id"],state=state,location=location,unfinished_obligation=obligation))
    return dict(second=second,boundary_events="pending",trains=rows,
        future_timing=[r for r in data.get("leg_timing",[]) if r["departure_second"]>=second],
        future_plan=[t for p in data["plans"] for t in p["trips"] if t["departure_second"]>=second],
        release_authorized=False)


def queue_area(state,start,end):
    area=0; since=0; count=0
    for event in state["queue_events"]+[dict(second=end,queued=0)]:
        stop=min(event["second"],end)
        area+=count*max(0,stop-max(start,since))
        if event["second"]>=end: break
        since=event["second"]; count=event["queued"]
    return area


def cohort_waiting(runner,state,start,end):
    ids={g for g,row in runner.by_id.items() if start<=row["turnstile_second"]<min(end,state["second"])}
    arrived=sum(runner.by_id[g]["count"] for g in ids)
    histogram=Counter()
    for row in state["board_events"]:
        if row["group_id"] in ids: histogram[row["wait_seconds"]]+=row["count"]
    boarded=sum(histogram.values()); area=sum(w*n for w,n in histogram.items())
    residual=sum(g["count"]*(state["second"]-g["platform_second"])
                 for queue in state["queues"].values() for g in queue if g["group_id"] in ids)
    rank=(95*boarded+99)//100; p95=None; cumulative=0
    for wait,count in sorted(histogram.items()):
        cumulative+=count
        if cumulative>=rank: p95=wait; break
    complete=arrived==boarded and state["second"]>=end
    return dict(arrived=arrived,boarded=boarded,unboarded=arrived-boarded,
        waiting_passenger_seconds_lower_bound=area+residual,complete=complete,
        mean_wait_seconds=area/boarded if complete and boarded else None,
        p95_wait_seconds=p95 if complete and boarded else None,
        boarded_mean_wait_seconds=area/boarded if boarded else None,
        boarded_p95_wait_seconds=p95,scope="all turnstile arrivals in [start,end); platform waiting through saved end")


def overload_metrics(runner,state,start,end):
    loads=defaultdict(list)
    for row in state["load_events"]: loads[(row["train_id"],row["car_id"])].append(row)
    exceeded=passenger_area=excess_area=0; maximum=0
    for rows in loads.values():
        prior=0; since=0; threshold=rows[0]["normal_threshold"]
        for row in rows+[dict(second=end,load=0)]:
            stop=min(row["second"],end)
            duration=max(0,stop-max(since,start))
            if duration:
                maximum=max(maximum,prior)
                if prior>threshold:
                    exceeded+=duration; passenger_area+=duration*prior; excess_area+=duration*(prior-threshold)
            if row["second"]>=end: break
            prior=row["load"]; since=row["second"]
    return dict(max_car_load=maximum,exceeded_car_seconds=exceeded,
        passenger_seconds_in_exceeded_cars=passenger_area,excess_passenger_seconds=excess_area,
        affected_unique_passengers=None,
        definition="sum of car-time above the car threshold; passenger-seconds count every occupant during that time")


def peak_exchange(runner,state,start,end):
    events=runner.motion_by_id
    rows=[r for r in state["load_events"] if start<=r["second"]<end]
    if not rows: return None
    peak=max(rows,key=lambda r:r["load"]); train,car=peak["train_id"],peak["car_id"]
    motion=events[peak["motion_event_id"]]
    track=runner.tracks[motion["location"]] if motion["event_type"]=="departure" else None
    car_rows=[r for r in state["load_events"] if r["train_id"]==train and r["car_id"]==car]
    index=car_rows.index(peak); previous=car_rows[:index]
    arrival=next((r for r in reversed(previous) if r["event_type"]=="arrival"),None)
    departure=next((r for r in reversed(previous[:previous.index(arrival)] if arrival else previous)
                    if r["event_type"]=="departure"),None)
    before=departure["load"] if departure else 0
    after_arrival=arrival["load"] if arrival else before
    if motion["event_type"]=="arrival":
        before=previous[-1]["load"] if previous else 0; after_arrival=peak["load"]
    queue_before=queue_after=None
    if track:
        direction=1 if runner.positions[track["to_station"]]>runner.positions[track["from_station"]] else -1
        origin=track["from_station"]
        queue_before=sum(d["count"] for g in runner.by_id.values() if g["origin"]==origin and g["platform_second"]<=peak["second"]
            for d in g["destinations"] if d["direction"]==direction)
        for board in state["board_events"]:
            event=events[board["motion_event_id"]]; section=runner.tracks[event["location"]]
            same=section["from_station"]==origin and (runner.positions[section["to_station"]]>runner.positions[origin])==(direction==1)
            if same and (event["second"]<peak["second"] or event["second"]==peak["second"] and event["event_id"]<motion["event_id"]):
                queue_before-=board["count"]
        boarding=sum(r["count"] for r in state["board_events"] if r["motion_event_id"]==motion["event_id"])
        queue_after=queue_before-boarding
    return dict(second=peak["second"],timestamp=motion["timestamp"],train_id=train,car_id=car,
        direction=("up" if runner.positions[track["to_station"]]>runner.positions[track["from_station"]] else "down") if track else None,
        track_id=motion["location"] if track else None,capacity=peak["max_capacity"],normal_threshold=peak["normal_threshold"],
        load_before_arrival=before,alighted=before-after_arrival,
        boarded=peak["load"]-after_arrival if track else 0,load_after_departure=peak["load"],
        queue_before_boarding=queue_before,queue_after_boarding=queue_after,
        source="reconstructed from per-car load ledger, group destinations and boarding events")


def work_between(data,motion,start,end):
    tracks={t["track_id"]:t for t in data["tracks"]}; trains={t["train_id"]:t for t in data["trains"]}
    trips={t["trip_id"]:t for p in data["plans"] for t in p["trips"]}
    kilometers=dict.fromkeys(("passenger_train_km","nonpassenger_train_km","passenger_car_km","nonpassenger_car_km"),Decimal(0))
    state_seconds=Counter(); by_train=defaultdict(list)
    for event in motion["events"]:
        by_train[event["train_id"]].append(event)
        if event["event_type"]=="arrival" and start<event["second"]<=end:
            # Arrival location is a station; the last departure identifies its directed track.
            prior=next((e for e in reversed(by_train[event["train_id"]][:-1]) if e["event_type"]=="departure"),None)
            track=prior["location"] if prior else trains[event["train_id"]]["location"]
            passenger=trips[event["trip_id"]]["passenger"] if event["trip_id"]!="@initial" else trains[event["train_id"]]["initial_movement"]["passenger"]
            prefix="passenger" if passenger else "nonpassenger"
            km=Decimal(str(tracks[track]["length_km"]))
            kilometers[prefix+"_train_km"]+=km
            kilometers[prefix+"_car_km"]+=km*trains[event["train_id"]]["cars"]
    for train in trains.values():
        state=train["state"]; since=0
        for event in by_train[train["train_id"]]+[dict(second=end,next_state=state)]:
            second=min(event["second"],end)
            state_seconds[state]+=max(0,second-max(since,start))
            if event["second"]>=end: break
            since=event["second"]; state=event["next_state"]
    holding=sum(max(0,min(r["departure_second"],end)-max(start,r["departure_second"]-r.get("holding_seconds",0)))
                for r in data.get("leg_timing",[]))
    if holding>state_seconds["standby"]: raise ValueError("holding exceeds recorded standby time")
    state_seconds["holding"]=holding; state_seconds["standby"]-=holding
    if any(r.get("approach_holding_seconds", 0) for r in data.get("leg_timing", [])):
        approach = 0
        for row in data["leg_timing"]:
            hold = row.get("approach_holding_seconds", 0)
            track = (trains[row["train_id"]]["location"] if row["trip_id"] == "@initial"
                     else trips[row["trip_id"]]["route"][row["leg_index"]])
            at_signal = row["departure_second"] + tracks[track]["travel_seconds"]
            approach += max(0, min(at_signal + hold, end) - max(start, at_signal))
        if approach > state_seconds["moving"]: raise ValueError("approach holding exceeds movement time")
        state_seconds["approach_holding"] = approach
        state_seconds["moving"] -= approach
    kilometers["train_km"]=kilometers["passenger_train_km"]+kilometers["nonpassenger_train_km"]
    kilometers["car_km"]=kilometers["passenger_car_km"]+kilometers["nonpassenger_car_km"]
    return {**{k:number(v) for k,v in kilometers.items()},"state_seconds":dict(state_seconds),
        "distance_status":"partial arrival ledger; boundary movements retained separately",
        "energy_kWh":None,"paid_person_hours":None,"cash_total_RUB":None}

