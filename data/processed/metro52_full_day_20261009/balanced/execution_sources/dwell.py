"""Keyed service times and a precedence compiler for the existing simulator."""

from __future__ import annotations

import copy
import hashlib
import heapq
import random
from collections import defaultdict

from simulation import integer, simulate


def timing_key(train, trip, index):
    return (train,trip,index)


def keyed_service(profile, seed, key, nominal):
    integer(seed,"seed"); integer(nominal,"nominal service")
    settings = {"profile":profile} if isinstance(profile,str) else profile
    name = settings["profile"]
    if name == "fixed": return nominal
    if name == "constant": return integer(settings["seconds"],"service seconds")
    bounds = {"mild":(15,20,25),"wider":(10,20,30)}
    if name not in bounds: raise ValueError("unknown dwell profile")
    low,mode,high = bounds[name]
    value = hashlib.sha256(repr(("dwell-v1",seed,tuple(key))).encode()).digest()
    sampled = random.Random(int.from_bytes(value,"big")).triangular(low,high,mode)
    return max(settings.get("technical_minimum_seconds",0),int(sampled+0.5))


def compile_timing(data, plan_id, profile="fixed", seed=0, *, committed=None, as_of=0):
    """Preserve directed-track order, propagate service and headway delays."""
    integer(seed,"seed"); integer(as_of,"as_of")
    name = profile if isinstance(profile,str) else profile["profile"]
    if name == "fixed" and committed is None and not data.get("leg_timing"):
        # Original regular plans need no compiler and remain byte-compatible.
        try:
            simulate(data,plan_id)
            return copy.deepcopy(data),dict(profile=name,seed=seed,holding_seconds=0,
                service_mean_seconds=None,service_seconds=[],infrastructure_status="unknown",status="assumed")
        except ValueError as exc:
            if "headway" not in str(exc): raise
    original = copy.deepcopy(data); original.pop("leg_timing",None)
    stations={s["station_id"]:s for s in original["stations"]}
    tracks={t["track_id"]:t for t in original["tracks"]}
    plan=next(p for p in original["plans"] if p["plan_id"]==plan_id)
    old={timing_key(r["train_id"],r["trip_id"],r["leg_index"]):r for r in (committed or data.get("leg_timing",[]))}
    nodes={}; by_train=defaultdict(list); by_track=defaultdict(list)
    initial_ready={}; predecessor={}

    def add(train,trip,index,track,departure,initial=False,first=False):
        key=timing_key(train,trip,index); section=tracks[track]; station=stations[section["to_station"]]
        wait=station["turnaround_seconds"] or station["stop_seconds"]
        service=wait if station["turnaround_seconds"] or station["kind"]=="depot" else keyed_service(profile,seed,key,wait)
        prior=old.get(key); frozen=bool(prior and prior["departure_second"]<as_of)
        if prior and prior["departure_second"]+section["travel_seconds"]<as_of:
            service=prior["service_seconds"]
        anchored=initial or first and not by_train[train]
        earliest=departure if anchored else -10**12
        if not initial and not frozen: earliest=max(earliest,as_of)
        if frozen: earliest=prior["departure_second"]
        nodes[key]=dict(key=key,train=train,trip=trip,index=index,track=track,nominal=departure,
            travel=section["travel_seconds"],service=service,earliest=earliest,initial=initial,
            frozen=frozen,anchored=anchored,order=prior["departure_second"] if prior else departure)
        by_train[train].append(key); by_track[track].append(key)
        return departure+section["travel_seconds"]+wait

    for train in original["trains"]:
        name_train=train["train_id"]; ready=train["ready_after_seconds"]
        if train["state"]=="moving":
            track=train["location"]; section=tracks[track]
            departure=train["initial_movement"]["arrival_second"]-section["travel_seconds"]
            ready=add(name_train,"@initial",0,track,departure,True)
        initial_ready[name_train]=ready
        for trip in sorted((t for t in plan["trips"] if t["train_id"]==name_train),key=lambda t:(t["departure_second"],t["trip_id"])):
            departure=trip["departure_second"]
            for i,track in enumerate(trip["route"]):
                departure=add(name_train,trip["trip_id"],i,track,departure,first=i==0)

    edges=defaultdict(dict); incoming=dict.fromkeys(nodes,0)
    def edge(a,b,delay):
        if b not in edges[a]: incoming[b]+=1
        edges[a][b]=max(edges[a].get(b,0),delay)

    for train,keys in by_train.items():
        for a,b in zip(keys,keys[1:]): edge(a,b,nodes[a]["travel"]+nodes[a]["service"])
        if keys and not nodes[keys[0]]["initial"]:
            nodes[keys[0]]["earliest"]=max(nodes[keys[0]]["earliest"],initial_ready[train])
        predecessor.update({b:a for a,b in zip(keys,keys[1:])})
    history={r["track_id"]:r["departure_second"] for r in original.get("previous_departures",[])}
    for track,keys in by_track.items():
        keys.sort(key=lambda k:(nodes[k]["order"],k))
        for a,b in zip(keys,keys[1:]): edge(a,b,113)
        if keys and track in history:
            first=nodes[keys[0]]
            if not (first["initial"] and first["nominal"]==history[track]):
                first["earliest"]=max(first["earliest"],history[track]+113)
    ready=[k for k,n in incoming.items() if not n]; heapq.heapify(ready)
    times={k:n["earliest"] for k,n in nodes.items()}; seen=0
    while ready:
        key=heapq.heappop(ready); node=nodes[key]; seen+=1
        if (node["initial"] and times[key]!=node["nominal"]
                or node["frozen"] and times[key]!=old[key]["departure_second"]):
            raise ValueError("regulation would change committed movement")
        for nxt,delay in edges[key].items():
            times[nxt]=max(times[nxt],times[key]+delay)
            incoming[nxt]-=1
            if incoming[nxt]==0: heapq.heappush(ready,nxt)
    if seen!=len(nodes): raise ValueError("infeasible cyclic precedence constraints")
    rows=[]; holds=0; service=[]
    for key,node in sorted(nodes.items()):
        previous=predecessor.get(key)
        physical=(times[previous]+nodes[previous]["travel"]+nodes[previous]["service"]) if previous else initial_ready[node["train"]]
        planned=node["nominal"] if node['anchored'] else -10**12
        unheld=max(physical,planned,as_of if not node["frozen"] else -10**12)
        holding=0 if node["initial"] else max(0,times[key]-unheld)
        holds+=holding
        if stations[tracks[node["track"]]["to_station"]]["stop_seconds"]:
            service.append(node["service"])
        rows.append(dict(train_id=node["train"],trip_id=node["trip"],leg_index=node["index"],
            departure_second=times[key],service_seconds=node["service"],holding_seconds=holding))
    original["leg_timing"]=rows
    simulate(original,plan_id)  # The shared movement validator remains authoritative.
    return original,dict(profile=name,seed=seed,holding_seconds=holds,service_seconds=service,
        service_mean_seconds=sum(service)/len(service) if service else None,infrastructure_status="unknown",
        status="assumed",random_process="dwell-v1; sha256(seed,train_id,trip_id,leg_index); no mutable RNG cursor")


def regular_plan_at_limit(config,count,**kwargs):
    from decision_engine import build_regular_plan
    sections=len(config['station_ids'])-1
    cycle=2*(sections*config['travel_seconds']+(sections-1)*config['stop_seconds']+config['turnaround_seconds'])
    extra=max(0,113*count-cycle)
    return build_regular_plan(config,count,terminal_holds=(extra//2,extra-extra//2),**kwargs)


def compile_operational_timing(data, plan_id, profile="fixed", seed=0, *, policy,
                               incidents=(), as_of=0, departure_not_before=None):
    """Compile departure/arrival precedence over the shared movement engine."""
    from datetime import timedelta
    from goal43_policy import minimum_headway
    from simulation import instant

    integer(seed, "seed"); integer(as_of, "as_of")
    out = copy.deepcopy(data)
    stations = {r["station_id"]: r for r in out["stations"]}
    tracks = {r["track_id"]: r for r in out["tracks"]}
    plan = next(p for p in out["plans"] if p["plan_id"] == plan_id)
    old = {timing_key(r["train_id"], r["trip_id"], r["leg_index"]): r for r in out.get("leg_timing", [])}
    out.pop("leg_timing", None)
    out["headway_policy"] = copy.deepcopy(policy)
    start = instant(out["period_start"])
    nodes, legs = {}, {}
    train_keys, track_keys, platform_keys = defaultdict(list), defaultdict(list), defaultdict(list)
    fixed, lower, edges, incoming = {}, {}, defaultdict(dict), {}

    def variable(key, earliest, freeze=None):
        nodes[key] = True; lower[key] = earliest; incoming[key] = 0
        if freeze is not None: fixed[key] = freeze

    def edge(a, b, delay):
        if b not in edges[a]: incoming[b] += 1
        edges[a][b] = max(edges[a].get(b, 0), delay)

    def direction(track):
        if track.endswith("_up"): return "up"
        if track.endswith("_down"): return "down"
        return track

    def add(train, trip, index, track_id, nominal, passenger, initial=False):
        key = timing_key(train, trip, index)
        track = tracks[track_id]; station = stations[track["to_station"]]
        prior = old.get(key)
        nominal_wait = station["turnaround_seconds"] or station["stop_seconds"]
        service = nominal_wait if station["turnaround_seconds"] or station["kind"] == "depot" else keyed_service(profile, seed, key, nominal_wait)
        old_d = prior["departure_second"] if prior else nominal
        old_a = old_d + track["travel_seconds"] + (prior.get("approach_holding_seconds", 0) if prior else 0)
        if prior and old_a < as_of: service = prior["service_seconds"]
        first = not train_keys[train]
        earliest = nominal if first else -10**12
        if not initial: earliest = max(earliest, as_of)
        freeze_d = old_d if initial or old_d < as_of else None
        if freeze_d is not None: earliest = freeze_d
        command = (departure_not_before or {}).get(key)
        if command is not None:
            integer(command, "departure lower bound")
            if freeze_d is not None and command > freeze_d:
                raise ValueError("command rewrites past departure")
            earliest = max(earliest, command)
        d, a = ("D",) + key, ("A",) + key
        variable(d, earliest, freeze_d)
        variable(a, old_a if old_a < as_of else -10**12, old_a if old_a < as_of else None)
        edge(d, a, track["travel_seconds"])
        row = dict(key=key, d=d, a=a, train=train, trip=trip, index=index,
                   track=track_id, travel=track["travel_seconds"], service=service,
                   nominal=nominal, initial=initial, passenger=passenger,
                   old_departure=old_d, old_arrival=old_a)
        legs[key] = row
        if train_keys[train]:
            previous = legs[train_keys[train][-1]]
            edge(previous["a"], d, previous["service"])
        train_keys[train].append(key); track_keys[track_id].append(key)
        if station["kind"] == "line": platform_keys[station["station_id"], direction(track_id)].append(key)
        return nominal + track["travel_seconds"] + nominal_wait

    for train in out["trains"]:
        train_id = train["train_id"]
        if train["state"] == "moving":
            track_id = train["location"]
            departure = train["initial_movement"]["arrival_second"] - tracks[track_id]["travel_seconds"]
            add(train_id, "@initial", 0, track_id, departure, train["initial_movement"]["passenger"], True)
        for trip in sorted((r for r in plan["trips"] if r["train_id"] == train_id), key=lambda r: (r["departure_second"], r["trip_id"])):
            second = trip["departure_second"]
            for index, track_id in enumerate(trip["route"]):
                second = add(train_id, trip["trip_id"], index, track_id, second, trip["passenger"])
        keys = train_keys[train_id]
        if keys and not legs[keys[0]]["initial"]:
            lower[legs[keys[0]]["d"]] = max(lower[legs[keys[0]]["d"]], train["ready_after_seconds"])

    next_departure = {}
    for keys in train_keys.values():
        next_departure.update({a: legs[b]["d"] for a, b in zip(keys, keys[1:])})
    visits = {}
    for platform, keys in platform_keys.items():
        for key in keys:
            leg = legs[key]; release = next_departure.get(key)
            visits[key] = dict(train=leg["train"], platform=platform, a=leg["a"],
                old_arrival=leg["old_arrival"], release=release,
                old_release=legs[release[1:]]["old_departure"] if release else None, initial=False)
    for train in out["trains"]:
        station = train["location"]
        if train["state"] not in {"stop", "turnaround"} or stations[station]["kind"] != "line": continue
        keys = train_keys[train["train_id"]]
        outgoing = direction(legs[keys[0]]["track"]) if keys else None
        incoming_directions = {direction(track_id) for track_id, track in tracks.items()
            if track["to_station"] == station and stations[track["from_station"]]["kind"] == "line"}
        preferred = outgoing if train["state"] == "stop" else {"up": "down", "down": "up"}.get(outgoing)
        if preferred in incoming_directions: incoming_direction = preferred
        elif len(incoming_directions) == 1: incoming_direction = next(iter(incoming_directions))
        else: raise ValueError(f"initial platform direction unknown for {train['train_id']} at {station}")
        key = timing_key(train["train_id"], "@platform", 0); anchor = ("P", train["train_id"])
        variable(anchor, 0, 0)
        release = legs[keys[0]]["d"] if keys else None
        platform = (station, incoming_direction)
        visits[key] = dict(train=train["train_id"], platform=platform, a=anchor, old_arrival=0,
            release=release, old_release=legs[keys[0]]["old_departure"] if keys else None, initial=True)
        platform_keys[platform].append(key)

    assignments = {}
    for platform, keys in platform_keys.items():
        keys.sort(key=lambda key: (visits[key]["old_arrival"], not visits[key]["initial"],
                                  legs[key]["old_departure"] if not visits[key]["initial"] else 0, key))
        demo = policy.get("demo", {})
        capacity = demo.get("platform_capacity_by_station", {}).get(platform[0], demo.get("platform_capacity_default", 1))
        integer(capacity, "directional platform capacity", 1)
        slots = []
        for index, key in enumerate(keys):
            if index: edge(visits[keys[index-1]]["a"], visits[key]["a"], 0)
            if len(slots) < capacity:
                slot = len(slots); slots.append(key)
            else:
                available = [i for i, before in enumerate(slots) if visits[before]["release"] is not None]
                if not available:
                    raise ValueError(f"infeasible platform {platform}: no actual release or clearance movement")
                slot = min(available, key=lambda i: (visits[slots[i]]["old_release"], slots[i]))
                before = visits[slots[slot]]
                edge(before["release"], visits[key]["a"], 1)
                slots[slot] = key
            assignments[key] = slot

    incident_rows = []
    for incident in incidents:
        detected = integer(incident["detected_second"], "incident detected")
        end = incident.get("expected_end_second")
        if end is None:
            end = max(detected, as_of) + 480
            end_status = "assumed rolling protection; recovery time unknown"
        else:
            integer(end, "expected incident end")
            if end < detected: raise ValueError("incident end precedes detection")
            end_status = "controller estimate"
        matches = [r for r in visits.values() if r["train"] == incident["train_id"]
                   and r["platform"] == (incident["station_id"], incident["direction"])
                   and r["old_arrival"] <= detected
                   and (r["old_release"] is None or r["old_release"] >= detected)]
        if not matches: raise ValueError("incident does not refer to a train at the directional platform")
        affected = max(matches, key=lambda r: r["old_arrival"])
        release = affected["release"]
        if release is None: raise ValueError("stopped platform has no supported release movement")
        lower[release] = max(lower[release], end)
        incident_rows.append({**copy.deepcopy(incident), "used_end_second": end, "end_status": end_status})

    history = {r["track_id"]: r["departure_second"] for r in out.get("previous_departures", [])}
    pairs = []
    for track_id, keys in track_keys.items():
        keys.sort(key=lambda key: (legs[key]["old_departure"], key))
        for before, after in zip(keys, keys[1:]):
            pairs.append((legs[before]["d"], legs[after]["d"], track_id))
            edge(legs[before]["d"], legs[after]["d"], 113)
        if keys and track_id in history and not any(legs[key]["initial"] for key in keys):
            first = legs[keys[0]]
            anchor = ("H", track_id)
            variable(anchor, history[track_id], history[track_id])
            edge(anchor, first["d"], 113)
            pairs.append((anchor, first["d"], track_id))

    def propagate():
        indegrees = dict(incoming)
        ready = [key for key, count in indegrees.items() if not count]; heapq.heapify(ready)
        times = dict(lower); seen = 0
        while ready:
            key = heapq.heappop(ready); seen += 1
            if key in fixed and times[key] != fixed[key]:
                raise ValueError(f"regulation would rewrite committed movement {key}: {times[key]} != {fixed[key]}")
            for after, elapsed in edges[key].items():
                times[after] = max(times[after], times[key] + elapsed)
                indegrees[after] -= 1
                if not indegrees[after]: heapq.heappush(ready, after)
        if seen != len(nodes): raise ValueError("infeasible cyclic platform precedence")
        return times

    for _ in range(len(nodes) + 1):
        times = propagate(); changed = False
        for before, after, track_id in pairs:
            minimum = minimum_headway(start + timedelta(seconds=times[after]), policy) if stations[tracks[track_id]["from_station"]]["kind"] == "line" else 113
            if times[after] < times[before] + minimum:
                lower[after] = max(lower[after], times[before] + minimum); changed = True
        if not changed: break
    else: raise ValueError("headway policy did not converge")

    rows = []; approach_total = holds = 0
    previous_by_train = {}
    for train_id, keys in train_keys.items():
        for key in keys:
            leg = legs[key]; departure = times[leg["d"]]
            previous = previous_by_train.get(train_id)
            physical = times[previous["a"]] + previous["service"] if previous else next(t["ready_after_seconds"] for t in out["trains"] if t["train_id"] == train_id)
            unheld = max(physical, leg["nominal"] if previous is None else -10**12)
            holding = 0 if leg["initial"] else max(0, departure - unheld)
            approach = times[leg["a"]] - departure - leg["travel"]
            approach_total += approach; holds += holding
            rows.append(dict(train_id=train_id, trip_id=leg["trip"], leg_index=leg["index"],
                             departure_second=departure, service_seconds=leg["service"],
                             holding_seconds=holding, approach_holding_seconds=approach))
            previous_by_train[train_id] = leg
    out["leg_timing"] = sorted(rows, key=lambda r: timing_key(r["train_id"], r["trip_id"], r["leg_index"]))
    simulate(out, plan_id)
    occupancy = [dict(train_id=visits[key]["train"], platform=list(visits[key]["platform"]),
        slot_index=assignments[key], arrival_second=times[visits[key]["a"]],
        release_second=times[visits[key]["release"]] if visits[key]["release"] else None,
        initial=visits[key]["initial"]) for key in visits]
    return out, dict(profile=profile, seed=seed, holding_seconds=holds,
                     approach_holding_seconds=approach_total, incidents=incident_rows,
                     platform_policy="directional platform; clearance 1 second; no overtaking",
                     terminal_clearance_assumptions=[], platform_occupancy=occupancy,
                     initial_platform_occupancy=[row for row in occupancy if row["initial"]],
                     infrastructure_status="assumed platform policy; signalling and switches unknown")

