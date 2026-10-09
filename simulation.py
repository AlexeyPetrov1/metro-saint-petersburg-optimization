"""Deterministic event simulation of prescribed trips; no dispatch authorization."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import heapq
import io
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from economics import (Evaluation, Quantity, compare, decimal, exact_sum, load_costs,
                       number, product, ratio, write_json, write_text)


MODEL_VERSION = "simulation-v1.1"
STATES = ("moving", "stop", "turnaround", "standby")
HEADWAY_SECONDS = 113
REQUIRED_CASH = ("purchased_energy", "paid_labor", "avoidable_maintenance",
                 "entry_preparation", "withdrawal_preparation")
UNKNOWN_UNITS = {"purchased_energy": "kWh", "paid_person_hours": "person_h",
                 "avoidable_maintenance_events": "event", "entry_events": "event",
                 "withdrawal_events": "event", "implementation_events": "event"}
ABSENT_CHECKS = ["passenger demand, queues and capacity", "block occupancy and signalling",
                 "switch routes and dispatcher permissions", "crew schedules and paid work",
                 "maintenance history, EO validity and regulatory applicability",
                 "real energy consumption, tariffs and actual train release readiness"]
OUTPUT_NAMES = {f"{plan}.{suffix}" for plan in ("baseline", "candidate")
                for suffix in ("simulation.json", "events.csv", "evaluation.json")} | {
                    "control_check.json", "economic_comparison.json", "assumptions.json", "summary.md"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def fields_only(data, required, optional=()):
    if not isinstance(data, dict):
        raise ValueError("record must be an object")
    extra = set(data) - set(required) - set(optional)
    missing = set(required) - set(data)
    if extra or missing:
        raise ValueError(f"unsupported fields {sorted(extra)}; missing fields {sorted(missing)}")


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} needs a nonempty string")
    return value


def indexed(rows, key):
    if not isinstance(rows, list):
        raise ValueError(f"{key} records must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"missing {key}")
        name = identifier(row[key], key)
        if name in result:
            raise ValueError(f"duplicate {key}: {name}")
        result[name] = row
    return dict(sorted(result.items()))


def instant(value):
    if not isinstance(value, str):
        raise ValueError("time must be a timezone-aware string")
    if any(int(fraction) for fraction in re.findall(r"[.,](\d+)", value)):
        raise ValueError("time must have whole seconds")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("time must include a timezone")
    if result.microsecond:
        raise ValueError("time must have whole seconds")
    return result


def simulate(data, plan_id):
    fields_only(data, ("schema_version", "source", "period_start", "period_end",
                      "stations", "tracks", "trains", "plans"),
                ("previous_departures", "max_trains_on_line", "leg_timing", "headway_policy", "withdrawal_stabling"))
    if data["schema_version"] != "1.0":
        raise ValueError("unsupported simulation schema")
    identifier(data["source"], "source")
    start, end = map(instant, (data["period_start"], data["period_end"]))
    if start.utcoffset() != end.utcoffset() or end <= start:
        raise ValueError("invalid period or timezone offsets")
    duration = (end - start).days * 86400 + (end - start).seconds
    stations = indexed(data["stations"], "station_id")
    tracks = indexed(data["tracks"], "track_id")
    trains = indexed(data["trains"], "train_id")
    plans = indexed(data["plans"], "plan_id")
    if not stations or not trains or plan_id not in plans:
        raise ValueError("stations, trains and selected plan must exist")
    for station in stations.values():
        fields_only(station, ("station_id", "kind", "stop_seconds", "turnaround_seconds"))
        if station["kind"] not in {"line", "depot", "siding"}:
            raise ValueError("station kind must be line, depot or siding")
        for key in ("stop_seconds", "turnaround_seconds"):
            integer(station[key], key)
        if station["stop_seconds"] and station["turnaround_seconds"]:
            raise ValueError("station cannot require both stop and turnaround")
    for track in tracks.values():
        fields_only(track, ("track_id", "from_station", "to_station", "length_km", "travel_seconds"))
        if track["from_station"] not in stations or track["to_station"] not in stations:
            raise ValueError("track station does not exist")
        if track["from_station"] == track["to_station"]:
            raise ValueError("track must connect distinct stations")
        if not isinstance(track["length_km"], (str, int)) or isinstance(track["length_km"], bool):
            raise ValueError("length_km must be an exact decimal string or integer")
        if decimal(track["length_km"]) <= 0:
            raise ValueError("track length must be positive")
        integer(track["travel_seconds"], "travel_seconds", 1)
    for plan in plans.values():
        fields_only(plan, ("plan_id", "trips"))
    trips = indexed(plans[plan_id]["trips"], "trip_id")
    for trip in trips.values():
        fields_only(trip, ("trip_id", "train_id", "departure_second", "route", "passenger"))
        if trip["train_id"] not in trains:
            raise ValueError("trip train does not exist")
        integer(trip["departure_second"], "departure_second")
        if type(trip["passenger"]) is not bool:
            raise ValueError("passenger must be boolean")
        if not isinstance(trip["route"], list) or not trip["route"]:
            raise ValueError("route must be a nonempty list of directed tracks")
        for track_id in trip["route"]:
            if not isinstance(track_id, str) or track_id not in tracks:
                raise ValueError(f"unknown route track: {track_id}")
        route = [tracks[t] for t in trip["route"]]
        if any(a["to_station"] != b["from_station"] for a, b in zip(route, route[1:])):
            raise ValueError("discontinuous route")

    limit = data.get("max_trains_on_line")
    if limit is not None:
        integer(limit, "line capacity", 1)
    previous = data.get("previous_departures", [])
    if not isinstance(previous, list):
        raise ValueError("previous_departures must be a list")
    departures = {key: set() for key in tracks}
    for row in previous:
        fields_only(row, ("track_id", "departure_second"))
        if row["track_id"] not in tracks or type(row["departure_second"]) is not int or row["departure_second"] >= 0:
            raise ValueError("previous departure needs a known track and negative integer second")
        departures[row["track_id"]].add(row["departure_second"])

    queue, legs, runtime = [], [], {}
    timing, used_timing = {}, set()
    for row in data.get("leg_timing", []):
        fields_only(row, ("train_id", "trip_id", "leg_index", "departure_second", "service_seconds"),
                    ("holding_seconds", "approach_holding_seconds"))
        key = (row["train_id"], row["trip_id"], integer(row["leg_index"], "leg index"))
        if key in timing or row["train_id"] not in trains or type(row["departure_second"]) is not int:
            raise ValueError("invalid or duplicate leg timing")
        integer(row["service_seconds"], "service seconds")
        if "holding_seconds" in row: integer(row["holding_seconds"], "holding seconds")
        if "approach_holding_seconds" in row: integer(row["approach_holding_seconds"], "approach holding seconds")
        timing[key] = row

    def wait_at(station_id):
        station = stations[station_id]
        if station["turnaround_seconds"]:
            return "turnaround", station["turnaround_seconds"]
        return "stop", station["stop_seconds"]

    def schedule(second, event_type, leg):
        priority = 0 if event_type == "arrival" else 2 if event_type == "departure" else 1
        heapq.heappush(queue, (second, priority, leg["train_id"], leg["trip_id"],
                               leg["leg_index"], event_type, leg))

    def append_leg(train_id, trip_id, leg_index, track_id, departure, passenger, initial=False):
        track = tracks[track_id]
        key = (train_id, trip_id, leg_index)
        override = timing.get(key)
        if override:
            if initial and override["departure_second"] != departure or override["departure_second"] < departure:
                raise ValueError("leg timing precedes physical or committed readiness")
            departure = override["departure_second"]
            used_timing.add(key)
        approach = override.get("approach_holding_seconds", 0) if override else 0
        arrival = departure + track["travel_seconds"] + approach
        wait_kind, wait_seconds = wait_at(track["to_station"])
        if override: wait_seconds = override["service_seconds"]
        leg = {"train_id": train_id, "trip_id": trip_id, "leg_index": leg_index,
               "track_id": track_id, "departure_second": departure,
               "arrival_second": arrival, "ready_second": arrival + wait_seconds,
               "wait_kind": wait_kind, "wait_seconds": wait_seconds,
               "passenger": passenger, "initial": initial}
        if approach:
            leg["approach_holding_seconds"] = approach
        legs.append(leg)
        if not initial:
            schedule(departure, "departure", leg)
        schedule(arrival, "arrival", leg)
        if wait_seconds:
            schedule(arrival + wait_seconds, wait_kind + "_complete", leg)
        return leg

    for train_id, train in trains.items():
        fields_only(train, ("train_id", "train_type", "cars", "location", "state", "ready_after_seconds"), ("initial_movement",))
        identifier(train["train_type"], "train_type")
        integer(train["cars"], "cars", 1)
        ready = integer(train["ready_after_seconds"], "ready_after_seconds")
        state, location = train["state"], train["location"]
        previous_track = None
        if state not in STATES:
            raise ValueError("unknown train state")
        if state == "standby" and ready:
            raise ValueError("standby train must already be ready")
        rt = {"state": state, "location": location, "since": 0, "movement": None, "ready_second": ready}
        if state == "moving":
            if location not in tracks or ready or "initial_movement" not in train:
                raise ValueError("initial moving train needs track location and initial_movement")
            movement = train["initial_movement"]
            fields_only(movement, ("arrival_second", "passenger"))
            arrival = integer(movement["arrival_second"], "initial arrival_second", 1)
            if arrival >= tracks[location]["travel_seconds"] or type(movement["passenger"]) is not bool:
                raise ValueError("initial movement must have departed before the window")
            leg = append_leg(train_id, "@initial", 0, location,
                             arrival - tracks[location]["travel_seconds"], movement["passenger"], True)
            rt["movement"] = leg
            rt["ready_second"] = leg["ready_second"]
            previous_track = tracks[location]
            ready, location = leg["ready_second"], tracks[location]["to_station"]
        else:
            if location not in stations or "initial_movement" in train:
                raise ValueError("initial station location does not exist or has inconsistent movement")
            if state in {"stop", "turnaround"}:
                expected, maximum = wait_at(location)
                if state != expected or ready <= 0 or ready > maximum:
                    raise ValueError("invalid initial stop or turnaround remaining time")
                schedule(ready, state + "_complete", {"train_id": train_id, "trip_id": "@initial",
                         "leg_index": 0, "ready_second": ready})
        runtime[train_id] = rt
        ordered = sorted((t for t in trips.values() if t["train_id"] == train_id),
                         key=lambda t: (t["departure_second"], t["trip_id"]))
        for trip in ordered:
            if trip["trip_id"] == "@initial":
                raise ValueError("reserved trip_id @initial")
            departure = trip["departure_second"]
            first_timing = timing.get((train_id, trip["trip_id"], 0))
            if first_timing:
                departure = first_timing["departure_second"]
            if departure < ready:
                raise ValueError(f"train {train_id} busy until {ready}; trip {trip['trip_id']} at {departure}")
            if tracks[trip["route"][0]]["from_station"] != location:
                raise ValueError(f"train {train_id} location does not match trip origin")
            for i, track_id in enumerate(trip["route"]):
                track = tracks[track_id]
                if (previous_track and previous_track["from_station"] == track["to_station"]
                        and not stations[location]["turnaround_seconds"]):
                    release = data.get('headway_policy', {}).get('parameters', {}).get('depot_release_s', {}).get('value', {})
                    depot = {'DEPOT_AVTOVO': 'Avtovo', 'DEPOT_NORTHERN': 'Northern'}.get(location)
                    seconds = release.get(depot)
                    if (stations[location]['kind'] != 'depot' or not track_id.startswith('ENTRY_')
                            or type(seconds) is not int or seconds <= 0):
                        raise ValueError(f"reversal needs a turnaround station at {location}")
                    lead = track['travel_seconds'] + stations[track['to_station']]['turnaround_seconds']
                    if departure < ready + seconds - lead:
                        raise ValueError(f"depot preparation at {location} needs {seconds} seconds")
                leg = append_leg(train_id, trip["trip_id"], i, track_id, departure, trip["passenger"])
                departure = leg["ready_second"]
                location = track["to_station"]
                previous_track = track
            ready = departure

    if used_timing != set(timing):
        raise ValueError("leg timing references an absent movement")

    # Preserve every actual departure, including negative initial departures.
    # A history row at the same instant is evidence of that event, not a new train.
    headways = {track_id: [] for track_id in tracks}
    for leg in legs:
        headways[leg["track_id"]].append((leg["departure_second"], leg["train_id"]))
    for track_id, seconds in departures.items():
        actual_times = {second for second, _ in headways[track_id]}
        headways[track_id] += [(second, "history") for second in seconds - actual_times]
    for track_id, values in headways.items():
        values.sort()
        for a, b in zip(values, values[1:]):
            minimum = HEADWAY_SECONDS
            if "headway_policy" in data and stations[tracks[track_id]["from_station"]]["kind"] == "line":
                from goal43_policy import minimum_headway
                minimum = minimum_headway(start + timedelta(seconds=b[0]), data["headway_policy"])
            if b[0] - a[0] < minimum:
                raise ValueError(f"headway on {track_id}: {b[0] - a[0]} < {minimum} seconds")

    work = {train_id: {"train_id": train_id, "train_type": train["train_type"], "cars": train["cars"],
            "passenger_train_km": Decimal(0), "nonpassenger_train_km": Decimal(0),
            "passenger_car_km": Decimal(0), "nonpassenger_car_km": Decimal(0),
            "state_seconds": dict.fromkeys(STATES, 0)} for train_id, train in trains.items()}
    log = []

    def on_line(rt):
        if rt["state"] == "moving":
            track = tracks[rt["location"]]
            return any(stations[track[k]]["kind"] == "line" for k in ("from_station", "to_station"))
        return stations[rt["location"]]["kind"] == "line"

    line_count = sum(on_line(rt) for rt in runtime.values())
    peak = line_count
    if limit is not None and peak > limit:
        raise ValueError(f"initial line capacity exceeded: {peak} > {limit}")
    while queue:
        second, _, train_id, _, _, kind, leg = heapq.heappop(queue)
        if second > duration:
            break
        # The closing instant completes arrivals and waits; a new departure at
        # that instant belongs outside this window and remains an obligation.
        if second == duration and kind == "departure":
            continue
        rt, row = runtime[train_id], work[train_id]
        was_on_line = on_line(rt)
        before = rt["state"]
        row["state_seconds"][before] += second - rt["since"]
        rt["since"] = second
        if kind == "departure":
            track = tracks[leg["track_id"]]
            if before != "standby" or rt["location"] != track["from_station"]:
                raise ValueError("departure from a busy train or wrong location")
            rt.update(state="moving", location=leg["track_id"], movement=leg, ready_second=leg["ready_second"])
        elif kind == "arrival":
            if before != "moving" or rt["movement"] != leg:
                raise ValueError("arrival does not match active movement")
            track = tracks[leg["track_id"]]
            prefix = "passenger" if leg["passenger"] else "nonpassenger"
            km = decimal(track["length_km"])
            for key, value in ((prefix + "_train_km", km),
                               (prefix + "_car_km", product(km, Decimal(trains[train_id]["cars"])))):
                row[key] = exact_sum((row[key], value))
            rt.update(state=leg["wait_kind"] if leg["wait_seconds"] else "standby",
                      location=track["to_station"], movement=None, ready_second=leg["ready_second"])
        else:
            if before != kind.removesuffix("_complete") or rt["ready_second"] != second:
                raise ValueError("wait completion does not match train state")
            rt["state"] = "standby"
        log.append({"event_id": f"{plan_id}:E{len(log) + 1:06d}",
                    "timestamp": (start + timedelta(seconds=second)).isoformat(),
                    "second": second, "train_id": train_id, "event_type": kind,
                    "location": rt["location"], "previous_state": before,
                    "next_state": rt["state"], "source_plan_id": plan_id,
                    "trip_id": leg["trip_id"]})
        line_count += int(on_line(rt)) - int(was_on_line)
        peak = max(peak, line_count)
        if limit is not None and peak > limit:
            raise ValueError(f"line capacity exceeded at {second}: {peak} > {limit}")

    final, unfinished, obligations = [], [], []
    for train_id, rt in runtime.items():
        row = work[train_id]
        row["state_seconds"][rt["state"]] += duration - rt["since"]
        if sum(row["state_seconds"].values()) != duration:
            raise ValueError("state time conservation failed")
        row["car_km"] = exact_sum((row["passenger_car_km"], row["nonpassenger_car_km"]))
        row["train_km"] = exact_sum((row["passenger_train_km"], row["nonpassenger_train_km"]))
        for key in list(row):
            if key.endswith("_km"):
                row[key] = number(row[key])
        final.append({"train_id": train_id, "location": rt["location"], "state": rt["state"],
                      "ready_to_depart": rt["state"] == "standby",
                      "ready_second": rt["ready_second"] if rt["state"] != "standby" else duration})
        if rt["movement"]:
            unfinished.append(dict(rt["movement"]))
            obligations.append({"kind": "movement", **rt["movement"]})
        if rt["state"] in {"stop", "turnaround"}:
            obligations.append({"kind": rt["state"], "train_id": train_id,
                                "location": rt["location"], "ready_second": rt["ready_second"]})
    for leg in sorted(legs, key=lambda x: (x["departure_second"], x["train_id"], x["trip_id"], x["leg_index"])):
        if leg["departure_second"] >= duration:
            obligations.append({"kind": "planned_leg", **leg})
    crossing = [dict(leg) for leg in legs if leg["initial"]]
    context = {"stations": list(stations.values()), "tracks": list(tracks.values()),
               "trains": list(trains.values()), "previous_departures": sorted(previous, key=canonical),
               "max_trains_on_line": limit, "previous_departures_supplied": "previous_departures" in data}
    checks = [{"name": name, "status": "passed"} for name in
              ("references and positive lengths/times", "route continuity and train location",
               "train exclusivity, stops and turnarounds", "directed track departure headways",
               "state time conservation", "arrival-derived distance ledger")]
    checks += [{"name": "initial departure boundary context", "status": "passed" if "previous_departures" in data else "unknown"},
               {"name": "line capacity", "status": "passed" if limit is not None else "not_configured"}]
    result = {"schema_version": "1.0", "model_version": MODEL_VERSION, "evaluation_mode": "fixture",
            "source": data["source"], "plan_id": plan_id, "initial_state_id": digest(context),
            "demand_id": "passengers-not-modeled", "period_start": start.isoformat(), "period_end": end.isoformat(),
            "duration_seconds": duration, "events": log, "transport_work": list(work.values()),
            "initial_state": copy.deepcopy(list(trains.values())), "final_state": final,
            "unfinished_movements": unfinished, "unfinished_obligations": obligations,
            "boundary_crossing_movements": crossing, "mileage_complete": not unfinished and not crossing,
            "obligations_closed": not obligations, "obligations_scope": "scheduled_movement_and_station_waits_only",
            "status": "partial" if obligations or crossing else "complete",
            "checks": {"performed": checks, "absent": list(ABSENT_CHECKS),
                       "supported_constraints_status": "passed" if all(c["status"] == "passed" for c in checks) else "partial",
                       "headway_seconds": HEADWAY_SECONDS, "peak_trains_on_line": peak},
            "passenger_impact_checked": False, "release_authorized": False}
    if "withdrawal_stabling" in data:
        from withdrawal_stabling import report
        result["withdrawal_stabling"] = report(data, result)
    return result


def validate_result(result):
    """Validate the recorded contract; this is not a second movement simulation."""
    if not isinstance(result, dict) or result.get("schema_version") != "1.0":
        raise ValueError("unsupported simulation result schema")
    if result.get("model_version") != MODEL_VERSION or result.get("evaluation_mode") != "fixture":
        raise ValueError("incompatible simulation result version or mode")
    if result.get("obligations_scope") != "scheduled_movement_and_station_waits_only":
        raise ValueError("unsupported simulation obligations scope")
    for key in ("plan_id", "source", "demand_id", "initial_state_id"):
        identifier(result.get(key), key)
    for key in ("mileage_complete", "obligations_closed"):
        if type(result.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if result.get("release_authorized") is not False or result.get("passenger_impact_checked") is not False:
        raise ValueError("fixture cannot establish passenger impact or release authorization")
    start, end = map(instant, (result["period_start"], result["period_end"]))
    duration = integer(result["duration_seconds"], "duration_seconds", 1)
    if start.utcoffset() != end.utcoffset() or end - start != timedelta(seconds=duration):
        raise ValueError("result period and duration disagree")
    work = indexed(result["transport_work"], "train_id")
    initial = indexed(result["initial_state"], "train_id")
    final = indexed(result["final_state"], "train_id")
    if not work or work.keys() != initial.keys() or work.keys() != final.keys():
        raise ValueError("result train ledgers disagree")
    for train_id, row in work.items():
        cars = integer(row["cars"], "cars", 1)
        if cars != initial[train_id]["cars"] or row["train_type"] != initial[train_id]["train_type"]:
            raise ValueError("result train metadata disagree")
        counts = row["state_seconds"]
        if not isinstance(counts, dict) or set(counts) != set(STATES):
            raise ValueError("invalid state seconds ledger")
        if sum(integer(v, "state seconds") for v in counts.values()) != duration:
            raise ValueError("state seconds do not match duration")
        quantities = {key: decimal(row[key]) for key in ("passenger_train_km", "nonpassenger_train_km",
                      "passenger_car_km", "nonpassenger_car_km", "train_km", "car_km")}
        if any(value < 0 for value in quantities.values()):
            raise ValueError("negative result mileage")
        for prefix in ("passenger", "nonpassenger"):
            if quantities[prefix + "_car_km"] != product(quantities[prefix + "_train_km"], Decimal(cars)):
                raise ValueError("result car mileage does not match train mileage and cars")
        for unit in ("train_km", "car_km"):
            if quantities[unit] != exact_sum(quantities[p + "_" + unit] for p in ("passenger", "nonpassenger")):
                raise ValueError("result total mileage disagrees with components")
        state = final[train_id]["state"]
        if state not in STATES or final[train_id]["ready_to_depart"] is not (state == "standby"):
            raise ValueError("invalid final readiness")
    for key in ("unfinished_movements", "unfinished_obligations", "boundary_crossing_movements"):
        if not isinstance(result[key], list):
            raise ValueError(f"{key} must be a list")
        if any(not isinstance(row, dict) or row.get("train_id") not in work for row in result[key]):
            raise ValueError(f"invalid train in {key}")
    obligations = result["unfinished_obligations"]
    moving_ids = {row["train_id"] for row in result["unfinished_movements"]}
    if moving_ids != {t for t, row in final.items() if row["state"] == "moving"}:
        raise ValueError("unfinished movements disagree with final states")
    for row in result["unfinished_movements"]:
        if row["arrival_second"] <= duration or not any(
                o.get("kind") == "movement" and all(o.get(k) == row.get(k) for k in row) for o in obligations):
            raise ValueError("unfinished movement is missing its obligation")
    for train_id, row in final.items():
        if row["state"] in {"stop", "turnaround"} and not any(
                o.get("kind") == row["state"] and o["train_id"] == train_id
                and o.get("ready_second") == row["ready_second"] > duration for o in obligations):
            raise ValueError("unfinished wait is missing its obligation")
    closed = not obligations
    complete_mileage = not result["unfinished_movements"] and not result["boundary_crossing_movements"]
    if result["obligations_closed"] != closed or result["mileage_complete"] != complete_mileage:
        raise ValueError("result completeness flags disagree with recorded obligations")
    if result["status"] != ("complete" if closed and complete_mileage else "partial"):
        raise ValueError("result status disagrees with completeness")


def to_evaluation(result):
    """Convert seconds to train-hours here only; cars were counted at arrival."""
    validate_result(result)
    source = f"{MODEL_VERSION}; {result['source']}; plan={result['plan_id']}; fixture"
    quantities = {}
    status = "assumed" if result["mileage_complete"] else "partial"
    for key in ("passenger_train_km", "nonpassenger_train_km", "passenger_car_km", "nonpassenger_car_km", "train_km", "car_km"):
        amount = exact_sum(decimal(row[key]) for row in result["transport_work"])
        quantities[key] = Quantity(amount, "car_km" if "car_km" in key else "train_km", source, status)
    for state in STATES:
        seconds = sum(row["state_seconds"][state] for row in result["transport_work"])
        quantities[state + "_hours"] = Quantity(ratio(Decimal(seconds), Decimal(3600)), "train_h", source, "assumed")
    stopped = sum(row["state_seconds"][s] for row in result["transport_work"] for s in ("stop", "turnaround"))
    quantities["stopped_hours"] = Quantity(ratio(Decimal(stopped), Decimal(3600)), "train_h", source, "assumed")
    for key, unit in UNKNOWN_UNITS.items():
        quantities[key] = Quantity(None, unit, "not modeled or supplied; no inference from train work")
    return Evaluation(result["plan_id"], result["demand_id"], result["initial_state_id"],
                      result["period_start"], result["period_end"], result["period_end"], quantities,
                      model_version=MODEL_VERSION, evaluation_mode="fixture",
                      obligations_closed=result["obligations_closed"], constraint_status="unknown",
                      required_cash_articles=REQUIRED_CASH)


def control_check(result):
    # Literal acceptance oracle, not derived from tracks, times or cars.
    expected = {"passenger_train_km": "6", "passenger_car_km": "48", "nonpassenger_train_km": "0",
                "moving": 600, "stop": 120, "turnaround": 240, "standby": 0,
                "total_seconds": 960, "final_location": "A", "ready_to_depart": True,
                "unfinished_movements": 0, "unfinished_obligations": 0}
    if len(result["transport_work"]) != 1:
        raise ValueError("control cycle requires one train")
    row, final = result["transport_work"][0], result["final_state"][0]
    actual = {**{key: row[key] for key in ("passenger_train_km", "passenger_car_km", "nonpassenger_train_km")},
              **row["state_seconds"], "total_seconds": sum(row["state_seconds"].values()),
              "final_location": final["location"], "ready_to_depart": final["ready_to_depart"],
              "unfinished_movements": len(result["unfinished_movements"]),
              "unfinished_obligations": len(result["unfinished_obligations"])}
    checks = [{"quantity": key, "expected": value, "actual": actual[key],
               "status": "passed" if value == actual[key] else "failed"} for key, value in expected.items()]
    return {"oracle_source": "GOAL 3.1 manual values for educational A-B-C cycle",
            "status": "passed" if all(r["status"] == "passed" for r in checks) else "failed", "checks": checks}


def csv_events(events):
    buffer = io.StringIO(newline="")
    columns = ("event_id", "timestamp", "second", "train_id", "event_type", "location",
               "previous_state", "next_state", "source_plan_id", "trip_id")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    return buffer.getvalue()


def safe_output(input_path, output, costs):
    input_path, output, costs = map(lambda p: Path(p).resolve(), (input_path, output, costs))
    root = Path(__file__).resolve().parent
    protected = [root / p for p in ("data/raw", "docs/reference", "data/processed/metro", "data/processed/metro_repeat",
                                    "data/processed/economics", "data/processed/economics_repeat", "data/processed/economics_audit",
                                    "examples", "config", "metro_data", "tests", "docs")]
    if output == root or input_path.is_relative_to(output) or costs.is_relative_to(output):
        raise ValueError("output cannot contain input files or be the project root")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output cannot overwrite preserved data or project inputs")
    if output.exists():
        if not output.is_dir():
            raise ValueError("output must be a directory")
        entries = {p.name for p in output.iterdir()}
        if entries:
            manifest_path = output / "build_manifest.json"
            try:
                owner = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("output contains unrelated files or an unreadable ownership manifest") from exc
            if (not isinstance(owner, dict) or owner.get("model_version") not in {"simulation-v1", MODEL_VERSION}
                    or owner.get("status") not in {"success", "error"}
                    or not entries <= OUTPUT_NAMES | {"build_manifest.json"}):
                raise ValueError("output is not an owned simulation result directory")
    return input_path, output, costs


def run_example(input_path, output, costs):
    input_path, output, costs = safe_output(input_path, output, costs)
    try:
        data = json.loads(input_path.read_text(encoding="utf-8-sig"))
        results = {name: simulate(data, name) for name in ("baseline", "candidate")}
        control = control_check(results["baseline"])
        if control["status"] != "passed":
            raise ValueError("manual A-B-C control check failed; use simulate() for other plans")
        evaluations = {name: to_evaluation(result) for name, result in results.items()}
        comparison = compare(evaluations["baseline"], evaluations["candidate"], load_costs(costs))
        notes = {"evaluation_mode": "fixture", "source": data["source"],
                 "assumptions": ["prescribed trips and constant directed segment times; no speed model",
                                 "distance credited once on arrival, including closing instant; no prorating",
                                 "arrival, wait completion, departure; ties sorted by train_id, trip_id and leg",
                                 "all listed trains are present throughout the whole window",
                                 "line includes moving, stopping, turning and standby trains at line locations",
                                 "depot-to-line counts on departure; line-to-depot until arrival",
                                 "obligations_closed covers only scheduled movements and station waits in this fixture"],
                 "unknown_quantities": UNKNOWN_UNITS, "absent_checks": ABSENT_CHECKS,
                 "passenger_impact_checked": False, "release_authorized": False,
                 "recommendation": "Влияние на пассажиров не проверялось. Сравнение не даёт рекомендации снять поезд."}
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="simulation-", dir=output.parent) as directory:
            staging = Path(directory)
            for name, result in results.items():
                write_json(staging / f"{name}.simulation.json", result)
                write_text(staging / f"{name}.events.csv", csv_events(result["events"]))
                evaluation = asdict(evaluations[name])
                evaluation["quantities"] = {key: q.as_dict() for key, q in evaluations[name].quantities.items()}
                write_json(staging / f"{name}.evaluation.json", evaluation)
            write_json(staging / "control_check.json", control)
            write_json(staging / "economic_comparison.json", comparison)
            write_json(staging / "assumptions.json", notes)
            lines = ["Учебный симулятор GOAL 3.1", "", "| Величина | Ожидается | Получено |",
                     "|---|---:|---:|"]
            lines += [f"| {r['quantity']} | {r['expected']} | {r['actual']} |" for r in control["checks"]]
            saving = comparison["groups"]["cash"]["savings"]
            saving_text = saving if saving is not None else "неизвестна (null)"
            lines += ["", f"Денежный статус: {comparison['status']}; экономия RUB: {saving_text}.",
                      "Реальные ставки, энергия и оплаченный труд неизвестны.", notes["recommendation"],
                      "Проверки движения ограничены перечисленными в assumptions.json. Выпуск не разрешён."]
            write_text(staging / "summary.md", "\n".join(lines) + "\n")
            inputs = [input_path, costs, Path(__file__).resolve(), Path(__file__).resolve().with_name("economics.py")]
            manifest = {"status": "success", "model_version": MODEL_VERSION,
                        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.iterdir())}}
            write_json(staging / "build_manifest.json", manifest)
            output.mkdir(parents=True, exist_ok=True)
            for p in sorted(staging.iterdir(), key=lambda p: (p.name == "build_manifest.json", p.name)):
                p.replace(output / p.name)
        return manifest
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        write_json(output / "build_manifest.json", {"status": "error", "model_version": MODEL_VERSION, "message": str(exc)})
        write_text(output / "summary.md", f"Расчёт завершился с ошибкой: {exc}\n")
        raise


def check_output(output):
    output = Path(output)
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "success" or manifest.get("model_version") != MODEL_VERSION:
        raise ValueError("simulation manifest does not describe a successful current build")
    fields_only(manifest, ("status", "model_version", "inputs", "outputs"))
    for key in ("inputs", "outputs"):
        if not isinstance(manifest[key], dict) or any(
                not isinstance(path, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                for path, sha in manifest[key].items()):
            raise ValueError(f"invalid simulation manifest {key}")
    input_paths = {Path(p).resolve() for p in manifest["inputs"]}
    code = Path(__file__).resolve()
    if (len(manifest["inputs"]) != 4 or len(input_paths) != 4
            or not {code, code.with_name("economics.py")} <= input_paths
            or any(not Path(p).is_absolute() for p in manifest["inputs"])):
        raise ValueError("simulation manifest must identify both sources and both code inputs")
    if set(manifest["outputs"]) != OUTPUT_NAMES or {p.name for p in output.iterdir()} != OUTPUT_NAMES | {"build_manifest.json"}:
        raise ValueError("simulation manifest has an incomplete or unsupported output set")
    for path, expected in manifest["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed simulation input: {path}")
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed simulation result: {name}")
    return {"status": "passed", "files": len(manifest["outputs"]) + 1, "release_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--input", default="examples/simulation/cycle.json")
    parser.add_argument("--costs", default="config/costs/default.json")
    parser.add_argument("--output", default="data/processed/simulation")
    args = parser.parse_args()
    try:
        result = run_example(args.input, args.output, args.costs) if args.command == "run" else check_output(args.output)
        print(f"simulation {args.command}: {result['status']}; fixture; release_authorized=false")
        return 0
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
