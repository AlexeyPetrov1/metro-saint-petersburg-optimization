"""Independent arithmetic audit of saved GOAL 4.3 journals, without simulation."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from goal43_policy import load_policy, parameter


ROOT = Path(__file__).resolve().parent
RUN_FILES = ("movement_input.json", "motion.json", "passenger_checkpoint.json.gz",
             "groups.json.gz", "profiles.json", "summary.json")


def _read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _assert(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(actual, expected, label):
    if expected is None or isinstance(expected, bool):
        ok = type(actual) is type(expected) and actual == expected
    else:
        try:
            a, b = Decimal(str(actual)), Decimal(str(expected))
            ok = a.is_finite() and b.is_finite() and abs(a - b) <= Decimal("0.00000001")
        except Exception:
            ok = actual == expected
    _assert(ok, f"{label}: recorded {actual!r}, reconstructed {expected!r}")


def _count(value, label):
    _assert(type(value) is int and value >= 0, f"Invalid {label}")
    return value


def _indexed(rows, key, label):
    result = {row[key]: row for row in rows}
    _assert(len(result) == len(rows), f"Duplicate {label}")
    return result


def _line_occupied(state, location, stations, tracks):
    if location in tracks:
        track = tracks[location]
        return any(stations[track[key]]["kind"] == "line" for key in ("from_station", "to_station"))
    _assert(location in stations, "Unknown train location")
    return stations[location]["kind"] == "line"


def _incident_ranges(data):
    rows = []
    for name in ("incidents", "platform_blocks", "station_blocks"):
        rows.extend(data.get(name, []))
    if isinstance(data.get("incident"), dict):
        rows.append(data["incident"])
    result = []
    for row in rows:
        start = row.get("start_second", row.get("detected_second", row.get("second")))
        end = row.get("end_second")
        if end is None and start is not None:
            duration = row.get("duration_seconds")
            if duration is None and row.get("duration_minutes") is not None:
                duration = row["duration_minutes"] * 60
            if duration is not None:
                end = start + duration
        if start is not None and end is not None:
            result.append((start, end))
    return result


def _service_headways(departures, tracks, stations, scope, duration, policy, ranges):
    first, last = scope.get("start_second", 0), scope["end_second"]
    _assert(type(first) is int and type(last) is int and 0 <= first < last <= duration, "Invalid service scope")
    intervals, violations, unknown = [], [], []
    maximum = parameter(policy, "headway_max_s")
    for track_id, track in tracks.items():
        if not all(stations[track[key]]["kind"] == "line" for key in ("from_station", "to_station")):
            continue
        times = sorted({second for second, _ in departures.get(track_id, []) if second <= last})
        if not any(second <= first for second in times):
            unknown.append(track_id)
        for left, right in zip(times, times[1:]):
            if first <= right < last:
                intervals.append(right-left)
                if right-left > maximum:
                    violations.append(dict(kind="maximum_headway", track_id=track_id, previous_second=left,
                        second=right, actual_seconds=right-left, limit_seconds=maximum, excess_seconds=right-left-maximum))
        for instant, boundary in ((first, "start"), (last, "end")):
            previous = next((second for second in reversed(times) if second <= instant), None)
            age = instant-(previous if previous is not None else first)
            if age > maximum:
                violations.append(dict(kind="maximum_headway", track_id=track_id, boundary=boundary,
                    previous_second=previous, second=instant, actual_seconds=age, limit_seconds=maximum,
                    excess_seconds=age-maximum, age_is_lower_bound=previous is None))
        earliest = next((second for second in times if first < second <= last), None)
        if track_id in unknown and earliest is not None and earliest-first > maximum:
            violations.append(dict(kind="maximum_headway", track_id=track_id, boundary="first_departure_lower_bound",
                previous_second=None, second=earliest, actual_seconds=earliest-first, limit_seconds=maximum,
                excess_seconds=earliest-first-maximum, age_is_lower_bound=True))
    for row in violations:
        left = row["previous_second"] if row["previous_second"] is not None else first
        row["cause"] = "overlaps_recorded_incident" if any(a <= row["second"] and b >= left for a, b in ranges) else "not_linked_to_recorded_incident"
    return dict(passed=not violations and not unknown, checked_intervals=len(intervals), violations=violations,
        unknown_tracks=unknown, minimum_seconds=min(intervals, default=None), maximum_seconds=max(intervals, default=None),
        scope=scope)


def _movement(data, motion, policy, service_scope=None, incident_ranges=()):
    start, end = map(datetime.fromisoformat, (data["period_start"], data["period_end"]))
    _assert(start.tzinfo is not None and end.tzinfo is not None, "Movement timezone is missing")
    duration = int((end - start).total_seconds())
    _equal(motion["duration_seconds"], duration, "movement duration")
    if "period_start" in motion:
        _assert(datetime.fromisoformat(motion["period_start"]) == start, "Movement start mismatch")
        _assert(datetime.fromisoformat(motion["period_end"]) == end, "Movement end mismatch")
    trains = _indexed(data["trains"], "train_id", "train_id")
    stations = _indexed(data["stations"], "station_id", "station_id")
    tracks = _indexed(data["tracks"], "track_id", "track_id")
    plans = {trip["trip_id"]: trip for plan in data["plans"] for trip in plan["trips"]}
    events = motion["events"]
    by_event = _indexed(events, "event_id", "motion event_id")
    _assert(all(a["second"] <= b["second"] for a, b in zip(events, events[1:])), "Movement events are not chronological")
    _assert(all(type(row["second"]) is int and 0 <= row["second"] <= duration for row in events), "Movement event outside period")
    runtime = {key: dict(state=row["state"], location=row["location"], since=0) for key, row in trains.items()}
    limit = parameter(policy, "line_train_limit")
    peak = sum(_line_occupied(row["state"], row["location"], stations, tracks) for row in runtime.values())
    count = peak
    _assert(peak <= limit, f"line capacity exceeds {limit}")

    departures = defaultdict(list)
    for event in events:
        if event["event_type"] == "departure":
            departures[event["location"]].append((event["second"], event["train_id"]))
    for row in data.get("previous_departures", []):
        key, second = row["track_id"], row["departure_second"]
        if not any(second == value[0] for value in departures[key]):
            departures[key].append((second, "history"))
    violations = []
    ranges = [*_incident_ranges(data), *incident_ranges]
    for track_id, values in departures.items():
        track = tracks[track_id]
        values.sort()
        for previous, current in zip(values, values[1:]):
            if current[0] < 0:
                continue
            minimum = parameter(policy, "physical_headway_min_s")
            line_station = all(stations[track[key]]["kind"] == "line" for key in ("from_station", "to_station"))
            if line_station:
                local = (start + timedelta(seconds=current[0])).astimezone(ZoneInfo("Europe/Moscow"))
                minute = local.hour * 60 + local.minute
                peak_hours = local.weekday() < 5 and (480 <= minute < 600 or 1020 <= minute < 1140)
                minimum = parameter(policy, "peak_headway_min_s" if peak_hours else "offpeak_headway_min_s")
            gap = current[0] - previous[0]
            _assert(gap >= minimum, f"headway {track_id}: {gap} < {minimum}")
            maximum = parameter(policy, "headway_max_s")
            if service_scope is None and line_station and gap > maximum:
                caused = any(left <= current[0] and right >= previous[0] for left, right in ranges)
                violations.append(dict(kind="maximum_headway", track_id=track_id,
                    previous_second=previous[0], second=current[0], actual_seconds=gap,
                    limit_seconds=maximum, excess_seconds=gap - maximum,
                    cause="incident_blocking" if caused else "not_linked_to_recorded_incident"))

    service = _service_headways(departures, tracks, stations, service_scope, duration, policy, ranges) if service_scope else None
    if service is not None:
        violations.extend(service["violations"])

    pending = {}
    for key, row in trains.items():
        if row["state"] == "moving":
            pending[key] = dict(track_id=row["location"], passenger=row["initial_movement"]["passenger"],
                                trip_id="@initial")
    mileage = {key: Counter() for key in trains}
    seconds = {key: Counter() for key in trains}
    for event in events:
        key = event["train_id"]
        _assert(key in runtime, "Motion references unknown train")
        rt = runtime[key]
        _assert(event["previous_state"] == rt["state"], "Movement previous state mismatch")
        seconds[key][rt["state"]] += event["second"] - rt["since"]
        kind = event["event_type"]
        if kind == "departure":
            _assert(key not in pending and event["location"] in tracks, "departure while train is busy")
            track = tracks[event["location"]]
            _assert(rt["location"] == track["from_station"], "Departure origin mismatch")
            trip = plans[event["trip_id"]]
            pending[key] = dict(track_id=event["location"], passenger=trip["passenger"], trip_id=event["trip_id"])
        elif kind == "arrival":
            _assert(key in pending, "arrival lacks preceding departure")
            movement = pending.pop(key)
            track = tracks[movement["track_id"]]
            _assert(event["location"] == track["to_station"] and event["trip_id"] == movement["trip_id"], "Arrival route mismatch")
            prefix = "passenger" if movement["passenger"] else "nonpassenger"
            km = Decimal(str(track["length_km"]))
            _assert(km.is_finite() and km > 0, "Invalid track length")
            mileage[key][prefix + "_train_km"] += km
            mileage[key][prefix + "_car_km"] += km * trains[key]["cars"]
        previous_on_line = _line_occupied(rt["state"], rt["location"], stations, tracks)
        rt.update(state=event["next_state"], location=event["location"], since=event["second"])
        count += _line_occupied(rt["state"], rt["location"], stations, tracks) - previous_on_line
        peak = max(peak, count)
        _assert(count <= limit, f"line capacity exceeds {limit}")
    work = _indexed(motion["transport_work"], "train_id", "transport_work train_id")
    _assert(set(work) == set(trains), "Movement work fleet mismatch")
    totals = Counter()
    total_seconds = Counter()
    per_train = []
    for key, row in trains.items():
        rt = runtime[key]
        seconds[key][rt["state"]] += duration - rt["since"]
        _assert(sum(seconds[key].values()) == duration, "State time conservation mismatch")
        for prefix in ("passenger", "nonpassenger"):
            for suffix in ("train_km", "car_km"):
                name = prefix + "_" + suffix
                _equal(work[key][name], mileage[key][name], f"mileage {key}/{name}")
                totals[name] += mileage[key][name]
        for suffix in ("train_km", "car_km"):
            name = suffix
            value = mileage[key]["passenger_" + suffix] + mileage[key]["nonpassenger_" + suffix]
            _equal(work[key][name], value, f"mileage {key}/{name}")
            totals[name] += value
        for state in set(seconds[key]) | set(work[key]["state_seconds"]):
            _equal(work[key]["state_seconds"].get(state, 0), seconds[key][state], f"state seconds {key}/{state}")
        total_seconds.update(seconds[key])
        per_train.append(dict(train_id=key, **{name: float(value) for name, value in mileage[key].items()}))
    final = _indexed(motion.get("final_state", []), "train_id", "final state train_id")
    for key, row in final.items():
        _assert(row["state"] == runtime[key]["state"] and row["location"] == runtime[key]["location"], "Final movement stock mismatch")
    if "peak_trains_on_line" in motion.get("checks", {}):
        _equal(motion["checks"]["peak_trains_on_line"], peak, "line peak")
    holding = sum(max(0, min(row["departure_second"], duration) - max(0, row["departure_second"] - row.get("holding_seconds", 0)))
                  for row in data.get("leg_timing", []))
    _assert(holding <= total_seconds["standby"], "Holding exceeds standby time")
    total_seconds["standby"] -= holding
    total_seconds["holding"] = holding
    approach = 0
    for row in data.get("leg_timing", []):
        hold = row.get("approach_holding_seconds", 0)
        if not hold:
            continue
        key = row["train_id"]
        track_id = trains[key]["location"] if row["trip_id"] == "@initial" else plans[row["trip_id"]]["route"][row["leg_index"]]
        at_signal = row["departure_second"] + tracks[track_id]["travel_seconds"]
        approach += max(0, min(at_signal + hold, duration) - max(0, at_signal))
    _assert(approach <= total_seconds["moving"], "Approach holding exceeds moving time")
    if approach:
        total_seconds["moving"] -= approach
        total_seconds["approach_holding"] = approach
    return dict(duration=duration, trains=trains, tracks=tracks, plans=plans, events=by_event,
                resources={**{key: float(value) for key, value in totals.items()}, "state_seconds": dict(total_seconds)},
                peak=peak, per_train=per_train, violations=violations, service_headways=service)


def _exposure(load_events, profiles, final_loads, duration, policy):
    history = defaultdict(list)
    index = {(train, car["car_id"]): i for train, cars in profiles.items() for i, car in enumerate(cars)}
    max_load = 0
    for row in load_events:
        key = (row["train_id"], row["car_id"])
        _assert(key in index, "Unknown car load identity")
        load = _count(row["load"], "car load")
        second = _count(row["second"], "car observation second")
        _assert(second < duration, "Car load observation outside checkpoint")
        _assert(load <= profiles[key[0]][index[key]]["max_capacity"], "Car load exceeds declared capacity")
        _assert(not history[key] or history[key][-1][0] <= second, "Car load history is not chronological")
        history[key].append((second, load))
        max_load = max(max_load, load)
    step = parameter(policy, "decision_step_s")
    windows = [Counter() for _ in range((duration + step - 1) // step)]
    total = Counter()
    for key, car_index in index.items():
        train = key[0]
        values = history[key]
        _equal(final_loads[train][car_index], values[-1][1] if values else 0, "final car load")
        prior, since = 0, 0
        for second, load in values + [(duration, 0)]:
            if second > since:
                flags = dict(passenger_car_seconds=prior > 0,
                             comfort_car_seconds=prior > parameter(policy, "comfort_limit"),
                             exceptional_car_seconds=prior > parameter(policy, "normal_limit"),
                             hard_limit_car_seconds=prior > parameter(policy, "hard_limit"))
                for name, active in flags.items():
                    if active:
                        total[name] += second - since
                left = since
                while left < second:
                    slot = left // step
                    right = min(second, (slot + 1) * step)
                    for name, active in flags.items():
                        if active:
                            windows[slot][name] += right - left
                    left = right
            since, prior = second, load
    def summarize(counts):
        occupied = counts["passenger_car_seconds"]
        share = counts["exceptional_car_seconds"] / occupied if occupied else None
        status = "unknown" if not occupied else "fail" if counts["hard_limit_car_seconds"] or counts["exceptional_car_seconds"] * 100 > occupied else "pass"
        return {**{name: counts[name] for name in ("passenger_car_seconds", "comfort_car_seconds", "exceptional_car_seconds", "hard_limit_car_seconds")},
                "exceptional_time_share": share, "status": status, "hard_limit_ok": None if not occupied else not bool(counts["hard_limit_car_seconds"])}
    summary = dict(summarize(total), max_load=max_load)
    steps = [dict(summarize(row), start=i * step, end=min(duration, (i + 1) * step)) for i, row in enumerate(windows)]
    return summary, steps


def _scoped_resources(data, movement, first, last):
    """Filter the verified journal to a declared resource interval."""
    _assert(0 <= first <= last <= movement["duration"], "Invalid resources scope")
    totals = {prefix + suffix: Decimal(0) for prefix in ("passenger_", "nonpassenger_")
              for suffix in ("train_km", "car_km")}
    seconds = Counter()
    by_train = defaultdict(list)
    pending = {key: row["location"] for key, row in movement["trains"].items() if row["state"] == "moving"}
    overlap = lambda left, right: max(0, min(right, last) - max(left, first))
    for event in movement["events"].values():
        key = event["train_id"]
        by_train[key].append(event)
        if event["event_type"] == "departure":
            pending[key] = event["location"]
        elif event["event_type"] == "arrival":
            track = movement["tracks"][pending.pop(key)]
            if first < event["second"] <= last:
                passenger = (movement["trains"][key]["initial_movement"]["passenger"] if event["trip_id"] == "@initial"
                             else movement["plans"][event["trip_id"]]["passenger"])
                prefix = "passenger_" if passenger else "nonpassenger_"
                distance = Decimal(str(track["length_km"]))
                totals[prefix + "train_km"] += distance
                totals[prefix + "car_km"] += distance * movement["trains"][key]["cars"]
    for key, train in movement["trains"].items():
        state, since = train["state"], 0
        for event in by_train[key] + [dict(second=last, next_state=state)]:
            seconds[state] += overlap(since, event["second"])
            if event["second"] >= last:
                break
            since, state = event["second"], event["next_state"]
    holding = sum(overlap(row["departure_second"] - row.get("holding_seconds", 0), row["departure_second"])
                  for row in data.get("leg_timing", []))
    seconds["standby"] -= holding
    seconds["holding"] = holding
    approach = 0
    for row in data.get("leg_timing", []):
        if row.get("approach_holding_seconds", 0):
            track = movement["trains"][row["train_id"]]["location"] if row["trip_id"] == "@initial" else movement["plans"][row["trip_id"]]["route"][row["leg_index"]]
            at_signal = row["departure_second"] + movement["tracks"][track]["travel_seconds"]
            approach += overlap(at_signal, at_signal + row["approach_holding_seconds"])
    if approach:
        seconds["moving"] -= approach
        seconds["approach_holding"] = approach
    for suffix in ("train_km", "car_km"):
        totals[suffix] = totals["passenger_" + suffix] + totals["nonpassenger_" + suffix]
    return {**{key: float(value) for key, value in totals.items()}, "state_seconds": dict(seconds)}


def _passengers(groups, profiles, state, movement, policy, cohort_scope):
    duration = movement["duration"]
    _equal(state["second"], duration, "checkpoint duration")
    if "sha256" in state:
        raw = json.dumps({k: v for k, v in state.items() if k != "sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        _assert(hashlib.sha256(raw).hexdigest() == state["sha256"], "Checkpoint checksum mismatch")
    by_group = _indexed(groups, "group_id", "group_id")
    _assert(set(profiles) == set(movement["trains"]), "Passenger profile fleet mismatch")
    for train, cars in profiles.items():
        _assert(len(cars) == movement["trains"][train]["cars"], "Profile formation mismatch")
    for row in groups:
        _count(row["count"], "source group count")
        _assert(0 <= row["turnstile_second"] <= row["platform_second"], "Invalid source group arrival")
        _equal(sum(dest["count"] for dest in row["destinations"]), row["count"], "OD counts")
    entered = sum(row["count"] for row in groups if row["turnstile_second"] < duration)
    platform = sum(row["count"] for row in groups if row["platform_second"] < duration)
    _equal(state["entered"], entered, "entered")
    _equal(state["platform_arrived"], platform, "platform arrivals")
    _equal(state["approaching"], entered - platform, "approaching")
    boarded_by_group, histogram = Counter(), Counter()
    full_wait = 0
    for row in state["board_events"]:
        group = by_group[row["group_id"]]
        count = _count(row["count"], "boarding count")
        _assert(group["platform_second"] <= row["second"] < duration, "Boarding before arrival or after checkpoint")
        _equal(row["wait_seconds"], row["second"] - group["platform_second"], "boarding waiting")
        event = movement["events"].get(row["motion_event_id"])
        _assert(event is not None and event["event_type"] == "departure" and event["second"] == row["second"], "Boarding motion identity mismatch")
        track = movement["tracks"][event["location"]]
        _assert(track["from_station"] == group["origin"] and movement["plans"][event["trip_id"]]["passenger"] is True, "Boarding on wrong or nonpassenger train")
        boarded_by_group[row["group_id"]] += count
        histogram[str(row["wait_seconds"])] += count
        full_wait += count * row["wait_seconds"]
    _assert(dict(histogram) == state["wait_histogram"], "waiting histogram mismatch")
    boarded = sum(boarded_by_group.values())
    _equal(state["boarded"], boarded, "boarded count")
    board_by_event = Counter()
    loads_by_event = defaultdict(list)
    for row in state["board_events"]:
        board_by_event[row["motion_event_id"]] += row["count"]
    for row in state["load_events"]:
        loads_by_event[row["motion_event_id"]].append(row)
    prior_loads = {train: dict.fromkeys((car["car_id"] for car in cars), 0) for train, cars in profiles.items()}
    expected_load_events = set()
    for event_id, event in movement["events"].items():
        if event["second"] >= duration:
            continue
        kind = event["event_type"]
        if kind != "arrival" and not (kind == "departure" and movement["plans"][event["trip_id"]]["passenger"]):
            continue
        expected_load_events.add(event_id)
        train = event["train_id"]
        rows = loads_by_event[event_id]
        observed = {row["car_id"]: row["load"] for row in rows}
        _assert(len(rows) == len(profiles[train]) and set(observed) == set(prior_loads[train]), "Incomplete car load observations")
        _assert(all(row["second"] == event["second"] and row["train_id"] == train and row["event_type"] == kind for row in rows), "Car load event identity mismatch")
        previous = prior_loads[train]
        if kind == "departure":
            _assert(all(observed[car] >= previous[car] for car in observed), "Car load decreased during boarding")
            _equal(sum(observed.values()) - sum(previous.values()), board_by_event[event_id], "boarding load identity")
        else:
            _assert(all(observed[car] <= previous[car] for car in observed), "Car load increased during alighting")
        prior_loads[train] = observed
    _assert(set(loads_by_event) == expected_load_events, "Unexpected car load event history")
    queued_by_group, queue_counts = Counter(), Counter()
    residual = 0
    for key, rows in state["queues"].items():
        for row in rows:
            group = by_group[row["group_id"]]
            _assert(row["platform_second"] == group["platform_second"] and row["origin"] == group["origin"], "Queue group identity mismatch")
            count = _count(row["count"], "queue cohort count")
            queued_by_group[row["group_id"]] += count
            queue_counts[key] += count
            residual += count * (duration - group["platform_second"])
    for key in set(queue_counts) | set(state["queue_counts"]):
        _equal(state["queue_counts"].get(key, 0), queue_counts[key], "direction queue stock")
    queued = sum(queue_counts.values())
    _equal(state["queued"], queued, "queued count")
    for key, group in by_group.items():
        expected = group["count"] if group["platform_second"] < duration else 0
        _equal(queued_by_group[key] + boarded_by_group[key], expected, "source group conservation")
    onboard = 0
    for train, cars in profiles.items():
        _assert(len(state["onboard"][train]) == len(cars) and len(state["loads"][train]) == len(cars), "Final formation mismatch")
        for i, rows in enumerate(state["onboard"][train]):
            count = sum(row["count"] for row in rows)
            _equal(state["loads"][train][i], count, "onboard car load")
            onboard += count
    _equal(state["onboard_count"], onboard, "onboard stock")
    completed = boarded - onboard
    _equal(state["completed"], completed, "completed count")
    _equal(entered, completed + entered - platform + queued + onboard, "passenger conservation")
    waiting = full_wait + residual
    _equal(state["waiting_passenger_seconds"], waiting, "waiting passenger-seconds")
    queue = area = since = 0
    timeline = [(group["platform_second"], group["count"]) for group in groups if group["platform_second"] < duration]
    timeline += [(row["second"], -row["count"]) for row in state["board_events"]]
    for second, change in sorted(timeline, key=lambda row: (row[0], -row[1])):
        area += queue * (second - since)
        queue += change
        _assert(queue >= 0, "Negative reconstructed queue")
        since = second
    area += queue * (duration - since)
    _equal(area, waiting, "integrated queue waiting")
    exposure, windows = _exposure(state["load_events"], profiles, state["loads"], duration, policy)
    _equal(state["max_car_load"], exposure["max_load"], "max car load")
    first, last = cohort_scope
    ids = {key for key, row in by_group.items() if first <= row["turnstile_second"] < min(last, duration)}
    arrived = sum(by_group[key]["count"] for key in ids)
    cohort_hist = Counter()
    for row in state["board_events"]:
        if row["group_id"] in ids:
            cohort_hist[row["wait_seconds"]] += row["count"]
    n = sum(cohort_hist.values())
    cohort_wait = sum(wait * count for wait, count in cohort_hist.items())
    cohort_residual = sum(row["count"] * (duration - row["platform_second"]) for rows in state["queues"].values() for row in rows if row["group_id"] in ids)
    rank, cumulative, p95 = (95 * n + 99) // 100, 0, None
    for wait, count in sorted(cohort_hist.items()):
        cumulative += count
        if cumulative >= rank:
            p95 = wait
            break
    complete = arrived == n and duration >= last
    cohort = dict(arrived=arrived, boarded=n, unboarded=arrived - n,
                  waiting_passenger_seconds_lower_bound=cohort_wait + cohort_residual, complete=complete,
                  mean_wait_seconds=cohort_wait / n if complete and n else None,
                  p95_wait_seconds=p95 if complete and n else None,
                  boarded_mean_wait_seconds=cohort_wait / n if n else None, boarded_p95_wait_seconds=p95)
    return dict(entered=entered, boarded=boarded, queued=queued, onboard=onboard, approaching=entered - platform,
                completed=completed, waiting_passenger_seconds=waiting, full_boarded_wait_passenger_seconds=full_wait,
                residual_queue_age_passenger_seconds=residual, max_car_load=exposure["max_load"],
                **exposure, exposure_windows=windows, cohort=cohort)


def _recorded_metrics(recorded, rebuilt, label):
    for key in ("max_car_load", "entered", "boarded", "queued", "completed", "waiting_passenger_seconds"):
        if key in recorded:
            _equal(recorded[key], rebuilt[key], f"{label} {key}")
    if "train_km" in recorded:
        _equal(recorded["train_km"], rebuilt["resources"]["train_km"], f"{label} train mileage")
    resources = recorded.get("resources", {})
    for key in ("passenger_train_km", "nonpassenger_train_km", "passenger_car_km", "nonpassenger_car_km", "train_km", "car_km"):
        if key in resources:
            _equal(resources[key], rebuilt["resources"][key], f"{label} resource mileage {key}")
    for key, value in resources.get("state_seconds", {}).items():
        _equal(value, rebuilt["resources"]["state_seconds"].get(key, 0), f"{label} resource seconds {key}")
    for key, value in recorded.get("cohort", {}).items():
        if key in rebuilt["cohort"]:
            _equal(value, rebuilt["cohort"][key], f"{label} cohort {key}")
    exposure = recorded.get("policy_exposure", recorded.get("exposure", {}))
    summary = exposure.get("global", exposure)
    for key in ("passenger_car_seconds", "comfort_car_seconds", "exceptional_car_seconds", "hard_limit_car_seconds", "exceptional_time_share", "max_load", "hard_limit_ok", "status"):
        if key in summary:
            _equal(summary[key], rebuilt[key], f"{label} exposure {key}")
    if "windows" in exposure:
        _assert(len(exposure["windows"]) == len(rebuilt["exposure_windows"]), f"{label} exposure window count")
        for actual, expected in zip(exposure["windows"], rebuilt["exposure_windows"]):
            for key in ("start", "end", "passenger_car_seconds", "exceptional_car_seconds", "exceptional_time_share", "status"):
                if key in actual:
                    _equal(actual[key], expected[key], f"{label} exposure window {key}")


def _manifest(output):
    candidates = [output / "sourcebaseline_manifest.json", output / "baseline_manifest.json",
                  output.parent / "goal4_3_verification" / "baseline_manifest.json"]
    path = next((value for value in candidates if value.exists()), None)
    if path is None:
        return dict(status="not_provided", checked_files=0)
    manifest = _read(path)
    workspace = Path(manifest.get("workspace_root", ROOT))
    archive = Path(manifest.get("archived_baseline_source", path.parent / "baseline_source"))
    if not archive.is_absolute():
        archive = path.parent / archive
    files = manifest.get("files", manifest.get("immutable_files", {}))
    source_files = manifest.get("source_files", [])
    if isinstance(source_files, dict):
        files = {**files, **source_files}
    rows = []
    for name, expected in files.items():
        source = name in source_files
        actual_path = archive / name if source else workspace / name
        _assert(actual_path.is_file(), f"Missing {'archived' if source else 'immutable'} file {name}")
        actual = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        _assert(actual == expected, f"{'archived' if source else 'immutable'} bytes changed: {name}")
        rows.append(dict(path=name, verified_path=str(actual_path), status="passed", sha256=actual))
    return dict(status="passed", checked_files=len(rows), files=rows, manifest=str(path))


def verify(output):
    """Write verification_report.json and raise ValueError on an integrity failure."""
    output = Path(output).resolve()
    report = dict(schema_version=1, status="passed", run_count=0, runs=[], manifest_checks={})
    policy = load_policy()
    try:
        records = _read(output / "runs.json")
        if isinstance(records, dict):
            records = records["runs"]
        _assert(isinstance(records, list) and records, "runs.json must contain saved runs")
        _indexed(records, "run_id", "run_id")
        for record in records:
            folder = Path(record["folder"])
            if not folder.is_absolute():
                folder = output / folder
            files = {name: _read(folder / name) for name in RUN_FILES}
            summary = files["summary.json"]
            scenario_path = output / "scenarios" / record.get("scenario_id", "") / "summary.json"
            scenario = _read(scenario_path) if scenario_path.exists() else {}
            incident = scenario.get("incident") or {}
            actual_end = scenario.get("injected_actual_end_second", incident.get("expected_end_second"))
            ranges = [(incident["detected_second"], actual_end)] if "detected_second" in incident and actual_end is not None else []
            movement = _movement(files["movement_input.json"], files["motion.json"], policy, summary.get("service_scope"), ranges)
            if movement["service_headways"] is not None:
                for key in ("passed", "checked_intervals", "minimum_seconds", "maximum_seconds"):
                    if key in summary.get("headways", {}):
                        _equal(summary["headways"][key], movement["service_headways"][key], f"service headways {key}")
            scope = summary.get("cohort_scope", summary.get("metric_scope", {}))
            cohort_scope = (scope.get("start_second", 0), scope.get("end_second", movement["duration"]))
            resource_scope = summary.get("resources_scope", {})
            resource_first = resource_scope.get("start_second", 0)
            resource_last = resource_scope.get("end_second", movement["duration"])
            if (resource_first, resource_last) != (0, movement["duration"]):
                movement["resources"] = _scoped_resources(files["movement_input.json"], movement, resource_first, resource_last)
            groups = files["groups.json.gz"]
            if isinstance(groups, dict):
                groups = groups["groups"]
            passengers = _passengers(groups, files["profiles.json"], files["passenger_checkpoint.json.gz"], movement, policy, cohort_scope)
            rebuilt = dict(passengers, resources=movement["resources"], peak_trains_on_line=movement["peak"], per_train=movement["per_train"])
            _recorded_metrics(summary.get("metrics", summary), rebuilt, "summary")
            _recorded_metrics(record.get("metrics", {}), rebuilt, "runs")
            violations = movement["violations"]
            if passengers["hard_limit_car_seconds"]:
                violations.append(dict(kind="hard_car_limit", limit=parameter(policy, "hard_limit"), car_seconds=passengers["hard_limit_car_seconds"]))
            for row in passengers["exposure_windows"]:
                if row["status"] == "fail":
                    violations.append(dict(kind="exceptional_time", start=row["start"], end=row["end"], actual_share=row["exceptional_time_share"], limit_share=0.01))
            report["runs"].append(dict(run_id=record["run_id"], checks={name: "passed" for name in
                ("arrival_mileage", "state_seconds", "passenger_conservation", "waiting_identity", "car_time_exposure", "minimum_headways", "line_capacity")},
                violations=violations, reconstructed_metrics=rebuilt, full_day=record.get("full_day")))
        report["run_count"] = len(report["runs"])
        report["manifest_checks"] = _manifest(output)
    except Exception as exc:
        report.update(status="failed", error=str(exc), run_count=len(report["runs"]))
        output.mkdir(parents=True, exist_ok=True)
        (output / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"GOAL 4.3 integrity verification failed: {exc}") from exc
    (output / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    result = verify(parser.parse_args().output)
    print(json.dumps(dict(status=result["status"], run_count=result["run_count"]), ensure_ascii=False))
