"""Operational wrappers over the existing action and movement engines."""

import copy
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import median
from time import monotonic

from actions import expand_action
from dwell import compile_operational_timing, timing_key
from goal43_optimizer import solve_departures, solve_withdrawals
from goal43_policy import parameter, minimum_headway, validate_train_registry
from simulation import simulate, instant, integer
from withdrawal_stabling import attach_stabling


def attach_operational_depots(data, fleet_registry, policy):
    """Keep the entire assigned fleet in every branch, with explicit demo depot routes."""
    validate_train_registry(fleet_registry, policy)
    out = copy.deepcopy(data)
    rows = {row["train_id"]: row for row in fleet_registry}
    existing = {row["train_id"]: row for row in out["trains"]}
    if existing.keys() - rows.keys():
        raise ValueError("simulation contains trains outside assigned fleet")
    if any(row["state"] == "on_line" and row["train_id"] not in existing for row in fleet_registry):
        raise ValueError("on-line fleet ID is missing from initial movement")
    up = sorted((row for row in out["tracks"] if row["track_id"].endswith("_up")), key=lambda row: row["track_id"])
    if not up:
        raise ValueError("line requires ordered up/down routes")
    terminals = {"Avtovo": up[0]["from_station"], "Northern": up[-1]["to_station"]}
    stations = {row["station_id"]: row for row in out["stations"]}
    ready = {}
    for depot, terminal in terminals.items():
        if stations[terminal]["turnaround_seconds"] != parameter(policy, "turnaround_s"):
            raise ValueError("terminal turnaround does not match operational policy")
        depot_id = f"DEPOT_{depot.upper()}"
        if depot_id in stations:
            raise ValueError("operational depots already attached")
        out["stations"].append(dict(station_id=depot_id, kind="depot", stop_seconds=0, turnaround_seconds=0))
        for prefix, start, finish in (("ENTRY", depot_id, terminal), ("EXIT", terminal, depot_id)):
            out["tracks"].append(dict(track_id=f"{prefix}_{depot.upper()}", from_station=start,
                                      to_station=finish, length_km="1.35" if depot == "Avtovo" else "1", travel_seconds=120))
    start = instant(out["period_start"])
    for train_id, row in rows.items():
        depot = row["location"].get("depot", "Avtovo")
        if depot not in terminals:
            raise ValueError("unsupported depot route")
        if train_id not in existing:
            out["trains"].append(dict(train_id=train_id, train_type=row["type"],
                                      cars=parameter(policy, "cars_per_train"), state="standby",
                                      location=f"DEPOT_{depot.upper()}", ready_after_seconds=0))
        ready_at = row.get("ready_at")
        ready_second = int((instant(ready_at)-start).total_seconds()) if ready_at is not None else None
        release = parameter(policy, "depot_release_s")[depot]
        preparation = release-120-stations[terminals[depot]]["turnaround_seconds"]
        if preparation < 0:
            raise ValueError("inclusive release is shorter than the assumed physical entry route")
        ready[train_id] = dict(ready=row["state"] == "reserve" and ready_at is not None,
                              ready_at_second=ready_second, preparation_seconds=preparation,
                              inclusive_release_seconds=release, depot=depot,
                              readiness_status=row["readiness_status"],
                              source="GOAL 4.3 explicit demo depot route and roster",
                              infrastructure_status="user access distance; assumed terminal connection and 120 seconds",
                              location_status="explicit demo depot" if row["location"].get("kind") == "depot"
                                              else "assumed placeholder; actual depot unknown",
                              maintenance_status="unknown", crew_status="unknown")
    out["max_trains_on_line"] = parameter(policy, "line_train_limit")
    out["source"] += "; GOAL 4.3 depot connections and unavailable-train locations are assumed"
    return attach_stabling(out), ready


def _prune_timing(data):
    assigned = {trip["trip_id"] for plan in data["plans"] for trip in plan["trips"]}
    data["leg_timing"] = [row for row in data.get("leg_timing", [])
                          if row["trip_id"] == "@initial" or row["trip_id"] in assigned]


def _declare_final_return(data, train_id):
    plan = data["plans"][0]
    assigned = [row for row in plan["trips"] if row["train_id"] == train_id]
    if not assigned:
        return
    last = max(assigned, key=lambda row: (row["departure_second"], row["trip_id"]))
    if not last["passenger"]:
        return
    tracks = {row["track_id"]: row for row in data["tracks"]}
    stations = {row["station_id"]: row for row in data["stations"]}
    ready = last["departure_second"]
    for track_id in last["route"]:
        track = tracks[track_id]; station = stations[track["to_station"]]
        ready += track["travel_seconds"]+station["stop_seconds"]+station["turnaround_seconds"]
    terminal = tracks[last["route"][-1]]["to_station"]
    exit_track = next((row for row in tracks.values() if row["from_station"] == terminal
                       and row["track_id"].startswith("EXIT_")), None)
    if exit_track is None:
        raise ValueError("final service has no declared depot return route")
    plan["trips"].append(dict(trip_id=f"{train_id}:ACTION_FINAL_EXIT", train_id=train_id,
                             departure_second=ready, route=[exit_track["track_id"]], passenger=False))


def expand_operational_action(data, kind, train_id, as_of, registry, *, policy, incidents=(),
                              profile="fixed", seed=0):
    integer(as_of, "as_of")
    if kind == "ADD":
        info = registry.get(train_id, {})
        if info.get("ready") is not True or info.get("ready_at_second") is None or info["ready_at_second"] > as_of:
            raise ValueError("reserve is not explicitly ready at the decision time")
    plan_id = data["plans"][0]["plan_id"]
    baseline, _ = compile_operational_timing(data, plan_id, profile, seed, policy=policy,
                                            incidents=incidents, as_of=as_of)
    changed, action = expand_action(baseline, kind, train_id, as_of, registry)
    if kind in {"ADD", "DEFER_WITHDRAWAL"}:
        _declare_final_return(changed, train_id)
    _prune_timing(changed)
    changed, timing = compile_operational_timing(changed, plan_id, profile, seed, policy=policy,
                                                incidents=incidents, as_of=as_of)
    motion = simulate(changed, plan_id)
    trips = {row["trip_id"]: row for row in changed["plans"][0]["trips"]}
    tracks = {row["track_id"]: row for row in changed["tracks"]}
    effects = {}
    if kind in {"ADD", "DEFER_WITHDRAWAL"}:
        for event in motion["events"]:
            if (event["train_id"] == train_id and event["event_type"] == "departure"
                    and event["second"] >= as_of and trips.get(event["trip_id"], {}).get("passenger")):
                effects.setdefault(tracks[event["location"]]["from_station"], event["second"])
    if kind == "WITHDRAW":
        arrivals = [event for event in motion["events"] if event["train_id"] == train_id
                    and event["event_type"] == "arrival" and event["second"] >= as_of
                    and event["location"].startswith(("DEPOT_", "SIDING_"))]
        actual_effect = min((event["second"] for event in arrivals), default=None)
        final = next(row for row in motion["final_state"] if row["train_id"] == train_id)
        closed = final["state"] == "standby" and final["location"].startswith(("DEPOT_", "SIDING_"))
        closed &= not any(row["train_id"] == train_id for row in motion["unfinished_obligations"])
        action["withdrawal_return_closed"] = closed
        action["withdrawal_closure_status"] = "closed" if closed else "partial return obligation"
    else:
        actual_effect = min(effects.values()) if effects else as_of if kind == "KEEP" else None
    action.update(actual_effect_second=actual_effect, actual_effect_by_station=effects,
                  movement_checked=True, passengers_checked=False, timing=timing,
                  support_scope="existing movement engine; demo depot routes")
    if kind == "ADD":
        action["inclusive_release_seconds"] = registry[train_id]["inclusive_release_seconds"]
        action["additional_delay_seconds"] = None if actual_effect is None else actual_effect-action["planned_effect_second"]
    return changed, action


def _headway_windows(start, lower, upper, policy):
    boundaries = {lower, upper+1}
    day = (start+timedelta(seconds=lower)).date()
    last = (start+timedelta(seconds=upper)).date()
    while day <= last:
        for window in parameter(policy, "peak_windows"):
            if day.weekday() not in window["weekdays"]:
                continue
            for clock in (window["start"], window["end"]):
                at = datetime.fromisoformat(f"{day.isoformat()}T{clock}").replace(tzinfo=start.tzinfo)
                boundary = int((at-start).total_seconds())
                if lower < boundary <= upper:
                    boundaries.add(boundary)
        day += timedelta(days=1)
    ordered = sorted(boundaries)
    return [(a, b, minimum_headway(start+timedelta(seconds=a), policy)) for a, b in zip(ordered, ordered[1:])]


def regulate_plan(data, as_of, policy, incidents=(), profile="fixed", seed=0, *,
                   target_headway_seconds=None, time_limit_seconds=10.0, deadline=None):
    """Optimize future terminal departures, then recompile and check actual movement."""
    started = monotonic()
    end = deadline if deadline is not None else float("inf")
    def timeout(value):
        return value, dict(status="evaluation_timeout", solver_status="NOT_STARTED",
                           search_complete=False, movement_checked=False, passengers_checked=False,
                           release_authorized=False, elapsed_seconds=monotonic()-started,
                           holding_location_status="assumed terminal control only")
    integer(as_of, "as_of")
    if monotonic() >= end:
        return timeout(copy.deepcopy(data))
    plan_id = data["plans"][0]["plan_id"]
    baseline, _ = compile_operational_timing(data, plan_id, profile, seed, policy=policy,
                                            incidents=incidents, as_of=as_of)
    if monotonic() >= end:
        return timeout(baseline)
    stations = {row["station_id"]: row for row in baseline["stations"]}
    tracks = {row["track_id"]: row for row in baseline["tracks"]}
    trips = {row["trip_id"]: row for row in baseline["plans"][0]["trips"]}
    rows, keys, grouped, fixed = [], {}, defaultdict(list), {}
    for timing in baseline["leg_timing"]:
        trip = trips.get(timing["trip_id"])
        if trip is None or not trip["passenger"] or timing["leg_index"] != 0:
            continue
        track = tracks[trip["route"][0]]
        if not stations[track["from_station"]]["turnaround_seconds"]:
            continue
        event = f"{timing['train_id']}|{timing['trip_id']}|0"
        row = dict(event_id=event, train_id=timing["train_id"], station_id=track["from_station"],
                   direction="up" if track["track_id"].endswith("_up") else "down",
                   scheduled_seconds=timing["departure_second"], earliest_seconds=timing["departure_second"])
        grouped[(row["station_id"], row["direction"])].append(row)
        keys[event] = timing_key(timing["train_id"], timing["trip_id"], 0)
    for platform, entries in grouped.items():
        future = [row for row in entries if row["scheduled_seconds"] >= as_of]
        past = [row for row in entries if row["scheduled_seconds"] < as_of]
        if past:
            anchor = max(past, key=lambda row: (row["scheduled_seconds"], row["event_id"]))
            rows.append(anchor); fixed[anchor["event_id"]] = anchor["scheduled_seconds"]
        else:
            track = next(row for row in tracks.values() if row["from_station"] == platform[0]
                         and row["track_id"].endswith("_"+platform[1]))
            history = next((row["departure_second"] for row in baseline.get("previous_departures", [])
                            if row["track_id"] == track["track_id"]), None)
            if history is not None:
                event = f"@history|{track['track_id']}"
                rows.append(dict(event_id=event, train_id="@history", station_id=platform[0], direction=platform[1],
                                 scheduled_seconds=history, earliest_seconds=history))
                fixed[event] = history
        rows.extend(future)
    gaps = [b["scheduled_seconds"]-a["scheduled_seconds"] for entries in grouped.values()
            for a, b in zip(sorted(entries, key=lambda row: row["scheduled_seconds"]),
                            sorted(entries, key=lambda row: row["scheduled_seconds"])[1:])
            if b["scheduled_seconds"] >= as_of]
    target = target_headway_seconds if target_headway_seconds is not None else round(median(gaps)) if gaps else 150
    target = max(parameter(policy, "physical_headway_min_s"), target)
    by_train = defaultdict(list)
    for row in rows:
        if row["event_id"] not in fixed:
            by_train[row["train_id"]].append(row)
    links = []
    for entries in by_train.values():
        entries.sort(key=lambda row: row["scheduled_seconds"])
        links.extend((a["event_id"], b["event_id"], b["scheduled_seconds"]-a["scheduled_seconds"])
                     for a, b in zip(entries, entries[1:]))
    duration = int((instant(baseline["period_end"])-instant(baseline["period_start"])).total_seconds())
    windows = _headway_windows(instant(baseline["period_start"]),
                               min((row["scheduled_seconds"] for row in rows), default=0), duration+480, policy)
    solution = solve_departures(rows, min_headway_seconds=parameter(policy, "physical_headway_min_s"),
                                max_headway_seconds=parameter(policy, "headway_max_s"), max_hold_seconds=480,
                                fixed_departures=fixed, links=links, target_headway_seconds=target,
                                headway_windows=windows, time_limit_seconds=time_limit_seconds, deadline=deadline)
    audit = dict(status=solution.status, solver_status=solution.solver_status, search_complete=solution.search_complete,
                 aggregate_objective=solution.objective, target_headway_seconds=target,
                 solver_time_limit_seconds=time_limit_seconds, solver_elapsed_seconds=solution.elapsed_seconds,
                 search_holding_bound_s=480, search_holding_bound_status="assumed finite search bound",
                 passengers_checked=False, holding_location_status="assumed terminal control only",
                 holding_capacity_status="requires demo platform validation; no remote command authorized",
                 future_terminal_events=len(rows)-len(fixed), movement_checked=False,
                 release_authorized=False, elapsed_seconds=monotonic()-started)
    if not solution.times:
        return baseline, audit
    commands = {keys[event]: second for event, second in solution.times.items()
                if event in keys and event not in fixed}
    changed, timing = compile_operational_timing(baseline, plan_id, profile, seed, policy=policy,
                                                incidents=incidents, as_of=as_of, departure_not_before=commands)
    simulate(changed, plan_id)
    if monotonic() >= end:
        audit.update(status="evaluation_timeout", elapsed_seconds=monotonic()-started)
        return baseline, audit
    audit.update(status="movement_checked" if solution.search_complete else "search_incomplete",
                 movement_checked=True, timing=timing, elapsed_seconds=monotonic()-started,
                 commanded_departures=[dict(train_id=key[0], trip_id=key[1], leg_index=key[2],
                                             not_before_second=second) for key, second in sorted(commands.items())])
    return changed, audit


def _withdrawal_proxy(data, as_of):
    motion = simulate(data, data["plans"][0]["plan_id"])
    stations = {row["station_id"]: row for row in data["stations"]}
    tracks = {row["track_id"]: row for row in data["tracks"]}
    state = {row["train_id"]: (row["state"], row["location"]) for row in data["trains"]}
    for event in motion["events"]:
        if event["second"] >= as_of:
            break
        state[event["train_id"]] = event["next_state"], event["location"]
    on_line = set()
    for train, (kind, location) in state.items():
        if kind == "moving":
            track = tracks[location]
            active = any(stations[track[key]]["kind"] == "line" for key in ("from_station", "to_station"))
        else:
            active = stations[location]["kind"] == "line"
        if active:
            on_line.add(train)
    opportunities = {}
    for event in motion["events"]:
        if (event["train_id"] in on_line and event["second"] >= as_of and event["event_type"] == "arrival"
                and stations[event["location"]]["kind"] == "line"
                and stations[event["location"]]["turnaround_seconds"]):
            opportunities.setdefault(event["train_id"], event)
    candidates = []
    for train, arrival in opportunities.items():
        exit_track = next((row for row in tracks.values() if row["from_station"] == arrival["location"]
                           and stations[row["to_station"]]["kind"] == "depot"), None)
        if exit_track is None:
            continue
        ordered = sorted((row for row in data["plans"][0]["trips"] if row["train_id"] == train),
                         key=lambda row: (row["departure_second"], row["trip_id"]))
        current = next((i for i, row in enumerate(ordered) if row["trip_id"] == arrival["trip_id"]), -1)
        future_km = sum((Decimal(str(tracks[track]["length_km"])) for row in ordered[current+1:]
                         if row["passenger"] for track in row["route"]), Decimal(0))
        saving = int((future_km-Decimal(str(exit_track["length_km"]))) * 1000)
        candidates.append(dict(train_id=train, saving_units=max(0, saving)))
    return sorted(candidates, key=lambda row: row["train_id"]), len(on_line)


def generate_strategy_candidates(data, as_of, policy, registry, *, minimum_remaining, max_withdrawals=2,
                                 top_k=1, incidents=(), profile="fixed", seed=0,
                                 time_limit_seconds=10.0, deadline=None, include_recovery=True,
                                 recovery_end_second=None, headway_targets_seconds=()):
    """Build bounded strategy finalists; callers must evaluate their passengers from one snapshot."""
    integer(top_k, "top_k", 1); integer(max_withdrawals, "max_withdrawals")
    targets = []
    for value in headway_targets_seconds:
        target = integer(value, "headway target", 1)
        if not parameter(policy, "physical_headway_min_s") <= target <= parameter(policy, "headway_max_s"):
            raise ValueError("headway target is outside physical policy bounds")
        targets.append(target)
    targets = sorted(set(targets))
    if deadline is not None and monotonic() >= deadline:
        rows = [dict(candidate_id=kind, strategy=kind, status="evaluation_timeout",
                     action=dict(kind=kind), reason="budget expired before current-state movement checks")
                for kind in ("KEEP", "REGULATE_ONLY", "WITHDRAW_ONLY", "COMBINED")]
        return rows, dict(minimum_remaining=minimum_remaining, max_withdrawals=max_withdrawals,
                          top_k=top_k, proxy_status="assumed aggregate service floor", withdrawal_search=[],
                          passengers_checked=False, release_authorized=False)
    plan_id = data["plans"][0]["plan_id"]
    baseline, timing = compile_operational_timing(data, plan_id, profile, seed, policy=policy,
                                                 incidents=incidents, as_of=as_of)
    rows = [dict(candidate_id="KEEP", strategy="KEEP", status="movement_checked", _data=baseline,
                 action=dict(kind="KEEP", decision_second=as_of, movement_checked=True,
                             passengers_checked=False, release_authorized=False), timing=timing)]

    def add_regulated(source, strategy, candidate_id, action, target=None):
        try:
            options = {} if target is None else dict(target_headway_seconds=target)
            changed, audit = regulate_plan(source, as_of, policy, incidents, profile, seed,
                                           time_limit_seconds=time_limit_seconds, deadline=deadline, **options)
            rows.append(dict(candidate_id=candidate_id, strategy=strategy, status=audit["status"],
                             _data=changed, action=action, timing=audit))
        except ValueError as error:
            rows.append(dict(candidate_id=candidate_id, strategy=strategy, status="rejected",
                             action=action, reason=str(error)))

    add_regulated(baseline, "REGULATE_ONLY", "REGULATE_ONLY", dict(kind="REGULATE_ONLY", decision_second=as_of))
    for target in targets:
        add_regulated(baseline, "REGULATE_ONLY", f"REGULATE_ONLY:h{target}",
            dict(kind="REGULATE_ONLY", decision_second=as_of, target_headway_seconds=target), target)
    proxy, initial_count = _withdrawal_proxy(baseline, as_of)
    excluded, subsets, search_rows = [], [], []
    for _ in range(top_k):
        solution = solve_withdrawals(proxy, initial_count=initial_count, minimum_remaining=minimum_remaining,
                                    max_withdrawals=max_withdrawals, line_capacity=parameter(policy, "line_train_limit"),
                                    time_limit_seconds=time_limit_seconds, deadline=deadline, excluded_subsets=excluded)
        search_rows.append(dict(status=solution.status, search_complete=solution.search_complete,
                                elapsed_seconds=solution.elapsed_seconds, selected_train_ids=list(solution.selected_train_ids),
                                proxy_saving_m=solution.saving_units))
        if not solution.selected_train_ids:
            break
        subsets.append(solution.selected_train_ids); excluded.append(solution.selected_train_ids)
    if not subsets:
        for strategy in ("WITHDRAW_ONLY", "COMBINED"):
            rows.append(dict(candidate_id=strategy, strategy=strategy, status="no_feasible_candidate",
                             action=dict(kind=strategy), reason="no positive bounded withdrawal subset found"))
    for subset in subsets:
        suffix = "+".join(subset)
        action = dict(kind="WITHDRAW_ONLY", decision_second=as_of, withdrawn_train_ids=list(subset), actions=[])
        try:
            changed = baseline
            for train in subset:
                changed, single = expand_operational_action(changed, "WITHDRAW", train, as_of, registry,
                                                            policy=policy, incidents=incidents, profile=profile, seed=seed)
                action["actions"].append(single)
            rows.append(dict(candidate_id=f"WITHDRAW_ONLY:{suffix}", strategy="WITHDRAW_ONLY",
                             status="movement_checked", _data=changed, action=action))
            combined = copy.deepcopy(action); combined["kind"] = "COMBINED"
            add_regulated(changed, "COMBINED", f"COMBINED:{suffix}", combined)
            for target in targets:
                controlled = copy.deepcopy(combined)
                controlled["target_headway_seconds"] = target
                add_regulated(changed, "COMBINED", f"COMBINED:{suffix}:h{target}", controlled, target)
        except ValueError as error:
            for strategy in ("WITHDRAW_ONLY", "COMBINED"):
                rows.append(dict(candidate_id=f"{strategy}:{suffix}", strategy=strategy, status="rejected",
                                 action=action, reason=str(error)))
    if include_recovery:
        duration = int((instant(baseline["period_end"])-instant(baseline["period_start"])).total_seconds())
        recovery_end = min(duration, as_of+7200 if recovery_end_second is None else
                           integer(recovery_end_second, "recovery_end_second"))
        departure_times = {timing_key(row["train_id"], row["trip_id"], row["leg_index"]): row["departure_second"]
                           for row in baseline["leg_timing"]}
        recovery = [("ADD", train) for train, info in registry.items() if info.get("ready")]
        recovery += [("DEFER_WITHDRAWAL", trip["train_id"]) for trip in baseline["plans"][0]["trips"]
                     if not trip["passenger"] and trip["route"][0].startswith("EXIT_")
                     and as_of <= departure_times.get(timing_key(trip["train_id"], trip["trip_id"], 0), trip["departure_second"]) < recovery_end]
        for kind, train in sorted(set(recovery)):
            try:
                changed, action = expand_operational_action(baseline, kind, train, as_of, registry,
                                                            policy=policy, incidents=incidents, profile=profile, seed=seed)
                rows.append(dict(candidate_id=f"{kind}:{train}", strategy=kind, status="movement_checked",
                                 _data=changed, action=action))
            except ValueError as error:
                rows.append(dict(candidate_id=f"{kind}:{train}", strategy=kind, status="rejected",
                                 action=dict(kind=kind, train_id=train), reason=str(error)))
    audit = dict(minimum_remaining=minimum_remaining, max_withdrawals=max_withdrawals, top_k=top_k,
                 proxy_status="assumed aggregate service floor", proxy_unit="m of future scheduled passenger work less depot return",
                 withdrawal_search=search_rows, passengers_checked=False, release_authorized=False)
    if targets:
        audit["headway_targets_seconds"] = targets
    return rows, audit
