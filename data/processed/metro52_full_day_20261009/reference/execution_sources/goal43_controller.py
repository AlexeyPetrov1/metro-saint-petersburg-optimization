"""Repeated decisions with observed history and a separate offline actual world."""

import copy
import json
from collections import defaultdict
from datetime import datetime, timedelta

from dwell import compile_operational_timing, timing_key
from passenger_groups import GroupedPassengerSimulation


def control_times(start, steps, *, revision_second=None):
    if steps < 1: raise ValueError("at least one decision is required")
    times = {start+900*i for i in range(steps)}
    if revision_second is not None and revision_second >= start:
        times.add(revision_second)
    return sorted(times)


def observed_planning_data(planning, actual, as_of):
    """Copy only completed events and current movement progress from the actual world."""
    out = copy.deepcopy(planning)
    tracks = {r["track_id"]: r for r in out["tracks"]}
    trips = {r["trip_id"]: r for p in out["plans"] for r in p["trips"]}
    trains = {r["train_id"]: r for r in out.get("trains", [])}
    seen = {timing_key(r["train_id"], r["trip_id"], r["leg_index"]): r for r in actual["leg_timing"]}
    future = defaultdict(list)
    for i, row in enumerate(out["leg_timing"]):
        observed = seen[timing_key(row["train_id"], row["trip_id"], row["leg_index"])]
        track_id = trains[row["train_id"]]["location"] if row["trip_id"] == "@initial" else trips[row["trip_id"]]["route"][row["leg_index"]]
        travel = tracks[track_id]["travel_seconds"]
        departure = observed["departure_second"]
        arrival = departure+travel+observed.get("approach_holding_seconds", 0)
        if arrival < as_of:
            changed = copy.deepcopy(observed)
            if arrival+observed["service_seconds"] >= as_of:
                changed["service_seconds"] = max(row["service_seconds"], as_of-arrival)
            out["leg_timing"][i] = changed
        elif departure < as_of:
            row.update(departure_second=departure,
                approach_holding_seconds=max(0, as_of-departure-travel))
        else:
            future[track_id].append((row["departure_second"], timing_key(row["train_id"], row["trip_id"], row["leg_index"]), row))
            row["departure_second"] = max(as_of, row["departure_second"])
            row["approach_holding_seconds"] = 0
    for rows in future.values():
        previous = as_of-1
        for _, _, row in sorted(rows):
            row["departure_second"] = max(row["departure_second"], previous+1)
            previous = row["departure_second"]
    return out


def run_controller(data, sources, snapshots, profiles, ready, frame, policy, *, scenario_id,
                   minimum_remaining, groups, incident=None, actual_end_second=None,
                   revision_second=None, revised_end_second=None, steps=4, profile="fixed", seed=7,
                   first_live=None, ledger_sink=None, forecast_stress=False, forecast_provider=None,
                   candidate_headway_targets=None):
    from goal43_replay import live_decision
    start = next(iter(snapshots.values()))["second"]
    planning = data; actual = data; world = sources; states = snapshots
    decisions = []; ledger = []; trace = []; bundles = []; previous = start
    estimates = [copy.deepcopy(incident)] if incident else []
    times = control_times(start, steps, revision_second=revision_second)
    for index, second in enumerate(times):
        if index:
            states = {od: runner.run(start_second=previous, end_second=second,
                checkpoint=states[od])["checkpoint"] for od, runner in world.items()}
            planning = observed_planning_data(planning, actual, second)
        if revision_second is not None and second >= revision_second and estimates:
            estimates[0]["expected_end_second"] = revised_end_second
        recovered = incident is not None and second >= actual_end_second
        if recovered: estimates = []
        def transform(bundle):
            factor = 1.5 if index % 2 else .7
            changed = copy.deepcopy(bundle)
            changed["model_version"] += ":assumed-forecast-stress"
            changed["source_versions"]["stress"] = dict(factor=factor, status="assumed forecast error stress")
            for row in changed["rows"]: row["predicted_count"] = round(row["predicted_count"]*factor)
            return changed
        live = first_live if index == 0 and first_live is not None else live_decision(
            planning, world, states, profiles, ready, frame, policy, scenario_id=scenario_id,
            minimum_remaining=minimum_remaining, incidents=estimates, profile=profile, seed=seed, ledger=ledger,
            ledger_sink=ledger_sink, forecast_transform=transform if forecast_stress else None,
            forecast_provider=forecast_provider, candidate_headway_targets=candidate_headway_targets,
            incident_update=dict(severity="resolved", recovery_at=(datetime.fromisoformat(data["period_start"])+timedelta(seconds=second)).isoformat())
                if recovered else None)
        decision = live["decision"]; ledger = live["ledger"]
        decision["trigger"] = "incident_revision" if second == revision_second else "15_minute_control"
        decision["recovery_observed"] = recovered
        decision["forecast_stress"] = forecast_stress and index > 0
        decisions.append(decision)
        print(json.dumps(dict(stage="goal43_control", scenario=scenario_id, second=second,
            elapsed_seconds=decision.get("elapsed_seconds"), selection=decision["selection"]), ensure_ascii=False), flush=True)
        if live.get("bundle") is not None: bundles.append(live["bundle"])
        selection = decision["selection"]
        selected = selection.get("selected_candidate_id") or selection.get("mitigation_candidate_id")
        candidate = next((r for r in live["candidates"] if r["candidate_id"] == selected), None)
        if candidate is None:
            candidate = next((r for r in live["candidates"] if r["candidate_id"] == "KEEP"), None)
        planning = candidate["_data"] if candidate and candidate.get("_data") is not None else planning
        actual_incidents = [{**incident, "expected_end_second": actual_end_second}] if incident else []
        actual, timing = compile_operational_timing(planning, planning["plans"][0]["plan_id"],
            profile, seed, policy=policy, incidents=actual_incidents, as_of=second)
        new_world = {od: GroupedPassengerSimulation(actual, actual["plans"][0]["plan_id"], rows, profiles)
                     for od, rows in groups.items()}
        states = {od: runner.branch_checkpoint(states[od], world[od]) for od, runner in new_world.items()}
        world = new_world; previous = second
        trace.append(dict(second=second, estimated_incidents=copy.deepcopy(estimates),
            selected_candidate_id=selected, actual_world_is_offline=True, recovery_observed=recovered,
            pending_passengers={od: dict(queued=s["queued"], approaching=s["approaching"], onboard=s["onboard_count"])
                                for od, s in states.items()}, timing=timing))
    final = {od: runner.run(start_second=previous, checkpoint=states[od])["checkpoint"] for od, runner in world.items()}
    return dict(decisions=decisions, ledger=ledger, trace=trace, bundles=bundles,
                data=actual, sources=world, states=final)
