"""GOAL 4.3 orchestration over the existing forecasts, movement and passenger engines."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Condition, Lock, Thread

import pandas as pd

from decision_engine import build_regular_plan
from dwell import compile_operational_timing
from economics import write_json, write_text
from forecasting import hourly_median
from goal43_actions import attach_operational_depots, generate_strategy_candidates
from goal43_demo import write_demo
from goal43_economics import compare_train_economics, maintenance_registry, revenue_sensitivity
from goal43_notifications import update_incident, record_calculation_started, record_recommendation, analytics
from goal43_policy import build_train_registry, load_policy, parameter
from line1 import assign_destinations, minute_groups
from passenger_groups import GroupedPassengerSimulation
from replay_metrics import cohort_waiting, work_between
from rolling import decide_operational, operational_exposure, operational_headways
from simulation import digest, simulate


ROOT = Path(__file__).resolve().parent
VERSION = "goal4.3-replay-v1"


class OrderedNotificationSink:
    """Persist ledger snapshots serially; disk completion is outside local delivery."""
    def __init__(self, callback):
        self.callback = callback
        self.condition = Condition()
        self.pending = None
        self.running = False
        self.generation = self.saved = self.failed = 0
        self.saved_elapsed = None
        self.error = None

    def submit(self, rows, detected_monotonic):
        snapshot = copy.deepcopy(rows)
        with self.condition:
            self.generation += 1
            self.pending = (self.generation, snapshot, detected_monotonic)
            if not self.running:
                self.running = True
                Thread(target=self._run, name="goal43-notification-persistence", daemon=True).start()

    def _run(self):
        while True:
            with self.condition:
                if self.pending is None:
                    self.running = False
                    self.condition.notify_all()
                    return
                generation, rows, detected = self.pending
                self.pending = None
            try:
                self.callback(rows)
                error = None
            except Exception as exception:
                error = str(exception)
            with self.condition:
                if error is None:
                    self.saved = generation
                    self.saved_elapsed = time.monotonic() - detected
                else:
                    self.failed, self.error = generation, error
                self.condition.notify_all()

    def state(self):
        with self.condition:
            status = "not_requested" if not self.generation else "persisted" if self.saved == self.generation else "failed" if self.failed == self.generation else "pending"
            return dict(status=status, persisted_elapsed_seconds=self.saved_elapsed if status == "persisted" else None,
                        error=self.error if status == "failed" else None, latest_generation=self.generation,
                        completion_scope="callback completed; fsync and external delivery are unknown")

    def flush(self, timeout=1):
        with self.condition:
            self.condition.wait_for(lambda: max(self.saved, self.failed) == self.generation, timeout)
        return self.state()


_SINKS, _SINK_LOCK = {}, Lock()


def _ordered_sink(callback):
    if callback is None or isinstance(callback, OrderedNotificationSink):
        return callback
    with _SINK_LOCK:
        if id(callback) not in _SINKS:
            _SINKS[id(callback)] = OrderedNotificationSink(callback)
        return _SINKS[id(callback)]


def live_decision(data, sources, snapshots, profiles, ready, frame, policy, *, scenario_id,
                  minimum_remaining, incidents=(), profile="fixed", seed=7, ledger=(), budget_seconds=480,
                  ledger_sink=None, forecast_transform=None, incident_update=None, forecast_provider=None,
                  candidate_headway_targets=None):
    """Notify first; bound the whole forecast, search and validation pipeline."""
    from queue import Empty, Queue
    began = time.monotonic()
    measured_start = time.perf_counter()
    budget = min(float(budget_seconds), parameter(policy, "notification_budget_s"))
    sink = _ordered_sink(ledger_sink)
    second = next(iter(snapshots.values()))["second"]
    now = datetime.fromisoformat(data["period_start"]) + timedelta(seconds=second)
    estimate = incidents[0].get("expected_end_second") if incidents else None
    incident_state = dict(incident_id=scenario_id,
        severity="stop" if incidents else "low_demand", allowed_actions=["regulate", "withdraw", "reserve"],
        estimated_end_at=(datetime.fromisoformat(data["period_start"]) + timedelta(seconds=estimate)).isoformat() if estimate is not None else None,
        recovery_at=None)
    if incident_update: incident_state.update(incident_update)
    ledger = update_incident(list(ledger), incident_state, now.isoformat())
    ledger = record_calculation_started(ledger, scenario_id, now.isoformat())
    if sink is not None: sink.submit(ledger, began)
    results = Queue(maxsize=1)
    def calculate():
        try:
            ids = [s["station_id"] for s in data.get("stations", []) if s["kind"] == "line"]
            if forecast_provider is None:
                bundle = hourly_median(frame, ids, now.replace(minute=0, second=0, microsecond=0).isoformat())
            else:
                observed = frame[frame.interval_end.le(now) & frame.is_complete.eq(True)].copy()
                if "available_at" in observed:
                    observed = observed[observed.available_at.notna() & observed.available_at.le(now)].copy()
                bundle = forecast_provider(observed, ids, now.isoformat())
            if forecast_transform is not None: bundle = forecast_transform(bundle)
            candidate_options = {} if candidate_headway_targets is None else dict(headway_targets_seconds=candidate_headway_targets)
            candidates, search = generate_strategy_candidates(data, second, policy, ready,
                minimum_remaining=minimum_remaining, incidents=incidents, profile=profile, seed=seed,
                include_recovery=bool(incidents), time_limit_seconds=5, deadline=began+budget-min(20, budget/10),
                **candidate_options)
            remaining = max(0, began + budget-time.monotonic())
            decision = decide_operational(data, sources, snapshots, profiles, ready, bundle, policy,
                incidents=incidents, candidates=candidates, profile=profile, seed=seed,
                include_recovery=bool(incidents), budget_seconds=remaining)
            decision["search"] = search
            if search.get("search_complete") is False:
                decision["search_complete"] = False
                if decision["selection"]["selected_candidate_id"] is not None:
                    decision["selection"]["status"] = "search_incomplete"
            results.put((True, dict(decision=decision, candidates=candidates, bundle=bundle)))
        except Exception as error: results.put((False, error))
    Thread(target=calculate, name="goal43-live-pipeline", daemon=True).start()
    try: ok, result = results.get(timeout=max(0, began+budget-time.monotonic()))
    except Empty: ok, result = True, None
    elapsed = time.perf_counter()-measured_start
    if result is None or elapsed >= budget:
        result = dict(decision=dict(decision_time=now.isoformat(), decision_second=second, candidate_table=[],
            selection=dict(status="evaluation_timeout", selected_candidate_id=None, mitigation_candidate_id=None,
                release_authorized=False), search_complete=False, release_authorized=False), candidates=[], bundle=None)
    elif not ok: raise result
    decision = result["decision"]; decision["elapsed_seconds"] = elapsed
    decision["latency_scope"] = "detection through forecast, candidate generation and detailed validation"
    decision["incident_estimates"] = copy.deepcopy(incidents)
    selection = decision["selection"]
    selected = selection.get("selected_candidate_id") or selection.get("mitigation_candidate_id")
    checked = next((r for r in decision["candidate_table"] if r["candidate_id"] == selected), None)
    recommendation = dict(selection, validated=bool(checked and checked["status"] == "evaluated" and
                                                     checked.get("physical_passed") is not False))
    ledger = record_recommendation(ledger, scenario_id, recommendation,
        (now+timedelta(seconds=elapsed)).isoformat(), elapsed, budget_seconds=budget)
    delivery_elapsed = time.perf_counter()-measured_start
    if delivery_elapsed >= budget and selection.get("status") != "evaluation_timeout":
        decision["selection"] = dict(status="evaluation_timeout", selected_candidate_id=None,
            mitigation_candidate_id=None, release_authorized=False)
        decision["search_complete"] = False
        result.update(candidates=[], bundle=None)
        ledger = record_recommendation(ledger[:-1], scenario_id, dict(decision["selection"], validated=False),
            (now+timedelta(seconds=delivery_elapsed)).isoformat(), delivery_elapsed, budget_seconds=budget)
        delivery_elapsed = time.perf_counter()-measured_start
    decision.update(computation_elapsed_seconds=elapsed, elapsed_seconds=delivery_elapsed,
                    local_delivery_elapsed_seconds=delivery_elapsed)
    ledger[-1].update(elapsed_seconds=delivery_elapsed, within_budget=delivery_elapsed <= budget,
                     delivery_basis="in_memory", persistence_status="unknown_at_delivery")
    result["ledger"] = ledger
    result["notification_delivery"] = dict(basis="in_memory", elapsed_seconds=delivery_elapsed,
        within_budget=delivery_elapsed <= budget, external_delivery_status="not_sent")
    if sink is not None: sink.submit(ledger, began)
    result["notification_persistence"] = sink.state() if sink is not None else dict(
        status="not_requested", persisted_elapsed_seconds=None)
    return result


def build_operational_day(policy, *, count=38, start_at="2026-09-01T06:00:00+03:00",
                          service_hours=18, profile="fixed", seed=7, small_line=False):
    config = json.loads((ROOT / "config/line1/experiment.json").read_text(encoding="utf-8"))
    config["turnaround_seconds"] = parameter(policy, "turnaround_s")
    if small_line:
        ids = ["A", "B"] if small_line is True else list(small_line)
        config.update(station_ids=ids, travel_seconds=30, stop_seconds=20 if len(ids) > 2 else 0, length_m=2000)
        policy = copy.deepcopy(policy)
        policy["demo"]["platform_capacity_by_station"].update({ids[0]: 2, ids[-1]: 2})
    service_seconds = int(service_hours * 3600)
    data, meta = build_regular_plan(config, count, end_second=service_seconds)
    allocation = {"Baltiets": (count * 37 + 67) // 68, "Yubileyny": count - (count * 37 + 67) // 68}
    roster = build_train_registry(policy, initial_on_line=allocation)
    active = [row for row in roster if row["state"] == "on_line"]
    rename = {old["train_id"]: row["train_id"] for old, row in zip(data["trains"], active)}
    types = {row["train_id"]: row["type"] for row in roster}
    for train in data["trains"]:
        train["train_id"] = rename[train["train_id"]]
        train["train_type"] = types[train["train_id"]]
    for trip in data["plans"][0]["trips"]:
        old = trip["train_id"]; trip["train_id"] = rename[old]
        trip["trip_id"] = trip["trip_id"].replace(old + ":", rename[old] + ":", 1)
    origin = datetime.fromisoformat(start_at)
    data["period_start"] = origin.isoformat()
    data["period_end"] = (origin + timedelta(seconds=service_seconds + 7200)).isoformat()
    data, ready = attach_operational_depots(data, roster, policy)
    tracks = {r["track_id"]: r for r in data["tracks"]}
    stations = {r["station_id"]: r for r in data["stations"]}
    plan = data["plans"][0]
    for train in active:
        trips = sorted((r for r in plan["trips"] if r["train_id"] == train["train_id"]), key=lambda r: r["departure_second"])
        if not trips: continue
        last = trips[-1]; second = last["departure_second"]
        for track_id in last["route"]:
            section = tracks[track_id]; station = stations[section["to_station"]]
            second += section["travel_seconds"] + station["stop_seconds"] + station["turnaround_seconds"]
        terminal = tracks[last["route"][-1]]["to_station"]
        exit_track = next(t for t in data["tracks"] if t["from_station"] == terminal and t["track_id"].startswith("EXIT_"))
        plan["trips"].append(dict(trip_id=train["train_id"] + ":DAYEXIT", train_id=train["train_id"],
            departure_second=second, route=[exit_track["track_id"]], passenger=False))
    data, timing = compile_operational_timing(data, plan["plan_id"], profile, seed, policy=policy)
    profiles = {row["train_id"]: [dict(car_id=f"C{i + 1}", car_type=f"demo-{row['type']}",
        max_capacity=parameter(policy, "hard_limit"), normal_threshold=parameter(policy, "comfort_limit"),
        source="GOAL4.3 per-car operating limits; manufacturer distribution not inferred") for i in range(8)] for row in roster}
    meta.update(service_end_second=service_seconds, full_day=service_hours == 18 and origin.hour == 6,
        timing=timing, initial_passenger_status="assumed empty; subsequent observed warmup retains all passengers",
        initial_allocation=allocation, depot_routes_status="assumed", signalling_status="unknown")
    return data, roster, ready, profiles, meta


def mileage_context(data, motion, roster, demand_id, incident_id, seed, *, full_day):
    return dict(plan_id=motion["plan_id"], demand_id=demand_id,
        initial_state_id=digest(dict(trains=data["trains"], previous=data.get("previous_departures", []))),
        incident_id=incident_id, random_world_id=f"keyed-dwell:{seed}", period_start=data["period_start"],
        period_end=data["period_end"], closure_end=data["period_end"],
        coverage_complete=full_day, obligations_closed=motion["obligations_closed"], cash_scope_complete=False,
        trains={r["train_id"]: dict(train_type=r["train_type"], passenger_km=r["passenger_train_km"],
            nonpassenger_km=r["nonpassenger_train_km"], moving_seconds=r["state_seconds"]["moving"],
            standby_seconds=r["state_seconds"]["standby"],
            obligations=[o for o in motion["unfinished_obligations"] if o["train_id"] == r["train_id"]]) for r in motion["transport_work"]})


def fixture_history(station_ids, count_per_minute=2):
    records = []
    for day in ("2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01"):
        for hour in range(6, 24):
            for quarter in range(4):
                at = pd.Timestamp(f"{day}T{hour:02}:{quarter * 15:02}:00+03:00")
                for station in station_ids:
                    records.append(dict(station_id=station, interval_start=at, interval_end=at + pd.Timedelta(minutes=15),
                        entry_count=count_per_minute * 15 if hour < 22 else 0, is_complete=True,
                        source_id="explicit constant low-demand fixture"))
    return pd.DataFrame(records)


def actual_groups(frame, data, service_end):
    ids = [s["station_id"] for s in data["stations"] if s["kind"] == "line"]
    start = pd.Timestamp(data["period_start"]); end = start + pd.Timedelta(seconds=service_end)
    selected = frame[(frame.interval_start >= start) & (frame.interval_start < end) & frame.station_id.isin(ids)]
    expected = len(ids) * int((end - start).total_seconds() / 900)
    if len(selected) != expected or not selected.is_complete.all() or selected.entry_count.isna().any():
        raise ValueError("actual demand coverage is incomplete; no silent zero fill")
    groups = minute_groups(selected, ids, start, end, 120)
    return {s: assign_destinations(groups, ids, s) for s in ("S1", "S2", "S3")}


def save_run(output, run_id, data, runner, state, roster, policy, scenario_id, strategy, od, seed, *, full_day, cohort_start, cohort_end, service_end_second=None):
    folder = Path(output) / "runs" / run_id; folder.mkdir(parents=True, exist_ok=True)
    inferred = service_end_second is None
    if inferred:
        service_end_second = runner.duration-7200 if full_day else cohort_end
    if type(service_end_second) is not int or not 0 < service_end_second <= runner.duration:
        raise ValueError("service boundary must lie inside the movement horizon")
    resources = work_between(data, runner.motion, 0, runner.duration)
    exposure = operational_exposure(state, runner.profiles, policy, 0, runner.duration)
    cohort = cohort_waiting(runner, state, cohort_start, cohort_end)
    summary = dict(run_id=run_id, scenario_id=scenario_id, strategy=strategy, od=od, seed=seed,
        metrics=dict(policy_exposure=exposure, cohort=cohort, resources=resources,
            max_car_load=state["max_car_load"], waiting_passenger_seconds=state["waiting_passenger_seconds"]),
        cohort_scope=dict(start_second=cohort_start, end_second=cohort_end),
        resources_scope=dict(start_second=0, end_second=runner.duration), full_day=full_day,
        service_scope=dict(start_second=0, end_second=service_end_second, return_tail_excluded=True,
            source="generated full-day duration minus explicit 7200 s return tail" if inferred and full_day else "explicit service boundary" if not inferred else "partial cohort boundary"),
        entered=state["entered"], completed=state["completed"], queued=state["queued"],
        headways=operational_headways(data, runner.motion, 0, service_end_second), release_authorized=False)
    write_json(folder / "movement_input.json", data); write_json(folder / "motion.json", runner.motion)
    write_json(folder / "profiles.json", runner.profiles)
    for name, value in (("groups.json.gz", list(runner.by_id.values())), ("passenger_checkpoint.json.gz", state)):
        with gzip.open(folder / name, "wb", compresslevel=6) as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    write_json(folder / "summary.json", summary)
    return dict(run_id=run_id, folder=str(folder.relative_to(output)).replace("\\", "/"), **{k: v for k, v in summary.items() if k != "run_id"})


def _series(runner, state, start, end, bundle, data, *, service_end_second=None):
    if service_end_second is not None: end = min(end, service_end_second)
    stations = {r["station_id"]: r for r in data["stations"]}
    tracks = {r["track_id"]: r for r in data["tracks"]}
    line_tracks = {track_id for track_id, row in tracks.items()
                   if all(stations[row[k]]["kind"] == "line" for k in ("from_station", "to_station"))}
    load = defaultdict(int)
    for row in state["load_events"]:
        if start <= row["second"] < end: load[row["second"] // 60 * 60] = max(load[row["second"] // 60 * 60], row["load"])
    by_track = defaultdict(list)
    for event in runner.motion["events"]:
        if event["event_type"] == "departure" and event["location"] in line_tracks:
            by_track[event["location"]].append(event["second"])
    headways = defaultdict(int)
    for times in by_track.values():
        for a, b in zip(times, times[1:]):
            if start <= b < end: headways[b // 60 * 60] = max(headways[b // 60 * 60], b-a)
    forecast = defaultdict(int)
    origin = datetime.fromisoformat(runner.motion["period_start"])
    for row in bundle["rows"]:
        second = int((datetime.fromisoformat(row["interval_start"]) - origin).total_seconds())
        forecast[second] += row["predicted_count"]
    entries = defaultdict(int)
    for row in runner.by_id.values():
        if start <= row["turnstile_second"] < end: entries[row["turnstile_second"] // 900 * 900] += row["count"]
    queue = defaultdict(int)
    for row in state["queue_samples"]:
        if start <= row["second"] < end: queue[row["second"]] += row["queued"]
    def active(row):
        if row["state"] == "moving":
            return any(stations[tracks[row["location"]][k]]["kind"] == "line" for k in ("from_station", "to_station"))
        return stations[row["location"]]["kind"] == "line"
    positions = {r["train_id"]: dict(state=r["state"], location=r["location"]) for r in data["trains"]}
    total = sum(active(r) for r in positions.values()); fleet = []
    for event in runner.motion["events"]:
        row = positions[event["train_id"]]; before = active(row)
        row.update(state=event["next_state"], location=event["location"])
        total += int(active(row))-int(before)
        if start <= event["second"] < end:
            if not fleet or total != fleet[-1]["count"]: fleet.append(dict(second=event["second"], count=total))
    return dict(load=[dict(second=s, load=n) for s, n in sorted(load.items())],
        queue=[dict(second=s, queued=n) for s, n in sorted(queue.items())],
        headway=[dict(second=s, seconds=n) for s, n in sorted(headways.items())],
        headway_label="Максимальный интервал по направлениям за минуту", fleet=fleet,
        entries=[dict(second=s, actual=n, forecast=forecast.get(s)) for s, n in sorted(entries.items())])


def scenario(output, scenario_id, *, historical, seed=7, n_minutes=None, small_line=False, profile="fixed", decision_steps=2,
             estimate_mode="known", forecast_stress=False):
    began = time.perf_counter(); policy = load_policy()
    peak = n_minutes is not None
    start_at = "2026-09-01T08:15:00+03:00" if peak else "2026-09-01T06:00:00+03:00"
    count = 53 if peak else 3 if small_line else 38
    service_hours = 4 if peak else 18
    data, roster, ready, profiles, meta = build_operational_day(policy, count=count, start_at=start_at,
        service_hours=service_hours, profile=profile, seed=seed, small_line=small_line)
    policy = data["headway_policy"]
    station_ids = [s["station_id"] for s in data["stations"] if s["kind"] == "line"]
    path = ROOT / "data/processed/goal4_3_source_refresh/entries_station_15m.parquet"
    frame = pd.read_parquet(path if path.exists() else ROOT / "data/processed/metro/entries_station_15m.parquet") if historical else fixture_history(station_ids, 4 if small_line else 2)
    groups = actual_groups(frame, data, meta["service_end_second"])
    source = {s: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], groups[s], profiles) for s in groups}
    incident = None; decision_second = 1800 if peak else 21600
    if peak:
        event = next(e for e in source["S1"].motion["events"] if e["second"] >= 1800 and e["event_type"] == "arrival"
                     and e["location"] == "L1_07")
        decision_second = event["second"]
        incoming = next(e for e in reversed(source["S1"].motion["events"][:source["S1"].motion["events"].index(event)])
                        if e["train_id"] == event["train_id"] and e["event_type"] == "departure")
        incident = dict(incident_id=scenario_id, train_id=event["train_id"], station_id=event["location"],
            direction="up" if incoming["location"].endswith("_up") else "down",
            detected_second=decision_second, expected_end_second=decision_second + int(n_minutes * 60))
        if estimate_mode == "revised": incident["expected_end_second"] = decision_second+300
        if estimate_mode == "unknown": incident["expected_end_second"] = None
    snapshots = {s: runner.run(end_second=decision_second)["checkpoint"] for s, runner in source.items()}
    incidents = [incident] if incident else []
    folder = Path(output) / "scenarios" / scenario_id; folder.mkdir(parents=True, exist_ok=True)
    ledger_sink = OrderedNotificationSink(lambda rows: write_json(folder / "notifications.json", rows))
    live = live_decision(data, source, snapshots, profiles, ready, frame, policy, scenario_id=scenario_id,
        minimum_remaining=max(1, count - (1 if small_line else 2)), incidents=incidents, profile=profile, seed=seed,
        ledger_sink=ledger_sink)
    decision, candidates, bundle, ledger = (live[k] for k in ("decision", "candidates", "bundle", "ledger"))
    if not candidates:
        public = dict(scenario_id=scenario_id, title=scenario_id, evaluation_mode="historical_entries_injected_incident_assumed_timetable" if historical else "fixture",
            decisions=[decision], comparisons=[], incident=incident, status="evaluation_timeout", release_authorized=False)
        write_json(folder / "summary.json", public)
        write_json(folder / "decision.json", decision)
        return public, []
    comparisons, records, economics_contexts = [], [], {}
    write_json(folder / "decision.json", decision); write_json(folder / "forecast.json", bundle)
    write_json(folder / "fleet_registry.json", roster)
    write_json(folder / "notification_analytics.json", analytics(ledger)); write_json(folder / "assumptions.json", meta)
    for candidate in candidates:
        if candidate.get("_data") is None: continue
        changed = candidate["_data"]; trial_rows = []
        if incident:
            changed, _ = compile_operational_timing(changed, changed["plans"][0]["plan_id"], profile, seed,
                policy=policy, incidents=[{**incident, "expected_end_second": decision_second+int(n_minutes*60)}], as_of=decision_second)
        for od, original in source.items():
            runner = GroupedPassengerSimulation(changed, changed["plans"][0]["plan_id"], groups[od], profiles)
            branch = runner.branch_checkpoint(snapshots[od], original)
            out = runner.run(start_second=decision_second, checkpoint=branch)
            state = out["checkpoint"]
            run_id = f"{scenario_id}_{candidate['candidate_id'].replace(':', '_').replace('+', '-')}_{od}_{seed}"
            record = save_run(output, run_id, changed, runner, state, roster, policy,
                scenario_id, candidate["strategy"], od, seed, full_day=meta["full_day"],
                cohort_start=decision_second, cohort_end=decision_second + 7200, service_end_second=meta["service_end_second"])
            records.append(record); trial_rows.append(record)
            context = mileage_context(changed, runner.motion, roster, digest(groups[od]),
                scenario_id if incident else "none", seed, full_day=meta["full_day"])
            economics_contexts[candidate["candidate_id"], od] = context
            if od == "S1":
                comparisons.append(dict(strategy=candidate["strategy"], candidate_id=candidate["candidate_id"],
                    status=next((r["status"] for r in decision["candidate_table"] if r["candidate_id"] == candidate["candidate_id"]), candidate["status"]),
                    elapsed_seconds=decision["elapsed_seconds"], metrics=dict(
                        mean_wait_seconds=record["metrics"]["cohort"]["mean_wait_seconds"],
                        p95_wait_seconds=record["metrics"]["cohort"]["p95_wait_seconds"], max_car_load=record["metrics"]["max_car_load"],
                        exceptional_time_share=record["metrics"]["policy_exposure"]["global"]["exceptional_time_share"],
                        train_km=record["metrics"]["resources"]["train_km"], saving_km=None,
                        to1_resource_value_rub=None, lost_revenue_rub=None),
                    series=_series(runner, state, decision_second, decision_second + 7200, bundle, changed),
                    per_train=[], release_authorized=False))
    from goal43_controller import run_controller
    controlled = run_controller(data, source, snapshots, profiles, ready, frame, policy, scenario_id=scenario_id,
        minimum_remaining=max(1, count-(1 if small_line else 2)), groups=groups, incident=incident,
        actual_end_second=decision_second+int(n_minutes*60) if incident else None,
        revision_second=decision_second+180 if estimate_mode == "revised" else None,
        revised_end_second=decision_second+int(n_minutes*60) if estimate_mode == "revised" else None,
        steps=decision_steps, profile=profile, seed=seed, first_live=live, forecast_stress=forecast_stress,
        ledger_sink=ledger_sink)
    write_json(folder / "decisions.json", controlled["decisions"])
    write_json(folder / "forecasts.json", controlled["bundles"])
    write_json(folder / "controller_trace.json", controlled["trace"])
    persistence = ledger_sink.flush(2)
    write_json(folder / "notification_analytics.json", analytics(controlled["ledger"]))
    for od, runner in controlled["sources"].items():
        state = controlled["states"][od]
        record = save_run(output, f"{scenario_id}_ROLLING_SELECTED_{od}_{seed}", controlled["data"], runner,
            state, roster, policy, scenario_id, "ROLLING_SELECTED", od, seed, full_day=meta["full_day"],
            cohort_start=decision_second, cohort_end=decision_second+7200, service_end_second=meta["service_end_second"])
        records.append(record)
        economics_contexts["ROLLING_SELECTED", od] = mileage_context(controlled["data"], runner.motion, roster,
            digest(groups[od]), scenario_id if incident else "none", seed, full_day=meta["full_day"])
        if od == "S1":
            comparisons.append(dict(strategy="ROLLING_SELECTED", candidate_id="ROLLING_SELECTED",
                status=controlled["decisions"][-1]["selection"]["status"], elapsed_seconds=max(d["elapsed_seconds"] for d in controlled["decisions"]),
                metrics=dict(mean_wait_seconds=record["metrics"]["cohort"]["mean_wait_seconds"],
                    p95_wait_seconds=record["metrics"]["cohort"]["p95_wait_seconds"], max_car_load=record["metrics"]["max_car_load"],
                    exceptional_time_share=record["metrics"]["policy_exposure"]["global"]["exceptional_time_share"],
                    train_km=record["metrics"]["resources"]["train_km"], saving_km=None, to1_resource_value_rub=None, lost_revenue_rub=None),
                series=_series(runner, state, decision_second, decision_second+7200, bundle, controlled["data"]), per_train=[], release_authorized=False))
    fleet = {row["train_id"]: row["type"] for row in roster}
    economic_rows = []
    for comparison in comparisons:
        economy = compare_train_economics(economics_contexts["KEEP", "S1"],
            economics_contexts[comparison["candidate_id"], "S1"], fleet, policy=policy)
        comparison["metrics"].update(saving_km=economy["totals"]["saving_km"],
            to1_resource_value_rub=economy["totals"]["to1_resource_value_rub"])
        comparison["per_train"] = [dict(train_id=row["train_id"], train_type=row["train_type"],
            baseline_km=row["baseline_km"], candidate_km=row["candidate_km"], saving_km=row["saving_km"])
            for row in economy["per_train"].values()]
        economic_rows.append(dict(candidate_id=comparison["candidate_id"], **economy))
    write_json(folder / "economics.json", economic_rows)
    public = dict(scenario_id=scenario_id, title=scenario_id,
        evaluation_mode="historical_entries_injected_incident_assumed_timetable" if historical and incident else "historical_entries_assumed_timetable" if historical else "fixture",
        incident=incident, injected_actual_end_second=decision_second+int(n_minutes*60) if incident else None,
        estimate_mode=estimate_mode, decisions=controlled["decisions"], comparisons=comparisons, full_day=meta["full_day"],
        selection=decision["selection"], elapsed_seconds=time.perf_counter() - began,
        notification_persistence=persistence,
        positive_historical_economy_confirmed=False, release_authorized=False)
    write_json(folder / "summary.json", public)
    print(json.dumps(dict(stage="goal43_scenario", scenario=scenario_id, seconds=round(public["elapsed_seconds"], 2),
        selection=decision["selection"], runs=len(records)), ensure_ascii=False), flush=True)
    return public, records


def collect_runs(output):
    output = Path(output)
    records = []
    for path in sorted((output / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        records.append(dict(summary, folder=path.parent.relative_to(output).as_posix()))
    for path in sorted((output / "campaign").glob("*/manifest.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")).get("records", []))
    return records


def refresh_service_summaries(output, *, return_tail_seconds=7200):
    """Correct service scopes using identical saved movement, without simulation."""
    output = Path(output)
    records = collect_runs(output)
    by_run, closed = {}, {}
    for record in records:
        folder = output / record["folder"]
        data_bytes = (folder / "movement_input.json").read_bytes()
        motion_bytes = (folder / "motion.json").read_bytes()
        data, motion = json.loads(data_bytes), json.loads(motion_bytes)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        assumptions_path = output / "scenarios" / record.get("scenario_id", "") / "assumptions.json"
        assumptions = json.loads(assumptions_path.read_text(encoding="utf-8")) if assumptions_path.exists() else {}
        service_end = summary.get("service_scope", {}).get("end_second", assumptions.get("service_end_second"))
        if service_end is None:
            service_end = motion["duration_seconds"]-return_tail_seconds
        if type(service_end) is not int or not 0 < service_end <= motion["duration_seconds"]:
            raise ValueError(f"No valid generated service boundary for {record['run_id']}")
        scope = dict(start_second=0, end_second=service_end, return_tail_excluded=True,
            source="saved generated service boundary; journal-only refresh")
        headways = operational_headways(data, motion, 0, service_end)
        evidence = dict(movement_input_sha256=hashlib.sha256(data_bytes).hexdigest(),
                        motion_sha256=hashlib.sha256(motion_bytes).hexdigest(), simulation_rerun=False)
        summary.update(headways=headways, service_scope=scope, service_scope_evidence=evidence)
        write_json(folder / "summary.json", summary)
        record.update(headways=headways, service_scope=scope, service_scope_evidence=evidence)
        by_run[record["run_id"]] = record
        closed[record["run_id"]] = motion.get("obligations_closed") is True
    for path in sorted((output / "campaign").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["records"] = [by_run[row["run_id"]] for row in manifest.get("records", [])]
        for trial in manifest.get("trials", []):
            record = by_run.get(trial.get("run_id"))
            if record:
                trial["service_admissible"] = (record["metrics"]["policy_exposure"]["admissible"] is True
                    and record["headways"]["passed"] and closed[record["run_id"]]
                    and record["entered"] == record["completed"] and record.get("queued", 0) == 0)
        if manifest.get("trials"):
            from goal43_campaign import _aggregate
            manifest["aggregate"] = _aggregate(manifest["trials"])
        manifest["service_scope_refreshed"] = True
        write_json(path, manifest)
    write_json(output / "runs.json", records)
    return dict(runs_updated=len(records), simulation_rerun=False,
                evidence="per-run service_scope_evidence hashes; movement and passenger journals unchanged")


def run(output, *, scenario_ids=None, campaign=False, seed=7, finalize_report=True):
    output = Path(output).resolve()
    if output == ROOT or ROOT not in output.parents or "goal4_2" in str(output):
        raise ValueError("GOAL4.3 output must be an isolated descendant of the workspace")
    output.mkdir(parents=True, exist_ok=True)
    cases = {"fixture_offpeak": dict(historical=False), "historical_offpeak": dict(historical=True),
             **{f"station_stop_{n}": dict(historical=True, n_minutes=n, decision_steps=4) for n in (3, 8, 15, 30)},
             "station_stop_5_to_15": dict(historical=True, n_minutes=15, estimate_mode="revised", decision_steps=4),
             "station_stop_unknown": dict(historical=True, n_minutes=30, estimate_mode="unknown", decision_steps=4),
             "forecast_stress": dict(historical=False, small_line=["L1_01", "L1_07", "L1_19"], forecast_stress=True, decision_steps=3, profile="mild")}
    selected = list(cases) if scenario_ids is None else list(scenario_ids)
    unknown = set(selected) - cases.keys()
    if unknown: raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    scenarios = []; records = collect_runs(output); began = time.perf_counter()
    for name in selected:
        summary, rows = scenario(output, name, seed=seed, **cases[name]); scenarios.append(summary); records.extend(rows)
        records = list({r["run_id"]: r for r in records}.values())
        write_json(output / "runs.json", records)
    if campaign:
        from goal43_campaign import run_campaign
        stages = [output / "campaign" / stage / "manifest.json" for stage in ("preliminary", "validation")]
        expected = [120, 120]
        complete = all(path.exists() and len(json.loads(path.read_text(encoding="utf-8")).get("records", [])) == count
                       for path, count in zip(stages, expected))
        if not complete:
            preliminary = run_campaign(output, stage="preliminary", workers=3)
            run_campaign(output, stage="validation", selected_plan=preliminary["selected_plan"], workers=3)
    records = collect_runs(output); write_json(output / "runs.json", records)
    write_json(output / "maintenance_registry.json", maintenance_registry())
    write_json(output / "performance.json", dict(total_seconds=time.perf_counter() - began,
        decision_seconds=[d["elapsed_seconds"] for row in scenarios for d in row["decisions"]], budget_seconds=480))
    if finalize_report:
        from goal43_report import finalize
        finalize(output)
    return dict(status="replay_generated", output=str(output), scenarios=len(scenarios), runs=len(records),
                campaign_requested=campaign, release_authorized=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/goal4_3")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--campaign", action="store_true", help="Run the 10 preliminary and 20 frozen validation worlds")
    parser.add_argument("--no-report", action="store_true", help="Defer reporting when running separate scenario batches")
    args = parser.parse_args(argv)
    if args.command == "check":
        from goal43_verify import verify
        result = verify(args.output)
    else: result = run(args.output, scenario_ids=args.scenarios, campaign=args.campaign, finalize_report=not args.no_report)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
