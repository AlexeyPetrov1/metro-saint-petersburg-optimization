"""Forecast-only choices from saved snapshots; actual future entries stay in replay."""

import copy
from datetime import datetime, timedelta
from statistics import median

from actions import evaluate_candidates, select_candidate
from forecasting import forecast_groups, validate_bundle
from line1 import assign_destinations


def preserve_past_groups(source,future,start):
    return [copy.deepcopy(g) for g in source.by_id.values() if g["turnstile_second"]<start]+[
        copy.deepcopy(g) for g in future if g["turnstile_second"]>=start]


def committed_overload(sources,snapshots,profiles,start):
    for scenario,source in sources.items():
        source._validate_checkpoint(snapshots[scenario],start)
        if source.profiles!=profiles: raise ValueError('decision changes the saved car profiles')
    for scenario,source in sources.items():
        position={t['train_id']:t['state'] for t in source.motion['initial_state']}
        for event in source.motion['events']:
            if event['second']>=start: break
            position[event['train_id']]=event['next_state']
        for train,loads in snapshots[scenario]['loads'].items():
            if position[train]!='moving': continue
            arrival=next(e['second'] for e in source.motion['events']
                if e['train_id']==train and e['event_type']=='arrival' and e['second']>=start)
            if arrival==start: continue
            for car,load in zip(source.profiles[train],loads):
                if load>car['normal_threshold']:
                    return dict(scenario=scenario,train_id=train,car_id=car['car_id'],load=load,
                        normal_threshold=car['normal_threshold'],start_second=start,until_second=arrival,
                        reason='load persists on an already committed movement until arrival')
    return None


def decide_snapshot(data,sources,snapshots,profiles,registry,bundle,candidates,profile,seeds,*,early_rejection=False):
    start=next(iter(snapshots.values()))["second"]
    if any(s["second"]!=start for s in snapshots.values()): raise ValueError("snapshots must share a decision instant")
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("distinct fixed seeds are required")
    origin=datetime.fromisoformat(data["period_start"]); now=origin+timedelta(seconds=start)
    station_ids=[s["station_id"] for s in data["stations"] if s["kind"]=="line"]
    horizon=validate_bundle(bundle,station_ids,now.isoformat())["available_horizon_seconds"]
    horizon=min(7200,horizon)
    planning=copy.deepcopy(data); planning["period_end"]=(now+timedelta(seconds=horizon)).isoformat()
    groups=forecast_groups(bundle,station_ids,data["period_start"])
    groups=[g for g in groups if start<=g["turnstile_second"]<start+horizon]
    forecasts={s:assign_destinations(groups,station_ids,s) for s in sources}
    if early_rejection:
        witness=committed_overload(sources,snapshots,profiles,start)
        if witness:
            table=[dict(candidate_id=f'{kind}:{train or "plan"}',action=dict(kind=kind,train_id=train),
                status='rejected',reasons=[witness['reason']],normal_load_passed=False,
                tested_seeds=[],requested_seeds=len(seeds),trials=0,early_rejection=True,
                cash_delta_RUB=None) for kind,train in candidates]
            return dict(decision_time=now.isoformat(),decision_second=start,forecast_version=bundle['model_version'],
                forecast_issued_at=bundle['issued_at'],available_horizon_seconds=horizon,target=bundle['target'],
                decision_inputs='validated observed snapshot and issued forecast; future truth excluded',
                candidate_table=table,selection=select_candidate(table),trials=[],seed_set=list(seeds),
                committed_violation=witness,infrastructure_status='unknown',release_authorized=False)
    records=[]; active=list(candidates)
    for seed in seeds:
        for scenario,source in sources.items():
            demand=preserve_past_groups(source,forecasts[scenario],start)
            rows=evaluate_candidates(planning,demand,profiles,registry,snapshots[scenario],source,active,profile,seed)
            for row in rows:
                public={k:v for k,v in row.items() if not k.startswith("_")}
                public.update(seed=seed,scenario=scenario)
                records.append(public)
        if early_rejection:
            active=[(kind,train) for kind,train in active if all(r["status"]=="evaluated" and r["physical_passed"]
                and r["normal_load_passed"] for r in records if r["candidate_id"]==f"{kind}:{train or 'plan'}")]
            if not active: break
    table=[]
    for kind,train in candidates:
        name=f"{kind}:{train or 'plan'}"; trials=[r for r in records if r["candidate_id"]==name]
        evaluated=[r for r in trials if r["status"]=="evaluated"]
        if len(evaluated)!=len(trials):
            table.append(dict(candidate_id=name,action=dict(kind=kind,train_id=train),status="rejected",
                reasons=sorted({r["reason"] for r in trials if r["status"]=="rejected"}),cash_delta_RUB=None))
            continue
        peaks=[r["metrics"]["max_car_load"] for r in evaluated]
        waits=[r["metrics"]["waiting_passenger_seconds"] for r in evaluated]
        table.append(dict(candidate_id=name,action=evaluated[0]["action"],status="evaluated",
            physical_passed=all(r["physical_passed"] for r in evaluated),
            normal_load_passed=all(r["normal_load_passed"] for r in evaluated),
            metrics=dict(max_car_load=max(peaks),waiting_passenger_seconds=max(waits)),
            resources=dict(train_km=str(max(float(r["resources"]["train_km"]) for r in evaluated))),
            forecast_entries=sum(g["count"] for g in groups),cash_delta_RUB=None,
            tested_seeds=sorted({r["seed"] for r in trials}),requested_seeds=len(seeds),
            early_rejection=len({r["seed"] for r in trials})<len(seeds),
            trials=len(trials),violation_fraction=sum(not r["normal_load_passed"] for r in evaluated)/len(trials),
            max_car_load=max(peaks),median_waiting_passenger_seconds=median(waits),
            waiting_min=min(waits),waiting_max=max(waits),worst_checked_load=max(peaks)))
    table.sort(key=lambda r:r["candidate_id"])
    return dict(decision_time=now.isoformat(),decision_second=start,forecast_version=bundle["model_version"],
        forecast_issued_at=bundle["issued_at"],available_horizon_seconds=horizon,
        target=bundle["target"],decision_inputs="observed past snapshot and issued forecast; future truth excluded",
        candidate_table=table,selection=select_candidate(table),trials=records,seed_set=list(seeds),
        infrastructure_status="unknown",release_authorized=False)


def operational_exposure(state, profiles, policy, start, end):
    """Use the car-load ledger; positive load includes platform and approach waits."""
    from collections import defaultdict
    from goal43_policy import occupancy_exposure

    by_car = defaultdict(list)
    for row in state["load_events"]:
        by_car[row["train_id"], row["car_id"]].append(row)
    indexes = {(train, car["car_id"]): index for train, cars in profiles.items() for index, car in enumerate(cars)}
    intervals = []
    for (train, car), rows in by_car.items():
        previous_load = 0; since = 0
        for row in rows + [dict(second=end, load=0)]:
            stop = min(row["second"], end)
            left = max(start, since)
            if stop > left:
                intervals.append(dict(train_id=train, car_index=indexes[train, car],
                                      start_s=left, end_s=stop, load=previous_load))
            if row["second"] >= end: break
            since = row["second"]; previous_load = row["load"]
    return occupancy_exposure(intervals, policy, window_start_s=start, window_end_s=end)


def operational_headways(data, motion, start, end):
    from collections import defaultdict
    from goal43_policy import minimum_headway, parameter

    origin = datetime.fromisoformat(data["period_start"])
    stations = {r["station_id"]: r for r in data["stations"]}
    tracks = {r["track_id"]: r for r in data["tracks"]}
    policy = data["headway_policy"]
    times = defaultdict(list)
    for row in data.get("previous_departures", []): times[row["track_id"]].append(row["departure_second"])
    for event in motion["events"]:
        if event["event_type"] == "departure": times[event["location"]].append(event["second"])
    gaps, boundaries, violations, unknown = [], [], [], []
    maximum = parameter(policy, "headway_max_s")
    for track, section in tracks.items():
        if any(stations[section[key]]["kind"] != "line" for key in ("from_station", "to_station")): continue
        seconds = sorted({second for second in times[track] if second <= end})
        prior = [second for second in seconds if second <= start]
        if not prior: unknown.append(track)
        for instant, kind in ((start, "boundary_start"), (end, "boundary_end")):
            seen = [second for second in seconds if second <= instant]
            age = instant-seen[-1] if seen else instant-start
            row = dict(track_id=track, second=instant, seconds=age, kind=kind,
                       age_is_lower_bound=not seen, maximum_seconds=maximum)
            boundaries.append(row)
            if age > maximum: violations.append(row)
        for left, right in zip(seconds, seconds[1:]):
            if not start <= right < end: continue
            gap = right - left
            minimum = minimum_headway(origin + timedelta(seconds=right), policy)
            row = dict(track_id=track, second=right, seconds=gap, minimum_seconds=minimum, kind="completed_interval")
            gaps.append(row)
            if gap < minimum or gap > maximum: violations.append(row)
        first = next((second for second in seconds if start < second <= end), None)
        if not prior and first is not None and first-start > maximum:
            violations.append(dict(track_id=track, second=first, seconds=first-start,
                kind="first_departure_lower_bound", age_is_lower_bound=True, maximum_seconds=maximum))
    return dict(checked_intervals=len(gaps), violations=violations, passed=not violations and not unknown,
                coverage_status="unknown" if unknown else "complete", unknown_tracks=unknown,
                boundary_ages=boundaries,
                minimum_seconds=min((r["seconds"] for r in gaps), default=None),
                maximum_seconds=max((r["seconds"] for r in gaps), default=None), intervals=gaps)


def committed_schedule_km(data):
    from decimal import Decimal
    tracks = {r["track_id"]: Decimal(str(r["length_km"])) for r in data["tracks"]}
    return str(sum((tracks[track] for plan in data["plans"] for trip in plan["trips"]
                    for track in trip["route"]), Decimal(0)))


def select_operational_candidate(rows, *, timed_out=False, search_incomplete=False):
    from decimal import Decimal
    feasible = [r for r in rows if r.get("status") == "evaluated" and r.get("passengers_checked")
                and r.get("physical_passed") and r.get("policy_passed") and r.get("service_passed")]
    if not feasible:
        checked = [r for r in rows if r.get("status") == "evaluated" and r.get("physical_passed")]
        mitigation = min(checked, key=lambda r: (r["metrics"]["max_car_load"],
            r["metrics"]["waiting_passenger_seconds"], r["candidate_id"]), default=None)
        return dict(status="evaluation_timeout" if timed_out else "no_feasible_candidate",
                    selected_candidate_id=None,
                    mitigation_candidate_id=mitigation["candidate_id"] if mitigation else None,
                    release_authorized=False, economic_optimality=False)
    chosen = min(feasible, key=lambda r: (Decimal(r["committed_schedule_km"]),
        r["metrics"]["waiting_passenger_seconds"], r["metrics"]["comfort_car_seconds"],
        r["candidate_id"] != "KEEP", r["candidate_id"]))
    return dict(status="search_incomplete" if timed_out or search_incomplete else "conditional_resource_policy",
                selected_candidate_id=chosen["candidate_id"], mitigation_candidate_id=None,
                policy_version="closed-schedule-km-comfort-soft-v1", release_authorized=False,
                economic_optimality=False)


def decide_operational(data, sources, snapshots, profiles, registry, bundle, policy, *,
                       incidents=(), candidates=None, profile="fixed", seed=7,
                       budget_seconds=480, include_recovery=True, minimum_remaining=None,
                       monitoring_context=None, passenger_service_check=None, forecast_horizon_seconds=7200,
                       service_end_second=None):
    """Bound the caller's wait; discard late work and cancel its passenger loop."""
    from math import isfinite
    from queue import Empty, Queue
    from threading import Event, Thread
    from time import monotonic
    from goal43_policy import parameter

    began = monotonic()
    if type(forecast_horizon_seconds) is not int or forecast_horizon_seconds <= 0:
        raise ValueError('positive integer forecast horizon required')
    if service_end_second is not None and (type(service_end_second) is not int
            or not next(iter(snapshots.values()))['second'] < service_end_second <= min(s.duration for s in sources.values())):
        raise ValueError('service boundary must follow decision and lie within observation horizon')
    if isinstance(budget_seconds, bool) or not isfinite(float(budget_seconds)) or budget_seconds < 0:
        raise ValueError("budget_seconds must be finite and nonnegative")
    deadline = began + min(float(budget_seconds), parameter(policy, "notification_budget_s"))
    cancelled, result = Event(), Queue(maxsize=1)

    def timeout():
        cancelled.set()
        start = next(iter(snapshots.values()))["second"]
        now = datetime.fromisoformat(data["period_start"]) + timedelta(seconds=start)
        return dict(decision_time=now.isoformat(), decision_second=start, target=bundle["target"],
            forecast_version=bundle["model_version"], forecast_issued_at=bundle["issued_at"],
            available_horizon_seconds=None, candidate_table=[],
            decision_inputs="observed snapshot and available forecast; future actual demand excluded",
            incident_estimates=copy.deepcopy(incidents), selection=select_operational_candidate([], timed_out=True),
            elapsed_seconds=monotonic()-began, search_complete=False, release_authorized=False,
            physical_effect_deadline_is_notification_deadline=False)

    if monotonic() >= deadline: return timeout()

    def calculate():
        try:
            answer = _decide_operational(data, sources, snapshots, profiles, registry, bundle, policy,
                incidents=incidents, candidates=candidates, profile=profile, seed=seed,
                budget_seconds=budget_seconds, include_recovery=include_recovery,
                minimum_remaining=minimum_remaining, began=began, deadline=deadline, cancelled=cancelled,
                monitoring_context=monitoring_context, passenger_service_check=passenger_service_check,
                forecast_horizon_seconds=forecast_horizon_seconds, service_end_second=service_end_second)
            check_operational_deadline(deadline, cancelled)
            result.put((True, answer))
        except Exception as error:
            if not cancelled.is_set(): result.put((False, error))

    Thread(target=calculate, name="goal43-dispatch", daemon=True).start()
    try: succeeded, answer = result.get(timeout=max(0, deadline-monotonic()))
    except Empty: return timeout()
    if monotonic() >= deadline or isinstance(answer, TimeoutError): return timeout()
    if not succeeded: raise answer
    answer["elapsed_seconds"] = monotonic()-began
    return answer


def check_operational_deadline(deadline, cancelled):
    from time import monotonic
    if cancelled.is_set() or monotonic() >= deadline:
        raise TimeoutError("operational calculation deadline or cancellation")


def initial_service_obligations(runner, before, after, start):
    queued = sum(row["count"] for queue in before["queues"].values() for row in queue)
    approaching = sum(row["count"] for row in runner.by_id.values()
                      if row["turnstile_second"] < start <= row["platform_second"])
    remaining_queue = sum(row["count"] for queue in after["queues"].values() for row in queue
                          if row["turnstile_second"] < start)
    remaining_approach = sum(row["count"] for row in runner.by_id.values()
                            if row["turnstile_second"] < start and row["platform_second"] >= after["second"])
    pending = queued+approaching; unboarded = remaining_queue+remaining_approach
    return dict(queued_at_start=queued, approaching_at_start=approaching, pending_at_start=pending,
                boarded=pending-unboarded, unboarded=unboarded, complete=unboarded == 0,
                initial_onboard=before["onboard_count"],
                scope="boarding obligations for passengers already queued or approaching at decision time")


def _decide_operational(data, sources, snapshots, profiles, registry, bundle, policy, *,
                        incidents, candidates, profile, seed, budget_seconds, include_recovery,
                        minimum_remaining, began, deadline, cancelled, monitoring_context=None,
                        passenger_service_check=None, forecast_horizon_seconds=7200, service_end_second=None):
    from time import monotonic
    from goal43_actions import generate_strategy_candidates
    from goal43_policy import parameter
    from passenger_groups import GroupedPassengerSimulation
    from replay_metrics import cohort_waiting, work_between

    check_operational_deadline(deadline, cancelled)
    start = next(iter(snapshots.values()))["second"]
    if any(state["second"] != start for state in snapshots.values()): raise ValueError("decision snapshots disagree")
    now = datetime.fromisoformat(data["period_start"]) + timedelta(seconds=start)
    station_ids = [s["station_id"] for s in data["stations"] if s["kind"] == "line"]
    horizon = min(forecast_horizon_seconds, validate_bundle(bundle, station_ids, now.isoformat())["available_horizon_seconds"],
                  next(iter(sources.values())).duration - start)
    check_operational_deadline(deadline, cancelled)
    if horizon <= 0: raise ValueError("no available decision horizon")
    metadata = dict(decision_time=now.isoformat(), decision_second=start, target=bundle["target"],
        forecast_version=bundle["model_version"], forecast_issued_at=bundle["issued_at"],
        available_horizon_seconds=horizon,
        decision_inputs="observed snapshot and available forecast; future actual demand excluded",
        incident_estimates=copy.deepcopy(incidents), release_authorized=False)
    if monitoring_context is not None:
        from metro_monitoring import control_guard, guard_candidates
        metadata['anomaly_guard'] = control_guard(monitoring_context['observations'],
            monitoring_context.get('event_registry'), now, station_ids,
            movement=monitoring_context.get('movement'),
            knowledge_mode=monitoring_context.get('knowledge_mode', 'strict'),
            require_movement_coverage=monitoring_context.get('require_movement_coverage', False),
            horizon_seconds=horizon)
        check_operational_deadline(deadline, cancelled)
        if monitoring_context.get('enable_anomaly_monitor',False):
            from metro_anomaly_monitor import monitor_anomalies
            metadata['anomaly_monitor'] = monitor_anomalies(monitoring_context['observations'],
                monitoring_context.get('event_registry'),now,vestibules=monitoring_context.get('vestibules'),
                movement=monitoring_context.get('movement'),
                knowledge_mode=monitoring_context.get('knowledge_mode','strict'))
            check_operational_deadline(deadline,cancelled)
    if monotonic() >= deadline:
        return dict(**metadata, candidate_table=[], selection=select_operational_candidate([], timed_out=True),
                    elapsed_seconds=monotonic() - began, search_complete=False)
    approach_delay = 120
    allowance = approach_delay + parameter(policy, "headway_max_s")
    mature_end = max(start, start+horizon-allowance)
    forecast = forecast_groups(bundle, station_ids, data["period_start"], delay=approach_delay)
    forecast = [g for g in forecast if start <= g["turnstile_second"] < start + horizon]
    demand = {s: preserve_past_groups(source, assign_destinations(forecast, station_ids, s), start)
              for s, source in sources.items()}
    check_operational_deadline(deadline, cancelled)
    if candidates is None:
        current = sum(r["location"] in {s["station_id"] for s in data["stations"] if s["kind"] == "line"}
                      or r["state"] == "moving" for r in data["trains"])
        candidates, search = generate_strategy_candidates(data, start, policy, registry,
            minimum_remaining=max(1, current - 2) if minimum_remaining is None else minimum_remaining,
            incidents=incidents, profile=profile, seed=seed, deadline=deadline-min(20, budget_seconds/10),
            include_recovery=include_recovery)
    else: search = dict(scope="supplied movement plans")
    check_operational_deadline(deadline, cancelled)
    search_incomplete = search.get("search_complete") is False or any(
        r.get("search_complete") is False for r in search.get("withdrawal_search", []))
    search_incomplete |= any(r.get("status") == "search_incomplete" or
                             r.get("timing", {}).get("search_complete") is False for r in candidates)
    table = []; timed_out = any(r.get("status") == "evaluation_timeout" for r in candidates)
    if monitoring_context is not None:
        candidates = guard_candidates(candidates, metadata['anomaly_guard'])
        check_operational_deadline(deadline, cancelled)
    for candidate in candidates:
        if monotonic() >= deadline - min(5, budget_seconds / 10):
            timed_out = True; break
        row = {k: copy.deepcopy(v) for k, v in candidate.items() if not k.startswith("_")}
        if candidate.get("status") not in {"movement_checked", "search_incomplete"} or candidate.get("_data") is None:
            table.append(row); continue
        changed = copy.deepcopy(candidate["_data"])
        full_schedule_km = committed_schedule_km(changed)
        changed["period_end"] = (now + timedelta(seconds=horizon)).isoformat()
        trials = []
        try:
            for scenario, source in sources.items():
                if monotonic() >= deadline - min(5, budget_seconds / 10):
                    raise TimeoutError("no remaining detailed validation budget")
                runner = GroupedPassengerSimulation(changed, changed["plans"][0]["plan_id"], demand[scenario], profiles,
                    car_preference=getattr(source, 'car_preference', None),
                    boarding_seed=getattr(source, 'boarding_seed', None),
                    boarding_capacity_per_car=getattr(source, 'boarding_capacity_per_car', None))
                check_operational_deadline(deadline, cancelled)
                state = runner.branch_checkpoint(snapshots[scenario], source)
                check_operational_deadline(deadline, cancelled)
                out = runner.run(start_second=start, checkpoint=state, deadline=deadline, cancel_event=cancelled)
                check_operational_deadline(deadline, cancelled)
                exposure = operational_exposure(out["checkpoint"], profiles, policy, start, start + horizon)
                check_operational_deadline(deadline, cancelled)
                headway_end = min(start + horizon, service_end_second) if service_end_second is not None else start + horizon
                headways = operational_headways(changed, runner.motion, start, headway_end)
                check_operational_deadline(deadline, cancelled)
                instant_peak = max((r["load"] for r in out["load_events"]), default=0)
                trial = dict(scenario=scenario, metrics=out["metrics"], policy_exposure=exposure,
                    verification_scope=dict(headway_end_second=headway_end, observation_end_second=start+horizon),
                    headways=headways, cohort=cohort_waiting(runner, out["checkpoint"], start, start + horizon),
                    mature_cohort=cohort_waiting(runner, out["checkpoint"], start, mature_end),
                    cohort_boundary=dict(approach_delay_seconds=approach_delay,
                        headway_max_seconds=parameter(policy, "headway_max_s"), allowance_seconds=allowance,
                        mature_end_second=mature_end, late_entries_are_censored=True),
                    initial_obligations=initial_service_obligations(runner, snapshots[scenario], out["checkpoint"], start),
                    hard_passed=instant_peak <= parameter(policy, "hard_limit"),
                    resources=work_between(changed, runner.motion, start, start + horizon))
                if passenger_service_check is not None:
                    trial['peak_boarding_service'] = passenger_service_check(runner, out['checkpoint'], start, mature_end)
                trials.append(trial)
                check_operational_deadline(deadline, cancelled)
            row.update(status="evaluated", passengers_checked=True,
                physical_passed=all(r["hard_passed"] for r in trials),
                policy_passed=all(r["policy_exposure"]["admissible"] is True or
                    r["policy_exposure"]["status"] == "unknown" and r["metrics"]["entered"] == 0
                    and r["initial_obligations"]["pending_at_start"] == 0
                    and snapshots[r["scenario"]]["onboard_count"] == 0 for r in trials),
                service_passed=all(r["headways"]["passed"] and r["mature_cohort"]["complete"]
                                  and r["initial_obligations"]["complete"]
                                  and r.get('peak_boarding_service', {}).get('passed', True) for r in trials),
                committed_schedule_km=full_schedule_km,
                metrics=dict(max_car_load=max(r["metrics"]["max_car_load"] for r in trials),
                    waiting_passenger_seconds=max(r["metrics"]["waiting_passenger_seconds"] for r in trials),
                    comfort_car_seconds=max(r["policy_exposure"]["global"]["comfort_car_seconds"] for r in trials)),
                trials=trials, horizon_resources_are_partial=True, cash_delta_RUB=None)
        except TimeoutError as error:
            row.update(status="evaluation_timeout", passengers_checked=False, reason=str(error)); timed_out = True
        except ValueError as error:
            row.update(status="rejected", passengers_checked=False, reason=str(error))
        table.append(row)
    check_operational_deadline(deadline, cancelled)
    selection = select_operational_candidate(table, timed_out=timed_out, search_incomplete=search_incomplete)
    return dict(**metadata, candidate_table=table, selection=selection, search=search,
                elapsed_seconds=monotonic() - began, search_complete=not timed_out and not search_incomplete,
                physical_effect_deadline_is_notification_deadline=False)
