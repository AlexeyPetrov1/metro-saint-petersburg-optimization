"""Offline, fixed-past robustness campaign on an explicit three-station fixture."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from dwell import compile_operational_timing
from economics import write_json
from forecasting import hourly_median
from goal43_actions import generate_strategy_candidates
from goal43_economics import compare_train_economics
from goal43_policy import load_policy
from goal43_replay import (actual_groups, build_operational_day, fixture_history,
                           mileage_context, save_run)
from passenger_groups import GroupedPassengerSimulation
from rolling import decide_operational
from simulation import digest


PAST_SEED = 7
DECISION_SECOND = 21600
STATION_IDS = ["L1_01", "L1_07", "L1_19"]
STAGES = {"preliminary": tuple(range(100, 110)), "validation": tuple(range(200, 220))}
_CONTEXT = None
_SOURCES = None


def future_world(data, policy, as_of, seed, commands=()):
    """Keep committed movement and service; draw only the future keyed dwell world."""
    lower = {(row["train_id"], row["trip_id"], row["leg_index"]): row["not_before_second"] for row in commands}
    changed, _ = compile_operational_timing(data, data["plans"][0]["plan_id"], "mild", seed,
                                           policy=policy, as_of=as_of, departure_not_before=lower)
    return changed


def _seed_contract(seeds, stage, selected_plan):
    if stage not in STAGES:
        raise ValueError("stage must be preliminary or validation")
    chosen = list(STAGES[stage] if seeds is None else seeds)
    if not chosen or any(type(seed) is not int or seed < 0 for seed in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError("seeds must contain distinct nonnegative integer worlds")
    if stage == "validation" and selected_plan is not None:
        if set(chosen) & set(selected_plan.get("preliminary_seeds", STAGES["preliminary"])):
            raise ValueError("preliminary and validation seeds overlap")
    return chosen


def _prepare(stage, seeds, selected_plan):
    policy = load_policy()
    data, roster, ready, profiles, meta = build_operational_day(policy, count=3, service_hours=18,
        profile="mild", seed=PAST_SEED, small_line=STATION_IDS)
    policy = data["headway_policy"]
    ids = [row["station_id"] for row in data["stations"] if row["kind"] == "line"]
    if ids != STATION_IDS:
        raise ValueError("campaign requires the explicit three-station fixture with stochastic intermediate dwell")
    frame = fixture_history(ids, count_per_minute=4)
    groups = actual_groups(frame, data, meta["service_end_second"])
    sources = {od: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], rows, profiles)
               for od, rows in groups.items()}
    snapshots = {od: source.run(end_second=DECISION_SECOND)["checkpoint"] for od, source in sources.items()}
    at = datetime.fromisoformat(data["period_start"])+timedelta(seconds=DECISION_SECOND)
    forecast = hourly_median(frame, ids, at.isoformat())
    context = dict(base_data=data, roster=roster, profiles=profiles, policy=policy, meta=meta,
                   groups=groups, snapshots=snapshots, stage=stage, seeds=seeds)
    if stage == "preliminary":
        candidates, search = generate_strategy_candidates(data, DECISION_SECOND, policy, ready,
            minimum_remaining=2, max_withdrawals=1, include_recovery=False, profile="mild", seed=PAST_SEED)
        decision = decide_operational(data, sources, snapshots, profiles, ready, forecast, policy,
            candidates=candidates, profile="mild", seed=PAST_SEED, include_recovery=False)
        selected_id = decision["selection"].get("selected_candidate_id")
        selected = next((row for row in candidates if row["candidate_id"] == selected_id), None)
        if selected is None or selected.get("_data") is None:
            raise ValueError("forecast-only preparation found no frozen checked campaign plan")
        selected_plan = dict(schema_version=1, candidate_id=selected_id, strategy=selected["strategy"],
            withdrawn_train_ids=copy.deepcopy(selected["action"].get("withdrawn_train_ids", [])),
            commanded_departures=copy.deepcopy(selected.get("timing", {}).get("commanded_departures", [])),
            movement_input=copy.deepcopy(selected["_data"]), plan_digest=digest(selected["_data"]),
            planning_seed=PAST_SEED, past_seed=PAST_SEED, decision_second=DECISION_SECOND,
            preliminary_seeds=seeds, validation_seeds_reserved=list(STAGES["validation"]),
            selection=copy.deepcopy(decision["selection"]), selected_before_future_trials=True,
            selection_inputs="observed common snapshot and hourly station median; no future actual trial outcomes")
        context.update(candidates=candidates, decision=decision, search=search, forecast=forecast)
    else:
        if (selected_plan.get("selected_before_future_trials") is not True
                or selected_plan.get("decision_second") != DECISION_SECOND
                or selected_plan.get("planning_seed") != PAST_SEED
                or selected_plan.get("plan_digest") != digest(selected_plan.get("movement_input"))):
            raise ValueError("invalid frozen campaign descriptor or movement digest")
        frozen = selected_plan["movement_input"]
        if frozen["trains"] != data["trains"] or frozen["period_start"] != data["period_start"] or frozen["period_end"] != data["period_end"]:
            raise ValueError("frozen plan does not share campaign fleet and boundaries")
        candidates = [dict(candidate_id="KEEP", strategy="KEEP", status="movement_checked", _data=data, timing={})]
        if selected_plan["candidate_id"] != "KEEP":
            candidates.append(dict(candidate_id=selected_plan["candidate_id"], strategy=selected_plan["strategy"],
                status="movement_checked", _data=frozen,
                timing=dict(commanded_departures=selected_plan.get("commanded_departures", []))))
        context.update(candidates=candidates, decision=None, search=None, forecast=forecast)
    context["selected_plan"] = copy.deepcopy(selected_plan)
    return context


def _init_worker(context):
    global _CONTEXT, _SOURCES
    _CONTEXT = context
    data = context["base_data"]
    _SOURCES = {od: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], rows, context["profiles"])
                for od, rows in context["groups"].items()}


def _trial(seed):
    context, sources = _CONTEXT, _SOURCES
    began = perf_counter()
    trials, records, baselines = [], [], {}
    roster, policy, profiles = (context[key] for key in ("roster", "policy", "profiles"))
    output = Path(context["stage_output"])
    fleet = {row["train_id"]: row["type"] for row in roster}
    for candidate in context["candidates"]:
        candidate_id, strategy = candidate["candidate_id"], candidate["strategy"]
        if candidate.get("_data") is None or candidate["status"] not in {"movement_checked", "search_incomplete"}:
            for od in sources:
                trials.append(dict(stage=context["stage"], future_seed=seed, past_seed=PAST_SEED,
                    od=od, candidate_id=candidate_id, strategy=strategy, status=candidate["status"],
                    reason=candidate.get("reason"), service_admissible=False, full_day=True,
                    obligations_closed=False, saving_km=None))
            continue
        try:
            changed = future_world(candidate["_data"], policy, DECISION_SECOND, seed,
                                   candidate.get("timing", {}).get("commanded_departures", []))
        except ValueError as error:
            for od in sources:
                trials.append(dict(stage=context["stage"], future_seed=seed, past_seed=PAST_SEED,
                    od=od, candidate_id=candidate_id, strategy=strategy, status="rejected",
                    reason=str(error), service_admissible=False, full_day=True,
                    obligations_closed=False, saving_km=None))
            continue
        for od, source in sources.items():
            runner = GroupedPassengerSimulation(changed, changed["plans"][0]["plan_id"], context["groups"][od], profiles)
            snapshot = runner.branch_checkpoint(context["snapshots"][od], source)
            state = runner.run(start_second=DECISION_SECOND, checkpoint=snapshot)["checkpoint"]
            suffix = candidate_id.replace(":", "_").replace("+", "-")
            run_id = f"campaign_{context['stage']}_{seed}_{suffix}_{od}"
            record = save_run(output, run_id, changed, runner, state, roster, policy,
                "synthetic_three_station_campaign", strategy, od, seed, full_day=True,
                cohort_start=DECISION_SECOND, cohort_end=context["meta"]["service_end_second"],
                service_end_second=context["meta"]["service_end_second"])
            record["folder"] = str(Path("campaign") / context["stage"] / record["folder"]).replace("\\", "/")
            record.update(past_seed=PAST_SEED, future_seed=seed, campaign_stage=context["stage"])
            records.append(record)
            mileage = mileage_context(changed, runner.motion, roster, digest(context["groups"][od]),
                                       "none", seed, full_day=True)
            mileage["random_world_id"] = f"keyed-dwell:past{PAST_SEED}:future{seed}"
            if candidate_id == "KEEP":
                baselines[od] = mileage
            economy = compare_train_economics(baselines[od], mileage, fleet, policy=policy)
            write_json(Path(context["root_output"]) / record["folder"] / "economics.json", economy)
            service = (record["metrics"]["policy_exposure"]["admissible"] is True
                       and record["headways"]["passed"] and runner.motion["obligations_closed"]
                       and state["entered"] == state["completed"])
            cohort = record["metrics"]["cohort"]
            trials.append(dict(stage=context["stage"], future_seed=seed, past_seed=PAST_SEED,
                od=od, candidate_id=candidate_id, strategy=strategy, run_id=run_id, status="evaluated",
                service_admissible=service, full_day=True, obligations_closed=runner.motion["obligations_closed"],
                entered=state["entered"], completed=state["completed"], queued=state["queued"],
                mean_wait_seconds=cohort["mean_wait_seconds"], p95_wait_seconds=cohort["p95_wait_seconds"],
                max_car_load=state["max_car_load"], train_km=record["metrics"]["resources"]["train_km"],
                saving_km=economy["totals"]["saving_km"],
                saving_km_per_assigned_train=economy["totals"].get("mean_saving_km_per_train"),
                to1_resource_value_rub=economy["totals"]["to1_resource_value_rub"],
                future_movement_digest=digest(changed["leg_timing"])))
    return dict(trials=trials, records=records, future_seed=seed, elapsed_seconds=perf_counter()-began)


def _aggregate(trials):
    grouped = defaultdict(list)
    for row in trials:
        grouped[row["strategy"], row["od"]].append(row)
    result = []
    for (strategy, od), rows in sorted(grouped.items()):
        values = [Decimal(str(row["saving_km"])) for row in rows if row.get("saving_km") is not None]
        waits = [float(row["mean_wait_seconds"]) for row in rows if row.get("mean_wait_seconds") is not None]
        result.append(dict(strategy=strategy, od=od, worlds=len(rows),
            admissible_worlds=sum(row["service_admissible"] for row in rows),
            saving_km_min=str(min(values)) if values else None, saving_km_max=str(max(values)) if values else None,
            mean_wait_seconds_median=median(waits) if waits else None,
            worst_mean_wait_seconds=max(waits) if waits else None,
            worst_max_car_load=max((row.get("max_car_load", 0) for row in rows), default=0)))
    return result


def run_campaign(output, seeds=None, stage="preliminary", selected_plan=None, workers=1):
    """Run 10 selection checks or 20 new worlds against a frozen forecast-selected plan.

    The caller appends returned records to its global runs.json. All files written here live
    under campaign/<stage>. These trials do not establish safety or savings on the real line.
    """
    began = perf_counter()
    chosen = _seed_contract(seeds, stage, selected_plan)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(output).resolve()
    stage_output = output / "campaign" / stage
    if stage == "validation" and selected_plan is None:
        descriptor = output / "campaign/preliminary/selected_plan.json"
        if not descriptor.exists():
            raise ValueError("validation requires a frozen preliminary selected_plan")
        selected_plan = json.loads(descriptor.read_text(encoding="utf-8"))
        chosen = _seed_contract(chosen, stage, selected_plan)
    context = _prepare(stage, chosen, copy.deepcopy(selected_plan))
    context.update(root_output=str(output), stage_output=str(stage_output))
    stage_output.mkdir(parents=True, exist_ok=True)
    write_json(stage_output / "selected_plan.json", context["selected_plan"])
    write_json(stage_output / "forecast.json", context["forecast"])
    if context["decision"] is not None:
        write_json(stage_output / "decision.json", context["decision"])
    scope = dict(evaluation_mode="explicit_synthetic_three_station_fixture", historical_line53_validation=False,
        stations=STATION_IDS, initial_active_trains=3, assigned_fleet=68,
        demand="assumed constant 4 entries per station per minute, 06:00–22:00, then zero",
        movement="30 s per section; keyed mild intermediate dwell; 180 s terminal turnaround",
        past_seed=PAST_SEED, decision_second=DECISION_SECOND, full_day="06:00–00:00 plus common depot-return tail",
        randomness="independent future keyed dwell; identical observed past, demand and infrastructure",
        selection="forecast-only decision before future trials; validation never reranks",
        probability_guarantee=False, release_authorized=False)
    manifest = dict(stage=stage, seeds=chosen, seed_count=len(chosen), scope=scope,
        selected_plan_digest=context["selected_plan"]["plan_digest"],
        selection_recomputed=stage == "preliminary", aggregate=[], trials=[], records=[])
    write_json(stage_output / "manifest.json", manifest)
    if workers == 1:
        _init_worker(context)
        batches = []
        for seed in chosen:
            batch = _trial(seed); batches.append(batch)
            print(json.dumps(dict(stage="goal43_campaign", campaign_stage=stage, seed=seed,
                                  seconds=round(batch["elapsed_seconds"], 2), runs=len(batch["records"]))), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(context,)) as executor:
            batches = []
            for batch in executor.map(_trial, chosen):
                batches.append(batch)
                print(json.dumps(dict(stage="goal43_campaign", campaign_stage=stage, seed=batch["future_seed"],
                                      seconds=round(batch["elapsed_seconds"], 2), runs=len(batch["records"]))), flush=True)
    trials = [row for batch in batches for row in batch["trials"]]
    records = [row for batch in batches for row in batch["records"]]
    manifest.update(trials=trials, records=records, aggregate=_aggregate(trials),
                    per_world_seconds=[dict(seed=batch["future_seed"], seconds=batch["elapsed_seconds"]) for batch in batches],
                    elapsed_seconds=perf_counter()-began)
    write_json(stage_output / "manifest.json", manifest)
    return dict(**manifest, selected_plan=context["selected_plan"])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--stage", choices=("both", "preliminary", "validation"), default="both")
    args = parser.parse_args(argv)
    result = []
    descriptor = None
    if args.stage in {"both", "preliminary"}:
        initial = run_campaign(args.output, stage="preliminary", workers=args.workers)
        descriptor = initial["selected_plan"]
        result.append(initial)
    if args.stage in {"both", "validation"}:
        result.append(run_campaign(args.output, stage="validation", workers=args.workers, selected_plan=descriptor))
    print(json.dumps([dict(stage=row["stage"], seeds=len(row["seeds"]), runs=len(row["records"]),
                           seconds=round(row["elapsed_seconds"], 2), frozen_plan=row["selected_plan_digest"],
                           aggregate=row["aggregate"]) for row in result]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
