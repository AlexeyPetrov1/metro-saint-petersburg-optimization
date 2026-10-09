"""Frozen-model versus median forecasts in the same causal dispatch controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from economics import write_json
from forecasting import hourly_median
from goal43_controller import run_controller
from goal43_economics import maintenance_registry
from goal43_policy import load_policy
from goal43_replay import actual_groups, build_operational_day, save_run, _series
from passenger_groups import GroupedPassengerSimulation
from replay_metrics import cohort_waiting, work_between
from rolling import operational_exposure, operational_headways
from simulation import digest


ROOT = Path(__file__).resolve().parent
VERSION = "metro-ml-controller-comparison-v1"
REAL_QUARTER_SOURCE = "uploaded-spb-line1-15min"
WINDOWS = {"peak": dict(start_hour=8, start_minute=15, decision_hour=9, train_count=53, headway_targets=[130]),
           "offpeak": dict(start_hour=11, start_minute=0, decision_hour=12, train_count=38, headway_targets=[180])}


def median_provider(frame, station_ids, as_of, *, horizon_hours=2):
    at = datetime.fromisoformat(as_of).replace(minute=0, second=0, microsecond=0)
    return hourly_median(frame, station_ids, at.isoformat(), horizon_hours=horizon_hours)


def forecast_error(bundle, frame, decision_time):
    """Score future targets only; observations never become model features here."""
    now = pd.Timestamp(decision_time)
    rows = pd.DataFrame(bundle["rows"])
    rows["interval_start"] = pd.to_datetime(rows.interval_start, utc=True)
    rows = rows[rows.interval_start.ge(now)].copy()
    if rows.empty or rows.duplicated(["station_id", "interval_start"]).any():
        raise ValueError("forecast needs unique future intervals")
    truth = frame[["station_id", "interval_start", "entry_count", "is_complete", "source_id"]]
    joined = rows.merge(truth, on=["station_id", "interval_start"], how="left", validate="one_to_one")
    if joined.entry_count.isna().any() or not joined.is_complete.eq(True).all():
        raise ValueError("forecast evaluation requires complete actual coverage")
    if not joined.source_id.eq(REAL_QUARTER_SOURCE).all():
        raise ValueError("evaluation requires real quarter-hour targets; hourly warmup is not actual truth")
    error = joined.predicted_count - joined.entry_count
    total = int(joined.entry_count.sum())
    return dict(rows=len(joined), actual_entries=total, predicted_entries=float(joined.predicted_count.sum()),
        absolute_error=float(error.abs().sum()), wape_percent=100*float(error.abs().sum())/total if total else None,
        mae_passengers_per_15m=float(error.abs().mean()),
        rmse_passengers_per_15m=float((error.pow(2).mean())**.5),
        signed_bias_percent=100*float(error.sum())/total if total else None,
        decision_time=decision_time, first_target=joined.interval_start.min().isoformat(),
        last_target=joined.interval_start.max().isoformat(),
        overlapping_horizons_must_not_be_summed=True)


def metric_delta(baseline, candidate, policy=None):
    base = {r["train_id"]: r for r in baseline["per_train"]}
    changed = {r["train_id"]: r for r in candidate["per_train"]}
    if base.keys() != changed.keys() or not base:
        raise ValueError("comparison requires an identical fixed fleet")
    rates = maintenance_registry(policy)["to1_resource_rates_rub_per_train_km"]
    per_train = []
    for train in sorted(base):
        left, right = base[train], changed[train]
        if left["train_type"] != right["train_type"]:
            raise ValueError("fixed fleet train types changed")
        saving = float(left["km"]) - float(right["km"])
        rate = float(rates[left["train_type"].casefold()])
        per_train.append(dict(train_id=train, train_type=left["train_type"], baseline_km=float(left["km"]),
            candidate_km=float(right["km"]), saving_km=saving, to1_resource_equivalent_rub=saving*rate))
    saving = float(baseline["train_km"]) - float(candidate["train_km"])
    wait = None if baseline["mean_wait_seconds"] is None or candidate["mean_wait_seconds"] is None else candidate["mean_wait_seconds"]-baseline["mean_wait_seconds"]
    return dict(observed_saving_train_km=saving, observed_saving_km_per_fixed_train=saving/len(base),
        mean_wait_delta_seconds=wait, per_train=per_train,
        observed_to1_resource_equivalent_rub=sum(r["to1_resource_equivalent_rub"] for r in per_train),
        cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None,
        scope="closed four-hour experiment including common warmup and depot return; no daily or annual scaling",
        resource_value_is_cash_saving=False)


def _metrics(data, runner, state, roster, policy, start, service_end):
    resources = work_between(data, runner.motion, 0, runner.duration)
    controlled_resources = work_between(data, runner.motion, start, service_end)
    cohort = cohort_waiting(runner, state, start, service_end)
    mature_end = service_end-600
    mature = cohort_waiting(runner,state,start,mature_end)
    exposure = operational_exposure(state, runner.profiles, policy, start, service_end)
    headways = operational_headways(data, runner.motion, start, service_end)
    types = {r["train_id"]: r["type"] for r in roster}
    per_train = [dict(train_id=row["train_id"], train_type=types[row["train_id"]],
        km=float(row["passenger_train_km"])+float(row["nonpassenger_train_km"]))
        for row in runner.motion["transport_work"]]
    events = [r for r in state["queue_events"] if start <= r["second"] < service_end]
    prior = next((r for r in reversed(state["queue_events"]) if r["second"] < start), None)
    if prior is not None: events.append(prior)
    peak_load = exposure["global"]["max_load"]
    return dict(train_km=float(resources["train_km"]),
        mean_km_per_fixed_train=float(resources["train_km"])/len(roster),
        fixed_fleet_size=len(roster), per_train=per_train, resources=resources,
        resources_scope=dict(start_second=0, end_second=runner.duration, includes_common_warmup=True, includes_depot_return=True),
        control_window_resources=controlled_resources,
        control_window_scope=dict(start_second=start, end_second=service_end, boundary_movement_is_partial=True),
        mean_wait_seconds=mature["mean_wait_seconds"], p95_wait_seconds=mature["p95_wait_seconds"],
        cohort=cohort,mature_cohort=mature,
        mean_wait_scope=dict(start_second=start,end_second=mature_end,
            late_arrival_allowance_seconds=600,identical_across_strategies=True),
        max_queue=max((r["queued"] for r in events), default=0),
        max_platform_direction_queue=max((r["max_platform_direction_queue"] for r in events), default=0),
        max_car_load=peak_load, policy_exposure=exposure, headways=headways,
        entered=state["entered"], completed=state["completed"], queued_at_closure=state["queued"],
        approaching_at_closure=state["approaching"], onboard_at_closure=state["onboard_count"],
        movement_obligations_closed=runner.motion["obligations_closed"],
        actual_service_admissible=exposure["admissible"] is True and headways["passed"] and mature["complete"],
        all_service_arrivals_completed=cohort["complete"],
        cash_saving_rub=None, lost_revenue_rub=None, release_authorized=False)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(output, date, window, frame, forecaster, *, steps=2, seed=7, od="S1"):
    """Use one shared observed checkpoint and independent future worlds for each strategy."""
    began = time.perf_counter()
    case_id = f"{date}_{window}_{od}"
    folder = Path(output)/"cases"/case_id
    folder.mkdir(parents=True, exist_ok=True)
    spec = WINDOWS[window]
    policy = load_policy()
    origin = f"{date}T{spec['start_hour']:02}:{spec['start_minute']:02}:00+03:00"
    actual_window = frame[frame.interval_start.ge(pd.Timestamp(origin)) &
        frame.interval_start.lt(pd.Timestamp(origin)+pd.Timedelta(hours=4))]
    if actual_window.empty or not actual_window.source_id.eq(REAL_QUARTER_SOURCE).all():
        raise ValueError("simulation requires real quarter-hour targets; synthetic hourly warmup cannot be scored")
    data, roster, ready, profiles, meta = build_operational_day(policy, count=spec["train_count"],
        start_at=origin, service_hours=4, seed=seed)
    policy = data["headway_policy"]
    all_groups = actual_groups(frame, data, meta["service_end_second"])
    groups = {od: all_groups[od]}
    sources = {od: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], groups[od], profiles)}
    first = (spec["decision_hour"]-spec["start_hour"])*3600
    snapshots = {od: sources[od].run(end_second=first)["checkpoint"]}
    ids = [s["station_id"] for s in data["stations"] if s["kind"] == "line"]
    day_rows = frame[frame.station_id.isin(ids) & frame.interval_start.ge(pd.Timestamp(origin)) &
        frame.interval_start.lt(pd.Timestamp(origin)+pd.Timedelta(hours=4))].copy()
    contract = dict(version=VERSION, case_id=case_id, date=date, window=window, od=od, seed=seed,
        requested_line_id=2, actual_simulated_line_id=1, blue_line_mapping_status="unresolved; supplied simulator and model describe line 1",
        initial_count=spec["train_count"], steps=steps, control_seconds=900, forecast_refresh_seconds=3600,
        forecast_horizon_seconds=7200, candidate_headway_targets_seconds=spec["headway_targets"],
        minimum_remaining=spec["train_count"]-2, service_seconds=meta["service_end_second"],
        actual_groups_digest=digest(groups[od]), initial_checkpoint_digest=digest(snapshots[od]),
        movement_digest=digest(data), policy_digest=digest(policy), station_ids=ids,
        actual_rows=len(day_rows), actual_entries=int(day_rows.entry_count.sum()),
        actual_interval_digest=digest(day_rows.assign(interval_start=day_rows.interval_start.map(lambda x:x.isoformat()),
            interval_end=day_rows.interval_end.map(lambda x:x.isoformat()))[["station_id","interval_start","interval_end","entry_count"]].to_dict("records")),
        source_hashes={str(path.relative_to(ROOT)): _file_hash(path) for path in
            [ROOT/"data/raw/metro/data.zip", ROOT/"data/raw/metro/lgbm_final.zip", ROOT/"goal43_actions.py",
             ROOT/"metro_ml_compare.py", ROOT/"goal43_controller.py", ROOT/"goal43_replay.py", ROOT/"metro_ml.py"] if path.exists()},
        observation_rule="completed intervals only; hourly refresh uses features available at the hour boundary",
        warmup="one hour real station entries; initially empty trains and platforms are assumed",
        actual_future_only_for_evaluation=True, assumptions=meta,
        forecast_status="held out after frozen training cutoff" if date > "2026-09-29" else "diagnostic; training/calibration overlap",
        model_station_coverage="18 model stations plus explicit hourly median fallback for L1_07",
        daily_or_annual_savings_claimed=False, release_authorized=False)
    identity = digest(contract)
    manifest = folder/"experiment_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["experiment_identity"] != identity:
            raise ValueError(f"existing case identity differs: {case_id}; choose a new isolated output")
        if (folder/"metrics.json").exists():
            return json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    else:
        write_json(manifest, dict(experiment_identity=identity, contract=contract))
        write_json(folder/"warmup_checkpoint.json", snapshots[od])
        write_json(folder/"actual_groups.json", groups[od])
        write_json(folder/"fleet_registry.json", roster)
        write_json(folder/"policy.json", policy)
    strategies = {}
    for strategy in ("KEEP", "MEDIAN", "ML"):
        strategy_file = folder/f"{strategy.lower()}_result.json"
        if strategy_file.exists():
            strategies[strategy] = json.loads(strategy_file.read_text(encoding="utf-8")); continue
        started = time.perf_counter()
        if strategy == "KEEP":
            runner = sources[od]
            state = runner.run(start_second=first, checkpoint=snapshots[od])["checkpoint"]
            changed = data; decisions = []; bundles = []; trace = []
        else:
            provider = median_provider if strategy == "MEDIAN" else forecaster.forecast_bundle
            controlled = run_controller(data, sources, snapshots, profiles, ready, frame, policy,
                scenario_id=f"{case_id}_{strategy}", minimum_remaining=spec["train_count"]-2,
                groups=groups, steps=steps, seed=seed, forecast_provider=provider,
                candidate_headway_targets=spec["headway_targets"])
            changed = controlled["data"]; runner = controlled["sources"][od]; state = controlled["states"][od]
            decisions = controlled["decisions"]; bundles = controlled["bundles"]; trace = controlled["trace"]
        run_id = f"{case_id}_{strategy}_{seed}"
        record = save_run(output, run_id, changed, runner, state, roster, policy, case_id, strategy, od, seed,
            full_day=False, cohort_start=first, cohort_end=meta["service_end_second"], service_end_second=meta["service_end_second"])
        metrics = _metrics(changed, runner, state, roster, policy, first, meta["service_end_second"])
        scores = [forecast_error(bundle, frame, decision["decision_time"]) for bundle, decision in zip(bundles, decisions)]
        issue_ids = set(); unique_scores = []
        for bundle, score in zip(bundles, scores):
            if bundle["issued_at"] not in issue_ids:
                unique_scores.append(score); issue_ids.add(bundle["issued_at"])
        actions = [dict(decision_time=d["decision_time"], decision_second=d["decision_second"],
            selection=d["selection"], candidates=[{k:v for k,v in r.items() if k in ("candidate_id","strategy","status","action","service_passed","policy_passed","physical_passed","committed_schedule_km")}
                for r in d["candidate_table"]], elapsed_seconds=d["elapsed_seconds"],
            forecast_version=d.get("forecast_version"), forecast_issued_at=d.get("forecast_issued_at"),
            available_horizon_seconds=d.get("available_horizon_seconds")) for d in decisions]
        result = dict(strategy=strategy, initial_checkpoint_digest=contract["initial_checkpoint_digest"],
            actual_groups_digest=contract["actual_groups_digest"], metrics=metrics,
            decisions=actions, forecast_errors=scores, unique_issue_forecast_errors=unique_scores,
            run_folder=record["folder"], elapsed_seconds=time.perf_counter()-started,
            series=_series(runner,state,first,meta["service_end_second"],bundles[0],changed) if bundles else None,
            release_authorized=False)
        strategies[strategy] = result
        write_json(folder/f"{strategy.lower()}_decisions.json", decisions)
        write_json(folder/f"{strategy.lower()}_forecasts.json", bundles)
        write_json(folder/f"{strategy.lower()}_trace.json", trace)
        write_json(strategy_file,result)
        print(json.dumps(dict(stage="metro_ml_strategy", case_id=case_id, strategy=strategy,
            elapsed_seconds=result["elapsed_seconds"], metrics={k:metrics[k] for k in
                ("train_km","mean_wait_seconds","max_queue","max_car_load","actual_service_admissible")}),ensure_ascii=False),flush=True)
    for strategy in strategies:
        strategies[strategy]["delta_vs_keep"] = metric_delta(strategies["KEEP"]["metrics"], strategies[strategy]["metrics"], policy)
    result = dict(case_id=case_id, date=date, window=window, od=od, experiment_identity=identity,
        forecast_status=contract["forecast_status"], strategies=strategies,
        ml_vs_median=metric_delta(strategies["MEDIAN"]["metrics"],strategies["ML"]["metrics"],policy),
        elapsed_seconds=time.perf_counter()-began, requested_line_id=2, actual_simulated_line_id=1,
        release_authorized=False)
    write_json(folder/"metrics.json",result)
    return result


def collect_report(output):
    output = Path(output)
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output/"cases").glob("*/metrics.json"))]
    rows = []
    for case in cases:
        for strategy,result in case["strategies"].items():
            metrics = result["metrics"]; delta = result["delta_vs_keep"]
            scores = result["unique_issue_forecast_errors"]
            rows.append(dict(date=case["date"],window=case["window"],od=case["od"],strategy=strategy,
                forecast_status=case["forecast_status"],train_km=metrics["train_km"],
                km_per_fixed_train=metrics["mean_km_per_fixed_train"],
                observed_saving_km_per_fixed_train=delta["observed_saving_km_per_fixed_train"],
                mean_wait_seconds=metrics["mean_wait_seconds"],p95_wait_seconds=metrics["p95_wait_seconds"],
                wait_cohort_start_second=metrics["mean_wait_scope"]["start_second"],
                wait_cohort_end_second=metrics["mean_wait_scope"]["end_second"],
                late_arrival_allowance_seconds=metrics["mean_wait_scope"]["late_arrival_allowance_seconds"],
                whole_cohort_complete=metrics["cohort"]["complete"],
                mature_cohort_complete=metrics["mature_cohort"]["complete"],
                max_queue=metrics["max_queue"],max_car_load=metrics["max_car_load"],
                actual_service_admissible=metrics["actual_service_admissible"],
                forecast_wape_percent=scores[0]["wape_percent"] if scores else None,
                observed_to1_resource_equivalent_rub=delta["observed_to1_resource_equivalent_rub"],
                max_decision_seconds=max((d["elapsed_seconds"] for d in result["decisions"]),default=None),
                cash_saving_rub=None,lost_revenue_rub=None,run_folder=result["run_folder"]))
    report = dict(version=VERSION,cases=cases,case_count=len(cases),rows=rows,
        requested_line_id=2,actual_simulated_line_id=1,
        blue_line_mapping_status="unresolved; model and simulator line 1 are explicitly retained",
        comparison="same optimizer and candidate headway grid; only frozen forecast provider differs",
        limitations=["simulated timetable and OD S1 are assumed", "four-hour cases with one-hour warmup and two rolling decisions",
            "no daily savings or cash effect", "ML is hourly; 15-minute allocation is an assumed causal historical profile",
            "ML18stations plus median fallback for missing L1_07", "only dates after 2026-09-29 are frozen-model temporal holdout"],
        release_authorized=False)
    write_json(output/"report.json",report)
    pd.DataFrame(rows).to_csv(output/"rows.csv",index=False,encoding="utf-8-sig")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/processed/metro_ml_comparison")
    parser.add_argument("--dates",nargs="+",default=["2026-09-30","2026-09-29"])
    parser.add_argument("--windows",nargs="+",choices=list(WINDOWS),default=list(WINDOWS))
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--seed",type=int,default=7)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == ROOT or ROOT not in output.parents or "goal4_2" in str(output) or "goal4_3" in str(output):
        raise ValueError("comparison output must be a new isolated workspace descendant")
    output.mkdir(parents=True,exist_ok=True)
    from metro_ml import FrozenMetroForecaster
    model = FrozenMetroForecaster(allow_training_period=True)
    frame = model.load_simulator_frame()
    for date in args.dates:
        for window in args.windows:
            run_case(output,date,window,frame,model,steps=args.steps,seed=args.seed)
            collect_report(output)
    report = collect_report(output)
    print(json.dumps(dict(status="complete",case_count=report["case_count"],output=str(output)),ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
