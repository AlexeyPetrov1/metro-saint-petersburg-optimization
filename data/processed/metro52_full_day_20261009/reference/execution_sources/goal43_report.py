"""Rebuild GOAL 4.3 reports from saved journals, without rerunning simulations."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from goal43_demo import write_demo
from goal43_policy import build_train_registry, load_policy, validate_train_registry


ROOT = Path(__file__).resolve().parent
LATENCY_SCOPE = "detection through forecast, candidate generation and detailed validation"
METRICS = ("mean_wait_seconds", "p95_wait_seconds", "max_car_load", "exceptional_time_share",
           "train_km", "saving_km", "to1_resource_value_rub", "lost_revenue_rub")


def _read(path, default=None):
    if not path.is_file():
        return default
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return float(result) if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _instant(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _folder(output, record):
    path = Path(record["folder"])
    return path if path.is_absolute() else output / path


def _records(output):
    result = _read(output / "runs.json", [])
    return result.get("runs", []) if isinstance(result, dict) else result


def _matching(records, scenario_id, comparison):
    candidates = [row for row in records if row.get("scenario_id") == scenario_id
                  and row.get("strategy") == comparison.get("strategy")]
    candidate = comparison.get("candidate_id", comparison.get("strategy", ""))
    prefix = f"{scenario_id}_{candidate.replace(':', '_').replace('+', '-')}_"
    exact = [row for row in candidates if row.get("candidate_id") == candidate or row.get("run_id", "").startswith(prefix)]
    return exact or candidates


def _service(output, records, verified):
    if not records:
        return "unknown", ["No matching saved run"], None
    failures, unknown, full_day = [], [], all(row.get("full_day") is True for row in records)
    for record in records:
        folder = _folder(output, record)
        summary = _read(folder / "summary.json", record)
        motion = _read(folder / "motion.json", {})
        metrics = summary.get("metrics", {})
        exposure = metrics.get("policy_exposure", {}).get("admissible")
        headways = summary.get("headways", {}).get("passed")
        checks = {"car_time_policy": exposure, "headways": headways,
                  "obligations_closed": motion.get("obligations_closed")}
        if summary.get("entered") is not None and summary.get("completed") is not None:
            checks["passenger_obligations"] = (summary["entered"] == summary["completed"] and summary.get("queued", 0) == 0)
        else:
            checks["passenger_obligations"] = None
        for key, value in checks.items():
            if value is False:
                failures.append(f"{record['run_id']}: {key}")
            elif value is not True:
                unknown.append(f"{record['run_id']}: {key} unknown")
        if verified.get(record["run_id"], {}).get("violations"):
            failures.append(f"{record['run_id']}: independently reconstructed violations")
    return ("violated" if failures else "unknown" if unknown else "admissible"), failures + unknown, full_day


def _bundles(value):
    if isinstance(value, list):
        for item in value:
            yield from _bundles(item)
    elif isinstance(value, dict):
        if isinstance(value.get("rows"), list):
            yield value
        for name in ("bundles", "forecasts"):
            if name in value:
                yield from _bundles(value[name])


def _forecast_errors(output, scenarios, records):
    rows, seen = [], set()
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        folder = output / "scenarios" / scenario_id
        candidates = [row for row in records if row.get("scenario_id") == scenario_id and row.get("od") == "S1"]
        record = next((row for row in candidates if row.get("strategy") == "KEEP"), next(iter(candidates), None))
        motion, groups, actual_available = {}, [], False
        if record:
            run_folder = _folder(output, record)
            motion = _read(run_folder / "motion.json", {})
            groups = _read(run_folder / "groups.json.gz", [])
            actual_available = (run_folder / "groups.json.gz").is_file()
            groups = groups.get("groups", []) if isinstance(groups, dict) else groups
        origin = _instant(motion["period_start"]) if motion.get("period_start") else None
        assumptions = _read(folder / "assumptions.json", {})
        coverage = assumptions.get("service_end_second", motion.get("duration_seconds"))
        station_entries = defaultdict(list)
        for group in groups:
            station_entries[group.get("origin", group.get("station_id"))].append((group["turnstile_second"], group["count"]))
        index = {}
        for station, events in station_entries.items():
            events.sort()
            prefix = [0]
            for _, count in events:
                prefix.append(prefix[-1] + count)
            index[station] = ([second for second, _ in events], prefix)
        bundles = list(_bundles(scenario.get("forecasts", [])))
        for path in sorted(folder.rglob("*forecast*.json")):
            bundles.extend(_bundles(_read(path)))
        for bundle in bundles:
            for forecast in bundle["rows"]:
                identity = (scenario_id, bundle.get("issued_at", bundle.get("as_of")), forecast.get("station_id"),
                            forecast.get("interval_start"), forecast.get("interval_end"))
                if identity in seen:
                    continue
                seen.add(identity)
                actual, reason = None, "missing actual journal or coverage"
                if actual_available and origin is not None and coverage is not None:
                    first = (_instant(forecast["interval_start"]) - origin).total_seconds()
                    last = (_instant(forecast["interval_end"]) - origin).total_seconds()
                    if 0 <= first < last <= coverage:
                        seconds, prefix = index.get(forecast["station_id"], ([], [0]))
                        actual = prefix[bisect_left(seconds, last)] - prefix[bisect_left(seconds, first)]
                        reason = None
                predicted = _number(forecast.get("predicted_count"))
                rows.append(dict(scenario_id=scenario_id, issued_at=identity[1], station_id=forecast.get("station_id"),
                    interval_start=forecast.get("interval_start"), interval_end=forecast.get("interval_end"),
                    predicted_entries=forecast.get("predicted_count"), actual_entries=actual,
                    absolute_error=None if actual is None or predicted is None else abs(predicted - actual),
                    target=bundle.get("target", "turnstile_entries"), status="covered" if actual is not None else "unknown",
                    reason=reason, actual_run_id=None if record is None else record["run_id"]))
    covered = [row for row in rows if row["absolute_error"] is not None]
    error, actual = sum(row["absolute_error"] for row in covered), sum(row["actual_entries"] for row in covered)
    return rows, dict(target="turnstile_entries", covered_rows=len(covered), missing_fact_rows=sum(row["actual_entries"] is None for row in rows),
        absolute_error=error, actual_entries=actual, wape=error / actual if actual else None,
        scope="saved forecast windows; overlapping issues remain separate predictions; not an independent model holdout")


def _provenance(output):
    verification = ROOT / "data/processed/goal4_3_verification"
    candidates = [output / "sourcebaseline_manifest.json", output / "baseline_manifest.json", verification / "baseline_manifest.json"]
    path = next((value for value in candidates if value.is_file()), None)
    baseline = dict(status="not_provided", checked_files=0)
    if path:
        baseline = _read(path)
        baseline.setdefault("workspace_root", str(ROOT))
        baseline.setdefault("archived_baseline_source", str((verification / "baseline_source").resolve()))
        _write(output / "sourcebaseline_manifest.json", baseline)
        baseline = dict(status="referenced_not_rechecked", manifest=str(path.resolve()),
            report_manifest=str((output / "sourcebaseline_manifest.json").resolve()),
            archived_baseline_source=baseline["archived_baseline_source"], file_count=len(baseline.get("files", {})),
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    refresh_root = ROOT / "data/processed/goal4_3_source_refresh"
    semantic = _read(refresh_root / "semantic_comparison.json", {})
    refresh = _read(refresh_root / "source_refresh_manifest.json", {})
    rows = []
    for item in refresh.get("files", []):
        download = item.get("download", {})
        current = item.get("current_metadata", {})
        source_id = item.get("source_id")
        rows.append(dict(kind="source_refresh", name=source_id,
            value=download.get("workspace_path"), unit="file", source=current.get("url"),
            status=item.get("status"), sha256=download.get("sha256"),
            semantic_equal=semantic.get(source_id, {}).get("semantic_equal")))
    policy = load_policy()
    for name, value in policy["parameters"].items():
        rows.append(dict(kind="policy_parameter", name=name, **value))
    for name, value in policy.get("demo", {}).items():
        rows.append(dict(kind="demo_assumption", name=name, value=value, unit="configured demo value",
                         source="goal4_3.json demo profile", status="assumed"))
    usage_path = ROOT / "docs/goal43_parameter_usage.json"
    usage = _read(usage_path, {})
    changed = []
    for name, value in usage.get("code_files", {}).items():
        source = ROOT / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != value.get("sha256"):
            changed.append(name)
    return dict(baseline=baseline, source_refresh=refresh, semantic_comparison=semantic,
        policy=str(ROOT / "config/decision_engine/goal4_3.json"), incident_sources=_read(ROOT / "docs/goal43_incident_sources.json", {}),
        parameter_usage=dict(path=str(usage_path), status="stale_snapshot" if changed else "snapshot_consistent" if usage else "not_provided",
            changed_code_files=changed, evidence_scope=usage.get("evidence_scope"))), rows


def _campaign(output, records, verified):
    result, trials, scenarios, train_rows, economy_rows = [], [], [], [], []
    by_run = {row["run_id"]: row for row in records}
    for path in sorted((output / "campaign").glob("*/manifest.json")):
        manifest = _read(path)
        stage = manifest.get("stage", path.parent.name)
        expected = 20 if stage == "validation" else 10 if stage == "preliminary" else None
        count = manifest.get("seed_count", len(manifest.get("seeds", [])))
        trial_rows = manifest.get("trials", [])
        declared = set(manifest.get("seeds", []))
        completed = {row.get("future_seed") for row in trial_rows}
        result.append(dict(stage=stage, seed_count=count, expected_seed_count=expected,
            complete_seed_count=(count == expected and len(declared) == expected and declared <= completed) if expected is not None else None,
            completed_seed_count=len(completed),
            trial_count=len(trial_rows), elapsed_seconds=manifest.get("elapsed_seconds"),
            scope=manifest.get("scope", {}), aggregate=manifest.get("aggregate", []), manifest=str(path)))
        for trial in trial_rows:
            trial = {"stage": stage, **trial}
            if verified.get(trial.get("run_id"), {}).get("violations"):
                trial["service_admissible"] = False
            trials.append(trial)
            if trial.get("status") != "evaluated":
                continue
            identity = f"campaign_{stage}_{trial.get('future_seed')}_{trial.get('od')}"
            record = by_run.get(trial.get("run_id"))
            economy = _read(_folder(output, record) / "economics.json", {}) if record else {}
            per_train = [dict(train_id=train, train_type=row.get("train_type"), baseline_km=row.get("baseline_km"),
                candidate_km=row.get("candidate_km"), saving_km=row.get("saving_km")) for train, row in economy.get("per_train", {}).items()]
            complete = len(per_train) == 68 and all(_number(row.get("saving_km")) is not None for row in per_train)
            mean = _number(sum((Decimal(str(row["saving_km"])) for row in per_train), Decimal(0)) / 68) if complete else None
            train_rows.extend(dict(scenario_id=identity, candidate_id=trial.get("candidate_id"), strategy=trial.get("strategy"), **row)
                              for row in per_train)
            if economy:
                economy_rows.append(dict(scenario_id=identity, candidate_id=trial.get("candidate_id"), **economy.get("totals", {}),
                    cash_savings_rub=economy.get("cash_savings_rub"), cash_status=economy.get("cash_status"),
                    lost_revenue_rub=economy.get("revenue", {}).get("lost_revenue_rub"),
                    lost_profit_rub=economy.get("revenue", {}).get("lost_profit_rub")))
            scenario = next((row for row in scenarios if row["scenario_id"] == identity), None)
            if scenario is None:
                scenario = dict(scenario_id=identity, title=f"Искусственный тест: {stage}, seed {trial.get('future_seed')}, {trial.get('od')}",
                    evaluation_mode=manifest.get("scope", {}).get("evaluation_mode", "synthetic_fixture"),
                    incident=None, decisions=[], comparisons=[], release_authorized=False,
                    provenance={"scope": manifest.get("scope", {}), "saved_manifest": str(path)})
                scenarios.append(scenario)
            scenario["comparisons"].append(dict(strategy=trial.get("strategy"), candidate_id=trial.get("candidate_id"),
                status="pass" if trial.get("service_admissible") is True else "fail" if trial.get("service_admissible") is False else "unknown", elapsed_seconds=None,
                metrics={**{key: _number(trial.get(key)) for key in METRICS}, "mean_saving_km_per_train": mean}, series={},
                card_od=trial.get("od"), od_results=[dict(od=trial.get("od"), seed=trial.get("future_seed"),
                    mean_wait_seconds=trial.get("mean_wait_seconds"), p95_wait_seconds=trial.get("p95_wait_seconds"),
                    max_car_load=trial.get("max_car_load"), exceptional_time_share=trial.get("exceptional_time_share"),
                    headways_passed=(record or {}).get("headways", {}).get("passed"))],
                per_train=[{key: _number(value) if key.endswith("_km") else value for key, value in row.items()} for row in per_train],
                release_authorized=False))
    return result, trials, scenarios, train_rows, economy_rows


def _departures(output, record, cache):
    if record is None:
        return {}
    if record["run_id"] in cache:
        return cache[record["run_id"]]
    folder = _folder(output, record)
    data, motion = _read(folder / "movement_input.json", {}), _read(folder / "motion.json", {})
    tracks = {row["track_id"]: row for row in data.get("tracks", [])}
    stations = {row["station_id"]: row for row in data.get("stations", [])}
    trips = {row["trip_id"]: row for plan in data.get("plans", []) for row in plan.get("trips", [])}
    result = {}
    for event in motion.get("events", []):
        track = tracks.get(event.get("location"))
        if event.get("event_type") != "departure" or track is None or trips.get(event.get("trip_id"), {}).get("passenger") is not True:
            continue
        if not all(stations.get(track[name], {}).get("kind") == "line" for name in ("from_station", "to_station")):
            continue
        key = event["train_id"], event["trip_id"], event["location"]
        result[key] = event["second"], track["from_station"]
    cache[record["run_id"]] = result
    return result


def _actual_effects(output, scenario, records):
    cache = {}
    sid = scenario["scenario_id"]
    rows = [row for row in records if row.get("scenario_id") == sid and row.get("od") == "S1"]
    baseline = next((row for row in rows if row.get("strategy") == "KEEP"), None)
    keep = _departures(output, baseline, cache)
    rolling = next((row for row in rows if row.get("strategy") == "ROLLING_SELECTED"), None)
    for index, decision in enumerate(scenario.get("decisions", [])):
        start = decision.get("decision_second")
        if start is None:
            continue
        selected = decision.get("selection", {}).get("selected_candidate_id") or decision.get("selection", {}).get("mitigation_candidate_id")
        for candidate in decision.get("candidate_table", []):
            action = candidate.get("action")
            if not isinstance(action, dict) or action.get("kind") not in ("REGULATE_ONLY", "WITHDRAW_ONLY", "COMBINED"):
                continue
            if index and candidate.get("candidate_id") != selected:
                continue
            record = rolling if index else next(iter(_matching(rows, sid, candidate)), None)
            changed = _departures(output, record, cache)
            by_station = {}
            if baseline and record:
                for key in keep.keys() | changed.keys():
                    before, after = keep.get(key), changed.get(key)
                    if before == after:
                        continue
                    times = [row[0] for row in (before, after) if row is not None and row[0] >= start]
                    if not times:
                        continue
                    station = (after or before)[1]
                    by_station[station] = min(by_station.get(station, min(times)), min(times))
            first = min(by_station.values(), default=None)
            action.update(timetable_change_second=first, timetable_change_by_station=by_station,
                actual_effect_kind="timetable change", guaranteed_improvement=False,
                effect_evidence=dict(baseline_run_id=None if baseline is None else baseline["run_id"],
                    candidate_run_id=None if record is None else record["run_id"],
                    stable_key=["train_id", "trip_id", "track_id"],
                    scope="cumulative saved timetable difference after this decision" if index else "saved alternative against KEEP"))
            if action.get("actual_effect_second") is None:
                action["actual_effect_second"] = first
            if not action.get("actual_effect_by_station"):
                action["actual_effect_by_station"] = by_station
            if action["kind"] in ("WITHDRAW_ONLY", "COMBINED"):
                for child in action.get("actions", []):
                    if child.get("actual_effect_second") is not None:
                        child.setdefault("depot_return_second", child["actual_effect_second"])


def _saved_queue_age(state):
    result = dict(oldest_queue_age_seconds=None, oldest_turnstile_wait_age_seconds=None,
                  checkpoint_second=state.get("second"))
    second, queues = state.get("second"), state.get("queues")
    if type(second) is not int or not isinstance(queues, (dict, list)):
        return result
    rows = [row for group in queues.values() for row in group] if isinstance(queues, dict) else queues
    positive = [row for row in rows if isinstance(row, dict) and (_number(row.get("count")) or 0) > 0]
    for key, timestamp in (("oldest_queue_age_seconds", "platform_second"), ("oldest_turnstile_wait_age_seconds", "turnstile_second")):
        times = [row.get(timestamp) for row in positive]
        if all(type(value) is int and 0 <= value <= second for value in times):
            result[key] = second-min(times) if times else 0
    return result


def _presentation(source):
    shown = deepcopy(source)
    sid = shown.get("scenario_id", "")
    titles = {"fixture_offpeak": "Вне пика: искусственный контроль",
        "historical_offpeak": "Вне пика: исторические входы",
        "forecast_stress": "Ошибка прогноза: стрессовый спрос",
        "station_stop_5_to_15": "Пересмотр срока: с 5 до 15 минут",
        "station_stop_unknown": "Остановка с неизвестным сроком"}
    incident = shown.get("incident")
    if isinstance(incident, dict):
        start, end = _number(incident.get("detected_second")), _number(shown.get("injected_actual_end_second"))
        actual = _number(incident.get("actual_duration_seconds", incident.get("duration_seconds")))
        actual_source = "saved incident duration" if actual is not None else None
        if actual is None and start is not None and end is not None:
            actual, actual_source = end-start, "saved injected actual end minus detection"
        if actual is not None and actual < 0:
            actual, actual_source = None, None
        estimated_end = _number(incident.get("expected_end_second"))
        estimate = estimated_end-start if start is not None and estimated_end is not None else None
        incident.update(actual_duration_minutes=None if actual is None else actual/60,
            actual_duration_seconds=actual, actual_duration_source=actual_source,
            initial_estimated_duration_seconds=estimate if estimate is not None and estimate >= 0 else None,
            controller_estimate_history=[dict(decision_second=row.get("decision_second"), estimates=row.get("incident_estimates", []))
                for row in shown.get("decisions", [])])
        if sid.startswith("station_stop_") and sid not in titles:
            if actual is None:
                titles[sid] = "Остановка состава: фактическая длительность неизвестна"
            else:
                minutes = actual/60
                suffix = "минуту" if minutes.is_integer() and minutes % 10 == 1 and minutes % 100 != 11 else \
                    "минуты" if not minutes.is_integer() or minutes % 10 in (2, 3, 4) and minutes % 100 not in (12, 13, 14) else "минут"
                titles[sid] = f"Остановка состава на {minutes:g} {suffix}"
    if not shown.get("title") or shown["title"] == sid:
        shown["title"] = titles.get(sid, shown.get("title", sid))
    return shown


def _demo(scenarios, campaign, comparisons, provenance, output, records):
    policy = load_policy()
    value = lambda name: policy["parameters"][name]["value"]
    rows = [_presentation(row) for row in scenarios]
    service = {(row["scenario_id"], row["candidate_id"]): row for row in comparisons}
    for scenario in rows:
        _actual_effects(output, scenario, records)
        scenario["release_authorized"] = False
        for decision in scenario.get("decisions", []):
            selection = decision.get("selection", {})
            if decision.get("reason"):
                continue
            if selection.get("selected_candidate_id") and selection.get("policy_version") == "closed-schedule-km-comfort-soft-v1":
                decision["reason"] = ("Среди проверенных допустимых вариантов выбран план с меньшим запланированным пробегом; "
                    "при равенстве учитываются ожидание и время выше комфортной загрузки. Это не оценка денежной прибыли.")
                if selection.get("status") == "search_incomplete":
                    decision["reason"] += " Поиск не завершён; выбор относится только к проверенной части вариантов."
            elif selection.get("status") == "no_feasible_candidate":
                opening = ("Среди проверенных вариантов допустимый план не найден. Поиск не завершён; отсутствие допустимого решения не доказано. "
                    if decision.get("search_complete") is False else "Допустимый вариант не найден. ")
                decision["reason"] = opening + ("Проверенная мера смягчения, если она указана, "
                    "выбрана по максимальной загрузке и суммарному ожиданию; нарушения ограничений сохраняются.")
            elif selection.get("status") == "evaluation_timeout":
                decision["reason"] = "Время проверки исчерпано; рекомендация не выдана."
        for comparison in scenario.get("comparisons", []):
            comparison["od_results"] = []
            comparison["card_od"] = "S1"
            comparison["card_metrics_scope"] = "S1; alternative destinations are reported separately"
            comparison.setdefault("metrics", {})
            matching = _matching(records, scenario["scenario_id"], comparison)
            primary = next((record for record in matching if record.get("od") == "S1"), None)
            ages = _saved_queue_age(_read(_folder(output, primary) / "passenger_checkpoint.json.gz", {}) if primary else {})
            comparison["metrics"].update({key: ages[key] for key in ("oldest_queue_age_seconds", "oldest_turnstile_wait_age_seconds")})
            comparison["queue_age_scope"] = dict(od="S1", checkpoint_second=ages["checkpoint_second"],
                run_id=None if primary is None else primary["run_id"], includes_closure_tail=True,
                basis="saved positive-count platform queues after the resource horizon and closure tail")
            for record in matching:
                metrics = record.get("metrics", {})
                windows = metrics.get("policy_exposure", {}).get("windows")
                window_shares = [_number(row.get("exceptional_time_share")) for row in windows or []]
                failed = [dict(start_second=row.get("start"), end_second=row.get("end"),
                    exceptional_time_share=row.get("exceptional_time_share")) for row in windows or [] if row.get("status") == "fail"]
                comparison["od_results"].append(dict(od=record.get("od"), seed=record.get("seed"),
                    queued=record.get("queued"), entered=record.get("entered"), completed=record.get("completed"),
                    cohort_scope=record.get("cohort_scope"), resources_scope=record.get("resources_scope"),
                    failed_15m_windows=failed if windows is not None else None,
                    maximum_window_share=max((value for value in window_shares if value is not None), default=None),
                    headways_passed=record.get("headways", {}).get("passed"),
                    mean_wait_seconds=metrics.get("cohort", {}).get("mean_wait_seconds"),
                    p95_wait_seconds=metrics.get("cohort", {}).get("p95_wait_seconds"),
                    max_car_load=metrics.get("max_car_load"),
                    exceptional_time_share=metrics.get("policy_exposure", {}).get("global", {}).get("exceptional_time_share")))
            for key in METRICS:
                if key in comparison.get("metrics", {}):
                    comparison["metrics"][key] = _number(comparison["metrics"][key])
            for row in comparison.get("per_train", []):
                for key in ("baseline_km", "candidate_km", "saving_km"):
                    row[key] = _number(row.get(key))
            checked = service.get((scenario["scenario_id"], comparison.get("candidate_id", comparison.get("strategy"))))
            fixed = comparison.get("per_train", [])
            complete = checked and checked["fixed_train_count"] == len(fixed) == 68 and all(_number(row.get("saving_km")) is not None for row in fixed)
            comparison["metrics"]["mean_saving_km_per_train"] = _number(checked.get("mean_saving_km_per_train")) if complete else None
            if checked:
                comparison["audit_full_day"] = checked["full_day"]
                comparison["status"] = {"admissible": "pass", "violated": "fail"}.get(checked["service_status"], "unknown")
                comparison["service_findings"] = checked["service_findings"]
            comparison["release_authorized"] = False
        scenario["provenance"] = dict(source=scenario.get("evaluation_mode"), baseline_manifest=provenance["baseline"],
            conditional=True, profit_status="unknown", release_authorized=False)
    return dict(schema_version=1, title="GOAL 4.3: сохранённые расчёты и проверка условий",
        scenarios=rows + campaign, release_authorized=False,
        confirmed_parameters=[
            "Час пик по будням: 08:00–10:00 и 17:00–19:00.",
            f"Минимальный интервал: {value('peak_headway_min_s')} с в час пик, {value('offpeak_headway_min_s')} с вне пика; максимальный — {value('headway_max_s')} с.",
            f"Разворот: {value('turnaround_s')} с. Выпуск до первого пассажирского отправления: Автово — {value('depot_release_s')['Avtovo']} с, Северное — {value('depot_release_s')['Northern']} с.",
            f"Парк: {value('fleet_by_type')['Baltiets']} Балтийцев и {value('fleet_by_type')['Yubileyny']} Юбилейных; на линии не более {value('line_train_limit')} составов, включая остановленные.",
            f"В составе {value('cars_per_train')} вагонов. Комфорт — {value('comfort_limit')}, обычный предел — {value('normal_limit')}, жёсткий максимум — {value('hard_limit')} человек на вагон.",
            f"Загрузка выше обычного предела допустима не более {value('exceptional_time_share_max') * 100:g}% времени перевозки пассажиров.",
            f"Пересмотр каждые {value('decision_step_s')} с; локальное уведомление и проверенная рекомендация — за {value('notification_budget_s')} с.",
            f"ТО1 через {value('to1_interval_km')} км; валовой доход с оплаченной поездки — {value('fare_revenue_rub')} руб."
        ],
        assumptions=[
            "Направления, станции назначения и перемещение от турникета до платформы заданы сценариями.",
            "Демо задаёт пути, соединения депо, два места для направлений на конечных и одно на промежуточных станциях; парковка конечных ограничена одним составом.",
            "Начальная расстановка и готовность двух резервных составов заданы явно для демо.",
            "Искусственные инциденты и будущие реализации проверяют заданные условия; они не подтверждают вероятность эффекта на реальной линии.",
            "Расход ресурса ТО1 рассчитан по пробегу; таблица выручки использует явно предполагаемые 1000 потенциально неоплаченных поездок."
        ],
        unknowns=[
            "Фактические назначения пассажиров, схемы сигнализации и занятости путей, рабочие наряды и готовность резерва.",
            "Одометры после последнего ТО1 и будущий пробег каждого состава; реальные даты отложенного обслуживания.",
            "Отказы до оплаты, возвраты, переменные затраты и изменение денежных выплат; фактическая упущенная выручка и прибыль.",
            "Доставка уведомления внешнему адресату; возраст очереди неизвестен там, где нет полного сохранённого состояния."
        ])


def _economic_artifacts(output, scenarios, comparisons, trials):
    from goal43_economics import maintenance_registry, revenue_sensitivity, to1_deferral
    policy = load_policy()
    shares = ["0", "0.001", "0.01", "0.05"]
    registry = maintenance_registry(policy)
    sensitivity = []
    contexts = [*comparisons, *(dict(row, scenario_id=f"campaign_{row.get('stage')}_{row.get('future_seed')}_{row.get('od')}") for row in trials)]
    for row in contexts:
        benefit = row.get("to1_resource_value_rub")
        value = dict(scenario_id=row["scenario_id"], candidate_id=row.get("candidate_id"),
            service_status=row.get("service_status", "admissible" if row.get("service_admissible") is True else "violated"),
            benefit_kind="TO1_resource_accrual", potential_unpaid_trips=1000, status="assumed",
            actual_lost_revenue_rub=None, lost_profit_rub=None,
            demand_basis="explicit assumed potential unpaid trips; not paid queue stock")
        if _number(benefit) is None:
            value.update(status="unknown_benefit", benefit_rub=None, break_even_additional_unpaid_trips=None, scenarios=[])
        else:
            value.update(revenue_sensitivity(dict(benefit_rub=benefit, benefit_kind="TO1_resource_accrual",
                potential_unpaid_trips=1000, abandonment_shares=shares), policy=policy))
        sensitivity.append(value)
    _write(output / "revenue_sensitivity.json", dict(schema_version=1, status="assumed_sensitivity",
        fare_revenue_rub=registry["fare_revenue_rub"], potential_unpaid_trips=1000, abandonment_shares=shares,
        assumptions_source="GOAL 4.3 explicit report sensitivity grid; no measured abandonment supplied",
        actual_lost_revenue_rub=None, lost_profit_rub=None, comparisons=sensitivity, release_authorized=False))
    fleet_path = next((output / "scenarios" / row["scenario_id"] / "fleet_registry.json" for row in scenarios
                       if (output / "scenarios" / row["scenario_id"] / "fleet_registry.json").is_file()), None)
    fleet = _read(fleet_path) if fleet_path else build_train_registry(policy)
    validate_train_registry(fleet, policy)
    deferrals = []
    for train in fleet:
        result = to1_deferral(dict(km_since_to1=train.get("km_since_to1"), baseline_evaluation_km=None,
                                  candidate_evaluation_km=None, future_daily_km=None), policy=policy)
        deferrals.append(dict(train_id=train["train_id"], train_type=train["type"],
            odometer_km=train.get("odometer_km"), km_since_to1=train.get("km_since_to1"), **result))
    _write(output / "maintenance_deferral.json", dict(schema_version=1, fleet=deferrals,
        registry_source=str(fleet_path) if fleet_path else str(ROOT / "config/decision_engine/goal4_3.json"),
        known_comparison_mileage="tables/train_km.csv", future_daily_mileage_status="unknown",
        odometer_status="not measured; missing registry values remain unknown", cash_savings_rub=None,
        note="Per-train mileage differences do not determine a maintenance date without the last-TO1 odometer and a future plan",
        release_authorized=False))
    return dict(revenue_sensitivity="revenue_sensitivity.json", maintenance_deferral="maintenance_deferral.json")


def _markdown(report):
    lines = ["# GOAL 4.3: результаты сохранённых расчётов", "",
        f"Сформировано: {report['generated_at']}. Сценариев: {report['scenario_count']}; журналов: {report['run_count']}.", "",
        "Результат условный: входы берутся из исторических таблиц или явно искусственного сценария, "
        "направления и назначения задаются OD-сценариями. "
        "Инфраструктура и готовность резерва в демо заданы явно; разрешение на выпуск отсутствует.", "",
        f"Положительных допустимых примеров полного дня: {report['positive_admissible_examples']}. "
        f"В основных сценариях: {report['positive_admissible_scenario_examples']}; в искусственных будущих реализациях: {report['positive_admissible_future_trials']}. "
        "Сокращение километров при нарушении интервала или других ограничений не считается подтверждённым эффектом.", "",
        "## Пробег и ограничения", "", "| Сценарий | Стратегия | Δ км | Среднее Δ км/состав | ТО1, руб. ресурса | Условия обслуживания | Полный день |",
        "|---|---|---:|---:|---:|---|---|"]
    for row in report["comparisons"]:
        lines.append(f"| {row['scenario_id']} | {row['candidate_id']} | {row.get('saving_km')} | "
            f"{row.get('mean_saving_km_per_train')} | {row.get('to1_resource_value_rub')} | {row['service_status']} | {row['full_day']} |")
    lines += ["", "Среднее рассчитано на фиксированный набор составов, включая неиспользованные. "
        "[Нарушения по журналам](tables/service_runs.csv) включают области оценки и число превышений интервала; "
        "превышения во время искусственной аварии сохраняются в результате.", "", "ТО1: 7,4 руб./км для Юбилейного и 9,2 руб./км для Балтийца. Это оценка расхода ресурса, "
        "а не уменьшение текущих выплат. Денежная экономия, упущенная прибыль и выручка неизвестны без "
        "подтверждённых затрат, отказов до оплаты и возвратов. Ожидание уже оплатившего пассажира само по себе не создаёт потерю выручки.", "",
        "[Чувствительность к неоплаченным поездкам](revenue_sensitivity.json) использует явно предполагаемые 1000 потенциальных "
        "поездок и доли отказов 0 / 0,1% / 1% / 5%. Порог сравнивается только с ресурсом ТО1; это не упущенная прибыль. "
        "[Дата ТО1 по каждому из 68 составов](maintenance_deferral.json) остаётся неизвестной без одометра и будущего плана.", "",
        "## Решения и время поиска", "", "| Сценарий | Момент решения | Записано, с | Полное измерение | В пределах 480 с | Выбор |",
        "|---|---|---:|---|---|---|"]
    for row in report["decisions"]:
        lines.append(f"| {row['scenario_id']} | {row.get('decision_time')} | {row.get('elapsed_seconds')} | "
                     f"{row['complete_latency_scope']} | {row['within_budget']} | {row.get('selected_candidate_id')} |")
    forecast = report["forecast_errors"]
    lines += ["", "## Ошибка прогноза входов", "",
        f"Покрытых строк: {forecast['covered_rows']}; без факта: {forecast['missing_fact_rows']}; WAPE: {forecast['wape']}.", "",
        "Расчёт относится к сохранённым окнам прогноза медианы. Повторные прогнозы одного окна оцениваются отдельно. "
        "Это не независимая проверка обученной модели; норматив 6% не подтверждается этим отчётом.", "",
        "## Искусственные будущие реализации", "",
        "Предварительный набор: 10 seed; отдельная проверка: 20 seed. Выбранный план фиксируется до проверочного набора. "
        "Этот тест использует искусственный участок и не даёт вероятностной гарантии для линии с 53 составами.", "",
        "| Этап | Seed сохранено | Ожидалось | Набор полный | Реализаций | Длительность, с |", "|---|---:|---:|---|---:|---:|"]
    for row in report["campaign"]:
        lines.append(f"| {row['stage']} | {row['seed_count']} | {row['expected_seed_count']} | {row['complete_seed_count']} | "
                     f"{row['trial_count']} | {row['elapsed_seconds']} |")
    baseline = report["source_provenance"]["baseline"]
    lines += ["", "## Источники и целостность", "",
        f"Независимая проверка журналов: {report['verification']['status']}. "
        f"Baseline: {baseline.get('file_count', baseline.get('checked_files', 0))} файлов; архив `{baseline.get('archived_baseline_source')}`.", "",
        "Свежие D02/D05 сравниваются по значениям после адаптации формата. Результат семантического сравнения:", ""]
    for name, value in report["source_provenance"]["semantic_comparison"].items():
        if isinstance(value, dict):
            lines.append(f"- {name}: semantic_equal={value.get('semantic_equal')}; различающихся строк={value.get('value_rows_differing', 'unknown')}.")
    lines += ["", "Данные движения относятся к линии 1. 200,4 млн поездок за 2025 год относятся к синей линии "
        "и используются только как отдельная справка: при 50,9 руб. это 10,20036 млрд руб. валовой выручки. "
        "Переносить этот объём на линию 1 или масштабировать им эффект демо нельзя.", "",
        "Подробные таблицы: [решения](tables/decisions.csv), [километры составов](tables/train_km.csv), "
        "[экономика](tables/economics.csv), [ошибки прогноза](tables/forecast_errors.csv), "
        "[происхождение](tables/source_provenance.csv), [будущие реализации](tables/campaign_trials.csv). "
        "[Открыть демо](demo.html).", ""]
    return "\n".join(lines)


def finalize(output, *, verify_integrity=False):
    """Create JSON/Markdown/CSV/HTML artifacts; final verification is explicit."""
    began = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance, source_rows = _provenance(output)
    previous_check = _read(output / "verification_report.json", {})
    verification = dict(status="not_rechecked", previous_status=previous_check.get("status"),
                        previous_run_count=previous_check.get("run_count"))
    if verify_integrity:
        from goal43_verify import verify
        previous_check = verify(output)
        verification = dict(status=previous_check["status"], run_count=previous_check["run_count"],
                            manifest_checks=previous_check["manifest_checks"], report="verification_report.json")
    verified = {row["run_id"]: row for row in previous_check.get("runs", [])} if verify_integrity else {}
    scenarios = [_read(path) for path in sorted((output / "scenarios").glob("*/summary.json"))]
    records = _records(output)
    comparisons, decisions, train_rows, economy_rows, service_runs = [], [], [], [], []
    for record in records:
        summary = _read(_folder(output, record) / "summary.json", record)
        service_runs.append(dict(run_id=record["run_id"], scenario_id=record.get("scenario_id"), strategy=record.get("strategy"),
            od=record.get("od"), seed=record.get("seed"), full_day=record.get("full_day"),
            cohort_scope=summary.get("cohort_scope"), resources_scope=summary.get("resources_scope"),
            service_scope=summary.get("service_scope"),
            exposure_admissible=summary.get("metrics", {}).get("policy_exposure", {}).get("admissible"),
            headways_passed=summary.get("headways", {}).get("passed"),
            recorded_headway_violations=summary.get("headways", {}).get("violations", []),
            independently_reconstructed_violations=verified.get(record["run_id"], {}).get("violations")))
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        for index, decision in enumerate(scenario.get("decisions", [])):
            elapsed = _number(decision.get("elapsed_seconds"))
            decisions.append(dict(scenario_id=scenario_id, decision_index=index, decision_time=decision.get("decision_time"),
                elapsed_seconds=elapsed, latency_scope=decision.get("latency_scope"),
                complete_latency_scope=decision.get("latency_scope") == LATENCY_SCOPE,
                within_budget=None if elapsed is None else elapsed <= 480,
                selected_candidate_id=decision.get("selection", {}).get("selected_candidate_id"),
                selection_status=decision.get("selection", {}).get("status"), release_authorized=False))
        economy = _read(output / "scenarios" / scenario_id / "economics.json", [])
        for value in economy:
            economy_rows.append(dict(scenario_id=scenario_id, candidate_id=value.get("candidate_id"),
                **value.get("totals", {}), cash_savings_rub=value.get("cash_savings_rub"), cash_status=value.get("cash_status"),
                lost_revenue_rub=value.get("revenue", {}).get("lost_revenue_rub"),
                lost_profit_rub=value.get("revenue", {}).get("lost_profit_rub"), coverage_complete=value.get("coverage_complete")))
        for comparison in scenario.get("comparisons", []):
            candidate_id = comparison.get("candidate_id", comparison.get("strategy"))
            status, findings, full_day = _service(output, _matching(records, scenario_id, comparison), verified)
            metrics = comparison.get("metrics", {})
            saving = _number(metrics.get("saving_km"))
            fixed_count = len({row["train_id"] for row in comparison.get("per_train", [])})
            fixed = comparison.get("per_train", [])
            complete = fixed_count and fixed_count == len(fixed) and all(_number(row.get("saving_km")) is not None for row in fixed)
            mean = str(sum((Decimal(str(row["saving_km"])) for row in fixed), Decimal(0)) / fixed_count) if complete else None
            comparisons.append(dict(scenario_id=scenario_id, candidate_id=candidate_id, strategy=comparison.get("strategy"),
                evaluation_mode=scenario.get("evaluation_mode"), full_day=full_day, service_status=status, service_findings=findings,
                fixed_train_count=fixed_count, mean_saving_km_per_train=mean,
                positive_admissible_example=status == "admissible" and full_day is True and saving is not None and saving > 0,
                **{key: metrics.get(key) for key in METRICS}))
            for row in comparison.get("per_train", []):
                train_rows.append(dict(scenario_id=scenario_id, candidate_id=candidate_id, strategy=comparison.get("strategy"), **row))
    forecast_rows, forecasts = _forecast_errors(output, scenarios, records)
    campaign, trial_rows, campaign_scenarios, campaign_trains, campaign_economy = _campaign(output, records, verified)
    train_rows.extend(campaign_trains)
    economy_rows.extend(campaign_economy)
    positive_trials = sum(row.get("service_admissible") is True and row.get("full_day") is True
        and row.get("obligations_closed") is True and (_number(row.get("saving_km")) or 0) > 0 for row in trial_rows)
    positive_scenarios = sum(row["positive_admissible_example"] for row in comparisons)
    economic_artifacts = _economic_artifacts(output, scenarios, comparisons, trial_rows)
    payload = _demo(scenarios, campaign_scenarios, comparisons, provenance, output, records)
    _write(output / "demo_payload.json", payload)
    write_demo(output, payload)
    tables = output / "tables"
    _csv(tables / "decisions.csv", decisions, ("scenario_id", "decision_index", "decision_time", "elapsed_seconds", "latency_scope", "complete_latency_scope", "within_budget", "selected_candidate_id", "selection_status", "release_authorized"))
    _csv(tables / "train_km.csv", train_rows, ("scenario_id", "candidate_id", "strategy", "train_id", "train_type", "baseline_km", "candidate_km", "saving_km"))
    _csv(tables / "economics.csv", economy_rows or comparisons, ("scenario_id", "candidate_id", "baseline_km", "candidate_km", "train_km", "saving_km", "mean_saving_km_per_train", "to1_resource_value_rub", "cash_savings_rub", "cash_status", "lost_revenue_rub", "lost_profit_rub", "coverage_complete", "service_status"))
    _csv(tables / "service_runs.csv", service_runs, ("run_id", "scenario_id", "strategy", "od", "seed", "full_day", "cohort_scope", "resources_scope", "service_scope", "exposure_admissible", "headways_passed", "recorded_headway_violations", "independently_reconstructed_violations"))
    _csv(tables / "forecast_errors.csv", forecast_rows, ("scenario_id", "issued_at", "station_id", "interval_start", "interval_end", "predicted_entries", "actual_entries", "absolute_error", "target", "status", "reason", "actual_run_id"))
    _csv(tables / "source_provenance.csv", source_rows, ("kind", "name", "value", "unit", "source", "status", "sha256", "semantic_equal"))
    _csv(tables / "campaign_trials.csv", trial_rows, ("stage", "future_seed", "past_seed", "od", "candidate_id", "strategy", "run_id", "status", "service_admissible", "full_day", "obligations_closed", "entered", "completed", "queued", *METRICS, "saving_km_per_assigned_train", "future_movement_digest"))
    old_performance = _read(output / "performance.json", {})
    performance = dict(total_seconds=old_performance.get("total_seconds"), budget_seconds=480,
        total_seconds_scope="last recorded replay invocation; not the sum of all artifacts",
        decisions=decisions, decision_seconds=[row["elapsed_seconds"] for row in decisions],
        campaign=[{key: row[key] for key in ("stage", "seed_count", "trial_count", "elapsed_seconds")} for row in campaign],
        report_generation_seconds=time.perf_counter() - began,
        scope="recorded replay and detection-to-checked-decision timings; browser rendering is not search latency")
    report = dict(schema_version=1, status="generated" if scenarios else "partial",
        generated_at=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(), scenario_count=len(scenarios), run_count=len(records),
        positive_admissible_examples=positive_scenarios + positive_trials,
        positive_admissible_scenario_examples=positive_scenarios, positive_admissible_future_trials=positive_trials,
        conditional=True, release_authorized=False, comparisons=comparisons, decisions=decisions, service_runs=service_runs,
        forecast_errors=forecasts, campaign=campaign, source_provenance=provenance, verification=verification,
        performance=performance, economic_artifacts=economic_artifacts, cash_savings_rub=None, lost_profit_rub=None)
    _write(output / "performance.json", performance)
    _write(output / "report.json", report)
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true", help="Independently verify all saved journals before finalizing")
    args = parser.parse_args()
    result = finalize(args.output, verify_integrity=args.verify)
    print(json.dumps({key: result[key] for key in ("status", "scenario_count", "run_count", "positive_admissible_examples")}, ensure_ascii=False))
