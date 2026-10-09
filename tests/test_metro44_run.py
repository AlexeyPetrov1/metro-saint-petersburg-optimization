import gzip
import json

import pytest

import metro44_run as run


def test_forecast_baseline_is_a_single_alias_and_fixed_schedule_has_no_forecast():
    assert run.forecast_source("STANDARD") == "MEDIAN"
    assert run.forecast_source("MPC_BASELINE_FORECAST") == "MEDIAN"
    assert run.forecast_source("MPC_ML") == "ML"
    assert run.forecast_source("MPC_ORACLE") == "ORACLE"
    assert run.forecast_source("FIXED_SCHEDULE") is None
    with pytest.raises(ValueError, match="strategy"):
        run.forecast_source("MEDIAN")


def test_control_grid_has_all_68_quarters_without_reset():
    assert run.control_times(64800) == list(range(3600, 64800, 900))
    assert len(run.control_times(64800)) == 68


def test_clearance_extends_departures_without_changing_admission_or_control_grid():
    assert run.departure_end(64800, "FIXED_SCHEDULE") == 64800
    assert all(run.departure_end(64800, arm) == 65400 for arm in run.ARMS if arm != "FIXED_SCHEDULE")
    assert len(run.control_times(64800)) == 68


def test_station_offsets_include_each_link_and_intermediate_stop():
    data = {"stations": [{"station_id": "A", "kind": "line", "stop_seconds": 0},
                          {"station_id": "B", "kind": "line", "stop_seconds": 20},
                          {"station_id": "C", "kind": "line", "stop_seconds": 0}],
            "tracks": [{"from_station": "A", "to_station": "B", "travel_seconds": 100},
                       {"from_station": "B", "to_station": "A", "travel_seconds": 100},
                       {"from_station": "B", "to_station": "C", "travel_seconds": 120},
                       {"from_station": "C", "to_station": "B", "travel_seconds": 120}]}
    assert run.station_offsets(data) == {"up": [0, 120, 240], "down": [240, 140, 0]}


def test_immutable_checkpoint_refuses_overwrite(tmp_path):
    path = tmp_path / "resume_03600.json.gz"
    run.write_gzip_once(path, {"state": {"second": 3600}})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        run.write_gzip_once(path, {"state": {"second": 4500}})
    assert path.read_bytes() == before


def test_resume_requires_matching_source_input_and_mode_contract(tmp_path):
    record = dict(contract_identity="current", next_index=1, second=3600,
                  state={"second": 3600}, data={"plans": []}, decision={"decision_second": 3600})
    run.write_gzip_once(tmp_path / "resume_03600.json.gz", record)
    assert run.load_resume(tmp_path, "current")["next_index"] == 1
    with pytest.raises(ValueError, match="contract"):
        run.load_resume(tmp_path, "changed")


def test_resume_rejects_inconsistent_clock_instead_of_skipping_an_interval(tmp_path):
    record = dict(contract_identity="current", next_index=2, second=4500,
                  state={"second": 3600}, data={}, decision={})
    run.write_gzip_once(tmp_path / "resume_04500.json.gz", record)
    with pytest.raises(ValueError, match="clock"):
        run.load_resume(tmp_path, "current")


def test_resume_refuses_a_missing_executed_prefix(tmp_path):
    record = dict(contract_identity="current", next_index=2, second=4500,
                  state={"second": 4500}, data={}, decision={})
    run.write_gzip_once(tmp_path / "resume_04500.json.gz", record)
    with pytest.raises(ValueError, match="prefix"):
        run.load_resume(tmp_path, "current")


def test_frozen_sources_are_archived_by_content_and_never_replaced(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    code = source / "module.py"
    code.write_bytes(b"VALUE = 1\n")
    monkeypatch.setattr(run, "ROOT", source)
    sha = run._hash(code)
    output = tmp_path / "output"
    run.archive_sources(output, {"module.py": sha})
    archived = output / "source_versions" / f"{sha}_module.py"
    before = archived.read_bytes()
    run.archive_sources(output, {"module.py": sha})
    assert archived.read_bytes() == before == code.read_bytes()
    code.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="source"):
        run.archive_sources(output, {"module.py": sha})


def test_published_decision_excludes_private_caches_and_retains_public_checks():
    import copy
    decision = {"selection": {}, "candidate_table": [{"candidate_id": "MPC", "trials": [],
                "audit": {"movement_passed": True, "_transport_work": [1, 2], "_data_digest": "private"}}]}
    before = copy.deepcopy(decision)
    public = run.public_decision(decision)
    assert public["candidate_table"][0]["audit"] == {"movement_passed": True}
    assert decision == before


def test_published_trials_retain_wait_budget_and_future_exposure_evidence():
    summary = {"max_load": 150, "exceptional_time_share": 0.005, "exceptional_car_seconds": 50}
    trial = {"scenario": "S1", "policy_exposure": {"admissible": True, "global": summary, "windows": []},
             "headways": {"passed": True, "minimum_seconds": 150, "maximum_seconds": 160, "violations": []},
             "mature_cohort": {"complete": True, "arrived": 1000, "unboarded": 0,
                               "mean_wait_seconds": 80, "p95_wait_seconds": 150},
             "initial_obligations": {"complete": True, "unboarded": 0}}
    saved = run.public_decision({"candidate_table": [{"candidate_id": "KEEP", "trials": [trial]}]})
    evidence = saved["candidate_table"][0]["trials"][0]
    assert evidence["future_cohort"]["mean_wait_seconds"] == 80
    assert evidence["future_cohort"]["p95_wait_seconds"] == 150
    assert evidence["future_policy_summary"]["global"]["exceptional_car_seconds"] == 50


def test_pilot_and_full_runs_have_separate_namespaces():
    assert run.arm_folder("out", "MPC_ML", None).name == "MPC_ML"
    assert run.arm_folder("out", "MPC_ML", 2).name == "MPC_ML_pilot_2"
    with pytest.raises(ValueError, match="pilot"):
        run.arm_folder("out", "MPC_ML", 0)


def test_unchecked_selected_plan_cannot_be_applied():
    old, changed = {"identity": "old"}, {"identity": "changed"}
    candidates = [{"candidate_id": "KEEP", "_data": old}, {"candidate_id": "NEW", "_data": changed}]
    table = [{"candidate_id": "NEW", "status": "evaluation_timeout", "passengers_checked": False}]
    data, applied = run.selected_plan({"selected_candidate_id": "NEW"}, candidates, table)
    assert data is old
    assert applied == "KEEP"


def test_bounded_decision_discards_late_feasible_work_and_cancels_worker():
    from threading import Event
    import time
    cancelled = Event()

    def late(deadline, cancel_event):
        time.sleep(0.04)
        return {"selection": {"selected_candidate_id": "NEW", "mitigation_candidate_id": "RELIEF"}}

    receipt = run.bounded_decision(late, 0.01, cancel_event=cancelled)
    assert receipt["completed"] is False
    assert receipt["result"] is None
    assert receipt["selection"]["selected_candidate_id"] is None
    assert receipt["selection"]["mitigation_candidate_id"] is None
    assert receipt["selection"]["status"] == "evaluation_timeout"
    assert cancelled.is_set()


def test_bounded_decision_propagates_failure_without_applying_a_plan():
    def fail(deadline, cancel_event):
        raise RuntimeError("invalid calculation")

    with pytest.raises(RuntimeError, match="invalid calculation"):
        run.bounded_decision(fail, 1)


def test_bounded_decision_returns_timely_work_with_explicit_scope():
    receipt = run.bounded_decision(lambda deadline, cancel_event: {"answer": 42}, 1)
    assert receipt["completed"] is True
    assert receipt["result"] == {"answer": 42}
    assert receipt["elapsed_seconds"] < 1


def test_closed_mileage_can_reuse_a_previously_validated_movement_ledger(monkeypatch):
    roster = [{"train_id": f"T{i}", "type": "Baltiets" if i < 37 else "Yubileyny"} for i in range(68)]
    work = [{"train_id": r["train_id"], "passenger_train_km": "4", "nonpassenger_train_km": "1"} for r in roster]

    def forbidden(*args):
        raise AssertionError("do not rerun an identical closed movement simulation")

    monkeypatch.setattr(run, "simulate", forbidden)
    rows = run.closed_per_train({}, roster, transport_work=work)
    assert len(rows) == 68
    assert all(row["km"] == 5 for row in rows)


def test_checked_mitigation_remains_explicit_research_action():
    old, changed = {"identity": "old"}, {"identity": "changed"}
    candidates = [{"candidate_id": "KEEP", "_data": old}, {"candidate_id": "RELIEF", "_data": changed}]
    table = [{"candidate_id": "RELIEF", "status": "evaluated", "passengers_checked": True,
              "physical_passed": True, "trials": [{"headways": {"passed": True}}]}]
    data, applied = run.selected_plan({"selected_candidate_id": None, "mitigation_candidate_id": "RELIEF"},
                                     candidates, table)
    assert data is changed
    assert applied == "RELIEF"


def test_closed_mileage_keeps_all_inactive_trains_and_their_types(monkeypatch):
    roster = [{"train_id": f"T{i:02}", "type": "Baltiets" if i < 37 else "Yubileyny"} for i in range(68)]
    motion = {"transport_work": [{"train_id": r["train_id"], "passenger_train_km": "0",
                                 "nonpassenger_train_km": "0"} for r in roster], "obligations_closed": True}
    motion["transport_work"][0].update(passenger_train_km="10", nonpassenger_train_km="2")
    monkeypatch.setattr(run, "simulate", lambda *args: motion)
    rows = run.closed_per_train({"plans": [{"plan_id": "P"}]}, roster)
    assert len(rows) == 68
    assert rows[0] == {"train_id": "T00", "train_type": "Baltiets", "km": 12.0}
    assert rows[-1]["km"] == 0


def test_cp_profile_maps_directions_to_ordered_terminal_ids():
    data = {"stations": [{"station_id": "A", "kind": "line"}, {"station_id": "B", "kind": "line"}]}
    result = {"slots": [{"start_second": 3600, "end_second": 4500,
                          "up_headway_seconds": 120, "down_headway_seconds": 140}]}
    profile = run.directional_profile(data, result)
    assert profile == [{"terminal": "A", "start_second": 3600, "end_second": 4500, "headway_seconds": 120},
                       {"terminal": "B", "start_second": 3600, "end_second": 4500, "headway_seconds": 140}]


def test_slot_minimum_excludes_the_next_quarter_boundary(monkeypatch):
    monkeypatch.setattr(run, "forecast_section_rates", lambda *args, **kwargs: [
        {"start_second": 13500, "end_second": 14400, "up_entries": 0, "down_entries": 0}])
    monkeypatch.setattr(run, "station_offsets", lambda *args: {})
    monkeypatch.setattr(run, "minimum_headway", lambda at, policy: 113 if 8 <= at.hour < 10 else 150)
    data = {"period_start": "2026-09-30T06:00:00+03:00", "headway_policy": {},
            "stations": [{"station_id": "A", "kind": "line"}, {"station_id": "B", "kind": "line"}]}
    assert run._slots(data, {}, "2026-09-30T09:45:00+03:00", {})[0]["minimum_headway_seconds"] == 113


def test_small_standard_catalog_includes_same_headway_wear_assignment(monkeypatch):
    data = {"period_start": "2026-09-30T06:00:00+03:00", "headway_policy": {},
            "stations": [{"station_id": "A", "kind": "line"}, {"station_id": "B", "kind": "line"}]}
    monkeypatch.setattr(run, "_slots", lambda *args: [{"start_second": 3600, "end_second": 4500,
        "minimum_headway_seconds": 150, "up_rate": 1.0, "down_rate": 1.0}])
    calls = []

    def planner(data, *args, **kwargs):
        calls.append(kwargs)
        return {**data, "proposal": len(calls)}, {"movement_passed": True}

    monkeypatch.setattr(run, "rebuild_timetable", planner)
    candidates, _ = run.make_candidates(data, 3600, 65400, {}, "STANDARD", {})
    wear = next(row for row in candidates if row["candidate_id"] == "DENSE_WEAR")
    assert calls[0]["headway"] == calls[-1]["headway"]
    assert calls[-1]["wear_priority"] is True
    assert wear["action"]["wear_priority"] is True
    assert len(candidates) <= 4


def keep_fixture():
    data = {"period_start": "2026-09-30T06:00:00+03:00", "headway_policy": {},
            "stations": [{"station_id": "A", "kind": "line"}, {"station_id": "B", "kind": "line"}],
            "tracks": [{"track_id": "UP", "from_station": "A", "to_station": "B"},
                       {"track_id": "DOWN", "from_station": "B", "to_station": "A"}],
            "plans": [{"plan_id": "P", "trips": []}], "leg_timing": []}
    for index, second in enumerate(range(3630, 6000, 180)):
        for track in ("UP", "DOWN"):
            trip = f"{track}_{index}"
            data["plans"][0]["trips"].append({"trip_id": trip, "passenger": True, "route": [track]})
            data["leg_timing"].append({"trip_id": trip, "leg_index": 0, "departure_second": second})
    bundle = {"rows": [{"station_id": station, "interval_start": f"2026-09-30T07:{minute}:00+03:00",
                         "interval_end": f"2026-09-30T07:{end}:00+03:00", "predicted_count": count}
                        for minute, end, count in (("00", "15", 150), ("15", "30", 0))
                        for station in ("A", "B")]}
    return data, bundle


def test_surrogate_uses_keep_station_clocks_platform_delay_and_same_future_cohort():
    data, bundle = keep_fixture()
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["mean_wait_budget_seconds"] == 90
    assert result["passenger_weight"] == 300
    assert result["unknown_passenger_weight"] == 0
    assert result["platform_delay_seconds"] == 120
    assert result["capacity_checked"] is False
    assert result["mature_end_second"] == 4800


def test_budgeted_cp_generates_feasible_wait_point_instead_of_only_sluggish_capacity_plan(monkeypatch):
    import copy
    data, bundle = keep_fixture()
    before = copy.deepcopy(data)
    slots = [{"start_second": 3600, "end_second": 4500, "minimum_headway_seconds": 150,
              "up_rate": 0.5, "down_rate": 0.5, "up_entries": 100, "down_entries": 100, "ready_capacity": 53}]
    unbounded = run.solve_joint_profile(slots)
    assert unbounded["estimated_mean_wait_seconds"] == 240
    monkeypatch.setattr(run, "_slots", lambda *args: slots)
    proposals = []

    def planner(data, *args, **kwargs):
        proposals.append(kwargs)
        return {**data, "proposal": len(proposals)}, {"movement_passed": True}

    monkeypatch.setattr(run, "rebuild_timetable", planner)
    candidates, search = run.make_candidates(data, 3600, 65400, bundle, "MPC_ML", {})
    cp = search["cp_search"]
    assert cp["estimated_mean_wait_seconds"] <= 90
    assert cp["slots"][0]["up_headway_seconds"] == cp["slots"][0]["down_headway_seconds"] == 180
    assert {row["candidate_id"] for row in candidates} == {"KEEP", "MPC_BUDGETED", "DENSE", "MPC_BUDGETED_WEAR"}
    assert data == before


def test_missing_keep_departures_is_an_explicit_surrogate_assumption():
    data, bundle = keep_fixture()
    data["leg_timing"] = []
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["mean_wait_budget_seconds"] == 90
    assert result["unknown_passenger_weight"] == 300
    assert result["fallback_assumption_used"] is True


def phase_fixture(up_headway=150, down_headway=150, phase=0):
    data, bundle = keep_fixture()
    data["plans"][0]["trips"] = []
    data["leg_timing"] = []
    for track, headway in (("UP", up_headway), ("DOWN", down_headway)):
        for index, second in enumerate(range(3600 + phase, 6000, headway)):
            trip = f"{track}_{index}"
            data["plans"][0]["trips"].append({"trip_id": trip, "passenger": True, "route": [track]})
            data["leg_timing"].append({"trip_id": trip, "leg_index": 0, "departure_second": second})
    for row in bundle["rows"]:
        if row["interval_start"].endswith("07:00:00+03:00"):
            row["predicted_count"] = 75
    return data, bundle


def test_search_calibrates_half_headway_to_known_keep_phase_without_changing_entries():
    import copy
    data, bundle = phase_fixture()
    before = copy.deepcopy((data, bundle))
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["freeflow_keep_mean_seconds"] == 60
    assert result["known_headway_half_mean_seconds"] == 75
    assert result["surrogate_phase_correction_seconds"] == 15
    assert result["mean_wait_budget_seconds"] == 75
    assert result["headway_passenger_weight"] == result["passenger_weight"] == 150
    assert result["fallback_assumption_used"] is False
    assert (data, bundle) == before


def test_half_headway_calibration_is_weighted_by_passengers_not_directions():
    data, bundle = phase_fixture(down_headway=300)
    next(r for r in bundle["rows"] if r["station_id"] == "B"
         and r["interval_start"].endswith("07:00:00+03:00"))["predicted_count"] = 225
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["passenger_weight"] == 300
    assert result["freeflow_keep_mean_seconds"] == 105
    assert result["known_headway_half_mean_seconds"] == 131.25
    assert result["mean_wait_budget_seconds"] == 131.25


def test_surrogate_correction_never_shrinks_the_keep_freeflow_budget():
    data, bundle = phase_fixture(phase=29)
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["freeflow_keep_mean_seconds"] == 89
    assert result["known_headway_half_mean_seconds"] == 75
    assert result["surrogate_phase_correction_seconds"] == 0
    assert result["mean_wait_budget_seconds"] == 89


def test_missing_previous_departure_uses_labelled_fallback_not_an_invented_gap():
    data, bundle = phase_fixture()
    data["leg_timing"] = [r for r in data["leg_timing"] if r["departure_second"] > 3600]
    result = run.estimated_keep_wait_budget(data, bundle, 3600)
    assert result["unknown_passenger_weight"] == 0
    assert result["unknown_headway_passenger_weight"] == 10
    assert result["headway_passenger_weight"] == 140
    assert result["mean_wait_budget_seconds"] == 90
    assert result["fallback_assumption_used"] is True
    assert result["fallback_reasons"] == ["previous_departure_missing"]
    assert result["surrogate_phase_correction_seconds"] is None


def test_calibrated_search_budget_is_not_the_final_passenger_wait_budget(monkeypatch):
    from threading import Event
    import copy
    import time
    from test_metro44_selection import candidate

    data, bundle = phase_fixture()
    data["identity"] = "KEEP"
    changed = {**data, "identity": "NEW"}
    rows = [candidate("KEEP", mean=65.7, p95=135, km=10),
            candidate("NEW", mean=70, p95=134, km=5)]
    monkeypatch.setattr(run, "read_forecast", lambda *args: bundle)
    monkeypatch.setattr(run, "decide_operational", lambda *args, **kwargs: {
        "candidate_table": copy.deepcopy(rows), "search_complete": True})
    monkeypatch.setattr(run, "closed_per_train", lambda data, *args, **kwargs: next(
        row["committed_per_train"] for row in rows if row["candidate_id"] == data["identity"]))

    def proposals(data, second, planning_end, bundle, strategy, state, **kwargs):
        surrogate = run.estimated_keep_wait_budget(data, bundle, second)
        assert surrogate["mean_wait_budget_seconds"] == 75
        return [{"candidate_id": "KEEP", "_data": data}, {"candidate_id": "NEW", "_data": changed}], {
            "generation_seconds": 0, "cp_waiting_surrogate": surrogate}

    monkeypatch.setattr(run, "make_candidates", proposals)
    result = run._calculate(data, object(), {}, {}, [], None, "MPC_ML", ["A", "B"],
                            3600, 6000, 3, time.monotonic() + 10, Event())
    assert result["decision"]["selection"]["waiting_budgets"] == {"mean_seconds": 65.7, "p95_seconds": 135}
    assert result["decision"]["selection"]["selected_candidate_id"] == "KEEP"


def test_cp_waiting_cohort_uses_unshifted_counts_and_excludes_last_ten_minutes(monkeypatch):
    import copy
    data, bundle = phase_fixture()
    data["stations"][0]["stop_seconds"] = data["stations"][1]["stop_seconds"] = 0
    calls = []
    rows = [dict(start_second=3600, end_second=4500, up_rate=1, down_rate=2,
                 up_entries=400, down_entries=600),
            dict(start_second=4500, end_second=5400, up_rate=3, down_rate=4,
                 up_entries=700, down_entries=900)]

    def rates(bundle, ids, at, **kwargs):
        calls.append(kwargs)
        answer = copy.deepcopy(rows)
        if kwargs.get("station_offsets_seconds") is None:
            for row in answer:
                row.update(up_entries=150, down_entries=300)
        return answer

    monkeypatch.setattr(run, "forecast_section_rates", rates)
    monkeypatch.setattr(run, "station_offsets", lambda *args: {"up": [0, 140], "down": [140, 0]})
    result = run._slots(data, bundle, "2026-09-30T07:00:00+03:00", {"queues": {}})
    assert len(calls) == 2
    assert calls[0]["station_offsets_seconds"] is not None
    assert calls[1].get("station_offsets_seconds") is None
    assert [(r["up_rate"], r["down_rate"], r["up_entries"], r["down_entries"]) for r in result] == [
        (1, 2, 400, 600), (3, 4, 700, 900)]
    assert [(r["up_wait_entries"], r["down_wait_entries"]) for r in result] == [(150, 300), (50, 100)]
    assert [r["waiting_cohort_seconds"] for r in result] == [900, 300]
    assert all(r["waiting_cohort_end_second"] == 4800 for r in result)
    assert all(r["waiting_platform_delay_seconds"] == 120 for r in result)
    assert all(r["waiting_terminal_clock_mapping_exact"] is False for r in result)
    assert rows[1]["up_entries"] == 700


def test_wait_budgets_require_a_valid_same_forecast_keep_reference():
    keep = {"candidate_id": "KEEP", "status": "evaluated", "passengers_checked": True,
            "physical_passed": True, "service_passed": True,
            "trials": [{"policy_exposure": {"admissible": True, "global": {}, "windows": []},
                        "mature_cohort": {"complete": True, "mean_wait_seconds": 80, "p95_wait_seconds": 150},
                        "initial_obligations": {"complete": True}, "headways": {"passed": True}}]}
    assert run.keep_wait_budgets([keep]) == (80, 150)
    keep["trials"][0]["policy_exposure"]["admissible"] = False
    assert run.keep_wait_budgets([keep]) == (None, None)


def test_interrupted_controller_resumes_without_reset_or_rewriting_executed_prefix(tmp_path, monkeypatch):
    import copy
    data = {"period_start": "2026-09-30T06:00:00+03:00", "headway_policy": {},
            "stations": [{"station_id": "A", "kind": "line"}, {"station_id": "B", "kind": "line"}],
            "plans": [{"plan_id": "P"}], "version": 0}
    roster = [{"train_id": f"T{i}", "type": "Baltiets" if i < 37 else "Yubileyny"} for i in range(68)]
    shared = dict(data=data, checkpoint={"second": 3600, "history": []}, groups=[], profiles={}, roster=roster,
                  meta={"service_end_second": 64800}, groups_digest="groups", checkpoint_digest="warmup")
    monkeypatch.setattr(run, "prepare_shared", lambda *args: (shared, {"input_hashes": {}}))
    monkeypatch.setattr(run, "_hash", lambda *args: "unchanged")
    monkeypatch.setattr(run, "archive_sources", lambda *args: None)
    monkeypatch.setattr(run, "read_forecast", lambda *args: {})

    class Runner:
        def __init__(self, data, *args):
            self.data = data

        def run(self, *, start_second, end_second, checkpoint):
            assert start_second == checkpoint["second"]
            state = copy.deepcopy(checkpoint)
            state["history"].append([start_second, end_second])
            state["second"] = end_second
            return {"checkpoint": state}

        def branch_checkpoint(self, state, previous_runner):
            return copy.deepcopy(state)

    monkeypatch.setattr(run, "GroupedPassengerSimulation", Runner)
    fail_once = {"enabled": True}

    def candidates(data, second, *args, **kwargs):
        if second == 4500 and fail_once["enabled"]:
            raise RuntimeError("interruption")
        changed = copy.deepcopy(data)
        changed["version"] += 1
        return [{"candidate_id": "KEEP", "_data": data}, {"candidate_id": "NEW", "_data": changed}], {}

    monkeypatch.setattr(run, "make_candidates", candidates)

    def decide(*args, candidates, **kwargs):
        rows = []
        for candidate in candidates:
            summary = {"max_load": 160, "exceptional_time_share": 0.005, "exceptional_car_seconds": 50,
                       "hard_limit_car_seconds": 0, "passenger_car_seconds": 10000}
            trial = {"scenario": "S1", "policy_exposure": {"admissible": True, "global": summary, "windows": []},
                     "headways": {"passed": True, "minimum_seconds": 150, "maximum_seconds": 160, "violations": []},
                     "mature_cohort": {"complete": True, "unboarded": 0, "mean_wait_seconds": 80, "p95_wait_seconds": 150},
                     "initial_obligations": {"complete": True}}
            rows.append(dict(candidate_id=candidate["candidate_id"], status="evaluated", passengers_checked=True,
                             physical_passed=True, service_passed=True, trials=[trial],
                             metrics={"waiting_passenger_seconds": 80000}))
        return {"candidate_table": rows, "selection": {}, "search_complete": True}

    monkeypatch.setattr(run, "decide_operational", decide)
    monkeypatch.setattr(run, "closed_per_train", lambda data, roster: [
        dict(train_id=r["train_id"], train_type=r["type"], km=10 - data["version"]) for r in roster])
    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="interruption"):
        run.run_strategy(output, "STANDARD", source=tmp_path / "input", pilot_steps=2)
    folder = run.arm_folder(output, "STANDARD", 2)
    checkpoint = folder / "resume_03600.json.gz"
    preserved = checkpoint.read_bytes()
    (folder / "decision_03600.json").unlink()
    (folder / "plan_03600.json.gz").unlink()
    fail_once["enabled"] = False
    result = run.run_strategy(output, "STANDARD", source=tmp_path / "input", pilot_steps=2, resume=True)
    assert checkpoint.read_bytes() == preserved
    assert (folder / "decision_03600.json").exists()
    assert (folder / "plan_03600.json.gz").exists()
    assert len(result["decisions"]) == 2
    state = run.load_resume(folder, result["contract_identity"])["state"]
    assert state == {"second": 4500, "history": [[3600, 4500]]}
    assert result["checkpoint_second"] == 4500
