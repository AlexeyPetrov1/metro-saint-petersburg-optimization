from pathlib import Path
import json
from threading import Event
import time
from types import SimpleNamespace

import pytest

from goal43_policy import load_policy
from goal43_replay import build_operational_day, mileage_context
from simulation import simulate


def test_live_pipeline_starts_notification_before_slow_search(monkeypatch):
    import time
    import goal43_replay as replay
    data = {"period_start": "2026-09-01T06:00:00+03:00"}
    observed = []
    real_update = replay.update_incident
    def update(*args):
        observed.append("notification")
        return real_update(*args)
    def slow(*args, **kwargs):
        observed.append("search")
        time.sleep(2)
        return [], {}
    monkeypatch.setattr(replay, "update_incident", update)
    monkeypatch.setattr(replay, "hourly_median", lambda *args: {"target": "entries", "model_version": "test", "issued_at": "2026-09-01T06:00:00+03:00"})
    monkeypatch.setattr(replay, "generate_strategy_candidates", slow)
    started = time.monotonic()
    result = replay.live_decision(data, {}, {"S1": {"second": 0}}, {}, {}, None, load_policy(),
        scenario_id="slow", minimum_remaining=1, budget_seconds=.15)
    assert time.monotonic() - started < .7
    assert observed == ["notification", "search"]
    assert result["decision"]["selection"]["selected_candidate_id"] is None
    assert result["decision"]["selection"]["status"] == "evaluation_timeout"
    assert result["ledger"][0]["notification_at"] == data["period_start"]
    assert result["ledger"][-1]["kind"] == "evaluation_timeout"
    assert result["candidates"] == []


def test_live_pipeline_latency_includes_candidate_generation(monkeypatch):
    import time
    import goal43_replay as replay
    monkeypatch.setattr(replay, "hourly_median", lambda *args: {"target": "entries", "model_version": "test", "issued_at": "2026-09-01T06:00:00+03:00"})
    def search(*args, **kwargs):
        time.sleep(.07)
        return [{"candidate_id": "KEEP", "status": "movement_checked", "_data": {}}], {"search_complete": True}
    monkeypatch.setattr(replay, "generate_strategy_candidates", search)
    monkeypatch.setattr(replay, "decide_operational", lambda *args, **kwargs: {
        "selection": {"status": "conditional_resource_policy", "selected_candidate_id": "KEEP"},
        "candidate_table": [{"candidate_id": "KEEP", "status": "evaluated"}],
        "elapsed_seconds": .001, "search_complete": True})
    result = replay.live_decision({"period_start": "2026-09-01T06:00:00+03:00"}, {},
        {"S1": {"second": 0}}, {}, {}, None, load_policy(), scenario_id="normal", minimum_remaining=1)
    assert result["decision"]["elapsed_seconds"] >= .07
    assert result["ledger"][-1]["kind"] == "recommendation_ready"
    assert result["ledger"][-1]["validated"] is True


def test_operational_day_closes_service_and_depot_obligations():
    data, registry, ready, profiles, meta = build_operational_day(load_policy(), count=16,
        start_at="2026-09-01T06:00:00+03:00", service_hours=3)
    motion = simulate(data, data["plans"][0]["plan_id"])
    assert len(data["trains"]) == 68
    assert motion["obligations_closed"]
    assert all(r["location"].startswith("DEPOT_") for r in motion["final_state"])
    assert all(len(cars) == 8 and max(c["max_capacity"] for c in cars) == 160 for cars in profiles.values())
    assert meta["cycle_seconds"] == 6080
    assert ready["B030"]["inclusive_release_seconds"] == 720
    assert ready["Y025"]["inclusive_release_seconds"] == 780


def test_partial_window_never_establishes_final_saving():
    policy = load_policy()
    data, registry, _, _, meta = build_operational_day(policy, count=16,
        start_at="2026-09-01T06:00:00+03:00", service_hours=3)
    motion = simulate(data, data["plans"][0]["plan_id"])
    context = mileage_context(data, motion, registry, "demand", "none", 7, full_day=False)
    assert context["coverage_complete"] is False
    assert set(context["trains"]) == {r["train_id"] for r in registry}


def test_cli_accepts_new_profile_without_changing_legacy_default(monkeypatch, tmp_path):
    import goal43_replay
    import metro_replay
    called = []
    monkeypatch.setattr(goal43_replay, "run", lambda output, **kwargs: called.append(Path(output)) or {"status": "fixture"})
    assert metro_replay.main(["run", "--profile", "goal4_3", "--output", str(tmp_path)]) == 0
    assert called == [tmp_path]


def quick_pipeline(monkeypatch):
    import goal43_replay as replay
    monkeypatch.setattr(replay, "hourly_median", lambda *args: {"target": "turnstile_entries"})
    def search(*args, **kwargs):
        time.sleep(.02)
        return [{"candidate_id": "KEEP", "status": "movement_checked", "_data": {}}], {"search_complete": True}
    monkeypatch.setattr(replay, "generate_strategy_candidates", search)
    monkeypatch.setattr(replay, "decide_operational", lambda *args, **kwargs: dict(
        selection=dict(status="conditional_resource_policy", selected_candidate_id="KEEP"),
        candidate_table=[dict(candidate_id="KEEP", status="evaluated")]))
    return replay


@pytest.mark.parametrize("blocked_call", [1, 2])
def test_initial_and_final_persistence_cannot_block_in_memory_delivery(monkeypatch, blocked_call):
    replay = quick_pipeline(monkeypatch)
    entered, release = Event(), Event()
    calls = []
    def slow(rows):
        calls.append(rows)
        if len(calls) == blocked_call:
            entered.set()
            release.wait(2)
    sink = replay.OrderedNotificationSink(slow)
    started = time.monotonic()
    try:
        result = replay.live_decision({"period_start": "2026-09-01T06:00:00+03:00"}, {},
            {"S1": {"second": 0}}, {}, {}, None, load_policy(), scenario_id="sink", minimum_remaining=1,
            budget_seconds=.15, ledger_sink=sink)
        assert time.monotonic() - started < .3
        assert entered.wait(.5)
        assert result["ledger"][-1]["kind"] == "recommendation_ready"
        assert result["notification_delivery"]["basis"] == "in_memory"
        assert result["notification_delivery"]["elapsed_seconds"] >= .02
        assert result["notification_persistence"]["status"] == "pending"
        assert result["notification_persistence"]["persisted_elapsed_seconds"] is None
    finally:
        release.set()
        sink.flush(1)


def test_delayed_old_notification_write_cannot_finish_after_newer_write():
    from goal43_replay import OrderedNotificationSink
    entered, release = Event(), Event()
    writes = []
    def save(rows):
        if rows == ["old"]:
            entered.set()
            release.wait(2)
        writes.append(rows)
    sink = OrderedNotificationSink(save)
    sink.submit(["old"], time.monotonic())
    assert entered.wait(.5)
    sink.submit(["new"], time.monotonic())
    release.set()
    assert sink.flush(1)["status"] == "persisted"
    assert writes == [["old"], ["new"]]


def headway_fixture(seconds):
    policy = load_policy()
    data = dict(period_start="2026-09-01T06:00:00+03:00", period_end="2026-09-01T08:20:00+03:00",
        stations=[dict(station_id="A", kind="line"), dict(station_id="B", kind="line")],
        tracks=[dict(track_id="AB_up", from_station="A", to_station="B")], headway_policy=policy,
        previous_departures=[dict(track_id="AB_up", departure_second=0)], plans=[dict(plan_id="P")])
    motion = dict(duration_seconds=8400, events=[dict(event_type="departure", location="AB_up", second=s) for s in seconds])
    runner = SimpleNamespace(duration=8400, motion=motion, profiles={}, by_id={})
    return data, runner


def stub_passenger_metrics(monkeypatch):
    import goal43_replay as replay
    monkeypatch.setattr(replay, "work_between", lambda *args: dict(train_km="0"))
    monkeypatch.setattr(replay, "operational_exposure", lambda *args: dict(admissible=True))
    monkeypatch.setattr(replay, "cohort_waiting", lambda *args: {})
    return replay


def test_full_day_headways_cover_late_service_not_short_cohort(monkeypatch, tmp_path):
    replay = stub_passenger_metrics(monkeypatch)
    data, runner = headway_fixture((200, 400, 1100))
    state = dict(max_car_load=0, waiting_passenger_seconds=0, entered=0, completed=0, queued=0)
    record = replay.save_run(tmp_path, "late", data, runner, state, [], load_policy(), "fixture", "KEEP", "S1", 7,
        full_day=True, cohort_start=0, cohort_end=500)
    assert record["headways"]["passed"] is False
    assert record["service_scope"]["end_second"] == 1200
    assert record["cohort_scope"]["end_second"] == 500


def test_headways_exclude_return_tail_with_explicit_service_boundary(monkeypatch, tmp_path):
    replay = stub_passenger_metrics(monkeypatch)
    data, runner = headway_fixture((200, 400, 600, 800, 1000, 2500))
    state = dict(max_car_load=0, waiting_passenger_seconds=0, entered=0, completed=0, queued=0)
    record = replay.save_run(tmp_path, "tail", data, runner, state, [], load_policy(), "fixture", "KEEP", "S1", 7,
        full_day=True, cohort_start=0, cohort_end=500, service_end_second=1200)
    assert record["headways"]["passed"] is True
    assert record["service_scope"]["end_second"] == 1200


def test_journal_only_service_refresh_preserves_motion_and_mileage(tmp_path):
    import goal43_replay as replay
    data, runner = headway_fixture((200, 400, 1100))
    folder = tmp_path / "runs/late"
    folder.mkdir(parents=True)
    summary = dict(run_id="late", scenario_id="fixture", strategy="KEEP", full_day=True,
        metrics=dict(resources=dict(train_km="12.345"), policy_exposure=dict(admissible=True)),
        headways=dict(passed=True), cohort_scope=dict(start_second=0, end_second=500), entered=0, completed=0, queued=0)
    for name, value in (("movement_input.json", data), ("motion.json", runner.motion), ("summary.json", summary)):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "runs.json").write_text(json.dumps([dict(summary, folder="runs/late")]), encoding="utf-8")
    original = (folder / "motion.json").read_bytes()
    result = replay.refresh_service_summaries(tmp_path)
    refreshed = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert result["runs_updated"] == 1 and result["simulation_rerun"] is False
    assert (folder / "motion.json").read_bytes() == original
    assert refreshed["metrics"]["resources"]["train_km"] == "12.345"
    assert refreshed["headways"]["passed"] is False
    assert json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))[0]["service_scope"]["end_second"] == 1200
