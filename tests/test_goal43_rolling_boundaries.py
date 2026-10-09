import copy
import threading
import time

import pandas as pd
import pytest

import rolling
from forecasting import hourly_median
from passenger_groups import GroupedPassengerSimulation
from simulation import simulate
from test_goal43_actions import fixture
from test_goal43_rolling import setup


def regular_case(entry_count=30):
    data, registry, policy, _ = fixture()
    profiles = {row["train_id"]: [dict(car_id=f"C{i}", car_type="demo", max_capacity=160,
        normal_threshold=120, source="fixture") for i in range(8)] for row in data["trains"]}
    source = GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], [], profiles)
    state = source.run(end_second=700)["checkpoint"]
    origin = pd.Timestamp(data["period_start"]).floor("h")
    rows = []
    for station in "ABC":
        for q in range(8):
            at = origin - pd.Timedelta(days=1) + pd.Timedelta(minutes=15*q)
            rows.append(dict(station_id=station, interval_start=at,
                interval_end=at+pd.Timedelta(minutes=15), entry_count=entry_count,
                is_complete=True, source_id="fixture"))
    bundle = hourly_median(pd.DataFrame(rows), list("ABC"), origin.isoformat())
    candidate = dict(candidate_id="KEEP", strategy="KEEP", status="movement_checked",
                     _data=data, action={"kind": "KEEP"})
    return data, {"S1": source}, {"S1": state}, profiles, registry, bundle, policy, [candidate]


def test_last_departure_age_detects_missing_service_without_a_next_departure():
    data, sources, _, _, _, _, _ = setup()
    out = rolling.operational_headways(data, sources["S1"].motion, 300, 3600)
    assert out["passed"] is False
    assert any(row["kind"] == "boundary_end" and row["seconds"] > 480 for row in out["violations"])


def test_missing_history_is_unknown_even_when_no_completed_gap_was_seen():
    data, _, _, _, _, _, _ = setup()
    out = rolling.operational_headways(data, {"events": []}, 300, 400)
    assert out["coverage_status"] == "unknown"
    assert out["passed"] is False
    assert not out["violations"]


def test_no_departure_for_longer_than_maximum_is_a_violation_without_history():
    data, _, _, _, _, _, _ = setup()
    out = rolling.operational_headways(data, {"events": []}, 300, 1000)
    assert out["passed"] is False
    assert out["violations"]


def test_valid_boundary_age_does_not_require_future_actual_departure():
    data, _, _, _, _, _, _ = setup()
    data["previous_departures"] = [{"track_id": t["track_id"], "departure_second": 200}
                                  for t in data["tracks"]]
    out = rolling.operational_headways(data, {"events": []}, 300, 400)
    assert out["coverage_status"] == "complete"
    assert out["passed"] is True
    assert not out["violations"]


def test_horizon_entries_are_censored_but_mature_cohort_can_pass():
    args = regular_case()
    out = rolling.decide_operational(*args[:7], candidates=args[7])
    trial = out["candidate_table"][0]["trials"][0]
    assert trial["cohort"]["unboarded"] > 0
    assert trial["mature_cohort"]["complete"] is True
    assert trial["cohort_boundary"]["allowance_seconds"] == 120+480
    assert trial["cohort_boundary"]["mature_end_second"] == 3600-600
    assert out["selection"]["selected_candidate_id"] == "KEEP"


def test_maturity_allowance_uses_configured_maximum_instead_of_literal_600():
    args = regular_case()
    args[6]["parameters"]["headway_max_s"]["value"] = 360
    out = rolling.decide_operational(*args[:7], candidates=args[7])
    boundary = out["candidate_table"][0]["trials"][0]["cohort_boundary"]
    assert boundary["allowance_seconds"] == 480
    assert boundary["mature_end_second"] == 3120


@pytest.mark.parametrize("platform_second", [290, 350])
def test_existing_queue_and_approaching_passengers_remain_service_obligations(platform_second):
    data, sources, states, profiles, bundle, policy, candidates = setup()
    group = dict(group_id="BACKLOG", turnstile_second=290, platform_second=platform_second,
                 origin="A", count=8, destinations=[dict(station_id="C", count=8)])
    source = GroupedPassengerSimulation(data, "P", [group], profiles)
    state = source.run(end_second=300)["checkpoint"]
    out = rolling.decide_operational(data, {"S1": source}, {"S1": state}, profiles, {}, bundle,
                                     policy, candidates=candidates)
    trial = out["candidate_table"][0]["trials"][0]
    assert trial["initial_obligations"]["pending_at_start"] == 8
    assert trial["initial_obligations"]["unboarded"] == 8
    assert trial["initial_obligations"]["complete"] is False
    assert out["candidate_table"][0]["service_passed"] is False


def test_incomplete_solver_search_is_not_reported_as_complete_after_validation():
    args = regular_case()
    args[7][0]["status"] = "search_incomplete"
    args[7][0]["timing"] = {"search_complete": False}
    out = rolling.decide_operational(*args[:7], candidates=args[7])
    assert out["candidate_table"][0]["passengers_checked"] is True
    assert out["search_complete"] is False
    assert out["selection"]["status"] == "search_incomplete"


def test_generation_timeout_stays_explicit_even_before_outer_deadline():
    args = regular_case()
    args[7][0] = dict(candidate_id="KEEP", strategy="KEEP", status="evaluation_timeout")
    out = rolling.decide_operational(*args[:7], candidates=args[7])
    assert out["search_complete"] is False
    assert out["selection"]["status"] == "evaluation_timeout"
    assert out["selection"]["selected_candidate_id"] is None


@pytest.mark.parametrize("stage", ["validate_bundle", "operational_headways", "operational_exposure"])
def test_public_deadline_bounds_slow_preparation_and_final_validation(monkeypatch, stage):
    args = regular_case()
    before = copy.deepcopy(args[:7])
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = getattr(rolling, stage)

    def blocked(*call_args, **kwargs):
        entered.set()
        release.wait(2)
        try:
            return original(*call_args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(rolling, stage, blocked)
    started = time.monotonic()
    try:
        out = rolling.decide_operational(*args[:7], candidates=args[7], budget_seconds=.15)
        elapsed = time.monotonic()-started
        assert entered.is_set()
        assert elapsed < .7
        assert out["selection"]["status"] == "evaluation_timeout"
        assert out["selection"]["selected_candidate_id"] is None
        assert out["search_complete"] is False
        saved = copy.deepcopy(out)
    finally:
        release.set()
        assert finished.wait(2)
    assert out == saved
    assert args[0] == before[0] and args[2] == before[2]


def test_grouped_runner_cancels_within_256_events_and_preserves_legacy_output(monkeypatch):
    data, sources, _, _, _, _, _ = setup()
    runner = sources["S1"]
    expected = runner.run()
    assert runner.run(deadline=None, cancel_event=None) == expected
    event = threading.Event()
    original = runner._process
    processed = []

    def process(state, row):
        processed.append(row)
        event.set()
        return original(state, row)

    monkeypatch.setattr(runner, "_process", process)
    with pytest.raises(TimeoutError, match="deadline|cancel"):
        runner.run(cancel_event=event)
    assert 1 <= len(processed) <= 256
    with pytest.raises(TimeoutError, match="deadline|cancel"):
        runner.run(deadline=time.monotonic()-1)


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan"), True])
def test_invalid_budget_is_rejected(budget):
    args = regular_case()
    with pytest.raises(ValueError, match="budget"):
        rolling.decide_operational(*args[:7], candidates=args[7], budget_seconds=budget)
