import copy

import pandas as pd

from dwell import compile_operational_timing
from forecasting import hourly_median
from goal43_policy import load_policy
from passenger_groups import GroupedPassengerSimulation
from rolling import decide_operational, select_operational_candidate
from test_goal43_motion import line


def setup():
    policy = load_policy()
    data, _ = compile_operational_timing(line(), "P", policy=policy)
    profiles = {t["train_id"]: [dict(car_id=f"C{i}", car_type="demo", max_capacity=160,
        normal_threshold=120, source="user GOAL43") for i in range(8)] for t in data["trains"]}
    groups = [dict(group_id="G", turnstile_second=0, platform_second=0, origin="A", count=16,
                   destinations=[dict(station_id="C", count=16)])]
    source = GroupedPassengerSimulation(data, "P", groups, profiles)
    state = source.run(end_second=300)["checkpoint"]
    rows = []
    for station in "ABC":
        for hour in (8, 9):
            for q in range(4):
                start = pd.Timestamp(f"2026-08-31T{hour:02}:{q * 15:02}:00+03:00")
                rows.append(dict(station_id=station, interval_start=start,
                    interval_end=start + pd.Timedelta(minutes=15), entry_count=8,
                    is_complete=True, source_id="fixture"))
    bundle = hourly_median(pd.DataFrame(rows), list("ABC"), "2026-09-01T08:00:00+03:00")
    candidate = dict(candidate_id="KEEP", strategy="KEEP", status="movement_checked",
                     _data=data, action={"kind": "KEEP"})
    return data, {"S1": source}, {"S1": state}, profiles, bundle, policy, [candidate]


def test_forecast_only_operational_decision_preserves_snapshot():
    data, sources, states, profiles, bundle, policy, candidates = setup()
    before = copy.deepcopy(states)
    result = decide_operational(data, sources, states, profiles, {}, bundle, policy,
                                candidates=candidates, include_recovery=False)
    assert result["target"] == "turnstile_entries"
    assert result["decision_second"] == 300
    assert result["candidate_table"][0]["passengers_checked"] is True
    assert states == before
    assert result["elapsed_seconds"] < 480
    assert result["release_authorized"] is False


def test_zero_budget_returns_no_unchecked_recommendation():
    data, sources, states, profiles, bundle, policy, candidates = setup()
    result = decide_operational(data, sources, states, profiles, {}, bundle, policy,
                                candidates=candidates, budget_seconds=0)
    assert result["selection"]["status"] == "evaluation_timeout"
    assert result["selection"]["selected_candidate_id"] is None


def test_soft_comfort_and_closed_work_priority():
    rows = [dict(candidate_id=name, status="evaluated", service_passed=True,
        passengers_checked=True, policy_passed=True, physical_passed=True,
        committed_schedule_km=km, metrics={"waiting_passenger_seconds": wait,
                                         "comfort_car_seconds": 10, "max_car_load": 140})
        for name, km, wait in (("KEEP", "10", 100), ("WITHDRAW_ONLY:B1", "8", 200))]
    assert select_operational_candidate(rows)["selected_candidate_id"] == "WITHDRAW_ONLY:B1"
    rows[1]["physical_passed"] = False
    assert select_operational_candidate(rows)["selected_candidate_id"] == "KEEP"


def test_short_window_distance_does_not_win_by_postponement():
    rows = [dict(candidate_id=name, status="evaluated", service_passed=True,
        passengers_checked=True, policy_passed=True, physical_passed=True,
        committed_schedule_km="10", resources={"train_km": observed},
        metrics={"waiting_passenger_seconds": wait,"comfort_car_seconds": 0,"max_car_load": 30})
        for name, observed, wait in (("KEEP", "10", 100), ("REGULATE_ONLY", "5", 300))]
    assert select_operational_candidate(rows)["selected_candidate_id"] == "KEEP"


def test_future_actual_does_not_change_decision():
    data, sources, states, profiles, bundle, policy, candidates = setup()
    first = decide_operational(data, sources, states, profiles, {}, bundle, policy, candidates=candidates)
    changed = GroupedPassengerSimulation(data, "P", list(sources["S1"].by_id.values()) + [
        dict(group_id="FUTURE", turnstile_second=900, platform_second=900, origin="A", count=1000,
             destinations=[dict(station_id="C", count=1000)])], profiles)
    changed_state = changed.run(end_second=300)["checkpoint"]
    second = decide_operational(data, {"S1": changed}, {"S1": changed_state}, profiles, {}, bundle, policy, candidates=candidates)
    assert first["selection"] == second["selection"]
    assert first["candidate_table"][0]["metrics"] == second["candidate_table"][0]["metrics"]
