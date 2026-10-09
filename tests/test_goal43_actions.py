import copy
import time

import pytest

from decision_engine import build_regular_plan
from dwell import compile_operational_timing
from goal43_actions import (attach_operational_depots, expand_operational_action,
                            generate_strategy_candidates, regulate_plan)
from goal43_policy import build_train_registry, load_policy
from passenger_groups import GroupedPassengerSimulation
from simulation import simulate


def fixture(count=4, end_second=3600):
    policy = load_policy()
    registry = build_train_registry(policy, initial_on_line={"Baltiets": count//2,
                                                            "Yubileyny": count-count//2})
    config = dict(station_ids=["A", "B", "C"], date="2026-09-01", timezone="Europe/Moscow",
                  length_m=4000, travel_seconds=200, stop_seconds=20, turnaround_seconds=180,
                  train_type="Yubileyny", car_types=["fixture"]*8)
    data, _ = build_regular_plan(config, count, end_second=end_second)
    active = [row for row in registry if row["state"] == "on_line"]
    rename = {row["train_id"]: active[i]["train_id"] for i, row in enumerate(data["trains"])}
    for row in data["trains"]:
        old = row["train_id"]
        row["train_id"] = rename[old]
        row["train_type"] = next(train["type"] for train in active if train["train_id"] == rename[old])
    for trip in data["plans"][0]["trips"]:
        old = trip["train_id"]
        trip["train_id"] = rename[old]
        trip["trip_id"] = trip["trip_id"].replace(old, rename[old], 1)
    attached, ready = attach_operational_depots(data, registry, policy)
    tracks = {row["track_id"]: row for row in attached["tracks"]}
    stations = {row["station_id"]: row for row in attached["stations"]}
    plan = attached["plans"][0]
    for train in active:
        last = max((row for row in plan["trips"] if row["train_id"] == train["train_id"]),
                   key=lambda row: row["departure_second"])
        second = last["departure_second"]
        for track_id in last["route"]:
            track = tracks[track_id]; station = stations[track["to_station"]]
            second += track["travel_seconds"]+station["stop_seconds"]+station["turnaround_seconds"]
        terminal = tracks[last["route"][-1]]["to_station"]
        exit_track = next(row for row in tracks.values() if row["from_station"] == terminal
                          and row["track_id"].startswith("EXIT_"))
        plan["trips"].append(dict(trip_id=train["train_id"]+":DAYEXIT", train_id=train["train_id"],
                                  departure_second=second, route=[exit_track["track_id"]], passenger=False))
    compiled, _ = compile_operational_timing(attached, attached["plans"][0]["plan_id"], policy=policy)
    return compiled, ready, policy, registry


def test_attaching_depots_preserves_all_fleet_ids_and_unknown_reserves():
    data, ready, policy, registry = fixture()
    assert len(data["trains"]) == 68
    assert {train["train_id"] for train in data["trains"]} == {train["train_id"] for train in registry}
    assert {key for key, value in ready.items() if value["ready"]} == {"B030", "Y025"}
    assert not ready["B037"]["ready"] and ready["B037"]["readiness_status"] == "unknown"
    assert ready["B030"]["preparation_seconds"] == 720-120-180
    assert ready["Y025"]["preparation_seconds"] == 780-120-180
    assert all(row["turnaround_seconds"] == 180 for row in data["stations"]
               if row["station_id"] in {"A", "C"})
    assert simulate(data, data["plans"][0]["plan_id"])["checks"]["peak_trains_on_line"] == 4


@pytest.mark.parametrize("train,release", [("B030",720),("Y025",780)])
def test_add_release_is_inclusive_and_keeps_initial_fleet_unchanged(train, release):
    data, ready, policy, _ = fixture()
    before = copy.deepcopy(data)
    changed, action = expand_operational_action(data, "ADD", train, 30, ready, policy=policy)
    assert data == before
    assert changed["trains"] == data["trains"]
    assert action["planned_effect_second"] == 30+release
    assert action["actual_effect_second"] >= action["planned_effect_second"]
    assert action["actual_effect_by_station"]
    assert action["release_authorized"] is False


def test_not_ready_and_full_line_add_are_rejected():
    data, ready, policy, _ = fixture()
    with pytest.raises(ValueError, match="ready"):
        expand_operational_action(data, "ADD", "B037", 30, ready, policy=policy)
    data["max_trains_on_line"] = 4
    with pytest.raises(ValueError, match="capacity"):
        expand_operational_action(data, "ADD", "B030", 30, ready, policy=policy)


def test_reserve_cannot_be_used_before_explicit_ready_time():
    data, ready, policy, _ = fixture()
    ready["B030"]["ready_at_second"] = 1000
    with pytest.raises(ValueError, match="ready"):
        expand_operational_action(data, "ADD", "B030", 30, ready, policy=policy)


def test_withdrawal_finishes_passenger_trip_then_returns_and_closes_its_tail():
    data, ready, policy, _ = fixture()
    changed, action = expand_operational_action(data, "WITHDRAW", "B001", 30, ready, policy=policy)
    motion = simulate(changed, changed["plans"][0]["plan_id"])
    final = next(row for row in motion["final_state"] if row["train_id"] == "B001")
    assert final["state"] == "standby" and final["location"].startswith("DEPOT_")
    assert action["withdrawal_return_closed"]
    assert not any(row["train_id"] == "B001" for row in motion["unfinished_obligations"])
    assert action["actual_effect_second"] > 30
    assert action["passengers_checked"] is False


def test_deferred_withdrawal_restores_service_without_changing_past():
    data, ready, policy, _ = fixture()
    withdrawn, _ = expand_operational_action(data, "WITHDRAW", "B001", 30, ready, policy=policy)
    restored, action = expand_operational_action(withdrawn, "DEFER_WITHDRAWAL", "B001", 40,
                                               ready, policy=policy)
    assert action["kind"] == "DEFER_WITHDRAWAL"
    original = simulate(data, data["plans"][0]["plan_id"])
    final = simulate(restored, restored["plans"][0]["plan_id"])
    assert [row for row in original["events"] if row["second"] < 40] == [row for row in final["events"] if row["second"] < 40]
    assert not any(trip["train_id"] == "B001" and trip["trip_id"] == "B001:EXIT30"
                   for trip in restored["plans"][0]["trips"])


def test_regulation_preserves_past_routes_and_actual_departure_minima():
    data, _, policy, _ = fixture()
    before = copy.deepcopy(data)
    changed, audit = regulate_plan(data, 700, policy, target_headway_seconds=330)
    assert data == before and audit["status"] in {"movement_checked", "search_incomplete"}
    assert changed["plans"] == data["plans"] and changed["trains"] == data["trains"]
    first = simulate(data, data["plans"][0]["plan_id"])
    second = simulate(changed, changed["plans"][0]["plan_id"])
    assert [row for row in first["events"] if row["second"] < 700] == [row for row in second["events"] if row["second"] < 700]
    assert audit["passengers_checked"] is False
    assert audit["holding_location_status"] == "assumed terminal control only"


def test_candidate_families_share_initial_fleet_and_limit_withdrawal_count():
    data, ready, policy, _ = fixture()
    candidates, audit = generate_strategy_candidates(data, 700, policy, ready,
                                                     minimum_remaining=2, max_withdrawals=2,
                                                     top_k=2, include_recovery=False)
    assert {row["strategy"] for row in candidates} == {"KEEP", "REGULATE_ONLY", "WITHDRAW_ONLY", "COMBINED"}
    assert audit["minimum_remaining"] == 2 and audit["proxy_status"] == "assumed aggregate service floor"
    checked = [row for row in candidates if row["status"] == "movement_checked"]
    assert all(row["_data"]["trains"] == data["trains"] for row in checked)
    assert all(len(row["action"].get("withdrawn_train_ids", [])) <= 2 for row in candidates)


def test_incident_is_retained_when_withdrawal_plan_is_recompiled():
    data, ready, policy, _ = fixture()
    incident = dict(incident_id="I", train_id="B001", station_id="B", direction="up",
                    detected_second=200, expected_end_second=500)
    changed, action = expand_operational_action(data, "WITHDRAW", "B001", 200, ready,
                                               policy=policy, incidents=[incident])
    motion = simulate(changed, changed["plans"][0]["plan_id"])
    departure = next(row for row in motion["events"] if row["train_id"] == "B001"
                     and row["event_type"] == "departure" and row["location"] == "P02_up")
    assert departure["second"] >= 500 and action["withdrawal_return_closed"]


def test_add_branches_actual_passengers_on_same_68_train_snapshot():
    data, ready, policy, _ = fixture()
    profiles = {train["train_id"]: [dict(car_id=f"C{i}", car_type="demo", max_capacity=160,
                    normal_threshold=120, source="fixture") for i in range(8)] for train in data["trains"]}
    groups = [dict(group_id="G", turnstile_second=0, platform_second=0, origin="A", count=16,
                   destinations=[dict(station_id="C", count=16)])]
    source = GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], groups, profiles)
    checkpoint = source.run(end_second=30)["checkpoint"]
    changed, _ = expand_operational_action(data, "ADD", "B030", 30, ready, policy=policy)
    branch = GroupedPassengerSimulation(changed, changed["plans"][0]["plan_id"], groups, profiles)
    restored = branch.branch_checkpoint(checkpoint, source)
    assert restored["onboard"] == checkpoint["onboard"] and restored["queues"] == checkpoint["queues"]
    full = branch.run()["checkpoint"]
    continuation = branch.run(start_second=30, checkpoint=restored)["checkpoint"]
    assert continuation == full


def test_future_added_train_does_not_inflate_current_withdrawal_capacity():
    data, ready, policy, _ = fixture()
    added, _ = expand_operational_action(data, "ADD", "B030", 30, ready, policy=policy)
    candidates, audit = generate_strategy_candidates(added, 40, policy, ready, minimum_remaining=4,
                                                     top_k=1, include_recovery=False)
    assert audit["withdrawal_search"][0]["selected_train_ids"] == []
    assert not any(row["strategy"] == "WITHDRAW_ONLY" and row["status"] == "movement_checked"
                   for row in candidates)


def test_later_depot_return_is_not_an_immediate_defer_recovery_option():
    data, ready, policy, _ = fixture()
    withdrawn, _ = expand_operational_action(data, "WITHDRAW", "B001", 30, ready, policy=policy)
    candidates, _ = generate_strategy_candidates(withdrawn, 40, policy, ready, minimum_remaining=4,
                                                 recovery_end_second=100, include_recovery=True)
    assert not any(row["strategy"] == "DEFER_WITHDRAWAL" for row in candidates)


def test_expired_generation_budget_does_not_run_new_movement(monkeypatch):
    data, ready, policy, _ = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("expired budget must not begin new movement compilation")
    monkeypatch.setattr("goal43_actions.compile_operational_timing", forbidden)
    deadline = time.monotonic()-1
    changed, audit = regulate_plan(data, 30, policy, deadline=deadline)
    assert audit["status"] == "evaluation_timeout" and not audit["movement_checked"]
    candidates, _ = generate_strategy_candidates(data, 30, policy, ready, minimum_remaining=2,
                                                 deadline=deadline)
    assert len(candidates) == 4
    assert all(row["status"] == "evaluation_timeout" for row in candidates)


def test_solver_budget_starts_after_preparation_and_excludes_verification(monkeypatch):
    import goal43_actions as actions
    data, _, policy, _ = fixture()
    clock, calls = [100.0], []
    compile_actual, solve_actual = actions.compile_operational_timing, actions.solve_departures
    def prepare(*args, **kwargs):
        result = compile_actual(*args, **kwargs)
        clock[0] += 12
        return result
    def solve(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["deadline"] is None
        return solve_actual(*args, **kwargs)
    monkeypatch.setattr(actions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(actions, "compile_operational_timing", prepare)
    monkeypatch.setattr(actions, "solve_departures", solve)
    _, audit = regulate_plan(data, 700, policy, time_limit_seconds=5)
    assert calls[0]["time_limit_seconds"] == 5
    assert audit["movement_checked"] and audit["elapsed_seconds"] == 24


def test_shared_deadline_expiring_during_preparation_skips_solver(monkeypatch):
    import goal43_actions as actions
    data, _, policy, _ = fixture()
    clock = [100.0]
    compile_actual = actions.compile_operational_timing
    def prepare(*args, **kwargs):
        result = compile_actual(*args, **kwargs)
        clock[0] = 106
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("expired shared deadline must skip solving")
    monkeypatch.setattr(actions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(actions, "compile_operational_timing", prepare)
    monkeypatch.setattr(actions, "solve_departures", forbidden)
    _, audit = regulate_plan(data, 700, policy, time_limit_seconds=5, deadline=105)
    assert audit["status"] == "evaluation_timeout" and not audit["movement_checked"]


def test_shared_deadline_expiring_during_verification_discards_changed_plan(monkeypatch):
    import goal43_actions as actions
    data, _, policy, _ = fixture()
    clock = [100.0]
    compile_actual, solve_actual = actions.compile_operational_timing, actions.solve_departures
    def prepare(*args, **kwargs):
        result = compile_actual(*args, **kwargs)
        clock[0] += 4
        return result
    def solve(*args, **kwargs):
        assert kwargs["deadline"] == 107
        return solve_actual(*args, **dict(kwargs, deadline=None))
    monkeypatch.setattr(actions, "monotonic", lambda: clock[0])
    monkeypatch.setattr(actions, "compile_operational_timing", prepare)
    monkeypatch.setattr(actions, "solve_departures", solve)
    _, audit = regulate_plan(data, 700, policy, time_limit_seconds=5, deadline=107)
    assert audit["status"] == "evaluation_timeout" and not audit["movement_checked"]
    assert "commanded_departures" not in audit
