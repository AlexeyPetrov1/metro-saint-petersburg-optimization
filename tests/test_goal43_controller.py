import copy

from goal43_controller import observed_planning_data, control_times


def test_controller_grid_includes_immediate_incident_revision():
    assert control_times(1800, 8, revision_second=1980) == [1800, 1980, 2700, 3600, 4500, 5400, 6300, 7200, 8100]


def test_observed_merge_hides_future_actual_arrival_and_unexecuted_departure():
    planning = {"tracks": [{"track_id": "AB_up", "travel_seconds": 100}],
        "plans": [{"trips": [{"trip_id": "H", "route": ["AB_up"]}]}],
        "leg_timing": [{"train_id": "T", "trip_id": "H", "leg_index": 0,
                        "departure_second": 10, "service_seconds": 20, "holding_seconds": 0,
                        "approach_holding_seconds": 0}]}
    actual = copy.deepcopy(planning)
    actual["leg_timing"][0].update(departure_second=10, approach_holding_seconds=890)
    hidden = observed_planning_data(planning, actual, 200)
    assert hidden["leg_timing"][0]["departure_second"] == 10
    assert hidden["leg_timing"][0]["approach_holding_seconds"] == 90
    actual["leg_timing"][0]["approach_holding_seconds"] = 1890
    assert observed_planning_data(planning, actual, 200) == hidden
    actual["leg_timing"][0]["departure_second"] = 1000
    hidden = observed_planning_data(planning, actual, 200)
    assert hidden["leg_timing"][0]["departure_second"] == 200
    assert hidden["leg_timing"][0]["approach_holding_seconds"] == 0


def test_observed_merge_keeps_completed_actual_motion_exactly():
    planning = {"tracks": [{"track_id": "AB_up", "travel_seconds": 100}],
        "plans": [{"trips": [{"trip_id": "H", "route": ["AB_up"]}]}],
        "leg_timing": [{"train_id": "T", "trip_id": "H", "leg_index": 0,
                        "departure_second": 10, "service_seconds": 20, "holding_seconds": 0}]}
    actual = copy.deepcopy(planning)
    actual["leg_timing"][0].update(departure_second=20, service_seconds=25, approach_holding_seconds=15)
    hidden = observed_planning_data(planning, actual, 200)
    assert hidden["leg_timing"] == actual["leg_timing"]
    assert planning["leg_timing"][0]["departure_second"] == 10


def test_observed_merge_hides_remaining_duration_of_current_stop():
    planning = {"tracks": [{"track_id": "AB_up", "travel_seconds": 100}],
        "plans": [{"trips": [{"trip_id": "H", "route": ["AB_up"]}]}],
        "leg_timing": [{"train_id": "T", "trip_id": "H", "leg_index": 0,
                        "departure_second": 0, "service_seconds": 20, "holding_seconds": 0}]}
    actual = copy.deepcopy(planning)
    actual["leg_timing"][0]["service_seconds"] = 1000
    first = observed_planning_data(planning, actual, 150)
    actual["leg_timing"][0]["service_seconds"] = 2000
    assert observed_planning_data(planning, actual, 150) == first
    assert first["leg_timing"][0]["service_seconds"] == 50


def test_clamped_unexecuted_departures_keep_planned_fifo_order():
    planning = {"tracks": [{"track_id": "AB_up", "travel_seconds": 100}],
        "plans": [{"trips": [{"trip_id": "H1", "route": ["AB_up"]}, {"trip_id": "H2", "route": ["AB_up"]}]}],
        "leg_timing": [{"train_id": train, "trip_id": trip, "leg_index": 0,
            "departure_second": second, "service_seconds": 20, "holding_seconds": 0}
            for train, trip, second in (("Z", "H1", 10), ("A", "H2", 20))]}
    actual = copy.deepcopy(planning)
    for row in actual["leg_timing"]: row["departure_second"] += 1000
    observed = observed_planning_data(planning, actual, 200)
    assert observed["leg_timing"][0]["departure_second"] < observed["leg_timing"][1]["departure_second"]


def test_repeated_keep_preserves_passengers_against_continuous_real_engine(monkeypatch):
    import goal43_replay as replay
    from goal43_controller import run_controller
    from goal43_policy import load_policy
    from passenger_groups import GroupedPassengerSimulation
    policy = load_policy()
    data, _, ready, profiles, meta = replay.build_operational_day(policy, count=3, service_hours=1,
        small_line=["L1_01", "L1_07", "L1_19"])
    policy = data["headway_policy"]
    groups = replay.actual_groups(replay.fixture_history(["L1_01", "L1_07", "L1_19"], 4), data, meta["service_end_second"])
    sources = {od: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], rows, profiles)
               for od, rows in groups.items()}
    snapshots = {od: runner.run(end_second=1200)["checkpoint"] for od, runner in sources.items()}
    observed = []
    def keep(data, sources, snapshots, *args, **kwargs):
        observed.append({od: (s["second"], s["entered"], s["queued"], s["onboard_count"]) for od, s in snapshots.items()})
        return {"decision": {"selection": {"selected_candidate_id": "KEEP"}}, "ledger": [], "bundle": {},
                "candidates": [{"candidate_id": "KEEP", "_data": data}]}
    monkeypatch.setattr(replay, "live_decision", keep)
    controlled = run_controller(data, sources, snapshots, profiles, ready, None, policy,
        scenario_id="keep", minimum_remaining=2, groups=groups, steps=2)
    assert [r["S1"][0] for r in observed] == [1200, 2100]
    assert observed[1]["S1"][1] > observed[0]["S1"][1]
    for od, runner in sources.items():
        continuous = runner.run()["checkpoint"]
        state = controlled["states"][od]
        assert state["board_events"] == continuous["board_events"]
        for key in ("entered", "completed", "queued", "approaching", "onboard_count", "waiting_passenger_seconds"):
            assert state[key] == continuous[key]
