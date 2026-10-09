import copy

import pytest

from actions import demo_depots
from decision_engine import build_regular_plan
from dwell import compile_timing
from forecasting import hourly_median
from passenger_groups import GroupedPassengerSimulation
from rolling import decide_snapshot, preserve_past_groups
from tests.test_forecasting import history


def episode(future_count=1):
    config=dict(station_ids=["A","B"],date="2026-09-01",timezone="Europe/Moscow",length_m=2000,
        travel_seconds=120,stop_seconds=0,turnaround_seconds=10,train_type="fixture",car_types=["fixture"])
    data,_=build_regular_plan(config,2,end_second=14400)
    data,registry=demo_depots(data,config)
    data,_=compile_timing(data,"line1-baseline","fixed",0,committed=[])
    profiles={t["train_id"]:[dict(car_id="C1",car_type="fixture",max_capacity=5,
        normal_threshold=5,source="fixture")] for t in data["trains"]}
    groups=[dict(group_id="past",turnstile_second=7000,platform_second=7300,origin="A",count=1,
        destinations=[dict(station_id="B",count=1)]),dict(group_id="future",turnstile_second=7400,
        platform_second=7400,origin="A",count=future_count,destinations=[dict(station_id="B",count=future_count)])]
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    state=runner.run(end_second=7200)["checkpoint"]
    bundle=hourly_median(history(),["A","B"],"2026-09-01T08:00:00+03:00")
    return data,registry,profiles,runner,state,bundle


def test_pending_platform_arrivals_keep_actual_past_turnstile_counts():
    data,registry,profiles,runner,state,bundle=episode()
    groups=preserve_past_groups(runner,[dict(group_id="new",turnstile_second=7500)],7200)
    assert groups[0]["group_id"] == "past" and groups[0]["platform_second"] == 7300


def test_future_truth_does_not_change_forecast_candidate_scores():
    first=episode(1); second=episode(500)
    def decide(parts):
        data,registry,profiles,runner,state,bundle=parts
        return decide_snapshot(data,{"S1":runner},{"S1":state},profiles,registry,bundle,
            [("KEEP",None),("WITHDRAW","T01")],"fixed",[0])
    a,b=decide(first),decide(second)
    assert a["selection"] == b["selection"]
    assert a["candidate_table"] == b["candidate_table"]


def test_changed_forecast_changes_simulated_demand_without_multipliers():
    data,registry,profiles,runner,state,bundle=episode()
    a=decide_snapshot(data,{"S1":runner},{"S1":state},profiles,registry,bundle,[("KEEP",None)],"fixed",[0])
    changed=copy.deepcopy(bundle)
    for row in changed["rows"]: row["predicted_count"] *= 100
    b=decide_snapshot(data,{"S1":runner},{"S1":state},profiles,registry,changed,[("KEEP",None)],"fixed",[0])
    assert b["candidate_table"][0]["forecast_entries"] == a["candidate_table"][0]["forecast_entries"]*100
    assert b["candidate_table"][0]["waiting_max"] > a["candidate_table"][0]["waiting_max"]

