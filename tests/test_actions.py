import copy

import pytest

from actions import demo_depots, expand_action, evaluate_candidates, select_candidate
from decision_engine import build_regular_plan
from dwell import compile_timing
from passenger_groups import GroupedPassengerSimulation


def inputs():
    config=dict(station_ids=["A","B"],date="2026-09-01",timezone="Europe/Moscow",
        length_m=2000,travel_seconds=120,stop_seconds=0,turnaround_seconds=10,
        train_type="fixture",car_types=["fixture"],capacity_by_type={"fixture":5},normal_threshold=5)
    data,_=build_regular_plan(config,2,end_second=1820)
    data,registry=demo_depots(data,config)
    data,_=compile_timing(data,"line1-baseline","fixed",0,committed=[])
    profiles={t["train_id"]:[dict(car_id="C1",car_type="fixture",max_capacity=5,
        normal_threshold=5,source="fixture")] for t in data["trains"]}
    groups=[dict(group_id="G0",turnstile_second=0,platform_second=0,origin="A",count=4,
        destinations=[dict(station_id="B",count=4)]),dict(group_id="G1",turnstile_second=1,
        platform_second=1,origin="A",count=1,destinations=[dict(station_id="B",count=1)])]
    return data,registry,profiles,groups


def test_withdrawal_finishes_current_trip_and_uses_an_explicit_exit_route():
    data,registry,profiles,groups=inputs()
    before=copy.deepcopy(data)
    changed,action=expand_action(data,"WITHDRAW","T01",10,registry)
    changed,_=compile_timing(changed,"line1-baseline","fixed",0,committed=data["leg_timing"],as_of=10)
    runner=GroupedPassengerSimulation(changed,"line1-baseline",groups,profiles)
    out=runner.run()
    assert action["planned_effect_second"] >= 120
    assert out["metrics"]["balance"]["passed"]
    assert next(t for t in runner.motion["final_state"] if t["train_id"]=="T01")["location"] == "DEPOT_B"
    assert data == before


def test_added_train_stays_at_depot_until_preparation_and_entry_finish():
    data,registry,profiles,groups=inputs()
    changed,action=expand_action(data,"ADD","R01",10,registry)
    changed,_=compile_timing(changed,"line1-baseline","fixed",0,committed=data["leg_timing"],as_of=10)
    runner=GroupedPassengerSimulation(changed,"line1-baseline",groups,profiles)
    departures=[e for e in runner.motion["events"] if e["train_id"]=="R01" and e["event_type"]=="departure"]
    assert min(e["second"] for e in departures) >= 910
    assert action["planned_effect_second"] >= 1030
    assert runner.motion["checks"]["peak_trains_on_line"] == 3
    assert action["operational_status"] == "conditional_scenario"


def test_branch_keeps_people_in_cars_queues_and_approaches():
    data,registry,profiles,groups=inputs()
    source=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    state=source.run(end_second=10)["checkpoint"]
    changed,_=expand_action(data,"WITHDRAW","T01",10,registry)
    changed,_=compile_timing(changed,"line1-baseline","fixed",0,committed=data["leg_timing"],as_of=10)
    target=GroupedPassengerSimulation(changed,"line1-baseline",groups,profiles)
    branched=target.branch_checkpoint(state,source)
    assert {k:branched[k] for k in ("loads","onboard","queues","entered")} == {k:state[k] for k in ("loads","onboard","queues","entered")}
    assert target.run(start_second=10,checkpoint=branched)["metrics"]["balance"]["passed"]


def test_branch_rejects_rewriting_a_past_arrival():
    data,_,profiles,groups=inputs()
    source=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    state=source.run(end_second=10)["checkpoint"]
    changed=copy.deepcopy(groups); changed[0]["count"]=5; changed[0]["destinations"][0]["count"]=5
    target=GroupedPassengerSimulation(data,"line1-baseline",changed,profiles)
    with pytest.raises(ValueError,match="past"):
        target.branch_checkpoint(state,source)


def test_seeded_split_run_is_identical_to_continuous_run():
    data,_,profiles,groups=inputs()
    data,_=compile_timing(data,"line1-baseline","mild",13,committed=[])
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    full=runner.run()
    state=runner.run(end_second=101)["checkpoint"]
    split=runner.run(start_second=101,checkpoint=state)
    assert full["checkpoint"] == split["checkpoint"]


def test_candidate_order_does_not_change_actual_evaluations():
    data,registry,profiles,groups=inputs()
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    snapshot=runner.run(end_second=10)["checkpoint"]
    candidates=[("KEEP",None),("ADD","R01"),("WITHDRAW","T01")]
    a=evaluate_candidates(data,groups,profiles,registry,snapshot,runner,candidates,"fixed",0)
    b=evaluate_candidates(data,groups,profiles,registry,snapshot,runner,list(reversed(candidates)),"fixed",0)
    assert {r["candidate_id"]:r["metrics"] for r in a} == {r["candidate_id"]:r["metrics"] for r in b}
    assert all(r["cash_delta_RUB"] is None for r in a)


def row(kind,load,km):
    return dict(candidate_id=kind,action=dict(kind=kind),status="evaluated",normal_load_passed=load<=120,
        physical_passed=True,metrics=dict(max_car_load=load,waiting_passenger_seconds=0),
        resources=dict(train_km=str(km)),cash_delta_RUB=None)


def test_keep_when_unchanged_and_withdraw_only_when_service_passes():
    assert select_candidate([row("KEEP",90,10),row("WITHDRAW",140,9)])["selected_candidate_id"] == "KEEP"
    assert select_candidate([row("KEEP",90,10),row("WITHDRAW",100,9)])["selected_candidate_id"] == "WITHDRAW"


def test_growth_selects_add_but_impossible_growth_is_honest_refusal():
    assert select_candidate([row("KEEP",130,10),row("ADD",110,11)])["selected_candidate_id"] == "ADD"
    result=select_candidate([row("KEEP",150,10),row("ADD",130,11)])
    assert result["status"] == "no_feasible_candidate" and result["selected_candidate_id"] is None
    assert result["mitigation_candidate_id"] == "ADD"


def test_unready_reserve_is_rejected_and_limit_is_not_exceeded():
    data,registry,_,_=inputs(); registry["R01"]["ready"] = False
    with pytest.raises(ValueError,match="ready"): expand_action(data,"ADD","R01",10,registry)
    registry["R01"]["ready"] = True; data["max_trains_on_line"]=2
    changed,_=expand_action(data,"ADD","R01",10,registry)
    with pytest.raises(ValueError,match="capacity"):
        compile_timing(changed,"line1-baseline","fixed",0,committed=data["leg_timing"],as_of=10)

