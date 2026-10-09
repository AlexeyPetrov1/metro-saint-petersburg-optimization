import copy

import pytest

from passenger_groups import GroupedPassengerSimulation
from passengers import PassengerSimulation
from test_line1 import tiny_groups, tiny_movement


def inputs(count=960,movement=None):
    data=copy.deepcopy(movement or tiny_movement())
    for train in data["trains"]: train["cars"]=8
    profiles={t["train_id"]:[dict(car_id=f"C{i}",car_type="fixture",max_capacity=160,
        normal_threshold=120,source="fixture") for i in range(8)] for t in data["trains"]}
    return data,tiny_groups(count=count),profiles


def runner(seed=None, count=960, movement=None, preference=None, cap=None):
    data,groups,profiles=inputs(count,movement)
    return GroupedPassengerSimulation(data,"service",groups,profiles,
        boarding_seed=seed,car_preference=preference,boarding_capacity_per_car=cap)


def departure_loads(out, train="T1"):
    return [r["load"] for r in out["load_events"] if r["train_id"]==train and r["event_type"]=="departure"]


def test_default_keeps_legacy_context_and_checkpoint():
    current=runner()
    data,groups,profiles=inputs()
    legacy=GroupedPassengerSimulation(data,"service",groups,profiles)
    assert current.context_id==legacy.context_id
    assert current.run()==legacy.run()


def test_seeded_boarding_is_mildly_uneven_and_reproducible():
    a=runner(19).run(); b=runner(19).run(); c=runner(20).run()
    loads=departure_loads(a)
    assert len(set(loads))>1 and sum(loads)==960
    assert max(loads)-min(loads)<=30
    assert a==b and loads!=departure_loads(c)
    assert a["metrics"]["balance"]["passed"] and a["metrics"]["waiting_identity_passed"]


@pytest.mark.parametrize("seed",[0,1,19,10**20])
def test_stochastic_boarding_preserves_capacity_and_people(seed):
    out=runner(seed,count=2000).run()
    assert out["metrics"]["completed"]==2000
    assert out["metrics"]["queued"]==out["metrics"]["onboard"]==0
    assert all(0<=r["load"]<=160 for r in out["load_events"])
    assert out["metrics"]["balance"]["passed"]


def test_stochastic_split_run_matches_whole_without_rng_state():
    source=runner(19,count=2000); full=source.run(); state=None; start=0
    for end in (1,60,61,100,180,181,300,600):
        state=source.run(start_second=start,end_second=end,checkpoint=state)["checkpoint"]
        start=end
    assert state==full["checkpoint"]


def test_future_branch_preserves_seed_and_past_people():
    source=runner(19,count=2000); checkpoint=source.run(end_second=100)["checkpoint"]
    data=tiny_movement(); data["plans"][0]["trips"][1]["departure_second"]=200
    target=runner(19,count=2000,movement=data)
    first=target.branch_checkpoint(checkpoint,source)
    assert first==target.branch_checkpoint(checkpoint,source)
    assert first["loads"]==checkpoint["loads"] and first["queues"]==checkpoint["queues"]
    out=target.run(start_second=100,checkpoint=first)
    assert out["metrics"]["balance"]["passed"]
    assert out["checkpoint"]==target.run()["checkpoint"]


def test_branch_cannot_change_stochastic_seed():
    source=runner(19); checkpoint=source.run(end_second=100)["checkpoint"]
    with pytest.raises(ValueError,match="seed|boarding"):
        runner(20).branch_checkpoint(checkpoint,source)


@pytest.mark.parametrize("seed",[-1,True,1.5,"19"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError,match="seed"):
        runner(seed)


def test_stochastic_and_explicit_preference_are_mutually_exclusive():
    with pytest.raises(ValueError,match="preference|seed"):
        runner(19,preference=[1]*8)


def test_operating_cap_leaves_people_for_next_train_without_changing_profiles():
    source=runner(19,count=1250,cap=150); full=source.run()
    state=source.run(end_second=100)["checkpoint"]
    assert state["queued"]==50 and state["denied_boarding_attempts"]==50
    assert departure_loads(full)==[150]*8
    assert full["metrics"]["completed"]==1250
    assert all(c["max_capacity"]==160 for cars in source.profiles.values() for c in cars)
    assert source.run(start_second=100,checkpoint=state)["checkpoint"]==full["checkpoint"]


@pytest.mark.parametrize("cap",[0,-1,True,150.5,161])
def test_invalid_operating_cap_is_rejected(cap):
    with pytest.raises(ValueError,match="capacity"):
        runner(19,cap=cap)


def test_branch_cannot_change_operating_cap():
    source=runner(19,cap=150); state=source.run(end_second=100)["checkpoint"]
    with pytest.raises(ValueError,match="capacity|boarding"):
        runner(19,cap=149).branch_checkpoint(state,source)


def test_opt_in_boarding_ledger_preserves_destination_counts():
    out=runner(19,count=2000).run()
    rows=out["checkpoint"]["board_events"]
    assert sum(r["count"] for r in rows if r["destination"]=="B")==2000
    assert all("destination" not in r for r in runner().run()["checkpoint"]["board_events"])
    assert all(r["destination"]=="B" for r in runner(cap=150).run()["checkpoint"]["board_events"])


@pytest.mark.parametrize("destination",[None,"unknown"])
def test_resealed_wrong_boarding_destination_is_rejected(destination):
    source=runner(19); state=source.run(end_second=100)["checkpoint"]
    if destination is None: state["board_events"][0].pop("destination")
    else: state["board_events"][0]["destination"]=destination
    state["sha256"]=PassengerSimulation._seal(state)
    with pytest.raises(ValueError,match="destination"):
        source.run(start_second=100,checkpoint=state)
