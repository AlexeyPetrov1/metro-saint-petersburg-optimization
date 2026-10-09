from decision_engine import build_regular_plan
from replay_metrics import movement_snapshot
from simulation import simulate


def test_boundary_snapshot_keeps_events_at_boundary_pending():
    config=dict(station_ids=["A","B"],date="2026-09-01",timezone="Europe/Moscow",length_m=2000,
        travel_seconds=120,stop_seconds=0,turnaround_seconds=10,train_type="fixture",car_types=["fixture"])
    data,_=build_regular_plan(config,2,end_second=520); motion=simulate(data,"line1-baseline")
    at120=movement_snapshot(data,motion,120)
    train=next(t for t in at120["trains"] if t["train_id"]=="T01")
    assert train["state"]=="moving" and train["location"]=="P01_up"
    assert train["unfinished_obligation"]["arrival_second"]==120
    at121=movement_snapshot(data,motion,121)
    train=next(t for t in at121["trains"] if t["train_id"]=="T01")
    assert train["state"]=="turnaround" and train["location"]=="B"
    assert train["unfinished_obligation"]["ready_second"]==130

