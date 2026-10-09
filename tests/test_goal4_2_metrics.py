from decision_engine import build_regular_plan
from passenger_groups import GroupedPassengerSimulation
from replay_metrics import cohort_waiting, overload_metrics, peak_exchange


def example(no_trains=False):
    config=dict(station_ids=["A","B"],date="2026-09-01",timezone="Europe/Moscow",length_m=2000,
        travel_seconds=120,stop_seconds=0,turnaround_seconds=10,train_type="fixture",car_types=["fixture"])
    data,_=build_regular_plan(config,2,end_second=520)
    if no_trains: data["plans"][0]["trips"]=[]
    profiles={t["train_id"]:[dict(car_id="C1",car_type="fixture",max_capacity=5,
        normal_threshold=2,source="fixture")] for t in data["trains"]}
    groups=[dict(group_id="G0",turnstile_second=0,platform_second=0,origin="A",count=4,
        destinations=[dict(station_id="B",count=4)]),dict(group_id="G1",turnstile_second=1,
        platform_second=1,origin="A",count=1,destinations=[dict(station_id="B",count=1)])]
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    return runner,runner.run()


def test_all_arrivals_include_people_boarding_after_main_window():
    runner,out=example()
    result=cohort_waiting(runner,out["checkpoint"],0,100)
    assert result["arrived"] == result["boarded"] == 5
    assert result["waiting_passenger_seconds_lower_bound"] == 129
    assert result["mean_wait_seconds"] == 25.8
    assert result["p95_wait_seconds"] == 129 and result["complete"]


def test_unserved_cohort_keeps_waiting_lower_bound_and_incomplete_status():
    runner,out=example(True)
    result=cohort_waiting(runner,out["checkpoint"],0,100)
    assert result["unboarded"] == 5
    assert result["waiting_passenger_seconds_lower_bound"] == 2599
    assert result["mean_wait_seconds"] is None and not result["complete"]


def test_overload_is_integrated_in_seconds_and_passenger_seconds():
    runner,out=example()
    result=overload_metrics(runner,out["checkpoint"],0,520)
    assert result["max_car_load"] == 4
    assert result["exceeded_car_seconds"] == 120
    assert result["passenger_seconds_in_exceeded_cars"] == 480
    assert result["affected_unique_passengers"] is None


def test_peak_has_independent_exchange_and_queue_reconstruction():
    runner,out=example()
    peak=peak_exchange(runner,out["checkpoint"],0,520)
    assert peak["second"] == 0 and peak["load_before_arrival"] == 0
    assert peak["alighted"] == 0 and peak["boarded"] == peak["load_after_departure"] == 4
    assert peak["queue_before_boarding"] == 4 and peak["queue_after_boarding"] == 0

