from decision_engine import build_regular_plan
from dwell import compile_timing
from passenger_groups import GroupedPassengerSimulation


def test_nonconstant_service_resumes_inside_a_stop_without_changing_events():
    config=dict(station_ids=["A","B","C"],date="2026-09-01",timezone="Europe/Moscow",
        length_m=2000,travel_seconds=103,stop_seconds=20,turnaround_seconds=10,
        train_type="fixture",car_types=["fixture"])
    data,_=build_regular_plan(config,4,end_second=1500)
    data,timing=compile_timing(data,"line1-baseline","wider",0,committed=[])
    assert len(set(timing["service_seconds"]))>3
    profiles={t["train_id"]:[dict(car_id="C1",car_type="fixture",max_capacity=5,
        normal_threshold=2,source="manual control")] for t in data["trains"]}
    groups=[dict(group_id="G",origin="A",turnstile_second=0,platform_second=0,count=8,
        destinations=[dict(station_id="C",count=8)])]
    runner=GroupedPassengerSimulation(data,"line1-baseline",groups,profiles)
    full=runner.run()
    for boundary in (110,229,700):
        first=runner.run(end_second=boundary)
        resumed=runner.run(start_second=boundary,checkpoint=first["checkpoint"])
        assert resumed["checkpoint"]==full["checkpoint"]

