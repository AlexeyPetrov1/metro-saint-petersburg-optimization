import copy
from pathlib import Path

import pytest

from goal43_campaign import run_campaign, future_world, STATION_IDS
from goal43_policy import load_policy
from goal43_replay import build_operational_day
from goal43_replay import actual_groups, fixture_history
from simulation import simulate, digest


def test_future_world_preserves_shared_past_but_changes_intermediate_dwell():
    policy = load_policy()
    data, _, _, _, _ = build_operational_day(policy, count=3, small_line=["A", "M", "B"],
                                            service_hours=18, profile="mild", seed=7)
    as_of = 21600
    first = future_world(data, data["headway_policy"], as_of, 100)
    second = future_world(data, data["headway_policy"], as_of, 200)
    a, b = (simulate(world, world["plans"][0]["plan_id"]) for world in (first, second))
    assert [row for row in a["events"] if row["second"] < as_of] == [row for row in b["events"] if row["second"] < as_of]
    assert [row for row in first["leg_timing"] if row["departure_second"] >= as_of] != [row for row in second["leg_timing"] if row["departure_second"] >= as_of]


def test_campaign_compares_four_families_then_validates_frozen_plan(tmp_path):
    initial = run_campaign(tmp_path, seeds=[100], stage="preliminary")
    assert initial["scope"]["evaluation_mode"] == "explicit_synthetic_three_station_fixture"
    assert initial["scope"]["historical_line53_validation"] is False
    assert {row["strategy"] for row in initial["trials"]} == {"KEEP", "REGULATE_ONLY", "WITHDRAW_ONLY", "COMBINED"}
    assert {row["od"] for row in initial["trials"]} == {"S1", "S2", "S3"}
    assert all(row["future_seed"] == 100 and row["past_seed"] == 7 for row in initial["trials"])
    descriptor = copy.deepcopy(initial["selected_plan"])
    final = run_campaign(tmp_path, seeds=[200], stage="validation", selected_plan=descriptor)
    assert descriptor == initial["selected_plan"]
    assert final["selected_plan"]["plan_digest"] == descriptor["plan_digest"]
    assert final["selection_recomputed"] is False
    assert {row["strategy"] for row in final["trials"]} == {"KEEP", descriptor["strategy"]}
    assert all(row["future_seed"] == 200 and row["past_seed"] == 7 for row in final["trials"])
    assert len(final["records"]) == len(final["trials"])
    assert all((tmp_path / row["folder"] / "summary.json").exists() for row in final["records"])
    assert all(row["full_day"] and row["obligations_closed"] for row in final["trials"])
    assert any(float(row["saving_km"]) > 0 and row["service_admissible"] for row in final["trials"])


def test_campaign_od_scenarios_have_distinct_destination_allocations():
    data, _, _, _, meta = build_operational_day(load_policy(), count=3, small_line=STATION_IDS,
                                               service_hours=18, profile="mild", seed=7)
    groups = actual_groups(fixture_history(STATION_IDS, count_per_minute=4), data, meta["service_end_second"])
    assert len({digest(rows) for rows in groups.values()}) == 3


def test_validation_requires_frozen_selection_and_distinct_seeds(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        run_campaign(tmp_path, seeds=[200], stage="validation")
    with pytest.raises(ValueError, match="overlap"):
        run_campaign(tmp_path, seeds=[100], stage="validation", selected_plan={"preliminary_seeds": [100]})


@pytest.mark.parametrize("seeds", [[], [100,100], [True], [-1]])
def test_campaign_rejects_invalid_seed_contract_before_running(tmp_path, seeds):
    with pytest.raises(ValueError):
        run_campaign(tmp_path, seeds=seeds)
