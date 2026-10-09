import json
from copy import deepcopy
from pathlib import Path

import pytest

from interfaces.streamlit_dispatcher.adapter import Replay


@pytest.fixture
def replay():
    data = {
        "period_start": "2026-09-30T06:00:00+03:00",
        "stations": [{"station_id": "L1_01", "kind": "line"},
                     {"station_id": "L1_02", "kind": "line"}],
        "tracks": [{"track_id": "P01_up", "from_station": "L1_01",
                    "to_station": "L1_02", "travel_seconds": 100}],
        "trains": [{"train_id": "B001", "train_type": "Baltiets",
                    "location": "L1_01", "cars": 8}],
    }
    groups = [{"group_id": "g1", "origin": "L1_01", "turnstile_second": 0,
               "platform_second": 120, "count": 10,
               "destinations": [{"station_id": "L1_02", "direction": 1, "count": 10}]}]
    state = {"second": 1000, "board_events": [{"second": 150, "group_id": "g1",
             "destination": "L1_02", "count": 7, "wait_seconds": 30}],
             "load_events": [{"second": 150, "train_id": "B001", "car_id": "C1", "load": 7},
                             {"second": 250, "train_id": "B001", "car_id": "C1", "load": 0}]}
    motion = {"events": [{"second": 150, "event_type": "departure",
                         "location": "P01_up", "train_id": "B001"},
                        {"second": 250, "event_type": "arrival",
                         "location": "L1_02", "train_id": "B001"}]}
    forecasts = [{"issued_at": "2026-09-30T06:01:00+03:00", "model_version": "stack",
                  "rows": [{"station_id": "L1_01", "interval_start": "2026-09-30T06:00:00+03:00",
                            "interval_end": "2026-09-30T06:15:00+03:00", "predicted_count": 12}]},
                 {"issued_at": "2026-09-30T06:15:00+03:00", "model_version": "stack",
                  "rows": [{"station_id": "L1_01", "interval_start": "2026-09-30T06:15:00+03:00",
                            "interval_end": "2026-09-30T06:30:00+03:00", "predicted_count": 18}]}]
    decisions = [{"decision_second": 150, "applied_candidate_id": "KEEP",
                  "selection": {"status": "no_feasible_candidate", "release_authorized": False}}]
    return Replay(data, groups, state, motion, forecasts, decisions)


def test_queue_and_train_load_are_separate_and_conserved(replay):
    f = replay.frame(200)
    assert f["entered"] == 10
    assert f["waiting"] == 3
    assert f["onboard"] == 7
    assert f["completed"] == 0
    assert f["stations"][0]["north"] == 3
    assert f["trains"][0]["fraction"] == pytest.approx(.5)
    assert sum(f["trains"][0]["cars"]) == 7
    assert replay.frame(251)["completed"] == 7


def test_no_future_forecasts_or_decisions(replay):
    assert replay.frame(59)["forecast"] is None
    assert replay.frame(149)["decision"] is None
    assert replay.frame(899)["forecast"]["rows"][0]["q50"] == 12
    assert replay.frame(900)["forecast"]["rows"][0]["q50"] == 18
    assert replay.frame(200)["decision"]["selection"]["release_authorized"] is False


def test_approaching_are_not_platform_queue(replay):
    f = replay.frame(60)
    assert f["approaching"] == 10
    assert f["waiting"] == f["onboard"] == 0


def test_read_only_and_observation_boundary(replay):
    before = deepcopy(replay.data)
    replay.frame(200)
    assert replay.data == before
    with pytest.raises(ValueError, match="recorded"):
        replay.frame(1001)


def test_script_payload_does_not_close_script(replay):
    from interfaces.streamlit_dispatcher.player import player_html
    html = player_html({"times": [], "scenario": "</script><script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in html
    assert "__PAYLOAD__" not in html


def test_real_saved_arm_matches_checkpoint():
    arm = Path("data/processed/metro51_balanced_pilot_20261009_v2/ensemble_old")
    if not arm.exists():
        pytest.skip("local campaign unavailable")
    replay = Replay.from_arm(arm)
    f = replay.frame(replay.end)
    for key, actual in [("entered", "entered"), ("waiting", "queued"),
                        ("onboard", "onboard_count"), ("completed", "completed")]:
        assert f[key] == replay.state[actual]
    assert f["waiting"] == sum(x["north"] + x["south"] for x in f["stations"])
    assert f["onboard"] == sum(sum(x["cars"]) for x in f["trains"])


def test_partial_arm_uses_checkpoint_and_does_not_invent_future():
    arm = Path("data/processed/metro51_balanced_pilot_20261009_v2/balanced")
    if not arm.exists():
        pytest.skip("local campaign unavailable")
    replay = Replay.from_arm(arm)
    assert replay.partial
    assert replay.end == replay.state["second"]
    assert replay.frame(replay.end)["waiting"] == replay.state["queued"]


def test_checkpoint_geometry_matches_recorded_movement():
    from interfaces.streamlit_dispatcher.adapter import planned_motion, read
    arm = Path("data/processed/metro51_balanced_pilot_20261009_v2/ensemble_old")
    if not arm.exists():
        pytest.skip("local campaign unavailable")
    run = arm / read(arm / "result.json")["run"]["folder"]
    data, recorded = read(run / "movement_input.json"), read(run / "motion.json")
    key = lambda r: (r["train_id"], r["second"], r["event_type"], r["location"])
    expected = {key(r) for r in recorded["events"] if r["event_type"] in ("arrival", "departure") and 0 <= r["second"] < 23400}
    actual = {key(r) for r in planned_motion(data)["events"] if 0 <= r["second"] < 23400}
    assert actual == expected


def test_all_control_snapshots_match_saved_counts():
    from interfaces.streamlit_dispatcher.adapter import read
    arm = Path("data/processed/metro51_balanced_pilot_20261009_v2/balanced")
    if not arm.exists():
        pytest.skip("local campaign unavailable")
    replay = Replay.from_arm(arm)
    for path in sorted((arm / "controller").glob("resume_*.json.gz")):
        checkpoint = read(path)
        f = replay.frame(checkpoint["second"])
        for key, actual in [("entered", "entered"), ("waiting", "queued"),
                            ("onboard", "onboard_count"), ("completed", "completed")]:
            assert f[key] == checkpoint["state"][actual], (path.name, key)


def test_upstream_assets_are_preserved():
    import hashlib
    root = Path("interfaces/streamlit_dispatcher")
    manifest = json.loads((root / "upstream.json").read_text(encoding="utf8"))
    for name, sha in manifest["sha256"].items():
        assert hashlib.sha256((root / "upstream" / name).read_bytes()).hexdigest() == sha
