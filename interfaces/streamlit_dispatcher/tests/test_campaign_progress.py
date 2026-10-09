"""New and growing campaigns remain readable without computing new predictions."""

import gzip
import json
import weakref
from pathlib import Path

from streamlit.testing.v1 import AppTest

from interfaces.streamlit_dispatcher.adapter import Replay
from interfaces.streamlit_dispatcher.tests.test_adapter import replay

APP = Path(__file__).resolve().parents[1] / "app.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False)


def arm_files(root, replay, *, second=None, complete=False):
    arm = root / "balanced"
    write(arm / "common_scenario.json.gz", {"data": replay.data, "groups": replay.groups})
    if second is None:
        return arm
    state = {**replay.state, "second": second}
    if complete:
        run = arm / "runs/RUN51_BALANCED"
        write(run / "movement_input.json", replay.data)
        write(run / "passenger_checkpoint.json.gz", state)
        write(run / "motion.json", {"events": [e for rows in replay.motion.values() for e in rows
                                               if e["event_type"] != "initial"]})
        write(arm / "result.json", {"run": {"folder": "runs/RUN51_BALANCED"}})
    else:
        write(arm / "controller" / f"resume_{second:05d}.json.gz",
              {"second": second, "data": replay.data, "state": state})
    for i, forecast in enumerate(replay.forecasts):
        write(arm / "controller" / f"raw_forecast_{i:05d}.json", forecast)
    return arm


def test_pending_arm_is_progress_not_application_error(tmp_path, monkeypatch, replay):
    arm_files(tmp_path, replay)
    monkeypatch.setenv("METRO_CAMPAIGN_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert any("контрольного состояния" in row.value for row in app.info)


def test_refresh_reads_new_checkpoint_and_completed_day(tmp_path, monkeypatch, replay):
    arm = arm_files(tmp_path, replay, second=1000)
    monkeypatch.setenv("METRO_CAMPAIGN_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.slider(key="clock_second").max == 1000
    arm_files(tmp_path, replay, second=1800)
    app.button[0].click().run()
    assert not app.exception
    assert app.slider(key="clock_second").max == 1800
    arm_files(tmp_path, replay, second=72000, complete=True)
    app.button[0].click().run()
    assert not app.exception
    assert app.slider(key="clock_second").max == 72000
    saved = Replay.from_arm(arm)
    assert not saved.partial
    assert saved.end == 72000
    assert saved.frame(72000)["forecast"]["stale"]
    assert saved.frame(72000)["decision"] is None
    assert max(f["second"] for f in saved.browser_data()["forecasts"]) == 900


def test_saved_full_day_counts_and_empty_forecasts():
    arm = Path("data/processed/metro50_full_day_20261009_v4/reference")
    if not arm.exists():
        import pytest
        pytest.skip("local completed full-day campaign unavailable")
    saved = Replay.from_arm(arm)
    assert not saved.partial
    assert saved.end == 72000
    for second in [0, 3599, 7200, 14400, 36000, 64800, 72000]:
        frame = saved.frame(second)
        assert frame["entered"] == sum(frame[k] for k in ("approaching", "waiting", "onboard", "completed"))
        assert frame["forecast"] is None
        assert frame["decision"] is None
    final = saved.frame(saved.end)
    for key, actual in [("entered", "entered"), ("waiting", "queued"),
                        ("onboard", "onboard_count"), ("completed", "completed")]:
        assert final[key] == saved.state[actual]


def test_progress_cache_does_not_retain_five_passenger_ledgers(tmp_path, monkeypatch, replay):
    created = []
    original = Replay.from_arm

    def load(cls, path):
        saved = original(path)
        created.append(weakref.ref(saved))
        return saved

    monkeypatch.setattr(Replay, "from_arm", classmethod(load))
    monkeypatch.setenv("METRO_CAMPAIGN_DIR", str(tmp_path))
    arm_files(tmp_path, replay, second=1000)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    for second in [1800, 2700, 3600, 4500]:
        arm_files(tmp_path, replay, second=second)
        app.button[0].click().run()
        assert not app.exception
    assert len(created) == 5
    # AppTest keeps old script globals for page callbacks. Inspect the resource
    # cache itself: its bound must hold even while the test retains those globals.
    from streamlit.runtime.caching.cache_resource_api import _resource_caches
    caches = [cache for scopes in _resource_caches._function_caches.values() for cache in scopes.values()
              if cache.display_name.endswith(".load_replay")]
    assert caches
    assert all(len(cache._mem_cache) <= 2 for cache in caches)
    assert {item.value.end for cache in caches for item in cache._mem_cache.values()} == {3600, 4500}
