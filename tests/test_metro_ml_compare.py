import copy

import pandas as pd
import pytest


def history():
    rows = []
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        for minute in (0, 15, 30, 45):
            at = pd.Timestamp(f"{day}T08:{minute:02}:00+03:00")
            rows.append(dict(station_id="A", interval_start=at,
                interval_end=at + pd.Timedelta(minutes=15), entry_count=10,
                is_complete=True, source_id="uploaded-spb-line1-15min"))
    return pd.DataFrame(rows)


def test_provider_receives_only_available_completed_history_and_actual_decision(monkeypatch):
    import goal43_replay as replay
    from goal43_policy import load_policy
    frame = history()
    frame["available_at"] = frame.interval_end
    frame.loc[frame.interval_start.eq(pd.Timestamp("2026-09-03T08:00+03:00")), "available_at"] = pd.Timestamp("2026-09-03T09:00+03:00")
    before = frame.copy(deep=True)
    received = []
    def provider(observed, ids, at):
        received.append((observed, ids, at))
        return {"model_version": "injected"}
    monkeypatch.setattr(replay, "hourly_median", lambda *a: pytest.fail("unexpected median"))
    monkeypatch.setattr(replay, "generate_strategy_candidates", lambda *a, **kw: ([], {}))
    monkeypatch.setattr(replay, "decide_operational", lambda *a, **kw: dict(
        selection=dict(status="no_feasible_candidate", selected_candidate_id=None), candidate_table=[]))
    data = dict(period_start="2026-09-03T08:00:00+03:00", stations=[dict(station_id="A", kind="line")])
    result = replay.live_decision(data, {}, {"S1": {"second": 900}}, {}, {}, frame,
        load_policy(), scenario_id="provider", minimum_remaining=1, forecast_provider=provider)
    observed, ids, at = received[0]
    assert ids == ["A"] and at == "2026-09-03T08:15:00+03:00"
    assert len(observed) == 8
    assert observed.interval_end.max() <= pd.Timestamp(at)
    assert result["bundle"]["model_version"] == "injected"
    pd.testing.assert_frame_equal(frame, before)


def test_median_refresh_is_hourly_at_quarter_hour_decisions():
    from metro_ml_compare import median_provider
    from forecasting import validate_bundle
    frame = history()
    one = median_provider(frame, ["A"], "2026-09-03T08:00:00+03:00", horizon_hours=1)
    two = median_provider(frame, ["A"], "2026-09-03T08:15:00+03:00", horizon_hours=1)
    assert one == two
    assert validate_bundle(two, ["A"], "2026-09-03T08:15:00+03:00")["available_horizon_seconds"] == 2700


def test_forecast_error_excludes_observed_slots_and_requires_exact_future_truth():
    from metro_ml_compare import forecast_error
    frame = history()
    bundle = {"rows": [dict(station_id="A", interval_start=f"2026-09-03T08:{m:02}:00+03:00",
        predicted_count=0 if m == 0 else 12) for m in (0, 15, 30, 45)]}
    score = forecast_error(bundle, frame, "2026-09-03T08:15:00+03:00")
    assert score["rows"] == 3
    assert score["wape_percent"] == 20
    assert score["mae_passengers_per_15m"] == 2
    assert score["signed_bias_percent"] == 20
    with pytest.raises(ValueError, match="complete actual coverage"):
        forecast_error(bundle, frame.iloc[:-1], "2026-09-03T08:15:00+03:00")


def test_run_controller_forwards_provider_on_each_decision(monkeypatch):
    import goal43_replay as replay
    from goal43_controller import run_controller
    from goal43_policy import load_policy
    from passenger_groups import GroupedPassengerSimulation
    data, _, ready, profiles, meta = replay.build_operational_day(load_policy(), count=3,
        service_hours=1, small_line=["L1_01", "L1_07", "L1_19"])
    frame = replay.fixture_history(["L1_01", "L1_07", "L1_19"], 4)
    groups = replay.actual_groups(frame, data, meta["service_end_second"])
    groups = {"S1": groups["S1"]}
    sources = {od: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], rows, profiles) for od, rows in groups.items()}
    snapshots = {od: runner.run(end_second=1200)["checkpoint"] for od, runner in sources.items()}
    provider = object()
    observed = []
    def keep(data, sources, snapshots, *args, **kwargs):
        observed.append(kwargs.get("forecast_provider"))
        return dict(decision=dict(selection=dict(selected_candidate_id="KEEP")), ledger=[], bundle={},
            candidates=[dict(candidate_id="KEEP", _data=data)])
    monkeypatch.setattr(replay, "live_decision", keep)
    out = run_controller(data, sources, snapshots, profiles, ready, frame, data["headway_policy"],
        scenario_id="provider", minimum_remaining=2, groups=groups, steps=2, forecast_provider=provider)
    assert observed == [provider, provider]
    assert len(out["decisions"]) == 2


def test_metric_delta_uses_fixed_fleet_and_never_calls_resource_value_cash():
    from metro_ml_compare import metric_delta
    baseline = dict(train_km=100, per_train=[dict(train_id="A", train_type="Yubileyny", km=60),
        dict(train_id="B", train_type="Baltiets", km=40)], mean_wait_seconds=50)
    candidate = dict(train_km=90, per_train=[dict(train_id="A", train_type="Yubileyny", km=50),
        dict(train_id="B", train_type="Baltiets", km=40)], mean_wait_seconds=55)
    delta = metric_delta(baseline, candidate)
    assert delta["observed_saving_train_km"] == 10
    assert delta["observed_saving_km_per_fixed_train"] == 5
    assert delta["mean_wait_delta_seconds"] == 5
    assert delta["observed_to1_resource_equivalent_rub"] == 74
    assert delta["cash_saving_rub"] is None
    assert delta["per_train"][0]["saving_km"] == 10
    changed = copy.deepcopy(candidate)
    changed["per_train"].pop()
    with pytest.raises(ValueError, match="fixed fleet"):
        metric_delta(baseline, changed)


def test_peak_fleet_initial_movement_is_inside_peak():
    from metro_ml_compare import WINDOWS
    from goal43_replay import build_operational_day
    from goal43_policy import load_policy
    spec = WINDOWS["peak"]
    origin = f"2026-09-30T{spec['start_hour']:02}:{spec['start_minute']:02}:00+03:00"
    data, _, _, _, meta = build_operational_day(load_policy(), count=spec["train_count"],
        start_at=origin, service_hours=4)
    assert len([t for t in data["trains"] if not t["location"].startswith("DEPOT_")]) == 53
    assert meta["service_end_second"] == 14400


def test_forecast_scoring_rejects_uniform_hourly_warmup_as_actual():
    from metro_ml_compare import forecast_error
    frame = history()
    frame.loc[frame.interval_start.eq(pd.Timestamp("2026-09-03T08:15:00+03:00")),
        "source_id"] = "uploaded-hourly-uniform-warmup"
    bundle = {"rows": [dict(station_id="A", interval_start="2026-09-03T08:15:00+03:00",predicted_count=10)]}
    with pytest.raises(ValueError,match="real quarter-hour"):
        forecast_error(bundle,frame,"2026-09-03T08:15:00+03:00")


def test_case_rejects_synthetic_actual_targets_before_movement_or_manifest(monkeypatch,tmp_path):
    import metro_ml_compare as comparison
    frame = history()
    frame["source_id"] = "uploaded-hourly-uniform-warmup"
    monkeypatch.setattr(comparison,"build_operational_day",lambda *a,**kw: pytest.fail("must reject truth before simulation"))
    with pytest.raises(ValueError,match="real quarter-hour"):
        comparison.run_case(tmp_path,"2026-09-03","peak",frame,None)
    assert not list(tmp_path.glob("**/experiment_manifest.json"))
