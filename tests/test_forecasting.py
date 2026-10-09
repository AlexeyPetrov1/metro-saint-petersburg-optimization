import copy

import pandas as pd
import pytest

from forecasting import hourly_median, validate_bundle, forecast_groups


def history():
    rows = []
    for day, total in (("2026-08-29", 40), ("2026-08-30", 80), ("2026-08-31", 120), ("2026-09-01", 4000)):
        for station in ("A", "B"):
            for hour in (8, 9):
                for quarter in range(4):
                    start = pd.Timestamp(f"{day}T{hour:02}:{quarter*15:02}:00+03:00")
                    rows.append(dict(station_id=station, interval_start=start,
                        interval_end=start+pd.Timedelta(minutes=15), entry_count=total//4,
                        is_complete=True, source_id="fixture"))
    return pd.DataFrame(rows)


def bundle():
    return hourly_median(history(), ["A", "B"], "2026-09-01T08:00:00+03:00")


def test_hourly_total_median_uses_only_past_and_preserves_hour_sum():
    out = bundle()
    assert out["model_version"] == "station-hour-median-v1"
    assert sum(r["predicted_count"] for r in out["rows"] if r["station_id"] == "A") == 160
    groups = forecast_groups(out, ["A", "B"], "2026-09-01T06:00:00+03:00", delay=120)
    assert sum(g["count"] for g in groups) == 320
    assert all(g["turnstile_second"] >= 7200 for g in groups)
    assert all(g["platform_second"]-g["turnstile_second"] == 120 for g in groups)


def test_median_of_complete_hour_totals_is_not_sum_of_quarter_medians():
    data = history().query("station_id == 'A'").copy()
    data = data[data.interval_start.dt.date.astype(str) < "2026-09-01"]
    data.loc[data.interval_start.dt.hour == 8, "entry_count"] = [100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, 0]
    out = hourly_median(data, ["A"], "2026-09-01T08:00:00+03:00")
    assert sum(r["predicted_count"] for r in out["rows"][:4]) == 100


@pytest.mark.parametrize("damage", ["target", "unit", "negative", "nan", "duplicate", "missing", "future_features", "future_issue", "stale"])
def test_invalid_forecast_cannot_be_used_as_entries(damage):
    out = copy.deepcopy(bundle())
    if damage == "target": out["target"] = "station_occupancy"
    elif damage == "unit": out["unit"] = "passengers_on_station"
    elif damage == "negative": out["rows"][0]["predicted_count"] = -1
    elif damage == "nan": out["rows"][0]["predicted_count"] = float("nan")
    elif damage == "duplicate": out["rows"].append(out["rows"][0])
    elif damage == "missing": out["rows"].pop()
    elif damage == "future_features": out["features_available_at"] = "2026-09-01T08:01:00+03:00"
    elif damage == "future_issue": out["issued_at"] = "2026-09-01T08:01:00+03:00"
    elif damage == "stale": out["issued_at"] = out["available_at"] = "2026-09-01T06:00:00+03:00"
    with pytest.raises(ValueError): validate_bundle(out, ["A", "B"], "2026-09-01T08:00:00+03:00")


def test_incomplete_hour_is_excluded_and_unknown_coverage_is_visible():
    data = history(); data.loc[data.interval_start.dt.hour == 9, "is_complete"] = False
    with pytest.raises(ValueError, match="coverage"):
        hourly_median(data, ["A", "B"], "2026-09-01T08:00:00+03:00")


def test_events_already_in_prediction_are_not_multiplied_again():
    out = bundle(); out["included_event_ids"] = ["concert"]
    with pytest.raises(ValueError, match="already"):
        forecast_groups(out, ["A", "B"], "2026-09-01T06:00:00+03:00", extra_event_ids=["concert"])


def test_forecast_refresh_is_hourly_and_available_horizon_shrinks():
    out = bundle()
    report = validate_bundle(out, ["A", "B"], "2026-09-01T08:45:00+03:00")
    assert report["available_horizon_seconds"] == 4500
    with pytest.raises(ValueError, match="hour"):
        hourly_median(history(), ["A", "B"], "2026-09-01T08:15:00+03:00")


@pytest.mark.parametrize('receipt', ['2026-09-01T08:01:00+03:00', None])
def test_median_excludes_late_or_unknown_receipt_despite_past_publication(receipt):
    data = history()
    data['available_at'] = data.interval_end
    data['received_at'] = data.interval_end.copy()
    mask = data.interval_start.dt.date.astype(str).eq('2026-08-31')
    data.loc[mask, 'received_at'] = pd.Timestamp(receipt) if receipt else pd.NaT
    out = hourly_median(data, ['A'], '2026-09-01T08:00:00+03:00')
    assert sum(r['predicted_count'] for r in out['rows']) == 120


def test_median_rejects_timezone_naive_receipt():
    data = history()
    data['received_at'] = data.interval_end.dt.tz_localize(None)
    with pytest.raises(ValueError, match='timezone'):
        hourly_median(data, ['A'], '2026-09-01T08:00:00+03:00')


def test_median_feature_availability_includes_delayed_receipt():
    data = history()
    data['available_at'] = data.interval_end
    data['received_at'] = data.interval_end.copy()
    data.loc[data.interval_start.dt.date.astype(str).eq('2026-08-31'), 'received_at'] = pd.Timestamp('2026-08-31T19:00:00+03:00')
    out = hourly_median(data, ['A'], '2026-09-01T08:00:00+03:00')
    assert pd.Timestamp(out['features_available_at']) == pd.Timestamp('2026-08-31T19:00:00+03:00')

