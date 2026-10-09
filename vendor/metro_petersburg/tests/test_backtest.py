import json

import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import config, contract


# --- Фолды ------------------------------------------------------------------------
def test_folds_expanding_with_gap():
    fs = BT.folds()
    assert [f.name for f in fs] == ["май", "июнь", "июль", "август"]
    assert [f.test_start.month for f in fs] == [5, 6, 7, 8]
    for f in fs:
        assert f.train_end <= f.test_start - pd.Timedelta(days=BT.GAP_DAYS)   # зазор ≥ 7 суток
        assert f.test_end == f.test_start + pd.offsets.MonthEnd(0)
    assert all(a.train_end < b.train_end for a, b in zip(fs, fs[1:]))       # окно расширяется
    assert all(a.test_end < b.test_start for a, b in zip(fs, fs[1:]))       # тесты не пересекаются
    assert all(f.test_end < BT.SEPTEMBER for f in fs)


def test_september_only_with_final():
    with pytest.raises(PermissionError):
        BT.get_fold("final")
    f = BT.get_fold("final", final=True)
    assert f.test_start == BT.SEPTEMBER and f.test_end == pd.Timestamp("2026-09-29")
    assert f.train_end == BT.SEPTEMBER - pd.Timedelta(days=BT.GAP_DAYS)


def test_split_does_not_overlap():
    days = pd.date_range("2026-02-09", "2026-08-31", freq="D")
    rows = pd.DataFrame({"sday": days, "is_special": False, "y": 1.0})
    for f in BT.folds():
        train, test = BT.split(rows, f)
        assert train.sday.max() < f.test_start - pd.Timedelta(days=BT.GAP_DAYS - 1)
        assert not set(train.sday) & set(test.sday)
        assert test.sday.min() == f.test_start and test.sday.max() == f.test_end


# --- Метрики и квантили -------------------------------------------------------------
def test_metrics_toy():
    df = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0], "q50": [12.0, 18.0, 30.0, 44.0],
                       "q10": [8.0, 15.0, 31.0, 35.0], "q90": [15.0, 25.0, 35.0, 39.0]})
    m = BT.metrics(df)
    assert m["wape"] == pytest.approx(8 / 100)
    assert m["mae"] == pytest.approx(2.0)
    assert m["pinball_50"] == pytest.approx(0.5 * 2.0)
    # q10: ошибки y − q = 2, 5, −1, 5 → 0,1·2 + 0,1·5 + 0,9·1 + 0,1·5 = 2,1 → среднее 0,525
    assert m["pinball_10"] == pytest.approx(0.525)
    assert m["coverage"] == pytest.approx(0.5)      # 10 ∈ [8,15], 20 ∈ [15,25]; 30 < 31; 40 > 39
    assert m["pinball_norm"] == pytest.approx(np.mean([m["pinball_10"], m["pinball_50"], m["pinball_90"]]) / 25)


def test_ratio_quantiles_calibrate_on_train():
    rng = np.random.default_rng(0)
    n = 4000
    train = pd.DataFrame({"group": "g", "band": rng.choice(["07–09", "10–15"], n), "h": 1, "f": 100.0})
    train["y"] = train.f * np.where(train.band == "07–09", rng.uniform(0.8, 1.2, n), rng.uniform(0.5, 1.5, n))
    m = BT.RatioQuantiles("x", "x", lambda df: df.f).fit(train)
    p = m.predict(train)
    assert (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all()
    assert (p.q50 == 100).all()
    cov = ((train.y >= p.q10) & (train.y <= p.q90)).mean()
    assert cov == pytest.approx(0.8, abs=0.01)
    wide = p[train.band == "10–15"]
    assert wide.q10.iloc[0] == pytest.approx(60, abs=2) and wide.q90.iloc[0] == pytest.approx(140, abs=2)


def test_ratio_quantiles_center_is_cell_median():
    train = pd.DataFrame({"group": "g", "band": ["07–09"] * 5 + ["10–15"] * 5, "h": 1, "f": 100.0,
                          "y": [90.0, 95, 100, 120, 130, 50, 60, 70, 80, 200]})
    off = BT.RatioQuantiles("x", "x", lambda df: df.f).fit(train)
    on = BT.RatioQuantiles("x", "x", lambda df: df.f, center=True).fit(train)
    assert (off.predict(train).q50 == 100).all()                      # по умолчанию q50 = f
    p = on.predict(train)
    assert p.q50.iloc[0] == pytest.approx(100) and p.q50.iloc[5] == pytest.approx(70)


def test_holiday_slice():
    df = pd.DataFrame({"hour": [8, 12], "group": BT.E.GROUPS[0], "is_anomaly": False, "is_holiday": [True, False]})
    assert BT.slices(df)["праздники (контроль)"].tolist() == [True, False]


def test_small_forecast_keeps_quantiles_equal():
    train = pd.DataFrame({"group": "g", "band": "05–06", "h": 1, "f": [100.0] * 10, "y": np.linspace(50, 150, 10)})
    m = BT.RatioQuantiles("x", "x", lambda df: df.f).fit(train)
    p = m.predict(train.assign(f=0.4))
    assert (p.q10 == 0.4).all() and (p.q90 == 0.4).all()


# --- Реальные данные --------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists()):
        pytest.skip("нет interim СПб — запустите python -m src.clean_spb")
    return BT.load_panel()


def test_september_not_in_panel_without_final(panel):
    g = panel.d.grid
    sep = g.sday >= BT.SEPTEMBER
    assert np.isnan(g.entries[sep]).all() and np.isnan(g.y[sep]).all()
    assert (panel.d.df.sday < BT.SEPTEMBER).all()
    rows = BT.make_rows(panel)
    assert rows.sday.max() < BT.SEPTEMBER
    exp = BT.make_rows(panel, for_export=True)
    assert exp.loc[exp.sday >= BT.SEPTEMBER, "y"].isna().all()


def test_rows_targets_clean(panel):
    rows = BT.make_rows(panel)
    assert rows.hour.isin(BT.METRIC_HOURS).all()
    assert rows.y.notna().all() and rows.b4.notna().all()
    assert set(rows.h) == {1, 2}
    assert (rows.tau - rows.t == pd.to_timedelta(rows.h, unit="h")).all()
    assert set(rows.loc[rows.is_special, "sday"].dt.strftime("%Y-%m-%d")) == {"2026-05-29", "2026-06-27", "2026-08-31"}
    assert set(rows.loc[rows.is_holiday, "sday"].dt.strftime("%d.%m")) == {"23.02", "08.03", "09.03", "01.05", "09.05",
                                                                          "11.05", "12.06"}


def test_export_file_passes_contract():
    if not BT.AUG_OUT.exists():
        pytest.skip("нет baseline_aug.json — запустите python -m src.backtest --export-august")
    preds = contract.validate_file(BT.AUG_OUT)
    recs = json.loads(BT.AUG_OUT.read_text(encoding="utf-8"))
    assert len(preds) == 18 * 31 * 20 * 2
    assert len({p.station_id for p in preds}) == 18 and {p.horizon_min for p in preds} == {60, 120}
    ts = pd.to_datetime([r["ts"] for r in recs])
    assert ts.min() == pd.Timestamp("2026-08-01 05:00+03:00") and ts.max() == pd.Timestamp("2026-09-01 00:00+03:00")
    assert len({p.model_version for p in preds}) == 1 and preds[0].model_version.startswith("baseline_")
