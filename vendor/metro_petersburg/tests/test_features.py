import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import config
from src import eda_spb as E
from src import features as F

# признаки даты, месяца, номера дня или недели: в тестовом месяце они принимают невиданные значения
FORBIDDEN = re.compile(r"month|date|doy|dayofyear|day_of_year|week|year|sday|time|^ts$|^t$|^tau$")
CUTS = ["2026-07-08 09:00",    # обычный будний час
        "2026-08-31 08:00",    # инцидент: часы 09–10 под флагом, но в момент t об этом ещё неизвестно
        "2026-06-15 04:00"]    # ночь: τ = 05 и 06 — уже следующие сутки метро


def test_groups_from_eda_decisions():
    assert list(F.GROUPS) == ["база", "ядро динамики", "кросс", "скорость и ускорение", "вчера", "погода",
                              "особые дни"]
    assert F.ABLATION == list(F.GROUPS)[1:] and len(F.ABLATION) == 6
    assert len(F.ALL) == len(set(F.ALL))                    # группы не пересекаются
    assert "is_holiday" not in F.ALL                        # его несёт «тип дня τ» базы
    assert set(F.TITLES) == set(F.ALL)
    assert set(F.CATEGORICAL) <= set(F.GROUPS["база"])


def test_no_date_feature_names():
    assert not [f for f in F.ALL if FORBIDDEN.search(f)]


def test_cumulative_resets_at_service_day_start():
    ts = pd.date_range("2026-03-02 03:00", periods=48, freq="h")
    sday = (ts - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).normalize()
    y, f = np.full((48, 1), 30.0), np.full((48, 1), 20.0)
    y[ts.hour == 5] = 60.0
    cum = F._cumulative(y, f, sday)
    first5 = int(np.flatnonzero(ts.hour == 5)[0])
    assert cum[first5, 0] == pytest.approx(3.0)                       # с 05:00 — новые сутки: 60 / 20
    assert cum[first5 + 1, 0] == pytest.approx((60 + 30) / 40)
    assert cum[first5 - 1, 0] == pytest.approx(1.5)                   # 04:00 — ещё прошлые сутки: 30·k / 20·k


@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists()):
        pytest.skip("нет interim СПб — запустите python -m src.clean_spb")
    d = E.load()
    return d, BT.build_panel(d)


@pytest.fixture(scope="module")
def feats(spb):
    _, panel = spb
    return F.add_features(BT.make_rows(panel), panel)


def _rows_at(panel: BT.Panel, t: pd.Timestamp) -> pd.DataFrame:
    rows = BT.make_rows(panel, for_export=True)
    rows = rows[rows.t == t]
    return F.add_features(rows, panel)[["vestibule_id", *F.ALL]].reset_index(drop=True)


@pytest.mark.parametrize("cut", CUTS)
def test_features_do_not_look_ahead(spb, cut):
    """Признаки строки t не меняются, если удалить поток после конца часа t и снять будущие флаги инцидента."""
    d, panel = spb
    g = d.grid
    t0 = int(np.flatnonzero(g.local == pd.Timestamp(cut))[0])
    after = (np.arange(len(g.ts)) > t0)[:, None]
    entries = np.where(after, np.nan, g.entries)
    flagged = np.where(after, g.closed, g.flagged)
    regular = d.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    cut_panel = BT.build_panel(dataclasses.replace(d, grid=E.make_grid(g.ts, entries, g.closed, flagged, regular)))
    full, trunc = _rows_at(panel, g.ts[t0]), _rows_at(cut_panel, g.ts[t0])
    assert len(full) >= 20 and set(full.h) == {1, 2}
    pd.testing.assert_frame_equal(full, trunc)


def test_no_date_features_in_matrix(feats):
    X = feats[F.ALL]
    assert not [c for c in X if pd.api.types.is_datetime64_any_dtype(X[c])]
    hours = (feats.tau - feats.tau.min()).dt.total_seconds()
    sample = np.random.default_rng(0).choice(len(X), 30_000, replace=False)
    for c in X.columns.difference(F.CATEGORICAL):
        rho = X[c].iloc[sample].corr(hours.iloc[sample], method="spearman")
        assert np.isnan(rho) or abs(rho) < 0.8, f"{c} монотонно связан со временем: ρ = {rho:.2f}"


def test_twin_and_line_features(feats):
    """r второго вестибюля — это r_t соседа по станции; r линии одинаков у всех вестибюлей в час t."""
    key = ["t", "h"]
    a = feats[feats.vestibule_id == "devyatkino_1"].set_index(key)
    b = feats[feats.vestibule_id == "devyatkino_2"].set_index(key)
    common = a.index.intersection(b.index)
    np.testing.assert_allclose(a.loc[common, "r_twin"], b.loc[common, "r_t"], equal_nan=True)
    assert feats[feats.vestibule_id == "lesnaya_1"].r_twin.isna().all()
    assert (feats.groupby("t").r_line.nunique(dropna=True) <= 1).all()


def test_features_keep_backtest_columns(spb, feats):
    """Признаки добавляются рядом с колонками бэктеста и не подменяют их (r_b4_t читает бейзлайн «b × r(t)»)."""
    _, panel = spb
    rows = BT.make_rows(panel)
    pd.testing.assert_frame_equal(feats[rows.columns], rows)
    with pytest.raises(ValueError):
        F.add_features(rows.assign(r_t=1.0), panel)


def test_holiday_is_day_type(feats):
    days = set(feats.loc[feats.day_type_tau == "праздник", "sday"].dt.strftime("%d.%m"))
    assert days == {"23.02", "08.03", "09.03", "01.05", "09.05", "11.05", "12.06"}
