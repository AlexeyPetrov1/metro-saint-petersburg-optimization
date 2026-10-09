import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import baseline as B
from src import config
from src import eda_spb as E

START = pd.Timestamp("2026-03-02 00:00")   # понедельник
WEEKS = 10


def _synthetic(holiday_week: int = 8):
    """Один вестибюль, 10 недель. Будни: 100 + 10·день недели + номер недели; воскресенье: 500 + номер недели.
    Понедельник недели holiday_week — праздник."""
    ts = pd.date_range(START, periods=24 * 7 * WEEKS, freq="h").tz_localize(config.SPB_TZ).tz_convert("UTC")
    local = ts.tz_convert(config.SPB_TZ).tz_localize(None)
    sday = (local - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).normalize()
    week = ((sday - START).days // 7).to_numpy()
    dow = sday.dayofweek.to_numpy()
    entries = np.where(dow == 6, 500 + week, 100 + 10 * dow + week).astype(float)[:, None]
    dates = pd.date_range(START - pd.Timedelta(days=1), START + pd.Timedelta(weeks=WEEKS), freq="D")
    hol = START + pd.Timedelta(weeks=holiday_week)
    cal = pd.DataFrame({"date": dates})
    cal["day_type"] = np.select([cal.date == hol, cal.date.dt.dayofweek < 5, cal.date.dt.dayofweek == 5],
                                ["праздник", "рабочий", "суббота"], "воскресенье")
    cal["is_regular"] = cal.date != hol
    closed = np.isin(local.hour, config.SPB_CLOSED_HOURS)[:, None]
    regular = cal.set_index("date").is_regular.reindex(sday).eq(True).to_numpy()
    return E.make_grid(ts, np.where(closed, 0, entries), closed, closed, regular), cal


def _at(g, day: str, hour: int = 8) -> int:
    return int(np.flatnonzero((g.sday == pd.Timestamp(day)) & (g.hour == hour))[0])


def test_norm_skips_holiday_in_history():
    g, cal = _synthetic()
    b = B.norm(g, cal)
    monday9 = START + pd.Timedelta(weeks=9)
    # понедельник недели 8 — праздник, его пропускаем: берутся недели 7, 6, 5, 4 → медиана 105,5
    assert b[_at(g, str(monday9.date())), 0] == pytest.approx(105.5)


def test_holiday_uses_last_sundays():
    g, cal = _synthetic()
    b = B.norm(g, cal)
    holiday = START + pd.Timedelta(weeks=8)
    # праздник в понедельник — 4 прошлых воскресенья: недели 7, 6, 5, 4 → 500 + 5,5
    assert b[_at(g, str(holiday.date())), 0] == pytest.approx(505.5)


def test_sunday_lag():
    days = pd.DatetimeIndex(["2026-03-02", "2026-03-07", "2026-03-08"])   # пн, сб, вс
    assert B.sunday_lag(days).tolist() == [24, 144, 168]


def test_level_factor_uses_only_past_days():
    g, cal = _synthetic()
    b = B.norm(g, cal)
    lf = B.level_factor(g, cal, b)
    day = START + pd.Timedelta(weeks=7)
    entries = g.entries.copy()
    entries[g.sday >= day] *= 2                       # с этих суток поток вдвое выше
    regular = cal.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    g2 = E.make_grid(g.ts, entries, g.closed, g.flagged, regular)
    lf2 = B.level_factor(g2, cal, B.norm(g2, cal))
    first = g.sday == day
    np.testing.assert_allclose(lf2[first], lf[first])            # в первые сутки скачка фактор ещё старый
    assert lf2[g.sday == day + pd.Timedelta(days=4)].mean() > lf[g.sday == day + pd.Timedelta(days=4)].mean()


@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists()):
        pytest.skip("нет interim СПб — запустите python -m src.clean_spb")
    return E.load()


def test_norms_do_not_look_ahead(spb):
    """Норма и входы в строке t не меняются, если удалить поток после конца часа t."""
    g = spb.grid
    t0 = int(np.flatnonzero((g.sday == pd.Timestamp("2026-07-08")) & (g.hour == 9))[0])
    entries = g.entries.copy()
    entries[t0 + 1:] = np.nan
    regular = spb.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    cut = E.make_grid(g.ts, entries, g.closed, g.flagged, regular)
    full, trunc = B.norm_matrices(g, spb.calendar), B.norm_matrices(cut, spb.calendar)
    for name in B.NORMS:
        np.testing.assert_allclose(full[name][:t0 + 1], trunc[name][:t0 + 1], equal_nan=True, err_msg=name)
    r_full, r_cut = B.observed_ratio(g, full["b4"]), B.observed_ratio(cut, trunc["b4"])
    np.testing.assert_allclose(r_full[:t0 + 1], r_cut[:t0 + 1], equal_nan=True)
    e_full, e_cut = E.ewm_ratio(r_full, g.ts, 2), E.ewm_ratio(r_cut, g.ts, 2)
    np.testing.assert_allclose(e_full[:t0 + 1], e_cut[:t0 + 1], equal_nan=True)


def test_real_holiday_norm_is_sunday_like(spb):
    """23.02 (понедельник, праздник): норма — по воскресеньям, а не по понедельникам."""
    n = B.norm_matrices(spb.grid, spb.calendar)
    g = spb.grid
    rows = (g.sday == pd.Timestamp("2026-02-23")) & np.isin(g.hour, range(7, 22))
    fact, b4, old = np.nansum(g.y[rows]), np.nansum(n["b4"][rows]), np.nansum(g.b[rows])
    assert abs(fact / b4 - 1) < 0.15 and fact / old < 0.7
