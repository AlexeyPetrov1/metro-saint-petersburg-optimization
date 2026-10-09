import numpy as np
import pandas as pd
import pytest

from src import config, eda

TZ = config.TZ_LOCAL


def _frame(n_stations: int = 3, days: int = 3, value: float = 100.0) -> pd.DataFrame:
    """Мини-mta: станции подряд по линии, постоянный поток, все колонки, которые нужны правилу пропусков."""
    ts = pd.date_range(pd.Timestamp("2024-06-04 00:00", tz=TZ), periods=24 * days, freq="h").tz_convert("UTC")
    rows = [{"station_id": f"s{i}", "line_order": i, "ts_utc": t} for i in range(n_stations) for t in ts]
    m = pd.DataFrame(rows)
    m["ts_local"] = m.ts_utc.dt.tz_convert(TZ)
    local = m.ts_local.dt.tz_localize(None)
    m["date"], m["hour"], m["dow"] = local.dt.normalize(), local.dt.hour, local.dt.dayofweek
    m["day_type"] = eda.day_type(m.date, set())
    m["entries"] = value
    m["is_missing"] = False
    return m


def _set(m, station, local_from, local_to, entries):
    t = m.ts_local.dt.tz_localize(None)
    mask = (m.station_id == station) & t.between(pd.Timestamp(local_from), pd.Timestamp(local_to))
    m.loc[mask, "entries"] = entries
    m.loc[mask, "is_missing"] = np.isnan(entries)
    return m


def _label(lab, m, station, local):
    t = m.ts_local.dt.tz_localize(None)
    return lab[(m.station_id == station) & (t == pd.Timestamp(local))].iloc[0]


def test_short_night_gap_alone_is_zero():
    m = _set(_frame(), "s1", "2024-06-05 03:00", "2024-06-05 03:00", np.nan)
    lab = eda.gap_rule(m)
    assert _label(lab, m, "s1", "2024-06-05 03:00") == "ноль"
    assert (lab != "").sum() == 1


def test_night_gap_with_neighbor_is_closure():
    m = _frame()
    for s in ("s1", "s2"):
        m = _set(m, s, "2024-06-05 02:00", "2024-06-05 03:00", np.nan)
    lab = eda.gap_rule(m)
    assert _label(lab, m, "s1", "2024-06-05 02:00") == "закрытие"
    assert _label(lab, m, "s2", "2024-06-05 03:00") == "закрытие"


def test_long_night_gap_is_closure():
    m = _set(_frame(), "s0", "2024-06-05 01:00", "2024-06-05 04:00", np.nan)  # 4 ч > 3
    assert set(eda.gap_rule(m)[m.is_missing]) == {"закрытие"}


def test_day_gap_is_closure_and_low_edges_join_it():
    m = _set(_frame(), "s0", "2024-06-05 10:00", "2024-06-05 15:00", np.nan)
    m = _set(m, "s0", "2024-06-05 08:00", "2024-06-05 09:00", 10.0)   # 10 % нормы — край
    m = _set(m, "s0", "2024-06-05 16:00", "2024-06-05 16:00", 50.0)   # 50 % нормы — уже не край
    m = _set(m, "s2", "2024-06-05 12:00", "2024-06-05 12:00", 5.0)    # низкий час без закрытия рядом
    lab = eda.gap_rule(m)
    assert _label(lab, m, "s0", "2024-06-05 12:00") == "закрытие"
    assert _label(lab, m, "s0", "2024-06-05 08:00") == "край закрытия"
    assert _label(lab, m, "s0", "2024-06-05 09:00") == "край закрытия"
    assert _label(lab, m, "s0", "2024-06-05 16:00") == ""
    assert _label(lab, m, "s0", "2024-06-05 07:00") == ""
    assert _label(lab, m, "s2", "2024-06-05 12:00") == ""


def test_day_type():
    dates = pd.Series(pd.to_datetime(["2023-11-22", "2023-11-23", "2023-11-25", "2023-11-26"]))
    out = eda.day_type(dates, {pd.Timestamp("2023-11-23")})
    assert out.tolist() == ["будни", "праздник", "суббота", "воскресенье"]


def test_weighted_quantile():
    v, w = pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([1.0, 1.0, 1.0, 97.0])
    assert eda.weighted_quantile(v, w, 0.5) == 4.0
    assert eda.weighted_quantile(v, pd.Series([1.0] * 4), 0.5) == 2.0


def test_holiday_ru():
    assert eda.holiday_ru("New Year's Day (observed)") == "Новый год (перенос)"
    assert eda.holiday_ru("Christmas Eve / New Year's Eve") == "Сочельник / Канун Нового года"


# --- Реальные данные ----------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    if not (config.INTERIM / "mta_line7_hourly.parquet").exists():
        pytest.skip("нет data/interim — запустите python -m src.download && python -m src.clean")
    return eda.load()


def test_load_excludes_dst_rows(data):
    full = pd.read_parquet(config.INTERIM / "mta_line7_hourly.parquet", columns=["dst_flag"])
    assert len(data.mta) == (full.dst_flag == "").sum() == len(full) - 4 * 22
    dst_hours = pd.to_datetime(["2023-11-05 05:00", "2023-11-05 06:00", "2024-11-03 05:00", "2024-11-03 06:00"], utc=True)
    assert not data.mta.ts_utc.isin(dst_hours).any()


def test_station_groups(data):
    counts = data.stations.group.value_counts()
    assert counts.to_dict() == {"Остальные Квинса": 15, "Манхэттен": 4, "Пересадочные Квинса": 3}


def test_events_manual(data):
    us = data.events[data.events.event.str.startswith("US Open")]
    assert len(us) == 2 and not us.verified.any()
    assert pd.Timestamp("2024-09-01") in eda.event_dates(data.events, "US Open")
