import numpy as np
import pandas as pd
import pytest

from src import config
from src.clean import DST_ABSORBED, DST_MERGED, MTA_OUT, clean_mta

TZ = config.TZ_LOCAL
SPRING = ("2023-03-11 00:00", "2023-03-13 23:00")
FALL = ("2023-11-04 00:00", "2023-11-06 23:00")


def _bounds(window):
    start, end = (pd.Timestamp(t).tz_localize(TZ).tz_convert("UTC") for t in window)
    return start, end


def _mta_like_raw(window, stations=("A", "B")) -> pd.DataFrame:
    """Сырые строки как у MTA: местное время без смещения, весной нет 02:00, осенью один 01:00."""
    start, end = _bounds(window)
    labels = pd.date_range(start, end, freq="h").tz_convert(TZ).tz_localize(None).unique()
    return pd.DataFrame([
        {"transit_timestamp": t.strftime("%Y-%m-%dT%H:%M:%S.000"), "station_complex_id": s,
         "entries": float(10 + i), "transfers": 1.0}
        for s in stations for i, t in enumerate(labels)
    ])


def _clean(raw, window, stations=("A", "B")):
    start, end = _bounds(window)
    return clean_mta(raw, list(stations), start, end)


def _assert_invariants(df: pd.DataFrame, n_hours: int, n_stations: int) -> None:
    assert not df.duplicated(["station_id", "ts_utc"]).any()
    assert len(df) == n_hours * n_stations
    per_station = df.groupby("station_id").ts_utc
    assert (per_station.size() == n_hours).all()
    first = df[df.station_id == df.station_id.iloc[0]].ts_utc.reset_index(drop=True)
    assert all(g.reset_index(drop=True).equals(first) for _, g in per_station)  # одинаковая сетка у всех
    assert per_station.apply(lambda s: s.is_monotonic_increasing and s.is_unique).all()
    assert (per_station.diff().dropna() == pd.Timedelta(hours=1)).all()


# --- Синтетика: всегда, без сети -------------------------------------------
@pytest.mark.parametrize("window", [SPRING, FALL], ids=["spring", "fall"])
def test_grid_invariants(window):
    df = _clean(_mta_like_raw(window), window)
    start, end = _bounds(window)
    n_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    _assert_invariants(df, n_hours, 2)
    assert df.ts_utc.min() == start and df.ts_utc.max() == end


def test_missing_is_nan_and_zero_stays_zero():
    raw = _mta_like_raw(SPRING)
    gap = (raw.station_complex_id == "B") & (raw.transit_timestamp == "2023-03-12T10:00:00.000")
    zero = (raw.station_complex_id == "A") & (raw.transit_timestamp == "2023-03-12T04:00:00.000")
    raw.loc[zero, "entries"] = 0.0
    df = _clean(raw[~gap], SPRING)

    at = lambda s, t: df[(df.station_id == s) & (df.ts_local == pd.Timestamp(t).tz_localize(TZ))].iloc[0]
    g, z = at("B", "2023-03-12 10:00"), at("A", "2023-03-12 04:00")
    assert np.isnan(g.entries) and g.is_missing
    assert z.entries == 0 and not z.is_missing
    assert df.is_missing.sum() == 1


def test_spring_day_has_23_hours_without_0200():
    df = _clean(_mta_like_raw(SPRING), SPRING)
    a = df[df.station_id == "A"]
    day = a[a.ts_local.dt.date == pd.Timestamp("2023-03-12").date()]
    assert len(day) == 23
    assert 2 not in set(day.ts_local.dt.hour)
    assert (a.dst_flag == "").all()
    assert not a.is_missing.any()


def test_fall_merged_and_absorbed():
    df = _clean(_mta_like_raw(FALL), FALL)
    a = df[df.station_id == "A"].set_index("ts_utc")
    day = a[a.ts_local.dt.date == pd.Timestamp("2023-11-05").date()]
    assert len(day) == 25

    merged = a.loc[pd.Timestamp("2023-11-05 05:00", tz="UTC")]   # 01:00 EDT
    absorbed = a.loc[pd.Timestamp("2023-11-05 06:00", tz="UTC")]  # 01:00 EST
    assert merged.dst_flag == DST_MERGED and not np.isnan(merged.entries) and not merged.is_missing
    assert absorbed.dst_flag == DST_ABSORBED and np.isnan(absorbed.entries) and absorbed.is_missing
    assert (a.dst_flag != "").sum() == 2


def test_nonexistent_local_hour_raises():
    raw = _mta_like_raw(SPRING)
    raw.loc[0, "transit_timestamp"] = "2023-03-12T02:00:00.000"
    with pytest.raises(ValueError, match="несуществующий"):
        _clean(raw, SPRING)


def test_duplicates_in_raw_raise():
    raw = _mta_like_raw(SPRING)
    with pytest.raises(ValueError, match="Дубликаты"):
        _clean(pd.concat([raw, raw.head(1)]), SPRING)


# --- Реальные данные: если interim уже построен -----------------------------
@pytest.fixture(scope="module")
def real():
    if not MTA_OUT.exists():
        pytest.skip("нет data/interim — запустите python -m src.download && python -m src.clean")
    return pd.read_parquet(MTA_OUT)


def test_real_grid_invariants(real):
    n_hours = int((config.END_UTC - config.START_UTC) / pd.Timedelta(hours=1)) + 1
    assert n_hours == 17_544
    assert set(real.station_id) == set(config.LINE7_IDS)
    _assert_invariants(real, n_hours, len(config.LINE7_IDS))
    assert real.ts_utc.min() == config.START_UTC and real.ts_utc.max() == config.END_UTC
    assert real.ts_local.min() == config.START_LOCAL.tz_localize(TZ)
    assert real.ts_local.max() == config.END_LOCAL.tz_localize(TZ)


def test_real_dst_flags(real):
    counts = real.groupby("station_id").dst_flag.value_counts().unstack(fill_value=0)
    assert (counts[DST_MERGED] == 2).all() and (counts[DST_ABSORBED] == 2).all()
    assert real.loc[real.dst_flag == DST_ABSORBED, "entries"].isna().all()


def test_real_missing_is_nan(real):
    assert (real.is_missing == real.entries.isna()).all()
    assert (real.entries.dropna() >= 0).all()


def test_real_stations():
    path = config.INTERIM / "stations.parquet"
    if not path.exists():
        pytest.skip("нет data/interim/stations.parquet")
    st = pd.read_parquet(path)
    assert len(st) == 22 and st.station_id.is_unique
    assert set(st.loc[st.is_transfer_complex, "station_id"]) == {"616", "461", "606", "610", "609", "611"}
    assert st.line_order.tolist() == list(range(22))
