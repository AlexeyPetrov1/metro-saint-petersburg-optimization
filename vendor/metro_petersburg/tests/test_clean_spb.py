import numpy as np
import pandas as pd
import pytest

from src import config, reference
from src.clean_spb import (FLOW_HEADER, FOOTER_PREFIX, clean_flow, find_raw, label_to_local, parse_flow_sheet,
                           parse_ts, read_flow_xlsx)

TZ = config.SPB_TZ
FMT = config.SPB_TS_FORMAT
START, END = pd.Timestamp("2026-04-01 00:00"), pd.Timestamp("2026-04-03 23:00")  # три дня
VEST = pd.DataFrame({
    "vestibule_id": ["a_1", "b_1", "b_2"],
    "raw_name": ["А", "Б-1", "Б-2"],
    "station_id": ["a", "b", "b"],
    "vestibule_no": [1, 1, 2],
    "closes_early": [False, False, True],
    "first_hour": [5, 5, 6],
    "last_hour": [0, 0, 21],
})
NO_INCIDENTS = pd.DataFrame(columns=reference.INCIDENT_COLS)


def _label(t: pd.Timestamp) -> str:
    """Метка как у организаторов: часы 00–02 подписаны датой прошедших суток метро."""
    return (t - pd.Timedelta(days=1) if t.hour < config.SPB_PREV_DAY_LABEL_BEFORE else t).strftime(FMT)


def _raw(start=START, end=END) -> pd.DataFrame:
    """Сырые строки как у организаторов: дата строкой, ночью 01–04 нули, днём 100."""
    ts = pd.date_range(start, end, freq="h")
    return pd.DataFrame([{"date_raw": _label(t), "raw_name": n,
                          "entries": 0 if t.hour in config.SPB_CLOSED_HOURS else 100}
                         for n in VEST.raw_name for t in ts])


def _clean(raw, incidents=NO_INCIDENTS):
    return clean_flow(raw, VEST, incidents, START, END)


def _incidents(*rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["vestibule_id", "start_local", "end_local"])
    df[["start_local", "end_local"]] = df[["start_local", "end_local"]].apply(pd.to_datetime)
    return df.assign(incident_id="x", description="", source="")[reference.INCIDENT_COLS]


def _sheet(raw: pd.DataFrame, total=None, max_=None) -> pd.DataFrame:
    """Лист как у организаторов: «Сводка», пустая строка, заголовок, данные, итог."""
    e = raw.entries
    top = [["Сводка", None, None], ["Показатель", "Значение", None],
           ["Минимальное значение", e.min(), None], ["Максимальное значение", e.max() if max_ is None else max_, None],
           ["Среднее значение", f"{e.mean():.2f}", None], [None, None, None], FLOW_HEADER]
    total = e.sum() if total is None else total
    bottom = [[f"{FOOTER_PREFIX} за выбранный период:  {total}", None, None]]
    return pd.DataFrame(top + raw[["date_raw", "raw_name", "entries"]].values.tolist() + bottom)


def _at(df, vestibule, local):
    return df[(df.vestibule_id == vestibule) & (df.ts_local == pd.Timestamp(local).tz_localize(TZ))].iloc[0]


# --- Синтетика: всегда, без данных ------------------------------------------
def test_sheet_summary_and_footer_cut():
    raw = _raw()
    parsed = parse_flow_sheet(_sheet(raw))
    assert len(parsed) == len(raw)
    assert parsed.date_raw.tolist() == raw.date_raw.tolist()
    assert parsed.entries.dtype == "int64" and parsed.entries.sum() == raw.entries.sum()


def test_footer_mismatch_raises():
    raw = _raw()
    with pytest.raises(ValueError, match="итогом файла"):
        parse_flow_sheet(_sheet(raw, total=raw.entries.sum() + 1))


def test_summary_mismatch_raises():
    with pytest.raises(ValueError, match="Сводка"):
        parse_flow_sheet(_sheet(_raw(), max_=999))


@pytest.mark.parametrize("bad", ["2026-01-01 00", "1.1.2026 0", "01.01.2026", "01.01.2026 24", "32.01.2026 00"])
def test_parse_ts_is_strict(bad):
    with pytest.raises(ValueError, match="не в формате"):
        parse_ts(pd.Series(["01.01.2026 00", bad]))


def test_parse_ts_day_first():
    ts = parse_ts(pd.Series(["01.02.2026 05", "01.03.2026 23", "12.01.2026 08"]))
    assert ts.tolist() == [pd.Timestamp("2026-02-01 05:00"), pd.Timestamp("2026-03-01 23:00"),
                           pd.Timestamp("2026-01-12 08:00")]


def test_labels_before_3h_belong_to_next_day():
    """«27.06.2026 01» — это 28.06 01:00, а «28.06.2026 03» — 28.06 03:00: ночь «Алых парусов» непрерывна."""
    lab = parse_ts(pd.Series(["27.06.2026 00", "27.06.2026 01", "27.06.2026 02", "28.06.2026 03", "27.06.2026 23"]))
    assert label_to_local(lab).tolist() == [pd.Timestamp("2026-06-28 00:00"), pd.Timestamp("2026-06-28 01:00"),
                                            pd.Timestamp("2026-06-28 02:00"), pd.Timestamp("2026-06-28 03:00"),
                                            pd.Timestamp("2026-06-27 23:00")]


def test_row_after_grid_end_is_dropped_and_counted():
    raw = _raw()
    extra = pd.DataFrame({"date_raw": _label(END + pd.Timedelta(hours=2)), "raw_name": VEST.raw_name, "entries": 7})
    df = _clean(pd.concat([raw, extra], ignore_index=True))
    assert len(df) == len(raw) and df.attrs["dropped_tail"]["rows"] == 3 and df.attrs["dropped_tail"]["entries"] == 21


def test_row_before_grid_start_raises():
    raw = _raw()
    extra = pd.DataFrame({"date_raw": _label(START - pd.Timedelta(hours=5)), "raw_name": VEST.raw_name, "entries": 7})
    with pytest.raises(ValueError, match="Сетка неполная"):
        _clean(pd.concat([raw, extra], ignore_index=True))


def test_text_sorted_input_gives_sorted_grid():
    """Строки в файле отсортированы как текст (01.04, 02.04, … 01 ч раньше 10 ч) — после очистки порядок по времени."""
    raw = _raw().sort_values("date_raw", ignore_index=True)
    df = _clean(raw)
    assert all(g.ts_utc.is_monotonic_increasing for _, g in df.groupby("vestibule_id"))


def test_grid_and_utc():
    df = _clean(_raw())
    n_hours = int((END - START) / pd.Timedelta(hours=1)) + 1
    assert len(df) == 3 * n_hours and not df.duplicated(["vestibule_id", "ts_utc"]).any()
    assert (df.ts_local.dt.tz_localize(None) - df.ts_utc.dt.tz_localize(None) == pd.Timedelta(hours=3)).all()
    assert df.ts_local.min() == START.tz_localize(TZ) and df.ts_local.max() == END.tz_localize(TZ)
    assert df.station_id.tolist() == df.vestibule_id.map(VEST.set_index("vestibule_id").station_id).tolist()


def test_unknown_name_raises():
    raw = _raw()
    raw.loc[0, "raw_name"] = "Технологический институт"
    with pytest.raises(ValueError, match="не сопоставлены"):
        _clean(raw)


def test_unused_reference_name_raises():
    raw = _raw()
    with pytest.raises(ValueError, match="не сопоставлены"):
        _clean(raw[raw.raw_name != "Б-2"])


def test_incomplete_grid_raises():
    with pytest.raises(ValueError, match="Сетка неполная"):
        _clean(_raw().drop(index=5))


def test_duplicates_raise():
    raw = _raw()
    with pytest.raises(ValueError, match="Дубликаты"):
        _clean(pd.concat([raw, raw.head(1)], ignore_index=True))


def test_closed_hour_flag_ignores_value():
    raw = _raw()
    raw.loc[(raw.raw_name == "А") & (raw.date_raw == "02.04.2026 02"), "entries"] = 5000  # особая ночь
    df = _clean(raw)
    assert (df.is_closed_hour == df.ts_local.dt.hour.isin([1, 2, 3, 4])).all()
    assert _at(df, "a_1", "2026-04-02 02:00").is_closed_hour


def test_vestibule_closed_by_regime_and_one_off():
    raw = _raw()
    one_off = (raw.raw_name == "А") & (raw.date_raw == "02.04.2026 10")
    raw.loc[one_off, "entries"] = 1
    low = (raw.raw_name == "Б-1") & raw.date_raw.str.endswith(" 00")  # малый поток в 00 ч: 0 и 3
    raw.loc[low, "entries"] = [0, 3, 3]
    df = _clean(raw)
    hour = df.ts_local.dt.hour

    b2 = df[df.vestibule_id == "b_2"]
    assert set(hour[b2.index][b2.is_vestibule_closed]) == {5, 22, 23, 0}   # режим 06–21
    assert _at(df, "a_1", "2026-04-02 10:00").is_vestibule_closed          # разово: 1 вход при норме 100
    assert (df.is_vestibule_closed & (df.vestibule_id != "b_2")).sum() == 1
    assert not _at(df, "b_1", "2026-04-01 00:00").is_vestibule_closed      # ноль малого потока — без флага
    assert not (df.is_closed_hour & df.is_vestibule_closed).any()


def test_incident_whole_line_half_open():
    df = _clean(_raw(), _incidents(["*", "2026-04-02 07:00", "2026-04-02 11:00"]))
    inc = df[df.is_incident]
    assert len(inc) == 3 * 4
    assert set(inc.ts_local.dt.hour) == {7, 8, 9, 10}
    assert not _at(df, "a_1", "2026-04-02 11:00").is_incident


def test_incident_single_vestibule():
    df = _clean(_raw(), _incidents(["b_1", "2026-04-02 08:00", "2026-04-02 09:00"]))
    assert df.loc[df.is_incident, "vestibule_id"].tolist() == ["b_1"]


def test_incident_unknown_vestibule_raises():
    with pytest.raises(ValueError, match="неизвестные vestibule_id"):
        _clean(_raw(), _incidents(["zzz", "2026-04-02 08:00", "2026-04-02 09:00"]))


# --- Реальные данные: если interim уже построен -----------------------------
@pytest.fixture(scope="module")
def real():
    if not config.SPB_HOURLY.exists():
        pytest.skip("нет data/interim/spb_line1_hourly.parquet — запустите python -m src.clean_spb")
    return pd.read_parquet(config.SPB_HOURLY)


@pytest.fixture(scope="module")
def vestibules():
    return reference.load_vestibules()


def test_real_grid(real, vestibules):
    n_hours = int((config.SPB_END_UTC - config.SPB_START_UTC) / pd.Timedelta(hours=1)) + 1
    assert n_hours == 6_528
    assert len(real) == 23 * n_hours
    assert not real.duplicated(["vestibule_id", "ts_utc"]).any()
    assert set(real.vestibule_id) == set(vestibules.vestibule_id)
    per_v = real.groupby("vestibule_id").ts_utc
    assert (per_v.size() == n_hours).all()
    assert (per_v.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (per_v.min() == config.SPB_START_UTC).all() and (per_v.max() == config.SPB_END_UTC).all()
    assert real.ts_local.min() == config.SPB_START_LOCAL.tz_localize(TZ)
    assert real.ts_local.max() == config.SPB_END_LOCAL.tz_localize(TZ)
    assert real.entries.dtype == "int64" and (real.entries >= 0).all()


def test_real_names_and_totals(real, vestibules):
    stations = reference.load_stations()
    assert len(vestibules) == 23 and real.station_id.nunique() == 18
    assert set(real.station_id) == set(stations.loc[stations.has_data, "station_id"])
    assert "tekhnologichesky_institut" not in set(real.station_id)
    # строка итога в файле — 164 877 135; 418 входов — метка «30.09 00» (= 01.10 00:00) за концом сетки
    assert real.entries.sum() == 164_877_135 - 418


@pytest.fixture(scope="module")
def line_by_hour(real):
    return real.groupby(real.ts_local.dt.tz_localize(None)).entries.sum()


def test_real_control_days(line_by_hour):
    """1 февраля и 1 марта — воскресенья без утреннего пика, 1 апреля — среда с двумя пиками."""
    for day in ("2026-02-01", "2026-03-01"):
        p = line_by_hour[day].groupby(lambda t: t.hour).sum()
        assert pd.Timestamp(day).dayofweek == 6
        assert p.loc[7:9].max() < p.loc[12:17].max()
    p = line_by_hour["2026-04-01"].groupby(lambda t: t.hour).sum()
    assert pd.Timestamp("2026-04-01").dayofweek == 2
    assert p.loc[7:9].max() > 1.5 * p[12] and p.loc[17:19].max() > 1.5 * p[12]


def test_real_dates_in_right_month(line_by_hour):
    """Если день или месяц перепутаны, будний профиль окажется на выходном: 08 ч / 12 ч > 1,2 ⇔ рабочий день."""
    path = config.CALENDAR_OUT
    if not path.exists():
        pytest.skip("нет календаря в interim")
    cal = pd.read_parquet(path).assign(date=lambda d: pd.to_datetime(d.date)).set_index("date")
    h = line_by_hour.index.hour
    ratio = (line_by_hour[h == 8].set_axis(line_by_hour.index[h == 8].normalize())
             / line_by_hour[h == 12].set_axis(line_by_hour.index[h == 12].normalize()))
    workday = cal.is_workday.reindex(ratio.index)
    assert len(ratio) == 272 and workday.notna().all()
    assert ((ratio > 1.2) == workday).all(), ratio[(ratio > 1.2) != workday]
    per_month = line_by_hour.groupby(line_by_hour.index.month).size()
    assert per_month.to_dict() == {1: 741, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 699}


def test_real_raw_dates_round_trip():
    """Сырые строки «ДД.ММ.ГГГГ ЧЧ»: день и месяц берутся из своих позиций."""
    found = find_raw(config.SPB_FLOW_FILE)
    if len(found) != 1:
        pytest.skip("нет файла организаторов в data/raw/spb")
    raw = read_flow_xlsx(found[0])
    ts = parse_ts(raw.date_raw)
    assert (ts.dt.day == raw.date_raw.str[:2].astype(int)).all()
    assert (ts.dt.month == raw.date_raw.str[3:5].astype(int)).all()
    assert set(raw.raw_name) == set(reference.load_vestibules().raw_name)


def test_real_flags(real):
    hour = real.ts_local.dt.hour
    local = real.ts_local.dt.tz_localize(None)
    assert (real.is_closed_hour == hour.isin([1, 2, 3, 4])).all()

    parade = real[(real.vestibule_id == "vosstaniya_1") & (local.dt.normalize() == "2026-05-09")]
    assert set(parade.loc[parade.is_vestibule_closed, "ts_local"].dt.hour) == set(range(7, 13))

    lp2 = real[real.vestibule_id == "leninsky_prospekt_2"]
    assert set(hour[lp2.index][lp2.is_vestibule_closed]) == {5, 22, 23, 0}
    assert lp2.is_vestibule_closed.sum() == 4 * 272

    inc = real[real.is_incident]
    assert len(inc) == 23 * 4
    assert set(local[inc.index]) == set(pd.date_range("2026-08-31 07:00", "2026-08-31 10:00", freq="h"))

    # нули без флага — только малый поток в 00 ч на конечных и в 22 ч на пл.Ленина-2
    free = real[(real.entries == 0) & ~(real.is_closed_hour | real.is_vestibule_closed | real.is_incident)]
    assert set(zip(free.vestibule_id, hour[free.index])) <= {
        ("devyatkino_1", 0), ("devyatkino_2", 0), ("prospekt_veteranov_1", 0), ("prospekt_veteranov_2", 0),
        ("ploshchad_lenina_2", 22)}


def test_real_special_nights_are_events(line_by_hour):
    """Все часы с ночными входами по линии > 100 покрыты событиями «по данным потока» и наоборот."""
    ev = reference.load_events()
    nights = ev[ev.source == "по данным потока"]
    assert len(nights) == 4
    assert nights.set_index("event").start_local["Алые паруса"] == pd.Timestamp("2026-06-28 01:00")
    closed = line_by_hour[line_by_hour.index.hour.isin(config.SPB_CLOSED_HOURS)]
    busy = closed[closed > 100].index
    covered = lambda t: ((nights.start_local <= t) & (t < nights.end_local)).any()
    assert all(covered(t) for t in busy)
    for e in nights.itertuples():   # в каждом окне есть часы с ночными входами
        assert (line_by_hour[e.start_local:e.end_local - pd.Timedelta(hours=1)] > 100).any(), e


# --- Календарь, погода, МО-I ------------------------------------------------
def test_real_calendar():
    if not config.CALENDAR_OUT.exists():
        pytest.skip("нет календаря в interim")
    cal = pd.read_parquet(config.CALENDAR_OUT).assign(date=lambda d: d.date.astype(str)).set_index("date")
    assert len(cal) == 365
    assert set(cal.index[cal.is_transfer_dayoff]) == {"2026-01-09", "2026-03-09", "2026-05-11", "2026-12-31"}
    assert not cal.loc["2026-05-09"].is_workday and cal.loc["2026-05-09"].holiday_name == "День Победы"
    assert cal.loc["2026-04-30"].is_shortened and cal.loc["2026-04-30"].is_workday


@pytest.mark.parametrize("kind", list(config.WEATHER_URLS))
def test_real_weather(kind):
    path = config.INTERIM / f"weather_{kind}_spb.parquet"
    if not path.exists():
        pytest.skip(f"нет {path.name}")
    w = pd.read_parquet(path)
    assert len(w) == 6_528 and w.ts_utc.is_unique
    assert w.ts_utc.min() == config.SPB_START_UTC and w.ts_utc.max() == config.SPB_END_UTC
    assert not w.drop(columns=["ts_utc", "ts_local"]).isna().any().any()


def test_real_mo1():
    if not config.MO1_OUT.exists():
        pytest.skip("нет mo1_reports.parquet")
    mo1 = pd.read_parquet(config.MO1_OUT)
    assert set(mo1.period) == {"2025", "2026-Q2", "2026-Q3"}
    assert set(mo1.line) == {"1", "2", "3", "4", "5", "6", "total"}
    assert mo1.groupby("period").size().nunique() == 1
    l1 = mo1[mo1.line == "1"].set_index(["period", "indicator"]).value
    assert l1["2025", "Поездов по факту"] == 286_642
    assert (l1[:, "Максимальная парность"] == 32).all()
    assert np.isclose(l1["2025", "Общие ваг/км"], 68_508_622.71)
    total = mo1[(mo1.line == "total") & (mo1.indicator == "Максимальная парность")]
    assert total.value.isna().all()  # «-» в файле
