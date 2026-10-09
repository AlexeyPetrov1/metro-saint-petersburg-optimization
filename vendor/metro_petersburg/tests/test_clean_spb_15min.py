import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import config, reference
from src.clean_spb_15min import (FLAGS, OUT_COLS, WEEKDAYS_RU, clean_15min, parse_day_sheet, slot_closure,
                                 slot_times)

TZ = config.SPB_TZ
N = config.SPB_15MIN_SLOTS
VES15 = pd.DataFrame({
    "raw_name_15min": ["А-1", "Б-1"], "vestibule_id": ["a_1", "b_1"], "station_id": ["a", "b"],
    "vestibule_no": [1, 1], "in_hourly": [True, False], "first_hour": [5, 5], "last_hour": [0, 0],
})
NO_INCIDENTS = pd.DataFrame(columns=reference.INCIDENT_COLS)


def _day_values(name: str) -> list[int]:
    """Слоты листа: ночью 01:00–04:59 нули, днём 10 у «А-1» и 4 у «Б-1»."""
    times = slot_times()
    base = 10 if name == "А-1" else 4
    return [0 if t.hour in config.SPB_CLOSED_HOURS else base for t in times]


def _sheet(day: str, values: dict[str, list[int]] | None = None, letter: str | None = None,
           title_day: str | None = None, total_shift: int = 0, header=None) -> tuple[str, list[tuple]]:
    """Лист как у организаторов: заголовок, метки, итог линии по слотам, вестибюли, итог суток."""
    d = pd.Timestamp(day)
    values = values or {n: _day_values(n) for n in VES15.raw_name_15min}
    wd = WEEKDAYS_RU[d.dayofweek]
    name = f"ТВхП {d:%d%m%Y} {letter or wd[0]} 15-мин."
    line = [sum(v[i] for v in values.values()) for i in range(N)]
    total = sum(line)
    rows = [(f"Таблица входных потоков. {d:%d%m%Y} {title_day or wd}",) + (None,) * (N + 2),
            (None, None, *(header or slot_times()), None),
            (None, None, *line, total)]
    rows += [(n, sum(v), *v, None) for n, v in values.items()]
    rows += [(None, total + total_shift) + (None,) * (N + 1)]
    return name, rows


def _raw(days=("2026-04-01", "2026-04-02")) -> pd.DataFrame:
    return pd.concat([parse_day_sheet(*_sheet(d)) for d in days], ignore_index=True)


def _hourly(raw: pd.DataFrame, shift: dict | None = None) -> pd.DataFrame:
    """Часовой файл для «А-1»: сумма 4 слотов (+ сдвиг в отдельных часах), флаги по расписанию."""
    df, _ = clean_15min(raw, VES15, NO_INCIDENTS, _empty_hourly(), _calendar(raw))
    h = (df[df.vestibule_id == "a_1"].assign(ts_utc=lambda d: d.ts_utc.dt.floor("h"))
         .groupby(["vestibule_id", "ts_utc"]).entries.sum().reset_index())
    for ts, delta in (shift or {}).items():
        h.loc[h.ts_utc == pd.Timestamp(ts).tz_localize(TZ).tz_convert("UTC"), "entries"] += delta
    local = h.ts_utc.dt.tz_convert(TZ)
    return h.assign(is_closed_hour=local.dt.hour.isin(config.SPB_CLOSED_HOURS), is_vestibule_closed=False,
                    is_incident=False)


def _empty_hourly() -> pd.DataFrame:
    return pd.DataFrame({"vestibule_id": pd.Series(dtype=str), "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                         "entries": pd.Series(dtype="int64"), "is_closed_hour": pd.Series(dtype=bool),
                         "is_vestibule_closed": pd.Series(dtype=bool), "is_incident": pd.Series(dtype=bool)})


def _calendar(raw: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(raw.sheet_date.min() - pd.Timedelta(days=1), raw.sheet_date.max() + pd.Timedelta(days=1))
    return pd.DataFrame({"date": days, "day_type": "рабочий", "is_regular": True})


# --- Разбор листа ------------------------------------------------------------------
def test_slot_times():
    t = slot_times()
    assert len(t) == 96 and t[0] == dt.time(3, 0) and t[1] == dt.time(3, 15) and t[-1] == dt.time(2, 45)
    assert t[84] == dt.time(0, 0)


def test_parse_sheet():
    raw = parse_day_sheet(*_sheet("2026-04-01"))
    assert len(raw) == 2 * 96 and set(raw.raw_name) == {"А-1", "Б-1"}
    assert (raw.sheet_date == pd.Timestamp("2026-04-01")).all()
    assert raw.groupby("raw_name").entries.sum().to_dict() == {"А-1": 10 * 80, "Б-1": 4 * 80}


@pytest.mark.parametrize("kwargs, msg", [
    ({"letter": "П"}, "буква дня недели"),                # 01.04.2026 — среда
    ({"title_day": "Пятница"}, "первая строка"),
    ({"total_shift": 1}, "итог суток"),
    ({"header": [*slot_times()[:-1], dt.time(3, 0)]}, "в заголовке"),
])
def test_parse_sheet_raises(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        parse_day_sheet(*_sheet("2026-04-01", **kwargs))


def test_parse_sheet_row_total_mismatch_raises():
    name, rows = _sheet("2026-04-01")
    rows[3] = (rows[3][0], rows[3][1] + 1, *rows[3][2:])
    with pytest.raises(ValueError, match="сумма слотов"):
        parse_day_sheet(name, rows)


def test_parse_sheet_line_total_mismatch_raises():
    name, rows = _sheet("2026-04-01")
    rows[2] = (None, None, rows[2][2] + 1, *rows[2][3:])
    with pytest.raises(ValueError, match="итог линии"):
        parse_day_sheet(name, rows)


def test_parse_sheet_extra_column_raises():
    name, rows = _sheet("2026-04-01")
    rows[3] = rows[3] + (5,)
    with pytest.raises(ValueError, match="правее"):
        parse_day_sheet(name, rows)


# --- Очистка -----------------------------------------------------------------------
def test_slots_after_midnight_belong_to_next_day():
    raw = _raw()
    df, _ = clean_15min(raw, VES15, NO_INCIDENTS, _empty_hourly(), _calendar(raw))
    a = df[df.vestibule_id == "a_1"]
    local = a.ts_local.dt.tz_localize(None)
    assert local.min() == pd.Timestamp("2026-04-01 03:00") and local.max() == pd.Timestamp("2026-04-03 02:45")
    assert (local.diff().dropna() == pd.Timedelta(minutes=15)).all()      # отсортировано и без дыр
    # слот 84 листа 01.04 (00:00) — это 02.04 00:00, а сутки метро у него — 01.04
    row = a[local == pd.Timestamp("2026-04-02 00:00")].iloc[0]
    assert row.sday == pd.Timestamp("2026-04-01") and row.hour == 0 and row.quarter == 0
    assert list(df.columns) == OUT_COLS and df.entries.dtype == "int64"


def test_unknown_or_missing_name_raises():
    raw = _raw()
    with pytest.raises(ValueError, match="не сопоставлены"):
        clean_15min(raw.replace({"raw_name": {"Б-1": "В-1"}}), VES15, NO_INCIDENTS, _empty_hourly(), _calendar(raw))


def test_vestibule_missing_on_one_sheet_raises():
    raw = _raw()
    raw = raw[~((raw.raw_name == "Б-1") & (raw.sheet_date == "2026-04-02"))]
    with pytest.raises(ValueError, match="Не на всех листах"):
        clean_15min(raw, VES15, NO_INCIDENTS, _empty_hourly(), _calendar(raw))


def test_flags_from_hourly_and_by_rules():
    raw = _raw()
    hourly = _hourly(raw, shift={"2026-04-01 12:00": -500, "2026-04-01 13:00": 5})
    hourly.loc[hourly.ts_utc == pd.Timestamp("2026-04-01 15:00", tz=TZ).tz_convert("UTC"), "is_vestibule_closed"] = True
    df, recon = clean_15min(raw, VES15, NO_INCIDENTS, hourly, _calendar(raw))
    at = lambda v, t: recon[(recon.vestibule_id == v) & (recon.ts_local == pd.Timestamp(t, tz=TZ))].iloc[0]
    assert at("a_1", "2026-04-01 12:00").is_source_mismatch                # 40 против 540
    assert not at("a_1", "2026-04-01 13:00").is_source_mismatch            # 40 против 35: < 100
    assert at("a_1", "2026-04-01 15:00").is_vestibule_closed               # флаг взят из часового файла
    assert at("a_1", "2026-04-02 02:00").is_closed_hour
    assert at("b_1", "2026-04-02 02:00").is_closed_hour                    # нет в часовом файле — по расписанию
    assert pd.isna(at("b_1", "2026-04-01 12:00").entries_hourly)
    slots = df[(df.vestibule_id == "a_1") & (df.ts_local.dt.tz_localize(None).between("2026-04-01 12:00", "2026-04-01 12:45"))]
    assert len(slots) == 4 and slots.is_source_mismatch.all()              # флаг часа — на всех 4 слотах
    assert set(FLAGS) <= set(df.columns)


def test_closure_rule_for_vestibule_without_hourly():
    vals = {n: _day_values(n) for n in VES15.raw_name_15min}
    vals["Б-1"] = [v * 20 for v in vals["Б-1"]]           # медиана 80 ≥ 50
    days = ["2026-04-01", "2026-04-02", "2026-04-03"]
    sheets = [_sheet(d, vals) for d in days]
    v2 = {k: list(v) for k, v in vals.items()}
    hour10 = [i for i, t in enumerate(slot_times()) if t.hour == 10]
    for i in hour10:
        v2["Б-1"][i] = 0                                    # 10 ч 02.04 — вход закрыт целиком
    sheets[1] = _sheet(days[1], v2)
    raw = pd.concat([parse_day_sheet(*s) for s in sheets], ignore_index=True)
    _, recon = clean_15min(raw, VES15, NO_INCIDENTS, _empty_hourly(), _calendar(raw))
    r = recon[(recon.vestibule_id == "b_1") & (recon.ts_local == pd.Timestamp("2026-04-02 10:00", tz=TZ))].iloc[0]
    assert r.is_vestibule_closed and r.closure_rule == "разово"


def test_slot_closure_rule():
    t = pd.date_range("2026-04-01 08:00", periods=8, freq="15min", tz=TZ)
    days = 5
    out = pd.DataFrame({"vestibule_id": "a_1", "ts_local": np.tile(t, days), "day_type": "рабочий",
                        "is_closed_hour": False, "is_vestibule_closed": False,
                        "entries": np.tile([1000, 900, 800, 700, 60, 60, 60, 60], days)})
    out.loc[1, "entries"] = 6        # 6 ≤ 2 % × 900 = 18 — закрыт
    out.loc[2, "entries"] = 30       # 30 > 16 — открыт
    out.loc[4, "entries"] = 2        # медиана 60 ≥ 50, вход ≤ 2 — закрыт
    out.loc[5, "entries"] = 3        # 3 > max(2; 1,2) — открыт
    c = slot_closure(out)
    assert c.tolist()[:8] == [False, True, False, False, True, False, False, False]
    out["is_vestibule_closed"] = True
    assert not slot_closure(out).any()                     # слоты закрытого часа не рассматриваются


# --- Справочник ----------------------------------------------------------------------
def test_reference_15min_loads():
    ves15 = reference.load_vestibules_15min()
    assert len(ves15) == 24 and ves15.vestibule_id.is_unique and ves15.raw_name_15min.is_unique
    assert ves15.loc[~ves15.in_hourly, "vestibule_id"].tolist() == ["tekhnologichesky_institut_1"]
    assert ves15.station_id.iloc[0] == "devyatkino" and ves15.station_id.iloc[-1] == "prospekt_veteranov"


def test_reference_15min_must_match_hourly_reference(tmp_path):
    df = pd.read_csv(config.SPB_VESTIBULES_15MIN_CSV, dtype=str, keep_default_na=False)
    bad = df.copy()
    bad.loc[bad.vestibule_id == "leninsky_prospekt_2", "last_hour"] = "0"
    p = tmp_path / "v.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="не совпадают"):
        reference.load_vestibules_15min(p)
    bad = df.copy()
    bad.loc[bad.vestibule_id == "avtovo_1", "in_hourly"] = "false"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="in_hourly"):
        reference.load_vestibules_15min(p)


# --- Реальные данные: если interim уже построен -----------------------------------------
@pytest.fixture(scope="module")
def real():
    if not config.SPB_15MIN.exists() or not config.SPB_15MIN_RECON.exists():
        pytest.skip("нет data/interim/spb_line1_15min*.parquet — запустите python -m src.clean_spb_15min")
    return pd.read_parquet(config.SPB_15MIN), pd.read_parquet(config.SPB_15MIN_RECON)


def test_real_grid(real):
    df, _ = real
    assert len(df) == 24 * 120 * 96 == 276_480
    assert not df.duplicated(["vestibule_id", "ts_utc"]).any()
    per_v = df.groupby("vestibule_id").ts_utc
    assert (per_v.size() == 120 * 96).all()
    local = df.ts_local.dt.tz_localize(None)
    assert local.min() == pd.Timestamp("2026-02-01 03:00") and local.max() == pd.Timestamp("2026-10-01 02:45")
    assert set(local.dt.month) == {2, 3, 5, 6, 7, 8, 9, 10}              # 00–02 последних суток — уже следующий месяц
    assert df.entries.sum() == 75_929_175                                 # сумма итогов 120 листов
    assert df.entries.dtype == "int64" and (df.entries >= 0).all()


def test_real_flags(real):
    df, recon = real
    hour = df.ts_local.dt.hour
    assert (df.is_closed_hour == hour.isin(config.SPB_CLOSED_HOURS)).all()
    assert recon.is_source_mismatch.sum() == 31
    slot = recon[recon.is_slot_closed]
    local = slot.ts_local.dt.tz_localize(None)
    balt = slot[(slot.vestibule_id == "baltiyskaya_1") & (local.dt.hour == 8)]
    narv = slot[(slot.vestibule_id == "narvskaya_1") & (local.dt.hour == 8)]
    assert balt.sday.dt.month.eq(2).all() and len(balt) == 19            # все рабочие дни февраля
    assert narv.sday.dt.month.eq(7).all() and len(narv) == 23            # все рабочие дни июля
    ti = df[df.vestibule_id == "tekhnologichesky_institut_1"]
    assert not ti.in_hourly.any() and ti.entries.sum() > 1_000_000


def test_real_reconciliation_is_systematic(real):
    """Σ 4 слотов ≠ часу почти всегда, но 15-минутный источник стабильно выше на ~1,1 % (найдено на этапе 6)."""
    _, recon = real
    r = recon[recon.entries_hourly.notna() & ~recon.is_closed_hour & ~recon.is_vestibule_closed]
    assert (r["diff"] == 0).mean() < 0.05
    excess = r.groupby(r.sday.dt.month).apply(lambda g: g.entries.sum() / g.entries_hourly.sum() - 1, include_groups=False)
    assert excess.between(0.010, 0.013).all()
