"""Очистка данных 1 линии СПб: data/raw → data/interim, всё выровнено по ts_utc (раздел 3 ТЗ).

Запуск: python -m src.clean_spb

Поток («Пассажиропоток 2026 Линия 1.xlsx») → spb_line1_hourly.parquet: полная сетка «вестибюль × час» в UTC,
vestibule_id, station_id, ts_utc, ts_local, entries, is_closed_hour, is_vestibule_closed, is_incident.
Строки с любым из трёх флагов исключаются из обучения и метрик; флаги могут пересекаться.
- is_closed_hour — метро закрыто по расписанию (01–04 местного), независимо от значения. В особые ночи метро
  работало, но ночные часы мы не прогнозируем; сами ночи записаны в events_spb.csv.
- is_vestibule_closed — в часы работы метро вестибюль закрыт: по режиму из справочника (first_hour…last_hour)
  или разово — вход ≤ SPB_CLOSURE_MAX_ENTRIES при медиане вестибюля и часа ≥ SPB_CLOSURE_MIN_NORM.
  Нули малого потока (00 ч на конечных) — настоящие значения, флага у них нет.
- is_incident — ручная разметка из incidents.csv, интервал [start_local, end_local); "*" — вся линия.

Часы 00–02 в файле подписаны датой прошедших суток метро (config.SPB_PREV_DAY_LABEL_BEFORE): метка «D 01» — это D+1 01:00.

Ещё: погода СПб (weather_*_spb), производственный календарь (calendar_ru_2026), отчёты МО-I (mo1_reports).
"""
import fnmatch
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src import config, reference
from src.clean import build_grid, clean_weather
from src.reference import nfc, service_pos

FLOW_HEADER = ["Дата", "Станция", "Количество пассажиров"]
FOOTER_PREFIX = "Общее количество пассажиров"
SUMMARY_KEYS = {"Минимальное значение": "min", "Максимальное значение": "max", "Среднее значение": "mean"}
OUT_COLS = ["vestibule_id", "station_id", "ts_utc", "ts_local", "entries",
            "is_closed_hour", "is_vestibule_closed", "is_incident"]
REGIME, ONE_OFF = "режим", "разово"


# --- Сырые файлы --------------------------------------------------------------
def find_raw(pattern: str, root: Path = config.SPB_RAW) -> list[Path]:
    """Файлы организаторов по маске имени в любой подпапке; имена сравниваются в NFC."""
    pat = nfc(pattern)
    return sorted(p for p in root.rglob("*") if p.is_file() and fnmatch.fnmatchcase(nfc(p.name), pat))


def find_one(pattern: str, root: Path = config.SPB_RAW) -> Path:
    found = find_raw(pattern, root)
    if len(found) != 1:
        raise FileNotFoundError(f"В {root} ожидался ровно один файл «{pattern}», найдено: {[str(p) for p in found]}")
    return found[0]


# --- Поток ----------------------------------------------------------------------
def read_flow_xlsx(path: Path) -> pd.DataFrame:
    sheet = pd.read_excel(path, header=None, dtype=object, engine="openpyxl")
    return parse_flow_sheet(sheet)


def parse_flow_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """Лист как есть → date_raw, raw_name, entries.

    Отрезает блок «Сводка» сверху и строку «Общее количество пассажиров…» снизу и сверяет с ними данные:
    итог = сумме, min/max/mean = сводке.
    """
    cells = sheet.iloc[:, :3].set_axis(["a", "b", "c"], axis=1).reset_index(drop=True)
    is_header = (cells.a == FLOW_HEADER[0]) & (cells.b == FLOW_HEADER[1]) & (cells.c == FLOW_HEADER[2])
    if is_header.sum() != 1:
        raise ValueError(f"Строка-заголовок {FLOW_HEADER} найдена {is_header.sum()} раз, ожидалась 1")
    h = int(np.flatnonzero(is_header)[0])

    body = cells.iloc[h + 1:]
    stop = body.b.isna()
    n_data = int(np.argmax(stop.to_numpy())) if stop.any() else len(body)
    data, tail = body.iloc[:n_data], body.iloc[n_data:]
    tail_text = tail.a.dropna().astype(str).tolist()
    if not tail_text or not tail_text[0].startswith(FOOTER_PREFIX):
        raise ValueError(f"После данных ожидалась строка «{FOOTER_PREFIX}…», а там {tail_text[:1]}")
    if len(tail_text) > 1 or tail[["b", "c"]].notna().any().any():
        raise ValueError(f"После строки итога ещё есть данные: {tail_text[1:3]}")

    entries = pd.to_numeric(data.c, errors="coerce")
    bad = entries.isna() | (entries < 0) | (entries % 1 != 0)
    if bad.any():
        raise ValueError(f"Входы должны быть целыми ≥ 0: {data.loc[bad].head().values.tolist()}")
    entries = entries.astype("int64")
    total = int(re.sub(r"\D", "", tail_text[0].split(":")[-1]))
    if entries.sum() != total:
        raise ValueError(f"Сумма входов {entries.sum():,} не совпадает с итогом файла {total:,}")

    summary = {SUMMARY_KEYS[k]: float(v) for k, v in zip(cells.a.iloc[:h], cells.b.iloc[:h]) if k in SUMMARY_KEYS}
    actual = {"min": entries.min(), "max": entries.max(), "mean": round(entries.mean(), 2)}
    off = {k: (v, actual[k]) for k, v in summary.items() if abs(v - actual[k]) > 0.005}
    if off:
        raise ValueError(f"«Сводка» не совпадает с данными (в файле, по данным): {off}")

    return pd.DataFrame({"date_raw": data.a.astype(str).to_numpy(),
                         "raw_name": data.b.astype(str).map(nfc).to_numpy(),
                         "entries": entries.to_numpy()})


def parse_ts(s: pd.Series, fmt: str = config.SPB_TS_FORMAT) -> pd.Series:
    """Строго по формату. Обратная проверка strftime ловит то, что strptime прощает: «1.1.2026 0»."""
    ts = pd.to_datetime(s, format=fmt, errors="coerce")
    bad = ts.isna() | (ts.dt.strftime(fmt) != s)
    if bad.any():
        raise ValueError(f"Даты не в формате {fmt!r}: {sorted(s[bad].unique())[:5]}")
    return ts


def label_to_local(label: pd.Series, before: int = config.SPB_PREV_DAY_LABEL_BEFORE) -> pd.Series:
    """Метка файла → реальное местное время: часы раньше `before` относятся к следующему календарному дню."""
    return label + pd.to_timedelta((label.dt.hour < before).astype(int), unit="D")


def regime_open(hour: pd.Series, first: pd.Series, last: pd.Series) -> pd.Series:
    """Час внутри режима вестибюля first…last; сутки метро начинаются в 05:00, поэтому last = 0 — полночный час."""
    pos = service_pos(hour)
    return (pos >= service_pos(first)) & (pos <= service_pos(last))


def closure_reason(df: pd.DataFrame, vestibules: pd.DataFrame,
                   max_entries: int = config.SPB_CLOSURE_MAX_ENTRIES,
                   min_norm: float = config.SPB_CLOSURE_MIN_NORM) -> pd.Series:
    """Почему вестибюль закрыт в часы работы метро: «режим», «разово» или "" (открыт)."""
    hour = df.ts_local.dt.hour
    reg = vestibules.set_index("vestibule_id")
    in_service = ~hour.isin(config.SPB_CLOSED_HOURS)
    open_ = regime_open(hour, df.vestibule_id.map(reg.first_hour), df.vestibule_id.map(reg.last_hour))
    norm = df.entries.where(in_service & open_).groupby([df.vestibule_id, hour]).transform("median")
    one_off = in_service & open_ & (df.entries <= max_entries) & (norm >= min_norm)
    return pd.Series(np.select([in_service & ~open_, one_off], [REGIME, ONE_OFF], ""), index=df.index)


def incident_mask(df: pd.DataFrame, incidents: pd.DataFrame) -> pd.Series:
    local = df.ts_local.dt.tz_localize(None)
    mask = pd.Series(False, index=df.index)
    for inc in incidents.itertuples():
        who = True if inc.vestibule_id == reference.ALL_VESTIBULES else df.vestibule_id == inc.vestibule_id
        mask |= who & (local >= inc.start_local) & (local < inc.end_local)
    return mask


def clean_flow(raw: pd.DataFrame, vestibules: pd.DataFrame, incidents: pd.DataFrame,
               start_local: pd.Timestamp = config.SPB_START_LOCAL, end_local: pd.Timestamp = config.SPB_END_LOCAL,
               freq: str = config.SPB_FREQ, tz: str = config.SPB_TZ) -> pd.DataFrame:
    names = vestibules.set_index("raw_name")
    unknown, unused = set(raw.raw_name) - set(names.index), set(names.index) - set(raw.raw_name)
    if unknown or unused:
        raise ValueError(f"Названия вестибюлей не сопоставлены: нет в справочнике {sorted(unknown)}, "
                         f"нет в данных {sorted(unused)}")
    bad_inc = set(incidents.vestibule_id) - set(vestibules.vestibule_id) - {reference.ALL_VESTIBULES}
    if bad_inc:
        raise ValueError(f"incidents.csv: неизвестные vestibule_id {sorted(bad_inc)}")

    naive = label_to_local(parse_ts(raw.date_raw))
    ts_local = naive.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    ts_utc = ts_local.dt.tz_convert("UTC")
    offsets = (naive - ts_utc.dt.tz_localize(None)).unique()
    if len(offsets) != 1:
        raise ValueError(f"Смещение {tz} от UTC меняется внутри периода: {offsets}")
    obs = pd.DataFrame({"vestibule_id": raw.raw_name.map(names.vestibule_id), "ts_utc": ts_utc,
                        "entries": raw.entries})
    dup = obs.duplicated(["vestibule_id", "ts_utc"])
    if dup.any():
        raise ValueError(f"Дубликаты (vestibule_id, ts_utc): {obs[dup].head().to_dict('records')}")

    start_utc = start_local.tz_localize(tz).tz_convert("UTC")
    end_utc = end_local.tz_localize(tz).tz_convert("UTC")
    tail = obs.ts_utc > end_utc            # одиночная метка «30.09 00» = 01.10 00:00 — за концом сетки
    dropped = obs[tail]
    obs = obs[~tail]
    grid = build_grid(vestibules.vestibule_id.tolist(), start_utc, end_utc, freq, id_col="vestibule_id")
    key = ["vestibule_id", "ts_utc"]
    g_idx, o_idx = pd.MultiIndex.from_frame(grid[key]), pd.MultiIndex.from_frame(obs[key])
    missing, extra = g_idx.difference(o_idx), o_idx.difference(g_idx)
    if len(missing) or len(extra):
        raise ValueError(f"Сетка неполная: нет {len(missing)} строк (напр. {list(missing[:3])}), "
                         f"лишних {len(extra)} (напр. {list(extra[:3])})")

    df = grid.merge(obs, on=key, how="left", validate="one_to_one")
    df["station_id"] = df.vestibule_id.map(vestibules.set_index("vestibule_id").station_id)
    df["ts_local"] = df.ts_utc.dt.tz_convert(tz)
    df["is_closed_hour"] = df.ts_local.dt.hour.isin(config.SPB_CLOSED_HOURS)
    df["is_vestibule_closed"] = closure_reason(df, vestibules) != ""
    df["is_incident"] = incident_mask(df, incidents)
    out = df[OUT_COLS].sort_values(key, ignore_index=True)
    out.attrs["dropped_tail"] = {"rows": len(dropped), "entries": int(dropped.entries.sum()),
                                 "ts_local": sorted(dropped.ts_utc.dt.tz_convert(tz).astype(str).unique())}
    return out


# --- Календарь ------------------------------------------------------------------
def clean_calendar(codes: str, holidays_ru: pd.DataFrame, year: int = config.CALENDAR_YEAR) -> pd.DataFrame:
    """isdayoff (тип дня с переносами) + holidays.RU (названия). Перенесённый выходной — будний нерабочий без праздника."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    codes = codes.strip()
    if len(codes) != len(dates) or set(codes) - set("012"):
        raise ValueError(f"Календарь isdayoff: ожидалось {len(dates)} цифр 0/1/2")
    df = pd.DataFrame({"date": dates.date, "dow": dates.dayofweek, "day_code": [int(c) for c in codes]})
    df["is_workday"] = df.day_code != 1
    df["is_shortened"] = df.day_code == 2
    names = holidays_ru.assign(date=pd.to_datetime(holidays_ru.date).dt.date).groupby("date").name.agg("; ".join)
    df["holiday_name"] = df.date.map(names).fillna("")
    df["is_transfer_dayoff"] = ~df.is_workday & (df.dow < 5) & (df.holiday_name == "")
    df["is_working_weekend"] = df.is_workday & (df.dow >= 5)
    return df


# --- МО-I -------------------------------------------------------------------------
def _num(s: str) -> float:
    s = re.sub(r"\s", "", s).replace(",", ".")
    return np.nan if s in ("", "-") else float(s)


def _period(title: str) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    if m := re.search(r"за\s+(\d{4})\s+ГОД", title, flags=re.I):
        p = pd.Period(m.group(1), freq="Y")
        label = m.group(1)
    elif m := re.search(r"за\s+(I{1,3}|IV)\s+квартал\s+(\d{4})", title, flags=re.I):
        q = ["I", "II", "III", "IV"].index(m.group(1).upper()) + 1
        p = pd.Period(f"{m.group(2)}Q{q}", freq="Q")
        label = f"{m.group(2)}-Q{q}"
    else:
        raise ValueError(f"МО-I: не распознан отчётный период в «{title}»")
    return label, p.start_time.normalize(), p.end_time.normalize()


def parse_mo1(path: Path) -> pd.DataFrame:
    """Форма МО-I (HTML в KOI8-R с расширением .xls) → период × линия × показатель."""
    soup = BeautifulSoup(path.read_bytes().decode("koi8-r"), "html.parser")
    text = lambda el: re.sub(r"\s+", " ", el.get_text(" ").replace("\xa0", " ")).strip()
    rows = [(tr, [text(td) for td in tr.find_all("td", recursive=False)]) for tr in soup.find_all("tr")]

    title = next((c[0] for _, c in rows if c and "выполнения графика движения поездов" in c[0]), None)
    if title is None:
        raise ValueError(f"{path.name}: нет заголовка «выполнения графика движения поездов»")
    period, start, end = _period(title)
    i_lines = next((i for i, (_, c) in enumerate(rows) if c and all(x.isdigit() for x in c) and c[0] == "1"), None)
    if i_lines is None:
        raise ValueError(f"{path.name}: нет строки с номерами линий")
    lines = rows[i_lines][1] + ["total"]

    out = []
    for tr, cells in rows[i_lines + 1:]:
        if len(cells) != len(lines) + 1:
            continue
        m = re.match(r"^(.*?)\s*\(([^()]+)\)(.*)$", cells[0])
        name, unit = ((m.group(1) + m.group(3)), m.group(2)) if m else (cells[0], "")
        name = re.sub(r"\s+,", ",", name).strip(" ,")
        for line, v in zip(lines, cells[1:]):
            out.append({"source_file": nfc(path.name), "period": period, "period_start": start, "period_end": end,
                        "line": line, "indicator": name, "unit": unit, "value": _num(v),
                        "is_detail": "detail-vagon-km" in (tr.get("class") or [])})
    if not out:
        raise ValueError(f"{path.name}: в таблице нет строк показателей")
    return pd.DataFrame(out)


def clean_mo1(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError(f"Нет файлов {config.SPB_MO1_GLOB} в {config.SPB_RAW}")
    df = pd.concat([parse_mo1(p) for p in paths], ignore_index=True)
    per_period = df.groupby("period").source_file.nunique()
    if (per_period > 1).any():
        raise ValueError(f"МО-I: один период в нескольких файлах: {per_period[per_period > 1].to_dict()}")
    return df.sort_values(["period_start"], kind="stable", ignore_index=True)


# --- Отчёт ----------------------------------------------------------------------
def _service_day(ts_local: pd.Series) -> pd.Series:
    return (ts_local.dt.tz_localize(None) - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).dt.date


def report(df: pd.DataFrame, vestibules: pd.DataFrame, stations: pd.DataFrame,
           weather: dict[str, pd.DataFrame], cal: pd.DataFrame, mo1: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    v_name = vestibules.set_index("vestibule_id").raw_name
    hour = df.ts_local.dt.hour

    print("\n=== Период и строки ===")
    per_v = df.groupby("vestibule_id").size()
    print(f"ts_local: {df.ts_local.min()} … {df.ts_local.max()}")
    print(f"ts_utc:   {df.ts_utc.min()} … {df.ts_utc.max()}")
    tail = df.attrs.get("dropped_tail", {"rows": 0, "entries": 0, "ts_local": []})
    print(f"строк: {len(df):,}; вестибюлей: {df.vestibule_id.nunique()}; станций: {df.station_id.nunique()}; "
          f"часов на вестибюль: {sorted(per_v.unique().tolist())}; сумма входов: {df.entries.sum():,} "
          f"+ за концом сетки {tail['entries']:,} = итогу файла")
    print(f"часы 00–02 в файле подписаны датой прошедших суток метро — переведены в реальное время; "
          f"отброшено {tail['rows']} строк за концом сетки ({', '.join(tail['ts_local'])})")

    print("\n=== Сопоставление названий (порядок по линии: Девяткино → пр. Ветеранов) ===")
    names = vestibules.merge(stations[["station_id", "name", "line_order"]], on="station_id")
    names["rows"] = names.vestibule_id.map(per_v)
    print(names[["line_order", "raw_name", "vestibule_id", "station_id", "name", "rows"]].to_string(index=False))
    no_data = stations[~stations.has_data]
    print(f"станции без данных (has_data=false): {no_data.name.tolist()}; строк у них: "
          f"{int(df.station_id.isin(no_data.station_id).sum())}")

    print("\n=== Часы работы по данным: доля дней со входами > 0 ===")
    edge = [4, 5, 6, 21, 22, 23, 0]
    share = (df[hour.isin(edge)].assign(h=hour, pos=lambda d: d.entries > 0)
             .pivot_table(index="vestibule_id", columns="h", values="pos", aggfunc="mean")[edge])
    share.index = share.index.map(v_name)
    reg = vestibules.set_index("raw_name")
    share["режим"] = [f"{reg.at[n, 'first_hour']:02d}–{reg.at[n, 'last_hour']:02d}" for n in share.index]
    print(share.loc[vestibules.raw_name].to_string(float_format=lambda x: f"{x:.0%}"))
    print("(04 ч — служебные проходы до конца апреля; 00 ч на конечных — малый поток, не закрытие)")

    print("\n=== Топ-5 вестибюлей по рабочим дням (производственный календарь; сутки метро 05–05) ===")
    sday = _service_day(df.ts_local)
    daily = df.groupby([df.vestibule_id, sday.rename("day")]).entries.sum().reset_index()
    full_days = (daily.day >= config.SPB_START_LOCAL.date()) & (daily.day < config.SPB_END_LOCAL.date())
    work = set(cal.loc[cal.is_workday, "date"])
    wd = daily[full_days & daily.day.isin(work)]
    top = wd.groupby("vestibule_id").entries.agg(["mean", "max"]).sort_values("mean", ascending=False).head(5)
    top.index = top.index.map(v_name)
    print(f"рабочих дней: {wd.day.nunique()}; в среднем по линии {wd.groupby('day').entries.sum().mean():,.0f} входов в день")
    print(top.round(0).astype(int).rename(columns={"mean": "в среднем за день", "max": "максимум"}).to_string())

    print("\n=== Контрольные дни: входы по линии по часам ===")
    days = ["2026-02-01", "2026-03-01", "2026-04-01"]
    local_date = df.ts_local.dt.tz_localize(None).dt.normalize()
    prof = pd.DataFrame({f"{d} {pd.Timestamp(d):%a}": df[local_date == d].groupby(hour).entries.sum() for d in days})
    print(prof.T.to_string())
    for col in prof:
        p = prof[col]
        print(f"  {col}: утро 07–09 max {p.loc[7:9].max():,} ({p.loc[7:9].idxmax():02d} ч), "
              f"вечер 16–19 max {p.loc[16:19].max():,} ({p.loc[16:19].idxmax():02d} ч), 12 ч {p[12]:,}")

    print("\n=== Нули и флаги ===")
    reason = closure_reason(df, vestibules)
    label = pd.Series(np.select(
        [df.is_closed_hour, reason == REGIME, reason == ONE_OFF, df.is_incident],
        ["метро закрыто (01–04)", "вестибюль закрыт: режим", "вестибюль закрыт: разово", "инцидент"],
        "без флага"), index=df.index)
    tab = df.assign(label=label, zero=df.entries == 0).groupby("label").agg(
        строк=("entries", "size"), нулей=("zero", "sum"), ненулевых=("zero", lambda z: int((~z).sum())),
        входов=("entries", "sum"))
    print(tab.to_string())
    print(f"(метка — первый подходящий флаг по порядку; пересечения: инцидент ∩ прочие = "
          f"{int((df.is_incident & (df.is_closed_hour | df.is_vestibule_closed)).sum())})")

    z = df[(df.entries == 0) & (label == "без флага")]
    print(f"\nнули без флага: {len(z)} — по вестибюлю и часу:")
    print(pd.crosstab(z.vestibule_id.map(v_name), hour[z.index]).to_string())

    reg_rows = df[reason == REGIME]
    print("\nзакрыт по режиму — нулей / ненулевых по вестибюлю и часу:")
    print(reg_rows.assign(n=reg_rows.vestibule_id.map(v_name), h=hour, zero=reg_rows.entries == 0)
          .groupby(["n", "h"]).zero.agg(нулей="sum", всего="size").assign(ненулевых=lambda t: t.всего - t.нулей)
          .drop(columns="всего").T.to_string())

    one = df[reason == ONE_OFF]
    print("\nзакрыт разово:")
    print(one.assign(name=one.vestibule_id.map(v_name))[["name", "ts_local", "entries"]].to_string(index=False))

    night = df[df.is_closed_hour].groupby(df.ts_local.dt.tz_localize(None)).entries.sum()
    print("\nметро закрыто, но по линии > 100 входов за час (особые ночи → events_spb.csv):")
    print(night[night > 100].to_string())

    inc = df[df.is_incident]
    print(f"\nинцидент: {len(inc)} строк = {inc.vestibule_id.nunique()} вестибюлей × "
          f"{inc.ts_local.dt.hour.nunique()} ч ({inc.ts_local.min()} … {inc.ts_local.max()})")

    print("\n=== 5 строк из interim (Девяткино II, 31.08 07–11 ч) ===")
    print(df[(df.vestibule_id == "devyatkino_2") & (local_date == "2026-08-31") & hour.between(7, 11)].to_string(index=False))

    print("\n=== Внешние таблицы ===")
    for kind, w in weather.items():
        nan = w.drop(columns=["ts_utc", "ts_local"]).isna().sum()
        print(f"weather_{kind}_spb: {len(w)} ч, {w.ts_utc.min()} … {w.ts_utc.max()}, NaN: {nan[nan > 0].to_dict() or 0}")
    odd = cal[(cal.holiday_name != "") | cal.is_transfer_dayoff | cal.is_working_weekend | cal.is_shortened]
    print(f"calendar_ru_{config.CALENDAR_YEAR}: {len(cal)} дней, нерабочих {(~cal.is_workday).sum()}; "
          f"особые дни ({len(odd)}):")
    print(odd[["date", "dow", "day_code", "holiday_name", "is_transfer_dayoff", "is_working_weekend",
               "is_shortened"]].to_string(index=False))
    key = mo1[(mo1.line == "1") & ~mo1.is_detail]
    print(f"mo1_reports: {len(mo1)} строк, периоды {mo1.period.unique().tolist()}; линия 1:")
    print(key.pivot_table(index="indicator", columns="period", values="value", sort=False).to_string())


# --- main -------------------------------------------------------------------------
def main() -> None:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    stations = reference.load_stations()
    vestibules = reference.load_vestibules(stations=stations)
    incidents = reference.load_incidents(vestibule_ids=vestibules.vestibule_id.tolist())

    flow_path = find_one(config.SPB_FLOW_FILE)
    raw = read_flow_xlsx(flow_path)
    df = clean_flow(raw, vestibules, incidents)
    df.to_parquet(config.SPB_HOURLY, index=False)

    weather = {}
    for kind in config.WEATHER_URLS:
        path = config.RAW / f"weather_{kind}_spb.json"
        if not path.exists():
            raise FileNotFoundError(f"Нет {path} — запустите python -m src.download --only weather_spb")
        weather[kind] = clean_weather(json.loads(path.read_text(encoding="utf-8")),
                                      config.SPB_START_UTC, config.SPB_END_UTC, config.SPB_TZ)
        weather[kind].to_parquet(config.INTERIM / f"weather_{kind}_spb.parquet", index=False)

    year = config.CALENDAR_YEAR
    codes_path, hol_path = config.RAW / f"calendar_ru_{year}_isdayoff.txt", config.RAW / f"holidays_ru_{year}.csv"
    if not codes_path.exists() or not hol_path.exists():
        raise FileNotFoundError("Нет календаря — запустите python -m src.download --only calendar")
    cal = clean_calendar(codes_path.read_text(encoding="utf-8"), pd.read_csv(hol_path), year)
    cal.to_parquet(config.CALENDAR_OUT, index=False)

    mo1 = clean_mo1(find_raw(config.SPB_MO1_GLOB))
    mo1.to_parquet(config.MO1_OUT, index=False)

    print(f"поток: {flow_path.relative_to(config.ROOT)}")
    report(df, vestibules, stations, weather, cal, mo1)
    print(f"\nclean_spb: готово → {config.INTERIM.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
