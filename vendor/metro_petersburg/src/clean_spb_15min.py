"""Очистка 15-минутных данных 1 линии СПб (этап 6): data/raw/spb/15min → data/interim/spb_line1_15min.parquet.

Запуск: python -m src.clean_spb_15min | tee logs/clean_spb_15min.log

Файлы «Входные пассажиропотоки Линия 1 <месяц> 2026 по 15-мин.xlsx» (февраль, май, июль, сентябрь):
- лист на сутки метро «ТВхП ДДММГГГГ <буква дня недели> 15-мин.», в первой ячейке — «Таблица входных потоков.
  ДДММГГГГ <день недели>»; листы идут от последнего дня к первому, пустой «Лист1» пропускается;
- 96 колонок 03:00 … 02:45, метка — начало интервала; строка над вестибюлями — итог линии по слотам, справа — итог суток;
- 24 вестибюля (с Технологическим институтом, которого нет в часовом файле), снизу — итог суток.

Слоты 00:00–02:45 относятся к следующей календарной дате: та же подпись «датой суток метро», что в часовом файле
(config.SPB_PREV_DAY_LABEL_BEFORE). Сортировка — по vestibule_id и ts_utc, а не в порядке листов.
Перевода часов в СПб нет, поэтому dst_flag здесь не нужен (правило CLAUDE.md выполняется тривиально).

Выход — полная сетка «вестибюль × 15 мин»: vestibule_id, station_id, ts_utc, ts_local, sday, hour, quarter, entries,
in_hourly, флаги часа, day_type, is_regular. Флаги — те же, что в spb_line1_hourly.parquet, и стоят на всех
четырёх слотах часа:
- где час есть в часовом файле — флаги берутся оттуда;
- для Технологического института и суток 30.09 (их нет в часовом файле) — по тем же правилам clean_spb:
  01–04 — метро закрыто, режим вестибюля из справочника, разовое закрытие по медиане часа, инциденты из incidents.csv.
Два новых флага — часы с ними не идут в профиль и в оценку раннего сигнала:
- is_source_mismatch — |Σ 4 слотов − часовой файл| > max(SPB_15MIN_MISMATCH_ABS; REL × час): выгрузки накопленных
  данных или нули в одном из источников;
- is_slot_closed — частичное закрытие: в часе есть слот с входом ≤ max(SPB_CLOSURE_MAX_ENTRIES;
  SPB_15MIN_SLOT_CLOSED_SHARE × медиана) при медиане этого вестибюля, слота и типа дня ≥ SPB_CLOSURE_MIN_NORM
  (правило разового закрытия clean_spb на шаге 15 минут; доля — для служебных проходов в больших вестибюлях).
  Так видны регулярные закрытия входа на 30–60 минут в пик (Балтийская, Нарвская), которых часовые флаги не видят.
day_type и is_regular — как в eda_spb.service_calendar (производственный календарь и events_spb.csv).

Сверка «Σ 4 слотов = часу» по каждому часу — в data/interim/spb_line1_15min_reconciliation.parquet,
сводки и выбросы — в reports/intrahour/.
"""
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, reference
from src import eda_spb as E
from src.clean_spb import REGIME, ONE_OFF, closure_reason, find_raw, incident_mask
from src.reference import nfc

SHEET_RE = re.compile(r"^ТВхП (\d{8}) (\S) 15-мин\.$")
TITLE_PREFIX = "Таблица входных потоков."
WEEKDAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
FIRST_COL = 2                                   # A — название, B — итог суток, C… — слоты
HOUR_FLAGS = ["is_closed_hour", "is_vestibule_closed", "is_incident", "is_source_mismatch"]
FLAGS = [*HOUR_FLAGS, "is_slot_closed"]
OUT_COLS = ["vestibule_id", "station_id", "ts_utc", "ts_local", "sday", "hour", "quarter", "entries", "in_hourly",
            *FLAGS, "day_type", "is_regular"]
REPORTS = config.ROOT / "reports" / "intrahour"


def slot_times(n: int = config.SPB_15MIN_SLOTS) -> list[dt.time]:
    start = dt.datetime(2000, 1, 1) + config.SPB_15MIN_DAY_START
    return [(start + dt.timedelta(minutes=15 * i)).time() for i in range(n)]


# --- Разбор листа ----------------------------------------------------------------
def _count(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or v < 0:
        raise ValueError(f"{where}: ожидалось целое ≥ 0, а там {v!r}")
    return int(v)


def parse_day_sheet(name: str, rows: list[tuple]) -> pd.DataFrame:
    """Лист суток метро как есть (строки ячеек) → sheet_date, raw_name, slot, entries; со всеми сверками итогов."""
    m = SHEET_RE.match(nfc(name))
    if not m:
        raise ValueError(f"лист «{name}»: имя не в формате «ТВхП ДДММГГГГ <буква> 15-мин.»")
    day = pd.to_datetime(m.group(1), format="%d%m%Y", errors="coerce")
    if pd.isna(day) or day.strftime("%d%m%Y") != m.group(1):
        raise ValueError(f"лист «{name}»: дата {m.group(1)} не в формате ДДММГГГГ")
    weekday = WEEKDAYS_RU[day.dayofweek]
    if m.group(2) != weekday[0]:
        raise ValueError(f"лист «{name}»: буква дня недели «{m.group(2)}», а {day:%d.%m.%Y} — {weekday}")

    n = config.SPB_15MIN_SLOTS
    width = FIRST_COL + n + 1
    rows = [tuple(r) + (None,) * (width - len(r)) for r in rows]
    if any(v is not None for r in rows for v in r[width:]):
        raise ValueError(f"лист «{name}»: данные правее колонки итога")
    title = f"{TITLE_PREFIX} {m.group(1)} {weekday}"
    if len(rows) < 4 or nfc(str(rows[0][0])) != title or any(v is not None for v in rows[0][1:]):
        raise ValueError(f"лист «{name}»: первая строка должна быть «{title}», а там {rows[0][:2] if rows else None}")
    head = rows[1]
    if head[:FIRST_COL] != (None,) * FIRST_COL or list(head[FIRST_COL:FIRST_COL + n]) != slot_times(n) \
            or head[FIRST_COL + n] is not None:
        raise ValueError(f"лист «{name}»: в заголовке ожидались {n} меток 03:00 … 02:45")

    line = rows[2]
    if line[:FIRST_COL] != (None,) * FIRST_COL:
        raise ValueError(f"лист «{name}»: строка итога линии должна начинаться с пустых ячеек")
    body = rows[3:]
    n_ves = next((i for i, r in enumerate(body) if r[0] is None), len(body))
    ves, tail = body[:n_ves], body[n_ves:]
    if not tail or tail[0][0] is not None or any(v is not None for v in tail[0][FIRST_COL:]):
        raise ValueError(f"лист «{name}»: после вестибюлей ожидалась строка итога суток")
    if any(v is not None for r in tail[1:] for v in r):
        raise ValueError(f"лист «{name}»: после строки итога ещё есть данные")

    where = f"лист «{name}»"
    names = [nfc(str(r[0])) for r in ves]
    vals = np.array([[_count(v, f"{where}, {nm}") for v in r[FIRST_COL:FIRST_COL + n]] for nm, r in zip(names, ves)],
                    dtype="int64")
    if any(r[FIRST_COL + n] is not None for r in ves):
        raise ValueError(f"{where}: справа от слотов вестибюля есть данные")
    day_tot = np.array([_count(r[1], f"{where}, итог {nm}") for nm, r in zip(names, ves)])
    if (vals.sum(1) != day_tot).any():
        bad = [nm for nm, a, b in zip(names, vals.sum(1), day_tot) if a != b]
        raise ValueError(f"{where}: сумма слотов не равна итогу суток у {bad}")
    line_slots = np.array([_count(v, f"{where}, итог линии") for v in line[FIRST_COL:FIRST_COL + n]])
    if (line_slots != vals.sum(0)).any():
        raise ValueError(f"{where}: итог линии по слотам не равен сумме вестибюлей")
    total = _count(tail[0][1], f"{where}, итог суток")
    if _count(line[FIRST_COL + n], f"{where}, итог линии") != total or vals.sum() != total:
        raise ValueError(f"{where}: итог суток {total:,} не сходится с суммами ({vals.sum():,})")
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: повторяются названия вестибюлей")

    return pd.DataFrame({"sheet_date": day, "raw_name": np.repeat(names, n),
                         "slot": np.tile(np.arange(n), len(names)), "entries": vals.ravel()})


def read_15min_xlsx(path: Path) -> pd.DataFrame:
    """Все листы-сутки файла; пустые листы пропускаются. Файл обязан покрывать ровно один календарный месяц целиком."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if all(v is None for r in rows for v in r):
            continue
        parts.append(parse_day_sheet(ws.title, rows))
    wb.close()
    if not parts:
        raise ValueError(f"{path.name}: нет листов с данными")
    df = pd.concat(parts, ignore_index=True)
    days = pd.DatetimeIndex(df.sheet_date.unique())
    month = days.to_period("M").unique()
    full = pd.date_range(month[0].start_time, month[0].end_time.normalize(), freq="D") if len(month) == 1 else None
    if full is None or len(days) != len(full) or not days.sort_values().equals(full):
        raise ValueError(f"{nfc(path.name)}: листы должны покрывать один месяц целиком без повторов, "
                         f"а там {len(days)} суток: {sorted(days.strftime('%d.%m'))[:5]}…")
    return df.assign(source_file=nfc(path.name))


# --- Очистка -----------------------------------------------------------------------
def clean_15min(raw: pd.DataFrame, ves15: pd.DataFrame, incidents: pd.DataFrame, hourly: pd.DataFrame,
                calendar: pd.DataFrame, tz: str = config.SPB_TZ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сырые слоты → (сетка «вестибюль × 15 мин» с флагами, сверка «час × вестибюль» с часовым файлом).

    calendar — по суткам метро (eda_spb.service_calendar): date, day_type, is_regular.
    """
    names = ves15.set_index("raw_name_15min")
    expected = set(names.index)
    unknown, unused = set(raw.raw_name) - expected, expected - set(raw.raw_name)
    if unknown or unused:
        raise ValueError(f"Названия вестибюлей не сопоставлены: нет в справочнике {sorted(unknown)}, "
                         f"нет в данных {sorted(unused)}")
    per_sheet = raw.groupby("sheet_date").raw_name.agg(lambda s: frozenset(s))
    short = per_sheet[per_sheet != frozenset(expected)]
    if len(short):
        raise ValueError(f"Не на всех листах все {len(expected)} вестибюлей: {[f'{d:%d.%m}' for d in short.index[:5]]}")

    naive = raw.sheet_date + config.SPB_15MIN_DAY_START + pd.to_timedelta(15 * raw.slot, unit="min")
    ts_local = naive.dt.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    ts_utc = ts_local.dt.tz_convert("UTC")
    offsets = (naive - ts_utc.dt.tz_localize(None)).unique()
    if len(offsets) != 1:
        raise ValueError(f"Смещение {tz} от UTC меняется внутри периода: {offsets}")
    df = pd.DataFrame({"vestibule_id": raw.raw_name.map(names.vestibule_id), "ts_utc": ts_utc, "entries": raw.entries})
    dup = df.duplicated(["vestibule_id", "ts_utc"])
    if dup.any():
        raise ValueError(f"Дубликаты (vestibule_id, ts_utc): {df[dup].head().to_dict('records')}")
    df["hour_utc"] = df.ts_utc.dt.floor("h")

    # --- флаги на уровне часа
    ves = ves15.set_index("vestibule_id")
    h = df.groupby(["vestibule_id", "hour_utc"]).entries.agg(["sum", "size"]).reset_index()
    if (h["size"] != 4).any():
        raise ValueError("Есть часы не из 4 слотов — сетка неполная")
    h = h.rename(columns={"hour_utc": "ts_utc", "sum": "entries"}).drop(columns="size")
    h["ts_local"] = h.ts_utc.dt.tz_convert(tz)
    h["station_id"] = h.vestibule_id.map(ves.station_id)
    h["in_hourly"] = h.vestibule_id.map(ves.in_hourly).astype(bool)
    reason = closure_reason(h, ves15)
    rule = pd.DataFrame({"is_closed_hour": h.ts_local.dt.hour.isin(config.SPB_CLOSED_HOURS),
                         "is_vestibule_closed": reason != "", "is_incident": incident_mask(h, incidents)})
    hk = hourly[["vestibule_id", "ts_utc", "entries", "is_closed_hour", "is_vestibule_closed", "is_incident"]]
    h = h.merge(hk.rename(columns={"entries": "entries_hourly", **{f: f"{f}_h" for f in rule}}),
                on=["vestibule_id", "ts_utc"], how="left", validate="one_to_one")
    has_h = h.entries_hourly.notna()
    if (h.loc[has_h, "is_closed_hour_h"].astype(bool) != rule.loc[has_h, "is_closed_hour"]).any():
        raise ValueError("is_closed_hour расходится с часовым файлом — разная сетка часов")
    for f in rule:
        h[f] = np.where(has_h, h[f"{f}_h"].astype("boolean").fillna(False).astype(bool), rule[f])
    h["closure_rule"] = reason                                   # правило по 15-минутным данным — для отчёта
    h["vestibule_closed_by_rule"] = rule.is_vestibule_closed
    h = h.drop(columns=[f"{f}_h" for f in rule])
    h["diff"] = h.entries - h.entries_hourly
    lim = np.maximum(config.SPB_15MIN_MISMATCH_ABS, config.SPB_15MIN_MISMATCH_REL * h.entries_hourly)
    h["is_source_mismatch"] = has_h & (h["diff"].abs() > lim)

    local = h.ts_local.dt.tz_localize(None)
    h["sday"] = (local - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).dt.normalize()
    h["hour"] = local.dt.hour
    cal = calendar.set_index(pd.to_datetime(calendar.date))
    h["day_type"] = h.sday.map(cal.day_type)
    h["is_regular"] = h.sday.map(cal.is_regular).eq(True)
    if h.day_type.isna().any():
        raise ValueError(f"Нет типа дня для суток {sorted(h.sday[h.day_type.isna()].dt.strftime('%d.%m').unique())}")

    # --- слоты
    out = df.merge(h[["vestibule_id", "ts_utc", "station_id", "in_hourly", *HOUR_FLAGS, "sday", "hour", "day_type",
                      "is_regular"]].rename(columns={"ts_utc": "hour_utc"}),
                   on=["vestibule_id", "hour_utc"], how="left", validate="many_to_one")
    out["ts_local"] = out.ts_utc.dt.tz_convert(tz)
    out["quarter"] = (out.ts_local.dt.minute // 15).astype("int8")
    out["slot_closed"] = slot_closure(out)
    any_closed = out.groupby(["vestibule_id", "hour_utc"]).slot_closed.any().rename("is_slot_closed")
    out = out.merge(any_closed.reset_index(), on=["vestibule_id", "hour_utc"], how="left", validate="many_to_one")
    h = h.merge(any_closed.reset_index().rename(columns={"hour_utc": "ts_utc"}), on=["vestibule_id", "ts_utc"],
                how="left", validate="one_to_one")
    h = h.merge(out[out.slot_closed].groupby(["vestibule_id", "hour_utc"]).ts_local.agg(lambda t: ", ".join(t.dt.strftime("%H:%M")))
                .rename("closed_slots").reset_index().rename(columns={"hour_utc": "ts_utc"}),
                on=["vestibule_id", "ts_utc"], how="left")
    h["closed_slots"] = h.closed_slots.fillna("")
    out["entries"] = out.entries.astype("int64")
    order = {v: i for i, v in enumerate(ves15.vestibule_id)}
    out = out.assign(_o=out.vestibule_id.map(order)).sort_values(["_o", "ts_utc"], ignore_index=True)
    recon = h.assign(_o=h.vestibule_id.map(order)).sort_values(["_o", "ts_utc"], ignore_index=True).drop(columns="_o")
    return out[OUT_COLS], recon


def slot_closure(out: pd.DataFrame, max_entries: int = config.SPB_CLOSURE_MAX_ENTRIES,
                 min_norm: float = config.SPB_CLOSURE_MIN_NORM,
                 share: float = config.SPB_15MIN_SLOT_CLOSED_SHARE) -> pd.Series:
    """Слот закрыт: вход ≤ max(max_entries; share × медиана) при медиане вестибюля, слота (ЧЧ:ММ) и типа дня ≥ min_norm;
    праздник — как воскресенье. Медиана — по всем открытым слотам периода, как у разового закрытия в clean_spb (для флагов очистки
    допустимо, для признаков — нет). Слоты закрытых часов не рассматриваются."""
    open_ = ~(out.is_closed_hour | out.is_vestibule_closed)
    slot = out.ts_local.dt.strftime("%H:%M")
    day = out.day_type.replace({"праздник": "воскресенье"})
    norm = out.entries.where(open_).groupby([out.vestibule_id, slot, day]).transform("median")
    return open_ & (out.entries <= np.maximum(max_entries, share * norm)) & (norm >= min_norm)


# --- Сводки сверки -------------------------------------------------------------------
def recon_rows(recon: pd.DataFrame) -> pd.DataFrame:
    """Часы для сверки: есть в часовом файле, метро и вестибюль открыты (в закрытые часы почти одни нули)."""
    return recon[recon.entries_hourly.notna() & ~recon.is_closed_hour & ~recon.is_vestibule_closed]


def recon_summary(recon: pd.DataFrame) -> pd.DataFrame:
    """Сверка по срезам: доля точных совпадений, превышение Σ15 / Σчас − 1, медиана и доли |Δ| / час."""
    r = recon_rows(recon).copy()
    r["rel"] = r["diff"] / r.entries_hourly.clip(lower=1)
    slices = [("вся линия", pd.Series("все", index=r.index), ["все"]),
              ("месяц", r.sday.dt.month, sorted(r.sday.dt.month.unique())),
              ("день недели", r.sday.dt.dayofweek, list(range(7))),
              ("час", r.hour, [h for h in [*range(5, 24), 0] if h in set(r.hour)]),
              ("вестибюль", r.vestibule_id, list(dict.fromkeys(r.vestibule_id)))]
    rows = []
    for name, key, order in slices:
        groups = dict(list(r.groupby(key)))
        for k in order:
            g = groups[k]
            label = WEEKDAYS_RU[k] if name == "день недели" else str(k)
            rows.append({"slice": name, "key": label, "n_hours": len(g), "exact_share": (g["diff"] == 0).mean(),
                         "sum_15min": int(g.entries.sum()), "sum_hourly": int(g.entries_hourly.sum()),
                         "excess": g.entries.sum() / g.entries_hourly.sum() - 1, "median_rel": g.rel.median(),
                         "within_1pct": (g.rel.abs() <= 0.01).mean(), "within_5pct": (g.rel.abs() <= 0.05).mean(),
                         "mismatch_hours": int(g.is_source_mismatch.sum())})
    return pd.DataFrame(rows)


# --- Отчёт ----------------------------------------------------------------------------
def report(df: pd.DataFrame, recon: pd.DataFrame, raw: pd.DataFrame, ves15: pd.DataFrame,
           vestibules: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 300)
    name15 = ves15.set_index("vestibule_id").raw_name_15min
    loc = hourly.ts_local.dt.tz_localize(None)
    hourly_ref = hourly.assign(hour=loc.dt.hour, month=loc.dt.month, dow=loc.dt.dayofweek)
    pct = lambda x: f"{x * 100:+.2f} %".replace(".", ",")

    print("\n=== Файлы и сетка ===")
    files = raw.groupby("source_file").sheet_date.agg(["min", "max", "nunique"])
    print(files.rename(columns={"nunique": "суток"}).to_string())
    per_v = df.groupby("vestibule_id").size()
    print(f"строк: {len(df):,} = {df.vestibule_id.nunique()} вестибюлей × {raw.sheet_date.nunique()} суток × "
          f"{config.SPB_15MIN_SLOTS} слотов; слотов на вестибюль: {sorted(per_v.unique().tolist())}; "
          f"сумма входов: {df.entries.sum():,} (= сумме итогов листов: {raw.entries.sum():,})")
    print(f"ts_local: {df.ts_local.min()} … {df.ts_local.max()}; слоты 00:00–02:45 отнесены к следующей дате")

    print("\n=== Сопоставление названий ===")
    m = ves15.merge(vestibules[["vestibule_id", "raw_name"]], on="vestibule_id", how="left")
    m["raw_name"] = m.raw_name.fillna("— (нет в часовом файле)")
    print(m[["raw_name_15min", "raw_name", "vestibule_id", "station_id", "in_hourly"]].to_string(index=False))

    print("\n=== Подпись часов 00–02 и метка слота ===")
    q0 = raw[raw.slot.between(84, 87)].groupby("sheet_date").entries.sum()   # 00:00–00:45 листа
    by_wd = q0.groupby(q0.index.dayofweek).mean().round(0)
    by_wd.index = [WEEKDAYS_RU[i] for i in by_wd.index]
    print("вход линии в 00:00–00:59 по дню недели листа (если 00 ч — следующая дата, выше всего у пт и сб):")
    print(by_wd.astype(int).to_string())
    rh = recon[recon.entries_hourly.notna()]
    h0 = rh[rh.hour == 0]
    prev = h0[["vestibule_id", "ts_utc", "entries"]].assign(ts_utc=h0.ts_utc + pd.Timedelta(days=1))
    alt = prev.merge(rh[["vestibule_id", "ts_utc", "entries_hourly"]], on=["vestibule_id", "ts_utc"])
    print(f"00 ч, корреляция с часовым файлом: слоты — следующая дата {np.corrcoef(h0.entries, h0.entries_hourly)[0, 1]:.3f}, "
          f"та же дата листа {np.corrcoef(alt.entries, alt.entries_hourly)[0, 1]:.3f}")
    night = df[(df.ts_local.dt.tz_localize(None) >= "2026-05-29 22:00") & (df.ts_local.dt.tz_localize(None) < "2026-05-30 06:00")]
    nl = night.groupby(night.ts_local.dt.tz_localize(None)).entries.sum()
    print("особая ночь 29→30.05 (метро работало всю ночь): вход линии по часам, стык листов 29.05 | 30.05 — в 03:00")
    print(nl.groupby(nl.index.floor("h")).sum().to_string())
    slot_check = []
    hk = rh[["vestibule_id", "ts_utc", "entries_hourly"]]
    for k in (-1, 0, 1):
        s = (df.assign(h=(df.ts_utc + pd.Timedelta(minutes=15 * k)).dt.floor("h"))
             .groupby(["vestibule_id", "h"]).entries.sum().rename("e").reset_index()
             .merge(hk, left_on=["vestibule_id", "h"], right_on=["vestibule_id", "ts_utc"]))
        s = s.merge(recon[["vestibule_id", "ts_utc", "is_closed_hour"]], on=["vestibule_id", "ts_utc"])
        s = s[~s.is_closed_hour]
        slot_check.append(f"{k:+d} слот: {((s.e - s.entries_hourly).abs() / s.entries_hourly.clip(lower=1)).median():.2%}")
    print("медиана |Σ4 − час| / час при сдвиге разбивки на час (0 — метка = начало интервала): " + "; ".join(slot_check))

    print("\n=== Сверка: сумма 4 интервалов против spb_line1_hourly.parquet ===")
    allh = recon[recon.entries_hourly.notna()]
    r = recon_rows(recon)
    print(f"часов с обоими источниками: {len(allh):,}; точное совпадение — {(allh['diff'] == 0).mean():.1%} всех, "
          f"{(r['diff'] == 0).mean():.1%} часов работы ({len(r):,})")
    dq = r.groupby(["vestibule_id", "sday"])[["entries", "entries_hourly"]].sum()
    print(f"сутки вестибюля (часы работы): точное совпадение — {(dq.entries == dq.entries_hourly).mean():.1%}")
    line_day = r.groupby("sday")[["entries", "entries_hourly"]].sum()
    lr = line_day.entries / line_day.entries_hourly - 1
    print(f"линия: Σ15 / Σчас − 1 = {pct(r.entries.sum() / r.entries_hourly.sum() - 1)}; "
          f"по суткам p5…p95 {pct(lr.quantile(0.05))} … {pct(lr.quantile(0.95))}, min {pct(lr.min())}, max {pct(lr.max())}")
    a, b = np.polyfit(r.entries_hourly, r["diff"], 1)[::-1]
    print(f"форма: Δ ≈ {a:.1f} + {b:.4f} × час (МНК); в часах с потоком ≥ 200 Δ > 0 у "
          f"{(r[r.entries_hourly >= 200]['diff'] > 0).mean():.0%}")
    summ = recon_summary(recon)
    show = summ.assign(**{c: summ[c].map(lambda x: f"{x:.1%}") for c in ("exact_share", "within_1pct", "within_5pct")},
                       excess=summ.excess.map(pct), median_rel=summ.median_rel.map(pct))
    show["key"] = np.where(show.slice == "вестибюль", show.key.map(name15).fillna(show.key), show.key)
    print(show.drop(columns=["sum_15min", "sum_hourly"]).to_string(index=False))

    print(f"\n=== Выбросы: |Δ| > max({config.SPB_15MIN_MISMATCH_ABS}; {config.SPB_15MIN_MISMATCH_REL:.0%} часа) "
          f"→ is_source_mismatch ===")
    mm = recon[recon.is_source_mismatch]
    print(f"часов: {len(mm)} (из них в часы работы без флагов закрытия: {len(recon_rows(recon).query('is_source_mismatch'))})")
    print(mm.assign(name=mm.vestibule_id.map(name15), local=mm.ts_local.dt.strftime("%d.%m %H ч"))
          [["name", "local", "entries", "entries_hourly", "diff"]]
          .rename(columns={"entries": "Σ15", "entries_hourly": "час"}).to_string(index=False))

    print("\n=== Флаги (на уровне часа; у слотов — те же) ===")
    hours = recon.assign(label=np.select(
        [recon.is_closed_hour, recon.is_vestibule_closed, recon.is_incident, recon.is_source_mismatch,
         recon.is_slot_closed],
        ["метро закрыто (01–04)", "вестибюль закрыт", "инцидент", "расхождение источников", "частичное закрытие"],
        "без флага"))
    tab = hours.groupby("label").agg(часов=("entries", "size"), нулей=("entries", lambda e: int((e == 0).sum())),
                                     входов=("entries", "sum"))
    print(tab.to_string())
    src = np.where(recon.entries_hourly.notna(), "из часового файла", "по правилам")
    print(pd.crosstab(src, recon.in_hourly.map({True: "в часовом файле", False: "только 15 мин"})).to_string())
    both = recon[recon.entries_hourly.notna()]
    disagree = both[both.vestibule_closed_by_rule != both.is_vestibule_closed]
    print(f"закрытие вестибюля: правило по 15-минутным данным расходится с часовым флагом в {len(disagree)} часах "
          f"(флаг взят из часового файла):")
    if len(disagree):
        print(disagree.assign(name=disagree.vestibule_id.map(name15), local=disagree.ts_local.dt.strftime("%d.%m %H ч"))
              [["name", "local", "entries", "entries_hourly", "closure_rule", "is_vestibule_closed", "is_source_mismatch"]]
              .to_string(index=False))
    only15 = recon[recon.entries_hourly.isna()]
    print(f"часы без часового файла: {len(only15):,}, флагов: "
          + ", ".join(f"{f} {int(only15[f].sum())}" for f in ("is_closed_hour", "is_vestibule_closed", "is_incident")))

    print(f"\n=== Частичные закрытия: слот ≤ max({config.SPB_CLOSURE_MAX_ENTRIES}; "
          f"{config.SPB_15MIN_SLOT_CLOSED_SHARE:.0%} медианы) при медиане слота ≥ {config.SPB_CLOSURE_MIN_NORM} "
          f"→ is_slot_closed ===")
    sc = recon[recon.is_slot_closed]
    print(f"часов: {len(sc)}, из них с другими флагами: {int((sc.is_source_mismatch | sc.is_incident).sum())}")
    ep = sc.assign(month=sc.sday.dt.month, name=sc.vestibule_id.map(name15))
    print(ep.groupby(["name", "month", "day_type"]).agg(суток=("sday", "nunique"), часов=("hour", "size"),
                                                       слоты=("closed_slots", lambda x: ", ".join(sorted(set(", ".join(x).split(", ")))))
                                                       ).to_string())
    regular = ep.groupby(["vestibule_id", "hour"]).sday.nunique()
    for (v, hr) in regular[regular >= 5].index:
        x = hourly_ref[(hourly_ref.vestibule_id == v) & (hourly_ref.hour == hr) & (hourly_ref.dow < 5)]
        med = x.groupby(x.month).entries.median().round(0).astype(int)
        print(f"{name15[v]}, {hr:02d} ч, будни — медиана по часовому файлу по месяцам: "
              + ", ".join(f"{m}: {e:,}" for m, e in med.items()))

    print("\n=== Технологический институт и сутки 30.09 (нет в часовом файле) ===")
    ti = recon[~recon.in_hourly]
    tid = ti[~ti.is_closed_hour].groupby("sday").entries.sum()
    share_open = (ti.entries > 0).groupby(ti.hour).mean()
    open_hours = ", ".join(f"{h:02d}" for h in [*range(5, 24), *range(0, 5)] if share_open.get(h, 0) >= 0.5)
    print(f"Технологический институт: {len(tid)} суток, в среднем {tid.mean():,.0f} входов (от {tid.min():,} до {tid.max():,}); "
          f"часы со входом > 0 в большинство суток: {open_hours}; закрытий по правилам: {int(ti.is_vestibule_closed.sum())}")
    d30 = df[df.sday == "2026-09-30"]
    print(f"30.09: {d30.entries.sum():,} входов, из них Технологический институт {d30[~d30.in_hourly].entries.sum():,}")

    print("\n=== Типы дней (сутки метро с 05:00) ===")
    days = df.drop_duplicates("sday")[["sday", "day_type", "is_regular"]]
    days = days[days.sday.dt.month.isin(raw.sheet_date.dt.month.unique())]
    print(days.groupby([days.sday.dt.month.rename("месяц"), "day_type", "is_regular"]).size().unstack(fill_value=0).to_string())
    odd = days[~days.is_regular]
    print("необычные сутки:", ", ".join(f"{d:%d.%m} ({t})" for d, t in zip(odd.sday, odd.day_type)))
    return summ


# --- main -------------------------------------------------------------------------------
def main() -> None:
    stations = reference.load_stations()
    vestibules = reference.load_vestibules(stations=stations)
    ves15 = reference.load_vestibules_15min(vestibules=vestibules, stations=stations)
    incidents = reference.load_incidents(vestibule_ids=ves15.vestibule_id.tolist())
    events = reference.load_events(station_ids=stations.station_id.tolist())
    calendar = E.service_calendar(pd.read_parquet(config.CALENDAR_OUT), events)
    hourly = pd.read_parquet(config.SPB_HOURLY)

    paths = find_raw(config.SPB_15MIN_GLOB)
    if not paths:
        raise FileNotFoundError(f"Нет файлов «{config.SPB_15MIN_GLOB}» в {config.SPB_RAW}")
    raw = pd.concat([read_15min_xlsx(p) for p in paths], ignore_index=True)
    df, recon = clean_15min(raw, ves15, incidents, hourly, calendar)
    df.to_parquet(config.SPB_15MIN, index=False)
    recon.to_parquet(config.SPB_15MIN_RECON, index=False)

    print("файлы: " + "; ".join(str(p.relative_to(config.ROOT)) for p in paths))
    summ = report(df, recon, raw, ves15, vestibules, hourly)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summ.round(6).to_csv(REPORTS / "reconciliation_summary.csv", index=False)
    mm = recon[recon.is_source_mismatch]
    (mm.assign(ts_local=mm.ts_local.dt.tz_localize(None))
     [["vestibule_id", "ts_local", "entries", "entries_hourly", "diff", "closure_rule"]]
     .rename(columns={"entries": "entries_15min"}).to_csv(REPORTS / "source_mismatch.csv", index=False))
    sc = recon[recon.is_slot_closed]
    (sc.assign(ts_local=sc.ts_local.dt.tz_localize(None))
     [["vestibule_id", "ts_local", "day_type", "closed_slots", "entries", "entries_hourly", "is_source_mismatch"]]
     .rename(columns={"entries": "entries_15min"}).to_csv(REPORTS / "slot_closures.csv", index=False))
    print(f"\nclean_spb_15min: готово → {config.SPB_15MIN.relative_to(config.ROOT)}, "
          f"{config.SPB_15MIN_RECON.relative_to(config.ROOT)}, {REPORTS.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
