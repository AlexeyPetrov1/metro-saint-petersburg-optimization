"""Расчёты для EDA (этап 2). Каждая функция возвращает таблицу; графики — в notebooks/01_eda.ipynb.

Строки с dst_flag != "" отбрасываются при загрузке (CLAUDE.md), дальше их нигде нет.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config

DAY_TYPES = ["будни", "суббота", "воскресенье", "праздник"]
GROUPS = ["Манхэттен", "Пересадочные Квинса", "Остальные Квинса"]
WORK_HOURS = list(range(6, 24))   # метро круглосуточное; ночью потоки почти нулевые и раздувают относительные метрики
NIGHT_HOURS = list(range(1, 6))   # 01:00–05:59 местного времени
MORNING, EVENING = range(6, 11), range(16, 21)
METS_STATIONS = ["447", "448", "449"]  # Flushing-Main St, Mets-Willets Point, 111 St


@dataclass
class Data:
    mta: pd.DataFrame        # без DST-строк; + date, hour, dow, day_type, group, line_order, is_transfer_complex
    stations: pd.DataFrame
    weather: pd.DataFrame    # архив Open-Meteo (факт)
    forecast: pd.DataFrame   # исторический прогноз Open-Meteo
    games: pd.DataFrame
    holidays: pd.DataFrame
    events: pd.DataFrame     # data/reference/events_manual.csv


def station_groups(stations: pd.DataFrame) -> pd.Series:
    """Непересекающиеся группы: Манхэттен (4) / пересадочные Квинса (3) / остальные Квинса (15)."""
    return pd.Series(np.select([stations.borough == "Manhattan", stations.is_transfer_complex],
                               GROUPS[:2], GROUPS[2]), index=stations.index)


def day_type(date: pd.Series, public_dates: set) -> pd.Series:
    dow = date.dt.dayofweek
    out = np.select([date.isin(public_dates), dow == 5, dow == 6], ["праздник", "суббота", "воскресенье"], "будни")
    return pd.Series(pd.Categorical(out, categories=DAY_TYPES), index=date.index)


def load() -> Data:
    mta = pd.read_parquet(config.INTERIM / "mta_line7_hourly.parquet")
    mta = mta[mta.dst_flag == ""].drop(columns="dst_flag").reset_index(drop=True)

    st = pd.read_parquet(config.INTERIM / "stations.parquet")
    st["group"] = pd.Categorical(station_groups(st), categories=GROUPS)
    hol = pd.read_parquet(config.INTERIM / "holidays_us.parquet")
    hol["date"] = pd.to_datetime(hol["date"])
    public = set(hol.loc[hol.category == "public", "date"])

    mta = mta.merge(st[["station_id", "group", "line_order", "is_transfer_complex"]], on="station_id")
    local = mta["ts_local"].dt.tz_localize(None)
    mta["date"] = local.dt.normalize()
    mta["hour"] = local.dt.hour
    mta["dow"] = local.dt.dayofweek
    mta["day_type"] = day_type(mta["date"], public)

    events = pd.read_csv(config.EVENTS_MANUAL, parse_dates=["start_date", "end_date"], dtype={"nearest_station_id": str})
    return Data(
        mta=mta, stations=st,
        weather=pd.read_parquet(config.INTERIM / "weather_archive.parquet"),
        forecast=pd.read_parquet(config.INTERIM / "weather_hist_forecast.parquet"),
        games=pd.read_parquet(config.INTERIM / "mets_home_games.parquet"),
        holidays=hol, events=events,
    )


def event_dates(events: pd.DataFrame, pattern: str = "") -> set:
    rows = events[events.event.str.contains(pattern)] if pattern else events
    return {d for r in rows.itertuples() for d in pd.date_range(r.start_date, r.end_date)}


# --- Оформление -------------------------------------------------------------
# Палитра dataviz (проверена валидатором: 3 слота, все пары различимы при дальтонизме).
# Цвет закреплён за сущностью и одинаков на всех графиках.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # синий, оранжевый, бирюзовый
GROUP_COLORS = dict(zip(GROUPS, SERIES))
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style() -> None:
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.figsize": (11, 5.5), "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
        "legend.fontsize": 11, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 8, "axes.prop_cycle": cycler(color=SERIES),
    })


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png")


# --- 1. Суточные профили ----------------------------------------------------
def daily_profiles(mta: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """Средний поток на станцию по часам и доля часа в сутках — по группам и типам дня (без праздников)."""
    d = mta[mta.day_type != "праздник"]
    prof = d.groupby([by, "day_type", "hour"], observed=True).entries.mean().rename("mean").reset_index()
    prof["share"] = prof["mean"] / prof.groupby([by, "day_type"], observed=True)["mean"].transform("sum")
    return prof


def peak_ratios(mta: pd.DataFrame, by: str = "station_id") -> pd.DataFrame:
    """Утренний пик (06–10) против вечернего (16–20) по среднему профилю."""
    prof = daily_profiles(mta, by)
    rows = []
    for (key, dt), g in prof.groupby([by, "day_type"], observed=True):
        s = g.set_index("hour")["mean"]
        m, e = s.loc[list(MORNING)], s.loc[list(EVENING)]
        rows.append({by: key, "day_type": dt, "morning": m.max(), "morning_hour": m.idxmax(),
                     "evening": e.max(), "evening_hour": e.idxmax(), "ratio": m.max() / e.max()})
    return pd.DataFrame(rows)


# --- 2. Недельная сезонность ------------------------------------------------
def weekly_autocorr(mta: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(config.START_UTC, config.END_UTC, freq="h", tz="UTC")
    rows = []
    for sid, g in mta.groupby("station_id"):
        s = g.set_index("ts_utc").entries.reindex(full)  # исключённые DST-часы → NaN, пары с ними отбрасываются
        rows.append({"station_id": sid, "lag_24": s.autocorr(24), "lag_168": s.autocorr(168)})
    out = pd.DataFrame(rows).merge(stations[["station_id", "name", "line_order", "group"]], on="station_id")
    return out.sort_values("line_order", ignore_index=True)


# --- 3. Тренд ---------------------------------------------------------------
def monthly_trend(mta: pd.DataFrame) -> pd.DataFrame:
    """Средние сутки будней по месяцам: вся линия и группы. Не зависит от длины месяца и числа выходных."""
    wd = mta[mta.day_type == "будни"]
    parts = [wd.groupby("date").entries.sum().rename("total").reset_index().assign(level="Вся линия")]
    for grp, g in wd.groupby("group", observed=True):
        parts.append(g.groupby("date").entries.sum().rename("total").reset_index().assign(level=grp))
    daily = pd.concat(parts)
    daily["year"], daily["month"] = daily.date.dt.year, daily.date.dt.month
    m = daily.groupby(["level", "year", "month"]).total.mean().rename("mean_daily").reset_index()
    wide = m.pivot_table(index=["level", "month"], columns="year", values="mean_daily").reset_index()
    wide["yoy"] = wide[2024] / wide[2023] - 1
    return wide


# Найдено на EDA (п. 3): ступенчатая смена уровня на 82 St-Jackson Hts и 111 St в одни и те же дни,
# зеркально — у соседей 74 St-Broadway и 103 St-Corona Plaza (пассажиры уходили на соседние станции).
LEVEL_SHIFT_DATES = [pd.Timestamp("2023-05-15"), pd.Timestamp("2024-04-22")]
LEVEL_SHIFT_STATIONS = {"453": "сдвиг", "449": "сдвиг", "616": "сосед", "450": "сосед"}


def weekly_level(mta: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    """Средние будние сутки по неделям относительно медианы станции за весь период."""
    wd = mta[(mta.day_type == "будни") & mta.station_id.isin(station_ids)]
    daily = wd.groupby(["date", "station_id"]).entries.sum().unstack()
    weekly = daily.resample("W-SUN").mean()
    return weekly / daily.median()


def level_steps(mta: pd.DataFrame, dates=LEVEL_SHIFT_DATES, window: int = 10) -> pd.DataFrame:
    """Ступень уровня: средние будние сутки за window будней после даты / до даты − 1, по станциям и всей линии."""
    wd = mta[mta.day_type == "будни"]
    daily = wd.groupby(["date", "station_id"]).entries.sum().unstack()
    daily["вся линия"] = daily.sum(axis=1)
    rows = {dt: daily[daily.index >= dt].head(window).mean() / daily[daily.index < dt].tail(window).mean() - 1
            for dt in dates}
    return pd.DataFrame(rows).T


def missing_volume_share(mta: pd.DataFrame) -> float:
    """Оценка доли потока, потерянной в пропусках: пропуск заменяем средним своей ячейки (станция × час × тип дня × месяц)."""
    keys = ["station_id", "hour", "day_type", mta.date.dt.to_period("M").rename("ym")]
    fill = mta.groupby(keys, observed=True).entries.transform("mean")
    lost = fill[mta.entries.isna()].sum()
    return float(lost / (mta.entries.sum() + lost))


# --- 4. Праздники -----------------------------------------------------------
CATEGORY_ORDER = ["public", "ny_public", "unofficial", "предпраздничный", "послепраздничный"]
CATEGORY_RU = {"public": "федеральные", "ny_public": "штат NY", "unofficial": "неофициальные",
               "предпраздничный": "предпраздничный день", "послепраздничный": "послепраздничный день"}
HOLIDAY_RU = {
    "New Year's Day": "Новый год", "Martin Luther King Jr. Day": "День Мартина Лютера Кинга",
    "Washington's Birthday": "День президентов", "Memorial Day": "День памяти", "Juneteenth National Independence Day": "Джунтин",
    "Independence Day": "День независимости", "Labor Day": "День труда", "Columbus Day": "День Колумба",
    "Veterans Day": "День ветеранов", "Thanksgiving Day": "День благодарения", "Christmas Day": "Рождество",
    "Election Day": "День выборов", "Lincoln's Birthday": "День рождения Линкольна", "Susan B. Anthony Day": "День Сьюзен Энтони",
    "Christmas Eve": "Сочельник", "Easter Sunday": "Пасха", "Father's Day": "День отца", "Mother's Day": "День матери",
    "Good Friday": "Страстная пятница", "Groundhog Day": "День сурка", "Halloween": "Хэллоуин",
    "New Year's Eve": "Канун Нового года", "Saint Patrick's Day": "День святого Патрика", "Valentine's Day": "День святого Валентина",
}


def holiday_ru(name: str) -> str:
    """Русское название; «(observed)» — перенос выходного; составные имена — через « / »."""
    parts = []
    for p in name.split(" / "):
        base = p.replace(" (observed)", "")
        parts.append(HOLIDAY_RU.get(base, base) + (" (перенос)" if "(observed)" in p else ""))
    return " / ".join(parts)


def special_period(dates: pd.Series) -> pd.Series:
    """Неделя 24.12–01.01: просела целиком, «обычных» дней в ней нет."""
    return ((dates.dt.month == 12) & (dates.dt.day >= 24)) | ((dates.dt.month == 1) & (dates.dt.day == 1))


def holiday_calendar(hol: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Одна категория на дату: public > ny_public > unofficial > пред-/послепраздничный (government ⊂ public + сочельник)."""
    hol = hol[hol.category != "government"]
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    top = (hol.assign(r=hol.category.map(rank)).sort_values(["date", "r"])
           .groupby("date").agg(category=("category", "first"),
                                name=("name", lambda s: " / ".join(dict.fromkeys(s)))))
    public = set(hol.loc[hol.category == "public", "date"])
    all_dates = set(dates)
    is_workday = lambda d: d.dayofweek < 5 and d not in public
    extra = {}
    for h in sorted(public):
        for step, cat in ((-1, "предпраздничный"), (1, "послепраздничный")):
            d = h + pd.Timedelta(days=step)
            while d in all_dates and not is_workday(d):
                d += pd.Timedelta(days=step)
            if d in all_dates and d not in top.index and d not in extra:
                extra[d] = (cat, f"{cat}: {top.at[h, 'name']}")
    extra_df = pd.DataFrame.from_dict(extra, orient="index", columns=["category", "name"])
    return pd.concat([top, extra_df]).sort_index()


def holiday_effects(mta: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    """Суточный поток линии в праздник против (а) обычных будней той же недели и (б) того же дня недели ±1, ±2 недели."""
    daily = mta.groupby("date").entries.sum()
    dates = daily.index.to_series()
    cal = holiday_calendar(hol, dates)
    ordinary = dates[~dates.isin(cal.index) & ~special_period(dates)]
    iso = dates.dt.isocalendar()
    week = iso.year * 100 + iso.week
    rows = []
    for d, r in cal.iterrows():
        if d not in daily.index:
            continue
        same_week = ordinary[(week[ordinary.index] == week[d]) & (ordinary.dt.dayofweek < 5)]
        adj = [d + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
        adj = [a for a in adj if a in ordinary.index]
        # «та же неделя» — только при ≥ 2 обычных буднях: с одним днём база ломается от любого сбоя (снегопад 13.02.2024)
        b_week = daily[same_week.index].mean() if d.dayofweek < 5 and len(same_week) >= 2 else np.nan
        b_adj = daily[adj].mean() if adj else np.nan
        rows.append({"date": d, "category": r.category, "name": r["name"], "dow": d.dayofweek,
                     "total": daily[d], "base_week": b_week, "base_adj": b_adj,
                     "eff_week": daily[d] / b_week - 1, "eff_adj": daily[d] / b_adj - 1,
                     "special": bool(special_period(pd.Series([d])).iloc[0])})
    out = pd.DataFrame(rows)
    out["effect"] = out.eff_week.fillna(out.eff_adj)  # основной: та же неделя; для выходных — соседние недели
    return out


def weekday_levels(mta: pd.DataFrame, hol: pd.DataFrame) -> pd.Series:
    """Средний суточный поток обычных будней по дням недели относительно вт–чт: смещение базы «та же неделя»."""
    daily = mta.groupby("date").entries.sum()
    dates = daily.index.to_series()
    ordinary = dates[~dates.isin(holiday_calendar(hol, dates).index) & ~special_period(dates) & (dates.dt.dayofweek < 5)]
    lvl = daily[ordinary.index].groupby(ordinary.dt.dayofweek).mean()
    return lvl / lvl.loc[[1, 2, 3]].mean() - 1


def holiday_profile(mta: pd.DataFrame, hol: pd.DataFrame, name: str = "Thanksgiving Day") -> pd.DataFrame:
    """Часовой профиль линии (доля суток): праздник против обычного того же дня недели и обычного воскресенья того же месяца."""
    days = hol.loc[(hol.category == "public") & (hol.name == name), "date"]
    line = mta.groupby(["date", "hour"]).entries.sum().rename("v").reset_index()
    line["share"] = line.v / line.groupby("date").v.transform("sum")
    dates = line.date.drop_duplicates()
    cal = holiday_calendar(hol, dates)
    ordinary = set(dates[~dates.isin(cal.index) & ~special_period(dates)])
    months, dow = set(days.dt.month), days.dt.dayofweek.iloc[0]
    sel = {
        name: line[line.date.isin(days)],
        "обычный тот же день недели": line[line.date.isin(ordinary) & (line.date.dt.dayofweek == dow) & line.date.dt.month.isin(months)],
        "обычное воскресенье": line[line.date.isin(ordinary) & (line.date.dt.dayofweek == 6) & line.date.dt.month.isin(months)],
    }
    return pd.concat([g.groupby("hour").share.mean().rename(k) for k, g in sel.items()], axis=1)


# --- 5. Погода --------------------------------------------------------------
RAIN_MM, HEAVY_MM = 0.5, 2.5
WEATHER_CATS = {"rain": "дождь ≥ 0,5 мм/ч", "heavy": "сильный дождь ≥ 2,5 мм/ч", "snow": "снег > 0"}


def weather_flags(w: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts_utc": w.ts_utc,
        "rain": w.rain >= RAIN_MM,
        "heavy": w.rain >= HEAVY_MM,
        "snow": w.snowfall > 0,
        "dry": (w.precipitation == 0) & (w.snowfall == 0),
    })


def weather_effect(mta: pd.DataFrame, flags: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                   cats: dict[str, str] = WEATHER_CATS, ref: str = "dry") -> pd.DataFrame:
    """Поток в часы с погодой cats против среднего опорных часов ref той же ячейки (станция × час × тип дня × месяц).

    Эффект = Σ входов в часы с погодой / Σ опорной нормы − 1; 95 % интервал — бутстреп по дням.
    По умолчанию — осадки против сухих часов; для температуры опорные часы — «умеренные».
    """
    d = mta[mta.hour.isin(WORK_HOURS) & mta.entries.notna() & (mta.day_type != "праздник")].merge(flags, on="ts_utc")
    d["ym"] = d.date.dt.to_period("M")
    keys = ["station_id", "hour", "day_type", "ym"]
    dry = d[d[ref]].groupby(keys, observed=True).entries.agg(dry_mean="mean", dry_n="size")
    d = d.join(dry[dry.dry_n >= 3], on=keys)
    segments = {"все дни": d.day_type.notna(), "будни": d.day_type == "будни",
                "выходные": d.day_type.isin(["суббота", "воскресенье"])}
    rng = np.random.default_rng(seed)
    rows = []
    for cat, label in cats.items():
        for seg, mask in segments.items():
            w = d[d[cat] & mask & d.dry_mean.notna()]
            per_day = w.groupby("date").agg(num=("entries", "sum"), den=("dry_mean", "sum"))
            if per_day.empty:
                continue
            num, den = per_day.num.to_numpy(), per_day.den.to_numpy()
            idx = rng.integers(0, len(num), (n_boot, len(num)))
            boot = num[idx].sum(1) / den[idx].sum(1) - 1
            rows.append({"cat": cat, "label": label, "segment": seg, "effect": num.sum() / den.sum() - 1,
                         "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                         "lost": num.sum() - den.sum(),  # пассажиров «потеряно» за все такие часы
                         "n_hours": w.ts_utc.nunique(), "n_days": len(per_day)})
    return pd.DataFrame(rows)


def forecast_skill(arch: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Угадывает ли исторический прогноз осадки в том же часе: precision и recall при тех же порогах."""
    a, f = weather_flags(arch).set_index("ts_utc"), weather_flags(fc).set_index("ts_utc")
    f = f.reindex(a.index)
    rows = []
    for cat, label in WEATHER_CATS.items():
        fa, ff = a[cat], f[cat].fillna(False).astype(bool)
        near = ff | ff.shift(1, fill_value=False) | ff.shift(-1, fill_value=False)
        tp = (fa & ff).sum()
        rows.append({"cat": cat, "label": label, "hours_archive": int(fa.sum()), "hours_forecast": int(ff.sum()),
                     "precision": tp / max(ff.sum(), 1), "recall": tp / max(fa.sum(), 1),
                     "recall_pm1h": (fa & near).sum() / max(fa.sum(), 1)})
    return pd.DataFrame(rows)


# --- 6. Матчи «Метс» --------------------------------------------------------
def games_for_analysis(games: pd.DataFrame) -> pd.DataFrame:
    """Сыгранные матчи без двойных игр (окна накладываются)."""
    g = games[games.is_played & (games.double_header == "N")].copy()
    g["start_hour_utc"] = g.sched_start_utc.dt.floor("h")
    g["date"] = g.sched_start_utc.dt.tz_convert(config.TZ_LOCAL).dt.tz_localize(None).dt.normalize()
    return g.reset_index(drop=True)


def game_excess(d: Data, stations: list[str] | None = None, hours=range(-3, 6), window_days: int = 21) -> pd.DataFrame:
    """Превышение потока в часы вокруг матча над медианой тех же местных часов в дни без событий.

    k — час относительно часа старта: −3…−1 до игры, 0…2 игра, 3 — час est_end (старт + 3 ч), 3…5 — 3 ч после.
    База: тот же тип дня, ±window_days, без матчей на Citi Field (включая отложенные), US Open и праздников.
    """
    g = games_for_analysis(d.games)
    blocked = set(d.games.sched_start_utc.dt.tz_convert(config.TZ_LOCAL).dt.tz_localize(None).dt.normalize())
    blocked |= event_dates(d.events) | set(d.holidays.loc[d.holidays.category == "public", "date"])
    m = d.mta if stations is None else d.mta[d.mta.station_id.isin(stations)]
    local = m.ts_local.dt.tz_localize(None)
    e = dict(zip(zip(m.station_id, local), m.entries))
    dtype_by_date = d.mta.drop_duplicates("date").set_index("date").day_type
    all_dates = dtype_by_date.index
    rows = []
    for gm in g.itertuples():
        if gm.date in event_dates(d.events):
            continue  # матч в дни US Open — смешанный эффект
        near = all_dates[(abs(all_dates - gm.date) <= pd.Timedelta(days=window_days))]
        base_days = [b for b in near if b not in blocked and dtype_by_date[b] == dtype_by_date[gm.date]]
        if len(base_days) < 2:
            continue
        for k in hours:
            t = (gm.start_hour_utc + pd.Timedelta(hours=k)).tz_convert(config.TZ_LOCAL).tz_localize(None)
            offset = t - gm.date
            for sid in m.station_id.unique():
                val = e.get((sid, t), np.nan)
                base = np.nanmedian([e.get((sid, b + offset), np.nan) for b in base_days])
                rows.append({"game_pk": gm.game_pk, "station_id": sid, "k": k, "entries": val, "base": base,
                             "excess": val - base, "day_night": gm.day_night, "attendance": gm.attendance,
                             "actual_k": (gm.actual_end_utc - gm.start_hour_utc) / pd.Timedelta(hours=1),
                             "n_base": len(base_days)})
    return pd.DataFrame(rows)


def peak_shift(ex: pd.DataFrame, station: str = "448") -> pd.DataFrame:
    """Час максимального превышения после старта против est_end (k = 3) и фактического окончания."""
    s = ex[(ex.station_id == station) & ex.k.between(0, 5)].dropna(subset=["excess"])
    idx = s.groupby("game_pk").excess.idxmax()
    p = s.loc[idx, ["game_pk", "k", "actual_k", "day_night", "attendance"]].rename(columns={"k": "k_peak"})
    p["actual_k"] = p.actual_k.where(p.actual_k.between(0, 8))  # прерванные и доигранные позже матчи — без окончания
    p["actual_end_hour"] = np.floor(p.actual_k)
    p["shift_vs_est_end"] = p.k_peak - 3
    p["shift_vs_actual_end"] = p.k_peak - p.actual_end_hour
    return p.reset_index(drop=True)


def attendance_link(ex: pd.DataFrame, station: str = "448", ks=range(2, 6)) -> dict:
    """Суммарное превышение после игры (k = 2…5) против посещаемости."""
    s = ex[(ex.station_id == station) & ex.k.isin(list(ks))]
    per = s.groupby("game_pk").agg(excess=("excess", "sum"), attendance=("attendance", "first")).dropna()
    per = per[per.attendance > 0]
    slope, intercept = np.polyfit(per.attendance, per.excess, 1)
    return {"table": per, "pearson": stats.pearsonr(per.attendance, per.excess)[0],
            "spearman": stats.spearmanr(per.attendance, per.excess)[0],
            "per_1000": slope * 1000, "intercept": intercept, "n": len(per)}


# --- 7. Пропуски ------------------------------------------------------------
def missing_rows(mta: pd.DataFrame) -> pd.DataFrame:
    """Пропущенные часы: номер серии, её длина и пуста ли в этот час соседняя станция по линии."""
    m = mta.sort_values(["station_id", "ts_utc"])
    miss = m.pivot_table(index="ts_utc", columns="line_order", values="is_missing", aggfunc="first").astype(bool)
    neigh = (miss.shift(1, axis=1, fill_value=False) | miss.shift(-1, axis=1, fill_value=False)).stack()
    m = m.join(neigh.rename("neighbor_missing"), on=["ts_utc", "line_order"])
    mm = m[m.is_missing].copy()
    step = mm.groupby("station_id").ts_utc.diff() != pd.Timedelta(hours=1)
    mm["run"] = step.cumsum()
    mm["run_len"] = mm.groupby("run").ts_utc.transform("size")
    return mm


def gap_runs(mta: pd.DataFrame) -> pd.DataFrame:
    """Серии подряд идущих пропущенных часов по станции + доля часов, когда пуста и соседняя станция по линии."""
    mm = missing_rows(mta)
    runs = mm.groupby("run").agg(
        station_id=("station_id", "first"), line_order=("line_order", "first"),
        start_utc=("ts_utc", "min"), end_utc=("ts_utc", "max"), start_local=("ts_local", "min"),
        length_h=("ts_utc", "size"), all_night=("hour", lambda h: h.isin(NIGHT_HOURS).all()),
        weekend_share=("dow", lambda x: (x >= 5).mean()), neighbor_share=("neighbor_missing", "mean"),
    ).reset_index(drop=True)
    runs["start_hour"] = runs.start_local.dt.hour
    runs["start_dow"] = runs.start_local.dt.dayofweek
    return runs


def classify_gaps(runs: pd.DataFrame, max_zero_len: int = 3) -> pd.DataFrame:
    """Правило: «ноль» — короткая ночная серия, соседние станции открыты; всё остальное — «закрытие»."""
    zero = runs.all_night & (runs.length_h <= max_zero_len) & (runs.neighbor_share == 0)
    return runs.assign(kind=np.where(zero, "ноль", "закрытие"))


def gap_labels(mta: pd.DataFrame, runs: pd.DataFrame) -> pd.Series:
    """Метка для каждой строки mta: "" (есть данные), "ноль" или "закрытие"."""
    lab = pd.Series("", index=mta.index, dtype=object)
    by_station = {sid: g for sid, g in runs.groupby("station_id")}
    for sid, idx in mta[mta.is_missing].groupby("station_id").groups.items():
        rows = mta.loc[idx, "ts_utc"]
        for r in by_station[sid].itertuples():
            hit = rows.between(r.start_utc, r.end_utc)
            lab[hit[hit].index] = r.kind
    return lab


def _cell_ratio(mta: pd.DataFrame) -> pd.Series:
    """Поток относительно медианы ячейки (станция × час × тип дня × месяц)."""
    keys = ["station_id", "hour", "day_type", mta.date.dt.to_period("M").rename("ym")]
    return mta.entries / mta.groupby(keys, observed=True).entries.transform("median")


def closure_edges(mta: pd.DataFrame, labels: pd.Series, thr: float = 0.3, max_ext: int = 6) -> pd.Series:
    """Расширяет закрытия на соседние часы, где поток < thr от нормы: закрытие начинается и кончается внутри часа."""
    o = mta.sort_values(["station_id", "ts_utc"]).index
    lab = labels.loc[o].to_numpy(dtype=object).copy()
    low = (_cell_ratio(mta).loc[o] < thr).to_numpy()
    sid, ts = mta.station_id.loc[o].to_numpy(), mta.ts_utc.loc[o].to_numpy()
    linked = np.r_[False, (sid[1:] == sid[:-1]) & (np.diff(ts) == np.timedelta64(1, "h"))]  # i связан с i−1
    for _ in range(max_ext):
        closed = np.isin(lab, ["закрытие", "край закрытия"])
        prev_closed = np.r_[False, closed[:-1]] & linked
        next_closed = np.r_[closed[1:], False] & np.r_[linked[1:], False]
        add = low & (lab == "") & (prev_closed | next_closed)
        if not add.any():
            break
        lab[add] = "край закрытия"
    return pd.Series(lab, index=o).reindex(mta.index)


def gap_rule(mta: pd.DataFrame) -> pd.Series:
    """Итоговая метка строки: "" | "ноль" (заполняем 0) | "закрытие" | "край закрытия" (исключаем из обучения и метрик)."""
    runs = classify_gaps(gap_runs(mta))
    return closure_edges(mta, gap_labels(mta, runs))


def chance_zeros(mta: pd.DataFrame, hours: list[int]) -> float:
    """Сколько нулевых часов ожидалось бы случайно при пуассоновском потоке со средним ячейки."""
    keys = ["station_id", "hour", "day_type", mta.date.dt.to_period("M").rename("ym")]
    lam = mta.groupby(keys, observed=True).entries.transform("mean")
    return float(np.exp(-lam[mta.hour.isin(hours)]).sum())


def boundary_ratio(mta: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Поток за k = 1…4 ч до начала и после конца серии относительно нормы ячейки (медиана по сериям)."""
    v = mta.assign(ratio=_cell_ratio(mta)).set_index(["station_id", "ts_utc"]).ratio
    rows = []
    for k in range(1, 5):
        dt = pd.Timedelta(hours=k)
        before = v.reindex(list(zip(runs.station_id, runs.start_utc - dt))).to_numpy()
        after = v.reindex(list(zip(runs.station_id, runs.end_utc + dt))).to_numpy()
        rows.append({"k": k, "до": np.nanmedian(before), "после": np.nanmedian(after)})
    return pd.DataFrame(rows)


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    o = np.argsort(values.to_numpy())
    v, w = values.to_numpy()[o], weights.to_numpy()[o]
    return float(v[np.searchsorted(np.cumsum(w) / w.sum(), q)])


# --- 8–9. Шум и пересадочные ------------------------------------------------
def noise_frame(mta: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    """Рабочие часы обычных дней: без праздников, пред-/послепраздничных и недели 24.12–01.01."""
    dates = mta.date.drop_duplicates()
    cal = holiday_calendar(hol, dates)
    ok = ~mta.date.isin(cal.index) & ~special_period(mta.date) & mta.hour.isin(WORK_HOURS)
    return mta[ok & mta.entries.notna()]


def slot_cv(nf: pd.DataFrame) -> pd.DataFrame:
    """Коэффициент вариации слота (станция × час × тип дня) внутри квартала — без тренда и сезона."""
    q = nf.date.dt.to_period("Q").rename("q")
    cv = nf.groupby(["station_id", "hour", "day_type", q], observed=True).entries.agg(["mean", "std", "size"])
    cv = cv[cv["size"] >= 5].reset_index()
    cv["cv"] = cv["std"] / cv["mean"]
    return cv.merge(nf[["station_id", "group", "is_transfer_complex"]].drop_duplicates(), on="station_id")


def rolling_deviation(nf: pd.DataFrame, key: str = "day_type", window: str = "56D") -> pd.DataFrame:
    """Отклонение от медианы того же слота за предыдущие 8 недель (бейзлайн из раздела 6 ТЗ)."""
    keys = ["station_id", "hour", key]
    d = nf[keys + ["date", "entries"]].sort_values("date")
    med = (d.groupby(keys, observed=True, sort=False)
           .rolling(window, on="date", closed="left", min_periods=4)["entries"].median()
           .rename("base").reset_index())
    d = d.merge(med, on=keys + ["date"], how="left")
    d["dev"] = d.entries / d.base - 1
    return d.dropna(subset=["dev"])


def transfer_kinds(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Маски «пересадочные / обычные» — по всем станциям и только по Квинсу (чтобы отделить эффект Манхэттена)."""
    queens = df.group != "Манхэттен"
    return {
        "пересадочные (6)": df.is_transfer_complex,
        "обычные (16)": ~df.is_transfer_complex,
        "пересадочные Квинса (3)": df.is_transfer_complex & queens,
        "обычные Квинса (15)": ~df.is_transfer_complex & queens,
    }


def transfer_compare(mta: pd.DataFrame) -> pd.DataFrame:
    """Нормированные профили: пересадочные против обычных — все станции и только Квинс."""
    parts = [daily_profiles(mta[mask].assign(kind=k), by="kind") for k, mask in transfer_kinds(mta).items()]
    return pd.concat(parts, ignore_index=True)


def profile_distance(prof: pd.DataFrame, a: str, b: str, by: str = "kind") -> pd.Series:
    """Доля суточного потока, которую нужно «переставить» между часами, чтобы профиль a совпал с b (TVD)."""
    p = prof.pivot_table(index=["day_type", "hour"], columns=by, values="share", observed=True)
    return (p[a] - p[b]).abs().groupby(level="day_type", observed=True).sum() / 2
