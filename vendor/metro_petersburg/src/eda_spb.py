"""Расчёты для EDA по данным СПб, 1 линия (этап 2б). Каждая функция возвращает таблицу; графики — в notebooks/02_eda_spb.ipynb.

Строки с любым флагом (is_closed_hour, is_vestibule_closed, is_incident) исключены из всех расчётов, кроме п. 7:
в `y` и `r` они NaN, сырой поток остаётся в `entries`, отношение к норме без учёта флагов — в `r_raw`.

Норма b(t) — медиана y того же вестибюля, часа и дня недели за 4 последних обычных таких же дня (поиск до 8 недель
назад; только прошлое; в историю идут только обычные дни без флагов), r = y / b. Сутки метро — 05:00 … 04:59 (`sday`): час 00 относится к прошедшему дню.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config, eda, reference

# --- Константы ----------------------------------------------------------------
SCREENING_OUT = config.DATA / "features_screening.csv"
ANALYSIS_START = pd.Timestamp("2026-02-09")   # 4 полные недели обычных дней после 11.01 — норма определена
JAN_HOLIDAYS_END = pd.Timestamp("2026-01-11")
WORK_HOURS = [*range(6, 24), 0]               # часы работы для r, шума и прогнозов; в 05 ч метро работает полчаса
HOUR_POS = {h: i for i, h in enumerate(WORK_HOURS)}
B_MIN = 20                                    # при b < 20 входов r не считаем (раздел 5 ТЗ)
BASE_WEEKS, BASE_MIN_COUNT = 4, 3
DAY_TYPES = ["рабочий", "суббота", "воскресенье", "праздник"]
MORNING, EVENING = range(6, 11), range(16, 21)
HOUR_BANDS = {"утро 07–09": [7, 8, 9], "день 10–15": list(range(10, 16)), "вечер 16–19": [16, 17, 18, 19],
              "поздно 20–00": [20, 21, 22, 23, 0]}

GROUPS = ["Спальные конечные", "Центр и вокзалы", "Прочие"]
GROUP_COLORS = dict(zip(GROUPS, eda.SERIES))
STATION_GROUP = {
    **dict.fromkeys(["devyatkino", "grazhdansky_prospekt", "akademicheskaya",
                     "avtovo", "leninsky_prospekt", "prospekt_veteranov"], GROUPS[0]),
    **dict.fromkeys(["ploshchad_lenina", "chernyshevskaya", "vosstaniya", "vladimirskaya",
                     "pushkinskaya", "baltiyskaya", "tekhnologichesky_institut"], GROUPS[1]),
    **dict.fromkeys(["politekhnicheskaya", "ploshchad_muzhestva", "lesnaya", "vyborgskaya",
                     "narvskaya", "kirovsky_zavod"], GROUPS[2]),
}

# Учебный период школ СПб, 2025/26 (распоряжение Комитета по образованию СПб: зимние каникулы 31.12–11.01,
# весенние 28.03–05.04; доп. каникулы 1-х классов 16–22.02 не учитываем). Конец учебного года — 26.05
# по федеральной норме (ФОП), отдельно по документу СПб не сверен. 2026/27 — с 01.09.
SCHOOL_TERMS = [("2026-01-12", "2026-03-27"), ("2026-04-06", "2026-05-26"), ("2026-09-01", "2026-09-30")]

# Расходящаяся шкала для r: красный — ниже нормы, синий — выше, серый в центре (палитра dataviz: blue <-> red)
DIVERGING = ["#7d2120", "#b5302f", "#e34948", "#f0a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#256abf", "#0d366b"]


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("spb_div", DIVERGING)


# --- Загрузка -------------------------------------------------------------------
@dataclass
class Grid:
    """Широкие массивы час × вестибюль (вестибюли — в порядке по линии, Девяткино → пр. Ветеранов)."""
    ts: pd.DatetimeIndex     # UTC
    local: pd.DatetimeIndex  # местное время без смещения
    hour: np.ndarray
    sday: pd.DatetimeIndex   # сутки метро
    entries: np.ndarray      # сырой поток
    flagged: np.ndarray      # любой из трёх флагов
    closed: np.ndarray       # метро или вестибюль закрыт (без инцидента)
    y: np.ndarray            # поток без флагов (NaN)
    b: np.ndarray            # норма
    r: np.ndarray            # y / b в часы работы при b ≥ B_MIN
    r_raw: np.ndarray        # entries / b без учёта инцидента (п. 7)


@dataclass
class SpbData:
    df: pd.DataFrame         # длинная таблица: строка = вестибюль × час
    grid: Grid
    vestibules: pd.DataFrame
    stations: pd.DataFrame
    calendar: pd.DataFrame   # по суткам метро: day_type, is_regular, ...
    weather: pd.DataFrame    # архив Open-Meteo (факт)
    forecast: pd.DataFrame   # исторический прогноз Open-Meteo
    events: pd.DataFrame
    ops: dict


def event_days(events: pd.DataFrame) -> pd.Series:
    """Сутки метро события — по середине окна: ночь на 27.06 01–03 ч относится к суткам 26.06."""
    mid = events.start_local + (events.end_local - events.start_local) / 2
    return (mid - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).dt.normalize()


def service_calendar(cal: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Тип дня по производственному календарю и «обычный день»: тип совпадает с днём недели, не 1–11.01, без событий."""
    c = cal.copy()
    c["date"] = pd.to_datetime(c.date)
    holiday = (c.holiday_name != "") | (~c.is_workday & (c.dow < 5))
    c["day_type"] = np.select([holiday, c.is_workday, c.dow == 5], ["праздник", "рабочий", "суббота"], "воскресенье")
    expected = np.select([c.dow < 5, c.dow == 5], ["рабочий", "суббота"], "воскресенье")
    ev = set(event_days(events))
    c["is_event_day"] = c.date.isin(ev)
    c["is_regular"] = (c.day_type == expected) & (c.date > JAN_HOLIDAYS_END) & ~c.is_event_day
    terms = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in SCHOOL_TERMS]
    c["is_school"] = np.any([c.date.between(a, b) for a, b in terms], axis=0)
    return c


def baseline_matrix(y: np.ndarray, usable: np.ndarray, weeks: int = BASE_WEEKS, min_count: int = BASE_MIN_COUNT,
                    max_lookback: int = 2 * BASE_WEEKS, period: int = 168) -> np.ndarray:
    """Медиана последних `weeks` пригодных значений y[t − period·k], k = 1…max_lookback (только прошлое).

    Необычные дни (праздники, события) пропускаются, поэтому неделя после праздника не остаётся без нормы:
    берутся 4 последних обычных таких же дня недели. Меньше min_count значений → NaN.
    """
    hist = np.where(usable, y, np.nan)
    stack = np.full((max_lookback,) + y.shape, np.nan)
    for k in range(1, max_lookback + 1):
        stack[k - 1, k * period:] = hist[:-k * period]
    valid = ~np.isnan(stack)
    stack[valid.cumsum(0) > weeks] = np.nan   # только `weeks` самых свежих
    count = np.minimum(valid.sum(0), weeks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice
        med = np.nanmedian(stack, axis=0)
    return np.where(count >= min_count, med, np.nan)


def ratio(y: np.ndarray, b: np.ndarray, hour: np.ndarray) -> np.ndarray:
    work = np.isin(hour, WORK_HOURS)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(work & (b >= B_MIN), y / b, np.nan)


def make_grid(ts: pd.DatetimeIndex, entries: np.ndarray, closed: np.ndarray, flagged: np.ndarray,
              regular: np.ndarray) -> Grid:
    """Норма и отклонение по сырому потоку; regular — обычные ли сутки метро у каждого часа."""
    local = ts.tz_convert(config.SPB_TZ).tz_localize(None)
    hour = local.hour.to_numpy()
    sday = (local - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).normalize()
    y = np.where(flagged, np.nan, entries)
    b = baseline_matrix(y, ~flagged & regular[:, None])
    return Grid(ts=ts, local=local, hour=hour, sday=sday, entries=entries, flagged=flagged, closed=closed, y=y, b=b,
                r=ratio(y, b, hour), r_raw=ratio(np.where(closed, np.nan, entries), b, hour))


def load() -> SpbData:
    st = reference.load_stations()
    ves = reference.load_vestibules(stations=st).merge(st[["station_id", "name", "line_order"]], on="station_id")
    ves["group"] = ves.station_id.map(STATION_GROUP)
    ves["pos"] = np.arange(len(ves))
    events = reference.load_events(station_ids=st.station_id.tolist())
    cal = service_calendar(pd.read_parquet(config.CALENDAR_OUT), events)

    raw = pd.read_parquet(config.SPB_HOURLY)
    ts = pd.DatetimeIndex(np.sort(raw.ts_utc.unique()))
    wide = lambda col: (raw.pivot(index="ts_utc", columns="vestibule_id", values=col)
                        .reindex(index=ts, columns=ves.vestibule_id).to_numpy())
    entries = wide("entries").astype(float)
    closed = wide("is_closed_hour").astype(bool) | wide("is_vestibule_closed").astype(bool)
    flagged = closed | wide("is_incident").astype(bool)
    local = ts.tz_convert(config.SPB_TZ).tz_localize(None)
    hour = local.hour.to_numpy()
    sday = (local - pd.Timedelta(hours=config.SPB_SERVICE_DAY_START)).normalize()
    regular = cal.set_index("date").is_regular.reindex(sday).eq(True).to_numpy()

    g = make_grid(ts, entries, closed, flagged, regular)

    T, V = entries.shape
    df = pd.DataFrame({
        "vestibule_id": np.tile(ves.vestibule_id.to_numpy(), T),
        "pos": np.tile(ves.pos.to_numpy(), T),
        "ts_utc": np.repeat(ts, V), "local": np.repeat(local, V), "sday": np.repeat(sday, V),
        "hour": np.repeat(hour, V),
        "entries": entries.ravel(), "y": g.y.ravel(), "b": g.b.ravel(), "r": g.r.ravel(), "r_raw": g.r_raw.ravel(),
        "flagged": flagged.ravel(),
    })
    df = df.merge(ves[["vestibule_id", "station_id", "raw_name", "group", "line_order"]], on="vestibule_id")
    df = df.merge(cal[["date", "dow", "day_type", "is_regular", "is_workday", "is_shortened", "is_school"]]
                  .rename(columns={"date": "sday"}), on="sday", how="left")
    df["month"] = df.sday.dt.month
    df = df.sort_values(["ts_utc", "pos"], ignore_index=True)

    return SpbData(df=df, grid=g, vestibules=ves, stations=st, calendar=cal,
                   weather=pd.read_parquet(config.INTERIM / "weather_archive_spb.parquet"),
                   forecast=pd.read_parquet(config.INTERIM / "weather_hist_forecast_spb.parquet"),
                   events=events, ops=reference.load_operations())


def ok(df: pd.DataFrame) -> pd.Series:
    """Строки без флагов."""
    return ~df.flagged


def line_ratio(g: Grid, cols: np.ndarray | None = None) -> np.ndarray:
    """r всей линии (или набора вестибюлей): Σ y / Σ b по вестибюлям, где r определён."""
    m = ~np.isnan(g.r)
    if cols is not None:
        m = m & cols[None, :]
    y, b = np.where(m, g.y, 0).sum(1), np.where(m, g.b, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b >= B_MIN, y / b, np.nan)


def station_ratio(g: Grid, ves: pd.DataFrame) -> pd.DataFrame:
    """r станций (Σ по вестибюлям станции), колонки — станции по линии."""
    out = {}
    for sid, idx in ves.groupby("station_id", sort=False).pos:
        cols = np.zeros(len(ves), bool)
        cols[idx.to_numpy()] = True
        out[sid] = line_ratio(g, cols)
    return pd.DataFrame(out, index=g.ts)


# --- Статистика -----------------------------------------------------------------
def bh(p) -> np.ndarray:
    """Поправка Бенджамини–Хохберга (скорректированные p); NaN остаются NaN."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok_ = ~np.isnan(p)
    q = p[ok_]
    n = len(q)
    o = np.argsort(q)
    adj = q[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.minimum(adj, 1)
    out[ok_] = res
    return out


def boot_counts(n_days: int, n_boot: int, rng) -> np.ndarray:
    """Блочный бутстреп по дням: сколько раз каждый день попал в выборку, [n_boot, n_days]."""
    idx = rng.integers(0, n_days, (n_boot, n_days))
    return np.stack([np.bincount(i, minlength=n_days) for i in idx]).astype(float)


def strata_perms(strata: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """Перестановки дней внутри страт (месяц × рабочий / нерабочий), [n_perm, n_days]."""
    D = len(strata)
    perms = np.tile(np.arange(D), (n_perm, 1))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        perms[:, idx] = idx[np.argsort(rng.random((n_perm, len(idx))), axis=1)]
    return perms


def _ranks(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    m = ~np.isnan(a)
    out[m] = stats.rankdata(a[m])
    return out


def _pearson(n, sx, sy, sxx, syy, sxy):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def _gram(X: np.ndarray, Y: np.ndarray) -> dict:
    """Суммы для корреляции при любой паре дней (d, d'): G[d, d'] = Σ_слотов f(X[d]) · g(Y[d'])."""
    mx, my = (~np.isnan(X)).astype(float), (~np.isnan(Y)).astype(float)
    x, y = np.nan_to_num(X), np.nan_to_num(Y)
    return {"n": mx @ my.T, "sx": x @ my.T, "sy": mx @ y.T, "sxx": (x * x) @ my.T,
            "syy": mx @ (y * y).T, "sxy": x @ y.T}


def rank_corr_test(X: np.ndarray, Y: np.ndarray, perms: np.ndarray, counts: np.ndarray,
                   months: np.ndarray) -> dict:
    """Спирмен (ранги фиксированы по всей выборке) для матриц день × слот: оценка, ДИ бутстрепом по дням,
    p перестановочным тестом (дни X сопоставляются с днями Y по перестановке), оценка по месяцам.

    Перестановочная статистика считается за O(дней) через матрицы сумм по парам дней.
    """
    G = _gram(_ranks(X), _ranks(Y))
    D = X.shape[0]
    diag = {k: np.diag(v) for k, v in G.items()}
    est = _pearson(**{k: v.sum() for k, v in diag.items()})
    boot = _pearson(**{k: counts @ v for k, v in diag.items()})
    rows = np.arange(D)
    null = _pearson(**{k: v[rows, perms].sum(1) for k, v in G.items()})
    c = np.nanmean(null)   # центр нулевого распределения: связь, которую сохраняет перестановка
    p = (1 + np.sum(np.abs(null - c) >= abs(est - c))) / (1 + len(null))
    by_month = {m: _pearson(**{k: v[months == m].sum() for k, v in diag.items()}) for m in np.unique(months)}
    return {"effect": est, "lo": np.nanpercentile(boot, 2.5), "hi": np.nanpercentile(boot, 97.5), "p": p,
            "null_mean": c, "by_month": by_month}


def weighted_median_boot(vals: np.ndarray, days: np.ndarray, counts: np.ndarray, chunk: int = 50) -> np.ndarray:
    """Медиана vals при весах «сколько раз день попал в бутстреп-выборку» — для каждого повтора."""
    o = np.argsort(vals)
    v, dd = vals[o], days[o]
    out = np.empty(len(counts))
    for i in range(0, len(counts), chunk):
        w = counts[i:i + chunk][:, dd]
        cw = w.cumsum(1)
        idx = (cw < cw[:, -1:] / 2).sum(1)
        out[i:i + chunk] = v[np.minimum(idx, len(v) - 1)]
    return out


def median_diff_test(X: np.ndarray, Y: np.ndarray, perms: np.ndarray, counts: np.ndarray,
                     months: np.ndarray) -> dict:
    """Разница медиан Y при X = 1 и X = 0 (матрицы день × слот): ДИ бутстрепом по дням, p перестановкой дней."""
    D = X.shape[0]
    dmat = np.repeat(np.arange(D)[:, None], X.shape[1], 1)
    valid = ~np.isnan(X) & ~np.isnan(Y)
    m1, m0 = valid & (X == 1), valid & (X == 0)
    est = np.median(Y[m1]) - np.median(Y[m0])
    boot = weighted_median_boot(Y[m1], dmat[m1], counts) - weighted_median_boot(Y[m0], dmat[m0], counts)
    x1, x0 = X == 1, X == 0
    null = np.empty(len(perms))
    for i, pi in enumerate(perms):
        yp = Y[pi]
        vp = ~np.isnan(yp)
        a, b = yp[x1 & vp], yp[x0 & vp]
        null[i] = np.median(a) - np.median(b) if len(a) and len(b) else np.nan
    c = np.nanmean(null)
    p = (1 + np.nansum(np.abs(null - c) >= abs(est - c))) / (1 + np.sum(~np.isnan(null)))
    by_month = {}
    for m in np.unique(months):
        mm = (months == m)[:, None]
        a, b = Y[m1 & mm], Y[m0 & mm]
        days_pos = np.unique(dmat[m1 & mm]).size
        by_month[m] = np.median(a) - np.median(b) if days_pos >= 3 and len(b) else np.nan
    return {"effect": est, "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "p": p,
            "null_mean": c, "by_month": by_month}


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """Тест Диболда–Мариано: H0 — одинаковая точность; дисперсия — Ньюи–Уэст с лагом lag. Возвращает (stat, p)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def wape(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.abs(y - f).sum() / y.sum())


def compare_forecasts(y: np.ndarray, base: np.ndarray, alt: np.ndarray, day: np.ndarray,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """ΔWAPE = WAPE(base) − WAPE(alt) (> 0 — alt лучше): ДИ парным бутстрепом по дням, p — Диболд–Мариано."""
    days, inv = np.unique(day, return_inverse=True)
    ea = np.bincount(inv, np.abs(y - base))
    eb = np.bincount(inv, np.abs(y - alt))
    sy = np.bincount(inv, y)
    rng = np.random.default_rng(seed)
    c = boot_counts(len(days), n_boot, rng)
    boot = (c @ ea - c @ eb) / (c @ sy)
    _, p = dm_test(ea, eb)
    return {"wape_base": ea.sum() / sy.sum(), "wape_alt": eb.sum() / sy.sum(), "dwape": (ea.sum() - eb.sum()) / sy.sum(),
            "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "p_dm": p, "n_days": len(days)}


def day_bootstrap_ratio(num: pd.Series, den: pd.Series, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Σ num / Σ den − 1 с 95 % ДИ бутстрепом по дням (индекс — дни)."""
    rng = np.random.default_rng(seed)
    c = boot_counts(len(num), n_boot, rng)
    boot = (c @ num.to_numpy()) / (c @ den.to_numpy()) - 1
    return num.sum() / den.sum() - 1, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# --- 1. Профили -----------------------------------------------------------------
def daily_profiles(d: SpbData, by: str = "group") -> pd.DataFrame:
    """Средний поток на вестибюль по часам и доля часа в сутках: обычные дни по типам и праздники."""
    df = d.df[ok(d.df) & ((d.df.is_regular & (d.df.day_type != "праздник")) | (d.df.day_type == "праздник"))]
    df = df[df.sday >= d.calendar.date.min()]
    prof = df.groupby([by, "day_type", "hour"], observed=True).entries.mean().rename("mean").reset_index()
    prof["share"] = prof["mean"] / prof.groupby([by, "day_type"], observed=True)["mean"].transform("sum")
    return prof


def peak_ratios(d: SpbData, by: str = "vestibule_id") -> pd.DataFrame:
    """Утренний пик (06–10) против вечернего (16–20) по среднему профилю обычных рабочих дней."""
    prof = daily_profiles(d, by)
    p = prof[prof.day_type == "рабочий"].pivot(index=by, columns="hour", values="mean")
    m, e = p[list(MORNING)], p[list(EVENING)]
    return pd.DataFrame({"morning": m.max(1), "morning_hour": m.idxmax(1), "evening": e.max(1),
                         "evening_hour": e.idxmax(1), "ratio": m.max(1) / e.max(1)})


# --- 2. Дни недели --------------------------------------------------------------
def regular_daily(d: SpbData, day_type: str = "рабочий", by: str | None = None) -> pd.DataFrame:
    """Суточный поток обычных дней типа day_type: вся линия (и по группам, если by)."""
    df = d.df[ok(d.df) & d.df.is_regular & (d.df.day_type == day_type)]
    keys = ["sday"] + ([by] if by else [])
    return df.groupby(keys, observed=True).entries.sum().rename("total").reset_index()


def weekday_levels(d: SpbData, n_boot: int = 1000) -> pd.DataFrame:
    """Средние обычные рабочие сутки по дням недели относительно вт–чт, ДИ бутстрепом по дням."""
    daily = pd.concat([regular_daily(d).assign(level="Вся линия"),
                       regular_daily(d, by="group").rename(columns={"group": "level"})], ignore_index=True)
    daily["dow"] = daily.sday.dt.dayofweek
    rows = []
    rng = np.random.default_rng(0)
    for lvl, g in daily.groupby("level"):
        mid = g[g.dow.isin([1, 2, 3])].total
        for dow, gd in g.groupby("dow"):
            base = mid.mean()
            boot = [rng.choice(gd.total.to_numpy(), len(gd)).mean() / rng.choice(mid.to_numpy(), len(mid)).mean() - 1
                    for _ in range(n_boot)]
            rows.append({"level": lvl, "dow": dow, "effect": gd.total.mean() / base - 1,
                         "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "n_days": len(gd)})
    return pd.DataFrame(rows)


def baseline_daytype(g: Grid, cal: pd.DataFrame, window_days: int = 28, min_count: int = 3) -> np.ndarray:
    """Альтернативная норма: медиана того же часа по обычным дням того же типа за 28 прошлых суток."""
    c = cal.set_index("date").reindex(g.sday)
    dtype, regular = c.day_type.to_numpy(), c.is_regular.eq(True).to_numpy()
    usable = ~g.flagged & regular[:, None]
    hist = np.where(usable, g.y, np.nan)
    out = np.full(g.y.shape, np.nan)
    sd = g.sday.to_numpy()
    for h in range(24):
        idx = np.flatnonzero(g.hour == h)
        for i in idx:
            if not regular[i]:
                continue
            lo = sd[i] - np.timedelta64(window_days, "D")
            j = idx[(sd[idx] >= lo) & (sd[idx] < sd[i]) & (dtype[idx] == dtype[i])]
            vals = hist[j]
            cnt = (~np.isnan(vals)).sum(0)
            if len(j):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[i] = np.where(cnt >= min_count, np.nanmedian(vals, axis=0), np.nan)
    return out


def noise_by_key(d: SpbData) -> pd.DataFrame:
    """Шум |r − 1| на обычных днях при норме по дню недели (основная) и по типу дня — взвешено по потоку."""
    g = d.grid
    r_dt = ratio(g.y, baseline_daytype(g, d.calendar), g.hour)
    c = d.calendar.set_index("date").reindex(g.sday)
    keep = (c.is_regular.eq(True).to_numpy() & (g.sday >= ANALYSIS_START))[:, None]
    rows = []
    for key, r in (("день недели", g.r), ("тип дня", r_dt)):
        m = keep & ~np.isnan(g.r) & ~np.isnan(r_dt)   # одинаковые строки для обеих норм
        dev, w = np.abs(r[m] - 1), g.y[m]
        for dtp in ("рабочий", "суббота", "воскресенье", "все"):
            sel = np.ones(m.sum(), bool) if dtp == "все" else (np.broadcast_to(c.day_type.to_numpy()[:, None], m.shape)[m] == dtp)
            rows.append({"key": key, "day_type": dtp,
                         "p50": eda.weighted_quantile(pd.Series(dev[sel]), pd.Series(w[sel]), 0.5),
                         "p80": eda.weighted_quantile(pd.Series(dev[sel]), pd.Series(w[sel]), 0.8),
                         "p95": eda.weighted_quantile(pd.Series(dev[sel]), pd.Series(w[sel]), 0.95), "n": int(sel.sum())})
    return pd.DataFrame(rows)


# --- 3. Тренд -------------------------------------------------------------------
def monthly_trend(d: SpbData) -> pd.DataFrame:
    """Средние обычные рабочие сутки по месяцам: вся линия и группы."""
    parts = [regular_daily(d).assign(level="Вся линия"),
             regular_daily(d, by="group").rename(columns={"group": "level"})]
    daily = pd.concat(parts, ignore_index=True)
    daily["month"] = daily.sday.dt.month
    return daily.groupby(["level", "month"]).total.agg(mean="mean", n_days="size").reset_index()


def weekly_level(d: SpbData) -> pd.Series:
    daily = regular_daily(d).set_index("sday").total
    return daily.resample("W-SUN").mean()


def september_step(d: SpbData, n: int = 10) -> pd.DataFrame:
    """Ступень 01.09: n обычных рабочих дней после против n до (линия и группы)."""
    sep = pd.Timestamp("2026-09-01")
    rows = []
    for lvl, daily in [("Вся линия", regular_daily(d))] + [
            (gname, g) for gname, g in regular_daily(d, by="group").groupby("group")]:
        s = daily.set_index("sday").total
        before, after = s[s.index < sep].tail(n), s[s.index >= sep].head(n)
        rows.append({"level": lvl, "before": before.mean(), "after": after.mean(), "step": after.mean() / before.mean() - 1,
                     "before_from": before.index.min(), "after_to": after.index.max()})
    return pd.DataFrame(rows)


# --- 4. Праздники ---------------------------------------------------------------
def line_daily(d: SpbData) -> pd.Series:
    df = d.df[ok(d.df)]
    return df.groupby("sday").entries.sum()


def same_dow_base(daily: pd.Series, cal: pd.DataFrame, day: pd.Timestamp, weeks=(-2, -1, 1, 2)) -> float:
    regular = set(cal.loc[cal.is_regular, "date"])
    adj = [day + pd.Timedelta(weeks=k) for k in weeks]
    adj = [a for a in adj if a in regular and a in daily.index]
    return daily[adj].mean() if adj else np.nan


def special_days(cal: pd.DataFrame) -> pd.DataFrame:
    """Праздники, переносы, каникулы 1–11.01, первый рабочий день после них и сокращённые дни."""
    c = cal[(cal.date >= "2026-01-01") & (cal.date <= "2026-09-29")]
    sel = (c.day_type == "праздник") | c.is_shortened | (c.date <= "2026-01-12")
    out = c[sel].copy()
    out["kind"] = np.select(
        [out.is_transfer_dayoff, out.holiday_name != "", out.is_shortened, out.date == "2026-01-12"],
        ["перенос", "праздник", "сокращённый", "первый рабочий"], "каникулы")
    out["label"] = np.where(out.holiday_name != "", out.holiday_name, out.kind)
    return out[["date", "dow", "day_type", "kind", "label"]].reset_index(drop=True)


def holiday_effects(d: SpbData) -> pd.DataFrame:
    """Сутки линии в особые дни против того же дня недели ±1, ±2 недели (обычные дни)."""
    daily = line_daily(d)
    sp = special_days(d.calendar)
    sp["total"] = sp.date.map(daily)
    sp["base"] = [same_dow_base(daily, d.calendar, x) for x in sp.date]
    sp["effect"] = sp.total / sp.base - 1
    return sp


def hourly_vs_base(d: SpbData, day: pd.Timestamp) -> pd.Series:
    """Поток линии по часам в день day против того же дня недели ±1, ±2 недели (обычные дни)."""
    df = d.df[ok(d.df)]
    line = df.groupby(["sday", "hour"]).entries.sum().unstack()
    regular = set(d.calendar.loc[d.calendar.is_regular, "date"])
    adj = [day + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
    adj = [a for a in adj if a in regular and a in line.index]
    return (line.loc[day] / line.loc[adj].mean())[WORK_HOURS]


def holiday_profile(d: SpbData) -> pd.DataFrame:
    """Доля суток по часам: нерабочие будние праздники против обычных воскресений, суббот и рабочих дней тех же месяцев."""
    df = d.df[ok(d.df) & d.df.hour.isin(WORK_HOURS)]
    cal = d.calendar
    hol = cal[(cal.day_type == "праздник") & (cal.dow < 5) & (cal.date > "2026-01-02")]
    months = set(hol.date.dt.month)
    reg = cal[cal.is_regular & cal.date.dt.month.isin(months)]
    sets = {"праздник (будний)": set(hol.date),
            "обычное воскресенье": set(reg.loc[reg.day_type == "воскресенье", "date"]),
            "обычная суббота": set(reg.loc[reg.day_type == "суббота", "date"]),
            "обычный рабочий день": set(reg.loc[reg.day_type == "рабочий", "date"])}
    line = df.groupby(["sday", "hour"]).entries.sum().rename("v").reset_index()
    line["share"] = line.v / line.groupby("sday").v.transform("sum")
    return pd.concat({k: line[line.sday.isin(v)].groupby("hour").share.mean() for k, v in sets.items()},
                     axis=1).loc[WORK_HOURS]


# --- 5. Погода ------------------------------------------------------------------
FROST_C, HEAT_C = -15, 28
PRECIP_CATS = {"rain": "дождь ≥ 0,5 мм/ч", "snow": "снег > 0"}
TEMP_CATS = {"frost": f"мороз < {FROST_C} °C", "heat": f"жара > +{HEAT_C} °C"}


def weather_flags(w: pd.DataFrame) -> pd.DataFrame:
    f = eda.weather_flags(w)
    t = w.temperature_2m.to_numpy()
    f["frost"], f["heat"] = t < FROST_C, t > HEAT_C
    f["mild"] = (t >= FROST_C) & (t <= HEAT_C) & f.dry
    return f


def mta_like(d: SpbData) -> pd.DataFrame:
    """Таблица в колонках MTA для eda.weather_effect: обычные дни, без флагов; рабочий → «будни»."""
    df = d.df[ok(d.df) & d.df.is_regular]
    return pd.DataFrame({"station_id": df.vestibule_id, "ts_utc": df.ts_utc, "hour": df.hour, "entries": df.entries,
                         "date": df.sday, "day_type": df.day_type.replace({"рабочий": "будни"})})


def weather_effects(d: SpbData, w: pd.DataFrame) -> pd.DataFrame:
    """Эффект погоды к норме ячейки «вестибюль × час × тип дня × месяц»: осадки — к сухим часам, T — к умеренным."""
    m, f = mta_like(d), weather_flags(w)
    return pd.concat([eda.weather_effect(m, f, cats=PRECIP_CATS, ref="dry"),
                      eda.weather_effect(m, f, cats=TEMP_CATS, ref="mild")], ignore_index=True)


# --- 6. События -----------------------------------------------------------------
def event_matrix(d: SpbData, day: pd.Timestamp, hours=WORK_HOURS) -> pd.DataFrame:
    """r по вестибюлям (строки, по линии) и часам суток метро day."""
    df = d.df[(d.df.sday == day) & d.df.hour.isin(hours)]
    m = df.pivot(index="raw_name", columns="hour", values="r")
    return m.reindex(index=d.vestibules.raw_name, columns=hours)


def event_daily(d: SpbData, day: pd.Timestamp) -> pd.Series:
    """Σ y / Σ b за часы работы по вестибюлям в сутки day."""
    df = d.df[(d.df.sday == day) & d.df.r.notna()]
    s = df.groupby("raw_name").apply(lambda x: x.y.sum() / x.b.sum(), include_groups=False)
    return s.reindex(d.vestibules.raw_name)


def first_september(d: SpbData) -> pd.DataFrame:
    """1 сентября по вестибюлям: к норме (август) и к обычным рабочим дням 2–8 сентября."""
    df = d.df[ok(d.df) & d.df.hour.isin(WORK_HOURS)]
    sep1 = df[df.sday == "2026-09-01"].groupby("raw_name").entries.sum()
    week = df[df.sday.between("2026-09-02", "2026-09-08") & df.is_regular & (df.day_type == "рабочий")]
    wk = week.groupby(["raw_name", "sday"]).entries.sum().groupby("raw_name").mean()
    out = pd.DataFrame({"к норме": event_daily(d, pd.Timestamp("2026-09-01")), "к 2–8 сентября": sep1 / wk})
    return out.reindex(d.vestibules.raw_name)


def night_evenings(d: SpbData, hours=(18, 19, 20, 21, 22, 23, 0)) -> pd.DataFrame:
    """r линии вечером перед особыми ночами (сутки метро, в которые попадает ночное окно)."""
    g = d.grid
    rl = pd.Series(line_ratio(g), index=g.ts)
    nights = d.events[d.events.source == "по данным потока"]
    rows = {}
    for e, sd in zip(nights.itertuples(), event_days(nights)):
        m = (g.sday == sd) & np.isin(g.hour, hours)
        s = pd.Series(rl[m].to_numpy(), index=g.hour[m])
        rows[f"ночь на {e.start_local:%d.%m}: {e.event}"] = s.reindex(list(hours))
    return pd.DataFrame(rows).T


# --- 7. Инцидент ----------------------------------------------------------------
def incident_matrix(d: SpbData, day: str = "2026-08-31", hours=range(6, 14)) -> pd.DataFrame:
    """Доля от нормы понедельников (r_raw, флаг инцидента не исключаем) по вестибюлям и часам."""
    df = d.df[(d.df.sday == day) & d.df.hour.isin(list(hours))]
    return df.pivot(index="raw_name", columns="hour", values="r_raw").reindex(d.vestibules.raw_name)


def dip_candidates(d: SpbData, window: int = 3, thr_low: float = 0.75, thr_rebound: float = 1.05,
                   lookahead: int = 3, hours=range(6, 22)) -> pd.DataFrame:
    """Похожие на инцидент провалы: окно из `window` соседних станций, где Σy/Σb < thr_low, а в следующие
    1…lookahead ч в том же окне Σy/Σb > thr_rebound (откат). Тяжесть — «потерянные» входы Σb − Σy в окне;
    лучший час на сутки. Флаг инцидента не исключаем.
    """
    g, ves = d.grid, d.vestibules
    st_order = ves.drop_duplicates("station_id").station_id.tolist()
    yy = np.where(g.closed | np.isnan(g.r_raw), np.nan, g.entries)
    bb = np.where(np.isnan(yy), np.nan, g.b)
    ys = pd.DataFrame(yy).T.groupby(ves.station_id.to_numpy()).sum(min_count=1).T[st_order].to_numpy()
    bs = pd.DataFrame(bb).T.groupby(ves.station_id.to_numpy()).sum(min_count=1).T[st_order].to_numpy()
    T, S = ys.shape
    names = d.stations.set_index("station_id").name
    rows = []
    hour_ok = np.isin(g.hour, list(hours)) & (g.sday >= ANALYSIS_START)
    for i in np.flatnonzero(hour_ok):
        for s0 in range(S - window + 1):
            yw, bw = ys[i, s0:s0 + window], bs[i, s0:s0 + window]
            if np.isnan(yw).any() or np.nansum(bw) < 100:
                continue
            rw = yw.sum() / bw.sum()
            if rw >= thr_low:
                continue
            ahead = []
            for k in range(1, lookahead + 1):
                if i + k < T:
                    ya, ba = ys[i + k, s0:s0 + window], bs[i + k, s0:s0 + window]
                    if not np.isnan(ya).any() and np.nansum(ba) > 0:
                        ahead.append(ya.sum() / ba.sum())
            if not ahead or max(ahead) <= thr_rebound:
                continue
            rows.append({"local": g.local[i], "sday": g.sday[i], "hour": int(g.hour[i]),
                         "stations": " — ".join(names[st_order[s0:s0 + window]].tolist()),
                         "r_window": rw, "r_min_station": float(np.nanmin(yw / bw)),
                         "rebound": max(ahead), "lost": bw.sum() - yw.sum()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.sort_values("lost", ascending=False).drop_duplicates("sday")
    cal = d.calendar.set_index("date")
    best["day_type"] = best.sday.map(cal.day_type)
    best["is_regular"] = best.sday.map(cal.is_regular) & ~best.sday.map(cal.is_shortened)
    return best.reset_index(drop=True)


# --- 8. Динамика отклонения -------------------------------------------------------
def ewm_ratio(r: np.ndarray, ts: pd.DatetimeIndex, halflife_h: float) -> np.ndarray:
    """Экспоненциальное среднее r по времени (ночь затухает сама), состояние на конец каждого часа."""
    out = np.full(r.shape, np.nan)
    for v in range(r.shape[1]):
        s = pd.Series(r[:, v], index=ts).dropna()
        if s.empty:
            continue
        e = s.ewm(halflife=pd.Timedelta(hours=halflife_h), times=s.index).mean()
        out[:, v] = e.reindex(ts).ffill().to_numpy()
    return out


def persistence(d: SpbData, lags=(1, 2, 3)) -> pd.DataFrame:
    """Корреляция log r(t) и log r(t − k) внутри суток метро, по вестибюлям."""
    g = d.grid
    lr = np.log(g.r)
    keep = (g.sday >= ANALYSIS_START)
    rows = []
    for k in lags:
        same = np.r_[np.zeros(k, bool), g.sday[k:] == g.sday[:-k]] & keep
        for v, vid in enumerate(d.vestibules.raw_name):
            a, b = lr[k:, v], lr[:-k, v]
            m = same[k:] & ~np.isnan(a) & ~np.isnan(b)
            rows.append({"lag": k, "vestibule": vid, "rho": np.corrcoef(a[m], b[m])[0, 1], "n": int(m.sum())})
    return pd.DataFrame(rows)


def conditional_next(d: SpbData, thr: float = 1.15, horizons=(1, 2)) -> pd.DataFrame:
    """Распределение r(t+h) при r(t) > thr против безусловного (те же сутки метро)."""
    g = d.grid
    keep = (g.sday >= ANALYSIS_START)[:, None]
    rows = []
    for h in horizons:
        now, nxt = g.r[:-h], g.r[h:]
        same = (g.sday[h:] == g.sday[:-h])[:, None] & keep[h:]
        m = same & ~np.isnan(now) & ~np.isnan(nxt)
        for cond, sel in (("все часы", m), (f"r(t) > {thr}", m & (now > thr))):
            v = nxt[sel]
            rows.append({"h": h, "condition": cond, "n": len(v), "p10": np.percentile(v, 10), "p50": np.median(v),
                         "p90": np.percentile(v, 90), "share_gt_thr": float((v > thr).mean())})
    return pd.DataFrame(rows)


def cumulative_ratio(d: SpbData) -> np.ndarray:
    """Σ y / Σ b с начала суток метро (05:00) по конец часа t, по вестибюлям; NaN, пока Σ b < B_MIN."""
    g = d.grid
    m = ~np.isnan(g.y) & ~np.isnan(g.b)
    y, b = np.where(m, g.y, 0), np.where(m, g.b, 0)
    key = g.sday.to_numpy()
    cy = pd.DataFrame(y).groupby(key).cumsum().to_numpy()
    cb = pd.DataFrame(b).groupby(key).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cb >= B_MIN, cy / cb, np.nan)


def morning_vs_evening(d: SpbData, cum: np.ndarray | None = None) -> pd.DataFrame:
    """Σy/Σb с открытия к концу 09 ч против Σy/Σb в 17–18 ч того же дня, по вестибюлям и суткам."""
    g = d.grid
    cum = cumulative_ratio(d) if cum is None else cum
    m9 = (g.hour == 9) & (g.sday >= ANALYSIS_START)
    morning = pd.DataFrame(cum[m9], index=g.sday[m9], columns=d.vestibules.raw_name).stack().rename("morning")
    ev = np.isin(g.hour, [17, 18]) & (g.sday >= ANALYSIS_START)
    m = ~np.isnan(g.r[ev])
    ysum = pd.DataFrame(np.where(m, g.y[ev], 0), index=g.sday[ev]).groupby(level=0).sum()
    bsum = pd.DataFrame(np.where(m, g.b[ev], 0), index=g.sday[ev]).groupby(level=0).sum()
    evening = (ysum / bsum.where(bsum >= B_MIN)).set_axis(d.vestibules.raw_name, axis=1).stack().rename("evening")
    out = pd.concat([morning, evening], axis=1).dropna()
    out.index.names = ["sday", "vestibule"]
    return out.reset_index()


def target_rows(d: SpbData, h: int) -> dict:
    """Строки прогноза на час τ = t + h: τ в часах работы, без флагов, b(τ) ≥ B_MIN, с ANALYSIS_START."""
    g = d.grid
    T, V = g.y.shape
    tau_ok = (~np.isnan(g.r)) & (g.sday >= ANALYSIS_START)[:, None]
    tau, v = np.nonzero(tau_ok)
    keep = tau - h >= 0
    tau, v = tau[keep], v[keep]
    days, dix = np.unique(g.sday[tau], return_inverse=True)
    days = pd.DatetimeIndex(days)
    hp = np.array([HOUR_POS[x] for x in g.hour[tau]])
    return {"t": tau - h, "tau": tau, "v": v, "y": g.y[tau, v], "b": g.b[tau, v],
            "target": np.log((g.y[tau, v] + 1) / (g.b[tau, v] + 1)),
            "day": dix, "days": days, "slot": v * len(WORK_HOURS) + hp, "n_slots": V * len(WORK_HOURS),
            "month": days.month.to_numpy()[dix]}


def simple_forecasts(d: SpbData, horizons=(1, 2), clip=(0.3, 3.0)) -> pd.DataFrame:
    """WAPE на t+1, t+2: b; b × r(t); b × сглаженное r (полупериод 2 и 6 ч). Где r(t) нет — b."""
    g = d.grid
    variants = {"b × r(t)": g.r, "b × сглаж. r (2 ч)": ewm_ratio(g.r, g.ts, 2),
                "b × сглаж. r (6 ч)": ewm_ratio(g.r, g.ts, 6)}
    rows = []
    for h in horizons:
        tr = target_rows(d, h)
        base = tr["b"]
        for name, arr in variants.items():
            k = np.clip(np.nan_to_num(arr[tr["t"], tr["v"]], nan=1.0), *clip)
            for scope, sel in [("все месяцы", np.ones(len(base), bool))] + [
                    (m, tr["month"] == m) for m in np.unique(tr["month"])]:
                res = compare_forecasts(tr["y"][sel], base[sel], base[sel] * k[sel], tr["day"][sel])
                rows.append({"h": h, "forecast": name, "scope": scope, **res})
    return pd.DataFrame(rows)


# --- 9. Синхронность --------------------------------------------------------------
def sync(d: SpbData) -> dict:
    """Корреляции log r: вестибюль × вестибюль, с соседями по расстоянию на линии и с r всей линии (обычные дни)."""
    g = d.grid
    c = d.calendar.set_index("date").reindex(g.sday)
    keep = c.is_regular.eq(True).to_numpy() & (g.sday >= ANALYSIS_START)
    lr = pd.DataFrame(np.log(g.r[keep]), columns=d.vestibules.raw_name)
    corr = lr.corr()
    lo = d.vestibules.set_index("raw_name").line_order
    pairs = [{"a": a, "b": b, "dist": abs(lo[a] - lo[b]), "corr": corr.at[a, b]}
             for i, a in enumerate(corr.index) for b in corr.index[i + 1:]]
    pairs = pd.DataFrame(pairs)
    rl = np.log(line_ratio(g)[keep])
    with_line = lr.apply(lambda s: s.corr(pd.Series(rl, index=s.index)))
    return {"matrix": corr, "pairs": pairs, "with_line": with_line}


# --- 10. Шум ------------------------------------------------------------------------
def noise_table(d: SpbData) -> pd.DataFrame:
    """|r − 1| на обычных днях, взвешено по потоку: p50 / p80 / p90 / p95 по группам и периодам суток."""
    df = d.df[d.df.is_regular & d.df.r.notna() & (d.df.sday >= ANALYSIS_START)]
    df = df.assign(dev=(df.r - 1).abs())
    rows = []
    groups = [("Вся линия", df)] + list(df.groupby("group"))
    bands = [("все часы", WORK_HOURS)] + list(HOUR_BANDS.items())
    for gname, gd in groups:
        for band, hrs in bands:
            x = gd[gd.hour.isin(hrs)]
            q = {f"p{int(p * 100)}": eda.weighted_quantile(x.dev, x.y, p) for p in (0.5, 0.8, 0.9, 0.95)}
            rows.append({"level": gname, "band": band, **q, "p80_unweighted": x.dev.quantile(0.8), "n": len(x)})
    return pd.DataFrame(rows)


# --- 11. Ёмкость --------------------------------------------------------------------
def capacity(d: SpbData) -> pd.DataFrame:
    """Часовой вход линии (медиана обычных рабочих дней) против парность × вместимость по плану, на направление."""
    df = d.df[ok(d.df) & d.df.is_regular & (d.df.day_type == "рабочий") & d.df.hour.isin(WORK_HOURS)]
    line = df.groupby(["sday", "hour"]).entries.sum().rename("v").reset_index()
    cap = d.ops["train_capacity"]["used"]
    plan = d.ops["pairs_plan"]
    periods = {"сентябрь (с 02.09)": (line.sday >= "2026-09-02", plan["weekday_from_2026_09_01"]),
               "лето (июнь–август)": (line.sday.dt.month.isin([6, 7, 8]), plan["weekday_summer_from_2026_06_01"])}
    rows = []
    for name, (mask, pairs) in periods.items():
        med = line[mask].groupby("hour").v.median()
        for h in WORK_HOURS:
            rows.append({"period": name, "hour": h, "entries": med.get(h, np.nan), "pairs": pairs[h],
                         "capacity": pairs[h] * cap, "load": med.get(h, np.nan) / (pairs[h] * cap)})
    return pd.DataFrame(rows)


# --- 12. Отбор кандидатов в признаки ----------------------------------------------
@dataclass
class Candidate:
    name: str
    group: str
    kind: str        # "num" | "bin"
    desc: str


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[t − k] (k > 0 — прошлое, k < 0 — будущее), по первой оси."""
    out = np.full(a.shape, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _rolling_std(a: np.ndarray, w: int, min_periods: int) -> np.ndarray:
    return pd.DataFrame(a).rolling(w, min_periods=min_periods).std().to_numpy()


def _per_hour(values: np.ndarray, V: int) -> np.ndarray:
    return np.repeat(values[:, None], V, 1)


def candidate_matrices(d: SpbData, h: int, seed: int = 0) -> dict[str, tuple[Candidate, np.ndarray]]:
    """Кандидаты в признаки как массивы час × вестибюль, значение в строке t — только по данным до конца часа t;
    погода и календарь — на час t + h (прогноз и календарь известны заранее)."""
    g, ves, cal = d.grid, d.vestibules, d.calendar
    T, V = g.y.shape
    y, b, r = g.y, g.b, g.r
    out: dict[str, tuple[Candidate, np.ndarray]] = {}

    def add(name, group, kind, desc, arr):
        out[name] = (Candidate(name, group, kind, desc), arr.astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        # динамика
        for k in (1, 2, 3):
            add(f"d{k}_abs", "динамика", "num", f"прирост за {k} ч, входов", y - _shift(y, k))
            add(f"d{k}_pct", "динамика", "num", f"прирост за {k} ч, %", (y + 1) / (_shift(y, k) + 1) - 1)
        add("accel", "динамика", "num", "ускорение y(t) − 2y(t−1) + y(t−2)", y - 2 * _shift(y, 1) + _shift(y, 2))
        for k in (1, 2, 3):
            ex = (y - _shift(y, k)) - (b - _shift(b, k))
            add(f"excess_d{k}", "динамика", "num", f"прирост против обычного за {k} ч, входов", ex)
            add(f"excess_d{k}_rel", "динамика", "num", f"прирост против обычного за {k} ч / b(t)",
                np.where(b >= B_MIN, ex / b, np.nan))
        add("r", "динамика", "num", "отклонение r(t) = y / b", r)
        for k in (1, 2, 3):
            add(f"dr{k}", "динамика", "num", f"изменение r за {k} ч", r - _shift(r, k))
        add("accel_r", "динамика", "num", "ускорение r", r - 2 * _shift(r, 1) + _shift(r, 2))
        ewm2, ewm6 = ewm_ratio(r, g.ts, 2), ewm_ratio(r, g.ts, 6)
        add("ewm_r_2h", "динамика", "num", "сглаженное r, полупериод 2 ч", ewm2)
        add("ewm_r_6h", "динамика", "num", "сглаженное r, полупериод 6 ч", ewm6)
        add("cum_ratio", "динамика", "num", "Σy/Σb с открытия", cumulative_ratio(d))
        add("vol_r_3h", "динамика", "num", "std r за 3 ч", _rolling_std(r, 3, 3))
        add("vol_r_6h", "динамика", "num", "std r за 6 ч", _rolling_std(r, 6, 4))
        add("ratio_24", "динамика", "num", "y(t) / y(t−24)", (y + 1) / (_shift(y, 24) + 1))
        add("ratio_168", "динамика", "num", "y(t) / y(t−168)", (y + 1) / (_shift(y, 168) + 1))

        # кросс-признаки
        rl = line_ratio(g)
        add("r_line", "кросс", "num", "r всей линии", _per_hour(rl, V))
        sr = station_ratio(g, ves)
        st_order = list(sr.columns)
        lo = d.stations.set_index("station_id").line_order
        neigh = np.full((T, V), np.nan)
        for v, sid in enumerate(ves.station_id):
            near = [s for s in st_order if s != sid and abs(lo[s] - lo[sid]) <= 2]
            neigh[:, v] = sr[near].mean(axis=1, skipna=True).to_numpy()
        add("r_neighbors", "кросс", "num", "r соседних станций (±2)", neigh)
        twin = np.full((T, V), np.nan)
        for sid, idx in ves.groupby("station_id").pos:
            if len(idx) == 2:
                i, j = idx.to_numpy()
                twin[:, i], twin[:, j] = r[:, j], r[:, i]
        add("r_twin", "кросс", "num", "r второго вестибюля станции", twin)
        add("share_dev", "кросс", "num", "доля вестибюля в линии к обычной (r / r_line)", r / _per_hour(rl, V))

        # погода — исторический прогноз на час t + h
        fc = d.forecast.set_index("ts_utc").reindex(g.ts)
        prec, snow, temp = (fc[c].to_numpy() for c in ("precipitation", "snowfall", "temperature_2m"))
        at = lambda a: _per_hour(_shift(a, -h), V)
        add("fc_precip", "погода", "bin", "осадки ≥ 0,5 мм/ч на t+h (прогноз)", at((prec >= 0.5).astype(float)))
        add("fc_snow", "погода", "bin", "снег на t+h (прогноз)", at((snow > 0).astype(float)))
        bins = [(-np.inf, -15, "< −15 °C"), (-15, -5, "−15…−5 °C"), (-5, 5, "−5…+5 °C"), (5, 15, "+5…+15 °C"),
                (15, 25, "+15…+25 °C"), (25, np.inf, "> +25 °C")]
        for i, (a, bnd, lab) in enumerate(bins):
            add(f"fc_temp_bin{i}", "погода", "bin", f"температура {lab} на t+h (прогноз)",
                at(((temp >= a) & (temp < bnd)).astype(float)))
        add("fc_temp_drop24", "погода", "num", "T(t+h) − T(t+h−24), прогноз", at(temp - _shift(temp, 24)))
        local_date = g.local.normalize()
        snow_day = pd.Series(snow > 0).groupby(local_date).any()
        prev = snow_day.rolling(14, min_periods=1).max().shift(1).fillna(0).astype(bool)
        first = (snow_day & ~prev).reindex(local_date).to_numpy().astype(float)
        add("fc_first_snow", "погода", "bin", "первый снег после ≥ 14 дней без снега (прогноз, сутки t+h)", at(first))
        last3 = pd.Series(prec).rolling(3, min_periods=3).sum().to_numpy()
        add("fc_precip_last3h", "погода", "num", "осадки за t−2…t (прогноз), мм", _per_hour(last3, V))

        # календарь — на сутки метро часа t + h
        c = cal.set_index("date")
        sd_tau = _shift_index(g.sday, -h)
        cx = c.reindex(sd_tau)
        hol_days = cal.loc[cal.day_type == "праздник", "date"].sort_values().to_numpy()
        to_h, after_h = _days_to_holiday(sd_tau, hol_days)
        is_hol = (cx.day_type == "праздник").to_numpy()
        add("is_holiday", "календарь", "bin", "праздник / нерабочий будний (сутки t+h)", _per_hour(is_hol.astype(float), V))
        add("days_to_holiday", "календарь", "num", "дней до праздника (1…7, 8 — дальше)",
            _per_hour(np.where(is_hol, np.nan, to_h), V))
        add("days_after_holiday", "календарь", "num", "дней после праздника (1…7, 8 — дальше)",
            _per_hour(np.where(is_hol, np.nan, after_h), V))
        add("is_shortened", "календарь", "bin", "сокращённый день (сутки t+h)",
            _per_hour(cx.is_shortened.eq(True).to_numpy(dtype=float), V))
        add("is_school", "календарь", "bin", "учебный период школ (сутки t+h)",
            _per_hour(cx.is_school.eq(True).to_numpy(dtype=float), V))
        add("plan_pairs", "календарь", "num", "плановая парность на t+h", _per_hour(_plan_pairs(d, h), V))

        # свои
        add("r_yday_same_hour", "свои", "num", "r того же часа вчера (t+h−24)", _shift(r, 24 - h))
        add("r_week_same_hour", "свои", "num", "r того же часа неделю назад (t+h−168)", _shift(r, 168 - h))
        add("cum_ratio_yday", "свои", "num", "Σy/Σb вчерашних суток", _yesterday_ratio(d, sd_tau))
        add("ewm_r_line_2h", "свои", "num", "сглаженное r линии, полупериод 2 ч",
            _per_hour(ewm_ratio(rl[:, None], g.ts, 2)[:, 0], V))
        ev_days = set(event_days(d.events))
        night_days = set(event_days(d.events[d.events.source == "по данным потока"]))
        add("event_day", "свои", "bin", "сутки t+h — день события (events_spb)",
            _per_hour(sd_tau.isin(list(ev_days)).astype(float), V))
        add("night_ahead", "свои", "bin", "впереди особая ночь (сутки t+h)",
            _per_hour(sd_tau.isin(list(night_days)).astype(float), V))
        hr_tau = _per_hour(pd.Series(g.hour).shift(-h).to_numpy(), V)
        short = _per_hour(cx.is_shortened.eq(True).to_numpy(dtype=float), V)
        add("shortened_14_16", "свои", "bin", "сокращённый день, 14–16 ч (вечерний пик раньше)",
            short * np.isin(hr_tau, [14, 15, 16]))
        add("shortened_17_19", "свои", "bin", "сокращённый день, 17–19 ч (вечерний пик раньше)",
            short * np.isin(hr_tau, [17, 18, 19]))
        night = _per_hour(sd_tau.isin(list(night_days)).astype(float), V)
        add("night_ahead_late", "свои", "bin", "впереди особая ночь, 23–00 ч", night * np.isin(hr_tau, [23, 0]))
        rng = np.random.default_rng(seed + h)
        add("noise_control", "контроль", "num", "случайный шум — контроль процедуры", rng.normal(size=(T, V)))
    return out


def _shift_index(idx: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    """Значение индекса в позиции t − k (k < 0 — будущее); за краем — NaT."""
    s = pd.Series(idx)
    return pd.DatetimeIndex(s.shift(k))


def _days_to_holiday(days: pd.DatetimeIndex, hol: np.ndarray, cap: int = 8) -> tuple[np.ndarray, np.ndarray]:
    d = days.to_numpy()
    nxt = np.searchsorted(hol, d, side="left")
    prv = nxt - 1
    to_ = np.full(len(d), float(cap))
    after = np.full(len(d), float(cap))
    has_n = nxt < len(hol)
    to_[has_n] = (hol[nxt[has_n]] - d[has_n]) / np.timedelta64(1, "D")
    has_p = prv >= 0
    after[has_p] = (d[has_p] - hol[prv[has_p]]) / np.timedelta64(1, "D")
    nat = pd.isna(days)
    to_, after = np.minimum(to_, cap), np.minimum(after, cap)
    to_[nat] = after[nat] = np.nan
    return to_, after


def _plan_pairs(d: SpbData, h: int) -> np.ndarray:
    """Плановая парность на час t + h: выходные и праздники — график выходных; будни июнь–август — летний;
    остальные будни — с 01.09 (для января–мая — допущение, другого графика нет)."""
    g, plan = d.grid, d.ops["pairs_plan"]
    cal = d.calendar.set_index("date")
    sd = _shift_index(g.sday, -h)
    hr = pd.Series(g.hour).shift(-h).to_numpy()
    dt = cal.day_type.reindex(sd).to_numpy()
    out = np.full(len(sd), np.nan)
    for i in range(len(sd)):
        if pd.isna(sd[i]) or np.isnan(hr[i]) or int(hr[i]) not in plan["weekday_from_2026_09_01"]:
            continue
        hh = int(hr[i])
        if dt[i] != "рабочий":
            out[i] = plan["weekend_from_2026_09_01"][hh]
        elif 6 <= sd[i].month <= 8:
            out[i] = plan["weekday_summer_from_2026_06_01"][hh]
        else:
            out[i] = plan["weekday_from_2026_09_01"][hh]
    return out


def _yesterday_ratio(d: SpbData, sd_tau: pd.DatetimeIndex) -> np.ndarray:
    g = d.grid
    m = ~np.isnan(g.r)
    ys = pd.DataFrame(np.where(m, g.y, 0)).groupby(g.sday.to_numpy()).sum()
    bs = pd.DataFrame(np.where(m, g.b, 0)).groupby(g.sday.to_numpy()).sum()
    daily = ys / bs.where(bs >= B_MIN)
    prev = sd_tau - pd.Timedelta(days=1)
    return daily.reindex(prev).to_numpy()


def _dense(values: np.ndarray, tr: dict) -> np.ndarray:
    """Строки (день, слот) → матрица день × слот, пустые клетки NaN."""
    m = np.full((len(tr["days"]), tr["n_slots"]), np.nan)
    m[tr["day"], tr["slot"]] = values
    return m


def _slot_uniform(M: np.ndarray) -> np.ndarray:
    """Ранги внутри слота (столбца), шкала (0, 1]: сравниваются дни для одного вестибюля и часа."""
    return pd.DataFrame(M).rank(axis=0, pct=True).to_numpy()


def _slot_center(M: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return M - np.nanmedian(M, axis=0)


def _quintile_spread(xu: np.ndarray, yc: np.ndarray) -> float:
    """Размах эффекта в %: медиана цели (центрированной по слоту) в верхней пятой кандидата против нижней
    (пятые — внутри слота)."""
    m = ~np.isnan(xu) & ~np.isnan(yc)
    xu, yc = xu[m], yc[m]
    lo, hi = xu <= 0.2, xu > 0.8
    if not lo.any() or not hi.any():   # мало различных значений
        lo, hi = xu <= np.quantile(xu, 0.2), xu >= np.quantile(xu, 0.8)
    return float(np.exp(np.median(yc[hi]) - np.median(yc[lo])) - 1)


def screen(d: SpbData, horizons=(1, 2), n_perm: int = 1000, n_boot: int = 1000, seed: int = 0,
           dup_thr: float = 0.95) -> pd.DataFrame:
    """Отбор кандидатов: эффект и ДИ, p перестановкой дней внутри месяца × класса дня, BH, стабильность по месяцам,
    дубли. Пользу для прогноза (ΔWAPE) добавляет `forecast_gain`, вердикт — `verdict`."""
    rows = []
    cal = d.calendar.set_index("date")
    for h in horizons:
        tr = target_rows(d, h)
        cands = candidate_matrices(d, h, seed)
        Y = _dense(tr["target"], tr)
        Yc, Yu = _slot_center(Y), _slot_uniform(Y)
        months = pd.DatetimeIndex(tr["days"]).month.to_numpy()
        work = (cal.day_type.reindex(tr["days"]).to_numpy() == "рабочий")
        rng = np.random.default_rng(seed + 100 * h)
        perms = strata_perms(months * 2 + work, n_perm, rng)
        counts = boot_counts(len(tr["days"]), n_boot, rng)
        values = {}
        for name, (cand, arr) in cands.items():
            x = arr[tr["t"], tr["v"]]
            values[name] = x
            X = _dense(x, tr)
            valid = ~np.isnan(X) & ~np.isnan(Y)
            row = {"candidate": name, "group": cand.group, "kind": cand.kind, "description": cand.desc, "h": h,
                   "n_rows": int(valid.sum())}
            if cand.kind == "bin":
                n_pos_days = int(np.unique(np.nonzero(valid & (X == 1))[0]).size)
                row["days_with_flag"] = n_pos_days
                if n_pos_days < 2 or not (valid & (X == 0)).any():
                    rows.append({**row, "note": "нет данных"})
                    continue
                res = median_diff_test(X, Yc, perms, counts, months)
                row["effect_pct"] = float(np.exp(res["effect"]) - 1)
            else:
                Xu = _slot_uniform(X)
                res = rank_corr_test(Xu, Yu, perms, counts, months)
                row["effect_pct"] = _quintile_spread(Xu, Yc)
            bm = {m: v for m, v in res["by_month"].items() if not np.isnan(v)}
            same = sum(np.sign(v) == np.sign(res["effect"]) for v in bm.values())
            rows.append({**row, "effect": res["effect"], "lo": res["lo"], "hi": res["hi"], "p": res["p"],
                         "null_mean": res["null_mean"],
                         "months_same_sign": same, "months_eval": len(bm),
                         "by_month": " ".join(f"{m}:{v:+.2f}" for m, v in bm.items())})
        # дубли — Спирмен между кандидатами на подвыборке строк
        sub = np.random.default_rng(seed).choice(len(tr["t"]), min(40_000, len(tr["t"])), replace=False)
        corr = pd.DataFrame({k: v[sub] for k, v in values.items()}).corr(method="spearman")
        res_h = pd.DataFrame([r for r in rows if r["h"] == h]).set_index("candidate")
        strength = res_h.effect_pct.abs().fillna(0)
        dup_of = {}
        for a in corr.index:
            for b_ in corr.columns:
                if a < b_ and abs(corr.at[a, b_]) > dup_thr:
                    weak, strong = (a, b_) if strength.get(a, 0) < strength.get(b_, 0) else (b_, a)
                    dup_of.setdefault(weak, strong)
        for r_ in rows:
            if r_["h"] == h:
                r_["duplicate_of"] = dup_of.get(r_["candidate"], "")
    out = pd.DataFrame(rows)
    out["p_bh"] = bh(out.p)
    out["stable"] = np.where(out.months_eval >= 4, out.months_same_sign / out.months_eval.clip(lower=1) >= 0.75, np.nan)
    return out


def forecast_gain(d: SpbData, scr: pd.DataFrame, top: int | None = None, test_months=range(4, 9),
                  n_bins: int = 10) -> pd.DataFrame:
    """Для значимых кандидатов без дублей (или `top` сильнейших) каждого горизонта: прогноз b · e^поправка против b,
    поправка — медиана цели в квантильной корзине кандидата по прошлым месяцам (расширяющееся окно) минус общая
    медиана тех же месяцев. Тест — апрель–август: сентябрь — финальный тест, отбор признаков его не видит.
    ΔWAPE, ДИ парным бутстрепом по дням, p Диболда–Мариано."""
    rows = []
    for h, sh in scr.groupby("h"):
        pick = sh[(sh.p_bh < 0.05) & (sh.duplicate_of == "") & sh.effect_pct.notna()]
        pick = pick.reindex(pick.effect_pct.abs().sort_values(ascending=False).index).head(top or len(pick))
        tr = target_rows(d, h)
        cands = candidate_matrices(d, h)
        for name in pick.candidate:
            cand, arr = cands[name]
            x = arr[tr["t"], tr["v"]]
            corr = np.zeros(len(x))
            test = np.isin(tr["month"], list(test_months))
            for m in test_months:
                train, cur = (tr["month"] < m) & ~np.isnan(x), (tr["month"] == m) & ~np.isnan(x)
                if not cur.any() or train.sum() < 1000:
                    continue
                level = np.median(tr["target"][train])   # общий сдвиг обучающих месяцев — не заслуга кандидата
                if cand.kind == "bin":
                    med = {k: np.median(tr["target"][train & (x == k)]) - level for k in (0, 1)
                           if (train & (x == k)).any()}
                    corr[cur] = [med.get(k, 0.0) for k in x[cur]]
                else:
                    edges = np.unique(np.quantile(x[train], np.linspace(0, 1, n_bins + 1)[1:-1]))
                    bt, bc = np.searchsorted(edges, x[train]), np.searchsorted(edges, x[cur])
                    med = pd.Series(tr["target"][train]).groupby(bt).median() - level
                    corr[cur] = med.reindex(bc).fillna(0).to_numpy()
            base = tr["b"][test]
            res = compare_forecasts(tr["y"][test], base, base * np.exp(corr[test]), tr["day"][test])
            rows.append({"candidate": name, "h": h, "dwape": res["dwape"], "dwape_lo": res["lo"], "dwape_hi": res["hi"],
                         "p_dm": res["p_dm"], "wape_b": res["wape_base"], "wape_alt": res["wape_alt"]})
    return pd.DataFrame(rows)


def _pct1(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",") + " %"


def _pp2(x: float) -> str:
    return f"{x * 100:+.2f}".replace(".", ",") + " п. п."


def verdict(scr: pd.DataFrame, gain: pd.DataFrame, noise_p80: float) -> pd.DataFrame:
    """Вердикт «брать / проверить в абляции / отбросить» с причиной.

    Шум слота — разброс одного наблюдения, а эффект признака — систематический сдвиг, поэтому порог шума сам по себе
    кандидата не отбрасывает: если эффект меньше шума, но прогноз с поправкой по кандидату точнее (ΔWAPE > 0,
    ДИ выше нуля), кандидат идёт в абляцию.
    """
    out = scr.merge(gain, on=["candidate", "h"], how="left")
    noise = f"±{noise_p80 * 100:.1f}".replace(".", ",") + " %"
    v, why = [], []
    for r in out.itertuples():
        big = abs(r.effect_pct) >= noise_p80 if not pd.isna(r.effect_pct) else False
        rare = r.kind == "bin" and not pd.isna(r.days_with_flag) and r.days_with_flag < 15
        if getattr(r, "note", "") == "нет данных" or pd.isna(r.p):
            v.append("отбросить"); why.append("нет данных за период")
        elif r.duplicate_of:
            v.append("отбросить"); why.append(f"дубль {r.duplicate_of} (Спирмен > 0,95)")
        elif r.p_bh >= 0.05:
            if rare and big:
                v.append("проверить в абляции")
                why.append(f"эффект {_pct1(r.effect_pct)} ≥ шума, но всего {int(r.days_with_flag)} дн. — тест слабый")
            else:
                v.append("отбросить"); why.append(f"не значим (p_BH = {r.p_bh:.2f})".replace(".", ","))
        elif not big:
            if not pd.isna(r.dwape) and r.dwape > 0 and r.dwape_lo > 0:
                v.append("проверить в абляции")
                why.append(f"эффект {_pct1(r.effect_pct)} меньше шума {noise}, но ΔWAPE {_pp2(r.dwape)} (ДИ выше нуля)")
            else:
                gain_txt = "" if pd.isna(r.dwape) else f", ΔWAPE {_pp2(r.dwape)} не отличим от 0"
                v.append("отбросить"); why.append(f"эффект {_pct1(r.effect_pct)} меньше шума слота {noise}{gain_txt}")
        elif r.stable is not True and r.stable != 1.0:
            v.append("проверить в абляции")
            if r.stable in (False, 0.0):
                why.append("знак по месяцам нестабилен")
            else:
                days = f"всего {int(r.days_with_flag)} дн." if rare else f"месяцев с данными {int(r.months_eval)}"
                why.append(f"значим и ≥ шума, стабильность не проверить: {days}")
        elif pd.isna(r.dwape):
            v.append("проверить в абляции"); why.append("значим и ≥ шума, ΔWAPE не считали (вне топ-15)")
        elif r.dwape_lo > 0:
            v.append("брать"); why.append(f"значим, стабилен, ΔWAPE {_pp2(r.dwape)} (ДИ выше нуля)")
        else:
            v.append("проверить в абляции"); why.append(f"значим и стабилен, но ΔWAPE {_pp2(r.dwape)} не отличим от 0")
    out["verdict"], out["reason"] = v, why
    return out


SCREENING_COLS = ["candidate", "group", "kind", "description", "h", "n_rows", "days_with_flag", "effect", "lo", "hi",
                  "null_mean", "effect_pct", "p", "p_bh", "months_same_sign", "months_eval", "stable", "by_month",
                  "duplicate_of", "dwape", "dwape_lo", "dwape_hi", "p_dm", "verdict", "reason"]


def save_screening(v: pd.DataFrame, path=SCREENING_OUT) -> pd.DataFrame:
    out = v.reindex(columns=SCREENING_COLS).round(5)
    out.to_csv(path, index=False)
    return out


GAIN_COLS = ["dwape", "dwape_lo", "dwape_hi", "p_dm", "verdict", "reason"]


def revise_screening(d: SpbData, path=SCREENING_OUT) -> pd.DataFrame:
    """Пересмотр вердиктов без перезапуска перестановок: эффекты и p — из сохранённой таблицы, ΔWAPE — заново
    для всех значимых кандидатов (апрель–август), вердикт — по текущему правилу `verdict`."""
    scr = pd.read_csv(path).drop(columns=GAIN_COLS)
    scr["duplicate_of"] = scr.duplicate_of.fillna("")
    noise = noise_table(d)
    p80 = noise[(noise.level == "Вся линия") & (noise.band == "все часы")].p80.iloc[0]
    return save_screening(verdict(scr, forecast_gain(d, scr), p80), path)
