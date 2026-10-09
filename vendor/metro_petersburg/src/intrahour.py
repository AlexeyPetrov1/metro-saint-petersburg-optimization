"""Профиль внутри часа и ранний сигнал по 15-минутным данным (этап 6). Графики — notebooks/03_intrahour.ipynb,
выводы — docs/intrahour_findings.md.

Запуск без ноутбука: python -m src.intrahour → data/reference/intrahour_profile.csv и таблицы reports/intrahour/*.csv.
Данные — data/interim/spb_line1_15min.parquet (python -m src.clean_spb_15min). Часы с любым флагом, в том числе
с расхождением источников (is_source_mismatch) и частичным закрытием (is_slot_closed), исключены везде.

**Профиль.** Доля четверти j часа — Σ y_j / Σ y_часа по суткам. Берётся сумма, а не среднее дневных долей,
чтобы малые часы не шумели. Тип дня — рабочий / суббота / воскресенье; праздник считается воскресеньем, как в норме b4.
Только обычные сутки (eda_spb.service_calendar).

**Ранний сигнал** (вариант A, решение этапа 6):
- Истина — часовой файл: R = y / b4. Аномальный час — |R − 1| > p95 группы × периода суток
  (reports/backtest/anomaly_thresholds.csv).
- Частичное отклонение после k четвертей:

      r_k = Σ_{j<k} y_j / (b4 · s_v · Σ_{j<k} p_j),

  где p — профиль по прошлым суткам того же типа, а s_v — Σ15 / Σчас вестибюля за 14 последних общих суток
  (15-минутный источник выше часового на ~1 %).
- Детектор: тревога на первом k, где |r_k − 1| > c_k · p95 и знак совпадает с истиной.
  c_k выбирается по максимуму F1 на феврале + мае + июле; сентябрь — проверка. Часовые результаты по сентябрю уже
  известны (финальный тест этапа 5), поэтому чистой отложенной выборкой он не является.

**Только прошлое.**
- Профиль и s_v для суток D считаются по суткам строго до D.
- b4 и опорные прогнозы минуты 0 — замороженные baseline.norm и backtest.make_rows.
- Прогноз t+1 модели — из кэша data/interim/model_preds (май и июль; сентябрь не кэшировался).
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from src import backtest as BT
from src import config, reference
from src import eda_spb as E

OUT = config.ROOT / "reports" / "intrahour"
PROFILE_CSV = config.REFERENCE / "intrahour_profile.csv"
THRESHOLDS_CSV = BT.OUT / "anomaly_thresholds.csv"

Q = 4
QCOLS = [f"y{j}" for j in range(Q)]
PCOLS = [f"p{j}" for j in range(Q)]
MINUTES = [15 * k for k in range(1, Q + 1)]
# Вокзалы: Финляндский (пл. Ленина), Московский (пл. Восстания), Балтийский, ж/д платформа Девяткино
HUB_STATIONS = ["devyatkino", "ploshchad_lenina", "vosstaniya", "baltiyskaya"]
DAY3 = {"рабочий": "рабочий", "суббота": "суббота", "воскресенье": "воскресенье", "праздник": "воскресенье"}
DAY2 = {"рабочий": "рабочий", "суббота": "выходной", "воскресенье": "выходной"}
DAY_TYPES3 = ["рабочий", "суббота", "воскресенье"]
PROFILE_HOURS = [5, *E.WORK_HOURS]             # 05–00; в 05 ч метро работает с ~05:30
SELECT_MONTHS, CHECK_MONTH = (2, 5, 7), 9
MONTH_RU = {2: "февраль", 5: "май", 7: "июль", 9: "сентябрь"}
EVAL_START = E.ANALYSIS_START                  # 09.02: норма b4 определена, первая неделя февраля — разгон профиля
MIN_DAYS, MIN_ENTRIES = 3, 200                 # ячейка профиля: ≥ 3 суток и ≥ 200 входов за час в сумме
SV_DAYS, SV_MIN_DAYS = 14, 5                   # поправка источника: 14 последних общих суток, минимум 5
C_GRID = np.round(np.arange(0.30, 3.001, 0.05), 2)
PRECISION_INFO = 0.7                            # «минимальный c_k при precision ≥ 0,7» — только для информации
# Уровни профиля «только по прошлому»: свой вестибюль → станция → группа → вестибюль по «рабочий / выходной» → группа
PAST_LEVELS = [("vestibule_id", "day3"), ("station_id", "day3"), ("group", "day3"),
               ("vestibule_id", "day2"), ("group", "day2")]
SIM_NOTE_15MIN_ONLY = ("нет в часовых данных: профиль только по 4 месяцам 15-минутных данных "
                       "(февраль, май, июль, сентябрь)")


# --- Загрузка ------------------------------------------------------------------------
def load_hours(path=config.SPB_15MIN, recon_path=config.SPB_15MIN_RECON) -> pd.DataFrame:
    """15-минутная сетка → строка = вестибюль × час: y0…y3 (четверти), y15 = Σ, y_hourly (часовой файл), флаги."""
    q = pd.read_parquet(path)
    if len(q) % Q or not (q.quarter.to_numpy().reshape(-1, Q) == np.arange(Q)).all():
        raise ValueError("15-минутная сетка не разбивается на часы по 4 слота подряд — пересоберите clean_spb_15min")
    first = q.iloc[::Q].reset_index(drop=True)
    h = first[["vestibule_id", "station_id", "ts_utc", "sday", "hour", "in_hourly", "is_closed_hour",
               "is_vestibule_closed", "is_incident", "is_source_mismatch", "is_slot_closed", "day_type",
               "is_regular"]].copy()
    h[QCOLS] = q.entries.to_numpy().reshape(-1, Q)
    h["y15"] = h[QCOLS].sum(1)
    h["local"] = h.ts_utc.dt.tz_convert(config.SPB_TZ).dt.tz_localize(None)
    recon = pd.read_parquet(recon_path, columns=["vestibule_id", "ts_utc", "entries_hourly"])
    h = h.merge(recon.rename(columns={"entries_hourly": "y_hourly"}), on=["vestibule_id", "ts_utc"], how="left",
                validate="one_to_one")
    h["month"] = h.sday.dt.month
    h["day3"] = h.day_type.map(DAY3)
    h["day2"] = h.day3.map(DAY2)
    h["group"] = h.station_id.map(E.STATION_GROUP)
    h["hub"] = h.station_id.isin(HUB_STATIONS)
    h["ok"] = ~(h.is_closed_hour | h.is_vestibule_closed | h.is_incident | h.is_source_mismatch | h.is_slot_closed)
    return h


def nonuniformity(p: np.ndarray) -> np.ndarray:
    """½ Σ |p_j − 1/4|: какую долю часа пришлось бы переложить между четвертями до ровного профиля."""
    return 0.5 * np.abs(np.asarray(p) - 1 / Q).sum(-1)


def _shares(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Доли четвертей по сумме входов: keys → p0…p3, n_days, total."""
    g = df.groupby(keys, observed=True)
    out = g[QCOLS].sum()
    tot = out.sum(1)
    res = out.div(tot.where(tot > 0), axis=0).set_axis(PCOLS, axis=1)
    res["n_days"] = g.sday.nunique()
    res["total"] = tot
    return res.reset_index()


def profile_rows(h: pd.DataFrame) -> pd.DataFrame:
    """Строки для профиля: обычные сутки, без флагов, часы 05–00, тип дня из трёх."""
    return h[h.is_regular & h.ok & h.hour.isin(PROFILE_HOURS) & h.day3.isin(DAY_TYPES3)]


# --- 2. Профиль: неравномерность, стабильность, вокзалы ---------------------------------
def profile_cells(h: pd.DataFrame) -> pd.DataFrame:
    """Ячейки «вестибюль × тип дня × час» по всем 4 месяцам: доли, неравномерность U, максимальная четверть,
    сверхдисперсия φ (средний χ² Пирсона дня против профиля / 3: 1 — только счётный шум, > 1 — доли гуляют
    день ото дня сильнее, чем при случайном счёте)."""
    rows = profile_rows(h)
    cells = _shares(rows, ["vestibule_id", "station_id", "group", "hub", "day3", "hour"])
    cells["U"] = nonuniformity(cells[PCOLS].to_numpy())
    cells["max_share"] = cells[PCOLS].max(1)
    cells["max_quarter"] = cells[PCOLS].to_numpy().argmax(1)
    m = rows.merge(cells[["vestibule_id", "day3", "hour", *PCOLS]], on=["vestibule_id", "day3", "hour"])
    n = m.y15.to_numpy(float)
    P = m[PCOLS].to_numpy()
    ok = (n >= 50) & (P > 0).all(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = (((m[QCOLS].to_numpy() - n[:, None] * P) ** 2) / (n[:, None] * P)).sum(1)
    m["chi"] = np.where(ok, chi, np.nan)
    phi = m.groupby(["vestibule_id", "day3", "hour"]).chi.mean() / (Q - 1)
    return cells.merge(phi.rename("phi").reset_index(), on=["vestibule_id", "day3", "hour"], how="left")


def weighted(df: pd.DataFrame, col: str, by: list[str], w: str = "total") -> pd.Series:
    d = df[df[col].notna()]
    return (d[col] * d[w]).groupby([d[b] for b in by]).sum() / d[w].groupby([d[b] for b in by]).sum()


def hub_comparison(cells: pd.DataFrame) -> pd.DataFrame:
    """Вокзалы против остальных: U, максимальная четверть и φ по типу дня и часу, взвешено по потоку."""
    c = cells.assign(kind=np.where(cells.hub, "вокзалы", "остальные"))
    out = pd.concat({"U": weighted(c, "U", ["kind", "day3", "hour"]),
                     "max_share": weighted(c, "max_share", ["kind", "day3", "hour"]),
                     "phi": weighted(c, "phi", ["kind", "day3", "hour"])}, axis=1).reset_index()
    return out


def month_stability(h: pd.DataFrame, n_perm: int = 200, seed: int = 0) -> pd.DataFrame:
    """Стабильность профиля между месяцами: d_obs — среднее ½Σ|p_m − p_m'| по парам месяцев, нулевое распределение —
    перестановки суток между месяцами внутри ячейки «вестибюль × тип дня × час» (то же число суток в каждом месяце)."""
    rows = profile_rows(h)
    rng = np.random.default_rng(seed)
    months = sorted(rows.month.unique())
    out = []
    for (v, d3, hr), g in rows.groupby(["vestibule_id", "day3", "hour"], sort=False):
        mi = np.searchsorted(months, g.month.to_numpy())
        present = np.unique(mi)
        if len(present) < 2 or g.y15.sum() < MIN_ENTRIES * len(present):
            continue
        Y = g[QCOLS].to_numpy(float)
        perm = np.vstack([mi, np.array([rng.permutation(mi) for _ in range(n_perm)])])   # [1 + n_perm, n]
        onehot = (perm[:, :, None] == present[None, None, :]).astype(float)            # [P, n, M]
        S = np.einsum("pnm,nj->pmj", onehot, Y)                                           # [P, M, Q]
        tot = S.sum(2, keepdims=True)
        if (tot[0] == 0).any():
            continue
        prof = S / np.where(tot > 0, tot, np.nan)
        iu = np.triu_indices(len(present), 1)
        dist = 0.5 * np.abs(prof[:, iu[0]] - prof[:, iu[1]]).sum(-1).mean(-1)              # [P]
        d0, null = dist[0], dist[1:]
        out.append({"vestibule_id": v, "day3": d3, "hour": hr, "n_days": len(g), "n_months": len(present),
                    "total": g.y15.sum(), "d_obs": d0, "d_null": np.nanmean(null), "d_null_p95": np.nanquantile(null, 0.95),
                    "p": (1 + (null >= d0).sum()) / (1 + n_perm)})
    res = pd.DataFrame(out)
    res["p_bh"] = E.bh(res.p.to_numpy())
    return res


def monthly_profiles(h: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return _shares(profile_rows(h), [*keys, "month"])


# --- Таблица для симулятора -------------------------------------------------------------
def simulator_profile(h: pd.DataFrame, ves15: pd.DataFrame | None = None) -> pd.DataFrame:
    """Профиль для симулятора по всем 4 месяцам: вестибюль и станция целиком (vestibule_id = "*") × тип дня × час
    × четверть → доля. Ячейка с < MIN_DAYS суток или < MIN_ENTRIES входов → профиль станции, затем группы, затем линии
    (колонка source). Часы вне режима вестибюля не выводятся."""
    ves15 = reference.load_vestibules_15min() if ves15 is None else ves15
    rows = profile_rows(h)
    keys3 = ["day3", "hour"]
    lv = {"вестибюль": _shares(rows, ["vestibule_id", "station_id", *keys3]),
          "станция": _shares(rows, ["station_id", *keys3]),
          "группа": _shares(rows, ["group", *keys3]),
          "линия": _shares(rows, keys3)}
    good = lambda df: (df.n_days >= MIN_DAYS) & (df.total >= MIN_ENTRIES)
    st = reference.load_stations()
    order = st.set_index("station_id").line_order
    reg = ves15.set_index("vestibule_id")
    pos = reference.service_pos
    grid = [(v.vestibule_id, v.station_id, d, hr) for v in ves15.itertuples() for d in DAY_TYPES3 for hr in PROFILE_HOURS
            if pos(v.first_hour) <= pos(hr) <= pos(v.last_hour)]
    target = pd.DataFrame(grid, columns=["vestibule_id", "station_id", "day3", "hour"])
    stations = target[["station_id", "day3", "hour"]].drop_duplicates().assign(vestibule_id="*")
    target = pd.concat([target, stations], ignore_index=True)
    target["group"] = target.station_id.map(E.STATION_GROUP)

    def pick(level: str, on: list[str], mask: pd.Series) -> None:
        src = lv[level][good(lv[level])].set_index(on)[[*PCOLS, "n_days"]]
        need = mask & target.source.isna()
        vals = src.reindex(pd.MultiIndex.from_frame(target.loc[need, on]))
        hit = vals.p0.notna().to_numpy()
        rows = target.index[need][hit]
        target.loc[rows, PCOLS] = vals[PCOLS].to_numpy()[hit]
        target.loc[rows, "n_days"] = vals.n_days.to_numpy()[hit]
        target.loc[rows, "source"] = level

    target[PCOLS] = np.nan
    target["n_days"] = np.nan
    target["source"] = None
    is_v = target.vestibule_id != "*"
    pick("вестибюль", ["vestibule_id", "day3", "hour"], is_v)
    for level, on in (("станция", ["station_id", "day3", "hour"]), ("группа", ["group", "day3", "hour"]),
                      ("линия", ["day3", "hour"])):
        pick(level, on, pd.Series(True, index=target.index))
    if target.source.isna().any():
        raise ValueError(f"Нет профиля для {target[target.source.isna()].head().to_dict('records')}")

    target[PCOLS] = target[PCOLS].div(target[PCOLS].sum(1), axis=0)
    only15 = set(reg.index[~reg.in_hourly])
    st15 = set(reg.station_id[~reg.in_hourly])
    target["note"] = np.where(target.vestibule_id.isin(only15) | ((target.vestibule_id == "*") & target.station_id.isin(st15)),
                              SIM_NOTE_15MIN_ONLY, "")
    long = target.melt(id_vars=["station_id", "vestibule_id", "day3", "hour", "n_days", "source", "note"],
                       value_vars=PCOLS, var_name="quarter", value_name="share")
    long["quarter"] = long.quarter.str[1:].astype(int)
    long["n_days"] = long.n_days.astype(int)
    long = long.rename(columns={"day3": "day_type"})
    long["_o"] = long.station_id.map(order)
    long["_v"] = long.vestibule_id.eq("*")
    long["_d"] = long.day_type.map({d: i for i, d in enumerate(DAY_TYPES3)})
    long["_h"] = long.hour.map(reference.service_pos)
    long = long.sort_values(["_o", "_v", "vestibule_id", "_d", "_h", "quarter"]).reset_index(drop=True)
    long["share"] = long.share.round(6)
    return long[["station_id", "vestibule_id", "day_type", "hour", "quarter", "share", "n_days", "source", "note"]]


# --- 3. Ранний сигнал: профиль и поправка источника только по прошлому --------------------
def past_profile(h: pd.DataFrame) -> pd.DataFrame:
    """Для каждой строки (вестибюль × час суток D) — доли p0…p3 по обычным суткам строго до D того же типа дня;
    уровни — PAST_LEVELS по порядку, ячейка годна при ≥ MIN_DAYS суток и ≥ MIN_ENTRIES входов."""
    contrib = h.is_regular & h.ok
    left = h[["sday", "hour", "vestibule_id", "station_id", "group", "day3", "day2"]].copy()
    left["_i"] = np.arange(len(h))
    left = left.sort_values("sday", kind="stable")
    res = np.full((len(h), Q), np.nan)
    level = np.full(len(h), "", dtype=object)
    for key, day in PAST_LEVELS:
        a = (h[contrib].groupby([key, day, "hour", "sday"], observed=True)[QCOLS].sum().reset_index()
             .sort_values([key, day, "hour", "sday"]))
        g = a.groupby([key, day, "hour"], observed=True)
        a[QCOLS] = g[QCOLS].cumsum()              # включая сутки строки; merge_asof ниже берёт строго раньше D
        a["n_days"] = g.cumcount() + 1
        m = pd.merge_asof(left, a.sort_values("sday", kind="stable"), on="sday", by=[key, day, "hour"],
                          allow_exact_matches=False, direction="backward")
        tot = m[QCOLS].sum(1)
        ok = (m.n_days >= MIN_DAYS) & (tot >= MIN_ENTRIES)
        idx = m._i.to_numpy()
        take = ok.to_numpy() & np.isnan(res[idx, 0])
        res[idx[take]] = m.loc[take, QCOLS].div(tot[take], axis=0).to_numpy()
        level[idx[take]] = f"{key} × {day}"
    out = pd.DataFrame(res, columns=PCOLS, index=h.index)
    out["profile_level"] = level
    return out


def source_scale(h: pd.DataFrame) -> pd.Series:
    """s_v для суток D: Σ15 / Σчас вестибюля за SV_DAYS последних общих суток строго до D (часы 06–00 без флагов);
    меньше SV_MIN_DAYS суток — то же по линии; нет истории — NaN."""
    m = h.ok & h.in_hourly & h.hour.isin(E.WORK_HOURS) & h.y_hourly.notna()
    d = h[m].groupby(["vestibule_id", "sday"])[["y15", "y_hourly"]].sum().reset_index().sort_values(["vestibule_id", "sday"])
    roll = d.groupby("vestibule_id")[["y15", "y_hourly"]].rolling(SV_DAYS, min_periods=SV_MIN_DAYS).sum()
    roll = roll.reset_index(level=0, drop=True)
    d["s_v"] = roll.y15 / roll.y_hourly
    line = d.groupby("sday")[["y15", "y_hourly"]].sum().sort_index()
    lr = line.rolling(SV_DAYS, min_periods=SV_MIN_DAYS).sum()
    line_s = (lr.y15 / lr.y_hourly).rename("s_line").reset_index()
    left = h[["vestibule_id", "sday"]].assign(_i=np.arange(len(h))).sort_values("sday", kind="stable")
    mv = pd.merge_asof(left, d[["vestibule_id", "sday", "s_v"]].dropna().sort_values("sday"), on="sday",
                       by="vestibule_id", allow_exact_matches=False)
    ml = pd.merge_asof(left, line_s.dropna(), on="sday", allow_exact_matches=False)
    s = np.full(len(h), np.nan)
    s[mv._i.to_numpy()] = mv.s_v.to_numpy()
    sl = np.full(len(h), np.nan)
    sl[ml._i.to_numpy()] = ml.s_line.to_numpy()
    return pd.Series(np.where(np.isnan(s), sl, s), index=h.index, name="s_v")


# --- 3. Ранний сигнал: таблица часов ------------------------------------------------------
def model_t1() -> pd.DataFrame:
    """q50 замороженной модели на t+1 из кэша бэктеста (фолды май–август); пусто, если кэша нет."""
    from src import model as M

    step = M.load_config()["step_model"]
    found = sorted(p for p in M.CACHE.glob(f"{step}_*.parquet"))
    if len(found) != 1:
        return pd.DataFrame(columns=["vestibule_id", "tau", "f_model"])
    p = pd.read_parquet(found[0], columns=["vestibule_id", "h", "tau", "q50"])
    return p[p.h == 1].drop(columns="h").rename(columns={"q50": "f_model"})


def signal_table(h: pd.DataFrame | None = None, thresholds: pd.DataFrame | None = None,
                 with_model: bool = True) -> pd.DataFrame:
    """Строка = вестибюль × час с часовой истиной: r_1…r_4, R, порог p95, опорные прогнозы минуты 0 и признак `main`
    (обычные сутки с 09.02, 06–00, без флагов и расхождения источников, есть профиль и s_v)."""
    h = load_hours() if h is None else h
    thresholds = pd.read_csv(THRESHOLDS_CSV) if thresholds is None else thresholds
    h = pd.concat([h, past_profile(h), source_scale(h)], axis=1)

    panel = BT.build_panel(E.load(), final=True)       # норма и r(t) — только по прошлому; сентябрь виден как прошлое
    rows = BT.make_rows(panel, horizons=(1,))
    rows = rows[["vestibule_id", "tau", "y", "b4", "b4_level", "r_b4_t", "r_level_t", "band", "is_special"]]
    s = h.merge(rows.rename(columns={"tau": "ts_utc"}), on=["vestibule_id", "ts_utc"], how="inner", validate="one_to_one")
    if with_model:
        s = s.merge(model_t1().rename(columns={"tau": "ts_utc"}), on=["vestibule_id", "ts_utc"], how="left")
    else:
        s["f_model"] = np.nan

    s["R"] = s.y / s.b4
    s["thr"] = BT.anomaly_threshold(SimpleNamespace(thresholds=thresholds), s.group, s.hour, "p95")
    s["anom"] = (s.R - 1).abs() > s.thr
    s["up"] = s.R > 1
    cum_y = s[QCOLS].cumsum(1).to_numpy(float)
    cum_p = s[PCOLS].cumsum(1).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, Q + 1):
            s[f"r{k}"] = cum_y[:, k - 1] / (s.b4.to_numpy() * s.s_v.to_numpy() * cum_p[:, k - 1])
            s[f"rB{k}"] = cum_y[:, k - 1] / (s.b4.to_numpy() * cum_p[:, k - 1])     # вариант B: без поправки
    s["r4"] = s.y15 / (s.b4 * s.s_v)                                                      # Σ p = 1 точно
    s["rB4"] = s.y15 / s.b4
    s["f_b4"] = s.b4
    s["f_r"] = s.b4 * s.r_b4_t.clip(*BT.R_CLIP).fillna(1.0)
    s["f_level_r"] = s.b4_level * s.r_level_t.clip(*BT.R_CLIP).fillna(1.0)
    s["main"] = (s.is_regular & s.ok & s.in_hourly & s.hour.isin(E.WORK_HOURS) & (s.b4 >= E.B_MIN)
                 & (s.sday >= EVAL_START) & s.p0.notna() & s.s_v.notna())
    s["special"] = (~s.is_regular | s.is_special) & s.ok & s.in_hourly & s.hour.isin(E.WORK_HOURS) \
        & (s.b4 >= E.B_MIN) & (s.sday >= EVAL_START) & s.p0.notna() & s.s_v.notna()
    s["period"] = np.where(s.month.isin(SELECT_MONTHS), "фев + май + июль",
                           np.where(s.month == CHECK_MONTH, "сентябрь", "—"))
    return s


# --- 3. Предсказание всего часа ------------------------------------------------------------
def _boot_ratio(num: pd.Series, den: pd.Series, day: pd.Series, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % ДИ отношения сумм блочным бутстрепом по дням."""
    a = pd.DataFrame({"n": num, "d": den, "day": day}).groupby("day")[["n", "d"]].sum()
    counts = E.boot_counts(len(a), n_boot, np.random.default_rng(seed))
    b = (counts @ a.n.to_numpy()) / (counts @ a.d.to_numpy())
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def forecast_table(s: pd.DataFrame, mask_col: str = "main", n_boot: int = 1000) -> pd.DataFrame:
    """Насколько частичное отклонение предсказывает весь час: корреляции log r_k с log R, MAE |r_k − R|,
    WAPE прогноза ŷ = b4 · r_k. Опорные точки минуты 0 — норма, b × r(t), b × уровень × r(t), модель t+1."""
    rows = []
    sources = [("минута 0: норма b4", 0, "f_b4"), ("минута 0: b × r(t)", 0, "f_r"),
               ("минута 0: b × уровень × r(t)", 0, "f_level_r"), ("минута 0: модель t+1 (q50)", 0, "f_model"),
               *[(f"{m} мин: b4 · r_{k}", m, f"r{k}") for k, m in zip(range(1, Q + 1), MINUTES)]]
    for period, d in s[s[mask_col]].groupby("period"):
        for label, minute, col in sources:
            if col.startswith("r"):
                f = d.b4 * d[col]
            else:
                f = d[col]
            ok = f.notna() & d.y.notna()
            if ok.sum() == 0:
                continue
            x = d[ok]
            f = f[ok]
            err = (f - x.y).abs()
            lo, hi = _boot_ratio(err, x.y, x.sday, n_boot)
            r_hat = f / x.b4
            lr, lR = np.log(r_hat.clip(lower=1e-3)), np.log(x.R.clip(lower=1e-3))
            rows.append({"period": period, "label": label, "minute": minute, "n": int(ok.sum()),
                         "n_days": x.sday.nunique(), "wape": err.sum() / x.y.sum(), "wape_lo": lo, "wape_hi": hi,
                         "mae_r": (r_hat - x.R).abs().mean(),
                         "pearson_log": np.corrcoef(lr, lR)[0, 1] if r_hat.std() > 0 else np.nan,
                         "spearman": stats.spearmanr(r_hat, x.R).statistic if r_hat.std() > 0 else np.nan})
    return pd.DataFrame(rows)


def forecast_delta(s: pd.DataFrame, a: str, b: str, mask_col: str = "main", n_boot: int = 1000) -> pd.DataFrame:
    """ΔWAPE = WAPE(a) − WAPE(b) с ДИ блочным бутстрепом по дням (a, b — колонки прогноза или r_k)."""
    out = []
    for period, d in s[s[mask_col]].groupby("period"):
        fa = d.b4 * d[a] if a.startswith("r") else d[a]
        fb = d.b4 * d[b] if b.startswith("r") else d[b]
        ok = fa.notna() & fb.notna()
        if not ok.any():
            continue
        x = d[ok]
        ea, eb = (fa[ok] - x.y).abs(), (fb[ok] - x.y).abs()
        agg = pd.DataFrame({"ea": ea, "eb": eb, "y": x.y, "day": x.sday}).groupby("day")[["ea", "eb", "y"]].sum()
        counts = E.boot_counts(len(agg), n_boot, np.random.default_rng(1))
        boot = (counts @ (agg.ea - agg.eb).to_numpy()) / (counts @ agg.y.to_numpy())
        out.append({"period": period, "a": a, "b": b, "n": int(ok.sum()), "wape_a": ea.sum() / x.y.sum(),
                    "wape_b": eb.sum() / x.y.sum(), "delta_wape": (ea.sum() - eb.sum()) / x.y.sum(),
                    "lo": np.quantile(boot, 0.025), "hi": np.quantile(boot, 0.975)})
    return pd.DataFrame(out)


# --- 3. Детектор ---------------------------------------------------------------------------
def alarm(s: pd.DataFrame, k: int, c: float, prefix: str = "r") -> np.ndarray:
    return ((s[f"{prefix}{k}"] - 1).abs() > c * s.thr).to_numpy()


def _truth(s: pd.DataFrame, prefix: str = "r") -> tuple[np.ndarray, np.ndarray]:
    """Истина: вариант A — часовой файл (R); вариант B — 15-минутный час без поправки (rB4)."""
    if prefix == "r":
        return s.anom.to_numpy(), s.up.to_numpy()
    rb = s.rB4
    return ((rb - 1).abs() > s.thr).to_numpy(), (rb > 1).to_numpy()


def snapshot_scores(s: pd.DataFrame, k: int, c: float, prefix: str = "r") -> dict:
    """Тревога по k-й четверти (без учёта более ранних): TP — тревога на аномальном часе с тем же знаком."""
    anom, up = _truth(s, prefix)
    a = alarm(s, k, c, prefix)
    same = (s[f"{prefix}{k}"].to_numpy() > 1) == up
    tp = (a & anom & same).sum()
    n_al, n_an = a.sum(), anom.sum()
    prec = tp / n_al if n_al else np.nan
    rec = tp / n_an if n_an else np.nan
    f1 = 2 * prec * rec / (prec + rec) if n_al and n_an and (prec + rec) > 0 else 0.0
    return {"k": k, "minute": 15 * k, "c": c, "alarms": int(n_al), "anomalies": int(n_an), "tp": int(tp),
            "precision": prec, "recall": rec, "f1": f1}


def choose_c(s: pd.DataFrame, prefix: str = "r", grid: np.ndarray = C_GRID) -> pd.DataFrame:
    """c_k для k = 1…3: максимум F1 (зафиксированное правило) и, для информации, минимальный c с precision ≥ 0,7."""
    rows = []
    for k in range(1, Q):
        sc = pd.DataFrame([snapshot_scores(s, k, c, prefix) for c in grid])
        best = sc.loc[sc.f1.idxmax()]
        info = sc[sc.precision >= PRECISION_INFO]
        alt = info.iloc[0] if len(info) else None
        rows.append({"k": k, "minute": 15 * k, "c_f1": best.c, "f1": best.f1, "precision": best.precision,
                     "recall": best.recall,
                     "c_prec70": alt.c if alt is not None else np.nan,
                     "precision_prec70": alt.precision if alt is not None else np.nan,
                     "recall_prec70": alt.recall if alt is not None else np.nan})
    return pd.DataFrame(rows)


def sequential(s: pd.DataFrame, c: dict[int, float], prefix: str = "r") -> pd.DataFrame:
    """Детектор по ходу часа: тревога на первом k (1…3), где |r_k − 1| > c_k · p95; на 60-й минуте — полный час
    по 15-минутным данным (c = 1). Колонки: first_k (первая тревога, 0 — нет), first_up, detected_min (минута
    обнаружения аномалии с верным знаком; NaN — не найдена)."""
    anom, up = _truth(s, prefix)
    A = np.column_stack([alarm(s, k, c.get(k, 1.0), prefix) for k in range(1, Q + 1)])
    UP = np.column_stack([(s[f"{prefix}{k}"] > 1).to_numpy() for k in range(1, Q + 1)])
    any_a = A.any(1)
    first = np.where(any_a, A.argmax(1) + 1, 0)
    first_up = np.where(any_a, UP[np.arange(len(s)), np.maximum(first - 1, 0)], False)
    correct = A & (UP == up[:, None]) & anom[:, None]
    det = np.where(correct.any(1), 15 * (correct.argmax(1) + 1), np.nan)
    return pd.DataFrame({"anom": anom, "up": up, "first_k": first, "first_up": first_up, "detected_min": det},
                        index=s.index)


def detection_summary(s: pd.DataFrame, c: dict[int, float], prefix: str = "r", n_boot: int = 1000) -> pd.DataFrame:
    """По минутам 15/30/45/60: доля найденных аномалий к этой минуте (recall) и точность тревог, поднятых к этой минуте
    (первая тревога часа с верным знаком на аномальном часе); ложные тревоги на вестибюль в сутки; ДИ — бутстреп по дням."""
    seq = sequential(s, c, prefix)
    day = s.sday.to_numpy()
    days, di = np.unique(day, return_inverse=True)
    counts = E.boot_counts(len(days), n_boot, np.random.default_rng(2))
    n_vd = s.groupby("sday").vestibule_id.nunique().reindex(days).to_numpy()
    out = []
    for m in MINUTES:
        k = m // 15
        raised = (seq.first_k > 0) & (seq.first_k <= k)
        good = raised & seq.anom & (seq.first_up == seq.up)
        found = seq.detected_min <= m
        per = lambda x: np.bincount(di, weights=np.asarray(x, float), minlength=len(days))
        r_n, r_d = per(found), per(seq.anom)
        p_n, p_d = per(good), per(raised)
        fa = per(raised & ~seq.anom)
        br, bp = (counts @ r_n) / (counts @ r_d), (counts @ p_n) / (counts @ p_d)
        out.append({"minute": m, "anomalies": int(seq.anom.sum()), "recall": r_n.sum() / r_d.sum(),
                    "recall_lo": np.quantile(br, 0.025), "recall_hi": np.quantile(br, 0.975),
                    "alarms": int(raised.sum()), "precision": p_n.sum() / max(p_d.sum(), 1),
                    "precision_lo": np.nanquantile(bp, 0.025), "precision_hi": np.nanquantile(bp, 0.975),
                    "false_per_vest_day": fa.sum() / n_vd.sum()})
    det = seq.detected_min[seq.anom]
    res = pd.DataFrame(out)
    res["median_detect_min"] = det.fillna(np.inf).median()     # не найденные к 60-й минуте — «позже 60»
    res["missed_by_60"] = det.isna().mean()
    return res


def minute0_rule(s: pd.DataFrame) -> dict:
    """«Прошлый час уже аномален»: |r(t − 1) − 1| > p95 часа t — лучшее, что дают часовые данные к началу часа."""
    rt = s.r_b4_t
    a = ((rt - 1).abs() > s.thr) & rt.notna()
    tp = (a & s.anom & ((rt > 1) == s.up)).sum()
    return {"minute": 0, "alarms": int(a.sum()), "anomalies": int(s.anom.sum()),
            "precision": tp / max(a.sum(), 1), "recall": tp / max(s.anom.sum(), 1)}


def detection_slices(s: pd.DataFrame, c: dict[int, float]) -> pd.DataFrame:
    """Разрезы при выбранных c_k: направление, период суток, группа, вокзалы."""
    seq = sequential(s, c)
    d = s.assign(**{c_: seq[c_] for c_ in seq.columns if c_ not in ("anom", "up")})
    d["direction"] = np.where(d.up, "всплеск", "провал")
    d["band_eda"] = d.hour.map({h: b for b, hs in E.HOUR_BANDS.items() for h in hs}).fillna("06 ч")
    d["kind"] = np.where(d.hub, "вокзалы", "остальные")
    out = []
    for name, col in (("направление", "direction"), ("период суток", "band_eda"), ("группа", "group"),
                      ("вокзалы", "kind")):
        for key, g in d.groupby(col):
            an = g[g.anom]
            raised = (g.first_k > 0) & (g.first_k <= 3)
            good = raised & g.anom & (g.first_up == g.up)
            row = {"slice": name, "key": key, "rows": len(g), "anomalies": len(an),
                   "anomaly_rate": len(an) / len(g),
                   "precision_early": good.sum() / max(raised.sum(), 1),
                   "median_detect_min": an.detected_min.fillna(np.inf).median()}
            for m in MINUTES:
                row[f"recall_{m}"] = (an.detected_min <= m).mean() if len(an) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def lomo(s: pd.DataFrame) -> pd.DataFrame:
    """Разброс между месяцами отбора: c_k по двум месяцам, проверка на третьем."""
    out = []
    for test in SELECT_MONTHS:
        fit = s[s.month.isin([m for m in SELECT_MONTHS if m != test])]
        ev = s[s.month == test]
        cs = choose_c(fit)
        for r in cs.itertuples():
            sc = snapshot_scores(ev, r.k, r.c_f1)
            out.append({"test_month": MONTH_RU[test], "k": r.k, "minute": r.minute, "c": r.c_f1,
                        "f1": sc["f1"], "precision": sc["precision"], "recall": sc["recall"]})
    return pd.DataFrame(out)


# --- Всё вместе --------------------------------------------------------------------------------
@dataclass
class Results:
    hours: pd.DataFrame
    cells: pd.DataFrame
    hubs: pd.DataFrame
    stability: pd.DataFrame
    sim: pd.DataFrame
    signal: pd.DataFrame
    forecast: pd.DataFrame
    deltas: pd.DataFrame
    c_select: pd.DataFrame
    c: dict
    snapshot: pd.DataFrame
    detection: pd.DataFrame
    minute0: pd.DataFrame
    slices: pd.DataFrame
    lomo: pd.DataFrame
    special: pd.DataFrame
    variant_b: pd.DataFrame


def split_periods(sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Основные строки: отбор параметров (февраль, май, июль) и проверка (сентябрь)."""
    main_ = sig[sig.main]
    return main_[main_.month.isin(SELECT_MONTHS)], main_[main_.month == CHECK_MONTH]


def run(n_perm: int = 200) -> Results:
    h = load_hours()
    cells = profile_cells(h)
    sig = signal_table(h)
    sel, sep = split_periods(sig)

    c_sel = choose_c(sel)
    c = dict(zip(c_sel.k, c_sel.c_f1))
    snap = pd.DataFrame([{"period": p, **snapshot_scores(d, k, c[k])}
                         for p, d in (("фев + май + июль", sel), ("сентябрь", sep)) for k in range(1, Q)])
    det = pd.concat([detection_summary(d, c).assign(period=p) for p, d in (("фев + май + июль", sel), ("сентябрь", sep))])
    m0 = pd.DataFrame([{"period": p, **minute0_rule(d)} for p, d in (("фев + май + июль", sel), ("сентябрь", sep))])
    slices = pd.concat([detection_slices(d, c).assign(period=p) for p, d in (("фев + май + июль", sel), ("сентябрь", sep))])

    spec = sig[sig.special]
    special = []
    if len(spec):
        for day, d in spec.groupby("sday"):
            seq = sequential(d, c)
            special.append({"sday": day, "day_type": d.day_type.iloc[0], "rows": len(d), "anomalies": int(seq.anom.sum()),
                            "found_15": (seq.detected_min[seq.anom] <= 15).mean() if seq.anom.any() else np.nan,
                            "found_30": (seq.detected_min[seq.anom] <= 30).mean() if seq.anom.any() else np.nan,
                            "found_45": (seq.detected_min[seq.anom] <= 45).mean() if seq.anom.any() else np.nan,
                            "wape_r1": (d.b4 * d.r1 - d.y).abs().sum() / d.y.sum(),
                            "wape_r2": (d.b4 * d.r2 - d.y).abs().sum() / d.y.sum(),
                            "wape_f_level_r": (d.f_level_r - d.y).abs().sum() / d.y.sum()})
    special = pd.DataFrame(special)

    cb = choose_c(sel, prefix="rB")
    cB = dict(zip(cb.k, cb.c_f1))
    vb = pd.DataFrame([{"period": p, "variant": "B: истина и сигнал по 15-минутным, без поправки",
                        **snapshot_scores(d, k, cB[k], prefix="rB")}
                       for p, d in (("фев + май + июль", sel), ("сентябрь", sep)) for k in range(1, Q)])
    deltas = pd.concat([forecast_delta(sig, f"r{k}", ref) for ref in ("f_level_r", "f_model") for k in range(1, Q)],
                       ignore_index=True)       # с моделью — только май и июль: строки, где есть оба прогноза

    return Results(hours=h, cells=cells, hubs=hub_comparison(cells), stability=month_stability(h, n_perm),
                   sim=simulator_profile(h), signal=sig, forecast=forecast_table(sig), deltas=deltas,
                   c_select=c_sel, c=c, snapshot=snap, detection=det, minute0=m0, slices=slices, lomo=lomo(sel),
                   special=special, variant_b=vb)


def save(res: Results) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res.sim.to_csv(PROFILE_CSV, index=False)
    tables = {"profile_cells": res.cells, "profile_hubs": res.hubs, "profile_stability": res.stability,
              "signal_forecast": res.forecast, "signal_forecast_delta": res.deltas, "detection_c": res.c_select,
              "detection_snapshot": res.snapshot, "detection_by_minute": res.detection, "detection_minute0": res.minute0,
              "detection_slices": res.slices, "detection_lomo": res.lomo, "detection_special_days": res.special,
              "detection_variant_b": res.variant_b}
    for name, t in tables.items():
        t.round(6).to_csv(OUT / f"{name}.csv", index=False)


def main() -> None:
    res = run()
    save(res)
    pd.set_option("display.width", 200)
    print(res.c_select.round(3).to_string(index=False))
    print(res.detection.round(3).to_string(index=False))
    print(res.forecast.round(4).to_string(index=False))
    print(f"intrahour: готово → {PROFILE_CSV.relative_to(config.ROOT)}, {OUT.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
