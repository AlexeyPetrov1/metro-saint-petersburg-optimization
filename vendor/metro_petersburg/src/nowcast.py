"""Уточнение прогноза текущего часа по ходу часа (nowcast) и фильтр тревог (этап 7).

Запуск:
  python -m src.nowcast --fit         # подбор на мае + июле → reports/nowcast/nowcast_params.json и таблицы fit_*
  python -m src.nowcast --september   # проверка на сентябре — один раз; часовые результаты по нему уже известны
                                      # (финальный тест этапа 5), поэтому чистой отложенной выборкой он не является

**Смесь.** Час τ, прошло k = 1, 2, 3 четверти. Прогноз модели ŷ_м сделан в конце часа τ−1 (h = 1).
Частичное отклонение r_k — из этапа 6 (src/intrahour.py: b4 · s_v · доли профиля, только прошлое):

    log(ŷ_k + 1) = w_k · log(b4 · r_k + 1) + (1 − w_k) · log(ŷ_м + 1)

+1 защищает от нулей в первой четверти. На обычных потоках разница с log ŷ пренебрежима. w_k подбирается на мае
+ июле по минимуму WAPE. Вариант «w_k по периоду суток» остаётся, только если он лучше при перекрёстной проверке
май ↔ июль: ΔWAPE > 0 и 95 % ДИ выше нуля. Внутри выборки больше параметров всегда «лучше».

**Интервал.** [ŷ_k · Q10; ŷ_k · Q90], где Q — квантили отношения факт / ŷ_k по k × период суток на мае + июле.
Цель — покрытие 80 %. Отдельно — квантили станций (факт станции / Σ ŷ_k вестибюлей) для serve.

**Фильтр тревог.** Тревога показывается на первом k, где выполнено правило этапа 6 (|r_k − 1| > c_k · p95)
и |ŷ_k − b4| > N людей за час. Большая аномалия — аномальный час с |факт − b4| ≥ 500 входов. N выбирается
на мае + июле: минимум ложных тревог при recall больших аномалий к 45-й минуте ≥ 0,85. Без фильтра recall 0,88,
а условие 0,9 недостижимо: фильтр только убирает тревоги.

**Честность.**
- Прогнозы модели — только вне выборки: май и июль — models/lgbm_fold_2026-05 и lgbm_fold_2026-07 (совпадают
  с кэшем бэктеста), сентябрь — models/lgbm_holdout. Февраль не используется: он в обучении всех фолдов.
- Строки — оценка этапа 6 (`main`): обычные сутки, без флагов, is_source_mismatch и is_slot_closed, in_hourly.
- Параметры замораживаются в nowcast_params.json до прогона сентября.
"""
import argparse
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src import backtest as BT
from src import config, reference
from src import eda_spb as E
from src import features as F
from src import intrahour as I
from src import model as M

OUT = config.ROOT / "reports" / "nowcast"
PARAMS_JSON = OUT / "nowcast_params.json"
VERSION = "nowcast_v1"
FIT_MONTHS, CHECK_MONTH = (5, 7), 9
PERIOD = {5: "май + июль", 7: "май + июль", 9: "сентябрь"}
OOS_MODELS = {5: "lgbm_fold_2026-05", 7: "lgbm_fold_2026-07", 9: "lgbm_holdout"}
KS = (1, 2, 3)
BANDS = {"утро 06–09": [6, 7, 8, 9], "день 10–15": list(range(10, 16)), "вечер 16–19": [16, 17, 18, 19],
         "поздно 20–00": [20, 21, 22, 23, 0]}
BAND_OF = {h: b for b, hs in BANDS.items() for h in hs}
ALL = "все"
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
N_GRID = np.arange(0, 1001, 10)
BIG = 500                       # большая аномалия: |факт − b4| ≥ 500 входов за час (вестибюль)
RECALL_BIG_MIN, RECALL_MINUTE = 0.85, 45
LEVELS = (0.1, 0.9)


# --- Данные ------------------------------------------------------------------------
def stage6_c() -> dict[int, float]:
    """c_k детектора этапа 6 (максимум F1 на фев + мае + июле), зафиксированные в reports/intrahour/detection_c.csv."""
    c = pd.read_csv(I.OUT / "detection_c.csv")
    return {int(k): float(v) for k, v in zip(c.k, c.c_f1)}


def model_oos(months=(*FIT_MONTHS, CHECK_MONTH), data: E.SpbData | None = None) -> pd.DataFrame:
    """q10/q50/q90 модели на h = 1 по вестибюлям — каждый месяц моделью, обученной до него (+7 суток зазора)."""
    cfg = M.load_config()
    d = E.load() if data is None else data
    panel = replace(BT.build_panel(d, final=True), thresholds=pd.DataFrame(cfg["thresholds"]))
    rows = F.add_features(BT.make_rows(panel, horizons=(1,)), panel)
    out = []
    for month in months:
        name = OOS_MODELS[month]
        m, _, meta = M.load_bundle(M.MODELS / name)
        r = rows[rows.sday.dt.month == month]
        if pd.Timestamp(meta["train_end"]) + pd.Timedelta(days=BT.GAP_DAYS) >= r.sday.min():
            raise ValueError(f"{name}: обучена по {meta['train_end']} — для месяца {month} не вне выборки")
        p = m.predict(r)
        out.append(pd.DataFrame({"vestibule_id": r.vestibule_id.to_numpy(), "ts_utc": r.tau.to_numpy(),
                                 "m10": p.q10.to_numpy(), "m50": p.q50.to_numpy(), "m90": p.q90.to_numpy(),
                                 "model": name}))
    return pd.concat(out, ignore_index=True)


def table(months=(*FIT_MONTHS, CHECK_MONTH), h: pd.DataFrame | None = None) -> pd.DataFrame:
    """Строки оценки этапа 6 (`main`) за months + прогноз модели вне выборки, факт b4 · r_k, период суток."""
    h = I.load_hours() if h is None else h
    sig = I.signal_table(h, with_model=False)
    sig = sig[sig.main & sig.month.isin(months)].drop(columns=["f_model", "period"])
    s = sig.merge(model_oos(months), on=["vestibule_id", "ts_utc"], how="inner", validate="one_to_one")
    if len(s) != len(sig):
        raise ValueError(f"нет прогноза модели у {len(sig) - len(s)} строк")
    s["period"] = s.month.map(PERIOD)
    s["band4"] = s.hour.map(BAND_OF)
    s["peak"] = s.hour.isin(PEAK_HOURS)
    s["big"] = s.anom & ((s.y - s.b4).abs() >= BIG)
    for k in (*KS, 4):
        s[f"fact{k}"] = s.b4 * s[f"r{k}"]
    return s.reset_index(drop=True)


def fit_rows(s: pd.DataFrame) -> pd.DataFrame:
    """Строки подбора — только май и июль."""
    return s[s.month.isin(FIT_MONTHS)]


# --- Смесь ---------------------------------------------------------------------------
def blend(model, fact, w) -> np.ndarray:
    """Смесь в логарифмах: w = 0 — модель, w = 1 — факт b4 · r_k."""
    model, fact = np.asarray(model, float), np.clip(np.asarray(fact, float), 0, None)
    return np.expm1(w * np.log1p(fact) + (1 - w) * np.log1p(model))


def _keys(s: pd.DataFrame, by_band: bool) -> pd.Series:
    return s.band4 if by_band else pd.Series(ALL, index=s.index)


def fit_weights(s: pd.DataFrame, by_band: bool = False) -> dict[str, dict[str, float]]:
    """w по k (и периоду суток): минимум суммы |ŷ − y|, то есть WAPE, на сетке W_GRID."""
    out = {}
    for k in KS:
        out[str(k)] = {}
        for b, g in s.groupby(_keys(s, by_band)):
            err = [np.abs(blend(g.m50, g[f"fact{k}"], w) - g.y).sum() for w in W_GRID]
            out[str(k)][b] = float(W_GRID[int(np.argmin(err))])
    return out


def weight_of(s: pd.DataFrame, weights: dict, k: int) -> np.ndarray:
    wk = weights[str(k)]
    return np.array([wk[ALL]] * len(s)) if ALL in wk else s.band4.map(wk).to_numpy(float)


def nowcast(s: pd.DataFrame, weights: dict, k: int) -> np.ndarray:
    """ŷ_k по строкам; k = 0 — прогноз модели."""
    if k == 0:
        return s.m50.to_numpy(float)
    return blend(s.m50, s[f"fact{k}"], weight_of(s, weights, k))


def _boot_delta(err_a, err_b, y, day, n_boot=1000, seed=0) -> tuple[float, float, float]:
    """WAPE(a) − WAPE(b) и 95 % ДИ блочным бутстрепом по дням (минус — a точнее)."""
    agg = pd.DataFrame({"a": err_a, "b": err_b, "y": y, "d": day}).groupby("d")[["a", "b", "y"]].sum()
    counts = E.boot_counts(len(agg), n_boot, np.random.default_rng(seed))
    boot = (counts @ (agg.a - agg.b).to_numpy()) / (counts @ agg.y.to_numpy())
    return float((agg.a.sum() - agg.b.sum()) / agg.y.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def cross_check(s: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Перекрёстно май ↔ июль: w подбирается на одном месяце и проверяется на другом, варианты «по k» и
    «по k × период». Вариант с периодом оставляем, только если он лучше: ΔWAPE > 0 и ДИ выше нуля."""
    fit = fit_rows(s)
    preds = {False: [], True: []}
    for test in FIT_MONTHS:
        train, ev = fit[fit.month != test], fit[fit.month == test]
        for by_band in (False, True):
            w = fit_weights(train, by_band)
            st = stacked(train, w)
            q = fit_quantiles(st.y, st.blend, st.k, st.band4)
            for k in KS:
                f = nowcast(ev, w, k)
                lo, hi = interval(f, k, ev.band4, q)
                preds[by_band].append(pd.DataFrame({"k": k, "y": ev.y.to_numpy(), "sday": ev.sday.to_numpy(),
                                                    "f": f, "covered": (ev.y.to_numpy() >= lo) & (ev.y.to_numpy() <= hi),
                                                    "month": test}))
    a, b = pd.concat(preds[False], ignore_index=True), pd.concat(preds[True], ignore_index=True)
    rows = []
    for k in (*KS, ALL):
        m = (a.k == k) if k != ALL else np.ones(len(a), bool)
        d, lo, hi = _boot_delta(np.abs(a.f[m] - a.y[m]), np.abs(b.f[m] - b.y[m]), a.y[m], a.sday[m])
        rows.append({"k": k, "wape_k": np.abs(a.f[m] - a.y[m]).sum() / a.y[m].sum(),
                     "wape_k_band": np.abs(b.f[m] - b.y[m]).sum() / b.y[m].sum(),
                     "gain_band": d, "lo": lo, "hi": hi,      # gain_band > 0 — вариант с периодом точнее
                     "coverage_k": a.covered[m].mean(), "coverage_k_band": b.covered[m].mean()})
    res = pd.DataFrame(rows)
    tot = res[res.k == ALL].iloc[0]
    return res, bool(tot.gain_band > 0 and tot.lo > 0)


# --- Интервал --------------------------------------------------------------------------
def fit_quantiles(y, pred, k, band) -> dict[str, dict[str, list[float]]]:
    """Квантили LEVELS отношения факт / прогноз по k × периоду суток: {k: {период: [Q10, Q90]}}."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(y, float) / np.asarray(pred, float)
    df = pd.DataFrame({"r": r, "k": k, "band": band})
    df = df[np.isfinite(df.r)]
    out = {}
    for (kk, b), g in df.groupby(["k", "band"]):
        out.setdefault(str(kk), {})[b] = [float(np.quantile(g.r, q)) for q in LEVELS]
    return out


def interval(pred, k: int, band: pd.Series, q: dict) -> tuple[np.ndarray, np.ndarray]:
    qk = q[str(k)]
    lo = band.map(lambda b: qk[b][0]).to_numpy(float)
    hi = band.map(lambda b: qk[b][1]).to_numpy(float)
    pred = np.asarray(pred, float)
    return pred * lo, pred * hi


def stacked(s: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Строка = вестибюль × час × k (1…3): факт, прогноз модели, факт b4 · r_k, смесь."""
    parts = []
    for k in KS:
        parts.append(pd.DataFrame({"vestibule_id": s.vestibule_id, "station_id": s.station_id, "ts_utc": s.ts_utc,
                                   "sday": s.sday, "month": s.month, "band4": s.band4, "k": k, "y": s.y,
                                   "model": s.m50, "fact": s[f"fact{k}"], "blend": nowcast(s, weights, k)}))
    return pd.concat(parts, ignore_index=True)


def station_stack(st: pd.DataFrame) -> pd.DataFrame:
    """Сумма вестибюлей в станцию; только часы, где в строках есть все вестибюли станции из часового файла."""
    n_ves = reference.load_stations().set_index("station_id").n_vestibules
    agg = st.groupby(["station_id", "ts_utc", "k"]).agg(
        y=("y", "sum"), model=("model", "sum"), fact=("fact", "sum"), blend=("blend", "sum"),
        n=("vestibule_id", "size"), sday=("sday", "first"), month=("month", "first"),
        band4=("band4", "first")).reset_index()
    return agg[agg.n == agg.station_id.map(n_ves)].reset_index(drop=True)


# --- Фильтр тревог -----------------------------------------------------------------------
def alarm_table(s: pd.DataFrame, weights: dict, c: dict[int, float], n: float | None) -> pd.DataFrame:
    """По ходу часа: тревога на k = 1…4, если |r_k − 1| > c_k · p95 и (n is None или |ŷ_k − b4| > n).
    ŷ_k — смесь (k ≤ 3) или весь час по 15-минутным данным (k = 4). first_k — первая тревога (0 — нет),
    detected_min — минута первой тревоги с верным знаком на аномальном часе (NaN — не найдена)."""
    A, UP = [], []
    for k in (*KS, 4):
        yk = s.fact4.to_numpy(float) if k == 4 else nowcast(s, weights, k)
        rule = ((s[f"r{k}"] - 1).abs() > c.get(k, 1.0) * s.thr).to_numpy()
        if n is not None:
            rule &= np.abs(yk - s.b4.to_numpy(float)) > n
        A.append(rule)
        UP.append((s[f"r{k}"] > 1).to_numpy())
    A, UP = np.column_stack(A), np.column_stack(UP)
    anom, up = s.anom.to_numpy(), s.up.to_numpy()
    any_a = A.any(1)
    first = np.where(any_a, A.argmax(1) + 1, 0)
    first_up = np.where(any_a, UP[np.arange(len(s)), np.maximum(first - 1, 0)], False)
    correct = A & (UP == up[:, None]) & anom[:, None]
    det = np.where(correct.any(1), 15 * (correct.argmax(1) + 1), np.nan)
    return pd.DataFrame({"first_k": first, "first_up": first_up, "detected_min": det, "anom": anom, "up": up,
                         "big": s.big.to_numpy(), "sday": s.sday.to_numpy()}, index=s.index)


def alarm_summary(a: pd.DataFrame, minute: int = RECALL_MINUTE) -> dict:
    """Тревоги для диспетчера — поднятые к minute (k ≤ minute / 15): в сутки на линию, ложные, precision, recall
    всех и больших аномалий к minute и к 60-й минуте, медиана минут до тревоги (не найденные — «позже 60»)."""
    days = a.sday.nunique()
    kmax = minute // 15
    raised = (a.first_k > 0) & (a.first_k <= kmax)
    good = raised & a.anom & (a.first_up == a.up)
    det = a.detected_min
    med = lambda m: float(det[m].fillna(np.inf).median()) if m.any() else np.nan
    return {"days": int(days), "hours": int(len(a)), "anomalies": int(a.anom.sum()), "big": int(a.big.sum()),
            "alarms_per_day": raised.sum() / days, "false_per_day": (raised & ~a.anom).sum() / days,
            "precision": good.sum() / max(raised.sum(), 1),
            "recall_all": float((det[a.anom] <= minute).mean()), "recall_big": float((det[a.big] <= minute).mean()),
            "recall_all_60": float((det[a.anom] <= 60).mean()), "recall_big_60": float((det[a.big] <= 60).mean()),
            "median_min_all": med(a.anom), "median_min_big": med(a.big)}


def choose_n(s: pd.DataFrame, weights: dict, c: dict, grid=N_GRID) -> tuple[float, pd.DataFrame]:
    """N: минимум ложных тревог в сутки при recall больших аномалий к RECALL_MINUTE ≥ RECALL_BIG_MIN; при равенстве —
    меньший N. Нет допустимого N — фильтра нет (None)."""
    res = pd.DataFrame([{"N": n, **alarm_summary(alarm_table(s, weights, c, n))} for n in grid])
    ok = res[res.recall_big >= RECALL_BIG_MIN]
    if ok.empty:
        return None, res
    best = ok[ok.false_per_day == ok.false_per_day.min()].N.min()
    return float(best), res


# --- Сравнение -------------------------------------------------------------------------------
def compare(s: pd.DataFrame, params: dict, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """k = 0…3 × {модель, факт b4 · r_k, смесь} × срез: WAPE и покрытие; ΔWAPE смеси к модели и к факту с ДИ."""
    w = params["weights"]["w"]
    slices = {"все часы": np.ones(len(s), bool), "пики 07–09, 17–19": s.peak.to_numpy(),
              "аномальные": s.anom.to_numpy()}
    y, day = s.y.to_numpy(float), s.sday.to_numpy()
    rows, deltas = [], []
    for k in (0, *KS):
        methods = {"модель": (s.m50.to_numpy(float), s.m10.to_numpy(float), s.m90.to_numpy(float))}
        if k:
            f = s[f"fact{k}"].to_numpy(float)
            methods["факт b4 · r_k"] = (f, *interval(f, k, s.band4, params["quantiles"]["fact"]))
            b = nowcast(s, w, k)
            methods["смесь"] = (b, *interval(b, k, s.band4, params["quantiles"]["blend"]))
        for sl, m in slices.items():
            for name, (p, lo, hi) in methods.items():
                rows.append({"period": s.period.iloc[0], "k": k, "minute": 15 * k, "method": name, "slice": sl,
                             "n": int(m.sum()), "wape": np.abs(p[m] - y[m]).sum() / y[m].sum(),
                             "coverage": float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean())})
            if k:
                eb = np.abs(methods["смесь"][0] - y)
                for ref in ("модель", "факт b4 · r_k"):
                    d, lo_, hi_ = _boot_delta(eb[m], np.abs(methods[ref][0] - y)[m], y[m], day[m], n_boot)
                    deltas.append({"period": s.period.iloc[0], "k": k, "minute": 15 * k, "slice": sl, "vs": ref,
                                   "delta_wape": d, "lo": lo_, "hi": hi_})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def station_coverage(s: pd.DataFrame, params: dict) -> pd.DataFrame:
    st = station_stack(stacked(s, params["weights"]["w"]))
    lo, hi = np.empty(len(st)), np.empty(len(st))
    for k in KS:
        m = (st.k == k).to_numpy()
        lo[m], hi[m] = interval(st.blend[m], k, st.band4[m], params["quantiles"]["station"])
    st["cov"] = (st.y >= lo) & (st.y <= hi)
    return (st.groupby("k").agg(n=("y", "size"), coverage=("cov", "mean"),
                                wape=("y", lambda yy: np.abs(st.loc[yy.index, "blend"] - yy).sum() / yy.sum()))
            .reset_index().assign(period=s.period.iloc[0]))


# --- Подбор и заморозка ------------------------------------------------------------------------
def _sha(params: dict) -> str:
    body = {k: v for k, v in params.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def fit_params(s: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Все параметры — только по строкам мая и июля (fit_rows); остальные месяцы в s ни на что не влияют."""
    fit = fit_rows(s).reset_index(drop=True)
    if fit.empty:
        raise ValueError("нет строк мая и июля")
    cv, use_band = cross_check(fit)
    w = fit_weights(fit, by_band=use_band)
    st = stacked(fit, w)
    q = {"blend": fit_quantiles(st.y, st.blend, st.k, st.band4),
         "fact": fit_quantiles(st.y, st.fact, st.k, st.band4)}
    sst = station_stack(st)
    q["station"] = fit_quantiles(sst.y, sst.blend, sst.k, sst.band4)
    c = stage6_c()
    n, grid = choose_n(fit, w, c)
    params = {"version": VERSION, "fit_months": list(FIT_MONTHS), "rows": int(len(fit)),
              "bands": BANDS, "weights": {"variant": "k × период суток" if use_band else "k", "w": w},
              "quantiles": q, "levels": list(LEVELS),
              "alarm": {"c": {str(k): v for k, v in c.items()}, "N": n, "big": BIG,
                        "recall_big_min": RECALL_BIG_MIN, "minute": RECALL_MINUTE},
              "models": {str(m): OOS_MODELS[m] for m in (*FIT_MONTHS, CHECK_MONTH)}}
    params["sha256"] = _sha(params)
    return params, {"cv": cv, "n_grid": grid}


def load_params(path=PARAMS_JSON) -> dict:
    if not path.exists():
        raise SystemExit(f"нет {path} — сначала python -m src.nowcast --fit")
    p = json.loads(path.read_text(encoding="utf-8"))
    if _sha(p) != p.get("sha256"):
        raise SystemExit(f"{path.name} изменён после подбора: хэш не совпадает")
    return p


def alarm_compare(s: pd.DataFrame, params: dict) -> pd.DataFrame:
    """До и после фильтра."""
    w, c = params["weights"]["w"], {int(k): v for k, v in params["alarm"]["c"].items()}
    n = params["alarm"]["N"]
    return pd.DataFrame([{"period": s.period.iloc[0], "filter": label, "N": nn,
                          **alarm_summary(alarm_table(s, w, c, nn))}
                         for label, nn in (("без фильтра (правило этапа 6)", None), (f"фильтр N = {n}", n))])


def run_fit() -> dict:
    s = table(months=FIT_MONTHS)
    params, extra = fit_params(s)
    OUT.mkdir(parents=True, exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    cmp, dl = compare(s, params)
    extra["cv"].round(6).to_csv(OUT / "fit_cv.csv", index=False)
    extra["n_grid"].round(6).to_csv(OUT / "fit_n_grid.csv", index=False)
    cmp.round(6).to_csv(OUT / "fit_compare.csv", index=False)
    dl.round(6).to_csv(OUT / "fit_compare_delta.csv", index=False)
    station_coverage(s, params).round(6).to_csv(OUT / "fit_station.csv", index=False)
    alarm_compare(s, params).round(6).to_csv(OUT / "fit_alarms.csv", index=False)
    return params


def run_september(force: bool = False) -> None:
    done = OUT / "september_compare.csv"
    if done.exists() and not force:
        raise SystemExit("сентябрь уже проверен: reports/nowcast/september_compare.csv существует")
    params = load_params()
    s = table(months=(CHECK_MONTH,))
    cmp, dl = compare(s, params)
    cmp.round(6).to_csv(done, index=False)
    dl.round(6).to_csv(OUT / "september_compare_delta.csv", index=False)
    station_coverage(s, params).round(6).to_csv(OUT / "september_station.csv", index=False)
    alarm_compare(s, params).round(6).to_csv(OUT / "september_alarms.csv", index=False)
    (OUT / "september_run.json").write_text(json.dumps({"params_sha256": params["sha256"], "rows": int(len(s))},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true", help="подбор на мае + июле → nowcast_params.json")
    ap.add_argument("--september", action="store_true", help="проверка на сентябре (один раз)")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.fit:
        p = run_fit()
        print(f"веса: {p['weights']}; N = {p['alarm']['N']}")
        print(pd.read_csv(OUT / "fit_cv.csv").round(4).to_string(index=False))
    if args.september:
        run_september()
    for prefix in ("fit", "september"):
        path = OUT / f"{prefix}_compare.csv"
        if path.exists() and (args.fit or args.september):
            c = pd.read_csv(path)
            print(c[c.slice == "все часы"].pivot_table(index="k", columns="method", values=["wape", "coverage"]).round(4))
            print(pd.read_csv(OUT / f"{prefix}_alarms.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
