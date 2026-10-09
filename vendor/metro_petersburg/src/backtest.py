"""Каркас бэктеста (раздел 7 ТЗ) и лестница бейзлайнов (этап 3).

Запуск:
  python -m src.backtest                  # 4 фолда (тест май–август) → reports/backtest/
  python -m src.backtest --export-august  # прогнозы лучшего бейзлайна на август → data/predictions/baseline_aug.json
  python -m src.backtest --final          # + финальный тест на 1–29 сентября (один раз, в конце проекта)

Строка = (вестибюль, исходный час t, горизонт h): прогноз делается в конце часа t на час τ = t + h. Цели и метрики —
часы 05–00 без флагов; входы (r(t), сглаженное r) — по наблюдённому потоку, исключены только закрытые часы.
Без --final сентябрьский поток обнуляется в NaN до любых расчётов: финальный тест не видит никакой код.
Модель — что угодно с методами fit(train) и predict(rows) → q10, q50, q90 (на этапе 4 сюда встанет LightGBM).
"""
import argparse
import copy
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from src import baseline as B
from src import config, contract
from src import eda_spb as E

OUT = config.ROOT / "reports" / "backtest"
AUG_OUT = config.PREDICTIONS / "baseline_aug.json"
SEPTEMBER = pd.Timestamp("2026-09-01")
FINAL_END = pd.Timestamp("2026-09-29")
TRAIN_START = E.ANALYSIS_START                   # прогрев нормы: 4 полные недели обычных дней после 11.01
GAP_DAYS = 7
HORIZONS = (1, 2)
METRIC_HOURS = [5, *E.WORK_HOURS]                # 05–00 (раздел 7 ТЗ)
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
BANDS = {"05–06": [5, 6], "07–09": [7, 8, 9], "10–15": list(range(10, 16)), "16–19": [16, 17, 18, 19],
         "20–00": [20, 21, 22, 23, 0]}
BAND_OF = {h: b for b, hs in BANDS.items() for h in hs}
EDA_BAND_OF = {h: b for b, hs in E.HOUR_BANDS.items() for h in hs}   # для порогов аномалии; 05–06 → «все часы»
INCIDENT_DAY = pd.Timestamp("2026-08-31")
R_CLIP = (0.3, 3.0)
QUANTILES = (0.1, 0.5, 0.9)


# --- Фолды ------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp     # обучение — сутки метро < train_end
    test_start: pd.Timestamp
    test_end: pd.Timestamp      # включительно


def folds(final: bool = False) -> list[Fold]:
    """Расширяющееся окно: тест — май, июнь, июль, август; зазор 7 суток. Сентябрь — только с final=True."""
    out = []
    for month, name in ((5, "май"), (6, "июнь"), (7, "июль"), (8, "август")):
        start = pd.Timestamp(2026, month, 1)
        out.append(Fold(name, start - pd.Timedelta(days=GAP_DAYS), start, start + pd.offsets.MonthEnd(0)))
    if final:
        out.append(Fold("final", SEPTEMBER - pd.Timedelta(days=GAP_DAYS), SEPTEMBER, FINAL_END))
    return out


def get_fold(name: str, final: bool = False) -> Fold:
    if name == "final" and not final:
        raise PermissionError("Сентябрь — финальный тест: доступен только с --final")
    return {f.name: f for f in folds(final)}[name]


# --- Данные -----------------------------------------------------------------------
@dataclass
class Panel:
    d: E.SpbData
    norms: dict[str, np.ndarray]
    r_in: np.ndarray            # вход r(t) по наблюдённому потоку
    r_level_in: np.ndarray      # то же к норме с поправкой уровня
    ewm2_in: np.ndarray         # сглаженное r(t), полупериод 2 ч
    thresholds: pd.DataFrame    # p95 |r − 1| по группе × периоду суток (EDA), без сентября
    final: bool


def load_panel(final: bool = False) -> Panel:
    return build_panel(E.load(), final)


def build_panel(d: E.SpbData, final: bool = False) -> Panel:
    """Нормы, входы и пороги по потоку d.grid (тест утечки подаёт сюда обрезанный поток)."""
    g = d.grid
    if not final:   # сентябрь недоступен: поток в NaN до расчёта нормы и входов
        entries = np.where((g.sday >= SEPTEMBER)[:, None], np.nan, g.entries)
        regular = d.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
        d = replace(d, grid=E.make_grid(g.ts, entries, g.closed, g.flagged, regular),
                    df=d.df[d.df.sday < SEPTEMBER].reset_index(drop=True))
        g = d.grid
    norms = B.norm_matrices(g, d.calendar)
    r_in = B.observed_ratio(g, norms["b4"])
    nt = E.noise_table(d)   # пороги как в EDA, только на доступных данных
    return Panel(d=d, norms=norms, r_in=r_in, r_level_in=B.observed_ratio(g, norms["b4_level"]),
                 ewm2_in=E.ewm_ratio(r_in, g.ts, 2), thresholds=nt, final=final)


def anomaly_threshold(panel: Panel, group: pd.Series, hour: pd.Series, level: str = "p95") -> np.ndarray:
    """Квантиль `level` (p80 / p90 / p95) |r − 1| группы и периода суток (EDA); для 05–06 — по всем часам группы."""
    t = panel.thresholds.set_index(["level", "band"])[level]
    band = hour.map(EDA_BAND_OF).fillna("все часы")
    return t.reindex(pd.MultiIndex.from_arrays([group.to_numpy(), band.to_numpy()])).to_numpy()


DEFAULT_FLAG_RULE = {"kind": "threshold", "level": "p95"}


def anomaly_flag(df: pd.DataFrame, panel: Panel, rule: dict | None = None) -> np.ndarray:
    """Флаг аномалии прогноза. df — q10, q50, q90, норма b, group, hour.
    threshold: |q50 / b − 1| выше квантиля rule["level"] группы и периода; interval: норма вне [q10; q90]."""
    rule = rule or DEFAULT_FLAG_RULE
    if rule["kind"] == "interval":
        return ((df.b < df.q10) | (df.b > df.q90)).to_numpy()
    thr = anomaly_threshold(panel, df.group, df.hour, rule["level"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((df.q50 / df.b - 1).abs() > thr).to_numpy()


def make_rows(panel: Panel, horizons=HORIZONS, for_export: bool = False) -> pd.DataFrame:
    """Строки прогноза. Обычный режим: τ в часах 05–00, без флагов, с известным потоком и нормой.
    for_export: все часы 05–00, где метро и вестибюль открыты (в том числе часы инцидента), факт может быть неизвестен."""
    g, ves = panel.d.grid, panel.d.vestibules
    T, V = g.y.shape
    b4 = panel.norms["b4"]
    hour_ok = np.isin(g.hour, METRIC_HOURS)[:, None] & (g.sday >= TRAIN_START)[:, None] & ~np.isnan(b4)
    target_ok = hour_ok & (~g.closed if for_export else (~g.flagged & ~np.isnan(g.y)))
    naive = np.where(g.closed, np.nan, g.entries)
    parts = []
    for h in horizons:
        tau, v = np.nonzero(target_ok)
        keep = tau - h >= 0
        tau, v = tau[keep], v[keep]
        t = tau - h
        back = tau - 168
        parts.append(pd.DataFrame({
            "vi": v, "h": h, "t": g.ts[t], "tau": g.ts[tau], "tau_local": g.local[tau], "hour": g.hour[tau],
            "sday": g.sday[tau], "y": g.y[tau, v],
            **{k: m[tau, v] for k, m in panel.norms.items()},
            "naive168": np.where(back >= 0, naive[np.maximum(back, 0), v], np.nan),
            "r_b4_t": panel.r_in[t, v], "r_level_t": panel.r_level_in[t, v], "ewm2_t": panel.ewm2_in[t, v],
        }))
    rows = pd.concat(parts, ignore_index=True)
    rows["vestibule_id"] = ves.vestibule_id.to_numpy()[rows.vi]
    rows["station_id"] = ves.station_id.to_numpy()[rows.vi]
    rows["group"] = ves.group.to_numpy()[rows.vi]
    rows["band"] = rows.hour.map(BAND_OF)
    rows["month"] = rows.sday.dt.month
    rows["day_type"] = panel.d.calendar.set_index("date").day_type.reindex(rows.sday).to_numpy()
    rows["is_holiday"] = rows.day_type == "праздник"
    nights = E.event_days(panel.d.events[panel.d.events.source == "по данным потока"])
    rows["is_special"] = rows.sday.isin(set(nights) | {INCIDENT_DAY})
    with np.errstate(invalid="ignore", divide="ignore"):
        rows["r_tau"] = rows.y / rows.b4
    rows["is_anomaly"] = (rows.r_tau - 1).abs() > anomaly_threshold(panel, rows.group, rows.hour)
    return rows.drop(columns="vi")


# --- Модели -----------------------------------------------------------------------
class Model(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "Model": ...

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame: ...   # колонки q10, q50, q90, индекс как у rows


@dataclass
class RatioQuantiles:
    """Бейзлайн: q50 — точечный прогноз f; q10 и q90 — f × эмпирические квантили отношения y / f на обучении
    по ячейке «группа × период суток × h» (нет ячейки — по всему обучению). При f < 1 все квантили = f.
    center=True — q50 = f × медиана y / f в ячейке. По умолчанию выключено: медиана прошлых месяцев не переносится
    на тестовый месяц (уровень сдвигается), WAPE всей лестницы от неё не лучше (docs/backtest.md, --center-check)."""
    name: str
    label: str
    point: Callable[[pd.DataFrame], pd.Series]
    cells: tuple = ("group", "band", "h")
    center: bool = False
    q: pd.DataFrame | None = None
    q_all: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "RatioQuantiles":
        f = self.point(train)
        ok = (f >= 1) & train.y.notna()
        ratio = (train.y / f)[ok]
        keys = [train.loc[ok, c] for c in self.cells]
        self.q = ratio.groupby(keys).quantile(list(QUANTILES)).unstack()
        self.q_all = ratio.quantile(list(QUANTILES))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        f = self.point(rows).to_numpy(dtype=float)
        idx = pd.MultiIndex.from_frame(rows[list(self.cells)])
        k = {t: self.q[t].reindex(idx).fillna(self.q_all[t]).to_numpy() for t in QUANTILES}
        mid = k[0.5] if self.center else 1.0
        small = f < 1
        qs = np.column_stack([np.where(small, f, f * k[0.1]), np.where(small, f, f * mid),
                              np.where(small, f, f * k[0.9])])
        qs = np.sort(qs, axis=1)
        return pd.DataFrame(qs, columns=["q10", "q50", "q90"], index=rows.index)


def _times_ratio(col: str, norm: str = "b4") -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[norm] * df[col].clip(*R_CLIP).fillna(1.0)


def baseline_ladder() -> list[RatioQuantiles]:
    return [
        RatioQuantiles("seasonal_naive", "Сезонный наивный (неделю назад)", lambda df: df.naive168.fillna(df.b4)),
        RatioQuantiles("b4", "Профильный бейзлайн b (4 недели)", lambda df: df.b4),
        RatioQuantiles("b8", "b по 8 неделям", lambda df: df.b8),
        RatioQuantiles("b4_level", "b × уровень последних 7 дней", lambda df: df.b4_level),
        RatioQuantiles("b4_x_r", "b × r(t)", _times_ratio("r_b4_t")),
        RatioQuantiles("b4_x_ewm2", "b × сглаженное r (2 ч)", _times_ratio("ewm2_t")),
        RatioQuantiles("b4_level_x_r", "b × уровень × r(t)", _times_ratio("r_level_t", "b4_level")),
    ]


# --- Прогон и метрики -------------------------------------------------------------
def split(rows: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = rows[(rows.sday < fold.train_end) & ~rows.is_special & rows.y.notna()]
    test = rows[(rows.sday >= fold.test_start) & (rows.sday <= fold.test_end)]
    return train, test


def run(rows: pd.DataFrame, models: list[Model], fold_list: list[Fold],
        on_fit: Callable[[Fold, Model], None] | None = None) -> pd.DataFrame:
    """Прогнозы всех моделей на тестах всех фолдов (обучение — заново в каждом фолде).
    on_fit(fold, model) — вызывается с обученной моделью (например, чтобы сохранить её калибровку)."""
    out = []
    for fold in fold_list:
        train, test = split(rows, fold)
        for proto in models:
            m = copy.deepcopy(proto).fit(train)
            if on_fit is not None:
                on_fit(fold, m)
            p = m.predict(test)
            out.append(test[["vestibule_id", "station_id", "group", "h", "tau", "hour", "band", "sday", "y",
                             "is_special", "is_anomaly", "is_holiday"]].assign(model=m.name, fold=fold.name).join(p))
    return pd.concat(out, ignore_index=True)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def metrics(df: pd.DataFrame) -> dict:
    y, q10, q50, q90 = (df[c].to_numpy(dtype=float) for c in ("y", "q10", "q50", "q90"))
    pb = {f"pinball_{int(t * 100)}": pinball(y, q, t) for t, q in zip(QUANTILES, (q10, q50, q90))}
    mean_pb = np.mean(list(pb.values()))
    return {"n": len(y), "wape": float(np.abs(y - q50).sum() / y.sum()), "mae": float(np.abs(y - q50).mean()),
            **pb, "pinball": float(mean_pb), "pinball_norm": float(mean_pb / y.mean()),
            "coverage": float(np.mean((y >= q10) & (y <= q90)))}


def slices(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {"все часы": pd.Series(True, index=df.index), "пики 07–09, 17–19": df.hour.isin(PEAK_HOURS)}
    out.update({f"группа: {g}": df.group == g for g in E.GROUPS})
    out["аномальные часы"] = df.is_anomaly
    out["праздники (контроль)"] = df.is_holiday
    return out


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    """Метрики по модели × фолд (и «все») × h × срез; особые дни исключены."""
    gen = pred[~pred.is_special]
    rows = []
    for (model, h), g in gen.groupby(["model", "h"]):
        for fold, gf in [("все", g)] + list(g.groupby("fold", sort=False)):
            for name, mask in slices(gf).items():
                part = gf[mask]
                if len(part):
                    rows.append({"model": model, "fold": fold, "h": h, "slice": name, **metrics(part)})
    return pd.DataFrame(rows)


def evaluate_special(pred: pd.DataFrame) -> pd.DataFrame:
    """Особые дни (31.08, вечера перед ночами, когда метро работало всю ночь) — отдельно."""
    sp = pred[pred.is_special]
    return pd.DataFrame([{"model": m, "sday": d.date(), "h": h, **metrics(g)}
                         for (m, d, h), g in sp.groupby(["model", "sday", "h"])])


def best_model(res: pd.DataFrame) -> str:
    r = res[(res.fold == "все") & (res.slice == "все часы")]
    return r.groupby("model").wape.mean().idxmin()


def ablation_table(res: pd.DataFrame, labels: dict[str, str]) -> str:
    """Таблица раздела 7 ТЗ: WAPE все часы и пики, покрытие q10–q90 — среднее по h и отдельно t+1 / t+2."""
    r = res[res.fold == "все"]
    piv = lambda s, col: r[r.slice == s].pivot(index="model", columns="h", values=col)
    wa, wp, cv = piv("все часы", "wape"), piv("пики 07–09, 17–19", "wape"), piv("все часы", "coverage")
    pct = lambda x: f"{x * 100:.2f} %".replace(".", ",")
    lines = ["| Модель | WAPE, все часы | WAPE, пики | Покрытие q10–q90 | WAPE t+1 / t+2 |", "|---|---|---|---|---|"]
    for name in labels:
        if name in wa.index:
            lines.append(f"| {labels[name]} | {pct(wa.loc[name].mean())} | {pct(wp.loc[name].mean())} | "
                         f"{pct(cv.loc[name].mean())} | {pct(wa.loc[name, 1])} / {pct(wa.loc[name, 2])} |")
    return "\n".join(lines)


# --- Прогнозы для команды ---------------------------------------------------------
def station_frame(rows: pd.DataFrame, f: pd.Series, panel: Panel) -> pd.DataFrame:
    """Сумма вестибюлей в станцию; строка станции полная, если есть все открытые в этот час вестибюли станции."""
    g, ves = panel.d.grid, panel.d.vestibules
    df = rows.assign(f=f.to_numpy(), b=rows.b4, y_missing=rows.y.isna())
    agg = df.groupby(["station_id", "tau", "h"]).agg(
        y=("y", "sum"), y_missing=("y_missing", "sum"), f=("f", "sum"), b=("b", "sum"),
        n=("vestibule_id", "size"), hour=("hour", "first"), sday=("sday", "first"), tau_local=("tau_local", "first"),
        is_special=("is_special", "first")).reset_index()
    agg.loc[agg.y_missing > 0, "y"] = np.nan                     # факт станции — только если известен у всех
    open_ = (pd.DataFrame(~g.closed, index=g.ts, columns=ves.station_id.to_numpy())
             .T.groupby(level=0).sum().T.stack())                  # (час, станция) → открытых вестибюлей
    agg["n_open"] = open_.reindex(pd.MultiIndex.from_arrays([agg.tau, agg.station_id])).to_numpy()
    agg["group"] = agg.station_id.map(E.STATION_GROUP)
    agg["band"] = agg.hour.map(BAND_OF)
    return agg


def export_august(panel: Panel, model_name: str) -> list[dict]:
    """Прогнозы лучшего бейзлайна на август по станциям (фолд 4: обучение до 25.07)."""
    fold = get_fold("август")
    proto = {m.name: m for m in baseline_ladder()}[model_name]
    rows = make_rows(panel, for_export=True)
    train_rows, test_rows = split(rows, fold)
    st_train = station_frame(train_rows, proto.point(train_rows), panel)
    st_train = st_train[(st_train.n == st_train.n_open) & st_train.y.notna() & ~st_train.is_special]
    st_test = station_frame(test_rows, proto.point(test_rows), panel)
    q = RatioQuantiles(proto.name, proto.label, lambda df: df.f).fit(st_train).predict(st_test)
    return station_records(st_test.join(q), panel, f"baseline_{model_name}_v1")


def station_records(st: pd.DataFrame, panel: Panel, model_version: str, rule: dict | None = None) -> list[dict]:
    """Записи контракта из станционной таблицы с q10, q50, q90; is_anomaly — по правилу `anomaly_flag`
    (по умолчанию |q50 / норма − 1| выше p95 группы и периода суток)."""
    anomaly = anomaly_flag(st, panel, rule)
    records = []
    for r, an in zip(st.itertuples(), anomaly):
        qs = sorted(round(x) for x in (r.q10, r.q50, r.q90))
        records.append({"station_id": r.station_id, "ts": r.tau_local.tz_localize(config.SPB_TZ).isoformat(),
                        "horizon_min": 60 * r.h, "q10": qs[0], "q50": qs[1], "q90": qs[2], "baseline": round(r.b),
                        "is_anomaly": bool(an), "model_version": model_version})
    return sorted(records, key=lambda x: (x["ts"], x["station_id"], x["horizon_min"]))


# --- main -------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="добавить финальный тест на сентябре (один раз в конце)")
    ap.add_argument("--export-august", action="store_true", help="прогнозы лучшего бейзлайна на август по контракту")
    ap.add_argument("--center-check", action="store_true", help="сравнить q50 = f и q50 = f × медиана y / f")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(final=args.final)
    rows = make_rows(panel)
    models = baseline_ladder()
    labels = {m.name: m.label for m in models}
    pred = run(rows, models, folds(args.final))
    res = evaluate(pred)
    special = evaluate_special(pred)
    suffix = "_final" if args.final else ""
    res.round(6).to_csv(OUT / f"baselines{suffix}.csv", index=False)
    special.round(6).to_csv(OUT / f"special_days{suffix}.csv", index=False)
    panel.thresholds.round(5).to_csv(OUT / "anomaly_thresholds.csv", index=False)
    best = best_model(res if not args.final else evaluate(pred[pred.fold != "final"]))   # выбор — без сентября
    print(f"строк: {len(rows):,}; тест: {pred[pred.model == models[0].name].shape[0]:,} строк × {len(models)} моделей")
    print(ablation_table(res, labels))
    print(f"лучший бейзлайн: {best} — {labels[best]}")

    if args.center_check:
        centered = [replace(m, name=f"{m.name}_center", center=True) for m in models]
        res_c = evaluate(run(rows, centered, folds(args.final)))
        print("\nq50 = f × медиана y / f в ячейке, WAPE (среднее по h) — без центрирования → с ним:")
        for m in models:
            wa = lambda r, name: r[(r.model == name) & (r.fold == "все") & (r.slice == "все часы")].wape.mean()
            pct = lambda x: f"{x * 100:.2f} %".replace(".", ",")
            print(f"  {labels[m.name]}: {pct(wa(res, m.name))} → {pct(wa(res_c, m.name + '_center'))}")

    if args.export_august:
        if args.final:
            raise SystemExit("--export-august — без --final: август строится на фолде 4")
        recs = export_august(panel, best)
        contract._write_records(AUG_OUT, recs)
        preds = contract.validate_file(AUG_OUT)
        print(f"{AUG_OUT.relative_to(config.ROOT)}: {len(preds)} записей, контракт OK, "
              f"аномальных {sum(p.is_anomaly for p in preds)}, model_version {preds[0].model_version}")


if __name__ == "__main__":
    main()
