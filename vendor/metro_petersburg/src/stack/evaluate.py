"""Оценка стекинга: метрики по горизонтам и срезам, ΔWAPE с ДИ и тестом Диболда–Мариано, веса, демо, контракт.

**Метрики** — в потоке слота (z → (n + 1)·e^z − 1): WAPE по q50, покрытие q10–q90, pinball, нормированный на средний y
(как BT.metrics). Срезы: все слоты; пики (час слота 07–09, 17–19); аномальные (час слота аномален по истине этапа 6).

**ΔWAPE** = WAPE(база) − WAPE(стекинг), > 0 — стекинг точнее. ДИ — блочный бутстреп по суткам, p — Диболд–Мариано
на суточных суммах ошибок (eda_spb.compare_forecasts). База — B1, лучшая базовая модель (минимум WAPE на тех же
строках и горизонте: так строже для стекинга) и простое среднее.

**Демо.** Правило выбора зафиксировано до прогона сентября и смотрит только на факт и норму: рабочий день сентября
с максимальным |Σy / Σнорма − 1| линии в слотах 16:00–19:30; станция — с наибольшим |отклонением| в тот же вечер.
"""
import numpy as np
import pandas as pd

from src import backtest as BT
from src import config, contract, reference
from src import eda_spb as E
from src.stack import base as B
from src.stack import data as D

ALL = "все"
SLICES = {"все слоты": None, "пики 07–09, 17–19": "peak", "аномальные часы": "anom"}
DEMO_HOURS = (16, 17, 18, 19)
DEMO_ORIGINS = ("15:00", "20:00")


def _y(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[["y"]].join(B.to_y(df, name))


def metrics_table(df: pd.DataFrame, names, period: str) -> pd.DataFrame:
    """Модель × горизонт (и «все») × срез: n, WAPE, покрытие, pinball (норм.); у стекинга — покрытие до поправки."""
    out = []
    for name in names:
        d = _y(df, name)
        raw = _y(df, "stackraw") if name == "stack" and "stackraw_50" in df else None
        for h in (*D.HORIZONS, ALL):
            mh = np.ones(len(df), bool) if h == ALL else (df.horizon == h).to_numpy()
            for sl, col in SLICES.items():
                m = mh if col is None else mh & df[col].to_numpy(bool)
                if not m.any():
                    continue
                met = BT.metrics(d[m])
                row = {"period": period, "model": name, "label": B.LABELS[name], "horizon": h, "slice": sl,
                       "n": met["n"], "wape": met["wape"], "coverage": met["coverage"],
                       "pinball_norm": met["pinball_norm"]}
                if raw is not None:
                    row["coverage_raw"] = BT.metrics(raw[m])["coverage"]
                out.append(row)
    return pd.DataFrame(out)


def best_base(df: pd.DataFrame, names=B.BASES) -> dict:
    """Лучшая базовая модель по WAPE q50 на каждом горизонте (и «все»), срез «все слоты»."""
    out = {}
    for h in (*D.HORIZONS, ALL):
        m = np.ones(len(df), bool) if h == ALL else (df.horizon == h).to_numpy()
        w = {n: np.abs(df.y[m] - B.to_y(df[m], n).q50).sum() / df.y[m].sum() for n in names}
        out[h] = min(w, key=w.get)
    return out


def delta_table(df: pd.DataFrame, period: str, n_boot: int = 1000) -> pd.DataFrame:
    """ΔWAPE стекинга к B1, к лучшей базовой и к простому среднему по горизонтам и срезам."""
    best = best_base(df)
    q50 = {n: B.to_y(df, n).q50.to_numpy() for n in (*B.BASES, "mean", "stack")}
    y, day = df.y.to_numpy(float), df.sday.to_numpy()
    out = []
    for h in (*D.HORIZONS, ALL):
        mh = np.ones(len(df), bool) if h == ALL else (df.horizon == h).to_numpy()
        for sl, col in SLICES.items():
            m = mh if col is None else mh & df[col].to_numpy(bool)
            for vs, ref in (("B1", "b1"), ("лучшая базовая", best[h]), ("среднее", "mean")):
                r = E.compare_forecasts(y[m], q50[ref][m], q50["stack"][m], day[m], n_boot=n_boot)
                out.append({"period": period, "horizon": h, "slice": sl, "vs": vs, "ref_model": ref,
                            "wape_ref": r["wape_base"], "wape_stack": r["wape_alt"], "dwape": r["dwape"],
                            "lo": r["lo"], "hi": r["hi"], "p_dm": r["p_dm"], "n_days": r["n_days"], "n": int(m.sum())})
    return pd.DataFrame(out)


def minute_table(df: pd.DataFrame, period: str, names=(*B.BASES, "stack"), n_boot: int = 1000) -> pd.DataFrame:
    """WAPE по горизонту × минуте прогноза (:00 / :30) и ΔWAPE стекинга к B1 с ДИ. В :30 прогноз B1 сделан в :00
    (на 30 минут старее), а на горизонте 120 — с h = 3."""
    out = []
    for (h, mi), g in df.groupby(["horizon", "minute"]):
        row = {"period": period, "horizon": int(h), "minute": int(mi), "n": len(g)}
        for name in names:
            row[f"wape_{name}"] = np.abs(g.y - B.to_y(g, name).q50).sum() / g.y.sum()
        r = E.compare_forecasts(g.y.to_numpy(float), B.to_y(g, "b1").q50.to_numpy(), B.to_y(g, "stack").q50.to_numpy(),
                                g.sday.to_numpy(), n_boot=n_boot)
        out.append({**row, "dwape_vs_b1": r["dwape"], "lo": r["lo"], "hi": r["hi"], "p_dm": r["p_dm"]})
    return pd.DataFrame(out)


def weights_table(fits: dict[str, dict]) -> pd.DataFrame:
    """fits: {«обучение»: параметры весов} → длинная таблица w по квантилю и ячейке (и по горизонту)."""
    out = []
    for fit, p in fits.items():
        for level in ("w", "w_k"):
            for q, cells in p[level].items():
                for cell, w in cells.items():
                    out.append({"fit": fit, "variant": p["variant"], "level": "ячейка" if level == "w" else "горизонт",
                                "quantile": f"q{q}", "cell": cell, **{f"w_{n}": x for n, x in zip(p["names"], w)}})
    return pd.DataFrame(out)


# --- Станции и контракт ----------------------------------------------------------------
def station_sum(df: pd.DataFrame, name: str = "stack") -> pd.DataFrame:
    """Суммы вестибюлей в станцию по (станция, t, k): факт, норма, q50 модели; только полные станции."""
    n_ves = reference.load_stations().set_index("station_id").n_vestibules
    d = df.assign(q50=B.to_y(df, name).q50.to_numpy())
    st = d.groupby(["station_id", "t", "k"]).agg(
        y=("y", "sum"), n=("n", "sum"), q50=("q50", "sum"), cnt=("vestibule_id", "size"), slot=("slot", "first"),
        hour=("hour", "first"), band4=("band4", "first"), group=("group", "first"), sday=("sday", "first"),
        horizon=("horizon", "first")).reset_index()
    return st[st.cnt == st.station_id.map(n_ves)].reset_index(drop=True)


def fit_station_quantiles(st: pd.DataFrame) -> dict:
    """Квантили 0,1 / 0,9 отношения факт / Σq50 станции по ячейке «k × период» и по k."""
    r = st.y / st.q50.where(st.q50 > 0)
    ok = r.notna()
    cell = r[ok].groupby([st.k[ok], st.band4[ok]]).quantile([0.1, 0.9]).unstack()
    by_k = r[ok].groupby(st.k[ok]).quantile([0.1, 0.9]).unstack()
    return {"cell": {f"{k}|{b}": [float(v[0.1]), float(v[0.9])] for (k, b), v in cell.iterrows()},
            "k": {str(k): [float(v[0.1]), float(v[0.9])] for k, v in by_k.iterrows()}}


def station_interval(st: pd.DataFrame, sq: dict) -> pd.DataFrame:
    key = st.k.astype(str) + "|" + st.band4
    c = np.array([sq["cell"].get(x, sq["k"][str(k)]) for x, k in zip(key, st.k)], float)
    q = np.sort(np.column_stack([st.q50 * c[:, 0], st.q50, st.q50 * c[:, 1]]), axis=1)
    return pd.DataFrame(q, columns=["q10", "q50", "q90"], index=st.index)


def station_records(st: pd.DataFrame, rule: dict, thresholds: pd.DataFrame, version: str) -> list[dict]:
    """Записи контракта: станция, начало слота (местное время), horizon_min = 30k, квантили, норма, флаг аномалии
    по замороженному правилу (|q50 / норма − 1| выше порога группы и периода суток; пороги — часовые)."""
    from types import SimpleNamespace
    df = st.assign(b=st.n)
    flag = BT.anomaly_flag(df, SimpleNamespace(thresholds=thresholds), rule)
    out = []
    for r, an in zip(df.itertuples(), flag):
        qs = sorted(round(x) for x in (r.q10, r.q50, r.q90))
        out.append({"station_id": r.station_id, "ts": r.slot.tz_convert(config.SPB_TZ).isoformat(),
                    "horizon_min": int(r.horizon), "q10": qs[0], "q50": qs[1], "q90": qs[2], "baseline": round(r.b),
                    "is_anomaly": bool(an), "model_version": version})
    return sorted(out, key=lambda x: (x["ts"], x["station_id"], x["horizon_min"]))


def validate(records: list[dict]) -> int:
    return len(contract.validate_records(records))


# --- Демо ---------------------------------------------------------------------------------
def demo_choice(slots: pd.DataFrame, month: int = 9) -> tuple[pd.Timestamp, str, pd.DataFrame]:
    """Рабочий день месяца с максимальным |Σy / Σнорма − 1| линии в слотах 16:00–19:30 и станция с наибольшим
    |отклонением| в тот же вечер. Смотрит только на факт и норму."""
    s = slots[(slots.month == month) & (slots.day_type == "рабочий") & slots.hour.isin(DEMO_HOURS)]
    by_day = s.groupby("sday")[["y", "n"]].sum()
    by_day["dev"] = by_day.y / by_day.n - 1
    day = by_day.dev.abs().idxmax()
    st = s[s.sday == day].groupby("station_id")[["y", "n"]].sum()
    st["dev"] = st.y / st.n - 1
    return day, st.dev.abs().idxmax(), by_day.reset_index()


def demo_table(df: pd.DataFrame, day: pd.Timestamp, station: str, sq: dict) -> pd.DataFrame:
    """Прогнозы каждые 30 минут вечером дня day: линия и станция, факт, норма, B1 и стекинг (q50), интервал станции."""
    d = df[df.sday == day]
    loc = d.t.dt.tz_convert(config.SPB_TZ)
    lo, hi = (pd.Timestamp(f"{day.date()} {x}").tz_localize(config.SPB_TZ) for x in DEMO_ORIGINS)
    d = d[(loc >= lo) & (loc <= hi)]
    out = []
    for place, part in (("вся линия", d), (station, d[d.station_id == station])):
        part = part.assign(b1=B.to_y(part, "b1").q50.to_numpy(), stack=B.to_y(part, "stack").q50.to_numpy())
        agg = part.groupby(["t", "k"]).agg(slot=("slot", "first"), y=("y", "sum"), n=("n", "sum"), b1=("b1", "sum"),
                                           stack=("stack", "sum"), cnt=("vestibule_id", "size"),
                                           band4=("band4", "first")).reset_index()
        if place != "вся линия":
            iv = station_interval(agg.rename(columns={"stack": "q50"}), sq)
            agg["stack_q10"], agg["stack_q90"] = iv.q10.to_numpy(), iv.q90.to_numpy()
        agg.insert(0, "place", place)
        out.append(agg)
    res = pd.concat(out, ignore_index=True)
    res["t_local"] = res.t.dt.tz_convert(config.SPB_TZ).dt.strftime("%H:%M")
    res["slot_local"] = res.slot.dt.tz_convert(config.SPB_TZ).dt.strftime("%H:%M")
    return res
