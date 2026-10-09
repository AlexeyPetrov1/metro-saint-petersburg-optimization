"""Базовые модели стекинга: B1 (часовая LightGBM), B2 (персистентность), простое среднее. Выход — q10, q50, q90
в пространстве z = log((y + 1)/(n_slot + 1)), колонки `<имя>_10`, `<имя>_50`, `<имя>_90`.

**B1 — замороженная часовая LightGBM, только вне выборки и только чтение.**
- Май — models/lgbm_fold_2026-05, июль — lgbm_fold_2026-07, сентябрь — lgbm_holdout. Модель месяца обучена
  не позже чем за 7 суток до его начала, иначе ошибка.
- Прогноз часа H с последнего закрытого часа t0: h = H − t0 = 1, 2 или 3. h = 3 нужен в hh:30 для слота
  [hh+2:00, hh+2:30): прогноз, сделанный в hh:00, покрывает только часы hh и hh+1. Модель учили на h = 1, 2: деревья
  относят h = 3 к ветке h = 2, поправка интервала — общая (corr_all). Признаки строки h = 3 — те же функции
  замороженного features.py, данные — до конца t0.
- Перевод в слот: ŷ_slot = ŷ_H · s_v · p_half, z = log((ŷ_slot + 1)/(n_slot + 1)); квантили часа — уже откалиброванные
  замороженной моделью.

**B2 — персистентность.** r̂ = Σy / Σнорма за две последние четверти [t − 30, t) по валидным четвертям (нет ни одной —
r̂ = 1), clip как у бейзлайнов (BT.R_CLIP). q50 = log((n_slot · r̂ + 1)/(n_slot + 1)); q10 и q90 — q50 + квантили
остатка z − q50 на месяцах обучения по ячейке «горизонт × период суток» (меньше MIN_CELL строк — по горизонту).

**Простое среднее** — среднее z базовых моделей по каждому квантилю, затем сортировка.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from src import backtest as BT
from src import eda_spb as E
from src import features as F
from src import model as M
from src import nowcast as N
from src.stack import data as D

QUANTILES = M.QUANTILES
QS = ("10", "50", "90")
OOS_MODELS = N.OOS_MODELS                    # {5: lgbm_fold_2026-05, 7: lgbm_fold_2026-07, 9: lgbm_holdout}
H_HOURLY = (1, 2, 3)
PERSIST_QUARTERS = 2
MIN_CELL = 300
BASES = ("b1", "b2", "b3")
LABELS = {"b1": "B1: часовая LightGBM", "b2": "B2: персистентность", "b3": "B3: GRU",
          "mean": "Среднее B1–B3", "stack": "Стекинг"}


def cols(name: str) -> list[str]:
    return [f"{name}_{q}" for q in QS]


# --- B1 -------------------------------------------------------------------------------
def hourly_panel(data: E.SpbData | None = None) -> BT.Panel:
    """Панель замороженного пайплайна с порогами из model_final.json; final=True — сентябрь виден как прошлое."""
    cfg = M.load_config()
    d = E.load() if data is None else data
    return replace(BT.build_panel(d, final=True), thresholds=pd.DataFrame(cfg["thresholds"]))


def hourly_rows(panel: BT.Panel, months, for_export: bool = False) -> pd.DataFrame:
    """Строки часовой модели с признаками для часов τ в месяцах months, h = 1, 2, 3."""
    rows = BT.make_rows(panel, horizons=H_HOURLY, for_export=for_export)
    rows = rows[rows.sday.dt.month.isin(months)]
    return F.add_features(rows, panel)


def check_oos(meta: dict, month: int) -> None:
    start = pd.Timestamp(2026, month, 1)
    if pd.Timestamp(meta["train_end"]) + pd.Timedelta(days=BT.GAP_DAYS) >= start:
        raise ValueError(f"{meta['name']}: обучена по {meta['train_end']} — для месяца {month} не вне выборки")


def b1_hourly(rows: pd.DataFrame, month: int, models_dir=M.MODELS) -> pd.DataFrame:
    """q10/q50/q90 часа моделью месяца вне выборки: vestibule_id, t (начало t0), tau, h."""
    m, _, meta = M.load_bundle(models_dir / OOS_MODELS[month])
    check_oos(meta, month)
    r = rows[rows.sday.dt.month == month]
    p = m.predict(r)
    return pd.DataFrame({"vestibule_id": r.vestibule_id.to_numpy(), "t0": r.t.to_numpy(), "hour_ts": r.tau.to_numpy(),
                         "h_hour": r.h.to_numpy(), "m10": p.q10.to_numpy(), "m50": p.q50.to_numpy(),
                         "m90": p.q90.to_numpy(), "b1_model": OOS_MODELS[month]})


def b1_slot(rows: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """B1 в z слота для строк rows (вестибюль, t, k). Связь — по (вестибюль, час слота, h); t0 строки обязан совпасть."""
    key = ["vestibule_id", "hour_ts", "h_hour"]
    m = rows[[*key, "t0", "s_v", "p_half", "n"]].merge(hourly, on=key, how="left", validate="many_to_one",
                                                      suffixes=("", "_m"))
    if m.m50.isna().any():
        raise ValueError(f"нет прогноза часовой модели у {int(m.m50.isna().sum())} строк")
    if not (m.t0 == m.t0_m).all():
        raise RuntimeError("t0 строки и прогноза часовой модели не совпадают")
    out = pd.DataFrame(index=rows.index)
    scale = (m.s_v * m.p_half).to_numpy(float)
    n = m.n.to_numpy(float)
    for q in QS:
        out[f"b1_{q}"] = np.log((m[f"m{q}"].to_numpy(float) * scale + 1) / (n + 1))
    out["b1_model"] = m.b1_model.to_numpy()
    return out


# --- B2 -------------------------------------------------------------------------------
def persistence_ratio(q: D.Quarters, rows: pd.DataFrame) -> np.ndarray:
    """r̂ = Σy / Σнорма по валидным четвертям из PERSIST_QUARTERS последних перед t; нет — 1; clip R_CLIP."""
    key = rows[["t", "vestibule_id"]].drop_duplicates()
    i = q.index_of(key.t)
    if (i < PERSIST_QUARTERS).any():
        raise ValueError("момент t без истории четвертей")
    vi = pd.Index(q.vestibules).get_indexer(key.vestibule_id)
    idx = i[:, None] + np.arange(-PERSIST_QUARTERS, 0)[None, :]
    y, n = q.y[idx, vi[:, None]], q.norm[idx, vi[:, None]]
    ok = ~np.isnan(n)
    sy, sn = np.where(ok, y, 0).sum(1), np.where(ok, n, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(sn > 0, sy / sn, 1.0)
    r = pd.Series(np.clip(r, *BT.R_CLIP), index=pd.MultiIndex.from_frame(key))
    return r.reindex(pd.MultiIndex.from_frame(rows[["t", "vestibule_id"]])).to_numpy()


def b2_point(q: D.Quarters, rows: pd.DataFrame) -> np.ndarray:
    n = rows.n.to_numpy(float)
    return np.log((n * persistence_ratio(q, rows) + 1) / (n + 1))


def _cells(rows: pd.DataFrame) -> list[pd.Series]:
    return [rows.k, rows.band4]


def fit_residual_quantiles(rows: pd.DataFrame, point: np.ndarray) -> dict:
    """Квантили 0,1 и 0,9 остатка z − q50 по ячейке «k × период» (≥ MIN_CELL строк) и по k."""
    res = pd.Series(rows.z.to_numpy(float) - point, index=rows.index)
    by_cell = res.groupby(_cells(rows))
    size = by_cell.size()
    cell = by_cell.quantile([0.1, 0.9]).unstack()[size >= MIN_CELL]
    by_k = res.groupby(rows.k).quantile([0.1, 0.9]).unstack()
    return {"cell": {f"{k}|{b}": [float(v[0.1]), float(v[0.9])] for (k, b), v in cell.iterrows()},
            "k": {str(k): [float(v[0.1]), float(v[0.9])] for k, v in by_k.iterrows()}}


def apply_residual_quantiles(rows: pd.DataFrame, point: np.ndarray, rq: dict) -> np.ndarray:
    keys = rows.k.astype(str) + "|" + rows.band4
    lo = [rq["cell"].get(c, rq["k"][str(k)])[0] for c, k in zip(keys, rows.k)]
    hi = [rq["cell"].get(c, rq["k"][str(k)])[1] for c, k in zip(keys, rows.k)]
    return np.sort(np.column_stack([point + np.asarray(lo), point, point + np.asarray(hi)]), axis=1)


def b2(q: D.Quarters, train_rows: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """B2 для rows; квантили остатка — по train_rows (месяцы обучения)."""
    rq = fit_residual_quantiles(train_rows, b2_point(q, train_rows))
    z = apply_residual_quantiles(rows, b2_point(q, rows), rq)
    return pd.DataFrame(z, columns=cols("b2"), index=rows.index), rq


# --- Среднее --------------------------------------------------------------------------
def mean_of(df: pd.DataFrame, names=BASES, out: str = "mean") -> pd.DataFrame:
    z = np.column_stack([df[[f"{n}_{q}" for n in names]].mean(1) for q in QS])
    return pd.DataFrame(np.sort(z, axis=1), columns=cols(out), index=df.index)


def to_y(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Квантили модели name в потоке слота: (n + 1)·e^z − 1, снизу 0."""
    n = df.n.to_numpy(float)
    return pd.DataFrame({f"q{q}": M.from_z(df[f"{name}_{q}"], n, "ratio") for q in QS}, index=df.index)
