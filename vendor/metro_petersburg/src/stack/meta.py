"""Мета-модель: веса базовых моделей в пространстве z, выбор сложности, конформная поправка, заморозка.

**Вход** — только прогнозы вне выборки за май и июль (FIT_MONTHS). Февраль в стекинг не идёт: часовая LightGBM на нём
обучалась. Сентябрь — проверка после заморозки параметров.

**Веса.** На каждый квантиль и ячейку: w ≥ 0, Σw = 1. Минимум взвешенного pinball Σ n · ρ_τ(z − Σ w_m ẑ_m) на сетке
по симплексу с шагом GRID_STEP (231 точка для трёх моделей); при равенстве — первая точка сетки. После смешивания
квантили сортируются. Для ячейки, которой нет в обучении, — веса горизонта.

**Варианты ячеек:**
- V1 — горизонт;
- V2 — горизонт × период суток;
- V3 — горизонт × минута прогноза (:00 / :30). В :30 прогноз B1 устарел на 30 минут, а на горизонте 120 часовая
  модель работает с h = 3.

**Выбор сложности.** Обучение на мае → проверка на июле и наоборот. Метрика — средний по трём квантилям взвешенный
pinball в z. Берётся вариант, лучший в обоих направлениях; такого нет — V1. Затем выбранный вариант обучается
на мае + июле.

**Интервал.** Асимметричная конформная поправка в z (model.conformal, miss 0,1 на каждую сторону), ячейки «горизонт ×
период суток» (меньше MIN_CELL строк — по горизонту). Поправка — по остаткам того же стекинга на последнем месяце его
обучения: май → июль — по маю, июль → май — по июлю, сентябрь — по июлю (стекинг на мае + июле). Весов не больше 48
на квантиль, строк — десятки тысяч, поэтому оптимизм «внутри выборки» здесь пренебрежим.
"""
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from src import model as M
from src.stack import base as B
from src.stack import data as D

VERSION = "stack_30min_v1"
FIT_MONTHS = (5, 7)
CHECK_MONTH = 9
GRID_STEP = 0.05
VARIANTS = {"V1": ("k",), "V2": ("k", "band4"), "V3": ("k", "minute")}
VARIANT_LABELS = {"V1": "горизонт", "V2": "горизонт × период суток", "V3": "горизонт × минута прогноза"}
DEFAULT_VARIANT = "V1"
MISS = 0.1
MIN_CELL = 300
TAUS = {"10": 0.1, "50": 0.5, "90": 0.9}


def simplex(n_models: int, step: float = GRID_STEP) -> np.ndarray:
    """Точки симплекса с шагом step, по возрастанию лексикографически: [P, n_models]."""
    m = int(round(1 / step))
    pts = [c for c in itertools.product(range(m + 1), repeat=n_models - 1) if sum(c) <= m]
    return np.array([[*c, m - sum(c)] for c in pts], float) / m


def cell_key(df: pd.DataFrame, by: tuple[str, ...]) -> pd.Series:
    key = df[by[0]].astype(str)
    for c in by[1:]:
        key = key + "|" + df[c].astype(str)
    return key


def _pinball(e: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(tau * e, (tau - 1) * e)


def _best(Z: np.ndarray, z: np.ndarray, n: np.ndarray, tau: float, grid: np.ndarray) -> np.ndarray:
    loss = (n[:, None] * _pinball(z[:, None] - Z @ grid.T, tau)).sum(0)
    return grid[int(np.argmin(loss))]


def fit_weights(df: pd.DataFrame, variant: str, names=B.BASES, step: float = GRID_STEP) -> dict:
    """{"variant", "names", "by", "w": {квантиль: {ячейка: [w…]}}, "w_k": {квантиль: {горизонт: [w…]}}}."""
    grid = simplex(len(names), step)
    by = VARIANTS[variant]
    z, n = df.z.to_numpy(float), df.n.to_numpy(float)
    out = {"variant": variant, "names": list(names), "by": list(by), "w": {}, "w_k": {}}
    for q, tau in TAUS.items():
        Z = df[[f"{m}_{q}" for m in names]].to_numpy(float)
        for level, keys in (("w", cell_key(df, by)), ("w_k", df.k.astype(str))):
            out[level][q] = {}
            for key, ix in keys.groupby(keys).groups.items():
                i = df.index.get_indexer(ix)
                out[level][q][key] = [float(x) for x in _best(Z[i], z[i], n[i], tau, grid)]
    return out


def weights_of(df: pd.DataFrame, params: dict, q: str) -> np.ndarray:
    """Матрица весов [N, моделей] строк df: ячейка варианта или, если её нет, горизонт."""
    cell = cell_key(df, tuple(params["by"]))
    wc, wk = params["w"][q], params["w_k"][q]
    return np.array([wc.get(c, wk[str(k)]) for c, k in zip(cell, df.k)], float)


def apply_weights(df: pd.DataFrame, params: dict, out: str = "stack") -> pd.DataFrame:
    """z стекинга по квантилям, отсортированные."""
    zs = []
    for q in B.QS:
        Z = df[[f"{m}_{q}" for m in params["names"]]].to_numpy(float)
        zs.append((Z * weights_of(df, params, q)).sum(1))
    return pd.DataFrame(np.sort(np.column_stack(zs), axis=1), columns=B.cols(out), index=df.index)


def pinball_z(df: pd.DataFrame, name: str) -> float:
    """Средний по квантилям взвешенный pinball в z: Σ n · ρ_τ / Σ n."""
    z, n = df.z.to_numpy(float), df.n.to_numpy(float)
    return float(np.mean([(n * _pinball(z - df[f"{name}_{q}"].to_numpy(float), tau)).sum() / n.sum()
                          for q, tau in TAUS.items()]))


# --- Выбор сложности -------------------------------------------------------------------
def fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Строки обучения стекинга — только май и июль (прогнозы базовых моделей вне выборки)."""
    return df[df.month.isin(FIT_MONTHS)]


def select_variant(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Обучение на одном месяце, pinball на другом — для каждого варианта и направления."""
    fit = fit_rows(df)
    rows = []
    for train_m, test_m in ((5, 7), (7, 5)):
        tr, te = fit[fit.month == train_m], fit[fit.month == test_m]
        for v in VARIANTS:
            p = fit_weights(tr, v)
            rows.append({"train": D.MONTH_RU[train_m], "test": D.MONTH_RU[test_m], "variant": v,
                         "label": VARIANT_LABELS[v], "pinball_z": pinball_z(te.join(apply_weights(te, p)), "stack")})
    res = pd.DataFrame(rows)
    res["best"] = res.pinball_z == res.groupby("test").pinball_z.transform("min")
    winners = res[res.best].groupby("variant").test.nunique()
    both = winners[winners == 2].index.tolist()
    chosen = both[0] if both else DEFAULT_VARIANT
    res["chosen"] = res.variant == chosen
    return chosen, res


# --- Конформная поправка -----------------------------------------------------------------
def fit_conformal(df: pd.DataFrame, name: str = "stack") -> dict:
    """Поправки (lo, hi) в z: по ячейке «k × период» (≥ MIN_CELL строк) и по k."""
    z = df.z.to_numpy(float)
    sc = pd.DataFrame({"lo": df[f"{name}_10"].to_numpy(float) - z, "hi": z - df[f"{name}_90"].to_numpy(float),
                       "k": df.k.to_numpy(), "band4": df.band4.to_numpy()})
    out = {"cell": {}, "k": {}}
    for (k, b), g in sc.groupby(["k", "band4"]):
        if len(g) >= MIN_CELL:
            out["cell"][f"{k}|{b}"] = [M.conformal(g.lo.to_numpy(), MISS), M.conformal(g.hi.to_numpy(), MISS)]
    for k, g in sc.groupby("k"):
        out["k"][str(k)] = [M.conformal(g.lo.to_numpy(), MISS), M.conformal(g.hi.to_numpy(), MISS)]
    return out


def apply_conformal(df: pd.DataFrame, corr: dict, name: str = "stack", out: str = "stack") -> pd.DataFrame:
    key = df.k.astype(str) + "|" + df.band4
    c = np.array([corr["cell"].get(x, corr["k"][str(k)]) for x, k in zip(key, df.k)], float)
    z = np.column_stack([df[f"{name}_10"] - c[:, 0], df[f"{name}_50"], df[f"{name}_90"] + c[:, 1]])
    return pd.DataFrame(np.sort(z, axis=1), columns=B.cols(out), index=df.index)


# --- Перекрёстные прогнозы и итоговые параметры ---------------------------------------------
def cross_predictions(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict]:
    """Стекинг вне выборки на мае (обучен на июле) и июле (обучен на мае): сырые z (stackraw_*) и после поправки
    по месяцу обучения (stack_*). Возвращает и веса обоих направлений."""
    fit = fit_rows(df)
    parts, fits = [], {}
    for train_m, test_m in ((7, 5), (5, 7)):
        tr, te = fit[fit.month == train_m], fit[fit.month == test_m]
        p = fit_weights(tr, variant)
        corr = fit_conformal(tr.join(apply_weights(tr, p)))
        raw = apply_weights(te, p, out="stackraw")
        parts.append(raw.join(apply_conformal(te.join(raw), corr, name="stackraw")))
        fits[D.MONTH_RU[train_m]] = {"weights": p, "conformal": corr}
    return pd.concat(parts).reindex(fit.index), fits


def _sha(params: dict) -> str:
    body = {k: v for k, v in params.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def fit_params(df: pd.DataFrame) -> tuple[dict, dict]:
    """Все параметры — только по строкам мая и июля; остальные месяцы в df ни на что не влияют."""
    fit = fit_rows(df)
    if set(fit.month.unique()) != set(FIT_MONTHS):
        raise ValueError("нужны прогнозы вне выборки за май и июль")
    variant, sel = select_variant(fit)
    w = fit_weights(fit, variant)
    last = fit[fit.month == max(FIT_MONTHS)]
    corr = fit_conformal(last.join(apply_weights(last, w)))
    cross, fits = cross_predictions(fit, variant)
    params = {"version": VERSION, "fit_months": list(FIT_MONTHS), "rows": int(len(fit)), "variant": variant,
              "variant_label": VARIANT_LABELS[variant], "grid_step": GRID_STEP, "weights": w,
              "conformal": {"month": D.MONTH_RU[max(FIT_MONTHS)], **corr},
              "b1_models": {str(m): B.OOS_MODELS[m] for m in (*FIT_MONTHS, CHECK_MONTH)}}
    params["sha256"] = _sha(params)
    return params, {"selection": sel, "cross": cross, "cross_fits": fits}


def predict(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Итоговый стекинг: веса + поправка → stack_*; сырые z — stackraw_*."""
    raw = apply_weights(df, params["weights"], out="stackraw")
    return raw.join(apply_conformal(df.join(raw), params["conformal"], name="stackraw"))


def save_params(params: dict, path) -> None:
    path.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")


def load_params(path) -> dict:
    if not path.exists():
        raise SystemExit(f"нет {path} — сначала python -m src.stack --fit")
    p = json.loads(path.read_text(encoding="utf-8"))
    if _sha(p) != p.get("sha256"):
        raise SystemExit(f"{path.name} изменён после подбора: хэш не совпадает")
    return p
