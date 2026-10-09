"""Норма b для бэктеста и прогнозов (этап 3) — по решениям EDA (docs/eda_spb_findings.md).

Все варианты строятся только по прошлому. В историю идут только обычные сутки метро без флагов: не 1–11 января,
не праздники и не дни событий (`calendar.is_regular`).
- b4 — медиана того же вестибюля, часа и дня недели по 4 последним обычным таким же суткам (поиск до 8 недель).
  Для праздника и нерабочего буднего — по 4 последним обычным воскресеньям: по форме праздник — это воскресенье.
- b8 — то же по 8 последним (поиск до 16 недель, минимум 6 значений); где не определена — b4.
- b4_level — b4 × медиана Σy/Σb вестибюля по 7 последним обычным суткам до суток часа: норма отстаёт от тренда.
"""
import warnings

import numpy as np
import pandas as pd

from src import eda_spb as E

NORMS = ["b4", "b8", "b4_level"]
LEVEL_DAYS, LEVEL_MIN_DAYS = 7, 4
LEVEL_CLIP = (0.5, 2.0)


def median_recent(y: np.ndarray, usable: np.ndarray, base_lag: np.ndarray, weeks: int, min_count: int,
                  max_lookback: int, step: int = 168) -> np.ndarray:
    """Медиана `weeks` самых свежих пригодных значений y[t − base_lag[t] − step·j], j = 0…max_lookback−1.

    base_lag — сдвиг до ближайшего прошлого «такого же» дня у каждой строки (168 ч — тот же день недели неделю назад,
    24·k — ближайшее прошлое воскресенье). Меньше min_count значений → NaN.
    """
    T = y.shape[0]
    hist = np.where(usable, y, np.nan)
    stack = np.full((max_lookback,) + y.shape, np.nan)
    rows = np.arange(T)
    for j in range(max_lookback):
        src = rows - base_lag - step * j
        ok = src >= 0
        stack[j, ok] = hist[src[ok]]
    valid = ~np.isnan(stack)
    stack[valid.cumsum(0) > weeks] = np.nan
    count = np.minimum(valid.sum(0), weeks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice
        med = np.nanmedian(stack, axis=0)
    return np.where(count >= min_count, med, np.nan)


def day_info(g: E.Grid, calendar: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Для каждого часа: обычные ли сутки метро и праздник ли они."""
    c = calendar.set_index("date").reindex(g.sday)
    return c.is_regular.eq(True).to_numpy(), c.day_type.eq("праздник").to_numpy()


def sunday_lag(sday: pd.DatetimeIndex) -> np.ndarray:
    """Часов до того же часа ближайшего прошлого воскресенья (воскресенье → неделя назад)."""
    k = (sday.dayofweek.to_numpy() - 6) % 7
    k[k == 0] = 7
    return 24 * k


def norm(g: E.Grid, calendar: pd.DataFrame, weeks: int = 4, min_count: int = 3, max_lookback: int = 8) -> np.ndarray:
    """Норма по дню недели; для праздников и нерабочих будней — по воскресеньям."""
    regular, holiday = day_info(g, calendar)
    usable = ~g.flagged & regular[:, None]
    T = g.y.shape[0]
    b = median_recent(g.y, usable, np.full(T, 168), weeks, min_count, max_lookback)
    b_sun = median_recent(g.y, usable, sunday_lag(g.sday), weeks, min_count, max_lookback)
    return np.where(holiday[:, None], b_sun, b)


def level_factor(g: E.Grid, calendar: pd.DataFrame, b: np.ndarray, n_days: int = LEVEL_DAYS,
                 min_days: int = LEVEL_MIN_DAYS) -> np.ndarray:
    """Медиана суточного Σy/Σb вестибюля по `n_days` последним обычным суткам строго до суток часа; нет данных → 1."""
    regular, _ = day_info(g, calendar)
    work = np.isin(g.hour, E.WORK_HOURS)[:, None]
    m = work & regular[:, None] & ~np.isnan(g.y) & (b >= E.B_MIN)
    key = g.sday.to_numpy()
    ys = pd.DataFrame(np.where(m, g.y, 0)).groupby(key).sum()
    bs = pd.DataFrame(np.where(m, b, 0)).groupby(key).sum()
    daily = (ys / bs.where(bs >= E.B_MIN))
    daily = daily[daily.index.isin(g.sday[regular])]          # только обычные сутки
    med = daily.rolling(n_days, min_periods=min_days).median()  # по обычным суткам до и включая эти
    days = pd.date_range(g.sday.min(), g.sday.max(), freq="D")
    known = med.reindex(days).ffill().shift(1)                  # к суткам D — только сутки до D
    f = known.reindex(g.sday).to_numpy()
    return np.clip(np.nan_to_num(f, nan=1.0), *LEVEL_CLIP)


def norm_matrices(g: E.Grid, calendar: pd.DataFrame) -> dict[str, np.ndarray]:
    b4 = norm(g, calendar)
    b8 = norm(g, calendar, weeks=8, min_count=6, max_lookback=16)
    return {"b4": b4, "b8": np.where(np.isnan(b8), b4, b8), "b4_level": b4 * level_factor(g, calendar, b4)}


def observed_ratio(g: E.Grid, b: np.ndarray) -> np.ndarray:
    """Вход модели r(t) = поток / норма по наблюдённому потоку: час инцидента тоже входит — это реальные данные
    на момент прогноза. Исключены только закрытые часы (метро или вестибюль) и часы вне 06–00."""
    return E.ratio(np.where(g.closed, np.nan, g.entries), b, g.hour)
