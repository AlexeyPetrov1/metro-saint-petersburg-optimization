import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import config
from src import eda_spb as E


# --- Норма: синтетика ---------------------------------------------------------
def test_baseline_is_median_of_last_four_usable_weeks():
    T = 168 * 10
    y = np.zeros((T, 1))
    for w in range(10):
        y[w * 168:(w + 1) * 168] = 100 + w          # неделя w: 100 + w
    usable = np.ones((T, 1), bool)
    usable[8 * 168:9 * 168] = False                 # неделя 8 — необычная (праздник), её пропускаем
    b = E.baseline_matrix(y, usable)
    # неделя 9: 4 последних пригодных недели — 7, 6, 5, 4 (восьмая пропущена) → медиана 105,5
    assert b[9 * 168 + 10, 0] == pytest.approx(105.5)
    # неделя 5: недели 4, 3, 2, 1
    assert b[5 * 168, 0] == pytest.approx(102.5)
    # в первые 3 недели меньше 3 значений → NaN
    assert np.isnan(b[2 * 168, 0]) and not np.isnan(b[3 * 168, 0])


def test_baseline_uses_only_past():
    rng = np.random.default_rng(0)
    T = 168 * 9
    y = rng.uniform(50, 150, (T, 3))
    usable = np.ones_like(y, bool)
    t0 = 168 * 7 + 30
    y2 = y.copy()
    y2[t0 + 1:] = rng.uniform(0, 1000, y2[t0 + 1:].shape)   # будущее другое
    np.testing.assert_array_equal(E.baseline_matrix(y, usable)[:t0 + 1], E.baseline_matrix(y2, usable)[:t0 + 1])


# --- Статистика -----------------------------------------------------------------
def test_bh_matches_hand_example():
    adj = E.bh([0.01, 0.04, 0.03, 0.005, np.nan])
    np.testing.assert_allclose(adj[:4], [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(adj[4])


def _days(D=60, S=20, seed=0):
    rng = np.random.default_rng(seed)
    months = np.repeat([1, 2, 3], D // 3)
    perms = E.strata_perms(months, 500, rng)
    counts = E.boot_counts(D, 300, rng)
    return rng, months, perms, counts


def test_perm_corr_null_and_signal():
    rng, months, perms, counts = _days()
    X = rng.normal(size=(60, 20))
    noise = E.rank_corr_test(X, rng.normal(size=(60, 20)), perms, counts, months)
    signal = E.rank_corr_test(X, X + rng.normal(size=(60, 20)), perms, counts, months)
    assert noise["p"] > 0.05
    assert signal["p"] < 0.01 and signal["lo"] > 0.5


def test_gram_permutation_equals_direct():
    """Статистика через матрицы сумм по парам дней совпадает с прямым расчётом на переставленных днях."""
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(30, 8)), rng.normal(size=(30, 8))
    X[rng.random(X.shape) < 0.2] = np.nan
    Y[rng.random(Y.shape) < 0.2] = np.nan
    pi = rng.permutation(30)
    G = E._gram(X, Y)
    fast = E._pearson(**{k: v[np.arange(30), pi].sum() for k, v in G.items()})
    yp = Y[pi]
    m = ~np.isnan(X) & ~np.isnan(yp)
    assert fast == pytest.approx(np.corrcoef(X[m], yp[m])[0, 1])


def test_median_diff_null_and_signal():
    rng, months, perms, counts = _days()
    X = np.repeat((rng.random(60) < 0.3)[:, None], 20, 1).astype(float)   # признак уровня дня
    Y0 = rng.normal(size=(60, 20))
    assert E.median_diff_test(X, Y0, perms, counts, months)["p"] > 0.05
    res = E.median_diff_test(X, Y0 - 0.5 * X, perms, counts, months)
    assert res["p"] < 0.01 and res["hi"] < 0


def test_weighted_median_with_unit_weights_is_median():
    rng = np.random.default_rng(2)
    vals, days = rng.normal(size=101), rng.integers(0, 10, 101)
    out = E.weighted_median_boot(vals, days, np.ones((3, 10)))
    assert np.allclose(out, np.median(vals))


def test_dm_test():
    loss = np.abs(np.random.default_rng(3).normal(size=100))
    assert E.dm_test(loss, loss) == (0.0, 1.0)
    assert E.dm_test(loss + 1, loss)[1] < 0.01


# --- Реальные данные: признаки только из прошлого -----------------------------
@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists()):
        pytest.skip("нет interim СПб — запустите python -m src.clean_spb")
    return E.load()


@pytest.mark.parametrize("h", [1, 2])
def test_candidates_do_not_look_ahead(spb, h):
    """Признаки строки t не меняются, если удалить весь поток после конца часа t (раздел 5 ТЗ)."""
    g = spb.grid
    t0 = int(np.flatnonzero((g.sday == pd.Timestamp("2026-06-17")) & (g.hour == 9))[0])
    entries = g.entries.copy()
    entries[t0 + 1:] = np.nan
    regular = spb.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    cut = dataclasses.replace(spb, grid=E.make_grid(g.ts, entries, g.closed, g.flagged, regular))
    full, trunc = E.candidate_matrices(spb, h), E.candidate_matrices(cut, h)
    for name, (_, a) in full.items():
        np.testing.assert_allclose(a[:t0 + 1], trunc[name][1][:t0 + 1], equal_nan=True, err_msg=name)


def test_real_baseline_covers_analysis_period(spb):
    g = spb.grid
    work = np.isin(g.hour, E.WORK_HOURS) & (g.sday >= E.ANALYSIS_START)
    assert np.isnan(g.b[work]).mean() < 0.02
    assert ((g.r[~np.isnan(g.r)] > 0) & (g.b[~np.isnan(g.r)] >= E.B_MIN)).all()
