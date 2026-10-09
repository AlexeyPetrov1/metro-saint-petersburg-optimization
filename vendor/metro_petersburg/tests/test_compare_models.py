import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import compare_models as C
from src import model as M
from test_model import FAST, FEATS, _synthetic   # tests/ без __init__.py: модули импортируются по имени

SMALL = {"max_rounds": 30, "es_rounds": 5}


@pytest.fixture(scope="module")
def data():
    df = _synthetic()
    fold = BT.Fold("тест", pd.Timestamp("2026-05-25"), pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-08"))
    train, test = BT.split(df, fold)
    return df, fold, train, test


def test_generic_scheme_reproduces_frozen_lgbm(data):
    """LightGBM внутри общей схемы даёт ровно то же, что замороженная M.LGBMQuantile."""
    _, _, train, test = data
    ref = M.LGBMQuantile("m", "m", FEATS, params=FAST, max_rounds=60, es_rounds=10, min_cell=50).fit(train)
    raw = C.fit_raw(C.LGBMLearner(FEATS, params=dict(FAST), max_rounds=60, es_rounds=10), train, test)
    p, info = C.calibrate(raw, train, test, min_cell=50)
    pd.testing.assert_frame_equal(p, ref.predict(test))
    assert raw.iters == ref.best_iter
    assert info["calib_cov"] == pytest.approx(ref.info["calib_cov"])


def test_windows_match_frozen(data):
    _, _, train, _ = data
    ref = M.LGBMQuantile("m", "m", FEATS)
    assert C.windows(train.sday) == ref.windows(train.sday)


def test_linear_prep_uses_only_train(data):
    """Подготовка признаков и коэффициенты линейной модели не зависят от тестовых строк."""
    _, _, train, test = data
    noisy = test.assign(r_t=np.random.default_rng(3).lognormal(0, 2, len(test)), y=1e6)
    a, b = C.LinearLearner(FEATS), C.LinearLearner(FEATS)
    ra, rb = C.fit_raw(a, train, test), C.fit_raw(b, train, noisy)
    for q in M.QUANTILES:
        np.testing.assert_array_equal(a.beta[q], b.beta[q])
        np.testing.assert_array_equal(ra.z_cal[q], rb.z_cal[q])
    # после дообучения подготовка подогнана по всему обучению W — и только по нему
    pd.testing.assert_series_equal(a.prep["median"], train[["r_t", "level"]].astype(float).median())
    assert "station_group" not in a.prep["num"]


def test_ensemble_is_mean_of_raw_then_calibrated(data):
    _, _, train, test = data
    ra = C.fit_raw(C.LGBMLearner(FEATS, params=dict(FAST), max_rounds=40, es_rounds=5), train, test)
    rb = C.fit_raw(C.LinearLearner(FEATS), train, test)
    ens = C.average(ra, rb)
    for q in M.QUANTILES:
        np.testing.assert_allclose(ens.z_test[q], (ra.z_test[q] + rb.z_test[q]) / 2)
        np.testing.assert_allclose(ens.z_cal[q], (ra.z_cal[q] + rb.z_cal[q]) / 2)
    assert ens.seconds == pytest.approx(ra.seconds + rb.seconds)
    p, _ = C.calibrate(ens, train, test, min_cell=50)
    assert (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all()


@pytest.mark.parametrize("make", [
    lambda: C.LinearLearner(FEATS),
    lambda: C.CatBoostLearner(FEATS, **SMALL),
    lambda: C.XGBLearner(FEATS, **SMALL),
])
def test_each_learner_gives_sorted_nonnegative_quantiles(data, make):
    _, _, train, test = data
    raw = C.fit_raw(make(), train, test)
    p, info = C.calibrate(raw, train, test, min_cell=50)
    assert len(p) == len(test) and p.notna().all().all()
    assert (p.q10 >= 0).all() and (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all()
    assert 0.7 <= info["calib_cov"] <= 0.9


def test_frozen_model_untouched():
    if not M.FINAL_JSON.exists():
        pytest.skip("нет model_final.json")
    M.check_frozen(M.load_config())
