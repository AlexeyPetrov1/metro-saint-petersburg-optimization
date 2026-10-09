import json

import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import config
from src import intrahour as I
from src import model as M
from src import nowcast as N


# --- Смесь ---------------------------------------------------------------------------
def test_blend_ends():
    model, fact = np.array([100.0, 2500.0, 0.0]), np.array([150.0, 1800.0, 40.0])
    assert np.allclose(N.blend(model, fact, 0.0), model)
    assert np.allclose(N.blend(model, fact, 1.0), fact)
    mid = N.blend(model, fact, 0.5)
    assert np.allclose(mid, np.sqrt((model + 1) * (fact + 1)) - 1)       # среднее в логарифмах
    assert (np.minimum(model, fact) <= mid + 1e-9).all() and (mid <= np.maximum(model, fact) + 1e-9).all()


def test_nowcast_k0_is_model_and_band_weights():
    s = pd.DataFrame({"m50": [100.0, 200.0], "fact1": [150.0, 100.0], "band4": ["утро 06–09", "день 10–15"]})
    assert np.allclose(N.nowcast(s, {}, 0), s.m50)
    w = {"1": {"утро 06–09": 1.0, "день 10–15": 0.0}}
    assert np.allclose(N.nowcast(s, w, 1), [150.0, 200.0])
    assert np.allclose(N.nowcast(s, {"1": {N.ALL: 1.0}}, 1), s.fact1)


def test_interval_ordered():
    q = {"1": {"утро 06–09": [0.9, 1.1]}}
    lo, hi = N.interval(np.array([100.0, 0.0]), 1, pd.Series(["утро 06–09"] * 2), q)
    assert (lo <= hi).all() and lo[0] == pytest.approx(90) and hi[0] == pytest.approx(110)


def test_params_hash(tmp_path):
    p = {"version": N.VERSION, "alarm": {"N": 100}}
    p["sha256"] = N._sha(p)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(p), encoding="utf-8")
    assert N.load_params(path)["alarm"]["N"] == 100
    p["alarm"]["N"] = 50
    path.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(SystemExit, match="хэш"):
        N.load_params(path)


# --- Реальные данные и модели ---------------------------------------------------------------
@pytest.fixture(scope="module")
def fit_table():
    if not config.SPB_15MIN.exists() or not (M.MODELS / N.OOS_MODELS[5]).exists():
        pytest.skip("нет 15-минутных данных или моделей фолдов — clean_spb_15min и model --save-folds")
    return N.table(months=N.FIT_MONTHS)


def test_models_out_of_sample():
    for month, name in N.OOS_MODELS.items():
        path = M.MODELS / name / "meta.json"
        if not path.exists():
            pytest.skip(f"нет {name}")
        end = pd.Timestamp(json.loads(path.read_text(encoding="utf-8"))["train_end"])
        assert end + pd.Timedelta(days=BT.GAP_DAYS) < pd.Timestamp(f"2026-{month:02d}-01")
    assert 2 not in N.OOS_MODELS                     # февраль — в обучении всех фолдов


def test_fold_models_match_backtest_cache(fit_table):
    cfg = M.load_config()
    found = sorted(M.CACHE.glob(f"{cfg['step_model']}_*.parquet"))
    if len(found) != 1:
        pytest.skip("нет кэша бэктеста")
    cache = pd.read_parquet(found[0], columns=["vestibule_id", "h", "tau", "q50"])
    cache = cache[cache.h == 1].rename(columns={"tau": "ts_utc"})
    m = fit_table.merge(cache, on=["vestibule_id", "ts_utc"])
    assert len(m) == len(fit_table)
    assert np.abs(m.m50 - m.q50).max() < 1e-4


def test_thresholds_same_as_stage6():
    cfg = pd.DataFrame(M.load_config()["thresholds"]).set_index(["level", "band"]).p95
    csv = pd.read_csv(I.THRESHOLDS_CSV).set_index(["level", "band"]).p95
    assert np.allclose(cfg.reindex(csv.index), csv, atol=1e-5)


def test_fit_uses_only_may_and_july(fit_table):
    """Строки февраля и сентября (здесь — шум) ни на что в подборе не влияют."""
    p0, _ = N.fit_params(fit_table)
    rng = np.random.default_rng(0)
    noise = []
    for month in (2, 9):
        x = fit_table[fit_table.month == 5].copy()
        x["month"] = month
        x["sday"] = x.sday + pd.Timedelta(days=200 if month == 9 else -90)
        x["ts_utc"] = x.ts_utc + pd.Timedelta(days=200 if month == 9 else -90)
        for col in ("y", "m50", "fact1", "fact2", "fact3"):
            x[col] = rng.uniform(0, 5000, len(x))
        x["anom"] = rng.random(len(x)) < 0.5
        noise.append(x)
    p1, _ = N.fit_params(pd.concat([fit_table, *noise], ignore_index=True))
    assert p0 == p1
    assert set(N.fit_rows(fit_table).month) == {5, 7} and p0["fit_months"] == [5, 7]


def test_alarm_table_without_filter_is_stage6_detector(fit_table):
    c = N.stage6_c()
    w = {str(k): {N.ALL: 0.5} for k in N.KS}
    a = N.alarm_table(fit_table, w, c, None)
    seq = I.sequential(fit_table, c)
    assert (a.first_k.to_numpy() == seq.first_k.to_numpy()).all()
    assert np.array_equal(a.detected_min.to_numpy(), seq.detected_min.to_numpy(), equal_nan=True)


def test_filter_only_removes_alarms_and_choice_is_optimal(fit_table):
    p, extra = N.fit_params(fit_table)
    grid = extra["n_grid"]
    assert grid.alarms_per_day.is_monotonic_decreasing and grid.recall_big.is_monotonic_decreasing
    n = p["alarm"]["N"]
    chosen = grid[grid.N == n].iloc[0]
    assert chosen.recall_big >= N.RECALL_BIG_MIN
    feasible = grid[grid.recall_big >= N.RECALL_BIG_MIN]
    assert chosen.false_per_day == feasible.false_per_day.min()
