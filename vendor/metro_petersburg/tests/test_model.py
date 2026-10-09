import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import contract
from src import model as M

FEATS = ["vestibule", "hour_tau", "r_t", "level"]
FAST = {**M.PARAMS, "num_threads": 2, "min_data_in_leaf": 20}


def _synthetic(days: int = 120, seed: int = 0) -> pd.DataFrame:
    """2 вестибюля × 18 часов × 2 горизонта; y = f · e^(0,4·(r_t − 1) + шум)."""
    rng = np.random.default_rng(seed)
    sday = pd.date_range("2026-02-09", periods=days, freq="D")
    idx = pd.MultiIndex.from_product([sday, ["a", "b"], range(6, 24), [1, 2]], names=["sday", "vestibule_id", "hour", "h"])
    df = idx.to_frame(index=False)
    n = len(df)
    df["b4_level"] = np.where(df.vestibule_id == "a", 800.0, 200.0) * (1 + 0.5 * np.sin(df.hour / 4))
    df["r_t"] = rng.lognormal(0, 0.1, n)
    df["level"] = 1.0
    df["y"] = np.round(df.b4_level * np.exp(0.4 * np.log(df.r_t) + rng.normal(0, 0.08, n)))
    df["vestibule"] = pd.Categorical(df.vestibule_id, categories=["a", "b"])
    df["hour_tau"] = df.hour.astype(float)
    df["band"] = df.hour.map(BT.BAND_OF)
    df["tau"] = (df.sday + pd.to_timedelta(df.hour, unit="h")).dt.tz_localize("UTC")
    df["station_id"], df["group"] = df.vestibule_id, "g"
    df["is_special"] = df["is_holiday"] = False
    df["r_level_t"] = df.r_t                                      # вход бейзлайна «b × уровень × r(t)»
    df["ewm_r_line_2h"] = df.groupby("sday").r_t.transform("mean")
    df["is_anomaly"] = (df.y / df.b4_level - 1).abs() > 0.15
    return df


def _model() -> M.LGBMQuantile:
    return M.LGBMQuantile("m", "m", FEATS, params=FAST, max_rounds=60, es_rounds=10, min_cell=50)


@pytest.fixture(scope="module")
def fitted():
    df = _synthetic()
    fold = BT.Fold("тест", pd.Timestamp("2026-05-25"), pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-08"))
    train, test = BT.split(df, fold)
    return df, fold, train, test, _model().fit(train)


def test_target_roundtrip():
    y, f = np.array([0.0, 5, 120, 9000]), np.array([0.0, 10, 100, 8000])
    for target in ("ratio", "log1p"):
        np.testing.assert_allclose(M.from_z(M.to_z(y, f, target), f, target), y, atol=1e-9)


def test_conformal_quantile():
    s = np.random.default_rng(1).normal(size=5000)
    c = M.conformal(s, 0.1)
    assert (s > c).mean() <= 0.1 and c == pytest.approx(np.quantile(s, 0.9), abs=0.05)


def test_windows_inside_train(fitted):
    _, fold, train, _, m = fitted
    last = train.sday.max()
    assert last < fold.test_start - pd.Timedelta(days=BT.GAP_DAYS - 1)
    assert pd.Timestamp(m.info["calib_start"]) == last - pd.Timedelta(days=27)
    assert pd.Timestamp(m.info["es_start"]) == last - pd.Timedelta(days=13)
    cal = train.loc[m.calib_pred.index]
    assert cal.sday.min() == pd.Timestamp(m.info["calib_start"]) and cal.sday.max() == last
    assert set(m.calib_pred.index) <= set(train.index)
    assert (m.calib_pred.q10 <= m.calib_pred.q50).all() and (m.calib_pred.q50 <= m.calib_pred.q90).all()


def test_calibration_does_not_use_test(fitted):
    """Окна, поправки и прогнозы не меняются, если в тестовом месяце y заменить шумом или удалить."""
    df, fold, *_ = fitted
    is_test = (df.sday >= fold.test_start) & (df.sday <= fold.test_end)
    noise = np.random.default_rng(5).integers(0, 50_000, len(df))
    seen = []
    keep = lambda f, mm: seen.append(mm)
    base = BT.run(df, [_model()], [fold], on_fit=keep)
    alt = BT.run(df.assign(y=np.where(is_test, noise, df.y)), [_model()], [fold], on_fit=keep)
    BT.run(df.assign(y=np.where(is_test, np.nan, df.y)), [_model()], [fold], on_fit=keep)
    cols = ["q10", "q50", "q90", "q10_raw", "q90_raw"]
    pd.testing.assert_frame_equal(base[cols], alt[cols])
    for other in seen[1:]:
        assert other.info == seen[0].info
        pd.testing.assert_frame_equal(other.corr, seen[0].corr)


def test_coverage_after_correction(fitted):
    m = fitted[4]
    assert 0.78 <= m.info["calib_cov"] <= 0.86
    _, _, _, test, m = fitted
    p = m.predict(test)
    cov = ((test.y >= p.q10) & (test.y <= p.q90)).mean()
    assert 0.65 <= cov <= 0.92


def test_quantiles_sorted_and_small_norm(fitted):
    _, _, _, test, m = fitted
    p = m.predict(test)
    assert (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all() and (p.q10 >= 0).all()
    assert (p.q10_raw <= p.q50).all() and (p.q90_raw >= p.q50).all()
    tiny = test.head(5).assign(b4_level=0.4)
    pt = m.predict(tiny)
    assert np.allclose(pt[["q10", "q50", "q90"]].to_numpy(), 0.4)


def test_explain_contributions_add_up(fitted):
    _, _, _, test, m = fitted
    rows = test.head(50)
    c = M.contributions(m, rows)
    np.testing.assert_allclose(c.sum(axis=1), m.raw(rows)[0.5], rtol=1e-6, atol=1e-6)
    reasons = M.explain(m, rows, names={"a": "А", "b": "Б"})
    assert len(reasons) == 50 and all(len(r) == 3 for r in reasons)
    assert all(isinstance(x["text"], str) and x["text"] and x["feature"] in FEATS for r in reasons for x in r)
    top = c[FEATS].abs().to_numpy().argmax(axis=1)
    assert [r[0]["feature"] for r in reasons] == [FEATS[i] for i in top]


def test_deepcopy_is_unfitted():
    m = copy.deepcopy(_model())
    assert not m.boosters and m.corr is None


def test_model_export_passes_contract():
    if not M.AUG_OUT.exists():
        pytest.skip("нет model_aug.json — запустите python -m src.model --export-august")
    preds = contract.validate_file(M.AUG_OUT)
    assert len(preds) == 18 * 31 * 20 * 2
    assert {p.model_version for p in preds} == {M.MODEL_VERSION}
    expl = json.loads(M.EXPLAIN_OUT.read_text(encoding="utf-8"))
    assert len(expl) == len(preds) and all(len(e["reasons"]) == 3 for e in expl)


# --- Смесь, заморозка, флаг, бандл (этап 5) ----------------------------------------
def _blend(**kw) -> M.BlendModel:
    return M.BlendModel("blend", "blend", _model(), **kw)


def test_blend_without_candidates_is_the_model(fitted):
    _, _, train, test, _ = fitted
    b = _blend(grid_a=()).fit(train)
    assert b.a == 0 and b.w_max == 0
    pd.testing.assert_frame_equal(b.predict(test), b.model.predict(test))


def test_blend_weights_bounded(fitted):
    _, _, train, test, _ = fitted
    b = _blend().fit(train)
    b.a, b.w_max = 5.0, 0.4
    w = b.weights(test)
    assert (w >= 0).all() and (w <= 0.4).all() and (w > 0).any()
    p = b.predict(test)
    assert (p.q10 <= p.q50).all() and (p.q50 <= p.q90).all()


def test_blend_tuning_does_not_use_test(fitted):
    df, fold, *_ = fitted
    is_test = (df.sday >= fold.test_start) & (df.sday <= fold.test_end)
    seen = []
    keep = lambda f, mm: seen.append(mm)
    base = BT.run(df, [_blend()], [fold], on_fit=keep)
    noisy = df.assign(y=np.where(is_test, np.random.default_rng(7).integers(0, 50_000, len(df)), df.y),
                      is_anomaly=np.where(is_test, ~df.is_anomaly, df.is_anomaly))
    alt = BT.run(noisy, [_blend()], [fold], on_fit=keep)
    pd.testing.assert_frame_equal(base[["q10", "q50", "q90"]], alt[["q10", "q50", "q90"]])
    assert (seen[0].a, seen[0].w_max) == (seen[1].a, seen[1].w_max)


def test_check_frozen_detects_changes():
    cfg = {"features": FEATS, "target": "ratio", "anomaly_rule": {"kind": "interval"}, "blend": {"accepted": False}}
    cfg = {**cfg, "config_sha256": M.config_hash(cfg), "code_sha256": M.code_hashes(), "frozen": True}
    M.check_frozen(cfg)
    with pytest.raises(SystemExit):
        M.check_frozen({**cfg, "features": FEATS[:-1]})                 # конфигурацию поправили после заморозки
    with pytest.raises(SystemExit):
        M.check_frozen({**cfg, "code_sha256": {**cfg["code_sha256"], "src/model.py": "0" * 64}})   # код изменился
    with pytest.raises(SystemExit):
        M.check_frozen({k: v for k, v in cfg.items() if k != "frozen"})


def test_final_config_hash_matches():
    if not M.FINAL_JSON.exists():
        pytest.skip("нет model_final.json")
    cfg = json.loads(M.FINAL_JSON.read_text(encoding="utf-8"))
    if not cfg.get("frozen"):
        pytest.skip("конфигурация ещё не заморожена")
    assert M.config_hash(cfg) == cfg["config_sha256"]


def test_flag_metrics_toy():
    r = M._prf(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == pytest.approx(0.5)
    t = pd.DataFrame({"model": "модель", "level": "вестибюли", "h": "оба", "rule": list(M.FLAG_RULES),
                      "precision": [0.2, 0.3, 0.5, 0.4], "recall": [0.9, 0.7, 0.3, 0.65], "f1": [0.3, 0.4, 0.45, 0.5]})
    assert M.choose_flag_rule(t, "модель")[0] == "норма вне [q10; q90]"   # recall ≥ 0,6 и лучшая precision
    low = t.assign(recall=0.5)
    assert M.choose_flag_rule(low, "модель")[0] == "норма вне [q10; q90]"  # иначе — лучший F1


def test_bundle_roundtrip(fitted, tmp_path):
    _, _, train, test, m = fitted
    st_q = BT.RatioQuantiles("station", "s", lambda df: df.f).fit(train.assign(f=train.b4_level))
    cfg = {"anomaly_rule": {"kind": "threshold", "level": "p90"}, "thresholds": None, "config_sha256": "x"}
    M.save_bundle(tmp_path / "b", m, st_q, cfg, train, "b")
    m2, st_q2, meta = M.load_bundle(tmp_path / "b")
    pd.testing.assert_frame_equal(m.predict(test), m2.predict(test))
    st = test.assign(f=test.b4_level)
    pd.testing.assert_frame_equal(st_q.predict(st), st_q2.predict(st), check_names=False)
    assert meta["train_end"] == str(train.sday.max().date()) and meta["anomaly_rule"]["level"] == "p90"
