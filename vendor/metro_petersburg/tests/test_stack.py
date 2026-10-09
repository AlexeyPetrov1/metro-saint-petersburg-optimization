import json

import numpy as np
import pandas as pd
import pytest

from src import backtest as BT
from src import config
from src import eda_spb as E
from src import features as F
from src import intrahour as I
from src import model as M
from src import serve as S
from src.stack import __main__ as R
from src.stack import base as B
from src.stack import data as D
from src.stack import evaluate as V
from src.stack import meta as MT
from src.stack import nn as NN

TINY = {**NN.PARAMS, "max_epochs": 2, "patience": 1, "seeds": [0, 1], "es_days": 2}
ORIGIN = pd.Timestamp("2026-07-08 09:30", tz=config.SPB_TZ).tz_convert("UTC")


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=config.SPB_TZ).tz_convert("UTC")


# --- Время: что известно в момент t --------------------------------------------------------
def test_last_closed_hour_and_hourly_horizon():
    """В hh:00 и hh:30 последний закрытый час — hh − 1; горизонт часовой модели по слотам k = 1…4:
    в :00 — 1, 1, 2, 2; в :30 — 1, 2, 2, 3 (слот [hh+2:00, hh+2:30) — h = 3)."""
    for t, hs in ((_utc("2026-07-08 09:00"), [1, 1, 2, 2]), (_utc("2026-07-08 09:30"), [1, 2, 2, 3])):
        tt = pd.Series([t] * D.K)
        assert (D.last_closed_hour(tt) == _utc("2026-07-08 08:00")).all()
        slots = pd.Series([t + (k - 1) * D.SLOT for k in range(1, D.K + 1)])
        assert D.hourly_h(D.floor_hour(slots), tt).tolist() == hs
        assert (D.last_closed_hour(tt) < D.floor_hour(tt)).all()      # час, в котором стоит t, не используется


@pytest.fixture(scope="module")
def sd():
    if not config.SPB_15MIN.exists():
        pytest.skip("нет 15-минутных данных — python -m src.clean_spb_15min")
    return D.build()


def test_rows_definition(sd):
    r = sd.rows
    assert set(r.minute) == {0, 30} and set(r.horizon) == set(D.HORIZONS)
    assert D.local_hour(r.t).isin(D.HOURS).all() and r.hour.isin(D.HOURS).all()
    assert (r.slot == r.t + (r.k - 1) * D.SLOT).all()
    assert set(r.month) == {2, 5, 7, 9}
    assert np.allclose(r.z, np.log((r.y + 1) / (r.n + 1)))


# --- Окна не пересекают разрывы между месяцами ----------------------------------------------
def test_windows_do_not_cross_month_gaps_synthetic():
    """Два блока с разрывом: огромное значение в конце блока 1 не попадает ни в одно окно блока 2; моменты, у которых
    окно начиналось бы раньше блока, недопустимы."""
    a = pd.date_range("2026-02-27 00:00", periods=40, freq="15min", tz="UTC")
    b = pd.date_range("2026-05-01 00:00", periods=40, freq="15min", tz="UTC")
    ts = a.append(b)
    y = np.ones((len(ts), 1))
    y[len(a) - 1] = 1e9
    q = D.Quarters(ts=ts, vestibules=["v"], y=y, norm=np.ones_like(y), block=D.blocks_of(ts))
    assert q.block.tolist() == [0] * 40 + [1] * 40
    ok = q.window_ok(pd.Series(b))
    assert not ok[:D.WINDOW].any() and ok[D.WINDOW:].all()
    _, idx = q.windows(pd.Series(b[ok]))
    assert (q.block[idx] == 1).all() and (q.y[idx, 0] < 1e9).all()


def test_windows_real_data_inside_block(sd):
    keys = sd.rows[["vestibule_id", "t"]].drop_duplicates()
    i, idx = sd.q.windows(keys.t)
    ns = pd.DatetimeIndex(keys.t).asi8
    expected = ns[:, None] - (D.WINDOW - np.arange(D.WINDOW))[None, :] * D.QUARTER.value
    assert (sd.q.ts.asi8[idx] == expected).all()
    assert (sd.q.block[idx] == sd.q.block[i][:, None]).all()


# --- Утечка: шум после t ничего не меняет ----------------------------------------------------
def _tiny_gru(sd) -> NN.GRUQuantile:
    rows = sd.rows[(sd.rows.month == 2) & (sd.rows.sday >= "2026-02-20")]
    return NN.GRUQuantile(len(sd.q.vestibules), dict(TINY)).fit(NN.samples(sd.q, rows))


def test_15min_after_t_changes_nothing(sd):
    """15-минутные четверти с ts ≥ t (и часовой файл с часа t) заменены шумом: норма слота, окно GRU, её прогноз
    и персистентность B2 для момента t те же, а факт слотов изменился."""
    h = I.load_hours()
    rng = np.random.default_rng(0)
    noisy = h.copy()
    for j, c in enumerate(I.QCOLS):
        after = (noisy.ts_utc + pd.Timedelta(minutes=15 * j)) >= ORIGIN
        noisy.loc[after, c] = rng.integers(0, 5000, after.sum())
    noisy["y15"] = noisy[I.QCOLS].sum(1)
    later = noisy.ts_utc >= D.floor_hour(pd.Series([ORIGIN]))[0]
    noisy.loc[later, "y_hourly"] = rng.integers(0, 20000, later.sum())
    sd2 = D.build(noisy)
    a, b = sd.rows[sd.rows.t == ORIGIN], sd2.rows[sd2.rows.t == ORIGIN]
    assert len(a) > 50 and a[["vestibule_id", "k"]].equals(b[["vestibule_id", "k"]])
    np.testing.assert_array_equal(a.n.to_numpy(), b.n.to_numpy())
    np.testing.assert_array_equal(B.b2_point(sd.q, a), B.b2_point(sd2.q, b))
    sa, sb = NN.samples(sd.q, a), NN.samples(sd2.q, b)
    np.testing.assert_array_equal(sa.x, sb.x)
    np.testing.assert_array_equal(sa.codes, sb.codes)
    m = _tiny_gru(sd)
    np.testing.assert_array_equal(m.predict(sa), m.predict(sb))
    assert not np.array_equal(a.y.to_numpy(), b.y.to_numpy())          # шум действительно попал в будущее


def test_b1_uses_only_hours_up_to_t0():
    """B1 для момента hh:30: часовой поток после t0 = hh − 1 обрезан (serve.cut_data) — прогнозы часов h = 1, 2, 3
    те же, что на полных данных."""
    if not (M.MODELS / B.OOS_MODELS[7]).exists():
        pytest.skip("нет моделей фолдов — python -m src.model --save-folds")
    d = E.load()
    t0 = D.last_closed_hour(pd.Series([ORIGIN]))[0]
    full = B.b1_hourly(B.hourly_rows(B.hourly_panel(d), (7,)), 7)
    panel = B.hourly_panel(S.cut_data(d, t0))
    rows = BT.make_rows(panel, horizons=B.H_HOURLY, for_export=True)
    cut = B.b1_hourly(F.add_features(rows[rows.t == t0], panel), 7)
    m = cut.merge(full, on=["vestibule_id", "t0", "hour_ts", "h_hour"], suffixes=("_cut", ""))
    assert len(m) > 50 and set(m.h_hour) == {1, 2, 3}
    for q in B.QS:
        np.testing.assert_allclose(m[f"m{q}_cut"], m[f"m{q}"], rtol=0, atol=1e-9)


def test_b1_h1_matches_nowcast(sd):
    """Строки h = 1 в общей таблице h = 1, 2, 3 дают те же прогнозы, что nowcast.model_oos."""
    if not (M.MODELS / B.OOS_MODELS[5]).exists():
        pytest.skip("нет моделей фолдов")
    from src import nowcast as N
    hourly = B.b1_hourly(B.hourly_rows(B.hourly_panel(), (5,)), 5)
    ref = N.model_oos((5,)).rename(columns={"ts_utc": "hour_ts"})
    m = hourly[hourly.h_hour == 1].merge(ref, on=["vestibule_id", "hour_ts"])
    assert len(m) == len(ref)
    assert np.abs(m.m50_x - m.m50_y).max() < 1e-9


# --- Вне выборки: что видит мета-модель -------------------------------------------------------
def test_base_models_out_of_sample():
    for month, name in B.OOS_MODELS.items():
        path = M.MODELS / name / "meta.json"
        if not path.exists():
            pytest.skip(f"нет {name}")
        B.check_oos(json.loads(path.read_text(encoding="utf-8")), month)
    assert 2 not in B.OOS_MODELS
    assert all(max(ms) < m for m, ms in R.TRAIN_MONTHS.items())          # GRU и квантили B2 — по прошлым месяцам
    assert MT.FIT_MONTHS == (5, 7) and MT.CHECK_MONTH not in MT.FIT_MONTHS


@pytest.fixture(scope="module")
def cross():
    path = R.CACHE / "cross.parquet"
    if not path.exists():
        pytest.skip("нет прогнозов базовых моделей — python -m src.stack --fit")
    df = pd.read_parquet(path)
    return df[df.sday.dt.day % 3 == 0]                                    # каждый третий день — быстрее


def _base_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("stack")])


def test_meta_fit_uses_only_may_and_july(cross):
    """Шумовые строки февраля и сентября (прогнозы и факт — шум) ни на что в подборе не влияют."""
    base = _base_only(cross)
    p0, _ = MT.fit_params(base)
    rng = np.random.default_rng(1)
    noise = []
    for month, shift in ((2, -90), (9, 120)):
        x = base[base.month == 5].copy()
        x["month"] = month
        x["sday"] = x.sday + pd.Timedelta(days=shift)
        for c in ["z", *[f"{n}_{q}" for n in B.BASES for q in B.QS]]:
            x[c] = rng.normal(0, 1, len(x))
        x.index = x.index + 10 ** 7 * (month + 1)
        noise.append(x)
    p1, _ = MT.fit_params(pd.concat([base, *noise]))
    assert p0 == p1 and p0["fit_months"] == [5, 7]


def test_one_hot_weights_reproduce_base_model(cross):
    base = _base_only(cross)
    for j, name in enumerate(B.BASES):
        w = [0.0] * len(B.BASES)
        w[j] = 1.0
        p = {"names": list(B.BASES), "by": ["k"], "w": {q: {str(k): w for k in range(1, 5)} for q in B.QS},
             "w_k": {q: {str(k): w for k in range(1, 5)} for q in B.QS}}
        st = MT.apply_weights(base, p)
        for q in B.QS:
            np.testing.assert_array_equal(st[f"stack_{q}"].to_numpy(), base[f"{name}_{q}"].to_numpy())


def test_quantiles_sorted(cross):
    params = MT.load_params(R.PARAMS_JSON) if R.PARAMS_JSON.exists() else MT.fit_params(_base_only(cross))[0]
    out = MT.predict(_base_only(cross), params)
    for name in (*B.BASES, "mean"):
        z = cross[B.cols(name)].to_numpy()
        assert (np.diff(z, axis=1) >= 0).all(), name
    for name in ("stackraw", "stack"):
        z = out[B.cols(name)].to_numpy()
        assert (np.diff(z, axis=1) >= 0).all(), name
    assert all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0 for cells in params["weights"]["w"].values()
               for w in cells.values())


def test_simplex_grid():
    g = MT.simplex(3)
    assert len(g) == 231 and np.allclose(g.sum(1), 1) and (g >= 0).all()


# --- Контракт ----------------------------------------------------------------------------------
def test_station_records_pass_contract(cross):
    cfg = M.load_config()
    st = V.station_sum(cross)
    sq = V.fit_station_quantiles(st)
    st = st.join(V.station_interval(st, sq).drop(columns="q50"))
    one_day = st[st.sday == st.sday.max()]
    records = V.station_records(one_day, cfg["anomaly_rule"], pd.DataFrame(cfg["thresholds"]), MT.VERSION)
    assert V.validate(records) == len(one_day) > 0
    assert {r["horizon_min"] for r in records} == set(D.HORIZONS)
    assert all(pd.Timestamp(r["ts"]).minute in (0, 30) for r in records)


# --- GRU -------------------------------------------------------------------------------------------
def _synthetic_samples(n: int = 600, seed: int = 0) -> NN.Samples:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.2, (n, D.WINDOW, 4)).astype(np.float32)
    x[..., 1] = rng.random((n, D.WINDOW)) < 0.1
    x[..., 0][x[..., 1] > 0] = np.nan
    x[..., 3] = 0
    codes = np.column_stack([rng.integers(0, 5, n), rng.integers(0, 96, n), rng.integers(0, 7, n),
                             rng.integers(0, 4, n)]).astype(np.int64)
    z = x[:, -1, 2:3].astype(float) + rng.normal(0, 0.1, (n, D.K))
    w = rng.uniform(50, 500, (n, D.K))
    keys = pd.DataFrame({"vestibule_id": codes[:, 0].astype(str), "t": pd.Timestamp("2026-02-01", tz="UTC"),
                         "sday": pd.Timestamp("2026-02-01") + pd.to_timedelta(np.arange(n) % 10, unit="D")})
    return NN.Samples(keys, x, codes, z, w)


def test_gru_deterministic_and_sorted():
    s = _synthetic_samples()
    a = NN.GRUQuantile(5, dict(TINY)).fit(s)
    b = NN.GRUQuantile(5, dict(TINY)).fit(s)
    za, zb = a.predict(s), b.predict(s)
    np.testing.assert_array_equal(za, zb)
    assert za.shape == (len(s.keys), D.K, 3) and (np.diff(za, axis=-1) >= 0).all()
    assert a.info["epochs"] == b.info["epochs"]


def test_gru_early_stopping_window_is_last_days():
    s = _synthetic_samples()
    m = NN.GRUQuantile(5, dict(TINY))
    hold, es = m.split(s)
    last = s.keys.sday.max()
    assert (s.keys.sday[es] > last - pd.Timedelta(days=TINY["es_days"])).all()
    assert (s.keys.sday[hold] <= last - pd.Timedelta(days=TINY["es_days"])).all()


def test_gru_save_load_roundtrip(tmp_path):
    s = _synthetic_samples()
    m = NN.GRUQuantile(5, dict(TINY)).fit(s)
    m.save(tmp_path / "gru")
    np.testing.assert_array_equal(NN.GRUQuantile.load(tmp_path / "gru").predict(s), m.predict(s))


# --- Заморозка ----------------------------------------------------------------------------------
def test_frozen_model_untouched():
    if not M.FINAL_JSON.exists():
        pytest.skip("нет model_final.json")
    M.check_frozen(M.load_config())


def test_params_hash(tmp_path):
    p = {"version": MT.VERSION, "variant": "V1"}
    p["sha256"] = MT._sha(p)
    path = tmp_path / "p.json"
    MT.save_params(p, path)
    assert MT.load_params(path)["variant"] == "V1"
    p["variant"] = "V2"
    path.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(SystemExit, match="хэш"):
        MT.load_params(path)
