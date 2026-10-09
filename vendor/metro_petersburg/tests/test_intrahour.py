import numpy as np
import pandas as pd
import pytest

from src import config
from src import intrahour as I


# --- Арифметика ---------------------------------------------------------------------
def test_nonuniformity():
    assert I.nonuniformity([0.25] * 4) == pytest.approx(0)
    assert I.nonuniformity([1, 0, 0, 0]) == pytest.approx(0.75)
    assert I.nonuniformity([0.5, 0.5, 0, 0]) == pytest.approx(0.5)


def _toy(R=1.3, ys=(30, 35, 35, 30), thr=0.2, b4=100.0, s_v=1.0, p=(0.25,) * 4) -> pd.DataFrame:
    """Одна строка сигнала: часовая истина y = R · b4, четверти ys в 15-минутном источнике."""
    row = {"b4": b4, "s_v": s_v, "y": R * b4, "R": R, "thr": thr, **dict(zip(I.QCOLS, ys)), **dict(zip(I.PCOLS, p))}
    s = pd.DataFrame([row])
    s["anom"] = (s.R - 1).abs() > s.thr
    s["up"] = s.R > 1
    cy, cp = s[I.QCOLS].cumsum(1).to_numpy(float), s[I.PCOLS].cumsum(1).to_numpy(float)
    for k in range(1, 5):
        s[f"r{k}"] = cy[:, k - 1] / (s.b4.to_numpy() * s.s_v.to_numpy() * cp[:, k - 1])
    return s


def test_partial_ratio():
    s = _toy(ys=(25, 50, 25, 25), s_v=1.25, p=(0.2, 0.3, 0.3, 0.2))
    assert s.r1.item() == pytest.approx(25 / (100 * 1.25 * 0.2))           # 1,0
    assert s.r2.item() == pytest.approx(75 / (100 * 1.25 * 0.5))           # 1,2
    assert s.r4.item() == pytest.approx(125 / 125)


def test_sequential_detection_time_and_sign():
    c = {1: 1.0, 2: 1.0, 3: 1.0}
    # r1 = 1,2 (не выше 1 + 0,2), r2 = 1,3 → тревога на 30-й минуте; час аномален вверх
    s = pd.concat([_toy(ys=(30, 35, 35, 30)),
                   _toy(R=0.6, ys=(40, 40, 40, 40)),             # аномалия вниз, а r_k > 1 → знак не тот, не найдена
                   _toy(R=1.0, ys=(40, 25, 25, 10))], ignore_index=True)   # тревога на 15-й, час не аномален
    seq = I.sequential(s, c)
    assert seq.detected_min.iloc[0] == 30
    assert np.isnan(seq.detected_min.iloc[1]) and seq.first_k.iloc[1] == 1 and seq.first_up.iloc[1]
    assert seq.first_k.iloc[2] == 1 and not seq.anom.iloc[2]
    sc = I.snapshot_scores(s, 2, 1.0)
    assert sc["tp"] == 1 and sc["alarms"] == 3 and sc["anomalies"] == 2


def test_choose_c_maximizes_f1():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        R = 1.5 if i % 4 == 0 else 1.0
        y = R * 100 / 4 * (1 + rng.normal(0, 0.15, 4))
        rows.append(_toy(R=R, ys=tuple(y)))
    s = pd.concat(rows, ignore_index=True)
    cs = I.choose_c(s)
    for r in cs.itertuples():
        grid = [I.snapshot_scores(s, r.k, c)["f1"] for c in I.C_GRID]
        assert r.f1 == pytest.approx(max(grid))


# --- Реальные данные ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def hours():
    if not config.SPB_15MIN.exists():
        pytest.skip("нет data/interim/spb_line1_15min.parquet — запустите python -m src.clean_spb_15min")
    return I.load_hours()


def test_past_profile_and_scale_use_only_past(hours):
    """Шум вместо потока с суток D не меняет профиль и s_v ни для одних суток ≤ D (профиль и s_v — строго до D)."""
    cut = pd.Timestamp("2026-05-15")
    p0, s0 = I.past_profile(hours), I.source_scale(hours)
    noisy = hours.copy()
    after = noisy.sday >= cut
    rng = np.random.default_rng(1)
    noisy.loc[after, I.QCOLS] = rng.integers(0, 5000, (after.sum(), 4))
    noisy["y15"] = noisy[I.QCOLS].sum(1)
    noisy.loc[after, "y_hourly"] = rng.integers(0, 5000, after.sum())
    p1, s1 = I.past_profile(noisy), I.source_scale(noisy)
    keep = hours.sday <= cut
    pd.testing.assert_frame_equal(p0[keep], p1[keep])
    pd.testing.assert_series_equal(s0[keep], s1[keep])
    assert not p0[~keep].equals(p1[~keep])                 # а после D — меняется: тест не пустой


def test_past_profile_levels(hours):
    p = I.past_profile(hours)
    ok = p.p0.notna()
    assert np.allclose(p.loc[ok, I.PCOLS].sum(1), 1)
    first_day = hours.sday == hours.sday.min()
    assert p.loc[first_day, "p0"].isna().all()             # в первые сутки прошлого нет
    work = (hours.sday >= I.EVAL_START) & hours.hour.isin(I.E.WORK_HOURS) & hours.ok
    assert (p.loc[work, "profile_level"] != "").mean() > 0.99


def test_simulator_profile(hours):
    sim = I.simulator_profile(hours)
    sums = sim.groupby(["station_id", "vestibule_id", "day_type", "hour"]).share.sum()
    assert np.allclose(sums, 1, atol=1e-5)
    assert sim.share.between(0, 1).all() and sim.notna().all().all()
    assert set(sim.day_type) == set(I.DAY_TYPES3) and set(sim.quarter) == {0, 1, 2, 3}
    assert set(sim.vestibule_id) >= {"*", "tekhnologichesky_institut_1"}
    ti = sim[sim.station_id == "tekhnologichesky_institut"]
    assert (ti.note == I.SIM_NOTE_15MIN_ONLY).all() and (sim[sim.station_id != "tekhnologichesky_institut"].note == "").all()
    lp2 = sim[sim.vestibule_id == "leninsky_prospekt_2"]
    assert lp2.hour.map(lambda h: 6 <= h <= 21).all()      # вне режима вестибюля строк нет
    assert set(sim.source) <= {"вестибюль", "станция", "группа", "линия"}


def test_selection_never_sees_september(hours):
    sig = I.signal_table(hours, with_model=False)
    sel, sep = I.split_periods(sig)
    assert set(sel.month) == set(I.SELECT_MONTHS) and set(sep.month) == {I.CHECK_MONTH}
    assert sel.sday.min() >= I.EVAL_START and sel.main.all()
    assert not (sig.main & ~sig.ok).any()                   # флаги, в том числе расхождение и частичное закрытие, — вне оценки
