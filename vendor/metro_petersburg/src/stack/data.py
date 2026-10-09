"""Данные стекинга: четверти, слоты, строки прогноза и окна 4 ч.

**Постановка.** Момент прогноза t ∈ {hh:00, hh:30}, час t — 06–00. Слоты k = 1…4: [t + 30(k − 1), t + 30k),
horizon_min = 30k. Час слота — тоже 06–00.

**Что известно в момент t:**
- 15-минутные четверти с ts < t: четверть [ts, ts + 15) закончилась к t;
- часовые данные — до последнего закрытого часа t0 = floor_hour(t) − 1 ч. В hh:30 час hh ещё не закрыт, поэтому t0
  тот же, что в hh:00.

**Таблица часов** — intrahour.signal_table (этап 6): y0…y3, b4 (замороженная baseline.norm), профиль p0…p3
по суткам строго до D, поправка источника s_v по суткам строго до D, признак `main`, истина аномалии `anom`.

**Слот** — половина часа: y_slot = y0 + y1 или y2 + y3 (15-минутные данные), норма n_slot = b4 · s_v · (p0 + p1)
или b4 · s_v · (p2 + p3), цель z = log((y_slot + 1)/(n_slot + 1)), вес — n_slot.

**Строка** — (вестибюль, t, k), если час слота проходит `main` этапа 6: обычные сутки с 09.02, без флагов,
is_source_mismatch и is_slot_closed, in_hourly, b4 ≥ 20, есть профиль и s_v.

**Четверть для входа моделей** валидна, если её час есть в таблице часов без флагов (`ok`), b4 ≥ 20, есть профиль
и s_v. Норма четверти — b4 · s_v · p_j. Ночь, флаги и малые нормы — пропуск.

**Блоки.** 15-минутные данные — 4 месяца (февраль, май, июль, сентябрь), между ними разрывы. Окно [t − 4 ч, t)
собирается по меткам времени и допустимо, только если целиком лежит внутри блока момента t.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src import eda_spb as E
from src import intrahour as I
from src import nowcast as N

K = 4                                        # слотов вперёд
SLOT = pd.Timedelta(minutes=30)
QUARTER = pd.Timedelta(minutes=15)
HORIZONS = tuple(30 * k for k in range(1, K + 1))
WINDOW = 16                                  # четвертей во входе (4 ч)
HOURS = E.WORK_HOURS                         # 06–00: и момент t, и слот
PEAK_HOURS = N.PEAK_HOURS
BAND_OF = N.BAND_OF                          # утро 06–09, день 10–15, вечер 16–19, поздно 20–00
BANDS = list(N.BANDS)
MONTH_RU = {2: "февраль", 5: "май", 7: "июль", 9: "сентябрь"}


def floor_hour(ts: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return ts.dt.floor("h") if isinstance(ts, pd.Series) else ts.floor("h")


def last_closed_hour(t) -> pd.Series:
    """t0 — начало последнего закрытого часа в момент t: floor_hour(t) − 1 ч (в hh:30 час hh ещё идёт)."""
    return floor_hour(t) - pd.Timedelta(hours=1)


def hourly_h(slot_hour: pd.Series, t: pd.Series) -> np.ndarray:
    """Горизонт часовой модели для слота в часе slot_hour при прогнозе в момент t: (slot_hour − t0) в часах."""
    return ((slot_hour - last_closed_hour(t)) / pd.Timedelta(hours=1)).to_numpy().astype(int)


@dataclass
class Quarters:
    """Сетка четвертей «время × вестибюль» по всем блокам: факт, норма, валидность, блок."""
    ts: pd.DatetimeIndex                     # UTC, начало четверти, по возрастанию
    vestibules: list[str]
    y: np.ndarray                            # [T, V] входы (15-минутный источник)
    norm: np.ndarray                         # [T, V] b4 · s_v · p_j; NaN — пропуск
    block: np.ndarray                        # [T] номер блока (непрерывного куска сетки)

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.norm)

    def index_of(self, t) -> np.ndarray:
        """Позиция четверти, начинающейся в момент t (−1 — нет такой четверти)."""
        return self.ts.get_indexer(pd.DatetimeIndex(t))

    def z(self) -> np.ndarray:
        """log((y + 1)/(норма + 1)) по валидным четвертям, иначе NaN."""
        with np.errstate(invalid="ignore"):
            return np.where(self.valid, np.log((self.y + 1) / (self.norm + 1)), np.nan)

    def z_line(self) -> np.ndarray:
        """Отклонение всей линии по четвертям: log((Σy + 1)/(Σнорма + 1)) по валидным вестибюлям; нет ни одного — NaN."""
        v = self.valid
        sy = np.where(v, self.y, 0.0).sum(1)
        sn = np.where(v, self.norm, 0.0).sum(1)
        return np.where(v.any(1), np.log((sy + 1) / (sn + 1)), np.nan)

    def window_ok(self, t) -> np.ndarray:
        """Окно [t − 4 ч, t) целиком в блоке момента t (t — начало четверти в сетке)."""
        i = self.index_of(t)
        start = i - WINDOW
        ok = (i >= 0) & (start >= 0)
        same = np.zeros(len(i), bool)
        same[ok] = self.block[start[ok]] == self.block[i[ok]]
        span = np.zeros(len(i), bool)
        span[ok] = (self.ts[i[ok]] - self.ts[start[ok]]) == WINDOW * QUARTER
        return ok & same & span

    def windows(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Позиции четвертей окна [WINDOW] для каждого t: i − 16 … i − 1; t должны проходить window_ok."""
        i = self.index_of(t)
        if (i < 0).any():
            raise ValueError("момент t вне сетки четвертей")
        return i, i[:, None] + np.arange(-WINDOW, 0)[None, :]


def blocks_of(ts: pd.DatetimeIndex) -> np.ndarray:
    """Номер непрерывного куска сетки: новый блок там, где шаг больше 15 минут."""
    gap = np.r_[True, np.diff(ts.asi8) != QUARTER.value]
    return np.cumsum(gap) - 1


def quarter_grid(h: pd.DataFrame, s: pd.DataFrame) -> Quarters:
    """Сетка четвертей из таблицы часов этапа 6 (h — load_hours, s — signal_table). Вестибюли — только in_hourly."""
    h = h[h.in_hourly]
    ves = sorted(h.vestibule_id.unique())
    hours = pd.DatetimeIndex(np.sort(h.ts_utc.unique()))
    ts = (hours.repeat(I.Q) + pd.to_timedelta(np.tile(np.arange(I.Q) * 15, len(hours)), unit="min"))
    hi = hours.get_indexer(h.ts_utc)
    vi = pd.Index(ves).get_indexer(h.vestibule_id)
    y = np.full((len(hours), len(ves), I.Q), np.nan)
    y[hi, vi] = h[I.QCOLS].to_numpy(float)

    good = s[s.ok & (s.b4 >= E.B_MIN) & s[I.PCOLS].notna().all(1) & s.s_v.notna()]
    norm = np.full((len(hours), len(ves), I.Q), np.nan)
    gi = hours.get_indexer(good.ts_utc)
    gv = pd.Index(ves).get_indexer(good.vestibule_id)
    norm[gi, gv] = good[I.PCOLS].to_numpy(float) * (good.b4 * good.s_v).to_numpy(float)[:, None]
    norm = np.where(np.isnan(y), np.nan, norm)
    to2d = lambda a: a.transpose(0, 2, 1).reshape(len(hours) * I.Q, len(ves))
    return Quarters(ts=pd.DatetimeIndex(ts), vestibules=ves, y=to2d(y), norm=to2d(norm), block=blocks_of(pd.DatetimeIndex(ts)))


def slot_table(s: pd.DataFrame) -> pd.DataFrame:
    """Слоты (вестибюль × час × половина) из таблицы часов: факт, норма, цель z и срезы. Только строки `main`."""
    m = s[s.main].reset_index(drop=True)
    parts = []
    for half in (0, 1):
        a, b = 2 * half, 2 * half + 1
        p = m[f"p{a}"] + m[f"p{b}"]
        n = m.b4 * m.s_v * p
        y = (m[f"y{a}"] + m[f"y{b}"]).astype(float)
        parts.append(pd.DataFrame({
            "vestibule_id": m.vestibule_id, "station_id": m.station_id, "group": m.group, "hour_ts": m.ts_utc,
            "half": half, "slot": m.ts_utc + half * SLOT, "hour": m.hour, "sday": m.sday, "month": m.month,
            "day_type": m.day_type, "b4": m.b4, "s_v": m.s_v, "p_half": p, "n": n, "y": y,
            "z": np.log((y + 1) / (n + 1)), "anom": m.anom.astype(bool)}))
    out = pd.concat(parts, ignore_index=True)
    out["band4"] = out.hour.map(BAND_OF)
    out["peak"] = out.hour.isin(PEAK_HOURS)
    return out.sort_values(["slot", "vestibule_id"], kind="stable").reset_index(drop=True)


def local_hour(ts_utc: pd.Series) -> pd.Series:
    return ts_utc.dt.tz_convert(config.SPB_TZ).dt.hour


def make_rows(slots: pd.DataFrame, q: Quarters) -> pd.DataFrame:
    """Строки прогноза (вестибюль, t, k): слот k = 1…4 от момента t = slot − 30(k − 1). Момент t — в часах 06–00
    и с окном 4 ч внутри своего блока. t0 — последний закрытый час, h_hour — горизонт часовой модели."""
    parts = []
    for k in range(1, K + 1):
        r = slots.copy()
        r["k"] = k
        r["horizon"] = 30 * k
        r["t"] = r.slot - (k - 1) * SLOT
        parts.append(r)
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[local_hour(rows.t).isin(HOURS)]
    rows = rows[q.window_ok(rows.t)]
    rows["minute"] = rows.t.dt.minute
    rows["t0"] = last_closed_hour(rows.t)
    rows["h_hour"] = hourly_h(rows.hour_ts, rows.t)
    if not rows.h_hour.isin([1, 2, 3]).all():
        raise RuntimeError("горизонт часовой модели вне 1…3")
    return rows.sort_values(["t", "vestibule_id", "k"], kind="stable").reset_index(drop=True)


@dataclass
class StackData:
    q: Quarters
    slots: pd.DataFrame
    rows: pd.DataFrame
    hours: pd.DataFrame                      # signal_table (для B1: s_v и профиль часа уже в слотах)


def build(h: pd.DataFrame | None = None) -> StackData:
    """Всё для стекинга из 15-минутных данных: h — intrahour.load_hours (тест утечки подаёт сюда обрезанные данные)."""
    h = I.load_hours() if h is None else h
    s = I.signal_table(h, with_model=False)
    q = quarter_grid(h, s)
    slots = slot_table(s)
    return StackData(q=q, slots=slots, rows=make_rows(slots, q), hours=s)
