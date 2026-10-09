"""Признаки LightGBM (этап 4): строка (вестибюль, t, h) — только данные до конца часа t.

- Поток на входе — наблюдённый: закрытые часы (метро или вестибюль) → NaN, час инцидента входит — это реальные данные
  на момент прогноза (как r(t) в бэктесте).
- Отклонения r считаются к f_ref = b4 × уровень — к той же норме, что в цели модели и в лучшем бейзлайне.
- Признаки часа τ = t + h — норма, календарь, исторический прогноз погоды — известны заранее; лаги потока
  у τ — не ближе τ − 24.
- Признаков месяца и даты нет: в тестовом месяце они принимают невиданные значения.

Группы — из «Решений для этапа признаков» (docs/eda_spb_findings.md): база и 6 групп абляции в порядке добавления.
`is_holiday` в «особых днях» нет: его несёт «тип дня τ» базы — рабочих суббот в 2026 году нет, и сверх дня недели
тип дня означает только «праздник / перенос».
"""
import warnings

import numpy as np
import pandas as pd

from src import backtest as BT
from src import baseline as B
from src import eda_spb as E

GROUPS = {
    "база": ["vestibule", "station_group", "hour_tau", "dow_tau", "day_type_tau", "h", "b4_tau", "level",
             "y_t", "y_t1", "y_tau24", "y_tau168"],
    "ядро динамики": ["r_t", "ewm_r_2h", "ewm_r_6h", "cum_ratio", "ratio_168"],
    "кросс": ["r_line", "ewm_r_line_2h", "r_neighbors", "r_twin", "share_dev"],
    "скорость и ускорение": ["excess_d1_rel", "excess_d2_rel", "excess_d3_rel", "dr2", "dr3"],
    "вчера": ["r_yday", "cum_ratio_yday"],
    "погода": ["fc_precip_tau", "fc_precip_3h_tau"],
    "особые дни": ["shortened_14_16", "shortened_17_19", "night_ahead_late"],
}
ABLATION = list(GROUPS)[1:]
ALL = [f for fs in GROUPS.values() for f in fs]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
CATEGORICAL = ["vestibule", "station_group", "day_type_tau"]
COUNTS = ["b4_tau", "y_t", "y_t1", "y_tau24", "y_tau168"]     # входы: у станции складываются
PRECIP_MM = 0.5            # «дождь» — как в EDA: ≥ 0,5 мм/ч
NEIGHBOR_STATIONS = 2      # соседи — станции не дальше ±2 по линии
SHORTENED_EARLY, SHORTENED_LATE, NIGHT_LATE = [14, 15, 16], [17, 18, 19], [23, 0]


def _per_hour(a: np.ndarray, V: int) -> np.ndarray:
    return np.repeat(a[:, None], V, 1)


def _ratio_sum(y: np.ndarray, f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Σ y / Σ f по вестибюлям маски, для каждого часа; NaN, пока Σ f < B_MIN."""
    ys, fs = np.where(mask, y, 0).sum(1), np.where(mask, f, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fs >= E.B_MIN, ys / fs, np.nan)


def _cumulative(y: np.ndarray, f: np.ndarray, sday: pd.DatetimeIndex) -> np.ndarray:
    """Σ y / Σ f с начала суток метро (05:00) по конец часа, по вестибюлям; NaN, пока Σ f < B_MIN."""
    m = ~np.isnan(y) & ~np.isnan(f)
    key = sday.to_numpy()
    cy = pd.DataFrame(np.where(m, y, 0)).groupby(key).cumsum().to_numpy()
    cf = pd.DataFrame(np.where(m, f, 0)).groupby(key).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cf >= E.B_MIN, cy / cf, np.nan)


def t_matrices(panel: BT.Panel) -> dict[str, np.ndarray]:
    """Признаки момента t: массивы час × вестибюль, значение в строке t — по потоку до конца часа t."""
    g, ves, st = panel.d.grid, panel.d.vestibules, panel.d.stations
    T, V = g.entries.shape
    obs = np.where(g.closed, np.nan, g.entries)
    f = panel.norms["b4_level"]
    r = panel.r_level_in
    sh = E._shift
    known = ~np.isnan(r)
    rl = _ratio_sum(obs, f, known)

    st_r = {}
    for sid, idx in ves.groupby("station_id", sort=False).pos:
        cols = np.zeros(V, bool)
        cols[idx.to_numpy()] = True
        st_r[sid] = _ratio_sum(obs, f, known & cols[None, :])
    order = st.set_index("station_id").line_order
    neigh = np.full((T, V), np.nan)
    twin = np.full((T, V), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # Mean of empty slice
        for v, sid in enumerate(ves.station_id):
            near = [s for s in st_r if s != sid and abs(order[s] - order[sid]) <= NEIGHBOR_STATIONS]
            neigh[:, v] = np.nanmean(np.column_stack([st_r[s] for s in near]), axis=1)
    for _, idx in ves.groupby("station_id").pos:
        if len(idx) == 2:
            i, j = idx.to_numpy()
            twin[:, i], twin[:, j] = r[:, j], r[:, i]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "y_t": obs, "y_t1": sh(obs, 1),
            "r_t": r, "ewm_r_2h": E.ewm_ratio(r, g.ts, 2), "ewm_r_6h": E.ewm_ratio(r, g.ts, 6),
            "cum_ratio": _cumulative(obs, f, g.sday),
            "ratio_168": (obs + 1) / (sh(obs, 168) + 1),
            "r_line": _per_hour(rl, V),
            "ewm_r_line_2h": _per_hour(E.ewm_ratio(rl[:, None], g.ts, 2)[:, 0], V),
            "r_neighbors": neigh, "r_twin": twin, "share_dev": r / rl[:, None],
            "dr2": r - sh(r, 2), "dr3": r - sh(r, 3),
        }
        for k in (1, 2, 3):
            ex = (obs - sh(obs, k)) - (f - sh(f, k))
            out[f"excess_d{k}_rel"] = np.where(f >= E.B_MIN, ex / f, np.nan)
    return out


def tau_arrays(panel: BT.Panel) -> dict[str, np.ndarray]:
    """Признаки часа τ (индекс — час τ): норма, календарь и прогноз погоды известны заранее, лаги потока —
    не ближе τ − 24. Вектор длины T — общий для всех вестибюлей, матрица T × V — свой у каждого."""
    g, cal = panel.d.grid, panel.d.calendar
    obs = np.where(g.closed, np.nan, g.entries)
    r, f = panel.r_level_in, panel.norms["b4_level"]
    b4 = panel.norms["b4"]
    c = cal.set_index("date").reindex(g.sday)

    known = ~np.isnan(r)
    key = g.sday.to_numpy()
    ys = pd.DataFrame(np.where(known, obs, 0)).groupby(key).sum()
    fs = pd.DataFrame(np.where(known, f, 0)).groupby(key).sum()
    daily = ys / fs.where(fs >= E.B_MIN)
    yday = daily.reindex(g.sday - pd.Timedelta(days=1)).to_numpy()   # прошлые сутки метро целиком до t

    fc = panel.d.forecast.set_index("ts_utc").reindex(g.ts)
    prec = fc.precipitation.to_numpy(dtype=float)
    nights = set(E.event_days(panel.d.events[panel.d.events.source == "по данным потока"]))
    short = c.is_shortened.eq(True).to_numpy()
    return {
        "hour_tau": np.where(g.hour == 0, 24, g.hour).astype(float),      # 00 — конец суток метро, после 23
        "dow_tau": g.sday.dayofweek.to_numpy().astype(float),
        "day_type_tau": c.day_type.to_numpy(),
        "b4_tau": b4,
        "level": B.level_factor(g, cal, b4),
        "y_tau24": E._shift(obs, 24), "y_tau168": E._shift(obs, 168),
        "r_yday": E._shift(r, 24),
        "cum_ratio_yday": yday,
        "fc_precip_tau": (prec >= PRECIP_MM).astype(float),
        "fc_precip_3h_tau": pd.Series(prec).rolling(3, min_periods=3).sum().to_numpy(),
        "shortened_14_16": (short & np.isin(g.hour, SHORTENED_EARLY)).astype(float),
        "shortened_17_19": (short & np.isin(g.hour, SHORTENED_LATE)).astype(float),
        "night_ahead_late": (g.sday.isin(list(nights)) & np.isin(g.hour, NIGHT_LATE)).astype(float),
    }


def add_features(rows: pd.DataFrame, panel: BT.Panel) -> pd.DataFrame:
    """Строки бэктеста (BT.make_rows) + колонки признаков ALL; категории — с фиксированным списком значений."""
    g, ves = panel.d.grid, panel.d.vestibules
    ti, taui = g.ts.get_indexer(rows.t), g.ts.get_indexer(rows.tau)
    if (ti < 0).any() or (taui < 0).any():
        raise ValueError("t или τ вне сетки")
    vi = rows.vestibule_id.map(dict(zip(ves.vestibule_id, ves.pos))).to_numpy()
    out = {name: a[ti, vi] for name, a in t_matrices(panel).items()}
    for name, a in tau_arrays(panel).items():
        out[name] = a[taui] if a.ndim == 1 else a[taui, vi]
    out["vestibule"] = pd.Categorical(rows.vestibule_id, categories=ves.vestibule_id)
    out["station_group"] = pd.Categorical(rows.group, categories=E.GROUPS)
    out["day_type_tau"] = pd.Categorical(out["day_type_tau"], categories=E.DAY_TYPES)
    missing = set(ALL) - set(out) - {"h"}
    if missing:
        raise RuntimeError(f"не посчитаны признаки: {sorted(missing)}")
    clash = set(out) & set(rows.columns)
    if clash:   # признак не должен молча подменить колонку бэктеста (её читают бейзлайны)
        raise ValueError(f"признаки совпадают с колонками строк: {sorted(clash)}")
    return rows.assign(**out)


# --- Человекочитаемые описания (объяснения для LLM-слоя, подписи графиков) -----------
TITLES = {
    "vestibule": "вестибюль", "station_group": "группа станций", "hour_tau": "час прогноза",
    "dow_tau": "день недели", "day_type_tau": "тип дня", "h": "горизонт", "b4_tau": "норма b4 на час",
    "level": "уровень последних 7 дней", "y_t": "поток в последний час", "y_t1": "поток час назад",
    "y_tau24": "поток в этот час вчера", "y_tau168": "поток в этот час неделю назад",
    "r_t": "отклонение от нормы в последний час", "ewm_r_2h": "отклонение за последние 2 ч (сглаж.)",
    "ewm_r_6h": "отклонение за последние 6 ч (сглаж.)", "cum_ratio": "Σ потока / Σ нормы с открытия",
    "ratio_168": "поток к тому же часу неделю назад", "r_line": "отклонение всей линии",
    "ewm_r_line_2h": "отклонение линии за 2 ч (сглаж.)", "r_neighbors": "отклонение соседних станций",
    "r_twin": "отклонение второго вестибюля", "share_dev": "доля вестибюля в линии к обычной",
    "excess_d1_rel": "прирост против обычного за 1 ч", "excess_d2_rel": "прирост против обычного за 2 ч",
    "excess_d3_rel": "прирост против обычного за 3 ч", "dr2": "изменение отклонения за 2 ч",
    "dr3": "изменение отклонения за 3 ч", "r_yday": "отклонение в этот час вчера",
    "cum_ratio_yday": "Σ потока / Σ нормы вчера", "fc_precip_tau": "прогноз дождя на час",
    "fc_precip_3h_tau": "прогноз осадков за 3 ч", "shortened_14_16": "сокращённый день, 14–16 ч",
    "shortened_17_19": "сокращённый день, 17–19 ч", "night_ahead_late": "вечер перед ночью до утра",
}
DOW = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
DAY_TYPE_TEXT = {"рабочий": "рабочий день", "суббота": "суббота", "воскресенье": "воскресенье",
                 "праздник": "праздничный или перенесённый выходной день"}


def _num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{abs(x) * 100:.0f} %"


def _dev(r: float, what: str, more: str = "выше", less: str = "ниже", base: str = "нормы") -> str:
    """«{what} на 18 % выше нормы»; около нормы (±2 %) — «{what} на уровне нормы»."""
    if abs(r - 1) < 0.02:
        return f"{what} на уровне {base}"
    return f"{what} на {_pct(r - 1)} {more if r > 1 else less} {base}"


def describe(name: str, value, hour: int | None = None, place: str | None = None) -> str:
    """Текст причины: значение признака словами. hour — час τ, place — вестибюль или станция."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return f"{TITLES[name]}: нет данных"
    hh = f"{hour:02d} ч" if hour is not None else "этот час"
    v = value
    match name:
        case "vestibule":
            return f"особенности {place or 'вестибюля'}"
        case "station_group":
            return f"тип станции: {str(v).lower()}"
        case "hour_tau":
            return f"поправка на час {int(v) % 24:02d}:00"
        case "dow_tau":
            return f"поправка на день недели: {DOW[int(v)]}"
        case "day_type_tau":
            return DAY_TYPE_TEXT[str(v)]
        case "h":
            return f"поправка на горизонт {int(v)} ч"
        case "b4_tau":
            return f"поправка на величину нормы ({_num(v)} входов в {hh})"
        case "level":
            return _dev(v, "последнюю неделю поток")
        case "y_t":
            return f"в последний час вошло {_num(v)}"
        case "y_t1":
            return f"час назад вошло {_num(v)}"
        case "y_tau24":
            return f"вчера в {hh} вошло {_num(v)}"
        case "y_tau168":
            return f"неделю назад в {hh} вошло {_num(v)}"
        case "r_t":
            return _dev(v, "в последний час поток")
        case "ewm_r_2h":
            return _dev(v, "поток последние 2 ч")
        case "ewm_r_6h":
            return _dev(v, "поток последние 6 ч")
        case "cum_ratio":
            return _dev(v, "с открытия вошло", "больше", "меньше", "обычного")
        case "ratio_168":
            return _dev(v, "в последний час вошло", "больше", "меньше", "чем неделю назад")
        case "r_line":
            return _dev(v, "вся линия")
        case "ewm_r_line_2h":
            return _dev(v, "вся линия последние 2 ч")
        case "r_neighbors":
            return _dev(v, "соседние станции")
        case "r_twin":
            return _dev(v, "второй вестибюль станции")
        case "share_dev":
            return _dev(v, "доля вестибюля в потоке линии", base="обычной")
        case "excess_d1_rel" | "excess_d2_rel" | "excess_d3_rel":
            k = name[8]
            if abs(v) < 0.02:
                return f"за {k} ч поток менялся как обычно"
            return f"за {k} ч поток рос {'быстрее' if v > 0 else 'медленнее'} обычного на {_pct(v)} нормы"
        case "dr2" | "dr3":
            k = name[2]
            if abs(v) < 0.02:
                return f"за {k} ч отклонение от нормы не изменилось"
            return f"за {k} ч отклонение от нормы {'выросло' if v > 0 else 'снизилось'} на {abs(v) * 100:.0f} п. п."
        case "r_yday":
            return _dev(v, f"вчера в {hh} поток был")
        case "cum_ratio_yday":
            return _dev(v, "вчера за сутки вошло", "больше", "меньше", "нормы")
        case "fc_precip_tau":
            return f"по прогнозу дождь в {hh}" if v >= 0.5 else f"по прогнозу без дождя в {hh}"
        case "fc_precip_3h_tau":
            if v < 0.1:
                return f"по прогнозу без осадков за 3 ч до {hh}"
            return f"по прогнозу осадки {v:.1f} мм за 3 ч до {hh}".replace(".", ",")
        case "shortened_14_16" | "shortened_17_19":
            return "сокращённый предпраздничный день" if v >= 0.5 else "обычная продолжительность рабочего дня"
        case "night_ahead_late":
            return "впереди ночь, когда метро работает до утра" if v >= 0.5 else "обычный вечер"
    raise KeyError(name)
