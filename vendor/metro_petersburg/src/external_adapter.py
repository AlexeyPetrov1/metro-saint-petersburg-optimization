"""Переходник external_data (внешние данные, Денис) → прогноз (src/serve.py).

- Станции: русские названия external_data ↔ station_id — колонка name_external в data/reference/spb_line1_stations.csv.
- Погода: почасовой прогноз Яндекса → колонка precipitation той же таблицы, что исторический прогноз Open-Meteo,
  на котором обучена модель. Признаки fc_precip_tau и fc_precip_3h_tau из неё считает замороженный src/features.py —
  тем же кодом, что при обучении.
- Поезда и события в модель не идут (она на них не обучена) — только контекст для LLM-слоя в explanations.

Сдвиг часа. Open-Meteo метит осадки концом часа (сумма за предыдущий час), Яндекс — началом (ForecastHour.time,
accumulatedPrec — осадки «в этот час»): метка Open-Meteo = время Яндекса + 1 ч. Признак слота τ — осадки за
[τ − 1 ч, τ), сумма за 3 ч — за [τ − 3 ч, τ). Слотам t0 + 1 и t0 + 2 нужны часы Яндекса t0 − 2 … t0 + 1; к моменту
прогноза они уже начались, поэтому прогноз Яндекса берётся с past_hours.
"""
import os
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta

import numpy as np
import pandas as pd

from external_data.calendar.models import moscow_time
from external_data.events.models import Event
from external_data.locations import GeocoderError, LocationResolution, LocationResolutionError, resolve_location_to_line1
from external_data.railway import HUBS, RailwayHub, YandexRaspError, YandexRaspProvider, build_railway_features
from external_data.weather import WeatherObservation, YandexWeatherError, YandexWeatherProvider

from src import config, reference
from src import features as F

WEATHER_KEY_ENV = "YANDEX_WEATHER_API_KEY"
RASP_KEY_ENV = "YANDEX_RASP_API_KEY"
YANDEX_TO_OPENMETEO = pd.Timedelta(hours=1)   # час Яндекса [T, T + 1 ч) — метка Open-Meteo T + 1 ч
HORIZONS = (1, 2)                             # слоты t0 + 1 ч и t0 + 2 ч
PRECIP_WINDOW = 3                             # fc_precip_3h_tau: rolling(3) в features.tau_arrays
PAST_HOURS, FUTURE_HOURS = 6, 3               # окно запроса к Яндексу от serverTime, с запасом
RAILWAY_WINDOW = timedelta(minutes=120)       # самое длинное окно RailwayFeatures
EVENT_RADIUS_M = 1500                         # событие «рядом» со станцией — эвристика контекста, не признак
EVENT_WINDOW = pd.Timedelta(hours=3)          # слот τ в [начало − 3 ч, конец (или начало) + 3 ч]
EVENT_FIELDS = {"event_name", "event_type", "start_time", "end_time", "location_name", "expected_people", "source_url"}

LocatedEvent = tuple[Event, LocationResolution]


# --- Станции ---------------------------------------------------------------------
def station_by_external() -> dict[str, str]:
    """Название станции в external_data → station_id."""
    st = reference.load_stations()
    return dict(zip(st.name_external, st.station_id))


def external_name(station_id: str) -> str:
    st = reference.load_stations()
    return dict(zip(st.station_id, st.name_external))[station_id]


def hubs_by_station() -> dict[str, RailwayHub]:
    """station_id → вокзал рядом (4 станции)."""
    ids = station_by_external()
    return {ids[h.metro_station]: h for h in HUBS}


def slot_ts(tau: pd.Timestamp) -> str:
    """Слот как в контракте: местное время с поясом."""
    return pd.Timestamp(tau).tz_convert(config.SPB_TZ).isoformat()


# --- Погода ----------------------------------------------------------------------
def precip_hours(t0_utc) -> pd.DatetimeIndex:
    """Метки Open-Meteo (UTC), из которых считаются признаки погоды слотов t0 + 1 и t0 + 2: t0 − 1 … t0 + 2."""
    t0 = pd.Timestamp(t0_utc)
    return pd.date_range(t0 + pd.Timedelta(hours=HORIZONS[0] - PRECIP_WINDOW + 1), t0 + pd.Timedelta(hours=HORIZONS[-1]),
                         freq="h")


def yandex_precipitation(observations: Iterable[WeatherObservation]) -> pd.Series:
    """Осадки Яндекса, мм, по меткам Open-Meteo (UTC). Часы без значения (None) пропускаются."""
    obs = [o for o in observations if o.precipitation is not None]
    idx = pd.to_datetime([o.timestamp for o in obs], utc=True) + YANDEX_TO_OPENMETEO
    return pd.Series([o.precipitation for o in obs], index=idx, dtype=float, name="precipitation").sort_index()


def fetch_yandex(provider=None) -> list[WeatherObservation]:
    """Почасовой прогноз Яндекса в точке Open-Meteo (config.SPB_COORDS), включая уже начавшиеся часы сегодня.
    provider — объект с get_hourly_forecast; по умолчанию YandexWeatherProvider (ключ из окружения)."""
    own = provider is None
    if own:
        provider = YandexWeatherProvider()
    try:
        return provider.get_hourly_forecast(*config.SPB_COORDS, hours=FUTURE_HOURS, past_hours=PAST_HOURS)
    finally:
        if own:
            provider.close()


def _hours_text(hours: pd.DatetimeIndex) -> str:
    """Часы осадков словами: метка Open-Meteo T — час [T − 1 ч, T)."""
    return ", ".join(f"{t - YANDEX_TO_OPENMETEO:%H}–{t:%H}" for t in hours.tz_convert(config.SPB_TZ))


def weather_table(om: pd.DataFrame, t0_utc, source: str = "openmeteo", provider=None) -> tuple[pd.DataFrame, dict]:
    """Таблица прогноза погоды для признаков модели (колонки как у Open-Meteo) и сведения для meta.

    openmeteo — таблица как есть. yandex — в копии осадки нужных часов (precip_hours) заменены прогнозом Яндекса;
    чего Яндекс не дал — Open-Meteo, нет нигде — NaN (признак «дождь» тогда 0, сумма за 3 ч пуста, как в
    features.tau_arrays). Нет ключа или ошибка API — всё из Open-Meteo с предупреждением.
    meta: source (openmeteo | yandex | yandex+openmeteo), warning, hours — источник осадков по часам."""
    if source not in ("openmeteo", "yandex"):
        raise ValueError(f"источник погоды: openmeteo или yandex, а не {source!r}")
    hours = precip_hours(t0_utc)
    om_p = om.set_index("ts_utc").precipitation.reindex(hours)
    yx = pd.Series(np.nan, index=hours)
    warnings = []
    if source == "yandex":
        if provider is None and not os.environ.get(WEATHER_KEY_ENV, "").strip():
            warnings.append(f"нет ключа {WEATHER_KEY_ENV} — вместо Яндекса Open-Meteo")
        else:
            try:
                yx = yandex_precipitation(fetch_yandex(provider)).reindex(hours)
            except YandexWeatherError as e:
                warnings.append(f"{e} Вместо Яндекса — Open-Meteo")
            else:
                if yx.isna().any():
                    warnings.append(f"Яндекс не дал прогноз на часы {_hours_text(hours[yx.isna()])} — на них Open-Meteo")
    from_y = yx.notna().to_numpy()
    precip = yx.where(from_y, om_p)
    if precip.isna().any():
        warnings.append(f"нет прогноза осадков на часы {_hours_text(hours[precip.isna()])}")
    src = np.where(from_y, "yandex", np.where(om_p.notna(), "openmeteo", "нет данных"))
    info = {"source": "yandex" if from_y.all() else "yandex+openmeteo" if from_y.any() else "openmeteo",
            "warning": "; ".join(warnings) or None,
            "hours": {slot_ts(t - YANDEX_TO_OPENMETEO): s for t, s in zip(hours, src)}}
    if not from_y.any():
        return om, info
    tab = om.set_index("ts_utc")
    tab = tab.reindex(tab.index.union(hours)).rename_axis("ts_utc")
    tab.loc[hours, "precipitation"] = precip.to_numpy()
    if "ts_local" in tab:
        tab["ts_local"] = tab.index.tz_convert(config.SPB_TZ)
    return tab.reset_index(), info


def weather_features(precip: pd.Series, taus: Iterable) -> pd.DataFrame:
    """fc_precip_tau и fc_precip_3h_tau по осадкам с метками Open-Meteo — формулы features.tau_arrays:
    дождь ≥ PRECIP_MM мм (пропуск — не дождь), сумма за τ − 2 … τ (пуста, если хоть одного часа нет)."""
    rows = []
    for tau in taus:
        tau = pd.Timestamp(tau)
        win = precip.reindex(pd.date_range(tau - pd.Timedelta(hours=PRECIP_WINDOW - 1), tau, freq="h"))
        rows.append({"ts": slot_ts(tau), "fc_precip_tau": float(win.iloc[-1] >= F.PRECIP_MM),
                     "fc_precip_3h_tau": float(win.sum()) if win.notna().all() else np.nan})
    return pd.DataFrame(rows)


# --- Контекст для LLM-слоя -----------------------------------------------------------
def railway_context(taus: Iterable, provider) -> dict[tuple[str, str], dict]:
    """(station_id, слот) → прибытия поездов к вокзалу у станции на начало слота: RailwayFeatures без станции.
    arrivals_next_60m — прибытия за час слота. Расписание грузится один раз на (вокзал, дату)."""
    cache, out = {}, {}
    for sid, hub in hubs_by_station().items():
        for tau in taus:
            local = moscow_time(pd.Timestamp(tau).to_pydatetime())
            day, last, arrivals = local.date(), (local + RAILWAY_WINDOW).date(), []
            while day <= last:
                key = (hub.rasp_station_code, day)
                if key not in cache:
                    cache[key] = provider.get_arrivals(*key)
                arrivals += cache[key]
                day += timedelta(days=1)
            feats = build_railway_features(local, hub, arrivals)
            out[(sid, slot_ts(tau))] = feats.model_dump(mode="json", exclude={"metro_station"})
    return out


def locate_events(events: Iterable[Event], geocoder, top_k: int = 3) -> tuple[list[LocatedEvent], list[str]]:
    """Ближайшие станции событий (resolve_location_to_line1 из external_data): по адресу, иначе по площадке.
    Событие без площадки или не найденное геокодером пропускается — причина во втором списке."""
    located, skipped = [], []
    for ev in events:
        if not ev.location_name:
            skipped.append(f"{ev.event_name}: нет площадки")
            continue
        try:
            located.append((ev, resolve_location_to_line1(ev.location_name, geocoder, top_k=top_k, address=ev.address)))
        except (LocationResolutionError, GeocoderError) as e:
            skipped.append(f"{ev.event_name}: {e}")
    return located, skipped


def events_context(taus: Iterable, located: Iterable[LocatedEvent]) -> dict[tuple[str, str], list[dict]]:
    """(station_id, слот) → события рядом: станция среди ближайших не дальше EVENT_RADIUS_M, слот τ — от начала
    события − 3 ч до его конца (или начала) + 3 ч. События без времени начала не берутся."""
    ids, taus = station_by_external(), [pd.Timestamp(t) for t in taus]
    out = defaultdict(list)
    for ev, res in located:
        if ev.start_time is None:
            continue
        start, end = pd.Timestamp(ev.start_time), pd.Timestamp(ev.end_time or ev.start_time)
        item = ev.model_dump(mode="json", include=EVENT_FIELDS)
        for near in res.nearest_stations:
            if near.distance_m > EVENT_RADIUS_M:
                continue
            for tau in taus:
                if start - EVENT_WINDOW <= tau <= end + EVENT_WINDOW:
                    out[(ids[near.station_name], slot_ts(tau))].append({**item, "distance_m": round(near.distance_m)})
    return dict(out)


def add_context(explanations: list[dict], railway_provider=None, events: Iterable[LocatedEvent] | None = None,
                live: bool = False) -> dict:
    """Каждой записи explanations — context: {railway: прибытия или None, events: [...]}. Без провайдера поездов
    Rasp подключается сам только для «сейчас» (live) при ключе. Ошибки источников прогноз не роняют.
    Возвращает сведения для meta."""
    taus = sorted({pd.Timestamp(e["ts"]) for e in explanations})
    warnings, rail, ev = [], {}, {}
    own = railway_provider is None and live and bool(os.environ.get(RASP_KEY_ENV, "").strip())
    if own:
        railway_provider = YandexRaspProvider()
    if railway_provider is not None:
        try:
            rail = railway_context(taus, railway_provider)
        except (YandexRaspError, ValueError) as e:
            warnings.append(f"расписания поездов: {e}")
        finally:
            if own:
                railway_provider.close()
    events = list(events or [])
    if events:
        ev = events_context(taus, events)
    for e in explanations:
        key = (e["station_id"], e["ts"])
        e["context"] = {"railway": rail.get(key), "events": ev.get(key, [])}
    return {"railway": type(railway_provider).__name__ if railway_provider is not None else None,
            "events": len(events), "warning": "; ".join(warnings) or None}
