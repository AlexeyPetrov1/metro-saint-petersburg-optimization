"""Переходник external_data → прогноз на демонстрационных данных external_data: без сети и ключей."""
import json
from datetime import date, datetime

import httpx
import numpy as np
import pandas as pd
import pytest

from external_data.events.models import Event
from external_data.locations import LINE1_STATIONS, FakeGeocoderProvider, GeocodedLocation, clear_station_cache
from external_data.railway import FakeRailwayProvider, RailwayArrival
from external_data.railway.yandex_rasp import YandexRaspError
from external_data.weather import WeatherObservation
from external_data.weather.tests.test_yandex_weather import KEY, forecast, hour   # его демо-ответ API погоды

from src import config, reference
from src import external_adapter as A

REAL_CLIENT = httpx.Client
MSK = config.SPB_TZ


def msk(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=MSK)


# --- Станции -------------------------------------------------------------------------
def test_stations_match_external_data():
    st = reference.load_stations()
    assert st.name_external.tolist() == list(LINE1_STATIONS)
    assert A.station_by_external()["Площадь Восстания"] == "vosstaniya"
    assert A.external_name("tekhnologichesky_institut") == "Технологический институт"
    hubs = {sid: h.railway_name for sid, h in A.hubs_by_station().items()}
    assert hubs == {"vosstaniya": "Московский вокзал", "ploshchad_lenina": "Финляндский вокзал",
                    "baltiyskaya": "Балтийский вокзал", "devyatkino": "Девяткино"}


# --- Погода ----------------------------------------------------------------------------
@pytest.fixture()
def yandex_api(monkeypatch):
    """Его YandexWeatherProvider без сети: HTTP-клиент с MockTransport (как в его тестах), фиктивный ключ."""
    def serve(payload, status=200):
        def factory(**kwargs):
            transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
            return REAL_CLIENT(**kwargs, transport=transport)
        monkeypatch.setenv(A.WEATHER_KEY_ENV, KEY)
        monkeypatch.setattr("external_data.weather.yandex_weather.httpx.Client", factory)
    return serve


def yandex_hours(precip: dict[int, float], day="2026-10-07"):
    """Его hour(): RAIN и прочее из демо; accumulatedPrec — по часам."""
    return [hour(f"{day}T{h:02d}:00:00+03:00", accumulatedPrec=p) for h, p in precip.items()]


def om_table(start: str, end: str, value: float = 0.3) -> pd.DataFrame:
    """Таблица в формате исторического прогноза Open-Meteo (data/interim/weather_hist_forecast_spb.parquet)."""
    ts = pd.date_range(msk(start), msk(end), freq="h").tz_convert("UTC")
    return pd.DataFrame({"ts_utc": ts, "ts_local": ts.tz_convert(MSK), "temperature_2m": 10.0, "precipitation": value})


T0 = msk("2026-10-07 11:00").tz_convert("UTC")      # now = 12:30, serverTime его демо-ответа


def test_yandex_hour_is_openmeteo_label_plus_hour(yandex_api):
    yandex_api(forecast(yandex_hours({h: h / 10 for h in range(5, 16)})))
    obs = A.fetch_yandex()
    assert [o.timestamp.hour for o in obs] == list(range(7, 16))          # serverTime 12:30 − 6 ч … + 3 ч
    p = A.yandex_precipitation(obs)
    assert p[msk("2026-10-07 13:00")] == pytest.approx(1.2)                # час 12–13 Яндекса → метка 13:00


def test_needed_hours():
    assert A.precip_hours(T0).tz_convert(MSK).hour.tolist() == [10, 11, 12, 13]   # метки t0 − 1 … t0 + 2


def test_weather_table_yandex(yandex_api):
    yandex_api(forecast(yandex_hours({8: 0.0, 9: 0.0, 10: 1.2, 11: 0.0, 12: 0.0, 13: 0.0})))
    om = om_table("2026-10-07 00:00", "2026-10-07 23:00")
    tab, info = A.weather_table(om, T0, "yandex")
    assert info["source"] == "yandex" and info["warning"] is None
    assert info["hours"] == {f"2026-10-07T{h:02d}:00:00+03:00": "yandex" for h in (9, 10, 11, 12)}
    p = tab.set_index("ts_utc").precipitation
    labels = A.precip_hours(T0)
    assert p[labels].tolist() == [0.0, 1.2, 0.0, 0.0]
    assert (p.drop(labels) == 0.3).all()                 # остальные часы — Open-Meteo
    assert (om.precipitation == 0.3).all()               # исходная таблица не тронута
    assert tab.columns.tolist() == om.columns.tolist() and len(tab) == len(om)


def test_weather_table_partial_yandex(yandex_api):
    yandex_api(forecast(yandex_hours({11: 2.0, 12: 0.0})))
    tab, info = A.weather_table(om_table("2026-10-07 00:00", "2026-10-07 23:00"), T0, "yandex")
    assert info["source"] == "yandex+openmeteo"
    assert list(info["hours"].values()) == ["openmeteo", "openmeteo", "yandex", "yandex"]
    assert "09–10, 10–11" in info["warning"]
    assert tab.set_index("ts_utc").precipitation[A.precip_hours(T0)].tolist() == [0.3, 0.3, 2.0, 0.0]


def test_weather_table_without_key_is_openmeteo(monkeypatch):
    monkeypatch.delenv(A.WEATHER_KEY_ENV, raising=False)
    om = om_table("2026-10-07 00:00", "2026-10-07 23:00")
    tab, info = A.weather_table(om, T0, "yandex")
    assert tab is om and info["source"] == "openmeteo"
    assert A.WEATHER_KEY_ENV in info["warning"]


def test_weather_table_api_error_is_openmeteo(yandex_api):
    yandex_api({"errors": [{"message": "Access denied"}]}, status=403)
    om = om_table("2026-10-07 00:00", "2026-10-07 23:00")
    tab, info = A.weather_table(om, T0, "yandex")
    assert tab is om and info["source"] == "openmeteo" and "403" in info["warning"]


def test_weather_table_no_data_anywhere(monkeypatch):
    monkeypatch.delenv(A.WEATHER_KEY_ENV, raising=False)
    tab, info = A.weather_table(om_table("2026-10-07 00:00", "2026-10-07 11:00"), T0, "openmeteo")
    assert info["warning"] == "нет прогноза осадков на часы 11–12, 12–13"
    assert list(info["hours"].values()) == ["openmeteo", "openmeteo", "нет данных", "нет данных"]


def test_weather_features_on_demo():
    """Демо из external_data/features/cli.py: 10.10 18:00, 1,2 мм, RAIN; соседние часы — без осадков."""
    obs = [WeatherObservation(timestamp=datetime.fromisoformat(f"2026-10-10T{h:02d}:00:00+03:00"),
                              latitude=59.9343, longitude=30.3351, temperature=6.0,
                              precipitation=1.2 if h == 18 else 0.0, precipitation_type="RAIN" if h == 18 else "NO_TYPE",
                              wind_speed=4.3, humidity=85, pressure=750, condition="RAIN" if h == 18 else "CLEAR")
           for h in range(14, 23)]
    f = A.weather_features(A.yandex_precipitation(obs), [msk(f"2026-10-10 {h}:00") for h in range(18, 23)])
    assert f.fc_precip_tau.tolist() == [0, 1, 0, 0, 0]               # дождь 18–19 — признак слота 19:00
    assert f.fc_precip_3h_tau.tolist() == pytest.approx([0, 1.2, 1.2, 1.2, 0])


def test_weather_features_threshold_and_gap():
    p = pd.Series([0.0, 0.0, 0.5, 0.49], index=pd.date_range(msk("2026-10-10 10:00"), periods=4, freq="h"))
    f = A.weather_features(p.drop(p.index[1]), [msk("2026-10-10 12:00"), msk("2026-10-10 13:00")])
    assert f.fc_precip_tau.tolist() == [1.0, 0.0]                    # ≥ 0,5 мм — дождь
    assert np.isnan(f.fc_precip_3h_tau).all()                       # часа 11:00 нет — суммы нет


# --- Поезда -----------------------------------------------------------------------------
MOSCOW_STATION = "s9602494"      # Московский вокзал → Площадь Восстания


def arrival(t: str, kind: str = "suburban") -> RailwayArrival:
    return RailwayArrival(arrival_time=t, transport_type=kind, station_code=MOSCOW_STATION, title=f"рейс {t[11:16]}")


@pytest.fixture()
def railway():
    """Расписание прибытий на Московский вокзал 10.10 (как в демо features/cli.py: ближайшее через 18 мин)."""
    day = [arrival("2026-10-10T18:18:00+03:00", "train"), arrival("2026-10-10T18:45:00+03:00"),
           arrival("2026-10-10T18:59:00+03:00"), arrival("2026-10-10T19:20:00+03:00", "train"),
           arrival("2026-10-10T19:50:00+03:00"), arrival("2026-10-10T21:00:00+03:00")]
    night = [arrival("2026-10-11T00:30:00+03:00")]
    return FakeRailwayProvider({(MOSCOW_STATION, date(2026, 10, 10)): day, (MOSCOW_STATION, date(2026, 10, 11)): night})


def test_railway_context(railway):
    ctx = A.railway_context([msk("2026-10-10 18:00"), msk("2026-10-10 19:00")], railway)
    at18 = ctx[("vosstaniya", "2026-10-10T18:00:00+03:00")]
    assert at18["railway_name"] == "Московский вокзал" and "metro_station" not in at18
    assert [at18[f"arrivals_next_{w}m"] for w in (15, 30, 60, 120)] == [0, 1, 3, 5]
    assert at18["minutes_to_next_arrival"] == 18 and at18["train_arrivals_next_30m"] == 1
    assert ctx[("vosstaniya", "2026-10-10T19:00:00+03:00")]["arrivals_next_60m"] == 2   # 19:20, 19:50
    assert {sid for sid, _ in ctx} == {"vosstaniya", "ploshchad_lenina", "baltiyskaya", "devyatkino"}
    assert ctx[("devyatkino", "2026-10-10T18:00:00+03:00")]["arrivals_next_120m"] == 0


def test_railway_context_loads_each_day_once(railway):
    ctx = A.railway_context([msk("2026-10-10 22:00"), msk("2026-10-10 23:00")], railway)
    assert ctx[("vosstaniya", "2026-10-10T23:00:00+03:00")]["arrivals_next_120m"] == 1    # 00:30 следующих суток
    calls = [c for c in railway.calls if c[0] == MOSCOW_STATION]
    assert calls == [(MOSCOW_STATION, date(2026, 10, 10)), (MOSCOW_STATION, date(2026, 10, 11))]


# --- События -----------------------------------------------------------------------------
# демо-координаты станций и площадок (приблизительные, не справочник)
STATION_POINTS = {
    "Девяткино": (60.0502, 30.4432), "Гражданский проспект": (60.0350, 30.4183), "Академическая": (60.0128, 30.3959),
    "Политехническая": (60.0089, 30.3709), "Площадь Мужества": (59.9998, 30.3663), "Лесная": (59.9849, 30.3442),
    "Выборгская": (59.9709, 30.3474), "Площадь Ленина": (59.9556, 30.3557), "Чернышевская": (59.9445, 30.3599),
    "Площадь Восстания": (59.9306, 30.3609), "Владимирская": (59.9276, 30.3479), "Пушкинская": (59.9207, 30.3296),
    "Технологический институт": (59.9165, 30.3184), "Балтийская": (59.9072, 30.2995), "Нарвская": (59.9012, 30.2747),
    "Кировский завод": (59.8797, 30.2618), "Автово": (59.8673, 30.2613), "Ленинский проспект": (59.8512, 30.2683),
    "Проспект Ветеранов": (59.8421, 30.2502),
}
BKZ_ADDRESS = "Лиговский проспект, 6, Санкт-Петербург"


@pytest.fixture()
def geocoder():
    def loc(query, lat, lon, address=None):
        return GeocodedLocation(latitude=lat, longitude=lon, query=query, formatted_address=address)
    results = {f"метро {n}, Санкт-Петербург": loc(f"метро {n}, Санкт-Петербург", *p) for n, p in STATION_POINTS.items()}
    results |= {"Газпром Арена, Санкт-Петербург": loc("Газпром Арена, Санкт-Петербург", 59.9730, 30.2205),
                "Ледовый дворец, Санкт-Петербург": loc("Ледовый дворец, Санкт-Петербург", 59.9218, 30.4668),
                BKZ_ADDRESS: loc(BKZ_ADDRESS, 59.9313, 30.3633, "Россия, Санкт-Петербург, Лиговский проспект, 6")}
    yield FakeGeocoderProvider(results)
    clear_station_cache()


@pytest.fixture()
def events():
    """Его examples/sample_events.json (ответы LLM) + концерт в БКЗ «Октябрьский» из примера в его README."""
    sample = json.loads((config.ROOT / "external_data/events/examples/sample_events.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: [Event.model_validate(e) for e in s["response"]["events"]] for s in sample}
    bkz = Event(event_name="Barcelona Flamenco Ballet", event_type="concert", start_time="2026-10-12T19:30:00+03:00",
                location_name="БКЗ Октябрьский", address="Лиговский проспект, 6")
    return {"match": by_id["today"][0], "no_time": by_id["date_without_time"][0],
            "no_place": by_id["explicit_concert_end"][0], "club": by_id["multiple"][0], "bkz": bkz}


def test_locate_events(geocoder, events):
    located, skipped = A.locate_events(events.values(), geocoder)
    assert [ev.location_name for ev, _ in located] == ["Газпром Арена", "Ледовый дворец", "БКЗ Октябрьский"]
    assert skipped[0] == "Концерт: нет площадки" and skipped[1].startswith("Концерт: ")   # «Клуб» не найден
    bkz = located[-1][1]
    assert bkz.nearest_station.station_name == "Площадь Восстания" and bkz.nearest_station.distance_m < 300


def test_events_context(geocoder, events):
    located, _ = A.locate_events(events.values(), geocoder)
    taus = [msk(f"2026-10-12 {h}:00") for h in (16, 17, 22, 23)] + [msk("2026-10-07 20:00")]
    ctx = A.events_context(taus, located)
    near = {k for k in ctx if k[0] == "vosstaniya"}
    assert near == {("vosstaniya", "2026-10-12T17:00:00+03:00"), ("vosstaniya", "2026-10-12T22:00:00+03:00")}
    item = ctx[("vosstaniya", "2026-10-12T17:00:00+03:00")][0]
    assert item["event_name"] == "Barcelona Flamenco Ballet" and item["distance_m"] < 300
    assert set(item) == A.EVENT_FIELDS | {"distance_m"}
    assert all(e["location_name"] == "БКЗ Октябрьский" for es in ctx.values() for e in es)   # матч далеко
    assert all(d <= A.EVENT_RADIUS_M for es in ctx.values() for d in [e["distance_m"] for e in es])


# --- context в explanations ------------------------------------------------------------
def stub_explanations(*keys):
    return [{"station_id": s, "ts": ts, "horizon_min": 60, "reasons": []} for s, ts in keys]


def test_add_context(railway, geocoder, events):
    located, _ = A.locate_events([events["bkz"]], geocoder)
    expl = stub_explanations(("vosstaniya", "2026-10-10T18:00:00+03:00"), ("narvskaya", "2026-10-10T18:00:00+03:00"),
                             ("vosstaniya", "2026-10-12T20:00:00+03:00"))
    meta = A.add_context(expl, railway, located)
    assert meta == {"railway": "FakeRailwayProvider", "events": 1, "warning": None}
    assert expl[0]["context"]["railway"]["arrivals_next_60m"] == 3 and expl[0]["context"]["events"] == []
    assert expl[1]["context"] == {"railway": None, "events": []}                  # у Нарвской нет вокзала
    assert expl[2]["context"]["events"][0]["event_name"] == "Barcelona Flamenco Ballet"


def test_add_context_without_sources(monkeypatch):
    monkeypatch.delenv(A.RASP_KEY_ENV, raising=False)
    expl = stub_explanations(("vosstaniya", "2026-10-10T18:00:00+03:00"))
    assert A.add_context(expl, live=True) == {"railway": None, "events": 0, "warning": None}
    assert expl[0]["context"] == {"railway": None, "events": []}


def test_add_context_survives_railway_error():
    class Broken:
        def get_arrivals(self, station_code, date):
            raise YandexRaspError("Yandex Rasp HTTP 429: превышен лимит запросов.")
    expl = stub_explanations(("vosstaniya", "2026-10-10T18:00:00+03:00"))
    meta = A.add_context(expl, Broken())
    assert "429" in meta["warning"] and expl[0]["context"]["railway"] is None
