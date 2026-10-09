import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from external_data.weather import WeatherObservation

from src import config, contract
from src import eda_spb as E
from src import external_adapter as A
from src import model as M
from src import serve as S


def test_origin_hour_closes_at_59():
    assert S.origin("2026-08-31 08:59") == pd.Timestamp("2026-08-31 08:00")
    assert S.origin("2026-08-31 08:30") == pd.Timestamp("2026-08-31 07:00")
    assert S.origin("2026-08-31 09:00") == pd.Timestamp("2026-08-31 08:00")
    assert S.origin(pd.Timestamp("2026-08-31 05:59", tz="UTC")) == pd.Timestamp("2026-08-31 08:00")   # 08:59 MSK


# --- Выбор модели -----------------------------------------------------------------
TRAIN_END = {"lgbm_fold_2026-05": "2026-04-23", "lgbm_fold_2026-06": "2026-05-24", "lgbm_fold_2026-07": "2026-06-23",
             "lgbm_fold_2026-08": "2026-07-24", "lgbm_holdout": "2026-08-24", "lgbm_final": "2026-09-29"}


@pytest.fixture()
def fake_models(tmp_path):
    for name, end in TRAIN_END.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text(json.dumps({"name": name, "train_end": end}))
    return tmp_path


@pytest.mark.parametrize("now, expected", [
    ("2026-06-27 20:59", "lgbm_fold_2026-06"),
    ("2026-08-31 08:59", "lgbm_holdout"),
    ("2026-08-30 12:00", "lgbm_fold_2026-08"),      # holdout обучен по 24.08: 7 суток ещё не прошли
    ("2026-05-01 00:00", "lgbm_fold_2026-05"),
    ("2026-10-07 09:00", "lgbm_final"),
])
def test_choose_model_auto(fake_models, now, expected):
    name, oos, warning = S.choose_model(now, fake_models)
    assert name == expected and oos and warning is None


def test_auto_never_takes_model_trained_after_now_minus_gap(fake_models):
    """Перебор моментов: auto не берёт модель с train_end ≥ now − 7 суток; иначе — lgbm_final с предупреждением."""
    for now in pd.date_range("2026-04-20", "2026-10-10", freq="7h"):
        name, oos, warning = S.choose_model(now, fake_models)
        end = pd.Timestamp(TRAIN_END[name])
        if warning is None:
            assert oos and end + pd.Timedelta(days=7) < now, (now, name)
        else:
            assert name == S.FULL and not oos
            assert all(pd.Timestamp(e) + pd.Timedelta(days=7) >= now for e in TRAIN_END.values()), now
    assert S.choose_model("2026-04-30 00:00", fake_models) == (S.FULL, False, S.WARNING)   # ровно 7 суток — рано


def test_explicit_model_warns_if_not_out_of_sample(fake_models):
    assert S.choose_model("2026-08-31 08:59", fake_models, "lgbm_final") == ("lgbm_final", False, S.WARNING)
    assert S.choose_model("2026-08-31 08:59", fake_models, "lgbm_holdout") == ("lgbm_holdout", True, None)


# --- Прогноз на реальных данных ------------------------------------------------------
@pytest.fixture(scope="module")
def spb():
    if not (config.SPB_HOURLY.exists() and config.CALENDAR_OUT.exists() and M.FINAL_JSON.exists()):
        pytest.skip("нет interim СПб или model_final.json")
    return E.load()


@pytest.fixture(scope="module")
def tiny(spb, tmp_path_factory):
    """Финальная конфигурация с 20 деревьями — модель-малютка для проверки serve."""
    cfg = {**M.load_config(), "max_rounds": 20, "es_rounds": 5}
    d = tmp_path_factory.mktemp("models")
    S.train_full(cfg, d)
    return d


def _future_garbage(d: E.SpbData, now: str) -> E.SpbData:
    """Поток после конца часа t0 заменён шумом, инцидент после t0 снят — на прогноз это влиять не должно."""
    t0 = S.origin(now).tz_localize(config.SPB_TZ).tz_convert("UTC")
    g = d.grid
    after = (g.ts > t0)[:, None]
    noise = np.random.default_rng(3).integers(0, 20_000, g.entries.shape).astype(float)
    regular = d.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    grid = E.make_grid(g.ts, np.where(after, noise, g.entries), g.closed, np.where(after, g.closed, g.flagged), regular)
    df = d.df.copy()
    df.loc[df.ts_utc > t0, ["entries", "y"]] = 1e6
    return replace(d, grid=grid, df=df)


@pytest.mark.parametrize("now", ["2026-08-31 08:59", "2026-07-08 13:59"])
def test_forecast_does_not_see_after_now(spb, tiny, now):
    full = S.forecast(now, models_dir=tiny, data=spb)
    alt = S.forecast(now, models_dir=tiny, data=_future_garbage(spb, now))
    assert full.records == alt.records
    assert full.explanations == alt.explanations
    assert full.meta == alt.meta


def test_forecast_passes_contract(spb, tiny):
    fc = S.forecast("2026-08-31 08:59", models_dir=tiny, data=spb)
    preds = contract.validate_records(fc.records)
    assert len(preds) == 18 * 2
    assert {p.horizon_min for p in preds} == {60, 120}
    assert {r["ts"] for r in fc.records} == {"2026-08-31T09:00:00+03:00", "2026-08-31T10:00:00+03:00"}
    assert len(fc.explanations) == 36 and all(len(e["reasons"]) == 3 for e in fc.explanations)
    assert fc.meta["model"] == "lgbm_final" and fc.meta["warning"] == S.WARNING   # малютка обучена по 29.09
    assert fc.meta["train_end"] == "2026-09-29"


# --- Погода Яндекса и контекст external_data ------------------------------------------
NOW = "2026-08-31 08:59"


class FakeYandex:
    """Вместо YandexWeatherProvider: час Яндекса (начало, UTC) → осадки, мм."""

    def __init__(self, precip: dict):
        self.precip, self.calls = precip, []

    def get_hourly_forecast(self, lat, lon, hours=2, *, past_hours=0):
        self.calls.append({"hours": hours, "past_hours": past_hours})
        return [WeatherObservation(timestamp=t.to_pydatetime(), latitude=lat, longitude=lon, precipitation=p)
                for t, p in self.precip.items()]


def _yandex_hours(now) -> pd.DatetimeIndex:
    t0 = S.origin(now).tz_localize(config.SPB_TZ).tz_convert("UTC")
    return A.precip_hours(t0) - A.YANDEX_TO_OPENMETEO


@pytest.fixture(scope="module")
def om_fc(spb, tiny):
    return S.forecast(NOW, models_dir=tiny, data=spb, weather_source="openmeteo")


def test_weather_auto_for_past_is_openmeteo(spb, tiny, om_fc):
    auto = S.forecast(NOW, models_dir=tiny, data=spb)
    assert auto.records == om_fc.records and auto.explanations == om_fc.explanations
    assert auto.meta["weather"]["requested"] == "auto" and auto.meta["weather"]["source"] == "openmeteo"
    assert set(auto.meta["weather"]["hours"].values()) == {"openmeteo"} and auto.meta["weather"]["warning"] is None


def test_yandex_with_openmeteo_values_gives_same_forecast(spb, tiny, om_fc):
    """Яндекс с теми же осадками, что у Open-Meteo, — тот же прогноз: погода идёт в модель тем же путём."""
    om = spb.forecast.set_index("ts_utc").precipitation
    fake = FakeYandex({t: float(om[t + A.YANDEX_TO_OPENMETEO]) for t in _yandex_hours(NOW)})
    fc = S.forecast(NOW, models_dir=tiny, data=spb, weather_source="yandex", weather_provider=fake)
    assert fc.records == om_fc.records
    assert fc.meta["weather"]["source"] == "yandex" and fc.meta["weather"]["warning"] is None
    assert fc.meta["weather"]["features"] == om_fc.meta["weather"]["features"]
    assert fake.calls == [{"hours": A.FUTURE_HOURS, "past_hours": A.PAST_HOURS}]


def test_yandex_rain_reaches_model_features(spb, tiny, om_fc):
    """Демо-дождь external_data (1,2 мм) в час t0: признак слота t0 + 1 — дождь, сумма за 3 ч у обоих слотов."""
    hours = _yandex_hours(NOW)
    precip = {t: 1.2 if t == hours[2] else 0.0 for t in hours}
    fc = S.forecast(NOW, models_dir=tiny, data=spb, weather_source="yandex", weather_provider=FakeYandex(precip))
    feats = fc.meta["weather"]["features"]
    assert [f["fc_precip_tau"] for f in feats] == [1.0, 0.0]
    assert [f["fc_precip_3h_tau"] for f in feats] == pytest.approx([1.2, 1.2])
    assert feats != om_fc.meta["weather"]["features"]
    own = A.weather_features(A.yandex_precipitation(FakeYandex(precip).get_hourly_forecast(*config.SPB_COORDS)),
                             [pd.Timestamp(f["ts"]) for f in feats])
    assert own.fc_precip_tau.tolist() == [f["fc_precip_tau"] for f in feats]
    assert own.fc_precip_3h_tau.tolist() == pytest.approx([f["fc_precip_3h_tau"] for f in feats])
    contract.validate_records(fc.records)


def test_yandex_without_key_falls_back_to_openmeteo(spb, tiny, om_fc, monkeypatch):
    monkeypatch.delenv(A.WEATHER_KEY_ENV, raising=False)
    monkeypatch.delenv(A.RASP_KEY_ENV, raising=False)
    fc = S.forecast(NOW, models_dir=tiny, data=spb, weather_source="yandex")
    assert fc.records == om_fc.records
    assert fc.meta["weather"]["source"] == "openmeteo" and A.WEATHER_KEY_ENV in fc.meta["weather"]["warning"]
    monkeypatch.setattr(S, "_wall_clock", lambda: pd.Timestamp(NOW))          # «сейчас» = now
    live = S.forecast(NOW, models_dir=tiny, data=spb)
    assert live.records == om_fc.records
    assert live.meta["weather"]["requested"] == "auto" and A.WEATHER_KEY_ENV in live.meta["weather"]["warning"]


def test_railway_context_in_explanations(spb, tiny, om_fc):
    from external_data.railway import FakeRailwayProvider, RailwayArrival
    day = pd.Timestamp(NOW).date()
    arrivals = {(h.rasp_station_code, day): [RailwayArrival(arrival_time="2026-08-31T09:20:00+03:00",
                                                            transport_type="suburban", station_code=h.rasp_station_code)]
                for h in A.hubs_by_station().values()}
    fc = S.forecast(NOW, models_dir=tiny, data=spb, railway_provider=FakeRailwayProvider(arrivals))
    assert fc.records == om_fc.records
    assert all(set(e["context"]) == {"railway", "events"} for e in fc.explanations)
    with_rail = {(e["station_id"], e["ts"]): e["context"]["railway"] for e in fc.explanations if e["context"]["railway"]}
    assert {s for s, _ in with_rail} == {"vosstaniya", "ploshchad_lenina", "baltiyskaya", "devyatkino"}
    assert with_rail[("vosstaniya", "2026-08-31T09:00:00+03:00")]["arrivals_next_60m"] == 1
    assert with_rail[("vosstaniya", "2026-08-31T10:00:00+03:00")]["arrivals_next_60m"] == 0
    assert fc.meta["context"] == {"railway": "FakeRailwayProvider", "events": 0, "warning": None}
    assert all(e["context"] == {"railway": None, "events": []} for e in om_fc.explanations)
