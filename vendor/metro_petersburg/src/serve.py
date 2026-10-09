"""Прогноз «как будто сейчас» для демо и Streamlit (этап 5).

Запуск:
  python -m src.serve --train                         # замороженная конфигурация на всех данных до 29.09 → models/lgbm_final/
  python -m src.serve --train --holdout               # то же по 24.08 → models/lgbm_holdout/ (модель финального теста
                                                      # без повторного теста: те же данные — те же деревья)
  python -m src.serve --now "2026-08-31 08:59"        # прогноз на t+1 и t+2 по данным до now
  python -m src.serve --now "2026-06-27 20:59" --model auto --json out.json

forecast(now) берёт только данные до конца часа t0 = floor(now + 1 мин) − 1 ч (час hh считается закрытым в hh:59):
поток после t0 становится NaN, флаги инцидента после t0 снимаются, нормы и признаки пересчитываются заново.
Модель по умолчанию (model="auto") — самая свежая из сохранённых, у которой конец обучения + 7 суток раньше now:
модели фолдов (models/lgbm_fold_*), финального теста (lgbm_holdout) и финальная (lgbm_final). Если такой нет —
lgbm_final с предупреждением «модель видела этот период».

Погода (weather_source): auto — для «сейчас» (now в пределах часа от текущего времени) прогноз Яндекса, если задан
YANDEX_WEATHER_API_KEY, иначе — исторический прогноз Open-Meteo, как при обучении; openmeteo и yandex — явно.
Источник по часам и предупреждения — в meta["weather"]. Поезда и события (external_data) в модель не идут —
только контекст для LLM-слоя: explanations[…]["context"] (src/external_adapter.py).
"""
import argparse
import functools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src import backtest as BT
from src import config, contract
from src import eda_spb as E
from src import external_adapter as A
from src import features as F
from src import model as M

FULL = "lgbm_final"
HOLDOUT = "lgbm_holdout"
GAP = pd.Timedelta(days=BT.GAP_DAYS)
WARNING = "модель видела этот период: прогноз не вне выборки"
WEATHER_SOURCES = ("auto", "openmeteo", "yandex")
LIVE = pd.Timedelta(hours=1)     # now ближе часа к текущему времени — прогноз «сейчас»


class Forecast(NamedTuple):
    records: list[dict]          # контракт: станция × слот × горизонт
    explanations: list[dict]     # топ-3 причины на каждую запись
    meta: dict                   # какая модель, до какой даты обучена, вне выборки ли прогноз


def _local(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(config.SPB_TZ).tz_localize(None) if ts.tzinfo is not None else ts


def _wall_clock() -> pd.Timestamp:
    """Текущее местное время СПб (тесты подменяют)."""
    return pd.Timestamp.now(tz=config.SPB_TZ).tz_localize(None)


def is_live(now) -> bool:
    """Прогноз «сейчас», а не повтор прошлого: now в пределах часа от текущего времени."""
    return abs(_local(now) - _wall_clock()) < LIVE


def origin(now) -> pd.Timestamp:
    """Час t0 (местное время): последний закрытый час; час hh считается закрытым в hh:59."""
    return (_local(now) + pd.Timedelta(minutes=1)).floor("h") - pd.Timedelta(hours=1)


def bundles(models_dir: Path = M.MODELS) -> dict[str, dict]:
    """Сохранённые модели: имя папки → meta.json."""
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(models_dir).glob("lgbm_*/meta.json"))}


def choose_model(now, models_dir: Path = M.MODELS, model: str = "auto") -> tuple[str, bool, str | None]:
    """(модель, вне выборки ли прогноз, предупреждение). auto — самая свежая модель с train_end + 7 суток < now;
    нет такой — lgbm_final с предупреждением."""
    now = _local(now)
    found = bundles(models_dir)
    oos = {n: pd.Timestamp(m["train_end"]) + GAP < now for n, m in found.items()}
    if model != "auto":
        if model not in found:
            raise FileNotFoundError(f"нет модели {model} в {models_dir}")
        return model, oos[model], None if oos[model] else WARNING
    ok = [n for n in found if oos[n]]
    if ok:
        return max(ok, key=lambda n: found[n]["train_end"]), True, None
    if FULL not in found:
        raise FileNotFoundError(f"нет подходящей модели в {models_dir} — python -m src.serve --train")
    return FULL, False, WARNING


@functools.lru_cache(maxsize=1)
def _data() -> E.SpbData:
    return E.load()


@functools.lru_cache(maxsize=8)
def _bundle(path: str):
    return M.load_bundle(Path(path))


def cut_data(d: E.SpbData, t0_utc: pd.Timestamp) -> E.SpbData:
    """Данные на момент конца часа t0: поток после t0 → NaN, будущие флаги инцидента сняты, длинная таблица — до t0."""
    g = d.grid
    after = (g.ts > t0_utc)[:, None]
    regular = d.calendar.set_index("date").is_regular.reindex(g.sday).eq(True).to_numpy()
    grid = E.make_grid(g.ts, np.where(after, np.nan, g.entries), g.closed, np.where(after, g.closed, g.flagged), regular)
    return replace(d, grid=grid, df=d.df[d.df.ts_utc <= t0_utc].reset_index(drop=True))


def _weather_features(rows: pd.DataFrame) -> list[dict]:
    """Признаки погоды, которые видела модель, по слотам (у всех вестибюлей одинаковы); мм — до 0,001."""
    w = rows.drop_duplicates("tau").sort_values("tau")
    return [{"ts": A.slot_ts(tau), "fc_precip_tau": float(p),
             "fc_precip_3h_tau": None if math.isnan(p3) else round(float(p3), 3)}
            for tau, p, p3 in zip(w.tau, w.fc_precip_tau, w.fc_precip_3h_tau)]


def forecast(now, model: str = "auto", models_dir: Path = M.MODELS, data: E.SpbData | None = None,
             weather_source: str = "auto", weather_provider=None, railway_provider=None,
             events: list[A.LocatedEvent] | None = None) -> Forecast:
    """Прогноз по контракту на 18 станций, горизонты 60 и 120 мин, и топ-3 причины — по данным только до now.

    weather_source: auto | openmeteo | yandex (см. описание модуля); weather_provider — объект с get_hourly_forecast
    вместо YandexWeatherProvider. railway_provider (get_arrivals) и events (A.locate_events) — контекст для LLM-слоя;
    без провайдера поездов Rasp подключается сам только для «сейчас» при YANDEX_RASP_API_KEY."""
    if weather_source not in WEATHER_SOURCES:
        raise ValueError(f"weather_source: {' | '.join(WEATHER_SOURCES)}, а не {weather_source!r}")
    t0 = origin(now)
    t0_utc = t0.tz_localize(config.SPB_TZ).tz_convert("UTC")
    d = data if data is not None else _data()
    if t0_utc not in d.grid.ts or t0 < BT.TRAIN_START:
        raise ValueError(f"now = {now}: вне периода данных ({BT.TRAIN_START:%d.%m.%Y} – {d.grid.local[-1]:%d.%m.%Y})")
    name, oos, warning = choose_model(now, models_dir, model)
    m, st_q, meta = _bundle(str(Path(models_dir) / name))
    live = is_live(now)
    source = ("yandex" if live else "openmeteo") if weather_source == "auto" else weather_source
    weather_fc, weather = A.weather_table(d.forecast, t0_utc, source, weather_provider)
    d = replace(d, forecast=weather_fc)
    panel = BT.build_panel(cut_data(d, t0_utc), final=True)
    panel = replace(panel, thresholds=pd.DataFrame(meta["thresholds"]))
    rows = BT.make_rows(panel, for_export=True)
    rows = F.add_features(rows[rows.t == t0_utc], panel)
    records, expl, _ = M.station_forecast(m, st_q, rows, panel, meta["anomaly_rule"],
                                          f"{meta['model_version']}/{name}")
    context = A.add_context(expl, railway_provider, events, live)
    info = {"model": name, "model_version": meta["model_version"], "train_start": meta["train_start"],
            "train_end": meta["train_end"], "out_of_sample": oos, "warning": warning,
            "now": str(_local(now)), "t0": t0.tz_localize(config.SPB_TZ).isoformat(),
            "config_sha256": meta.get("config_sha256"),
            "weather": {"requested": weather_source, **weather, "features": _weather_features(rows)},
            "context": context}
    return Forecast(records, expl, info)


def train_full(cfg: dict | None = None, models_dir: Path = M.MODELS, end: pd.Timestamp = BT.FINAL_END,
               name: str = FULL) -> Path:
    """Замороженная конфигурация на всех сутках до end включительно (без особых дней и флагов) → бандл."""
    if cfg is None:
        cfg = M.load_config()
        M.check_frozen(cfg)
    panel = M.frozen_panel(cfg, final=True)
    if not cfg.get("thresholds"):
        cfg = {**cfg, "thresholds": json.loads(panel.thresholds.to_json(orient="records", force_ascii=False))}
    rows = F.add_features(BT.make_rows(panel), panel)
    train = rows[(rows.sday <= end) & ~rows.is_special & rows.y.notna()]
    m = M.model_from_cfg(cfg, name).fit(train)
    path = Path(models_dir) / name
    M.save_bundle(path, m, M.station_calibration(m, train, panel), cfg, train, name)
    return path


def table(fc: Forecast, stations: list[str] | None = None) -> pd.DataFrame:
    """Записи и причины одной таблицей (для печати и Streamlit)."""
    df = pd.DataFrame(fc.records)
    why = {(e["station_id"], e["ts"], e["horizon_min"]): e["reasons"] for e in fc.explanations}
    df["reasons"] = ["; ".join(f"{r['text']} ({r['effect_pct']:+.0f} %)" for r in why[(s, t, h)])
                     for s, t, h in zip(df.station_id, df.ts, df.horizon_min)]
    if stations:
        df = df[df.station_id.isin(stations)]
    return df


def _header(fc: Forecast) -> str:
    m = fc.meta
    span = f"{pd.Timestamp(m['train_start']):%d.%m}–{pd.Timestamp(m['train_end']):%d.%m.%Y}"
    oos = "вне выборки" if m["out_of_sample"] else f"НЕ вне выборки — {m['warning']}"
    t0 = pd.Timestamp(m["t0"])
    w = m["weather"]
    weather = {"openmeteo": "Open-Meteo", "yandex": "Яндекс", "yandex+openmeteo": "Яндекс и Open-Meteo"}[w["source"]]
    return (f"Прогноз по данным до {m['now']} (последний закрытый час — {t0:%d.%m %H}:00)\n"
            f"Модель: {m['model']} (обучена {span}), {oos}\n"
            f"Погода: {weather}" + (f"\nПредупреждение: {w['warning']}" if w["warning"] else ""))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="обучить финальную модель на всех данных до 29.09")
    ap.add_argument("--holdout", action="store_true", help="с --train: обучение по 24.08 → models/lgbm_holdout/")
    ap.add_argument("--now", help="момент прогноза, местное время: «2026-08-31 08:59»")
    ap.add_argument("--model", default="auto", help="auto (по умолчанию) или имя папки в models/")
    ap.add_argument("--weather", default="auto", choices=WEATHER_SOURCES,
                    help="погода: auto (по умолчанию: «сейчас» — Яндекс, прошлое — Open-Meteo), openmeteo, yandex")
    ap.add_argument("--stations", help="станции через запятую (для печати)")
    ap.add_argument("--json", help="записать записи, причины и meta в файл")
    args = ap.parse_args(argv)
    if args.train:
        last = BT.get_fold("final", final=True).train_end - pd.Timedelta(days=1)
        path = train_full(end=last, name=HOLDOUT) if args.holdout else train_full()
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        print(f"{path.relative_to(config.ROOT)}: обучение {meta['train_start']} – {meta['train_end']}, "
              f"деревьев {meta['best_iter']}")
    if args.now:
        fc = forecast(args.now, args.model, weather_source=args.weather)
        contract.validate_records(fc.records)
        print(_header(fc))
        df = table(fc, args.stations.split(",") if args.stations else None)
        with pd.option_context("display.max_colwidth", 200, "display.width", 250):
            print(df[["station_id", "ts", "horizon_min", "q10", "q50", "q90", "baseline", "is_anomaly",
                      "reasons"]].to_string(index=False))
        if args.json:
            Path(args.json).write_text(json.dumps(fc._asdict(), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
