"""Стекинг: прогноз каждые 30 минут (этап 7, ч. 2).

Запуск:
  python -m src.stack --fit [--reuse]   # базовые модели мая и июля вне выборки, выбор сложности май ↔ июль,
                                        # веса на мае + июле → reports/stack/stack_params.json, таблицы cross_*
  python -m src.stack --september       # один прогон на сентябре по замороженным параметрам; часовые результаты по
                                        # сентябрю уже известны (финальный тест этапа 5) — чистой отложенной выборкой
                                        # он не является; демо-вечер; data/predictions/stack_sep.json по контракту
  python -m src.stack --figures         # графики reports/figures/spb_17_*
  python -m src.stack --tables          # производные таблицы (by_minute) из кэша прогнозов, без переобучения

--reuse берёт прогнозы GRU из кэша data/interim/stack_preds/, если ключ (параметры, окно, код) совпал.

Окна базовых моделей (TRAIN_MONTHS): B3 обучается и квантили остатка B2 считаются на месяцах строго до тестового —
фев → май, фев + май → июль, фев + май + июль → сентябрь. B1 — модель месяца вне выборки (base.OOS_MODELS).
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src import config
from src import model as M
from src.stack import base as B
from src.stack import data as D
from src.stack import evaluate as V
from src.stack import meta as MT
from src.stack import nn as NN

OUT = config.ROOT / "reports" / "stack"
CACHE = config.INTERIM / "stack_preds"
PARAMS_JSON = OUT / "stack_params.json"
SEP_OUT = config.PREDICTIONS / "stack_sep.json"
TRAIN_MONTHS = {5: (2,), 7: (2, 5), 9: (2, 5, 7)}
CACHE_VERSION = 1
PERIODS = {5: "май (стекинг обучен на июле)", 7: "июль (стекинг обучен на мае)"}
CROSS = "май + июль (вне выборки)"
SEPT = "сентябрь"


def log(*a) -> None:
    print(*a, flush=True)


def _key(month: int, n_train: int) -> str:
    spec = {"v": CACHE_VERSION, "params": NN.PARAMS, "train": TRAIN_MONTHS[month], "n_train": n_train,
            "code": hashlib.sha256((config.ROOT / "src" / "stack" / "nn.py").read_bytes()).hexdigest()}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def gru(sd: D.StackData, month: int, rows: pd.DataFrame, reuse: bool) -> tuple[pd.DataFrame, dict]:
    """B3 для месяца month: обучение на TRAIN_MONTHS[month], прогноз строк rows. Кэш — по ключу."""
    train = sd.rows[sd.rows.month.isin(TRAIN_MONTHS[month])]
    s_tr, s_te = NN.samples(sd.q, train), NN.samples(sd.q, rows)
    key = _key(month, len(s_tr.keys))
    path, info_path = CACHE / f"b3_{month:02d}_{key}.npy", CACHE / f"b3_{month:02d}_{key}.json"
    if reuse and path.exists() and info_path.exists():
        z, info = np.load(path), json.loads(info_path.read_text(encoding="utf-8"))
    else:
        m = NN.GRUQuantile(len(sd.q.vestibules)).fit(s_tr)
        z, info = m.predict(s_te), m.info
        m.save(M.MODELS / f"stack_gru_2026-{month:02d}")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, z)
        info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return NN.to_rows(s_te, z, rows), {"month": month, **info}


def base_table(sd: D.StackData, month: int, hourly_rows: pd.DataFrame, reuse: bool) -> tuple[pd.DataFrame, dict]:
    """Строки месяца с прогнозами B1, B2, B3 и простым средним (всё — в z, вне выборки)."""
    if max(TRAIN_MONTHS[month]) >= month:
        raise ValueError("окно обучения не раньше тестового месяца")
    rows = sd.rows[sd.rows.month == month]
    train = sd.rows[sd.rows.month.isin(TRAIN_MONTHS[month])]
    b1 = B.b1_slot(rows, B.b1_hourly(hourly_rows, month))
    b2, rq = B.b2(sd.q, train, rows)
    b3, info = gru(sd, month, rows, reuse)
    df = rows.join(b1).join(b2).join(b3)
    df = df.join(B.mean_of(df))
    log(f"  {D.MONTH_RU[month]}: строк {len(df)}, B1 — {df.b1_model.iloc[0]}, GRU — эпохи {info['epochs']}")
    return df, {"b2": rq, "gru": info}


def summary(metrics: pd.DataFrame, period: str) -> pd.DataFrame:
    m = metrics[(metrics.period == period) & (metrics.slice == "все слоты")]
    return m.pivot_table(index="label", columns="horizon", values="wape", sort=False).round(4)


def run_fit(reuse: bool = False) -> dict:
    sd = D.build()
    panel = B.hourly_panel()
    hr = B.hourly_rows(panel, MT.FIT_MONTHS)
    log("базовые модели вне выборки:")
    parts, infos = [], []
    for month in MT.FIT_MONTHS:
        df, info = base_table(sd, month, hr, reuse)
        parts.append(df)
        infos.append(info)
    base = pd.concat(parts)
    params, extra = MT.fit_params(base)
    cross = base.join(extra["cross"])
    sq = V.fit_station_quantiles(V.station_sum(cross))
    params["station_quantiles"] = sq
    params["b2_quantiles"] = {str(i["gru"]["month"]): i["b2"] for i in infos}
    params["sha256"] = MT._sha(params)
    OUT.mkdir(parents=True, exist_ok=True)
    MT.save_params(params, PARAMS_JSON)

    names = [*B.BASES, "mean", "stack"]
    mt = pd.concat([*(V.metrics_table(cross[cross.month == m], names, PERIODS[m]) for m in MT.FIT_MONTHS),
                    V.metrics_table(cross, names, CROSS)], ignore_index=True)
    dl = pd.concat([*(V.delta_table(cross[cross.month == m], PERIODS[m]) for m in MT.FIT_MONTHS),
                    V.delta_table(cross, CROSS)], ignore_index=True)
    fits = {**{k: v["weights"] for k, v in extra["cross_fits"].items()}, "май + июль": params["weights"]}
    mt.round(6).to_csv(OUT / "cross_metrics.csv", index=False)
    dl.round(6).to_csv(OUT / "cross_delta.csv", index=False)
    extra["selection"].round(6).to_csv(OUT / "selection.csv", index=False)
    V.minute_table(cross, CROSS).round(6).to_csv(OUT / "cross_by_minute.csv", index=False)
    V.weights_table(fits).round(4).to_csv(OUT / "weights.csv", index=False)
    pd.DataFrame([i["gru"] for i in infos]).to_csv(OUT / "gru_training.csv", index=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    cross.to_parquet(CACHE / "cross.parquet")
    log(f"вариант весов: {params['variant']} ({params['variant_label']})")
    log(extra["selection"].round(5).to_string(index=False))
    log(summary(mt, CROSS).to_string())
    return params


def run_september(force: bool = False) -> None:
    done = OUT / "september_metrics.csv"
    if done.exists() and not force:
        raise SystemExit("сентябрь уже проверен: reports/stack/september_metrics.csv существует")
    params = MT.load_params(PARAMS_JSON)
    cfg = M.load_config()
    sd = D.build()
    panel = B.hourly_panel()
    hr = B.hourly_rows(panel, (MT.CHECK_MONTH,))
    log("сентябрь — один прогон по замороженным параметрам:")
    df, info = base_table(sd, MT.CHECK_MONTH, hr, reuse=False)
    df = df.join(MT.predict(df, params))
    names = [*B.BASES, "mean", "stack"]
    mt = V.metrics_table(df, names, SEPT)
    dl = V.delta_table(df, SEPT)
    mt.round(6).to_csv(done, index=False)
    dl.round(6).to_csv(OUT / "september_delta.csv", index=False)
    pd.DataFrame([info["gru"]]).to_csv(OUT / "september_gru_training.csv", index=False)
    V.minute_table(df, SEPT).round(6).to_csv(OUT / "september_by_minute.csv", index=False)
    df.to_parquet(CACHE / "september.parquet")

    st = V.station_sum(df)
    st = st.join(V.station_interval(st, params["station_quantiles"]).drop(columns="q50"))
    records = V.station_records(st, cfg["anomaly_rule"], pd.DataFrame(cfg["thresholds"]), params["version"])
    n_ok = V.validate(records)
    SEP_OUT.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    day, station, by_day = V.demo_choice(sd.slots)
    demo = V.demo_table(df, day, station, params["station_quantiles"])
    demo.round(2).to_csv(OUT / f"demo_{day:%d%m}.csv", index=False)
    by_day.round(6).to_csv(OUT / "demo_choice.csv", index=False)
    (OUT / "september_run.json").write_text(json.dumps(
        {"params_sha256": params["sha256"], "rows": int(len(df)), "records": n_ok, "demo_day": str(day.date()),
         "demo_station": station}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(summary(mt, SEPT).to_string())
    log(f"контракт: {n_ok} записей; демо — {day:%d.%m}, {station}")


def run_tables() -> None:
    """Производные таблицы из сохранённых прогнозов (data/interim/stack_preds): модели не обучаются, параметры
    и прогон сентября не повторяются."""
    for name, period, out in (("cross", CROSS, "cross_by_minute.csv"), ("september", SEPT, "september_by_minute.csv")):
        path = CACHE / f"{name}.parquet"
        if path.exists():
            V.minute_table(pd.read_parquet(path), period).round(6).to_csv(OUT / out, index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true", help="подбор на мае и июле → stack_params.json")
    ap.add_argument("--september", action="store_true", help="проверка на сентябре (один раз)")
    ap.add_argument("--figures", action="store_true", help="графики spb_17_*")
    ap.add_argument("--tables", action="store_true", help="производные таблицы из кэша прогнозов")
    ap.add_argument("--reuse", action="store_true", help="прогнозы GRU из кэша")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.fit:
        run_fit(args.reuse)
    if args.september:
        run_september()
    if args.tables:
        run_tables()
    if args.figures:
        from src.stack import figures
        figures.make_all()


if __name__ == "__main__":
    main()
