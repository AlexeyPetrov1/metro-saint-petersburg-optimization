"""Очистка: data/raw → data/interim, всё выровнено по ts_utc.

Запуск: python -m src.clean

Формат MTA: station_id, ts_utc, ts_local, entries, transfers, is_missing, dst_flag —
полная сетка «станция × час» в UTC; нет строки в выгрузке → entries = NaN и is_missing, а не 0.

Переход на летнее время (America/New_York; transit_timestamp в MTA — местное время без смещения):
- март: часа 02:00 не существует, MTA его и не пишет; сетка в UTC непрерывна, дыры нет;
- ноябрь: 01:00 бывает дважды, а MTA пишет одну строку, в которую сложены оба часа.
  Её относим к первому 01:00 (EDT) с dst_flag="fall_merged"; второй 01:00 (EST) — NaN
  с dst_flag="fall_absorbed". Строки с dst_flag != "" исключаются из обучения и метрик (CLAUDE.md).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

DST_MERGED, DST_ABSORBED = "fall_merged", "fall_absorbed"
MTA_OUT = config.INTERIM / "mta_line7_hourly.parquet"


# --- MTA ------------------------------------------------------------------
def load_mta_raw(raw_dir: Path = config.RAW / "mta") -> pd.DataFrame:
    files = sorted(raw_dir.glob("line7_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет {raw_dir}/line7_*.parquet — сначала запустите python -m src.download")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def localize_mta(raw: pd.DataFrame, tz: str = config.TZ_LOCAL) -> pd.DataFrame:
    """Местное время без смещения → ts_utc. Неоднозначный ноябрьский час относим к первому вхождению (EDT)."""
    local = pd.to_datetime(raw["transit_timestamp"])
    # ambiguous=True: неоднозначное время считаем летним (первое вхождение); на остальные не влияет.
    # nonexistent="NaT" + своя проверка: тип исключения при "raise" разный в pandas 2 и 3
    ts = local.dt.tz_localize(tz, ambiguous=np.ones(len(local), dtype=bool), nonexistent="NaT")
    bad = ts.isna() & local.notna()
    if bad.any():
        raise ValueError("В MTA есть несуществующий местный час (весенний переход) — разберитесь руками: "
                         f"{sorted(local[bad].astype(str).unique())[:5]}")
    return pd.DataFrame({
        "station_id": raw["station_complex_id"].astype(str),
        "ts_utc": ts.dt.tz_convert("UTC"),
        "entries": raw["entries"].astype(float),
        "transfers": raw["transfers"].astype(float),
    })


def build_grid(station_ids: list[str], start_utc: pd.Timestamp, end_utc: pd.Timestamp,
               freq: str = "h", id_col: str = "station_id") -> pd.DataFrame:
    ts = pd.date_range(start_utc, end_utc, freq=freq, tz="UTC")
    idx = pd.MultiIndex.from_product([sorted(station_ids), ts], names=[id_col, "ts_utc"])
    return idx.to_frame(index=False)


def dst_flags(df: pd.DataFrame) -> np.ndarray:
    """По календарю: повторный местный час внутри станции — первое вхождение merged, второе absorbed.

    df отсортирован по (station_id, ts_utc), поэтому первое вхождение — летнее время (EDT).
    """
    key = pd.DataFrame({"s": df["station_id"], "n": df["ts_local"].dt.tz_localize(None)})
    repeated, second = key.duplicated(keep=False), key.duplicated(keep="first")
    return np.select([repeated & ~second, second], [DST_MERGED, DST_ABSORBED], "")


def clean_mta(raw: pd.DataFrame, station_ids: list[str],
              start_utc: pd.Timestamp = config.START_UTC, end_utc: pd.Timestamp = config.END_UTC,
              tz: str = config.TZ_LOCAL) -> pd.DataFrame:
    obs = localize_mta(raw, tz)
    obs = obs[obs.station_id.isin(station_ids)]
    dup = obs.duplicated(["station_id", "ts_utc"])
    if dup.any():
        raise ValueError(f"Дубликаты (station_id, ts_utc) в сырых данных: {obs[dup].head().to_dict('records')}")

    grid = build_grid(station_ids, start_utc, end_utc)
    outside = ~obs.set_index(["station_id", "ts_utc"]).index.isin(grid.set_index(["station_id", "ts_utc"]).index)
    if outside.any():
        raise ValueError(f"{outside.sum()} строк MTA вне сетки {start_utc} … {end_utc}")

    df = grid.merge(obs, on=["station_id", "ts_utc"], how="left", validate="one_to_one")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(tz)
    df["is_missing"] = df["entries"].isna()
    df["dst_flag"] = pd.Series(dst_flags(df), index=df.index, dtype=str)
    cols = ["station_id", "ts_utc", "ts_local", "entries", "transfers", "is_missing", "dst_flag"]
    return df[cols].sort_values(["station_id", "ts_utc"], ignore_index=True)


def build_stations(complexes: pd.DataFrame) -> pd.DataFrame:
    first = complexes.drop_duplicates("station_complex_id").set_index("station_complex_id")
    rows = [{
        "station_id": cid,
        "name": name,
        "line_order": i,
        "borough": first.at[cid, "borough"],
        "lat": float(first.at[cid, "lat"]),
        "lon": float(first.at[cid, "lon"]),
        "is_transfer_complex": cid in config.TRANSFER_COMPLEX_IDS,
    } for i, (cid, name) in enumerate(config.LINE7_COMPLEXES)]
    return pd.DataFrame(rows)


# --- Внешние таблицы --------------------------------------------------------
def clean_weather(raw: dict, start_utc: pd.Timestamp = config.START_UTC, end_utc: pd.Timestamp = config.END_UTC,
                  tz: str = config.TZ_LOCAL) -> pd.DataFrame:
    if raw.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Погода должна быть в GMT (timezone=GMT)")
    hourly = pd.DataFrame(raw["hourly"])
    hourly["ts_utc"] = pd.to_datetime(hourly.pop("time"), unit="s", utc=True)
    grid = pd.DataFrame({"ts_utc": pd.date_range(start_utc, end_utc, freq="h", tz="UTC")})
    df = grid.merge(hourly, on="ts_utc", how="left", validate="one_to_one")
    df.insert(1, "ts_local", df["ts_utc"].dt.tz_convert(tz))
    return df


def clean_mets(raws: list[dict]) -> pd.DataFrame:
    """Домашние матчи «Метс» на Citi Field (регулярка и плей-офф), включая отложенные — со статусом."""
    rows = []
    for raw in raws:
        for day in raw["dates"]:
            for g in day["games"]:
                if (g["venue"]["id"] != config.CITI_FIELD_VENUE_ID or g["gameType"] not in config.MLB_GAME_TYPES
                        or g["teams"]["home"]["team"]["id"] != config.METS_TEAM_ID):
                    continue
                info = g.get("gameInfo") or {}
                rows.append({
                    "game_pk": g["gamePk"],
                    "official_date": g["officialDate"],
                    "game_type": g["gameType"],
                    "status": g["status"]["detailedState"],
                    "sched_start_utc": g["gameDate"],
                    "first_pitch_utc": info.get("firstPitch"),
                    "duration_min": info.get("gameDurationMinutes"),
                    "delay_min": info.get("delayDurationMinutes"),
                    "attendance": info.get("attendance"),
                    "day_night": g.get("dayNight"),
                    "double_header": g.get("doubleHeader"),
                })
    df = pd.DataFrame(rows).drop_duplicates(["game_pk", "official_date", "status"])
    df["official_date"] = pd.to_datetime(df["official_date"]).dt.date
    for c in ("sched_start_utc", "first_pitch_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in ("duration_min", "delay_min", "attendance"):
        df[c] = df[c].astype("Float64")
    df["is_played"] = df["status"].str.startswith(("Final", "Completed Early", "Game Over"))
    # известно заранее — для признаков
    df["est_end_utc"] = df["sched_start_utc"] + config.GAME_EST_DURATION
    # известно только после матча — для EDA, не для признаков
    df["actual_end_utc"] = (df["first_pitch_utc"]
                            + pd.to_timedelta(df["duration_min"].astype(float), unit="min")
                            + pd.to_timedelta(df["delay_min"].fillna(0).astype(float), unit="min"))
    cols = ["game_pk", "official_date", "game_type", "status", "is_played", "sched_start_utc", "est_end_utc",
            "first_pitch_utc", "actual_end_utc", "duration_min", "delay_min", "attendance", "day_night", "double_header"]
    return df[cols].sort_values("sched_start_utc", ignore_index=True)


def clean_holidays(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates().sort_values(["date", "category"], ignore_index=True)


# --- Отчёт ------------------------------------------------------------------
def _longest_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max())


def report(df: pd.DataFrame, stations: pd.DataFrame, weather: dict[str, pd.DataFrame],
           mets: pd.DataFrame, hol: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    names = stations.set_index("station_id")["name"]

    print("\n=== Комплексы линии 7 ===")
    print(stations[["line_order", "station_id", "name", "borough", "is_transfer_complex"]].to_string(index=False))

    print("\n=== Таблица interim/mta_line7_hourly.parquet ===")
    per_station = df.groupby("station_id").size()
    print(f"строк: {len(df):,}  станций: {df.station_id.nunique()}  часов на станцию: {per_station.unique().tolist()}")
    print(f"ts_local: {df.ts_local.min()} … {df.ts_local.max()}")
    print(f"ts_utc:   {df.ts_utc.min()} … {df.ts_utc.max()}")
    regular = df[df.dst_flag == ""]
    print(f"пропусков всего: {df.is_missing.sum():,} ({df.is_missing.mean():.2%}), из них по DST (fall_absorbed): "
          f"{(df.dst_flag == DST_ABSORBED).sum()}; нулей (entries == 0): {(df.entries == 0).sum()}")

    print("\n=== Доля пропусков по станциям (без DST-часов) ===")
    hour = regular.ts_local.dt.hour
    rows = []
    for sid, g in regular.groupby("station_id"):
        m = g.is_missing
        h = hour[g.index]
        rows.append({
            "station_id": sid, "name": names[sid][:34],
            "missing": int(m.sum()), "share": m.mean(),
            "night_01_04": (m & h.between(1, 4)).sum() / max(m.sum(), 1),
            "weekend": (m & (g.ts_local.dt.dayofweek >= 5)).sum() / max(m.sum(), 1),
            "longest_gap_h": _longest_run(m.reset_index(drop=True)),
        })
    miss = pd.DataFrame(rows).sort_values("share", ascending=False)
    fmt = {"share": "{:.2%}".format, "night_01_04": "{:.0%}".format, "weekend": "{:.0%}".format}
    print(miss.to_string(index=False, formatters=fmt))
    all_missing = regular.groupby("ts_utc").is_missing.all()
    print(f"часов, когда пусто сразу на всех {df.station_id.nunique()} станциях: {int(all_missing.sum())}")
    print("(night_01_04 и weekend — доля пропусков данной станции, пришедшихся на ночь и выходные)")

    print("\n=== 5 строк из interim (Times Sq, будний день) ===")
    sample = df[(df.station_id == "611") & (df.ts_local >= pd.Timestamp("2024-06-05 07:00", tz=config.TZ_LOCAL))].head(5)
    print(sample.to_string(index=False))

    print("\n=== Переход на летнее время ===")
    local_day = df.ts_local.dt.date
    hours_per_day = df[df.station_id == df.station_id.iloc[0]].groupby(local_day).size()
    odd = hours_per_day[hours_per_day != 24]
    print("дни с числом местных часов != 24:", {str(d): int(n) for d, n in odd.items()})
    for day in odd.index:
        d0 = pd.Timestamp(day).tz_localize(config.TZ_LOCAL)
        win = df[(df.station_id == "611") & (df.ts_local >= d0) & (df.ts_local < d0 + pd.Timedelta(hours=5))]
        print(f"\n{day} ({'весна, 23 ч' if odd[day] == 23 else 'осень, 25 ч'}), Times Sq (611):")
        print(win[["ts_utc", "ts_local", "entries", "is_missing", "dst_flag"]].to_string(index=False))
        if odd[day] == 25:
            _compare_merged(df, day)

    print("\n=== Внешние таблицы ===")
    for kind, w in weather.items():
        nan = w.drop(columns=["ts_utc", "ts_local"]).isna().sum()
        print(f"weather_{kind}: {len(w)} ч, {w.ts_utc.min()} … {w.ts_utc.max()}, NaN: {nan[nan > 0].to_dict() or 0}")
    seasons = pd.to_datetime(mets.official_date).dt.year
    played = mets[mets.is_played]
    print(f"mets_home_games: {len(mets)} записей, сыграно {played.groupby(seasons[played.index]).size().to_dict()}, "
          f"по типам {played.game_type.value_counts().to_dict()}, не сыграно {mets[~mets.is_played].status.value_counts().to_dict()}")
    print(f"holidays_us: {len(hol)} строк, по категориям {hol.category.value_counts().to_dict()}")


def _compare_merged(df: pd.DataFrame, day) -> None:
    """Значение fall_merged против того же местного часа в соседние ±1, ±2 воскресенья."""
    line = df.groupby("ts_local").entries.sum(min_count=1)
    by_label = line.groupby(line.index.tz_localize(None)).sum(min_count=1)
    t = pd.Timestamp(day) + pd.Timedelta(hours=1)
    near = [t + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
    vals = by_label.reindex(near)
    print(f"  сумма по линии в 01:00: {by_label.get(t, np.nan):,.0f} (fall_merged) против "
          f"{', '.join(f'{v:,.0f}' for v in vals)} в соседние воскресенья → ×{by_label.get(t) / vals.mean():.2f}")


# --- main -------------------------------------------------------------------
def main() -> None:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    complexes = pd.read_csv(config.RAW / "mta_line7_complexes.csv", dtype={"station_complex_id": str})

    df = clean_mta(load_mta_raw(), config.LINE7_IDS)
    df.to_parquet(MTA_OUT, index=False)
    stations = build_stations(complexes)
    stations.to_parquet(config.INTERIM / "stations.parquet", index=False)

    weather = {}
    for kind in config.WEATHER_URLS:
        raw = json.loads((config.RAW / f"weather_{kind}_queens.json").read_text(encoding="utf-8"))
        weather[kind] = clean_weather(raw)
        weather[kind].to_parquet(config.INTERIM / f"weather_{kind}.parquet", index=False)

    mets_raw = [json.loads((config.RAW / f"mets_schedule_{s}.json").read_text(encoding="utf-8"))
                for s in config.MLB_SEASONS]
    mets = clean_mets(mets_raw)
    mets.to_parquet(config.INTERIM / "mets_home_games.parquet", index=False)

    hol_path = config.RAW / f"holidays_us_{min(config.HOLIDAY_YEARS)}_{max(config.HOLIDAY_YEARS)}.csv"
    hol = clean_holidays(pd.read_csv(hol_path))
    hol.to_parquet(config.INTERIM / "holidays_us.parquet", index=False)

    report(df, stations, weather, mets, hol)
    print(f"\nclean: готово → {config.INTERIM.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
