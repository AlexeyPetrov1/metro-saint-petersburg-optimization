"""Загрузка сырых данных в data/raw — как есть, без правок.

Запуск: python -m src.download [--only complexes mta weather mets holidays weather_spb calendar] [--force]
Повторный запуск ничего не качает, если файл уже есть; --force перекачивает.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable

import holidays
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config

STEPS = ["complexes", "mta", "weather", "mets", "holidays", "weather_spb", "calendar"]


# --- Общие помощники --------------------------------------------------------
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    token = os.environ.get("SOCRATA_APP_TOKEN")  # бесплатный токен data.ny.gov — при троттлинге
    if token:
        s.headers["X-App-Token"] = token
    return s


def soql(session: requests.Session, params: dict, page: int = 50_000, timeout: int = 300) -> list[dict]:
    """Все строки SoQL-запроса постранично. Без $order страницы с $offset нестабильны."""
    if "$order" not in params:
        raise ValueError("SoQL-пагинация требует $order")
    rows, offset = [], 0
    while True:
        r = session.get(config.MTA_URL, params={**params, "$limit": page, "$offset": offset}, timeout=timeout)
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < page:
            return rows
        offset += page


def _get_json(session: requests.Session, url: str, params: dict, timeout: int = 120) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{url}: {data.get('reason', data)}")
    return data


def _write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Пишем во временный файл и переименовываем: прерванная загрузка не выглядит готовой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    _write_atomic(path, lambda p: p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8"))


def _rel(path: Path) -> str:
    return str(path.relative_to(config.ROOT))


def _skip(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  skip   {_rel(path)}")
        return True
    return False


# --- MTA ------------------------------------------------------------------
def fetch_line7_complexes(session: requests.Session, force: bool = False) -> pd.DataFrame:
    """Список станционных комплексов линии 7 и сверка с закреплённым в config списком."""
    path = config.RAW / "mta_line7_complexes.csv"
    if not _skip(path, force):
        # одна неделя вместо двух лет: агрегат по всему периоду идёт минуты, а пропавший
        # комплекс всё равно поймает сверка с config.LINE7_COMPLEXES
        start, end = config.MTA_COMPLEX_PROBE
        rows = soql(session, {
            "$select": "station_complex_id, station_complex, borough, "
                       "max(latitude) as lat, max(longitude) as lon",
            "$where": f"transit_timestamp between '{start}' and '{end}'",
            "$group": "station_complex_id, station_complex, borough",
            "$order": "station_complex_id, station_complex",
        })
        cx = pd.DataFrame(rows)
        print(f"  всего комплексов в датасете: {cx.station_complex_id.nunique()}")
        line7 = cx[cx.station_complex.str.contains(config.LINE7_REGEX, regex=True)]
        _write_atomic(path, lambda p: line7.to_csv(p, index=False))
        print(f"  saved  {_rel(path)}")

    line7 = pd.read_csv(path, dtype={"station_complex_id": str})
    _check_line7(line7)
    return line7


def _check_line7(line7: pd.DataFrame) -> None:
    found, expected = set(line7.station_complex_id), set(config.LINE7_IDS)
    if found != expected:
        raise RuntimeError(f"Комплексы линии 7 не совпали со списком в config: "
                           f"лишние {sorted(found - expected)}, пропали {sorted(expected - found)}")
    multi = line7.groupby("station_complex_id").station_complex.nunique()
    for cid in multi[multi > 1].index:
        print(f"  ВНИМАНИЕ: у комплекса {cid} несколько названий: {list(line7.station_complex[line7.station_complex_id == cid])}")

    order = {cid: i for i, cid in enumerate(config.LINE7_IDS)}
    shown = line7.drop_duplicates("station_complex_id").assign(o=lambda d: d.station_complex_id.map(order)).sort_values("o")
    print(f"  комплексы линии 7: {len(shown)} (совпадают со схемой)")
    for r in shown.itertuples():
        print(f"    {r.o:>2}  {r.station_complex_id:>4}  {r.station_complex}  [{r.borough}]")


def _months() -> list[pd.Timestamp]:
    return list(pd.date_range(config.START_LOCAL.normalize(), config.END_LOCAL, freq="MS"))


def fetch_mta_hourly(session: requests.Session, force: bool = False) -> int:
    """Почасовые входы по комплексам линии 7: сумма по типам оплаты на стороне сервера, по месяцу на файл."""
    ids = ", ".join(f"'{i}'" for i in config.LINE7_IDS)
    total = 0
    for month in _months():
        path = config.RAW / "mta" / f"line7_{month:%Y-%m}.parquet"
        if _skip(path, force):
            total += len(pd.read_parquet(path, columns=["station_complex_id"]))
            continue
        nxt = month + pd.offsets.MonthBegin(1)
        rows = soql(session, {
            "$select": "transit_timestamp, station_complex_id, "
                       "sum(ridership) as entries, sum(transfers) as transfers",
            "$where": f"station_complex_id in ({ids}) AND "
                      f"transit_timestamp >= '{month:%Y-%m-%dT%H:%M:%S}' AND "
                      f"transit_timestamp < '{nxt:%Y-%m-%dT%H:%M:%S}'",
            "$group": "transit_timestamp, station_complex_id",
            "$order": "transit_timestamp, station_complex_id",
        })
        if not rows:
            raise RuntimeError(f"MTA вернул 0 строк за {month:%Y-%m}")
        df = pd.DataFrame(rows).astype({"transit_timestamp": str, "station_complex_id": str,
                                        "entries": float, "transfers": float})
        if df.duplicated(["transit_timestamp", "station_complex_id"]).any():
            raise RuntimeError(f"MTA {month:%Y-%m}: дубликаты (час, станция) после агрегации")
        _write_atomic(path, lambda p: df.to_parquet(p, index=False))
        print(f"  saved  {_rel(path)}  {len(df):>6} строк")
        total += len(df)
    return total


# --- Погода -----------------------------------------------------------------
def fetch_weather(session: requests.Session, kind: str, place: str = "queens", force: bool = False) -> Path:
    """Open-Meteo почасово в GMT/unixtime: локальные метки Open-Meteo в дни перехода на летнее время неверны.

    kind: "archive" — фактическая погода; "hist_forecast" — что обещал прогноз (честный признак).
    place: ключ config.WEATHER_PLACES. Даты в GMT покрывают весь период в UTC; лишнее отрежет clean
    (Квинс: последний местный час 2024-12-31 23:00 EST = 04:00Z 1 января; СПб: 00:00 MSK 1 января = 21:00Z 31 декабря).
    """
    path = config.RAW / f"weather_{kind}_{place}.json"
    if _skip(path, force):
        return path
    (lat, lon), start, end = config.WEATHER_PLACES[place]
    data = _get_json(session, config.WEATHER_URLS[kind], {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": "GMT", "timeformat": "unixtime",
    })
    _write_json(path, data)
    print(f"  saved  {_rel(path)}  {len(data['hourly']['time'])} ч")
    return path


# --- Матчи «Метс» -----------------------------------------------------------
def fetch_mets_schedule(session: requests.Session, season: int, force: bool = False) -> Path:
    """Расписание «Метс» за сезон как есть; домашние игры на Citi Field отбирает clean."""
    path = config.RAW / f"mets_schedule_{season}.json"
    if _skip(path, force):
        return path
    data = _get_json(session, config.MLB_SCHEDULE_URL, {
        "sportId": 1, "teamId": config.METS_TEAM_ID, "season": season,
        "gameType": ",".join(config.MLB_GAME_TYPES), "hydrate": "venue,gameInfo",
    })
    _write_json(path, data)
    n = sum(len(d["games"]) for d in data["dates"])
    print(f"  saved  {_rel(path)}  {n} игр")
    return path


# --- Праздники --------------------------------------------------------------
def build_holidays(force: bool = False) -> Path:
    """Праздники США: федеральные (public, government), неофициальные и дополнительные дни штата NY."""
    path = config.RAW / f"holidays_us_{min(config.HOLIDAY_YEARS)}_{max(config.HOLIDAY_YEARS)}.csv"
    if _skip(path, force):
        return path
    years = config.HOLIDAY_YEARS
    rows = []
    for cat in ("public", "government", "unofficial"):
        rows += [(d, name, cat) for d, name in holidays.US(years=years, categories=(cat,)).items()]
    federal = set(holidays.US(years=years).items())
    rows += [(d, name, "ny_public") for d, name in holidays.US(subdiv="NY", years=years).items()
             if (d, name) not in federal]
    df = pd.DataFrame(rows, columns=["date", "name", "category"]).sort_values(["date", "category"])
    _write_atomic(path, lambda p: df.to_csv(p, index=False))
    print(f"  saved  {_rel(path)}  {len(df)} строк")
    return path


# --- Производственный календарь РФ ----------------------------------------
def fetch_isdayoff(session: requests.Session, year: int = config.CALENDAR_YEAR, force: bool = False) -> Path:
    """Тип каждого дня года с переносами выходных: 0 — рабочий, 1 — нерабочий, 2 — сокращённый (pre=1)."""
    path = config.RAW / f"calendar_ru_{year}_isdayoff.txt"
    if _skip(path, force):
        return path
    r = session.get(config.ISDAYOFF_URL, params={"year": year, "cc": "ru", "pre": 1}, timeout=60)
    r.raise_for_status()
    codes = r.text.strip()
    n_days = 366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
    if len(codes) != n_days or set(codes) - set("012"):
        raise RuntimeError(f"isdayoff: ожидалась строка из {n_days} цифр 0/1/2, получено {codes[:50]!r}")
    _write_atomic(path, lambda p: p.write_text(codes + "\n", encoding="utf-8"))
    print(f"  saved  {_rel(path)}  {len(codes)} дней, нерабочих {codes.count('1')}, сокращённых {codes.count('2')}")
    return path


def build_holidays_ru(year: int = config.CALENDAR_YEAR, force: bool = False) -> Path:
    """Названия праздников РФ (holidays.RU): переносов там нет, их даёт isdayoff."""
    path = config.RAW / f"holidays_ru_{year}.csv"
    if _skip(path, force):
        return path
    df = pd.DataFrame(sorted(holidays.RU(years=year).items()), columns=["date", "name"])
    _write_atomic(path, lambda p: df.to_csv(p, index=False))
    print(f"  saved  {_rel(path)}  {len(df)} строк")
    return path


# --- main -------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--force", action="store_true", help="перекачать, даже если файл есть")
    args = ap.parse_args(argv)

    session = _session()
    if "complexes" in args.only:
        print("[complexes] MTA: комплексы линии 7")
        fetch_line7_complexes(session, args.force)
    if "mta" in args.only:
        print("[mta] MTA: почасовые входы 2023–2024")
        n = fetch_mta_hourly(session, args.force)
        print(f"  итого {n} строк в {len(_months())} файлах")
    if "weather" in args.only:
        print("[weather] Open-Meteo, Квинс")
        for kind in config.WEATHER_URLS:
            fetch_weather(session, kind, "queens", args.force)
    if "mets" in args.only:
        print("[mets] MLB Stats API")
        for season in config.MLB_SEASONS:
            fetch_mets_schedule(session, season, args.force)
    if "holidays" in args.only:
        print("[holidays] библиотека holidays")
        build_holidays(args.force)
    if "weather_spb" in args.only:
        print("[weather_spb] Open-Meteo, Санкт-Петербург")
        for kind in config.WEATHER_URLS:
            fetch_weather(session, kind, "spb", args.force)
    if "calendar" in args.only:
        print(f"[calendar] производственный календарь РФ {config.CALENDAR_YEAR}: isdayoff.ru и holidays.RU")
        fetch_isdayoff(session, force=args.force)
        build_holidays_ru(force=args.force)
    print("download: готово")


if __name__ == "__main__":
    main()
