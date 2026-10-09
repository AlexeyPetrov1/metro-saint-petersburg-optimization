"""Ручные справочники 1 линии СПб из data/reference (в git): загрузка с проверкой колонок и ключей."""
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

from src import config

STATION_COLS = ["station_id", "name", "name_external", "line_order", "n_vestibules", "has_data"]
VESTIBULE_COLS = ["vestibule_id", "raw_name", "station_id", "vestibule_no", "closes_early", "first_hour", "last_hour"]
VESTIBULE_15MIN_COLS = ["raw_name_15min", "vestibule_id", "station_id", "vestibule_no", "in_hourly", "first_hour",
                        "last_hour"]
INCIDENT_COLS = ["incident_id", "vestibule_id", "start_local", "end_local", "description", "source"]
EVENT_COLS = ["event", "start_local", "end_local", "venue", "nearest_station_id", "source", "verified"]
ALL_VESTIBULES = "*"  # vestibule_id в incidents.csv: вся линия


def nfc(s: str) -> str:
    """Имена файлов на macOS бывают в NFD («й» = «и» + знак) — сравниваем всегда в NFC."""
    return unicodedata.normalize("NFC", s).strip()


def service_pos(hour):
    """Номер часа в сутках метро: 05:00 → 0 … 00:00 → 19, закрытые 01–04 → 20–23."""
    return (hour - config.SPB_SERVICE_DAY_START) % 24


def _read(path: Path, cols: list[str], text: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: str for c in text}, keep_default_na=False)
    if list(df.columns) != cols:
        raise ValueError(f"{path.name}: колонки {list(df.columns)}, ожидались {cols}")
    for c in text:
        df[c] = df[c].map(nfc)
    return df


def _bool(df: pd.DataFrame, col: str, path: Path) -> pd.Series:
    if df[col].dtype != bool:
        raise ValueError(f"{path.name}: {col} должен быть true/false, а не {sorted(df[col].astype(str).unique())}")
    return df[col]


def _unique(df: pd.DataFrame, col: str, path: Path) -> None:
    dup = df[col][df[col].duplicated()]
    if len(dup):
        raise ValueError(f"{path.name}: повторяется {col}: {sorted(dup.unique())}")


def load_stations(path: Path = config.SPB_STATIONS_CSV) -> pd.DataFrame:
    """19 станций 1 линии в порядке Девяткино → пр. Ветеранов; has_data=false — нет в данных организаторов.
    name_external — название станции в external_data (модуль внешних данных), см. src/external_adapter.py."""
    df = _read(path, STATION_COLS, ["station_id", "name", "name_external"])
    _bool(df, "has_data", path)
    _unique(df, "station_id", path)
    _unique(df, "name_external", path)
    df = df.sort_values("line_order", ignore_index=True)
    if df.line_order.tolist() != list(range(len(df))):
        raise ValueError(f"{path.name}: line_order должен идти 0…{len(df) - 1} без пропусков")
    if not (df.has_data == (df.n_vestibules > 0)).all():
        raise ValueError(f"{path.name}: has_data должен совпадать с n_vestibules > 0")
    return df


def load_vestibules(path: Path = config.SPB_VESTIBULES_CSV, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Вестибюли: raw_name — как в файле организаторов; режим работы first_hour…last_hour (местное время)."""
    stations = load_stations() if stations is None else stations
    df = _read(path, VESTIBULE_COLS, ["vestibule_id", "raw_name", "station_id"])
    _bool(df, "closes_early", path)
    for col in ("vestibule_id", "raw_name"):
        _unique(df, col, path)
    unknown = set(df.station_id) - set(stations.loc[stations.has_data, "station_id"])
    if unknown:
        raise ValueError(f"{path.name}: station_id нет среди станций с данными: {sorted(unknown)}")
    counts = df.groupby("station_id").size()
    expected = stations.set_index("station_id").n_vestibules
    if not counts.reindex(expected.index, fill_value=0).equals(expected):
        raise ValueError(f"{path.name}: число вестибюлей не совпадает с n_vestibules в {config.SPB_STATIONS_CSV.name}")
    for col in ("first_hour", "last_hour"):
        if df[col].isin(config.SPB_CLOSED_HOURS).any() or not df[col].between(0, 23).all():
            raise ValueError(f"{path.name}: {col} должен быть часом работы метро")
    if (service_pos(df.first_hour) > service_pos(df.last_hour)).any():
        raise ValueError(f"{path.name}: first_hour позже last_hour")
    if not (df.closes_early == (df.last_hour != 0)).all():
        raise ValueError(f"{path.name}: closes_early должен совпадать с last_hour != 0")
    order = stations.set_index("station_id").line_order
    return (df.assign(_o=df.station_id.map(order)).sort_values(["_o", "vestibule_no"])
              .drop(columns="_o").reset_index(drop=True))


def load_vestibules_15min(path: Path = config.SPB_VESTIBULES_15MIN_CSV, vestibules: pd.DataFrame | None = None,
                          stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Названия вестибюлей в 15-минутных файлах (этап 6). in_hourly=false — вестибюля нет в часовом файле
    (Технологический институт): его режим работы задан здесь, у остальных обязан совпасть со справочником вестибюлей."""
    stations = load_stations() if stations is None else stations
    vestibules = load_vestibules(stations=stations) if vestibules is None else vestibules
    df = _read(path, VESTIBULE_15MIN_COLS, ["raw_name_15min", "vestibule_id", "station_id"])
    _bool(df, "in_hourly", path)
    for col in ("raw_name_15min", "vestibule_id"):
        _unique(df, col, path)
    unknown = set(df.station_id) - set(stations.station_id)
    if unknown:
        raise ValueError(f"{path.name}: неизвестные station_id: {sorted(unknown)}")
    known = vestibules.set_index("vestibule_id")
    hourly = df[df.in_hourly].set_index("vestibule_id")
    if set(hourly.index) != set(known.index):
        raise ValueError(f"{path.name}: in_hourly=true должен быть ровно у вестибюлей {config.SPB_VESTIBULES_CSV.name}: "
                         f"лишние {sorted(set(hourly.index) - set(known.index))}, "
                         f"нет {sorted(set(known.index) - set(hourly.index))}")
    cols = ["station_id", "vestibule_no", "first_hour", "last_hour"]
    off = hourly[cols].ne(known.loc[hourly.index, cols]).any(axis=1)
    if off.any():
        raise ValueError(f"{path.name}: {cols} не совпадают со справочником вестибюлей: {sorted(hourly.index[off])}")
    for col in ("first_hour", "last_hour"):
        if df[col].isin(config.SPB_CLOSED_HOURS).any() or not df[col].between(0, 23).all():
            raise ValueError(f"{path.name}: {col} должен быть часом работы метро")
    order = stations.set_index("station_id").line_order
    return (df.assign(_o=df.station_id.map(order)).sort_values(["_o", "vestibule_no"])
              .drop(columns="_o").reset_index(drop=True))


def _interval(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    for col in ("start_local", "end_local"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M")
    if not (df.start_local < df.end_local).all():
        raise ValueError(f"{path.name}: start_local должен быть раньше end_local")
    return df


def load_incidents(path: Path = config.SPB_INCIDENTS_CSV, vestibule_ids: list[str] | None = None) -> pd.DataFrame:
    """Ручная разметка инцидентов: интервал [start_local, end_local), vestibule_id="*" — вся линия."""
    vestibule_ids = load_vestibules().vestibule_id.tolist() if vestibule_ids is None else vestibule_ids
    df = _interval(_read(path, INCIDENT_COLS, ["incident_id", "vestibule_id", "description", "source"]), path)
    unknown = set(df.vestibule_id) - set(vestibule_ids) - {ALL_VESTIBULES}
    if unknown:
        raise ValueError(f"{path.name}: неизвестные vestibule_id: {sorted(unknown)}")
    return df


def load_events(path: Path = config.SPB_EVENTS_CSV, station_ids: list[str] | None = None) -> pd.DataFrame:
    """События города; nearest_station_id пустой, если событие не привязано к станции 1 линии."""
    station_ids = load_stations().station_id.tolist() if station_ids is None else station_ids
    df = _interval(_read(path, EVENT_COLS, ["event", "venue", "nearest_station_id", "source"]), path)
    _bool(df, "verified", path)
    unknown = set(df.nearest_station_id) - set(station_ids) - {""}
    if unknown:
        raise ValueError(f"{path.name}: неизвестные nearest_station_id: {sorted(unknown)}")
    return df.sort_values("start_local", ignore_index=True)


def load_operations(path: Path = config.LINE1_OPERATIONS_YAML) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
