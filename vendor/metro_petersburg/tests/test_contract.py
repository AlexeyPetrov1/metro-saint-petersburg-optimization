import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src import config
from src.contract import ContractError, make_stub, stub_stations, validate_file, validate_records
from src.reference import load_stations

GOOD = {
    "station_id": "vosstaniya",
    "ts": "2026-11-14T18:00:00+03:00",
    "horizon_min": 30,
    "q10": 2100, "q50": 2600, "q90": 3150,
    "baseline": 1850,
    "is_anomaly": True,
    "model_version": "lgbm_events_v1",
}


def test_example_from_tz_is_valid():
    (p,) = validate_records([GOOD])
    assert p.ts.utcoffset().total_seconds() == 3 * 3600


def test_stub_is_valid():
    stub = make_stub()
    preds = validate_records(stub)
    stations = load_stations()
    with_data = set(stations.loc[stations.has_data, "station_id"])
    assert set(stub_stations()) == with_data and len(with_data) == 18
    assert len(preds) == 18 * 20 * 2
    assert {p.station_id for p in preds} == with_data
    assert {p.horizon_min for p in preds} == {60, 120}
    assert all(p.ts.minute == 0 for p in preds)  # часовые слоты
    tz = ZoneInfo(config.SPB_TZ)
    hours = sorted({p.ts for p in preds})
    assert hours[0] == datetime(2026, 11, 12, 5, tzinfo=tz) and hours[-1] == datetime(2026, 11, 13, 0, tzinfo=tz)
    assert len(hours) == 20
    assert preds[0].ts.weekday() == 3  # четверг
    assert any(p.is_anomaly for p in preds) and not all(p.is_anomaly for p in preds)


def test_stub_file_matches_generator():
    """stub.json на диске проходит контракт и не устарел относительно make_stub()."""
    path = config.PREDICTIONS / "stub.json"
    validate_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == make_stub()


def test_stub_is_deterministic():
    assert make_stub() == make_stub()


@pytest.mark.parametrize("patch", [
    {"q10": 2700},                         # q10 > q50
    {"ts": "2026-11-14T18:00:00"},         # нет смещения
    {"ts": "2026-11-14T18:05:00+03:00"},   # не граница слота
    {"horizon_min": 0},
    {"horizon_min": 135},
    {"horizon_min": 40},                   # не кратно 15
    {"q50": -1},
    {"is_anomaly": "true"},
    {"station_id": ""},
    {"extra_field": 1},
])
def test_bad_record_rejected(patch):
    with pytest.raises(ContractError):
        validate_records([{**GOOD, **patch}])


def test_missing_field_rejected():
    rec = {k: v for k, v in GOOD.items() if k != "baseline"}
    with pytest.raises(ContractError):
        validate_records([rec])


def test_duplicate_key_rejected():
    with pytest.raises(ContractError, match="дубликат"):
        validate_records([GOOD, GOOD])


def test_all_errors_collected():
    with pytest.raises(ContractError) as exc:
        validate_records([{**GOOD, "q10": 9999}, GOOD, {**GOOD, "horizon_min": 7}])
    assert len(exc.value.errors) == 2
