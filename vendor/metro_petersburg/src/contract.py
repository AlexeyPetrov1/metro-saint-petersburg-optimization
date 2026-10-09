"""Контракт выхода модели (раздел 1 ТЗ): схема, валидация и JSON-заглушка для команды.

Запуск: python -m src.contract — пишет data/predictions/stub.json и contract.schema.json.
Схема допускает слоты по 15 минут и горизонты 15–120 мин; модель для СПб отдаёт часовые слоты и горизонты 60/120.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import (AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictInt,
                      ValidationError, field_validator, model_validator)

from src import config, reference

NonNeg = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Prediction(BaseModel):
    """Одна строка прогноза: входы на станцию в слоте ts, прогноз сделан за horizon_min минут."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"description": "Дополнительно проверяется: q10 <= q50 <= q90; "
                                          "минуты ts кратны 15, секунды = 0"},
    )

    station_id: str = Field(min_length=1)
    ts: AwareDatetime = Field(description="Начало слота, локальное время со смещением")
    horizon_min: StrictInt = Field(ge=15, le=120, multiple_of=15)
    q10: NonNeg
    q50: NonNeg
    q90: NonNeg
    baseline: NonNeg = Field(description="Расчётный поток для слота и типа дня")
    is_anomaly: StrictBool
    model_version: str = Field(min_length=1)

    @field_validator("ts")
    @classmethod
    def _slot_boundary(cls, ts: datetime) -> datetime:
        if ts.minute % 15 or ts.second or ts.microsecond:
            raise ValueError("ts должен быть началом 15-минутного слота")
        return ts

    @model_validator(mode="after")
    def _quantiles_ordered(self) -> "Prediction":
        if not self.q10 <= self.q50 <= self.q90:
            raise ValueError(f"нарушено q10 <= q50 <= q90: {self.q10}, {self.q50}, {self.q90}")
        return self


class ContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        shown = "\n".join(errors[:20])
        more = f"\n... и ещё {len(errors) - 20}" if len(errors) > 20 else ""
        super().__init__(f"{len(errors)} ошибок контракта:\n{shown}{more}")


def validate_records(records: list[dict]) -> list[Prediction]:
    """Проверяет все записи и выбрасывает одно ContractError со списком всех ошибок."""
    if not isinstance(records, list):
        raise ContractError(["ожидается JSON-массив записей"])
    preds, errors, seen = [], [], {}
    for i, rec in enumerate(records):
        try:
            p = Prediction.model_validate(rec)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"]) or "<запись>"
                errors.append(f"[{i}] {loc}: {err['msg']}")
            continue
        key = (p.station_id, p.ts, p.horizon_min, p.model_version)
        if key in seen:
            errors.append(f"[{i}] дубликат ключа (station_id, ts, horizon_min, model_version) — см. [{seen[key]}]")
        else:
            seen[key] = i
        preds.append(p)
    if errors:
        raise ContractError(errors)
    return preds


def validate_file(path: Path) -> list[Prediction]:
    with open(path, encoding="utf-8") as f:
        return validate_records(json.load(f))


def json_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Predictions",
        "type": "array",
        "items": Prediction.model_json_schema(),
    }


# --- Заглушка ---------------------------------------------------------------
STUB_DAY = datetime(2026, 11, 12)          # четверг: будний двухпиковый профиль
STUB_N_SLOTS = 20                          # часы работы метро 05:00 … 00:00 (последний — уже 13 ноября)
STUB_HORIZONS = (60, 120)
STUB_EVENT = {"stations": ("vosstaniya", "chernyshevskaya", "vladimirskaya"),
              "start": "18:00", "end": "20:45", "mult": 1.4}


def stub_stations() -> list[str]:
    """Станции, по которым модель будет отдавать прогноз: из справочника, только с данными (has_data)."""
    st = reference.load_stations()
    return st.loc[st.has_data, "station_id"].tolist()


def make_stub(seed: int = 42) -> list[dict]:
    """Синтетический будний день 1 линии СПб по часам; вечером «событие» у трёх станций."""
    rng = np.random.default_rng(seed)
    tz = ZoneInfo(config.SPB_TZ)
    start = STUB_DAY.replace(hour=config.SPB_SERVICE_DAY_START, tzinfo=tz)
    slots = [start + timedelta(hours=k) for k in range(STUB_N_SLOTS)]
    hours = config.SPB_SERVICE_DAY_START + np.arange(STUB_N_SLOTS)  # 5…24: полночь — продолжение суток

    # двугорбый суточный профиль: утренний пик сильнее вечернего
    profile = (0.15 + 1.0 * np.exp(-((hours - 8.5) / 1.0) ** 2 / 2)
               + 0.8 * np.exp(-((hours - 18.0) / 1.3) ** 2 / 2))
    profile /= profile.sum()

    ev_start, ev_end = (datetime.strptime(STUB_EVENT[k], "%H:%M").time() for k in ("start", "end"))
    records = []
    for station in stub_stations():
        volume = rng.uniform(15_000, 60_000)  # входов за день
        for slot, share in zip(slots, profile):
            baseline = volume * share
            event = station in STUB_EVENT["stations"] and ev_start <= slot.time() <= ev_end
            mult = STUB_EVENT["mult"] if event else 1.0
            for h in STUB_HORIZONS:
                rel = 0.10 + 0.10 * h / 120  # интервал шире на дальнем горизонте
                q50 = baseline * mult * (1 + rng.normal(0, 0.02 + 0.03 * h / 120))
                records.append({
                    "station_id": station,
                    "ts": slot.isoformat(),
                    "horizon_min": h,
                    "q10": round(q50 * (1 - rel)),
                    "q50": round(q50),
                    "q90": round(q50 * (1 + rel)),
                    "baseline": round(baseline),
                    "is_anomaly": event,
                    "model_version": "stub_v1",
                })
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    # массив по одной записи на строку — читается и глазами, и любым JSON-парсером
    lines = ",\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")


def main() -> None:
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    stub_path = config.PREDICTIONS / "stub.json"
    schema_path = config.PREDICTIONS / "contract.schema.json"

    _write_records(stub_path, make_stub())
    schema_path.write_text(json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    preds = validate_file(stub_path)
    n_anom = sum(p.is_anomaly for p in preds)
    print(f"{stub_path.relative_to(config.ROOT)}: {len(preds)} записей, контракт OK, аномальных {n_anom}")
    print(f"{schema_path.relative_to(config.ROOT)}: JSON Schema")
    print("пример:", json.dumps(preds[0].model_dump(mode="json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
