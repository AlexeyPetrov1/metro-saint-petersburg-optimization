"""Date classification boundary; an official calendar can implement the protocol."""

import json
from datetime import date
from pathlib import Path
from typing import Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .models import DayType


class CalendarDay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day_type: DayType
    is_preholiday: bool | None = Field(default=None, strict=True)


class CalendarDataUnavailableError(ValueError):
    """Missing source data must not silently become a weekday/weekend guess."""


class CalendarProvider(Protocol):
    def get_day(self, day: date) -> CalendarDay:
        """Classify a Moscow calendar date; fail if the source doesn't cover it."""
        ...


class MappingCalendarProvider:
    """Explicit local date table. No inferred holidays, transfers or weekdays."""

    def __init__(self, days: Mapping[date, CalendarDay]) -> None:
        self._days = TypeAdapter(dict[date, CalendarDay]).validate_python(dict(days))

    @classmethod
    def from_file(cls, path: str | Path) -> "MappingCalendarProvider":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        days = TypeAdapter(dict[date, CalendarDay]).validate_python(payload)
        return cls(days)

    def get_day(self, day: date) -> CalendarDay:
        try:
            return self._days[day]
        except KeyError:
            raise CalendarDataUnavailableError(f"Нет данных производственного календаря для {day.isoformat()}.") from None


class FakeCalendarProvider(MappingCalendarProvider):
    """Fixed test table; never imitates or downloads a production calendar."""
