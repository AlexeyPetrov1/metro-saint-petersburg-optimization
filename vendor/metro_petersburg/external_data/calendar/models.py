"""Validated calendar features; civil time is always Europe/Moscow."""

from datetime import datetime
from typing import Literal, Self
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

MOSCOW = ZoneInfo("Europe/Moscow")
DayType = Literal["workday", "weekend", "holiday"]


def moscow_time(timestamp: datetime) -> datetime:
    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must have an explicit timezone")
    return timestamp.astimezone(MOSCOW)


class CalendarFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: AwareDatetime
    day_of_week: int = Field(ge=0, le=6, strict=True, description="Monday=0, Sunday=6, Moscow time.")
    hour: int = Field(ge=0, le=23, strict=True)
    minute: int = Field(ge=0, le=59, strict=True)
    is_weekend: bool = Field(strict=True, description="Calendar day off, excluding holidays; not simply Sat/Sun.")
    is_workday: bool = Field(strict=True)
    is_holiday: bool = Field(strict=True)
    is_preholiday: bool | None = Field(strict=True)
    day_type: DayType

    @field_validator("timestamp")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return moscow_time(value)

    @model_validator(mode="after")
    def consistent_features(self) -> Self:
        if (self.day_of_week, self.hour, self.minute) != (
            self.timestamp.weekday(), self.timestamp.hour, self.timestamp.minute,
        ):
            raise ValueError("Calendar components must match the Moscow timestamp")
        if (self.is_workday, self.is_weekend, self.is_holiday) != (
            self.day_type == "workday", self.day_type == "weekend", self.day_type == "holiday",
        ):
            raise ValueError("Day flags must match day_type")
        return self
