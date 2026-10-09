"""Compact output contract; source-specific timestamps/coordinates stay upstream."""

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from external_data.calendar.models import DayType, moscow_time
from external_data.railway.models import Count


class WeatherFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature: float | None = None
    feels_like: float | None = None
    precipitation: float | None = None
    precipitation_type: str | None = None
    wind_speed: float | None = None
    wind_gust: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    condition: str | None = None


class CalendarFeatureValues(BaseModel):
    """Projection of validated CalendarFeatures without its duplicate timestamp."""

    model_config = ConfigDict(extra="forbid")

    day_of_week: int = Field(ge=0, le=6, strict=True)
    hour: int = Field(ge=0, le=23, strict=True)
    minute: int = Field(ge=0, le=59, strict=True)
    is_weekend: bool = Field(strict=True)
    is_workday: bool = Field(strict=True)
    is_holiday: bool = Field(strict=True)
    is_preholiday: bool | None = Field(strict=True)
    day_type: DayType


class EventFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_count: int = Field(ge=0, strict=True)
    event_types: list[str]
    total_expected_people: int | None = Field(ge=0, strict=True)
    has_event: bool = Field(strict=True)


class RailwayFeatureValues(BaseModel):
    """Projection of ready RailwayFeatures without duplicate time/station fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    railway_name: str
    arrivals_next_15m: Count
    arrivals_next_30m: Count
    arrivals_next_60m: Count
    arrivals_next_120m: Count
    train_arrivals_next_30m: Count
    suburban_arrivals_next_30m: Count
    minutes_to_next_arrival: float | None = Field(ge=0)


class ExternalFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: AwareDatetime
    station: str | None = None
    weather: WeatherFeatures
    calendar: CalendarFeatureValues
    events: EventFeatures
    railway: RailwayFeatureValues | None = None

    @field_validator("timestamp")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return moscow_time(value)
