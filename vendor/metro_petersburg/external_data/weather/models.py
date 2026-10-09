"""Weather units and timestamps are explicit; missing measurements stay null."""

from datetime import datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class WeatherObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp: AwareDatetime = Field(description="API time with its UTC offset, never local system time.")
    latitude: float = Field(ge=-90, le=90, strict=True)
    longitude: float = Field(ge=-180, le=180, strict=True)
    temperature: float | None = Field(default=None, strict=True, description="Degrees Celsius.")
    feels_like: float | None = Field(default=None, strict=True, description="Degrees Celsius; requires API entitlement.")
    precipitation: float | None = Field(default=None, ge=0, strict=True, description="Hourly forecast total, mm; null for current weather.")
    precipitation_type: str | None = Field(default=None, strict=True)
    wind_speed: float | None = Field(default=None, ge=0, strict=True, description="m/s.")
    wind_gust: float | None = Field(default=None, ge=0, strict=True, description="m/s.")
    humidity: float | None = Field(default=None, ge=0, le=100, strict=True, description="Percent, 0–100.")
    pressure: float | None = Field(default=None, gt=0, strict=True, description="mm Hg.")
    condition: str | None = Field(default=None, strict=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def require_time(cls, value: object) -> object:
        if not isinstance(value, (str, datetime)):
            raise ValueError("timestamp must be an ISO 8601 string or aware datetime")
        return value
