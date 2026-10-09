"""Railway arrivals and demand features, never metro trains or passenger estimates."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field

from external_data.calendar.models import moscow_time


def _datetime_value(value: object) -> object:
    if not isinstance(value, (str, datetime)):
        raise ValueError("timestamp must be an ISO 8601 string or aware datetime")
    return value


MoscowTimestamp = Annotated[AwareDatetime, BeforeValidator(_datetime_value), AfterValidator(moscow_time)]
StationCode = Annotated[str, Field(pattern=r"^s[0-9]+$", strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]


class RailwayHub(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    railway_name: str = Field(min_length=1)
    rasp_station_code: StationCode
    metro_station: str = Field(min_length=1)


class RailwayArrival(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    arrival_time: MoscowTimestamp
    transport_type: Literal["train", "suburban"]
    train_number: str | None = Field(default=None, min_length=1, strict=True)
    title: str | None = Field(default=None, min_length=1, strict=True)
    station_code: StationCode


class RailwayFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    timestamp: MoscowTimestamp
    metro_station: str
    railway_name: str
    arrivals_next_15m: Count
    arrivals_next_30m: Count
    arrivals_next_60m: Count
    arrivals_next_120m: Count
    train_arrivals_next_30m: Count
    suburban_arrivals_next_30m: Count
    minutes_to_next_arrival: float | None = Field(ge=0)
