"""Geographic contracts. Coordinates are decimal degrees, distances are metres."""

from pydantic import BaseModel, ConfigDict, Field


class GeoPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    latitude: float = Field(ge=-90, le=90, strict=True)
    longitude: float = Field(ge=-180, le=180, strict=True)


class GeocodedLocation(GeoPoint):
    query: str = Field(min_length=1)
    formatted_address: str | None = None


class StationDistance(GeoPoint):
    station_name: str = Field(min_length=1)
    distance_m: float = Field(ge=0, strict=True)


class LocationResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location_name: str = Field(min_length=1)
    location: GeocodedLocation
    nearest_station: StationDistance
    nearest_stations: list[StationDistance] = Field(min_length=1)
