"""Resolve a place to the closest Line 1 stations by great-circle distance."""

from math import asin, cos, radians, sin, sqrt
import re
import sys

from .geocoder import GeocoderProvider
from .candidates import is_petersburg_address
from .models import GeoPoint, LocationResolution, StationDistance
from .stations import LINE1_STATIONS, geocode_station

EARTH_RADIUS_M = 6_371_008.8  # Mean spherical Earth radius; no routing semantics.


class LocationResolutionError(RuntimeError):
    """A location or station could not be geocoded; no partial ranking returned."""


def haversine_distance(point_a: GeoPoint, point_b: GeoPoint) -> float:
    """Great-circle distance in metres, including coincident/antipodal points."""
    lat_a, lat_b = radians(point_a.latitude), radians(point_b.latitude)
    delta_lat = lat_b - lat_a
    delta_lon = radians(point_b.longitude - point_a.longitude)
    value = sin(delta_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(min(1.0, max(0.0, value))))


def resolve_location_to_line1(
    location_name: str,
    geocoder: GeocoderProvider,
    top_k: int = 3,
    address: str | None = None,
) -> LocationResolution:
    if not isinstance(location_name, str) or not location_name.strip():
        raise ValueError("location_name must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(LINE1_STATIONS):
        raise ValueError("top_k must be an integer in [1, 19]")
    name = location_name.strip()
    if address is not None and (not isinstance(address, str) or not address.strip()):
        raise ValueError("address must be a non-empty string or None")
    location = None
    if address is not None:
        address_query = address.strip()
        if not re.search(r"\b(?:санкт-петербург|спб)\b|(?:^|,)\s*(?:г\.|город)\s+", address_query, re.I):
            address_query += ", Санкт-Петербург"
        address_lookup = getattr(geocoder, "geocode_address", None)
        if callable(address_lookup):
            location = address_lookup(address_query)
        else:
            # Existing custom providers remain valid. Without structured locality,
            # their returned formatted_address must explicitly confirm the city.
            location = geocoder.geocode(address_query)
            if location is not None and not is_petersburg_address(location.formatted_address):
                location = None
        if location is not None:
            _debug_source(geocoder, "address")
    query = name if "санкт-петербург" in name.casefold() else f"{name}, Санкт-Петербург"
    if location is None:
        location = geocoder.geocode(query)
        if location is not None:
            _debug_source(geocoder, "location_name")
    if location is None:
        raise LocationResolutionError("Место не найдено с достаточной уверенностью; уточните location_name или адрес.")
    distances = []
    for station_name in LINE1_STATIONS:
        point = geocode_station(station_name, geocoder)
        if point is None:
            raise LocationResolutionError(f"Станция «{station_name}» не найдена геокодером; расчёт остановлен.")
        distances.append(StationDistance(
            station_name=station_name, latitude=point.latitude, longitude=point.longitude,
            distance_m=haversine_distance(location, point),
        ))
    # Python's stable sort preserves canonical order when distances are identical.
    distances.sort(key=lambda station: station.distance_m)
    return LocationResolution(
        location_name=name, location=location, nearest_station=distances[0],
        nearest_stations=distances[:top_k],
    )


def _debug_source(geocoder: GeocoderProvider, source: str) -> None:
    if getattr(geocoder, "debug", False):
        # Source is an internal constant, never user data or provider credentials.
        print(f"DEBUG resolution source: {source}", file=sys.stderr)
