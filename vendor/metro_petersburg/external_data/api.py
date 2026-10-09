"""Public station × timestamp facade, with explicit one-time configuration.

configure_external_data(weather_coordinates=(lat, lon)) loads the bundled
Russian production calendar for 2026 and chooses a common weather point.
A mapping {station: (lat, lon)} instead selects per-station weather coordinates.
A calendar_provider or calendar_file can override the bundled calendar.
No coordinates, uncovered dates, or demo calendar are inferred.

Hourly forecasts apply unchanged within their Moscow hour [time, time + 1h).
The requested ML timestamp is preserved. There is no archive, interpolation,
nearest-hour substitution or use of current weather values. Missing hour points
give weather=null fields; API failures still raise. Events must already be selected
for the station/time. The facade never calls LLMs or geocoders.
"""

from collections.abc import Iterable, Mapping
from contextlib import ExitStack, closing
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Protocol

from .calendar import CalendarProvider, MappingCalendarProvider, get_calendar_features
from .calendar.models import moscow_time
from .events.models import Event, normalize_timestamp
from .features import ExternalFeatures, build_external_features
from .features.models import RailwayFeatureValues
from .locations.models import GeoPoint
from .locations.stations import LINE1_STATIONS
from .railway import HUBS, RailwayProvider, YandexRaspProvider, get_railway_features
from .weather import WeatherObservation, YandexWeatherProvider

Coordinates = tuple[float, float] | Mapping[str, tuple[float, float]]
_DEFAULT_CALENDAR_FILE = Path(__file__).resolve().parent / "calendar/data/ru_production_calendar_2026.json"


class ExternalDataConfigurationError(ValueError):
    """Invalid calendar source, weather coordinates, or missing configuration."""


class WeatherProvider(Protocol):
    """Existing weather adapter methods, injectable for offline tests."""

    def get_current_weather(self, lat: float, lon: float) -> WeatherObservation:
        ...

    def get_hourly_forecast(self, lat: float, lon: float, hours: int = 2) -> list[WeatherObservation]:
        ...


def _station(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("station must be a Line 1 station name")
    for name in LINE1_STATIONS:
        if name.casefold() == value.strip().casefold():
            return name
    raise ValueError("Неизвестная station: используйте название станции линии 1.")


def _point(value: tuple[float, float]) -> GeoPoint:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ExternalDataConfigurationError("weather_coordinates must contain (latitude, longitude)")
    return GeoPoint(latitude=value[0], longitude=value[1])


class ExternalDataClient:
    """Reusable configuration; production clients are opened/closed per call.

    Normally constructed by configure_external_data. Dependency injection is
    available here for tests or an application-owned calendar source. Injected
    providers remain owned by the caller and are never closed by this client.
    """

    def __init__(
        self, *, calendar_provider: CalendarProvider, weather_coordinates: Coordinates,
        weather_provider: WeatherProvider | None = None,
        railway_provider: RailwayProvider | None = None,
    ) -> None:
        if not callable(getattr(calendar_provider, "get_day", None)):
            raise ExternalDataConfigurationError("calendar_provider must implement get_day(date)")
        if isinstance(weather_coordinates, Mapping):
            if not weather_coordinates:
                raise ExternalDataConfigurationError("weather_coordinates mapping must not be empty")
            self._points = {_station(station): _point(point) for station, point in weather_coordinates.items()}
            self._common_point = None
        else:
            self._common_point = _point(weather_coordinates)
            self._points = {}
        self._calendar = calendar_provider
        self._weather = weather_provider
        self._railway = railway_provider

    def get_external_features(
        self, station: str, timestamp: datetime | str, events: list[Event] | None = None,
    ) -> ExternalFeatures:
        station = _station(station)
        local = normalize_timestamp(timestamp)
        if events is not None and (not isinstance(events, list) or any(not isinstance(event, Event) for event in events)):
            raise TypeError("events must be a list of Event or None")
        point = self._common_point or self._points.get(station)
        if point is None:
            raise ExternalDataConfigurationError("Нет weather_coordinates для указанной станции.")
        # Calendar coverage must be established before spending API quota.
        calendar = get_calendar_features(local, provider=self._calendar)
        hub = next((hub for hub in HUBS if hub.metro_station == station), None)
        with ExitStack() as stack:
            weather_provider = self._weather
            if weather_provider is None:
                weather_provider = stack.enter_context(closing(YandexWeatherProvider()))
            railway_provider = self._railway
            if hub is not None and railway_provider is None:
                railway_provider = stack.enter_context(closing(YandexRaspProvider()))
            weather = _weather_at(weather_provider, point, local)
            railway = get_railway_features(local, hub, railway_provider) if hub is not None else None
            return build_external_features(
                local, weather, calendar, events if events is not None else [],
                station=station, railway=railway,
            )

    def get_external_feature_rows(self, requests: Iterable[Mapping]) -> list[dict]:
        """Sequential list of {station, timestamp, events?}; fail on first error.

        No cross-request events reuse, hidden cache, parallelism or partial result.
        """
        return [external_features_to_ml_dict(self.get_external_features(**request)) for request in requests]


def _weather_at(provider: WeatherProvider, point: GeoPoint, timestamp: datetime) -> WeatherObservation | None:
    # Current weather supplies only API server time for the forecast horizon.
    # Its values must never substitute for a missing forecast hour.
    current = provider.get_current_weather(point.latitude, point.longitude)
    hours_ahead = (timestamp - current.timestamp).total_seconds() / 3600
    if not 0 <= hours_ahead <= 48:
        return None
    forecast = provider.get_hourly_forecast(point.latitude, point.longitude, hours=max(1, ceil(hours_ahead)))
    hour_start = moscow_time(timestamp).replace(minute=0, second=0, microsecond=0)
    return next((observation for observation in forecast if observation.timestamp == hour_start), None)


_default_client: ExternalDataClient | None = None


def configure_external_data(
    *, weather_coordinates: Coordinates, calendar_file: str | Path | None = None,
    calendar_provider: CalendarProvider | None = None,
) -> ExternalDataClient:
    """Configure once per process, without network I/O or API credentials.

    Priority: calendar_provider, then calendar_file, then the bundled 2026
    Russian production calendar. No demo or weekday-based fallback is selected.
    An explicit common weather point or per-station coordinate mapping is required.
    Returns the configured client; failed reconfiguration leaves the old one intact.
    """
    if calendar_provider is None:
        path = calendar_file if calendar_file is not None else _DEFAULT_CALENDAR_FILE
        try:
            calendar_provider = MappingCalendarProvider.from_file(path)
        except (OSError, UnicodeError, ValueError, TypeError):
            raise ExternalDataConfigurationError("Не удалось загрузить calendar_file: нужна корректная таблица производственного календаря.") from None
    client = ExternalDataClient(calendar_provider=calendar_provider, weather_coordinates=weather_coordinates)
    global _default_client
    _default_client = client
    return client


def get_external_features(
    station: str, timestamp: datetime | str, events: list[Event] | None = None,
) -> ExternalFeatures:
    """Get a station's external features after one configure_external_data call."""
    return _configured_client().get_external_features(station, timestamp, events)


def get_external_feature_rows(requests: Iterable[Mapping]) -> list[dict]:
    """Sequential batch: each mapping contains station, timestamp and optional events."""
    return _configured_client().get_external_feature_rows(requests)


def _configured_client() -> ExternalDataClient:
    if _default_client is None:
        raise ExternalDataConfigurationError(
            "Сначала вызовите configure_external_data с weather_coordinates."
        )
    return _default_client


def external_features_to_ml_dict(features: ExternalFeatures) -> dict:
    """New flat JSON-compatible row; event_types remains a list, null stays None."""
    if not isinstance(features, ExternalFeatures):
        raise TypeError("features must be ExternalFeatures")
    result = {"timestamp": moscow_time(features.timestamp).isoformat(), "station": features.station}
    for block in (features.weather, features.calendar, features.events):
        result.update(block.model_dump(mode="json"))
    railway = features.railway.model_dump(mode="json") if features.railway is not None else {}
    for name in RailwayFeatureValues.model_fields:
        output_name = f"railway_{name}" if name.startswith("arrivals_next_") else name
        result[output_name] = railway.get(name)
    return result
