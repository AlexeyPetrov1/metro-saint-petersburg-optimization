"""Combine already parsed inputs without I/O, filtering or feature engineering."""

from datetime import datetime

from external_data.calendar.models import CalendarFeatures, moscow_time
from external_data.events.models import Event
from external_data.weather.models import WeatherObservation
from external_data.railway.models import RailwayFeatures

from .models import CalendarFeatureValues, EventFeatures, ExternalFeatures, RailwayFeatureValues, WeatherFeatures


def build_external_features(
    timestamp: datetime,
    weather: WeatherObservation | None,
    calendar: CalendarFeatures,
    events: list[Event],
    station: str | None = None,
    railway: RailwayFeatures | None = None,
) -> ExternalFeatures:
    """Project weather/calendar and aggregate exactly the supplied event list.

    Calendar and railway must describe the same instant. Railway must match
    station. Weather selection and event relevance/deduplication are the caller's
    responsibility; no location resolution is performed here.
    """
    local = moscow_time(timestamp)
    if not isinstance(calendar, CalendarFeatures):
        raise TypeError("calendar must be CalendarFeatures")
    if calendar.timestamp != local:
        raise ValueError("calendar.timestamp must match the target timestamp")
    if weather is not None and not isinstance(weather, WeatherObservation):
        raise TypeError("weather must be WeatherObservation or None")
    if not isinstance(events, list) or any(not isinstance(event, Event) for event in events):
        raise TypeError("events must be a list of Event")
    if railway is not None:
        if not isinstance(railway, RailwayFeatures):
            raise TypeError("railway must be RailwayFeatures or None")
        if railway.metro_station != station:
            raise ValueError("railway.metro_station must match station (station cannot be None with railway)")
        if railway.timestamp != local:
            raise ValueError("railway.timestamp must match the target timestamp")

    known_people = [event.expected_people for event in events if event.expected_people is not None]
    event_types = list(dict.fromkeys(event.event_type for event in events if event.event_type is not None))
    return ExternalFeatures(
        timestamp=local,
        station=station,
        weather=WeatherFeatures(**weather.model_dump(include=set(WeatherFeatures.model_fields))) if weather is not None else WeatherFeatures(),
        calendar=CalendarFeatureValues(**calendar.model_dump(exclude={"timestamp"})),
        railway=RailwayFeatureValues(**railway.model_dump(exclude={"timestamp", "metro_station"})) if railway is not None else None,
        events=EventFeatures(
            event_count=len(events), event_types=event_types,
            total_expected_people=sum(known_people) if known_people else None,
            has_event=bool(events),
        ),
    )
