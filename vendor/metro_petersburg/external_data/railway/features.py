"""Inclusive arrival windows with no passenger estimation or feature integration."""

from datetime import datetime, timedelta

from external_data.calendar.models import moscow_time

from .models import RailwayArrival, RailwayFeatures, RailwayHub
from .provider import RailwayProvider


def deduplicate_arrivals(arrivals: list[RailwayArrival]) -> list[RailwayArrival]:
    """Stable deduplication; anonymous records are retained rather than merged."""
    seen = set()
    result = []
    for arrival in arrivals:
        if not isinstance(arrival, RailwayArrival):
            raise TypeError("arrivals must contain RailwayArrival models")
        if arrival.train_number is not None:
            identity = ("number", arrival.train_number)
        elif arrival.title is not None:
            identity = ("title", " ".join(arrival.title.casefold().split()))
        else:
            result.append(arrival)
            continue
        key = (arrival.station_code, arrival.arrival_time, arrival.transport_type, identity)
        if key not in seen:
            result.append(arrival)
            seen.add(key)
    return result


def build_railway_features(
    timestamp: datetime, hub: RailwayHub, arrivals: list[RailwayArrival],
) -> RailwayFeatures:
    local = moscow_time(timestamp)
    if not isinstance(hub, RailwayHub):
        raise TypeError("hub must be RailwayHub")
    future = [
        ((arrival.arrival_time - local).total_seconds() / 60, arrival.transport_type)
        for arrival in deduplicate_arrivals(arrivals)
        if arrival.station_code == hub.rasp_station_code and arrival.arrival_time >= local
    ]
    return RailwayFeatures(
        timestamp=local, metro_station=hub.metro_station, railway_name=hub.railway_name,
        **{f"arrivals_next_{window}m": sum(minutes <= window for minutes, _ in future)
           for window in (15, 30, 60, 120)},
        train_arrivals_next_30m=sum(minutes <= 30 and kind == "train" for minutes, kind in future),
        suburban_arrivals_next_30m=sum(minutes <= 30 and kind == "suburban" for minutes, kind in future),
        minutes_to_next_arrival=min((minutes for minutes, _ in future), default=None),
    )


def get_railway_features(timestamp: datetime, hub: RailwayHub, provider: RailwayProvider) -> RailwayFeatures:
    """Fetch every Moscow calendar date touched by the inclusive 120-minute window."""
    local = moscow_time(timestamp)
    if not isinstance(hub, RailwayHub):
        raise TypeError("hub must be RailwayHub")
    day, last_day = local.date(), (local + timedelta(minutes=120)).date()
    arrivals = []
    while day <= last_day:
        arrivals.extend(provider.get_arrivals(hub.rasp_station_code, day))
        day += timedelta(days=1)
    return build_railway_features(local, hub, arrivals)
