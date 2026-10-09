"""Deterministic time features plus provider-owned classification."""

from datetime import datetime

from .models import CalendarFeatures, moscow_time
from .providers import CalendarDay, CalendarProvider


def get_calendar_features(timestamp: datetime, *, provider: CalendarProvider) -> CalendarFeatures:
    """Use the source classification after converting to the Moscow date."""
    local = moscow_time(timestamp)
    day = CalendarDay.model_validate(provider.get_day(local.date()))
    return CalendarFeatures(
        timestamp=local,
        day_of_week=local.weekday(), hour=local.hour, minute=local.minute,
        is_weekend=day.day_type == "weekend",
        is_workday=day.day_type == "workday",
        is_holiday=day.day_type == "holiday",
        is_preholiday=day.is_preholiday,
        day_type=day.day_type,
    )
