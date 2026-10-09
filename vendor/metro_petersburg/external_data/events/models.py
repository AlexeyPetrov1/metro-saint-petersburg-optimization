"""Validated extraction contract; all timestamps carry an explicit timezone."""

from datetime import datetime
from typing import Annotated, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

MOSCOW = ZoneInfo("Europe/Moscow")


def normalize_timestamp(value: object) -> datetime:
    """Reject ambiguous dates/epochs and normalize aware timestamps to Moscow."""
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            raise ValueError("Timestamp must include a date, time and UTC offset")
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be an ISO 8601 string or datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must have an explicit timezone")
    return value.astimezone(MOSCOW)


Timestamp = Annotated[datetime, BeforeValidator(normalize_timestamp)]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_name: str = Field(min_length=1, strict=True)
    event_type: str | None = None
    start_time: Timestamp | None = None
    end_time: Timestamp | None = None
    location_name: str | None = None
    address: str | None = Field(default=None, min_length=1, strict=True)
    expected_people: int | None = Field(default=None, ge=0, strict=True)
    source_type: str | None = None
    source_url: str | None = None
    published_at: Timestamp | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, strict=True)
    source_fragment: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def chronological_order(self) -> Self:
        if self.start_time and self.end_time and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        return self


class EventExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_event: bool = Field(strict=True)
    events: list[Event]

    @model_validator(mode="after")
    def consistent_event_flag(self) -> Self:
        if self.is_event != bool(self.events):
            raise ValueError("is_event must equal bool(events)")
        return self


class StructuredEvent(Event):
    """LLM wire contract: required keys, nullable values, mandatory evidence.

    Keep Event's public constructor defaults for existing callers. The inherited
    validators and explicit required fields describe exactly what APIs must emit.
    """

    event_name: str = Field(min_length=1, strict=True, description=(
        "Short event, artist or show name without reporting verbs. Reuse the name "
        "for occurrences only when immediate source context unambiguously links them."
    ))
    event_type: str | None = Field(description=(
        "Classify from source text only. A match with no stated sport is sport_event. "
        "football_match requires explicit or unambiguous football evidence in the text; "
        "never infer it from external knowledge of teams or venues."
    ))
    start_time: Timestamp | None = Field(description=(
        "Start of this single occurrence in Europe/Moscow. Separate listed dates "
        "into separate events; expand date ranges only for explicitly daily occurrences."
    ))
    end_time: Timestamp | None = Field(description=(
        "Only an explicitly stated ending moment in source text; otherwise null. "
        "Never equal start_time, use 23:59:59 or end of day, infer duration, "
        "or encode the next occurrence or a list of dates. Duration alone is insufficient."
    ))
    location_name: str | None = Field(description=(
        "Venue stated in source text, including unambiguous immediate context "
        "shared by occurrences; null if absent or ambiguous."
    ))
    expected_people: int | None = Field(ge=0, strict=True)
    address: str | None = Field(min_length=1, strict=True, description=(
        "Address explicitly present in source text, copied as a contiguous substring; "
        "case and whitespace normalization only. Otherwise null. Never recall, "
        "look up or infer an address or house number from venue name or world knowledge."
    ))
    source_type: str | None = Field(description="Return null; parser supplies source metadata.")
    source_url: str | None = Field(description="Return null; parser supplies source metadata.")
    published_at: Timestamp | None = Field(description="Return null; parser supplies source metadata.")
    confidence: float | None = Field(ge=0, le=1, strict=True)
    source_fragment: str = Field(min_length=1, strict=True, description=(
        "Exact contiguous substring copied character-for-character from source text. "
        "Support this occurrence's date when a date is stated; shared name/venue may come from "
        "unambiguous preceding context. Never join disjoint fragments or alter characters."
    ))


class StructuredEventExtractionResult(EventExtractionResult):
    """Strict LLM response; public result remains EventExtractionResult."""

    events: list[StructuredEvent] = Field(description=(
        "One Event per separate occurrence, including separately listed dates. "
        "Expand a date range only if daily occurrences are explicit; do not split "
        "one continuous multi-day event. Never encode multiple occurrences in end_time."
    ))
