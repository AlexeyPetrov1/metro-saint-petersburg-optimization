"""Local calendar CLI. An explicit date table is required, no weekday fallback."""

import argparse
import json
import sys
from datetime import datetime

from pydantic import ValidationError

from .calendar_features import get_calendar_features
from .models import moscow_time
from .providers import CalendarDataUnavailableError, MappingCalendarProvider


def _timestamp(value: str) -> datetime:
    try:
        return moscow_time(datetime.fromisoformat(value))
    except ValueError:
        raise argparse.ArgumentTypeError("Укажите ISO 8601 datetime с часовым поясом, например 2026-10-10T18:00:00+03:00") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Календарные признаки по локальному производственному календарю")
    parser.add_argument("--timestamp", required=True, type=_timestamp)
    parser.add_argument("--calendar-file", required=True, help="JSON-таблица дат; без неё праздники и переносы неизвестны")
    args = parser.parse_args(argv)
    try:
        provider = MappingCalendarProvider.from_file(args.calendar_file)
        result = get_calendar_features(args.timestamp, provider=provider)
    except CalendarDataUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        print("Не удалось прочитать календарь: проверьте файл, даты, day_type и is_preholiday.", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
