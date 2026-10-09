"""Manual railway arrival features smoke test."""

import argparse
import sys
from contextlib import closing
from datetime import datetime

from external_data.calendar.models import moscow_time

from .features import get_railway_features
from .hubs import get_hub
from .yandex_rasp import YandexRaspError, YandexRaspProvider


def _timestamp(value: str) -> datetime:
    try:
        return moscow_time(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        raise argparse.ArgumentTypeError("Нужен ISO 8601 timestamp с часовым поясом.") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прибытия железнодорожных поездов как внешний фактор спроса")
    parser.add_argument("--hub", required=True)
    parser.add_argument("--timestamp", required=True, type=_timestamp)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    try:
        hub = get_hub(args.hub)
        with closing(YandexRaspProvider(debug=args.debug)) as provider:
            result = get_railway_features(args.timestamp, hub, provider)
    except (YandexRaspError, ValueError, TypeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
