"""Manual weather API smoke test."""

import argparse
import json
import sys
from contextlib import closing

from .yandex_weather import YandexWeatherError, YandexWeatherProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Текущая погода или почасовой прогноз Yandex Weather")
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--hours", type=int, help="Получить прогноз на 1–48 часов вместо текущей погоды")
    parser.add_argument("--with-feels-like", action="store_true", help="Запросить feelsLike (нужен соответствующий тариф)")
    parser.add_argument("--debug", action="store_true", help="Показать GraphQL query, variables и сообщения ошибок без ключа")
    args = parser.parse_args(argv)
    try:
        with closing(YandexWeatherProvider(include_feels_like=args.with_feels_like)) as provider:
            provider.debug = args.debug
            if args.hours is None:
                output = provider.get_current_weather(args.lat, args.lon).model_dump(mode="json")
            else:
                output = [item.model_dump(mode="json") for item in provider.get_hourly_forecast(args.lat, args.lon, args.hours)]
    except (YandexWeatherError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
