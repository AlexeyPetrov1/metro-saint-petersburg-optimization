"""Manual Yandex Geocoder smoke test; stdout JSON, stderr safe diagnostics."""

import argparse
import sys
from contextlib import closing

from .geocoder import GeocoderError, YandexGeocoderProvider
from .resolver import LocationResolutionError, resolve_location_to_line1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ближайшие станции линии 1 к месту события")
    parser.add_argument("--location", required=True)
    parser.add_argument("--address", help="Явный адрес площадки; имеет приоритет перед названием")
    parser.add_argument("--top-k", type=int, choices=range(1, 20), default=3)
    parser.add_argument("--debug", action="store_true", help="Показать query, HTTP status, адрес и координаты без ключа")
    args = parser.parse_args(argv)
    try:
        with closing(YandexGeocoderProvider(debug=args.debug)) as geocoder:
            result = resolve_location_to_line1(args.location, geocoder, args.top_k, address=args.address)
    except (GeocoderError, LocationResolutionError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
