"""Manual extraction: python -m external_data.events.cli --text '...'"""

import argparse
import sys
from contextlib import closing

from pydantic import ValidationError

from .models import normalize_timestamp
from .diagnostics import redact_secrets, validation_details
from .parser import AddressGroundingError, SourceFragmentError, parse_events
from .providers.openrouter import OpenRouterError, OpenRouterProvider
from .providers.yandexgpt import YandexGPTError, YandexGPTProvider


def _timestamp(value: str):
    try:
        return normalize_timestamp(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Ожидается ISO 8601 с датой, временем и часовым поясом, например 2026-10-07T12:00:00+03:00"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Извлечение городских событий через LLM")
    parser.add_argument("--provider", choices=("openrouter", "yandex"), default="openrouter")
    parser.add_argument("--text", required=True)
    parser.add_argument("--published-at", type=_timestamp)
    parser.add_argument("--source-type")
    parser.add_argument("--source-url")
    parser.add_argument("--debug", action="store_true", help="Показать content модели в stderr с маскированием секретов")
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error("--text не должен быть пустым")
    try:
        provider_class = YandexGPTProvider if args.provider == "yandex" else OpenRouterProvider
        with closing(provider_class()) as provider:
            provider.debug = args.debug
            result = parse_events(
                args.text, published_at=args.published_at,
                source_type=args.source_type, source_url=args.source_url, provider=provider,
            )
    except (OpenRouterError, YandexGPTError) as error:
        print("Ошибка: " + redact_secrets(str(error)), file=sys.stderr)
        return 1
    except SourceFragmentError as error:
        print(f"Ошибка source_fragment: {error}", file=sys.stderr)
        return 1
    except AddressGroundingError as error:
        print(f"Ошибка address: {error}", file=sys.stderr)
        return 1
    except ValidationError as error:
        print("Ошибка: " + validation_details(error), file=sys.stderr)
        return 1
    except (TypeError, ValueError):
        print("Ошибка входных данных или структуры ответа провайдера: проверьте типы и формат аргументов.", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
