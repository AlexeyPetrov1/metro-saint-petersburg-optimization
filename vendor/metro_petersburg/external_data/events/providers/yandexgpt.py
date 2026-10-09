"""Yandex AI Studio Chat Completions adapter, configured only via env."""

import json
import os
import re
from typing import Any

from openai import APIError, APIStatusError, OpenAI
from pydantic import ValidationError

from ..models import StructuredEventExtractionResult
from ..diagnostics import debug_content, validation_details


class YandexGPTError(RuntimeError):
    """Configuration, API or extraction error safe to display in the CLI."""


def _unsupported_schema(error: APIStatusError) -> bool:
    """Downgrade only on explicit lack of format support, not arbitrary 4xx."""
    if error.status_code not in (400, 422):
        return False
    body = json.dumps(error.body, ensure_ascii=False, default=str).lower()
    format_mentioned = any(term in body for term in (
        "json_schema", "response_format", "structured output", "structured_output",
    ))
    unsupported = any(term in body for term in (
        "not supported", "unsupported", "does not support", "not support",
        "не поддерживается", "не поддерживает",
    ))
    return format_mentioned and unsupported


class YandexGPTProvider:
    debug: bool = False

    def __init__(self) -> None:
        api_key = os.environ.get("YANDEX_API_KEY", "").strip()
        folder_id = os.environ.get("YANDEX_FOLDER_ID", "").strip()
        model = os.environ.get("YANDEX_MODEL", "").strip()
        missing = [name for name, value in (
            ("YANDEX_API_KEY", api_key), ("YANDEX_FOLDER_ID", folder_id),
            ("YANDEX_MODEL", model),
        ) if not value]
        if missing:
            raise YandexGPTError("Задайте переменные окружения: " + ", ".join(missing))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
            raise YandexGPTError("YANDEX_FOLDER_ID должен содержать идентификатор каталога.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*", model):
            raise YandexGPTError("YANDEX_MODEL должен быть коротким именем, например yandexgpt/latest, без gpt://.")
        self.model = f"gpt://{folder_id}/{model}"
        self._client = OpenAI(
            api_key=api_key,
            project=folder_id,
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={"Authorization": f"Api-Key {api_key}"},
            timeout=60.0,
            max_retries=0,
        )

    def close(self) -> None:
        self._client.close()

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt + "\nJSON Schema:\n" + json.dumps(json_schema, ensure_ascii=False)},
            {"role": "user", "content": user_prompt},
        ]
        # Follow Yandex's documented schema mode without altering the Pydantic schema.
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "EventExtractionResult", "schema": json_schema},
        }
        try:
            try:
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages, response_format=response_format,
                )
            except APIStatusError as error:
                if not _unsupported_schema(error):
                    raise
                # One fallback only; preserve the trusted prompt and schema.
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    response_format={"type": "json_object"},
                )
        except APIStatusError as error:
            raise YandexGPTError(
                f"Yandex AI Studio HTTP {error.status_code}: проверьте API key, права, "
                "каталог, модель и поддержку response_format."
            ) from None
        except APIError:
            raise YandexGPTError("Не удалось получить ответ Yandex AI Studio: проверьте сеть и доступность API.") from None

        if not response.choices:
            raise YandexGPTError("Yandex AI Studio вернул ответ без choices.")
        choice = response.choices[0]
        content = choice.message.content
        debug_content(content, enabled=self.debug)
        if choice.message.refusal:
            raise YandexGPTError("Модель отказалась выполнять extraction.")
        if choice.finish_reason != "stop":
            raise YandexGPTError("Ответ модели не завершён нормально (обрезан или заблокирован).")
        if not isinstance(content, str) or not content.strip():
            raise YandexGPTError("Модель вернула пустой ответ.")
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            raise YandexGPTError("JSON parsing error: модель вернула невалидный JSON; ожидается один JSON-объект без Markdown.") from None
        if not isinstance(result, dict):
            raise YandexGPTError("Structured output error: модель должна вернуть JSON-объект EventExtractionResult.")
        try:
            StructuredEventExtractionResult.model_validate(result)
        except ValidationError as error:
            raise YandexGPTError(validation_details(error)) from None
        return result
