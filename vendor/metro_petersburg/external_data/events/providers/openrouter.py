"""OpenRouter adapter using the official OpenAI client, configured via env."""

import json
import os
from copy import deepcopy
from typing import Any

from openai import APIError, APIStatusError, OpenAI
from pydantic import ValidationError

from ..models import StructuredEventExtractionResult
from ..diagnostics import debug_content, validation_details


class OpenRouterError(RuntimeError):
    """Configuration, request or response failure safe to display in the CLI."""


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Require nullable fields explicitly; do not mutate the parser's schema."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def _unsupported_schema(error: APIStatusError) -> bool:
    # Deliberately narrow: never downgrade auth, billing, rate-limit or server errors.
    if error.status_code not in (400, 404, 422):
        return False
    body = json.dumps(error.body, ensure_ascii=False, default=str).lower()
    mentions_format = any(word in body for word in (
        "json_schema", "response_format", "structured output", "structured_outputs",
    ))
    unsupported = any(word in body for word in (
        "not supported", "unsupported", "does not support", "don't support",
        "not support", "unavailable",
    ))
    no_endpoints = "no endpoints found" in body and "support" in body
    return (mentions_format and unsupported) or (
        no_endpoints and (mentions_format or "requested parameters" in body)
    )


class OpenRouterProvider:
    debug: bool = False

    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("OPENROUTER_MODEL", "").strip()
        missing = [name for name, value in (
            ("OPENROUTER_API_KEY", api_key), ("OPENROUTER_MODEL", self.model),
        ) if not value]
        if missing:
            raise OpenRouterError("Задайте переменные окружения: " + ", ".join(missing))
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
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
        schema = _strict_schema(json_schema)
        messages = [
            {"role": "system", "content": system_prompt + "\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": user_prompt},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "EventExtractionResult", "strict": True, "schema": schema},
        }
        try:
            try:
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages, response_format=response_format,
                    extra_body={"provider": {"require_parameters": True}},
                )
            except APIStatusError as error:
                if not _unsupported_schema(error):
                    raise
                # Exactly one downgrade. The same trusted schema stays in system context.
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    response_format={"type": "json_object"},
                    extra_body={"provider": {"require_parameters": True}},
                )
        except APIStatusError as error:
            raise OpenRouterError(
                f"OpenRouter HTTP {error.status_code}: проверьте ключ, баланс, модель "
                "и поддержку response_format."
            ) from None
        except APIError:
            raise OpenRouterError("Не удалось получить ответ OpenRouter: проверьте сеть и доступность API.") from None

        if not response.choices:
            raise OpenRouterError("OpenRouter вернул ответ без choices.")
        choice = response.choices[0]
        content = choice.message.content
        debug_content(content, enabled=self.debug)
        if choice.message.refusal:
            raise OpenRouterError("Модель отказалась выполнять extraction.")
        if choice.finish_reason != "stop":
            raise OpenRouterError("Ответ модели не завершён нормально (обрезан или заблокирован).")
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError("Модель вернула пустой ответ.")
        try:
            # Parse the whole response; fenced JSON and prose intentionally fail.
            result = json.loads(content)
        except (ValueError, TypeError):
            raise OpenRouterError("JSON parsing error: модель вернула невалидный JSON; ожидается один JSON-объект без Markdown.") from None
        if not isinstance(result, dict):
            raise OpenRouterError("Structured output error: модель должна вернуть JSON-объект EventExtractionResult.")
        try:
            StructuredEventExtractionResult.model_validate(result)
        except ValidationError as error:
            raise OpenRouterError(validation_details(error)) from None
        return result
