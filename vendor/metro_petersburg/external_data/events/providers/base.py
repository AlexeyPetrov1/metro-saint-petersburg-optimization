"""Provider boundary: adapters may use a native structured-output API."""

from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a decoded JSON object, never prose or Markdown.

        Adapters own SDK calls, API errors and JSON decoding. Use native schema
        support when available; otherwise decode the entire response with
        json.loads, without searching for braces or repairing model output.
        """
        ...
