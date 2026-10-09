"""Fixed fixture provider. It does not interpret the supplied text."""

from copy import deepcopy
from typing import Any


class FakeLLMProvider:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = deepcopy(response)

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return deepcopy(self._response)
