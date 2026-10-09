"""Content-only diagnostics. Never dump SDK objects, HTTP requests or headers."""

import json
import os
import re
import sys

from pydantic import ValidationError


def redact_secrets(value: str) -> str:
    secrets = {
        secret.strip() for name, secret in os.environ.items()
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and secret.strip()
    }
    for secret in sorted(secrets, key=len, reverse=True):
        # A model may echo a credential as a JSON-escaped string.
        for variant in {secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]}:
            value = value.replace(variant, "[REDACTED]")
    # Also hide explicit credential fields, even if their values aren't in env.
    value = re.sub(
        r'("[^"\n]*(?:api[_-]?key|token|secret|password|authorization|credential)[^"\n]*"\s*:\s*)'
        r'"(?:\\.|[^"\\])*"',
        r'\1"[REDACTED]"', value, flags=re.I,
    )
    return re.sub(r"\b(?:Bearer|Api-Key)\s+[^\s\"\\]+", "[REDACTED]", value, flags=re.I)


def debug_content(content: str | None, *, enabled: bool) -> None:
    if enabled:
        # json.dumps escapes terminal control characters and retains invalid JSON for diagnosis.
        print("DEBUG model content: " + json.dumps(redact_secrets(content or ""), ensure_ascii=False), file=sys.stderr)


def validation_details(error: ValidationError) -> str:
    errors = error.errors(include_input=False, include_context=False)
    locations = ", ".join(".".join(map(str, item["loc"])) or "result" for item in errors[:5])
    kinds = ", ".join(item["type"] for item in errors[:5])
    return redact_secrets(f"Pydantic ValidationError: Ответ не соответствует EventExtractionResult; поля: {locations}; типы: {kinds}")
