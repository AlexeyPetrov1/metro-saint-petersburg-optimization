"""Exact quotations, bounded correction and content-only debug diagnostics."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from external_data.events import parse_events
from external_data.events.cli import main
from external_data.events.diagnostics import debug_content
from external_data.events.parser import FRAGMENT_RETRY, SourceFragmentError

TEXT = "Завтра в 19:00 в Ледовом дворце состоится концерт"
BAD = "В Ледовом дворце состоится концерт."
PUBLISHED = datetime.fromisoformat("2026-10-07T12:00:00+03:00")


def extraction(fragment):
    return {"is_event": True, "events": [{
        "event_name": "Концерт", "start_time": "2026-10-08T19:00:00+03:00",
        "event_type": "concert", "end_time": None, "location_name": "Ледовый дворец",
        "expected_people": None, "source_type": None, "source_url": None,
        "published_at": None, "confidence": None, "address": None,
        "source_fragment": fragment,
    }]}


class FragmentTests(unittest.TestCase):
    def test_exact_fragment(self):
        provider = Mock()
        provider.generate_structured.return_value = extraction(TEXT)
        result = parse_events(TEXT, provider=provider)
        self.assertEqual(result.events[0].source_fragment, TEXT)
        provider.generate_structured.assert_called_once()

    def test_shorter_contiguous_fragment(self):
        fragment = "в Ледовом дворце состоится концерт"
        provider = Mock()
        provider.generate_structured.return_value = extraction(fragment)
        self.assertEqual(parse_events(TEXT, provider=provider).events[0].source_fragment, fragment)
        provider.generate_structured.assert_called_once()

    def test_paraphrase_retries_once_with_context(self):
        provider = Mock()
        provider.generate_structured.side_effect = [extraction(BAD), extraction(TEXT)]
        result = parse_events(TEXT, PUBLISHED, "telegram", "https://t.me/example/42", provider=provider)
        self.assertEqual(result.events[0].source_fragment, TEXT)
        self.assertEqual(result.events[0].published_at, PUBLISHED)
        self.assertEqual(result.events[0].source_url, "https://t.me/example/42")
        self.assertEqual(provider.generate_structured.call_count, 2)
        first, second = [call.kwargs for call in provider.generate_structured.call_args_list]
        self.assertEqual(second["json_schema"], first["json_schema"])
        self.assertIn(FRAGMENT_RETRY, second["system_prompt"])
        envelope = json.loads(second["user_prompt"])
        self.assertEqual(envelope["SOURCE_TEXT"], TEXT)
        self.assertEqual(envelope["previous_extraction"], extraction(BAD))
        self.assertEqual(envelope["text"], TEXT)

    def test_invalid_twice_raises_diagnostic_error(self):
        for fragment in (BAD, " " + TEXT + " ", None):
            with self.subTest(fragment=fragment):
                provider = Mock()
                provider.generate_structured.return_value = extraction(fragment)
                with self.assertRaisesRegex(SourceFragmentError, r"events\[0\].source_fragment.*Один повторный") as error:
                    parse_events(TEXT, provider=provider)
                self.assertEqual(provider.generate_structured.call_count, 2)
                self.assertNotIn(BAD, str(error.exception))

    def test_pydantic_failure_does_not_retry(self):
        from pydantic import ValidationError
        provider = Mock()
        provider.generate_structured.return_value = {"is_event": True, "events": []}
        with self.assertRaises(ValidationError):
            parse_events(TEXT, provider=provider)
        provider.generate_structured.assert_called_once()

    def test_cli_distinguishes_fragment_and_validation_errors(self):
        for response, expected in ((extraction(BAD), "Ошибка source_fragment:"),
                                   ({"is_event": True, "events": []}, "Pydantic ValidationError")):
            with self.subTest(expected=expected):
                provider = Mock()
                provider.generate_structured.return_value = response
                out, err = io.StringIO(), io.StringIO()
                with patch("external_data.events.cli.YandexGPTProvider", return_value=provider), redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(main(["--provider", "yandex", "--text", TEXT]), 1)
                self.assertIn(expected, err.getvalue())
                self.assertEqual(out.getvalue(), "")


class DebugTests(unittest.TestCase):
    def run_cli(self, provider_name, contents, *, debug):
        key = "test-sensitive-key-do-not-print"
        env = {
            "YANDEX_API_KEY": key, "YANDEX_FOLDER_ID": "test-folder", "YANDEX_MODEL": "yandexgpt/latest",
            "OPENROUTER_API_KEY": key, "OPENROUTER_MODEL": "test/model",
            "TEST_PASSWORD": "secret-password-do-not-print",
        }
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": contents.pop(0),
                }}],
            })

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            self.addCleanup(client.close)
            return client

        module = "yandexgpt" if provider_name == "yandex" else "openrouter"
        out, err = io.StringIO(), io.StringIO()
        args = ["--provider", provider_name, "--text", TEXT]
        if debug:
            args.append("--debug")
        with patch.dict("os.environ", env, clear=True), patch(
            f"external_data.events.providers.{module}.OpenAI", side_effect=factory,
        ), redirect_stdout(out), redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue(), requests

    def test_debug_redacts_keys_and_secrets_in_content(self):
        content = json.dumps({
            "event_name": "test-sensitive-key-do-not-print secret-password-do-not-print",
            "api_key": "another-secret-not-in-env", "authorization": "Bearer unknown-token",
        })
        for name in ("yandex", "openrouter"):
            with self.subTest(provider=name):
                status, out, err, requests = self.run_cli(name, [content], debug=True)
                self.assertEqual(status, 1)
                self.assertEqual(out, "")
                self.assertIn("DEBUG model content:", err)
                self.assertIn("Pydantic ValidationError", err)
                for secret in ("test-sensitive-key-do-not-print", "secret-password-do-not-print",
                               "another-secret-not-in-env", "unknown-token"):
                    self.assertNotIn(secret, err)
                self.assertIn("[REDACTED]", err)
                self.assertNotIn("OpenAI-Project", err)
                self.assertEqual(len(requests), 1)

    def test_debug_shows_both_fragment_attempts(self):
        status, out, err, requests = self.run_cli("yandex", [json.dumps(extraction(BAD)), json.dumps(extraction(TEXT))], debug=True)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)["events"][0]["source_fragment"], TEXT)
        self.assertEqual(err.count("DEBUG model content:"), 2)
        self.assertEqual(len(requests), 2)

    def test_no_debug_by_default(self):
        status, out, err, _ = self.run_cli("yandex", [json.dumps(extraction(TEXT))], debug=False)
        self.assertEqual(status, 0)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out)["events"][0]["source_fragment"], TEXT)

    def test_debug_shows_invalid_json_and_parsing_reason(self):
        status, out, err, _ = self.run_cli("yandex", ["not-json-test-marker"], debug=True)
        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertIn("not-json-test-marker", err)
        self.assertIn("JSON parsing error", err)

    def test_structured_output_error_is_distinct(self):
        status, out, err, _ = self.run_cli("yandex", ["[]"], debug=True)
        self.assertEqual(status, 1)
        self.assertIn("Structured output error", err)
        self.assertEqual(out, "")

    def test_debug_masks_json_escaped_secret_and_terminal_controls(self):
        secret = 'test-"secret\\value'
        out = io.StringIO()
        with patch.dict("os.environ", {"TEST_API_KEY": secret}, clear=True), redirect_stderr(out):
            debug_content(json.dumps({"value": secret}) + "\x1b[31m", enabled=True)
        self.assertNotIn("secret", out.getvalue())
        self.assertIn("[REDACTED]", out.getvalue())
        self.assertNotIn("\x1b", out.getvalue())
