"""Real OpenAI SDK with a local HTTP transport; no credentials or network."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from unittest.mock import patch

import httpx
from openai import OpenAI

from external_data.events.models import StructuredEventExtractionResult
from external_data.events import EventExtractionResult, parse_events
from external_data.events.cli import main
from external_data.events.prompt import SYSTEM_PROMPT
from external_data.events.providers.yandexgpt import YandexGPTError, YandexGPTProvider

ENV = {
    "YANDEX_API_KEY": "unit-test-placeholder",
    "YANDEX_FOLDER_ID": "test-folder",
    "YANDEX_MODEL": "yandexgpt/latest",
}
EMPTY = {"is_event": False, "events": []}


def completion(content=json.dumps(EMPTY), *, finish_reason="stop", refusal=None):
    return {
        "id": "test", "object": "chat.completion", "created": 0,
        "model": "gpt://test-folder/yandexgpt/latest",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content, "refusal": refusal,
        }}],
    }


class YandexGPTTests(unittest.TestCase):
    def provider(self, replies):
        self.requests = []

        def handle(request):
            self.requests.append(request)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            status, body = reply
            return httpx.Response(status, json=body)

        def client_factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            self.addCleanup(client.close)
            return client

        with patch.dict("os.environ", ENV, clear=True), patch(
            "external_data.events.providers.yandexgpt.OpenAI", side_effect=client_factory,
        ) as factory:
            provider = YandexGPTProvider()
        factory.assert_called_once_with(
            api_key=ENV["YANDEX_API_KEY"], project=ENV["YANDEX_FOLDER_ID"],
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={"Authorization": "Api-Key " + ENV["YANDEX_API_KEY"]},
            timeout=60.0, max_retries=0,
        )
        return provider

    def generate(self, provider):
        return provider.generate_structured(
            system_prompt=SYSTEM_PROMPT, user_prompt='{"text":"Новость"}',
            json_schema=StructuredEventExtractionResult.model_json_schema(),
        )

    def assert_missing(self, name):
        for value in (None, "", " "):
            env = {key: value for key, value in ENV.items() if key != name}
            if value is not None:
                env[name] = value
            with self.subTest(name=name, value=value), patch.dict("os.environ", env, clear=True), patch(
                "external_data.events.providers.yandexgpt.OpenAI",
            ) as factory:
                with self.assertRaisesRegex(YandexGPTError, name):
                    YandexGPTProvider()
                factory.assert_not_called()

    def test_missing_api_key(self):
        self.assert_missing("YANDEX_API_KEY")

    def test_missing_folder_id(self):
        self.assert_missing("YANDEX_FOLDER_ID")

    def test_missing_model(self):
        self.assert_missing("YANDEX_MODEL")

    def test_rejects_full_uri_and_invalid_folder(self):
        for key, value in (("YANDEX_MODEL", "gpt://test-folder/yandexgpt/latest"),
                           ("YANDEX_MODEL", "yandexgpt//latest"),
                           ("YANDEX_FOLDER_ID", "test/folder")):
            with self.subTest(key=key), patch.dict("os.environ", {**ENV, key: value}, clear=True), patch(
                "external_data.events.providers.yandexgpt.OpenAI",
            ) as factory:
                with self.assertRaisesRegex(YandexGPTError, key):
                    YandexGPTProvider()
                factory.assert_not_called()

    def test_model_uri_auth_and_project_headers(self):
        provider = self.provider([(200, completion())])
        self.assertEqual(provider.model, "gpt://test-folder/yandexgpt/latest")
        self.assertEqual(self.generate(provider), EMPTY)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://ai.api.cloud.yandex.net/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Api-Key " + ENV["YANDEX_API_KEY"])
        self.assertEqual(request.headers["OpenAI-Project"], ENV["YANDEX_FOLDER_ID"])
        body = json.loads(request.content)
        self.assertEqual(body["model"], provider.model)
        self.assertNotIn("provider", body)  # No OpenRouter-specific routing fields.
        self.assertNotIn(ENV["YANDEX_API_KEY"], request.content.decode())

    def test_structured_output_uses_existing_schema_and_prompt(self):
        provider = self.provider([(200, completion())])
        schema = StructuredEventExtractionResult.model_json_schema()
        original = deepcopy(schema)
        provider.generate_structured(system_prompt=SYSTEM_PROMPT, user_prompt="{}", json_schema=schema)
        self.assertEqual(schema, original)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["response_format"], {
            "type": "json_schema", "json_schema": {"name": "EventExtractionResult", "schema": schema},
        })
        self.assertTrue(body["messages"][0]["content"].startswith(SYSTEM_PROMPT))
        self.assertEqual(body["messages"][1], {"role": "user", "content": "{}"})

    def test_fallback_to_json_mode(self):
        for status, message in ((400, "json_schema is not supported for this model"),
                                (422, "response_format json_schema не поддерживается")):
            with self.subTest(status=status):
                provider = self.provider([(status, {"error": {"message": message}}), (200, completion())])
                self.assertEqual(self.generate(provider), EMPTY)
                self.assertEqual(len(self.requests), 2)
                first, second = [json.loads(request.content) for request in self.requests]
                self.assertEqual(second["response_format"], {"type": "json_object"})
                self.assertEqual(first["messages"], second["messages"])
                self.assertIn('"properties"', second["messages"][0]["content"])

    def test_no_fallback_on_unrelated_http_errors(self):
        for status, message in ((400, "Invalid schema for response_format"),
                                (401, "Unauthorized"), (403, "Permission denied"),
                                (404, "Model not found"), (429, "Rate limited"), (500, "Internal error")):
            with self.subTest(status=status):
                provider = self.provider([(status, {"error": {"message": message}})])
                with self.assertRaisesRegex(YandexGPTError, f"HTTP {status}"):
                    self.generate(provider)
                self.assertEqual(len(self.requests), 1)

    def test_fallback_failure_stops(self):
        provider = self.provider([
            (400, {"error": {"message": "json_schema not supported"}}),
            (400, {"error": {"message": "json_object not supported"}}),
        ])
        with self.assertRaises(YandexGPTError):
            self.generate(provider)
        self.assertEqual(len(self.requests), 2)

    def test_pydantic_validation_error_in_both_modes(self):
        for fallback in (False, True):
            replies = [(400, {"error": {"message": "json_schema not supported"}})] if fallback else []
            replies.append((200, completion(json.dumps({"is_event": True, "events": [{
                "event_name": "Концерт", "confidence": 2,
                "event_type": None, "start_time": None, "end_time": None,
                "location_name": None, "address": None, "expected_people": None, "source_type": None,
                "source_url": None, "published_at": None, "source_fragment": "Концерт",
            }]}))))
            provider = self.provider(replies)
            with self.assertRaisesRegex(YandexGPTError, r"EventExtractionResult; поля: events.0.confidence"):
                self.generate(provider)
            self.assertEqual(len(self.requests), 2 if fallback else 1)

    def test_full_json_only(self):
        for content in ('```json\n{}\n```', 'prefix {} suffix', '[]', 'null', '', '{}{}'):
            with self.subTest(content=content):
                provider = self.provider([(200, completion(content))])
                with self.assertRaises(YandexGPTError):
                    self.generate(provider)
                self.assertEqual(len(self.requests), 1)

    def test_incomplete_refused_and_empty_responses(self):
        for response in (completion(finish_reason="length"), completion(refusal="No"),
                         completion(None), {**completion(), "choices": []}):
            with self.subTest(response=response), self.assertRaises(YandexGPTError):
                self.generate(self.provider([(200, response)]))

    def test_network_timeout_does_not_retry(self):
        provider = self.provider([httpx.ReadTimeout("test timeout")])
        with self.assertRaisesRegex(YandexGPTError, "проверьте сеть"):
            self.generate(provider)
        self.assertEqual(len(self.requests), 1)

    def test_parser_integration_and_source_metadata(self):
        text = "Завтра в 19:00 в Ледовом дворце состоится концерт"
        provider = self.provider([(200, completion(json.dumps({"is_event": True, "events": [{
            "event_name": "Концерт", "start_time": "2026-10-08T19:00:00+03:00",
            "location_name": "Ледовый дворец", "address": None, "source_fragment": text,
            "event_type": "concert", "end_time": None, "expected_people": None,
            "source_type": None, "source_url": None, "published_at": None, "confidence": None,
        }]})))])
        published = datetime.fromisoformat("2026-10-07T12:00:00+03:00")
        result = parse_events(text, published, "telegram", "https://t.me/example/42", provider=provider)
        event = result.events[0]
        self.assertTrue(result.is_event)
        self.assertIn("концерт", event.event_name.lower())
        self.assertEqual(event.event_type, "concert")
        self.assertEqual(event.location_name, "Ледовый дворец")
        self.assertIn(event.source_fragment, text)
        self.assertEqual(event.start_time.isoformat(), "2026-10-08T19:00:00+03:00")
        self.assertEqual(event.published_at, published)
        self.assertEqual(event.source_type, "telegram")
        self.assertEqual(event.source_url, "https://t.me/example/42")
        self.assertIsNone(event.expected_people)

        body = json.loads(self.requests[0].content)
        schema = body["response_format"]["json_schema"]["schema"]
        event_schema = schema["$defs"][schema["properties"]["events"]["items"]["$ref"].split("/")[-1]]
        self.assertEqual(set(event_schema["required"]), {
            "event_name", "event_type", "start_time", "end_time", "location_name", "address",
            "expected_people", "source_type", "source_url", "published_at",
            "confidence", "source_fragment",
        })
        self.assertEqual(event_schema["properties"]["source_fragment"]["type"], "string")

    def test_sparse_real_response_is_rejected_without_retry(self):
        sparse = {"events": [{"event_name": "концерт", "event_type": "concert",
                              "start_time": "2026-10-08T19:00:00+03:00"}], "is_event": True}
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                replies = [(400, {"error": {"message": "json_schema not supported"}})] if fallback else []
                replies.append((200, completion(json.dumps(sparse))))
                provider = self.provider(replies)
                with self.assertRaisesRegex(YandexGPTError, "Pydantic ValidationError.*missing"):
                    parse_events("Завтра в 19:00 в Ледовом дворце состоится концерт", provider=provider)
                self.assertEqual(len(self.requests), 2 if fallback else 1)

    def test_cli_yandex_selection(self):
        provider = self.provider([(200, completion())])
        out, err = io.StringIO(), io.StringIO()
        with patch("external_data.events.cli.YandexGPTProvider", return_value=provider), patch(
            "external_data.events.cli.OpenRouterProvider",
        ) as other, redirect_stdout(out), redirect_stderr(err):
            status = main(["--provider", "yandex", "--text", "Новость", "--published-at",
                           "2026-10-07T12:00:00+03:00", "--source-type", "telegram"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out.getvalue()), EMPTY)
        self.assertEqual(err.getvalue(), "")
        other.assert_not_called()
        self.assertTrue(provider._client.is_closed())

    def test_cli_yandex_configuration_error(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(["--provider", "yandex", "--text", "Новость"]), 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("YANDEX_API_KEY", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_cli_explicit_openrouter_still_works(self):
        from external_data.events.providers import FakeLLMProvider
        provider = FakeLLMProvider(EMPTY)
        with patch("external_data.events.cli.OpenRouterProvider", return_value=provider) as factory, patch(
            "external_data.events.cli.YandexGPTProvider",
        ) as other, patch.object(provider, "close", create=True) as close, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--provider", "openrouter", "--text", "Новость"]), 0)
            close.assert_called_once()
        factory.assert_called_once()
        other.assert_not_called()
