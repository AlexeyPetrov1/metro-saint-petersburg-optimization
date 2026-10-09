"""Regression tests for required-nullable fields on the LLM wire contract."""

import unittest
from copy import deepcopy
from datetime import datetime

from pydantic import ValidationError

from external_data.events import Event, parse_events
from external_data.events.models import StructuredEventExtractionResult
from external_data.events.providers import FakeLLMProvider

FIELDS = {
    "event_name", "event_type", "start_time", "end_time", "location_name", "address",
    "expected_people", "source_type", "source_url", "published_at",
    "confidence", "source_fragment",
}
TEXT = "Завтра в 19:00 в Ледовом дворце состоится концерт"
EVENT = {
    "event_name": "концерт", "event_type": "concert",
    "start_time": "2026-10-08T19:00:00+03:00", "end_time": None,
    "location_name": "Ледовый дворец", "address": None, "expected_people": None,
    "source_type": None, "source_url": None, "published_at": None,
    "confidence": None, "source_fragment": TEXT,
}


class SchemaTests(unittest.TestCase):
    def test_all_event_keys_required_and_nullable_only_where_allowed(self):
        schema = StructuredEventExtractionResult.model_json_schema()
        ref = schema["properties"]["events"]["items"]["$ref"].split("/")[-1]
        event = schema["$defs"][ref]
        self.assertEqual(set(event["properties"]), FIELDS)
        self.assertEqual(set(event["required"]), FIELDS)
        self.assertFalse(event["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"is_event", "events"})
        for name, definition in event["properties"].items():
            with self.subTest(field=name):
                self.assertNotIn("default", definition)
                if name in {"event_name", "source_fragment"}:
                    self.assertEqual(definition["type"], "string")
                    self.assertEqual(definition["minLength"], 1)
                else:
                    self.assertIn({"type": "null"}, definition["anyOf"])

    def test_omitting_any_field_is_validation_error(self):
        for name in FIELDS:
            with self.subTest(field=name):
                event = deepcopy(EVENT)
                del event[name]
                with self.assertRaises(ValidationError) as error:
                    StructuredEventExtractionResult.model_validate({"is_event": True, "events": [event]})
                self.assertIn(("events", 0, name), [item["loc"] for item in error.exception.errors()])

    def test_unknown_values_are_null_but_evidence_is_required(self):
        event = {name: None for name in FIELDS}
        event.update(event_name="Концерт", source_fragment="Концерт")
        result = StructuredEventExtractionResult.model_validate({"is_event": True, "events": [event]})
        self.assertIsNone(result.events[0].location_name)
        for fragment in (None, "", " "):
            with self.subTest(fragment=fragment), self.assertRaises(ValidationError):
                StructuredEventExtractionResult.model_validate({
                    "is_event": True, "events": [{**event, "source_fragment": fragment}],
                })

    def test_non_event_is_empty_and_top_level_keys_are_required(self):
        value = {"is_event": False, "events": []}
        self.assertEqual(StructuredEventExtractionResult.model_validate(value).model_dump(), value)
        for response in ({"events": []}, {"is_event": False},
                         {"is_event": True, "events": []}, {"is_event": False, "events": [EVENT]}):
            with self.subTest(response=response), self.assertRaises(ValidationError):
                StructuredEventExtractionResult.model_validate(response)

    def test_metadata_is_authoritative_and_public_format_unchanged(self):
        published = datetime.fromisoformat("2026-10-07T12:00:00+03:00")
        event = {**EVENT, "source_type": "invented", "source_url": "invented",
                 "published_at": "2000-01-01T12:00:00+03:00"}
        # The wire contract allows null metadata; even plausible non-null model
        # values can never replace caller-provided metadata in the public result.
        StructuredEventExtractionResult.model_validate({"is_event": True, "events": [event]})
        provider = FakeLLMProvider({"is_event": True, "events": [event]})
        result = parse_events(TEXT, published, "telegram", "https://t.me/example/42", provider=provider)
        self.assertIs(type(result.events[0]), Event)
        self.assertEqual(set(result.model_dump()["events"][0]), FIELDS)
        self.assertEqual(result.events[0].published_at, published)
        self.assertEqual(result.events[0].source_type, "telegram")
        self.assertEqual(result.events[0].source_url, "https://t.me/example/42")
        without_metadata = parse_events(TEXT, provider=provider).events[0]
        self.assertIsNone(without_metadata.source_type)
        self.assertIsNone(without_metadata.source_url)
        self.assertIsNone(without_metadata.published_at)
