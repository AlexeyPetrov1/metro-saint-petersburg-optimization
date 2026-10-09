"""Semantic regression fixtures, not an evaluation of live LLM reasoning.

The fake returns gold answers. Tests check their wire validity, preservation by
the parser, exact quotations and delivery of rules/schema to the provider.
"""

import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from external_data.events import parse_events
from external_data.events.models import StructuredEventExtractionResult
from external_data.events.prompt import SYSTEM_PROMPT
from external_data.events.providers import FakeLLMProvider

CASES = {case["id"]: case for case in json.loads(
    (Path(__file__).parents[1] / "examples" / "sample_events.json").read_text(encoding="utf-8")
)}


class ExtractionRuleTests(unittest.TestCase):
    def extract(self, identifier):
        case = CASES[identifier]
        StructuredEventExtractionResult.model_validate(case["response"])
        provider = Mock(wraps=FakeLLMProvider(case["response"]))
        result = parse_events(
            case["text"], datetime.fromisoformat(case["published_at"]),
            case["source_type"], case["source_url"], provider=provider,
        )
        provider.generate_structured.assert_called_once()
        self.assertTrue(result.is_event)
        for event in result.events:
            self.assertIn(event.source_fragment, case["text"])
        return result

    def test_case_a_three_occurrences_with_shared_context(self):
        result = self.extract("separate_concert_occurrences")
        self.assertEqual(len(result.events), 3)
        self.assertEqual([event.start_time.isoformat() for event in result.events], [
            "2026-10-09T20:00:00+03:00", "2026-10-10T19:00:00+03:00", "2026-10-11T19:00:00+03:00",
        ])
        for event in result.events:
            self.assertEqual(event.event_name, "Сергей Лазарев")
            self.assertEqual(event.event_type, "concert")
            self.assertEqual(event.location_name, "СКА Арена")
            self.assertIsNone(event.end_time)
        # The date quote needn't repeat the artist or venue from the preceding sentence.
        for event in result.events[1:]:
            self.assertEqual(event.source_fragment, "Концерты также пройдут 10 и 11 октября в 19:00.")

    def test_case_b_match_without_stated_sport(self):
        result = self.extract("match_unspecified_sport")
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.event_type, "sport_event")
        self.assertEqual(event.event_name, "Шанхай Дрэгонс — Динамо Минск")
        self.assertEqual(event.start_time.isoformat(), "2026-10-12T19:30:00+03:00")
        self.assertIsNone(event.end_time)
        self.assertEqual(event.location_name, "СКА Арена")

    def test_case_c_explicit_end(self):
        result = self.extract("explicit_concert_end")
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.start_time.isoformat(), "2026-10-12T19:00:00+03:00")
        self.assertEqual(event.end_time.isoformat(), "2026-10-12T22:30:00+03:00")
        self.assertIsNone(event.location_name)

    def test_rules_and_schema_descriptions_reach_provider(self):
        provider = Mock(wraps=FakeLLMProvider({"is_event": False, "events": []}))
        parse_events("Городская новость", provider=provider)
        request = provider.generate_structured.call_args.kwargs
        self.assertEqual(request["system_prompt"], SYSTEM_PROMPT)
        # Guard against losing semantic rules while keeping the same JSON shape.
        for rule in ("Каждое отдельное проведение", "ТОЛЬКО при явном указании ежедневных",
                     "Слово «матч» без указания вида спорта", "ЗАПРЕЩЕНО end_time = start_time",
                     "Даже точная", "Не склеивай", "Do not add punctuation."):
            self.assertIn(rule, request["system_prompt"])
        schema = request["json_schema"]
        ref = schema["properties"]["events"]["items"]["$ref"].split("/")[-1]
        fields = schema["$defs"][ref]["properties"]
        for name, rule in {
            "event_name": "without reporting verbs",
            "event_type": "no stated sport is sport_event",
            "start_time": "Separate listed dates",
            "end_time": "Only an explicitly stated ending moment",
            "location_name": "unambiguous immediate context",
            "source_fragment": "character-for-character",
        }.items():
            self.assertIn(rule, fields[name]["description"])
        self.assertIn("One Event per separate occurrence", schema["properties"]["events"]["description"])
