import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from external_data.events import Event, parse_events
from external_data.events.cli import main
from external_data.events.models import StructuredEventExtractionResult
from external_data.events.parser import ADDRESS_RETRY, AddressGroundingError, SourceFragmentError

NO_ADDRESS = "12 октября в 19:30 на СКА Арене пройдет матч"
ADDRESS = "проспект Юрия Гагарина, 8"
WITH_ADDRESS = f"12 октября в 19:30 на СКА Арене по адресу {ADDRESS} пройдет матч"


def response(text, address):
    event = Event(event_name="Матч", location_name="СКА Арена", address=address,
                  source_fragment=text).model_dump()
    return {"is_event": True, "events": [event]}


class AddressTests(unittest.TestCase):
    def test_arena_without_address_is_null(self):
        provider = Mock()
        provider.generate_structured.return_value = response(NO_ADDRESS, None)
        result = parse_events(NO_ADDRESS, provider=provider)
        self.assertEqual(result.events[0].location_name, "СКА Арена")
        self.assertIsNone(result.events[0].address)
        provider.generate_structured.assert_called_once()

    def test_explicit_address_is_grounded(self):
        provider = Mock()
        provider.generate_structured.return_value = response(WITH_ADDRESS, ADDRESS)
        result = parse_events(WITH_ADDRESS, provider=provider)
        self.assertEqual(result.events[0].address, ADDRESS)
        self.assertIn(result.events[0].address, WITH_ADDRESS)

    def test_case_and_whitespace_normalization(self):
        text = WITH_ADDRESS.replace(ADDRESS, "ПРОСПЕКТ  Юрия\nГагарина,\t8")
        provider = Mock()
        provider.generate_structured.return_value = response(text, ADDRESS)
        self.assertEqual(parse_events(text, provider=provider).events[0].address, ADDRESS)

    def test_hallucinated_address_corrected_to_null_on_shared_retry(self):
        provider = Mock()
        provider.generate_structured.side_effect = [response(NO_ADDRESS, ADDRESS), response(NO_ADDRESS, None)]
        self.assertIsNone(parse_events(NO_ADDRESS, provider=provider).events[0].address)
        self.assertEqual(provider.generate_structured.call_count, 2)
        retry = provider.generate_structured.call_args.kwargs
        self.assertIn(ADDRESS_RETRY, retry["system_prompt"])
        self.assertEqual(json.loads(retry["user_prompt"])["SOURCE_TEXT"], NO_ADDRESS)

    def test_hallucinated_address_twice_fails_safely(self):
        provider = Mock()
        provider.generate_structured.return_value = response(NO_ADDRESS, ADDRESS)
        with self.assertRaisesRegex(AddressGroundingError, r"events\[0\].address.*Один повторный") as error:
            parse_events(NO_ADDRESS, provider=provider)
        self.assertNotIn(ADDRESS, str(error.exception))
        self.assertEqual(provider.generate_structured.call_count, 2)

    def test_retry_cannot_change_house_number(self):
        provider = Mock()
        provider.generate_structured.side_effect = [response(WITH_ADDRESS, ADDRESS.replace("8", "9")), response(WITH_ADDRESS, ADDRESS)]
        self.assertEqual(parse_events(WITH_ADDRESS, provider=provider).events[0].address, ADDRESS)
        self.assertEqual(provider.generate_structured.call_count, 2)

    def test_fragment_and_address_share_one_retry(self):
        first = response(NO_ADDRESS, None)
        first["events"][0]["source_fragment"] = "Несуществующая цитата"
        provider = Mock()
        provider.generate_structured.side_effect = [first, response(NO_ADDRESS, ADDRESS)]
        with self.assertRaises(AddressGroundingError):
            parse_events(NO_ADDRESS, provider=provider)
        self.assertEqual(provider.generate_structured.call_count, 2)

    def test_address_does_not_relax_exact_fragment(self):
        bad = response(WITH_ADDRESS, ADDRESS)
        bad["events"][0]["source_fragment"] = WITH_ADDRESS.upper()
        provider = Mock()
        provider.generate_structured.return_value = bad
        with self.assertRaises(SourceFragmentError):
            parse_events(WITH_ADDRESS, provider=provider)

    def test_required_nullable_schema_and_public_default(self):
        schema = StructuredEventExtractionResult.model_json_schema()["$defs"]["StructuredEvent"]
        self.assertIn("address", schema["required"])
        self.assertIn({"type": "null"}, schema["properties"]["address"]["anyOf"])
        self.assertIsNone(Event(event_name="Матч").address)
        StructuredEventExtractionResult.model_validate(response(NO_ADDRESS, None))

    def test_cli_address_error_is_diagnostic(self):
        provider = Mock()
        provider.generate_structured.return_value = response(NO_ADDRESS, ADDRESS)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("external_data.events.cli.YandexGPTProvider", return_value=provider), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["--provider", "yandex", "--text", NO_ADDRESS]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Ошибка address:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
