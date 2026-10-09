import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock

from pydantic import ValidationError

from external_data.calendar import (
    CalendarDataUnavailableError, CalendarDay, CalendarFeatures,
    FakeCalendarProvider, MappingCalendarProvider, get_calendar_features,
)
from external_data.calendar.cli import main


class CalendarTests(unittest.TestCase):
    def features(self, timestamp, day_type, preholiday=None):
        moment = datetime.fromisoformat(timestamp)
        day = date.fromisoformat(timestamp[:10])
        provider = FakeCalendarProvider({day: CalendarDay(day_type=day_type, is_preholiday=preholiday)})
        return get_calendar_features(moment, provider=provider)

    def test_monday(self):
        result = self.features("2026-10-05T18:42:17+03:00", "workday", False)
        self.assertEqual((result.day_of_week, result.hour, result.minute), (0, 18, 42))
        self.assertEqual(result.day_type, "workday")
        self.assertEqual((result.is_workday, result.is_weekend, result.is_holiday), (True, False, False))
        self.assertFalse(result.is_preholiday)
        self.assertEqual(result.timestamp.second, 17)

    def test_saturday(self):
        result = self.features("2026-10-10T18:00:00+03:00", "weekend")
        self.assertEqual(result.day_of_week, 5)
        self.assertEqual(result.day_type, "weekend")
        self.assertEqual((result.is_workday, result.is_weekend, result.is_holiday), (False, True, False))

    def test_sunday(self):
        result = self.features("2026-10-11T09:15:00+03:00", "weekend")
        self.assertEqual((result.day_of_week, result.hour, result.minute), (6, 9, 15))
        self.assertTrue(result.is_weekend)

    def test_official_holiday_from_fake(self):
        # Classification comes from the test source, not from date-specific code.
        result = self.features("2026-01-01T12:00:00+03:00", "holiday", False)
        self.assertEqual(result.day_type, "holiday")
        self.assertEqual((result.is_workday, result.is_weekend, result.is_holiday), (False, False, True))

    def test_working_saturday(self):
        # Synthetic override: no assertion about the official status of this date.
        result = self.features("2026-10-10T18:00:00+03:00", "workday")
        self.assertEqual(result.day_of_week, 5)
        self.assertTrue(result.is_workday)
        self.assertFalse(result.is_weekend)
        self.assertFalse(result.is_holiday)
        self.assertEqual(result.day_type, "workday")

    def test_transferred_weekday_off(self):
        result = self.features("2026-10-05T08:00:00+03:00", "weekend")
        self.assertEqual(result.day_of_week, 0)
        self.assertTrue(result.is_weekend)
        self.assertFalse(result.is_workday)

    def test_holiday_on_sunday(self):
        result = self.features("2026-10-11T09:00:00+03:00", "holiday")
        self.assertTrue(result.is_holiday)
        self.assertFalse(result.is_weekend)
        self.assertEqual(result.day_type, "holiday")

    def test_preholiday_is_only_from_provider(self):
        for flag in (True, False, None):
            with self.subTest(flag=flag):
                result = self.features("2026-10-05T08:00:00+03:00", "workday", flag)
                self.assertIs(result.is_preholiday, flag)

    def test_utc_conversion_uses_moscow_date_for_provider(self):
        provider = Mock()
        provider.get_day.return_value = CalendarDay(day_type="workday")
        result = get_calendar_features(datetime.fromisoformat("2026-10-04T22:15:00Z"), provider=provider)
        provider.get_day.assert_called_once_with(date(2026, 10, 5))
        self.assertEqual(result.timestamp.isoformat(), "2026-10-05T01:15:00+03:00")
        self.assertEqual((result.day_of_week, result.hour, result.minute), (0, 1, 15))

    def test_positive_offset_and_year_boundary(self):
        provider = Mock()
        provider.get_day.return_value = CalendarDay(day_type="holiday")
        result = get_calendar_features(datetime.fromisoformat("2027-01-01T01:20:00+09:00"), provider=provider)
        provider.get_day.assert_called_once_with(date(2026, 12, 31))
        self.assertEqual(result.timestamp.isoformat(), "2026-12-31T19:20:00+03:00")

    def test_naive_and_non_datetime_rejected_before_provider(self):
        provider = Mock()
        with self.assertRaises(ValueError):
            get_calendar_features(datetime(2026, 10, 10), provider=provider)
        with self.assertRaises(TypeError):
            get_calendar_features("2026-10-10", provider=provider)
        provider.get_day.assert_not_called()

    def test_unknown_date_does_not_guess(self):
        with self.assertRaisesRegex(CalendarDataUnavailableError, "2026-10-10"):
            get_calendar_features(datetime.fromisoformat("2026-10-10T18:00:00+03:00"), provider=FakeCalendarProvider({}))

    def test_provider_errors_propagate(self):
        provider = Mock()
        provider.get_day.side_effect = RuntimeError("calendar unavailable")
        with self.assertRaisesRegex(RuntimeError, "calendar unavailable"):
            get_calendar_features(datetime.fromisoformat("2026-10-10T18:00:00+03:00"), provider=provider)

    def test_roundtrip_and_inconsistent_values(self):
        result = self.features("2026-10-05T08:30:00+03:00", "workday")
        self.assertEqual(CalendarFeatures.model_validate_json(result.model_dump_json()), result)
        for change in ({"day_of_week": 6}, {"hour": 25}, {"minute": 60}, {"is_weekend": True},
                       {"is_workday": "true"}, {"day_type": "unknown"}, {"timestamp": "2026-10-05T08:30:00"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                CalendarFeatures.model_validate({**result.model_dump(), **change})

    def test_local_file_cli_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.json"
            path.write_text(json.dumps({"2026-10-10": {"day_type": "weekend", "is_preholiday": None}}))
            provider = MappingCalendarProvider.from_file(path)
            self.assertEqual(provider.get_day(date(2026, 10, 10)).day_type, "weekend")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                status = main(["--timestamp", "2026-10-10T18:00:00+03:00", "--calendar-file", str(path)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(out.getvalue())["day_type"], "weekend")
            self.assertEqual(err.getvalue(), "")
            for data in ({"2026-02-30": {"day_type": "workday"}}, {"2026-10-10": {"day_type": "other"}}, []):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValidationError):
                    MappingCalendarProvider.from_file(path)

    def test_cli_missing_source_or_naive_time(self):
        for args in (["--timestamp", "2026-10-10T18:00:00+03:00"],
                     ["--timestamp", "2026-10-10T18:00:00", "--calendar-file", "unused.json"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)

    def test_cli_unknown_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.json"
            path.write_text("{}")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(["--timestamp", "2026-10-10T18:00:00+03:00", "--calendar-file", str(path)]), 1)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("Нет данных", err.getvalue())


if __name__ == "__main__":
    unittest.main()
