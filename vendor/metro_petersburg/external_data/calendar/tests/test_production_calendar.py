"""Audit the shipped five-day federal calendar, independently of API services."""

import json
import unittest
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from external_data.calendar import CalendarDataUnavailableError, MappingCalendarProvider, get_calendar_features


class ProductionCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data/ru_production_calendar_2026.json"
        cls.pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=list)
        cls.days = json.loads(path.read_text(encoding="utf-8"))
        cls.provider = MappingCalendarProvider.from_file(path)

    def test_complete_year_without_duplicate_dates_or_unknown_flags(self):
        expected = {(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)}
        self.assertEqual(set(self.days), expected)
        self.assertEqual(len(self.pairs), 365)
        for day, record in self.days.items():
            with self.subTest(day=day):
                self.assertEqual(set(record), {"day_type", "is_preholiday"})
                self.assertIs(type(record["is_preholiday"]), bool)
                self.assertEqual(self.provider.get_day(date.fromisoformat(day)).model_dump(), record)

    def test_federal_holidays_including_weekend_holidays(self):
        expected = {f"2026-01-{day:02d}" for day in range(1, 9)} | {
            "2026-02-23", "2026-03-08", "2026-05-01", "2026-05-09", "2026-06-12", "2026-11-04"}
        self.assertEqual({day for day, record in self.days.items() if record["day_type"] == "holiday"}, expected)

    def test_transferred_days_are_weekends_not_new_statutory_holidays(self):
        for day in ("2026-01-09", "2026-03-09", "2026-05-11", "2026-12-31"):
            with self.subTest(day=day):
                self.assertEqual(self.days[day], {"day_type": "weekend", "is_preholiday": False})

    def test_exactly_four_shortened_workdays(self):
        shortened = {day for day, record in self.days.items() if record["is_preholiday"]}
        self.assertEqual(shortened, {"2026-04-30", "2026-05-08", "2026-06-11", "2026-11-03"})
        self.assertTrue(all(self.days[day]["day_type"] == "workday" for day in shortened))
        self.assertEqual(self.days["2026-12-30"], {"day_type": "workday", "is_preholiday": False})

    def test_monthly_and_annual_workday_totals(self):
        expected = [15, 19, 21, 22, 19, 21, 23, 21, 22, 22, 20, 22]
        actual = [sum(day.startswith(f"2026-{month:02d}-") and record["day_type"] == "workday"
                      for day, record in self.days.items()) for month in range(1, 13)]
        self.assertEqual(actual, expected)
        self.assertEqual(Counter(record["day_type"] for record in self.days.values()),
                         {"workday": 247, "weekend": 104, "holiday": 14})
        self.assertFalse(any(date.fromisoformat(day).weekday() >= 5 and record["day_type"] == "workday"
                             for day, record in self.days.items()))

    def test_moscow_date_at_year_boundary_and_out_of_coverage(self):
        result = get_calendar_features(datetime.fromisoformat("2025-12-31T22:00:00Z"), provider=self.provider)
        self.assertEqual(result.timestamp.isoformat(), "2026-01-01T01:00:00+03:00")
        self.assertTrue(result.is_holiday)
        for value in (date(2025, 12, 31), date(2027, 1, 1)):
            with self.subTest(day=value), self.assertRaises(CalendarDataUnavailableError):
                self.provider.get_day(value)


if __name__ == "__main__":
    unittest.main()
