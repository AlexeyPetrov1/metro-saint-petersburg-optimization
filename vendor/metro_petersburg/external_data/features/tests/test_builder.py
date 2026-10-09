import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from unittest.mock import patch

from pydantic import ValidationError

from external_data.calendar.models import CalendarFeatures
from external_data.events.models import Event
from external_data.weather.models import WeatherObservation
from external_data.railway.models import RailwayFeatures
from external_data.features import ExternalFeatures, build_external_features
from external_data.features.cli import main


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.timestamp = datetime.fromisoformat("2026-10-10T18:00:00+03:00")
        self.calendar = CalendarFeatures(
            timestamp=self.timestamp, day_of_week=5, hour=18, minute=0,
            is_weekend=False, is_workday=True, is_holiday=False,
            is_preholiday=True, day_type="workday",  # Synthetic working Saturday.
        )
        self.weather = WeatherObservation(
            timestamp=self.timestamp, latitude=59.9343, longitude=30.3351,
            temperature=6.0, feels_like=3.0, precipitation=1.2,
            precipitation_type="RAIN", wind_speed=4.3, wind_gust=8.2,
            humidity=85.0, pressure=750.0, condition="RAIN",
        )
        self.railway = RailwayFeatures(
            timestamp=self.timestamp, metro_station="Площадь Восстания", railway_name="Московский вокзал",
            arrivals_next_15m=0, arrivals_next_30m=1, arrivals_next_60m=3,
            arrivals_next_120m=7, train_arrivals_next_30m=1, suburban_arrivals_next_30m=0,
            minutes_to_next_arrival=18,
        )

    def build(self, events, **kwargs):
        args = dict(timestamp=self.timestamp, weather=self.weather, calendar=self.calendar, events=events)
        args.update(kwargs)
        return build_external_features(**args)

    def test_weather_calendar_no_events(self):
        result = self.build([])
        self.assertEqual(result.weather.model_dump(), self.weather.model_dump(exclude={"timestamp", "latitude", "longitude"}))
        self.assertEqual(result.calendar.model_dump(), self.calendar.model_dump(exclude={"timestamp"}))
        self.assertEqual(result.events.model_dump(), {
            "event_count": 0, "event_types": [], "total_expected_people": None, "has_event": False,
        })
        self.assertEqual(result.timestamp, self.timestamp)

    def test_no_weather_all_fields_null(self):
        result = self.build([], weather=None)
        self.assertEqual(set(result.weather.model_dump()), {
            "temperature", "feels_like", "precipitation", "precipitation_type", "wind_speed",
            "wind_gust", "humidity", "pressure", "condition",
        })
        self.assertTrue(all(value is None for value in result.weather.model_dump().values()))

    def test_one_event(self):
        result = self.build([Event(event_name="Концерт", event_type="concert", expected_people=150)])
        self.assertEqual(result.events.model_dump(), {
            "event_count": 1, "event_types": ["concert"], "total_expected_people": 150, "has_event": True,
        })

    def test_multiple_events_and_stable_unique_types(self):
        events = [Event(event_name=str(i), event_type=kind, expected_people=people) for i, (kind, people) in enumerate([
            ("festival", 200), ("concert", None), ("festival", 100), (None, None), ("sport_event", 0),
        ])]
        result = self.build(events)
        self.assertEqual(result.events.event_count, 5)
        self.assertEqual(result.events.event_types, ["festival", "concert", "sport_event"])
        self.assertEqual(result.events.total_expected_people, 300)
        self.assertTrue(result.events.has_event)

    def test_all_people_unknown(self):
        result = self.build([Event(event_name="A"), Event(event_name="B")])
        self.assertIsNone(result.events.total_expected_people)
        self.assertEqual(result.events.event_types, [])
        self.assertTrue(result.events.has_event)

    def test_known_zero_is_not_null(self):
        result = self.build([Event(event_name="A", expected_people=0), Event(event_name="B")])
        self.assertEqual(result.events.total_expected_people, 0)

    def test_station_none_and_supplied(self):
        self.assertIsNone(self.build([]).station)
        self.assertEqual(self.build([], station="Площадь Восстания").station, "Площадь Восстания")

    def test_naive_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.build([], timestamp=datetime(2026, 10, 10, 18))

    def test_utc_timestamp_normalized(self):
        result = self.build([], timestamp=datetime.fromisoformat("2026-10-10T15:00:00Z"))
        self.assertEqual(result.timestamp.isoformat(), "2026-10-10T18:00:00+03:00")
        self.assertEqual(result.calendar.hour, 18)

    def test_calendar_mismatch_is_not_recalculated(self):
        with self.assertRaisesRegex(ValueError, "calendar.timestamp"):
            self.build([], timestamp=datetime.fromisoformat("2026-10-10T19:00:00+03:00"))

    def test_calendar_required(self):
        with self.assertRaises(TypeError):
            self.build([], calendar=None)

    def test_input_models_not_raw_dicts(self):
        with self.assertRaises(TypeError):
            self.build([], weather=self.weather.model_dump())
        with self.assertRaises(TypeError):
            self.build([{"event_name": "A"}])

    def test_inputs_not_mutated(self):
        events = [Event(event_name="A", event_type="concert", expected_people=100)]
        before = [self.weather.model_dump(), self.calendar.model_dump(), events[0].model_dump()]
        result = self.build(events)
        result.events.event_types.append("festival")
        result.weather.temperature = 99
        self.assertEqual(before, [self.weather.model_dump(), self.calendar.model_dump(), events[0].model_dump()])

    def test_events_are_not_filtered_or_deduplicated(self):
        event = Event(event_name="Past event", start_time="2020-01-01T12:00:00+03:00", expected_people=10)
        result = self.build([event, event])
        self.assertEqual(result.events.event_count, 2)
        self.assertEqual(result.events.total_expected_people, 20)

    def test_json_contract_roundtrip(self):
        result = self.build([])
        self.assertEqual(ExternalFeatures.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(set(result.model_dump()), {"timestamp", "station", "weather", "calendar", "events", "railway"})
        self.assertNotIn("timestamp", result.calendar.model_dump())
        self.assertNotIn("latitude", result.weather.model_dump())
        with self.assertRaises(ValidationError):
            ExternalFeatures.model_validate({**result.model_dump(), "timestamp": "2026-10-10T18:00:00"})

    def test_demo_cli_is_local(self):
        output = io.StringIO()
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), redirect_stdout(output):
            self.assertEqual(main(), 0)
        result = ExternalFeatures.model_validate_json(output.getvalue())
        self.assertEqual(result.events.event_count, 2)
        self.assertEqual(result.events.total_expected_people, 1000)
        self.assertEqual(result.station, "Площадь Восстания")
        self.assertEqual(result.railway.railway_name, "Московский вокзал")
        self.assertEqual(result.railway.arrivals_next_120m, 7)

    def test_railway_present_counts_preserved(self):
        result = self.build([], station="Площадь Восстания", railway=self.railway)
        self.assertEqual(result.railway.model_dump(), self.railway.model_dump(exclude={"timestamp", "metro_station"}))
        self.assertEqual(ExternalFeatures.model_validate_json(result.model_dump_json()), result)
        result.railway.arrivals_next_30m = 99
        self.assertEqual(self.railway.arrivals_next_30m, 1)

    def test_railway_none_and_old_positional_call(self):
        result = build_external_features(self.timestamp, self.weather, self.calendar, [], "Автово")
        self.assertIsNone(result.railway)
        self.assertIsNone(self.build([], railway=None).railway)
        self.assertIsNone(self.build([]).station)
        # Old serialized inputs remain readable; new output adds railway=null.
        old_payload = result.model_dump(exclude={"railway"})
        self.assertIsNone(ExternalFeatures.model_validate(old_payload).railway)

    def test_railway_station_mismatch_or_absent_rejected(self):
        for station in ("Автово", None):
            with self.subTest(station=station), self.assertRaisesRegex(ValueError, "railway.metro_station"):
                self.build([], station=station, railway=self.railway)

    def test_railway_timestamp_mismatch_rejected(self):
        railway = self.railway.model_copy(update={"timestamp": datetime.fromisoformat("2026-10-10T19:00:00+03:00")})
        with self.assertRaisesRegex(ValueError, "railway.timestamp"):
            self.build([], station="Площадь Восстания", railway=railway)

    def test_railway_minutes_null_and_zero_counts_valid(self):
        railway = RailwayFeatures(**{**self.railway.model_dump(), "minutes_to_next_arrival": None,
            "arrivals_next_15m": 0, "arrivals_next_30m": 0, "arrivals_next_60m": 0,
            "arrivals_next_120m": 0, "train_arrivals_next_30m": 0, "suburban_arrivals_next_30m": 0})
        result = self.build([], station="Площадь Восстания", railway=railway)
        self.assertIsNone(result.railway.minutes_to_next_arrival)
        self.assertEqual(result.railway.arrivals_next_120m, 0)

    def test_railway_same_instant_utc_input(self):
        result = self.build([], station="Площадь Восстания", railway=self.railway,
                            timestamp=datetime.fromisoformat("2026-10-10T15:00:00Z"))
        self.assertEqual(result.timestamp.isoformat(), "2026-10-10T18:00:00+03:00")

    def test_railway_requires_ready_model(self):
        with self.assertRaisesRegex(TypeError, "RailwayFeatures"):
            self.build([], station="Площадь Восстания", railway=self.railway.model_dump())

    def test_railway_does_not_change_other_blocks(self):
        events = [Event(event_name="Концерт", event_type="concert", expected_people=4000)]
        old = self.build(events, station="Площадь Восстания")
        new = self.build(events, station="Площадь Восстания", railway=self.railway)
        self.assertEqual(old.model_dump(exclude={"railway"}), new.model_dump(exclude={"railway"}))


if __name__ == "__main__":
    unittest.main()
