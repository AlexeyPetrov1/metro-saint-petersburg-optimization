import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from external_data import (
    Event, ExternalDataClient, ExternalDataConfigurationError, ExternalFeatures,
    configure_external_data, external_features_to_ml_dict, get_external_feature_rows,
    get_external_features,
)
from external_data.calendar import CalendarDataUnavailableError, CalendarDay, FakeCalendarProvider
from external_data.railway import FakeRailwayProvider, RailwayArrival, YandexRaspError
from external_data.weather import WeatherObservation, YandexWeatherError, YandexWeatherProvider


class FacadeTests(unittest.TestCase):
    def setUp(self):
        self.no_network = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        config = patch("external_data.api._default_client", None)
        config.start()
        self.addCleanup(config.stop)
        self.timestamp = datetime.fromisoformat("2026-10-12T18:00:00+03:00")
        self.station = "Площадь Восстания"
        self.point = (59.9343, 30.3351)
        self.calendar = FakeCalendarProvider({self.timestamp.date(): CalendarDay(day_type="workday", is_preholiday=None)})
        self.observation = WeatherObservation(
            timestamp=self.timestamp, latitude=self.point[0], longitude=self.point[1],
            temperature=6.0, feels_like=None, precipitation=1.2, precipitation_type="RAIN",
            wind_speed=4.3, wind_gust=None, humidity=85, pressure=750, condition="RAIN",
        )
        self.weather = Mock(spec=YandexWeatherProvider)
        self.weather.get_current_weather.return_value = self.observation.model_copy(update={
            "timestamp": self.timestamp - timedelta(minutes=30), "temperature": 99.0})
        self.weather.get_hourly_forecast.return_value = [self.observation]
        self.arrivals = [RailwayArrival(
            station_code="s9602494", arrival_time=self.timestamp + timedelta(minutes=minutes),
            transport_type="suburban" if minutes == 60 else "train", train_number=str(index),
        ) for index, minutes in enumerate((18, 45, 60, 75, 90, 105, 120))]
        self.railway = FakeRailwayProvider({("s9602494", self.timestamp.date()): self.arrivals})
        self.client = ExternalDataClient(
            calendar_provider=self.calendar, weather_coordinates=self.point,
            weather_provider=self.weather, railway_provider=self.railway,
        )

    def get(self, *, station=None, timestamp=None, events=None):
        return self.client.get_external_features(station or self.station, timestamp or self.timestamp, events)

    def test_hub_weather_calendar_and_railway_preserved(self):
        result = self.get()
        self.assertEqual(result.weather.model_dump(), self.observation.model_dump(exclude={"timestamp", "latitude", "longitude"}))
        self.assertEqual(result.calendar.model_dump(), dict(day_of_week=0, hour=18, minute=0,
            is_weekend=False, is_workday=True, is_holiday=False, is_preholiday=None, day_type="workday"))
        self.assertEqual(result.railway.model_dump(), dict(railway_name="Московский вокзал",
            arrivals_next_15m=0, arrivals_next_30m=1, arrivals_next_60m=3, arrivals_next_120m=7,
            train_arrivals_next_30m=1, suburban_arrivals_next_30m=0, minutes_to_next_arrival=18.0))
        self.assertEqual(self.railway.calls, [("s9602494", self.timestamp.date())])
        self.weather.get_current_weather.assert_called_once_with(*self.point)

    def test_no_hub_does_not_create_railway_provider(self):
        client = ExternalDataClient(calendar_provider=self.calendar, weather_coordinates=self.point,
                                    weather_provider=self.weather)
        with patch("external_data.api.YandexRaspProvider") as factory:
            result = client.get_external_features("Автово", self.timestamp)
        self.assertIsNone(result.railway)
        factory.assert_not_called()

    def test_events_none_and_empty(self):
        for events in (None, []):
            with self.subTest(events=events):
                result = self.get(events=events)
                self.assertEqual(result.events.model_dump(), dict(event_count=0, event_types=[], total_expected_people=None, has_event=False))

    def test_multiple_events_are_forwarded_without_filtering(self):
        events = [Event(event_name="Концерт", event_type="concert", expected_people=1000),
                  Event(event_name="Фестиваль", event_type="festival")]
        result = self.get(events=events)
        self.assertEqual(result.events.event_count, 2)
        self.assertEqual(result.events.event_types, ["concert", "festival"])
        self.assertEqual(result.events.total_expected_people, 1000)
        self.assertIsNone(events[1].expected_people)

    def test_datetime_and_iso_string(self):
        self.assertEqual(self.get(), self.get(timestamp=self.timestamp.isoformat()))

    def test_utc_and_canonical_station(self):
        result = self.get(station=" площадь восстания ", timestamp="2026-10-12T15:00:00Z")
        self.assertEqual(result.station, self.station)
        self.assertEqual(result.timestamp.isoformat(), "2026-10-12T18:00:00+03:00")

    def test_naive_and_invalid_timestamp_rejected_before_api(self):
        for value in (datetime(2026, 10, 12, 18), "2026-10-12T18:00:00", "2026-10-12", 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.get(timestamp=value)
        self.weather.get_current_weather.assert_not_called()

    def test_unknown_station_and_bad_events_fail_before_api(self):
        with self.assertRaises(ValueError):
            self.get(station="Несуществующая")
        with self.assertRaises(TypeError):
            self.get(events=[{"event_name": "Концерт"}])
        self.weather.get_current_weather.assert_not_called()

    def test_flat_schema_values_and_no_nested_objects(self):
        features = self.get(events=[Event(event_name="Концерт", event_type="concert", expected_people=1000)])
        row = external_features_to_ml_dict(features)
        self.assertEqual(row["timestamp"], self.timestamp.isoformat())
        self.assertEqual(row["station"], self.station)
        self.assertEqual(row["temperature"], 6.0)
        self.assertEqual(row["day_type"], "workday")
        self.assertEqual(row["event_types"], ["concert"])
        self.assertEqual(row["railway_arrivals_next_120m"], 7)
        self.assertEqual(row["train_arrivals_next_30m"], 1)
        self.assertEqual(row["minutes_to_next_arrival"], 18.0)
        self.assertEqual(len(row), 31)
        self.assertFalse(any(isinstance(value, dict) for value in row.values()))
        for field in ("weather", "calendar", "events", "railway", "passenger_flow", "lags", "baseline", "prediction"):
            self.assertNotIn(field, row)
        self.assertEqual(json.loads(json.dumps(row)), row)
        row["event_types"].append("changed")
        self.assertEqual(features.events.event_types, ["concert"])

    def test_flat_absent_railway_is_null_not_zero(self):
        row = external_features_to_ml_dict(self.get(station="Автово"))
        fields = ["railway_name", "railway_arrivals_next_15m", "railway_arrivals_next_30m",
                  "railway_arrivals_next_60m", "railway_arrivals_next_120m", "train_arrivals_next_30m",
                  "suburban_arrivals_next_30m", "minutes_to_next_arrival"]
        self.assertTrue(all(row[field] is None for field in fields))
        self.assertEqual(row["event_count"], 0)

    def test_flat_rejects_raw_dict(self):
        with self.assertRaises(TypeError):
            external_features_to_ml_dict({})

    def test_missing_configuration_is_explicit(self):
        with self.assertRaisesRegex(ExternalDataConfigurationError, "configure_external_data"):
            get_external_features(self.station, self.timestamp)

    def test_configure_real_local_table_once_and_public_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.json"
            path.write_text(json.dumps({"2026-10-12": {"day_type": "workday", "is_preholiday": None}}))
            configure_external_data(calendar_file=path, weather_coordinates=self.point)
            path.unlink()  # Source loaded once, not read on each call.
            with patch("external_data.api.YandexWeatherProvider", return_value=self.weather), patch(
                "external_data.api.YandexRaspProvider", return_value=Mock(wraps=self.railway, close=Mock())
            ):
                self.assertIsInstance(get_external_features(self.station, self.timestamp), ExternalFeatures)
                rows = get_external_feature_rows([{"station": "Автово", "timestamp": self.timestamp}])
                self.assertIsNone(rows[0]["railway_name"])

    def test_default_calendar_and_public_ml_entry_points(self):
        configure_external_data(weather_coordinates=self.point)
        with patch("external_data.api.YandexWeatherProvider", return_value=self.weather), patch(
            "external_data.api.YandexRaspProvider", return_value=Mock(wraps=self.railway, close=Mock())
        ):
            row = external_features_to_ml_dict(get_external_features(self.station, self.timestamp))
        self.assertEqual(row["day_type"], "workday")
        self.assertIs(row["is_preholiday"], False)
        self.assertEqual(len(row), 31)

    def test_invalid_calendar_file_does_not_fall_back_or_replace_configuration(self):
        client = configure_external_data(weather_coordinates=self.point)
        with self.assertRaises(ExternalDataConfigurationError):
            configure_external_data(weather_coordinates=self.point, calendar_file="not-present.json")
        from external_data import api
        self.assertIs(api._default_client, client)

    def test_provider_takes_priority_over_file_and_default(self):
        configure_external_data(weather_coordinates=self.point, calendar_file="not-present.json", calendar_provider=self.calendar)
        with patch("external_data.api.YandexWeatherProvider", return_value=self.weather):
            result = get_external_features("Автово", self.timestamp)
        # The explicit provider has unknown preholiday; the built-in has False.
        self.assertIsNone(result.calendar.is_preholiday)

    def test_calendar_file_overrides_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.json"
            path.write_text(json.dumps({"2026-10-12": {"day_type": "holiday", "is_preholiday": False}}))
            configure_external_data(weather_coordinates=self.point, calendar_file=path)
        with patch("external_data.api.YandexWeatherProvider", return_value=self.weather):
            self.assertEqual(get_external_features("Автово", self.timestamp).calendar.day_type, "holiday")

    def test_default_calendar_outside_2026_fails_before_network(self):
        configure_external_data(weather_coordinates=self.point)
        with patch("external_data.api.YandexWeatherProvider") as weather:
            for timestamp in ("2025-12-31T12:00:00+03:00", "2027-01-01T12:00:00+03:00"):
                with self.subTest(timestamp=timestamp), self.assertRaises(CalendarDataUnavailableError):
                    get_external_features(self.station, timestamp)
            weather.assert_not_called()

    def test_default_calendar_path_is_independent_of_working_directory(self):
        from contextlib import chdir
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            configure_external_data(weather_coordinates=self.point)
        with patch("external_data.api.YandexWeatherProvider", return_value=self.weather):
            self.assertEqual(get_external_features("Автово", self.timestamp).calendar.day_type, "workday")

    def test_calendar_provider_configurable_without_file(self):
        client = configure_external_data(calendar_provider=self.calendar, weather_coordinates=self.point)
        self.assertIsInstance(client, ExternalDataClient)

    def test_uncovered_calendar_date_fails_before_network(self):
        with self.assertRaises(CalendarDataUnavailableError):
            self.get(timestamp="2026-10-13T18:00:00+03:00")
        self.weather.get_current_weather.assert_not_called()
        self.assertEqual(self.railway.calls, [])

    def test_per_station_weather_coordinates(self):
        client = ExternalDataClient(calendar_provider=self.calendar,
            weather_coordinates={self.station: self.point}, weather_provider=self.weather, railway_provider=self.railway)
        client.get_external_features(self.station, self.timestamp)
        self.weather.get_current_weather.assert_called_once_with(*self.point)
        with self.assertRaisesRegex(ExternalDataConfigurationError, "weather_coordinates"):
            client.get_external_features("Автово", self.timestamp)

    def test_invalid_coordinates_rejected(self):
        for point in (None, (), (91, 30), {}, (59, float("nan"))):
            with self.subTest(point=point), self.assertRaises(ValueError):
                ExternalDataClient(calendar_provider=self.calendar, weather_coordinates=point)

    def test_exact_forecast_selected_not_current(self):
        self.weather.get_current_weather.return_value = self.observation.model_copy(update={
            "timestamp": self.timestamp - timedelta(minutes=30), "temperature": 99.0})
        self.weather.get_hourly_forecast.return_value = [self.observation]
        result = self.get()
        self.assertEqual(result.weather.temperature, 6.0)
        self.weather.get_hourly_forecast.assert_called_once_with(*self.point, hours=1)

    def test_missing_hour_never_uses_previous_future_or_non_hour_point(self):
        self.weather.get_hourly_forecast.return_value = [self.observation.model_copy(update={
            "timestamp": self.timestamp + timedelta(minutes=minutes)}) for minutes in (-60, 30, 60)]
        result = self.get(timestamp=self.timestamp + timedelta(minutes=45))
        self.assertTrue(all(value is None for value in result.weather.model_dump().values()))

    def test_hourly_forecast_on_five_quarter_hour_requests(self):
        self.weather.get_current_weather.return_value = self.observation.model_copy(update={
            "timestamp": self.timestamp - timedelta(minutes=30), "temperature": 99.0})
        self.weather.get_hourly_forecast.return_value = [self.observation, self.observation.model_copy(update={
            "timestamp": self.timestamp + timedelta(hours=1), "temperature": 7.0})]
        for minutes, temperature in ((0, 6.0), (15, 6.0), (30, 6.0), (45, 6.0), (60, 7.0)):
            with self.subTest(minutes=minutes):
                timestamp = self.timestamp + timedelta(minutes=minutes)
                result = self.get(station="Автово", timestamp=timestamp)
                self.assertEqual(result.weather.temperature, temperature)
                expected = self.observation.model_dump(exclude={"timestamp", "latitude", "longitude"})
                self.assertEqual(result.weather.model_dump(), {**expected, "temperature": temperature})
                self.assertEqual(result.timestamp, timestamp)
                self.assertEqual(result.calendar.minute, timestamp.minute)
                row = external_features_to_ml_dict(result)
                self.assertEqual(row["timestamp"], timestamp.isoformat())
                self.assertEqual(len(row), 31)
        self.assertEqual(self.observation.timestamp, self.timestamp)

    def test_current_weather_at_exact_target_is_never_used(self):
        self.weather.get_current_weather.return_value = self.observation.model_copy(update={"temperature": 99.0})
        self.assertEqual(self.get().weather.temperature, 6.0)
        self.weather.get_hourly_forecast.assert_called_once_with(*self.point, hours=1)
        self.weather.get_hourly_forecast.return_value = []
        self.assertTrue(all(value is None for value in self.get().weather.model_dump().values()))

    def test_current_hour_missing_from_forecast_stays_null(self):
        self.weather.get_current_weather.return_value = self.observation.model_copy(update={
            "timestamp": self.timestamp + timedelta(minutes=5)})
        self.weather.get_hourly_forecast.return_value = [self.observation.model_copy(update={
            "timestamp": self.timestamp + timedelta(hours=1)})]
        result = self.get(timestamp=self.timestamp + timedelta(minutes=15))
        self.assertTrue(all(value is None for value in result.weather.model_dump().values()))

    def test_next_hour_boundary_does_not_extend_previous_forecast(self):
        just_before = self.timestamp + timedelta(hours=1, microseconds=-1)
        self.assertEqual(self.get(timestamp=just_before).weather.temperature, 6.0)
        result = self.get(timestamp=self.timestamp + timedelta(hours=1))
        self.assertTrue(all(value is None for value in result.weather.model_dump().values()))

    def test_floor_uses_moscow_time_and_accepts_forecast_utc_offset(self):
        self.weather.get_hourly_forecast.return_value = [self.observation.model_copy(update={
            "timestamp": datetime.fromisoformat("2026-10-12T15:00:00Z")})]
        result = self.get(timestamp="2026-10-12T20:15:12.345678+05:30")
        self.assertEqual(result.timestamp.isoformat(), "2026-10-12T17:45:12.345678+03:00")
        # The 18:00 forecast is in the future relative to 17:45.
        self.assertIsNone(result.weather.temperature)
        result = self.get(timestamp="2026-10-12T20:45:12.345678+05:30")
        self.assertEqual(result.timestamp.isoformat(), "2026-10-12T18:15:12.345678+03:00")
        self.assertEqual(result.weather.temperature, 6.0)

    def test_forecast_failure_is_not_replaced_with_current_weather(self):
        self.weather.get_hourly_forecast.side_effect = YandexWeatherError("HTTP 503")
        with self.assertRaisesRegex(YandexWeatherError, "503"):
            self.get()

    def test_past_or_beyond_forecast_returns_null_without_forecast_request(self):
        for current_offset in (1, -49):
            with self.subTest(current_offset=current_offset):
                self.weather.get_current_weather.return_value = self.observation.model_copy(update={"timestamp": self.timestamp + timedelta(hours=current_offset)})
                self.assertIsNone(self.get().weather.temperature)
        self.weather.get_hourly_forecast.assert_not_called()

    def test_missing_weather_env_error_names_variable(self):
        client = ExternalDataClient(calendar_provider=self.calendar, weather_coordinates=self.point)
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(YandexWeatherError, "YANDEX_WEATHER_API_KEY"):
            client.get_external_features("Автово", self.timestamp)

    def test_missing_rasp_env_error_names_variable(self):
        client = ExternalDataClient(calendar_provider=self.calendar, weather_coordinates=self.point)
        with patch.dict("os.environ", {}, clear=True), patch("external_data.api.YandexWeatherProvider", return_value=self.weather):
            with self.assertRaisesRegex(YandexRaspError, "YANDEX_RASP_API_KEY"):
                client.get_external_features(self.station, self.timestamp)
        self.weather.close.assert_called_once()
        self.weather.get_current_weather.assert_not_called()

    def test_injected_providers_not_closed_and_api_errors_not_hidden(self):
        self.weather.get_current_weather.side_effect = YandexWeatherError("HTTP 403")
        with self.assertRaisesRegex(YandexWeatherError, "403"):
            self.get()
        self.weather.close.assert_not_called()

    def test_owned_providers_closed_on_failure(self):
        client = ExternalDataClient(calendar_provider=self.calendar, weather_coordinates=self.point)
        self.weather.get_current_weather.side_effect = YandexWeatherError("HTTP 403")
        railway = Mock(wraps=self.railway, close=Mock())
        with patch("external_data.api.YandexWeatherProvider", return_value=self.weather), patch(
            "external_data.api.YandexRaspProvider", return_value=railway
        ), self.assertRaises(YandexWeatherError):
            client.get_external_features(self.station, self.timestamp)
        self.weather.close.assert_called_once()
        railway.close.assert_called_once()

    def test_batch_sequential_rows_with_individual_events(self):
        rows = self.client.get_external_feature_rows([
            {"station": "Автово", "timestamp": self.timestamp},
            {"station": self.station, "timestamp": self.timestamp.isoformat(), "events": [Event(event_name="Матч")]},
        ])
        self.assertEqual([row["event_count"] for row in rows], [0, 1])
        self.assertEqual([row["station"] for row in rows], ["Автово", self.station])
        self.assertIsNone(rows[0]["railway_name"])
        self.assertEqual(rows[1]["railway_name"], "Московский вокзал")
        self.assertEqual(self.client.get_external_feature_rows([]), [])


if __name__ == "__main__":
    unittest.main()
