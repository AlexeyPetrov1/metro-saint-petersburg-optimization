import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from external_data.weather import WeatherObservation, YandexWeatherError, YandexWeatherProvider
from external_data.weather.cli import main
from external_data.weather.yandex_weather import ENDPOINT

REAL_CLIENT = httpx.Client
KEY = "test-weather-key-not-a-secret"
NOW = {
    "time": "2026-10-07T12:30:00+03:00", "temperature": 6, "feelsLike": 3,
    "precType": "RAIN", "windSpeed": 4.3, "windGust": 8.2,
    "humidity": 85, "pressure": 750, "condition": "RAIN",
}


def current(row=None):
    now = deepcopy(NOW if row is None else row)
    server = now.pop("time", None)
    return {"data": {"serverTime": server, "weatherByPoint": {"now": now}}}


def forecast(rows, server="2026-10-07T12:30:00+03:00"):
    return {"data": {"serverTime": server, "weatherByPoint": {"forecast": {"days": [{"hours": rows}]}}}}


def hour(time, **values):
    return {**NOW, "time": time, "accumulatedPrec": 1.2, **values}


class WeatherTests(unittest.TestCase):
    def provider(self, payload=None, *, status=200, exception=None, include_feels_like=False, invalid_json=False):
        self.requests = []

        def handle(request):
            self.requests.append(request)
            if exception:
                raise exception
            if invalid_json:
                return httpx.Response(status, text="not-json " + KEY)
            return httpx.Response(status, json=current() if payload is None else payload)

        def factory(**kwargs):
            client = REAL_CLIENT(**kwargs, transport=httpx.MockTransport(handle))
            self.addCleanup(client.close)
            return client

        with patch.dict("os.environ", {"YANDEX_WEATHER_API_KEY": KEY}, clear=True), patch(
            "external_data.weather.yandex_weather.httpx.Client", side_effect=factory,
        ):
            return YandexWeatherProvider(include_feels_like=include_feels_like)

    def test_current_header_coordinates_units_and_timeout(self):
        provider = self.provider(include_feels_like=True)
        result = provider.get_current_weather(59.9343, 30.3351)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), ENDPOINT)
        self.assertEqual(request.headers["X-Yandex-Weather-Key"], KEY)
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(request.extensions["timeout"]["read"], 15.0)
        body = json.loads(request.content)
        self.assertEqual(body["variables"], {"point": {"lat": 59.9343, "lon": 30.3351}})
        self.assertIn("temperature(unit: CELSIUS)", body["query"])
        self.assertIn("pressure(unit: MM_HG)", body["query"])
        self.assertIn("windSpeed(unit: METERS_PER_SECOND)", body["query"])
        self.assertIn("feelsLike(unit: CELSIUS)", body["query"])
        self.assertNotIn("forecast", body["query"])
        self.assertNotIn("accumulatedPrec", body["query"])
        self.assertNotIn(KEY, request.content.decode())
        self.assertEqual(result.timestamp.isoformat(), NOW["time"])
        self.assertEqual(result.latitude, 59.9343)
        self.assertEqual(result.longitude, 30.3351)
        for name, value in {"temperature": 6.0, "feels_like": 3.0, "wind_speed": 4.3,
                            "wind_gust": 8.2, "humidity": 85.0, "pressure": 750.0,
                            "condition": "RAIN", "precipitation_type": "RAIN"}.items():
            self.assertEqual(getattr(result, name), value)
        self.assertIsNone(result.precipitation)

    def test_basic_plan_query_omits_feels_like(self):
        provider = self.provider(current({"time": NOW["time"]}))
        result = provider.get_current_weather(-33.9, 151.2)
        self.assertNotIn("feelsLike", json.loads(self.requests[0].content)["query"])
        self.assertIsNone(result.feels_like)
        self.assertEqual(result.latitude, -33.9)

    def test_current_test_plan_field_allowlist(self):
        provider = self.provider(current({"time": NOW["time"]}))
        provider.get_current_weather(60, 30)
        query = json.loads(self.requests[0].content)["query"]
        # Check the actual HTTP query, not just a shared constant.
        selected = query.split("now {", 1)[1].split("}", 1)[0]
        names = {line.strip().split("(", 1)[0] for line in selected.splitlines() if line.strip()}
        self.assertEqual(names, {"temperature", "condition", "humidity", "pressure", "precType", "windSpeed", "windGust"})
        self.assertIn("serverTime", query)
        self.assertNotIn("time", names)
        self.assertNotIn("timestamp", names)

    def test_feels_like_flag_for_both_queries(self):
        for include in (False, True):
            for hourly in (False, True):
                with self.subTest(include=include, hourly=hourly):
                    provider = self.provider(forecast([]) if hourly else current(), include_feels_like=include)
                    if hourly:
                        provider.get_hourly_forecast(60, 30)
                    else:
                        provider.get_current_weather(60, 30)
                    query = json.loads(self.requests[0].content)["query"]
                    self.assertEqual("feelsLike" in query, include)
                    if hourly:
                        self.assertIn(" time accumulatedPrec", query)
                        selected = query.split("hours {", 1)[1].split("}", 1)[0]
                        names = set(re.sub(r"\([^)]*\)", "", selected).split())
                        expected = {"time", "temperature", "condition", "humidity", "pressure",
                                    "precType", "windSpeed", "windGust", "accumulatedPrec"}
                        self.assertEqual(names, expected | ({"feelsLike"} if include else set()))

    def test_current_timestamp_uses_server_time_only(self):
        payload = current()
        payload["data"]["weatherByPoint"]["now"]["time"] = "2000-01-01T00:00:00Z"
        result = self.provider(payload).get_current_weather(60, 30)
        self.assertEqual(result.timestamp.isoformat(), NOW["time"])
        del payload["data"]["serverTime"]
        with self.assertRaises(YandexWeatherError):
            self.provider(payload).get_current_weather(60, 30)

    def test_missing_measurements_are_null(self):
        result = self.provider(current({"time": NOW["time"]})).get_current_weather(0, 0)
        for key, value in result.model_dump().items():
            if key not in ("timestamp", "latitude", "longitude"):
                self.assertIsNone(value, key)

    def test_zero_values_are_preserved(self):
        row = {"time": NOW["time"], "temperature": 0, "windSpeed": 0, "humidity": 0, "precType": "NO_TYPE"}
        result = self.provider(current(row)).get_current_weather(0, 0)
        self.assertEqual(result.temperature, 0)
        self.assertEqual(result.wind_speed, 0)
        self.assertEqual(result.humidity, 0)
        self.assertEqual(result.precipitation_type, "NO_TYPE")

    def test_hourly_horizon_sorting_and_precipitation(self):
        rows = [hour("2026-10-07T14:00:00+03:00", accumulatedPrec=0),
                hour("2026-10-07T12:00:00+03:00"), hour("2026-10-07T13:00:00+03:00"),
                hour("2026-10-07T15:00:00+03:00")]
        provider = self.provider(forecast(rows))
        result = provider.get_hourly_forecast(59.9343, 30.3351, hours=2)
        self.assertEqual([item.timestamp.isoformat() for item in result],
                         ["2026-10-07T13:00:00+03:00", "2026-10-07T14:00:00+03:00"])
        self.assertEqual([item.precipitation for item in result], [1.2, 0])
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["variables"]["days"], 2)
        self.assertIn("serverTime", body["query"])
        self.assertIn("accumulatedPrec", body["query"])
        self.assertNotIn("summary", body["query"])

    def test_past_hours_keep_started_hours_of_today(self):
        rows = [hour(f"2026-10-07T{h:02d}:00:00+03:00", accumulatedPrec=float(h)) for h in range(8, 16)]
        result = self.provider(forecast(rows)).get_hourly_forecast(60, 30, 1, past_hours=4)
        self.assertEqual([item.timestamp.hour for item in result], [9, 10, 11, 12, 13])
        self.assertEqual([item.precipitation for item in result], [9.0, 10.0, 11.0, 12.0, 13.0])
        default = self.provider(forecast(rows)).get_hourly_forecast(60, 30, 1)
        self.assertEqual([item.timestamp.hour for item in default], [13])

    def test_midnight_and_timezone_boundary(self):
        payload = forecast([], server="2026-10-07T20:30:00Z")
        payload["data"]["weatherByPoint"]["forecast"]["days"] = [
            {"hours": [hour("2026-10-07T23:00:00+03:00")]},
            {"hours": [hour("2026-10-08T00:00:00+03:00"), hour("2026-10-08T01:00:00+03:00")]},
        ]
        result = self.provider(payload).get_hourly_forecast(60, 30, 2)
        self.assertEqual([row.timestamp.isoformat() for row in result],
                         ["2026-10-08T00:00:00+03:00", "2026-10-08T01:00:00+03:00"])

    def test_forecast_does_not_invent_missing_slots_or_precipitation(self):
        payload = forecast([{"time": "2026-10-07T13:00:00+03:00"}])
        result = self.provider(payload).get_hourly_forecast(60, 30)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].precipitation)
        self.assertEqual(self.provider(forecast([])).get_hourly_forecast(60, 30), [])

    def test_duplicate_forecast_timestamps_rejected(self):
        row = hour("2026-10-07T13:00:00+03:00")
        with self.assertRaisesRegex(YandexWeatherError, "дублирующиеся"):
            self.provider(forecast([row, row])).get_hourly_forecast(60, 30)

    def test_http_errors_no_retries_no_secrets(self):
        for status in (401, 403, 429, 500, 503):
            with self.subTest(status=status):
                provider = self.provider({"message": KEY}, status=status)
                with self.assertRaisesRegex(YandexWeatherError, f"HTTP {status}") as error:
                    provider.get_current_weather(60, 30)
                self.assertNotIn(KEY, str(error.exception))
                self.assertEqual(len(self.requests), 1)

    def test_timeout(self):
        provider = self.provider(exception=httpx.ReadTimeout(KEY))
        with self.assertRaisesRegex(YandexWeatherError, "timeout") as error:
            provider.get_current_weather(60, 30)
        self.assertNotIn(KEY, str(error.exception))
        self.assertEqual(len(self.requests), 1)

    def test_network_error(self):
        with self.assertRaisesRegex(YandexWeatherError, "ошибка сети"):
            self.provider(exception=httpx.ConnectError(KEY)).get_current_weather(60, 30)

    def test_missing_weather_key_does_not_use_yandexgpt_key(self):
        for value in (None, "", " "):
            env = {"YANDEX_API_KEY": "llm-key"}
            if value is not None:
                env["YANDEX_WEATHER_API_KEY"] = value
            with self.subTest(value=value), patch.dict("os.environ", env, clear=True), patch(
                "external_data.weather.yandex_weather.httpx.Client",
            ) as factory:
                with self.assertRaisesRegex(YandexWeatherError, "YANDEX_WEATHER_API_KEY"):
                    YandexWeatherProvider()
                factory.assert_not_called()

    def test_graphql_errors_including_partial_success(self):
        for data in (None, current()["data"]):
            with self.subTest(data=data):
                provider = self.provider({"data": data, "errors": [{"message": KEY}]})
                with self.assertRaisesRegex(YandexWeatherError, "GraphQL error") as error:
                    provider.get_current_weather(60, 30)
                self.assertNotIn(KEY, str(error.exception))

    def test_graphql_messages_preserved_for_http_200_and_400(self):
        for status in (200, 400):
            with self.subTest(status=status):
                messages = ['Cannot query field "time" on type "Now".', 'Access denied for field "feelsLike".']
                payload = {"errors": [{"message": message, "extensions": {"headers": {"key": KEY}}} for message in messages]}
                provider = self.provider(payload, status=status)
                with self.assertRaises(YandexWeatherError) as error:
                    provider.get_current_weather(60, 30)
                for message in messages:
                    self.assertIn(message, str(error.exception))
                self.assertNotIn(KEY, str(error.exception))
                self.assertNotIn("headers", str(error.exception))

    def test_cli_debug_query_variables_and_redacted_errors(self):
        message = 'Cannot query field "time"; echoed credential: ' + KEY
        provider = self.provider({"errors": [{"message": message, "extensions": {"headers": "never-output"}}]})
        out, err = io.StringIO(), io.StringIO()
        with patch("external_data.weather.cli.YandexWeatherProvider", return_value=provider), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(["--lat", "59.9343", "--lon", "30.3351", "--debug"]), 1)
        diagnostics = err.getvalue()
        self.assertEqual(out.getvalue(), "")
        for value in ("DEBUG GraphQL query", "DEBUG variables", "DEBUG GraphQL errors", "59.9343", "Cannot query field", "[REDACTED]"):
            self.assertIn(value, diagnostics)
        for value in (KEY, "X-Yandex-Weather-Key", "never-output", "extensions", "feelsLike"):
            self.assertNotIn(value, diagnostics)

    def test_debug_off_does_not_log_query(self):
        provider = self.provider({"errors": [{"message": "Access denied for field"}]})
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaisesRegex(YandexWeatherError, "Access denied for field"):
            provider.get_current_weather(60, 30)
        self.assertEqual(err.getvalue(), "")

    def test_malformed_json_and_structure(self):
        with self.assertRaisesRegex(YandexWeatherError, "не является JSON"):
            self.provider(invalid_json=True).get_current_weather(60, 30)
        for payload in ([], {}, {"data": None}, {"data": {}}, {"data": {"weatherByPoint": {"now": None}}}):
            with self.subTest(payload=payload), self.assertRaises(YandexWeatherError):
                self.provider(payload).get_current_weather(60, 30)

    def test_invalid_weather_values_and_missing_timestamp(self):
        for update in ({"time": None}, {"time": "2026-10-07T12:00:00"}, {"time": "not-a-date"},
                       {"humidity": 101}, {"windSpeed": -1}, {"temperature": True}, {"pressure": "750"}):
            with self.subTest(update=update), self.assertRaises(YandexWeatherError):
                self.provider(current({**NOW, **update})).get_current_weather(60, 30)
        with self.assertRaises(YandexWeatherError):
            self.provider(forecast([], server=None)).get_hourly_forecast(60, 30)

    def test_invalid_coordinates_and_hours_before_request(self):
        provider = self.provider()
        for lat, lon in ((91, 30), (60, -181), (float("nan"), 0), (0, float("inf")), (True, 0)):
            with self.subTest(lat=lat, lon=lon), self.assertRaises(ValueError):
                provider.get_current_weather(lat, lon)
        for hours in (0, -1, 49, True, 2.5):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                provider.get_hourly_forecast(60, 30, hours)
        for past_hours in (-1, 25, True, 1.5):
            with self.subTest(past_hours=past_hours), self.assertRaises(ValueError):
                provider.get_hourly_forecast(60, 30, past_hours=past_hours)
        self.assertEqual(self.requests, [])

    def test_model_timezone_and_finite_values(self):
        for update in ({"timestamp": "2026-10-07"}, {"timestamp": 123}, {"temperature": float("nan")},
                       {"precipitation": -1}, {"latitude": 91}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                WeatherObservation.model_validate({"timestamp": NOW["time"], "latitude": 60, "longitude": 30, **update})

    def test_cli_current_and_forecast(self):
        for args, payload, is_list in (([], current(), False), (["--hours", "2"], forecast([hour("2026-10-07T13:00:00+03:00")]), True)):
            with self.subTest(args=args):
                provider = self.provider(payload)
                out, err = io.StringIO(), io.StringIO()
                with patch("external_data.weather.cli.YandexWeatherProvider", return_value=provider), redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(main(["--lat", "59.9343", "--lon", "30.3351", *args]), 0)
                self.assertEqual(isinstance(json.loads(out.getvalue()), list), is_list)
                self.assertEqual(err.getvalue(), "")
                self.assertNotIn(KEY, out.getvalue())
                self.assertTrue(provider._client.is_closed)

    def test_cli_missing_key_and_required_coordinates(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(["--lat", "60", "--lon", "30"]), 1)
        self.assertIn("YANDEX_WEATHER_API_KEY", err.getvalue())
        self.assertEqual(out.getvalue(), "")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
