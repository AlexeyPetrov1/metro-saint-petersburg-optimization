"""Minimal Yandex Weather v3 GraphQL client; no coupling to events or LLMs."""

import json
import math
import os
import re
import sys
from datetime import timedelta
from typing import Any

import httpx
from pydantic import ValidationError

from .models import WeatherObservation

ENDPOINT = "https://api.weather.yandex.ru/graphql/query"
FIELDS = """
    temperature(unit: CELSIUS)
    precType
    windSpeed(unit: METERS_PER_SECOND)
    windGust(unit: METERS_PER_SECOND)
    humidity
    pressure(unit: MM_HG)
    condition
"""


class YandexWeatherError(RuntimeError):
    """Safe diagnostic with redacted GraphQL messages, never full responses."""


class YandexWeatherProvider:
    def __init__(self, *, timeout: float = 15.0, include_feels_like: bool = False, debug: bool = False) -> None:
        key = os.environ.get("YANDEX_WEATHER_API_KEY", "").strip()
        if not key:
            raise YandexWeatherError("Задайте отдельный ключ YANDEX_WEATHER_API_KEY; ключ YandexGPT не подходит.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self._fields = FIELDS + ("    feelsLike(unit: CELSIUS)\n" if include_feels_like else "")
        self.debug = debug
        # Snapshot credentials so errors remain redacted even if env later changes.
        self._secrets = {key} | {
            value.strip() for name, value in os.environ.items()
            if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and value.strip()
        }
        self._client = httpx.Client(
            timeout=timeout, headers={"X-Yandex-Weather-Key": key}, follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def _redact(self, text: str) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            for variant in (secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]):
                text = text.replace(variant, "[REDACTED]")
        # Escape control characters from untrusted server messages, including newlines.
        return re.sub(r"[\x00-\x1f\x7f]", lambda match: json.dumps(match.group())[1:-1], text)

    def _debug(self, label: str, value: object) -> None:
        if self.debug:
            print(f"DEBUG {label}: " + self._redact(json.dumps(value, ensure_ascii=False)), file=sys.stderr)

    def _error_messages(self, payload: object) -> list[str]:
        if not isinstance(payload, dict) or not payload.get("errors"):
            return []
        errors = payload["errors"]
        messages = [
            self._redact(error["message"]) for error in errors
            if isinstance(error, dict) and isinstance(error.get("message"), str)
        ] if isinstance(errors, list) else []
        messages = messages or ["GraphQL errors без текстового message"]
        self._debug("GraphQL errors", messages)
        return messages

    @staticmethod
    def _point(lat: float, lon: float) -> dict[str, float]:
        for name, value, bound in (("latitude", lat, 90), ("longitude", lon, 180)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -bound <= value <= bound:
                raise ValueError(f"{name} must be a finite number in [-{bound}, {bound}]")
        return {"lat": float(lat), "lon": float(lon)}

    def _request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        self._debug("GraphQL query", query)
        self._debug("variables", variables)
        try:
            response = self._client.post(ENDPOINT, json={"query": query, "variables": variables})
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            reason = {
                401: "неверный или отсутствующий Weather API key",
                403: "доступ запрещён: проверьте Weather API key и тариф",
                429: "превышен лимит запросов; повторите позже",
            }.get(status, "сервис временно недоступен" if status >= 500 else "запрос отклонён")
            try:
                messages = self._error_messages(error.response.json())
            except ValueError:
                messages = []
            details = " GraphQL error: " + "; ".join(messages) if messages else ""
            raise YandexWeatherError(f"Yandex Weather HTTP {status}: {reason}.{details}") from None
        except httpx.TimeoutException:
            raise YandexWeatherError("Yandex Weather: превышен timeout запроса.") from None
        except httpx.RequestError:
            raise YandexWeatherError("Yandex Weather: ошибка сети при обращении к API.") from None
        try:
            payload = response.json()
        except ValueError:
            raise YandexWeatherError("Yandex Weather: ответ не является JSON.") from None
        if not isinstance(payload, dict):
            raise YandexWeatherError("Yandex Weather: ожидается JSON-объект.")
        messages = self._error_messages(payload)
        if messages:
            # Partial GraphQL success isn't silently turned into complete features.
            raise YandexWeatherError("Yandex Weather GraphQL error: " + "; ".join(messages))
        data = payload.get("data")
        if not isinstance(data, dict):
            raise YandexWeatherError("Yandex Weather: отсутствует объект data.")
        return data

    @staticmethod
    def _object(value: object, name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise YandexWeatherError(f"Yandex Weather: отсутствует или некорректен объект {name}.")
        return value

    @staticmethod
    def _observation(row: dict[str, Any], point: dict[str, float], *, forecast: bool) -> WeatherObservation:
        try:
            return WeatherObservation(
                timestamp=row.get("time"), latitude=point["lat"], longitude=point["lon"],
                temperature=row.get("temperature"), feels_like=row.get("feelsLike"),
                precipitation=row.get("accumulatedPrec") if forecast else None,
                precipitation_type=row.get("precType"), wind_speed=row.get("windSpeed"),
                wind_gust=row.get("windGust"), humidity=row.get("humidity"),
                pressure=row.get("pressure"), condition=row.get("condition"),
            )
        except ValidationError:
            raise YandexWeatherError("Yandex Weather: некорректные погодные значения или timestamp без часового пояса.") from None

    def get_current_weather(self, lat: float, lon: float) -> WeatherObservation:
        point = self._point(lat, lon)
        query = "query CurrentWeather($point: PointInput!) { serverTime weatherByPoint(request: $point) { now {" + self._fields + "} } }"
        data = self._request(query, {"point": point})
        weather = self._object(data.get("weatherByPoint"), "weatherByPoint")
        now = self._object(weather.get("now"), "now")
        # Now.time has no listed tariff. Root serverTime is available on Test plan.
        # This is API response time, not a meteorological station measurement time.
        return self._observation({**now, "time": data.get("serverTime")}, point, forecast=False)

    def get_hourly_forecast(
        self, lat: float, lon: float, hours: int = 2, *, past_hours: int = 0,
    ) -> list[WeatherObservation]:
        """API hour points within [serverTime - past_hours, serverTime + hours], no interpolation.

        By default (past_hours=0) the past portion of the current hour is excluded.
        past_hours > 0 also keeps today's already started hours, as the API still
        forecasts them; days before today aren't requested. Missing slots aren't
        synthesized; fewer samples may be returned if the API has limited data.
        """
        point = self._point(lat, lon)
        if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 48:
            raise ValueError("hours must be an integer in [1, 48]")
        if isinstance(past_hours, bool) or not isinstance(past_hours, int) or not 0 <= past_hours <= 24:
            raise ValueError("past_hours must be an integer in [0, 24]")
        query = """query HourlyWeather($point: PointInput!, $days: Int!) {
            serverTime
            weatherByPoint(request: $point) {
                forecast { days(limit: $days) { hours {
        """ + self._fields + " time accumulatedPrec } } } } }"
        data = self._request(query, {"point": point, "days": math.ceil(hours / 24) + 1})
        # Use API time, not the local clock; validate it just like an observation.
        server_time = self._observation({"time": data.get("serverTime")}, point, forecast=False).timestamp
        weather = self._object(data.get("weatherByPoint"), "weatherByPoint")
        forecast = self._object(weather.get("forecast"), "forecast")
        days = forecast.get("days")
        if not isinstance(days, list):
            raise YandexWeatherError("Yandex Weather: отсутствует список forecast.days.")
        observations = []
        for day in days:
            rows = self._object(day, "forecast.days[]").get("hours")
            if not isinstance(rows, list):
                raise YandexWeatherError("Yandex Weather: отсутствует список forecast.days[].hours.")
            for row in rows:
                observation = self._observation(self._object(row, "hours[]"), point, forecast=True)
                if server_time - timedelta(hours=past_hours) <= observation.timestamp <= server_time + timedelta(hours=hours):
                    observations.append(observation)
        observations.sort(key=lambda item: item.timestamp)
        if len({item.timestamp for item in observations}) != len(observations):
            raise YandexWeatherError("Yandex Weather: дублирующиеся timestamps прогноза.")
        return observations
