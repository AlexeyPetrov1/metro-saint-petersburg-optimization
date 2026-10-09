"""Replaceable geocoder and a small synchronous Yandex Maps API adapter."""

import json
import math
import os
import re
import sys
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import quote, quote_plus

import httpx

from .models import GeocodedLocation
from .candidates import GeocoderCandidate, is_petersburg_address, resolve_address_candidates, resolve_candidates

ENDPOINT = "https://geocode-maps.yandex.ru/v1/"


class GeocoderError(RuntimeError):
    """Safe error without request URLs, credentials or raw HTTP response bodies."""


class GeocoderProvider(Protocol):
    def geocode(self, query: str) -> GeocodedLocation | None:
        ...


class AddressGeocoderProvider(Protocol):
    """Optional capability: address matching, separate from venue-name scoring."""

    def geocode_address(self, query: str) -> GeocodedLocation | None:
        ...


class FakeGeocoderProvider:
    """Return only preconfigured coordinates; no inference or network access."""

    def __init__(self, results: Mapping[str, GeocodedLocation | list[GeocoderCandidate] | None]) -> None:
        self._results = dict(results)
        self.queries: list[str] = []

    def geocode(self, query: str) -> GeocodedLocation | None:
        if isinstance(self._results.get(query), list):
            return resolve_candidates(query, self._fetch_candidates)
        self.queries.append(query)
        return self._results.get(query)

    def _fetch_candidates(self, query: str) -> list[GeocoderCandidate]:
        self.queries.append(query)
        result = self._results.get(query)
        if result is None:
            return []
        if not isinstance(result, list):
            raise TypeError("Candidate fixtures must contain lists for fallback queries too")
        return result

    def geocode_address(self, query: str) -> GeocodedLocation | None:
        if isinstance(self._results.get(query), list):
            return resolve_address_candidates(query, self._fetch_candidates)
        self.queries.append(query)
        result = self._results.get(query)
        return result if result is not None and is_petersburg_address(result.formatted_address) else None


class YandexGeocoderProvider:
    def __init__(self, *, timeout: float = 15.0, debug: bool = False) -> None:
        key = os.environ.get("YANDEX_GEOCODER_API_KEY", "").strip()
        if not key:
            raise GeocoderError("Задайте YANDEX_GEOCODER_API_KEY — отдельный ключ Yandex Maps Geocoder.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self._key = key
        self.debug = debug
        self._secrets = {key} | {
            value.strip() for name, value in os.environ.items()
            if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and value.strip()
        }
        self._client = httpx.Client(timeout=timeout, follow_redirects=False)

    def close(self) -> None:
        self._client.close()

    def _debug(self, label: str, value: object) -> None:
        if not self.debug:
            return
        # Only whitelisted values, never request/response objects or headers.
        message = json.dumps(value, ensure_ascii=False)
        for secret in sorted(self._secrets, key=len, reverse=True):
            for variant in (secret, json.dumps(secret, ensure_ascii=False)[1:-1],
                            json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret)):
                message = message.replace(variant, "[REDACTED]")
        print(f"DEBUG {label}: {message}", file=sys.stderr)

    def geocode(self, query: str) -> GeocodedLocation | None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return resolve_candidates(query, self._fetch_candidates, self._debug)

    def geocode_address(self, query: str) -> GeocodedLocation | None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return resolve_address_candidates(query, self._fetch_candidates, self._debug)

    def _fetch_candidates(self, query: str) -> list[GeocoderCandidate]:
        self._debug("query", query)
        try:
            response = self._client.get(ENDPOINT, params={
                "apikey": self._key, "geocode": query, "format": "json",
                "lang": "ru_RU", "results": 5,
            })
            self._debug("HTTP status", response.status_code)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            reason = {
                401: "ошибка авторизации: проверьте Geocoder API key",
                403: "доступ запрещён: проверьте Geocoder API key и права доступа",
                429: "превышен лимит запросов; повторите позже",
            }.get(status, "сервис временно недоступен" if status >= 500 else "запрос отклонён")
            raise GeocoderError(f"Yandex Geocoder HTTP {status}: {reason}.") from None
        except httpx.TimeoutException:
            raise GeocoderError("Yandex Geocoder: превышен timeout запроса.") from None
        except httpx.RequestError:
            raise GeocoderError("Yandex Geocoder: ошибка сети при обращении к API.") from None
        try:
            payload = response.json()
        except ValueError:
            raise GeocoderError("Yandex Geocoder: ответ не является JSON.") from None
        try:
            members = payload["response"]["GeoObjectCollection"]["featureMember"]
            if not isinstance(members, list):
                raise ValueError
            candidates = []
            for member in members:
                obj = member["GeoObject"]
                # Yandex Point.pos order is longitude latitude, not latitude longitude.
                longitude, latitude = (float(value) for value in obj["Point"]["pos"].split())
                metadata = obj.get("metaDataProperty", {}).get("GeocoderMetaData", {})
                address_data = metadata.get("Address", {})
                address = address_data.get("formatted") or metadata.get("text")
                localities = [part["name"] for part in address_data.get("Components", []) if part.get("kind") == "locality"]
                candidates.append(GeocoderCandidate(
                    latitude=latitude, longitude=longitude, formatted_address=address,
                    name=obj.get("name"), description=obj.get("description"),
                    text=metadata.get("text"), kind=metadata.get("kind"),
                    locality=localities[-1] if localities else None,
                    country_code=address_data.get("country_code"),
                ))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise GeocoderError("Yandex Geocoder: некорректная структура ответа или координаты.") from None
        return candidates
