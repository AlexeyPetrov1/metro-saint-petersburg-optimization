"""Yandex Rasp schedule adapter: rail arrivals, both transport types, all pages."""

import json
import math
import os
import re
import sys
from datetime import date as Date
from urllib.parse import quote, quote_plus

import httpx

from .features import deduplicate_arrivals
from .models import RailwayArrival

ENDPOINT = "https://api.rasp.yandex-net.ru/v3.0/schedule/"
PAGE_LIMIT = 100
MAX_PAGES = 100


class YandexRaspError(RuntimeError):
    """Credential-safe API/configuration error."""


class YandexRaspProvider:
    def __init__(self, *, timeout: float = 15.0, debug: bool = False) -> None:
        key = os.environ.get("YANDEX_RASP_API_KEY", "").strip()
        if not key:
            raise YandexRaspError("Задайте YANDEX_RASP_API_KEY — отдельный ключ Яндекс Расписаний.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self._key = key
        self.debug = debug
        self._secrets = {key} | {value.strip() for name, value in os.environ.items()
                                if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and value.strip()}
        self._client = httpx.Client(timeout=timeout, follow_redirects=False)

    def close(self) -> None:
        self._client.close()

    def _debug(self, label: str, value: object) -> None:
        if self.debug:
            message = json.dumps(value, ensure_ascii=False)
            for secret in sorted(self._secrets, key=len, reverse=True):
                for variant in (secret, json.dumps(secret, ensure_ascii=False)[1:-1], quote(secret, safe=""), quote_plus(secret)):
                    message = message.replace(variant, "[REDACTED]")
            print(f"DEBUG {label}: {message}", file=sys.stderr)

    def _request(self, station_code: str, day: Date, kind: str, offset: int) -> dict:
        self._debug("schedule request", {"station_code": station_code, "date": day.isoformat(),
                                         "transport_type": kind, "offset": offset})
        try:
            response = self._client.get(ENDPOINT, params={
                "apikey": self._key, "station": station_code, "date": day.isoformat(),
                "event": "arrival", "result_timezone": "Europe/Moscow", "transport_types": kind,
                "format": "json", "lang": "ru_RU", "limit": PAGE_LIMIT, "offset": offset,
            })
            self._debug("HTTP status", response.status_code)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            reason = {401: "ошибка авторизации", 403: "доступ запрещён: проверьте ключ и права",
                      429: "превышен лимит запросов; повторите позже"}.get(
                          status, "сервис временно недоступен" if status >= 500 else "запрос отклонён")
            raise YandexRaspError(f"Яндекс Расписания HTTP {status}: {reason}.") from None
        except httpx.TimeoutException:
            raise YandexRaspError("Яндекс Расписания: превышен timeout запроса.") from None
        except httpx.RequestError:
            raise YandexRaspError("Яндекс Расписания: ошибка сети.") from None
        try:
            data = response.json()
        except ValueError:
            raise YandexRaspError("Яндекс Расписания: некорректный JSON.") from None
        if not isinstance(data, dict) or not isinstance(data.get("schedule"), list):
            raise YandexRaspError("Яндекс Расписания: отсутствует список schedule.")
        station = data.get("station", {})
        if not isinstance(station, dict) or station.get("code", station_code) != station_code:
            raise YandexRaspError("Яндекс Расписания: ответ относится к другой станции.")
        if "date" in data and data["date"] != day.isoformat():
            raise YandexRaspError("Яндекс Расписания: ответ относится к другой дате.")
        return data

    def get_arrivals(self, station_code: str, date: Date) -> list[RailwayArrival]:
        if not isinstance(station_code, str) or not re.fullmatch(r"s[0-9]+", station_code):
            raise ValueError("station_code must be a Yandex station code s<digits>")
        if type(date) is not Date:
            raise TypeError("date must be datetime.date, not datetime or string")
        arrivals = []
        skipped = 0
        for kind in ("train", "suburban"):
            offset = 0
            for _ in range(MAX_PAGES):
                data = self._request(station_code, date, kind, offset)
                rows = data["schedule"]
                page = data.get("pagination")
                if (not isinstance(page, dict) or any(type(page.get(name)) is not int for name in ("total", "limit", "offset"))
                        or page["total"] < 0 or page["limit"] <= 0 or page["offset"] != offset
                        or len(rows) > page["limit"] or offset + len(rows) > page["total"]):
                    raise YandexRaspError("Яндекс Расписания: некорректная pagination; неполный результат не возвращён.")
                for row in rows:
                    try:
                        thread = row["thread"]
                        transport = thread["transport_type"]
                        if transport in ("plane", "bus", "water", "helicopter"):
                            continue
                        if row.get("is_fuzzy") is True:
                            raise ValueError("arrival time is not precise")
                        arrival = RailwayArrival(
                            arrival_time=row.get("arrival"), transport_type=transport,
                            train_number=_optional_text(thread.get("number")),
                            title=_optional_text(thread.get("title")) or _optional_text(thread.get("short_title")),
                            station_code=station_code,
                        )
                        arrivals.append(arrival)
                    except (KeyError, TypeError, ValueError):
                        skipped += 1
                offset += len(rows)
                if offset >= page["total"]:
                    break
                if not rows:
                    raise YandexRaspError("Яндекс Расписания: пустая страница до конца pagination.")
            else:
                raise YandexRaspError("Яндекс Расписания: превышен лимит страниц; неполный результат не возвращён.")
        if skipped:
            # Static reason and count only: no raw records, URL or exception text.
            print(f"Предупреждение: пропущено записей расписания: {skipped}; некорректные поля или нет точного arrival.", file=sys.stderr)
            if not arrivals:
                raise YandexRaspError("Яндекс Расписания: нет валидных прибытий среди повреждённых записей.")
        result = sorted(deduplicate_arrivals(arrivals), key=lambda item: item.arrival_time)
        self._debug("arrivals received", len(result))
        return result


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected string or null")
    return value.strip() or None
