import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from unittest.mock import patch

import httpx

from external_data.railway import RailwayFeatures, YandexRaspError, YandexRaspProvider
from external_data.railway.cli import main
from external_data.railway.yandex_rasp import ENDPOINT


def row(kind="train", number="101", arrival="2026-10-08T18:10:00+03:00"):
    return {"arrival": arrival, "departure": "2026-10-08T23:00:00+03:00",
            "thread": {"transport_type": kind, "number": number, "title": "Москва — Санкт-Петербург"}}


def page(rows, *, offset=0, total=None):
    return {"schedule": rows, "pagination": {"offset": offset, "limit": 100, "total": len(rows) if total is None else total}}


class YandexRaspTests(unittest.TestCase):
    def setUp(self):
        self.key = "test-only-rasp-secret"
        env = patch.dict(os.environ, {"YANDEX_RASP_API_KEY": self.key}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.requests = []

    def provider(self, handler, *, debug=False):
        def record(request):
            self.requests.append(request)
            return handler(request)
        client = httpx.Client(transport=httpx.MockTransport(record), timeout=7)
        self.addCleanup(client.close)
        with patch("external_data.railway.yandex_rasp.httpx.Client", return_value=client) as factory:
            result = YandexRaspProvider(timeout=7, debug=debug)
        factory.assert_called_once_with(timeout=7, follow_redirects=False)
        return result

    def fetch(self, provider):
        return provider.get_arrivals("s9602494", date(2026, 10, 8))

    def test_requests_parameters_and_parsing(self):
        def handler(request):
            self.assertEqual(str(request.url).split("?")[0], ENDPOINT)
            params = dict(request.url.params)
            self.assertEqual(params, {"apikey": self.key, "station": "s9602494", "date": "2026-10-08",
                "event": "arrival", "result_timezone": "Europe/Moscow", "transport_types": params["transport_types"],
                "format": "json", "lang": "ru_RU", "offset": "0", "limit": "100"})
            self.assertEqual(request.extensions["timeout"]["read"], 7)
            return httpx.Response(200, json=page([row(params["transport_types"])]))
        result = self.fetch(self.provider(handler))
        self.assertEqual(len(result), 2)
        self.assertEqual([item.transport_type for item in result], ["train", "suburban"])
        self.assertEqual(result[0].arrival_time.isoformat(), "2026-10-08T18:10:00+03:00")
        self.assertEqual(result[0].train_number, "101")

    def test_pagination_includes_later_pages(self):
        def handler(request):
            if request.url.params["transport_types"] == "suburban":
                return httpx.Response(200, json=page([]))
            offset = int(request.url.params["offset"])
            return httpx.Response(200, json=page([row(number=str(offset))], offset=offset, total=2))
        self.assertEqual(len(self.fetch(self.provider(handler))), 2)
        self.assertEqual([r.url.params["offset"] for r in self.requests], ["0", "1", "0"])

    def test_combined_responses_deduplicated(self):
        result = self.fetch(self.provider(lambda request: httpx.Response(200, json=page([row()]))))
        self.assertEqual(len(result), 1)

    def test_page_limit_prevents_unbounded_requests(self):
        def handler(request):
            offset = int(request.url.params["offset"])
            return httpx.Response(200, json=page([row(number=str(offset))], offset=offset, total=1000))
        provider = self.provider(handler)
        with patch("external_data.railway.yandex_rasp.MAX_PAGES", 2), self.assertRaisesRegex(YandexRaspError, "лимит страниц"):
            self.fetch(provider)
        self.assertEqual(len(self.requests), 2)

    def test_missing_key_does_not_use_other_keys(self):
        with patch.dict(os.environ, {"YANDEX_API_KEY": "other", "YANDEX_WEATHER_API_KEY": "other",
                                     "YANDEX_GEOCODER_API_KEY": "other"}, clear=True):
            with self.assertRaisesRegex(YandexRaspError, "YANDEX_RASP_API_KEY"):
                YandexRaspProvider()

    def check_status(self, status):
        provider = self.provider(lambda request: httpx.Response(status, text=self.key), debug=True)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(YandexRaspError, f"HTTP {status}") as caught:
            self.fetch(provider)
        self.assertNotIn(self.key, stderr.getvalue() + str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    def test_401(self):
        self.check_status(401)

    def test_403(self):
        self.check_status(403)

    def test_429(self):
        self.check_status(429)

    def test_500(self):
        self.check_status(500)

    def test_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout(str(request.url), request=request)
        with self.assertRaisesRegex(YandexRaspError, "timeout") as caught:
            self.fetch(self.provider(handler))
        self.assertNotIn(self.key, str(caught.exception))

    def test_network_error(self):
        def handler(request):
            raise httpx.ConnectError(str(request.url), request=request)
        with self.assertRaisesRegex(YandexRaspError, "ошибка сети"):
            self.fetch(self.provider(handler))

    def test_invalid_json(self):
        with self.assertRaisesRegex(YandexRaspError, "JSON"):
            self.fetch(self.provider(lambda request: httpx.Response(200, text="not json")))

    def test_malformed_envelope_and_pagination(self):
        for payload in (None, {}, {"schedule": None}, {"schedule": []},
                        {**page([]), "station": None}, {**page([]), "date": "2026-10-09"},
                        page([], total=1), page([], offset=10)):
            with self.subTest(payload=payload), self.assertRaises(YandexRaspError):
                self.fetch(self.provider(lambda request: httpx.Response(200, content=json.dumps(payload))))

    def test_bad_rows_skipped_with_safe_diagnostic(self):
        data = [row(), row(arrival=None), row(arrival="2026-10-08T18:10:00"),
                row(arrival=self.key), {"departure": "2026-10-08T18:10:00+03:00"},
                {**row(), "is_fuzzy": True}]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = self.fetch(self.provider(lambda request: httpx.Response(200, json=page(data))))
        self.assertEqual(len(result), 1)
        self.assertIn("пропущено записей", stderr.getvalue())
        self.assertNotIn(self.key, stderr.getvalue())

    def test_all_bad_rows_not_silently_zero(self):
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(YandexRaspError, "нет валидных"):
            self.fetch(self.provider(lambda request: httpx.Response(200, json=page([row(arrival=None)]))))

    def test_other_transport_excluded_and_missing_metadata_null(self):
        data = [row(kind=kind) for kind in ("plane", "bus", "water", "helicopter")]
        data.append({"arrival": "2026-10-08T18:10:00+03:00", "thread": {"transport_type": "train"}})
        result = self.fetch(self.provider(lambda request: httpx.Response(200, json=page(data))))
        self.assertTrue(all(item.transport_type == "train" for item in result))
        self.assertIsNone(result[0].train_number)
        self.assertIsNone(result[0].title)

    def test_short_title_fallback(self):
        data = row()
        data["thread"].update(title="", short_title="Краткое название")
        result = self.fetch(self.provider(lambda request: httpx.Response(200, json=page([data]))))
        self.assertEqual(result[0].title, "Краткое название")

    def test_empty_schedule(self):
        self.assertEqual(self.fetch(self.provider(lambda request: httpx.Response(200, json=page([])))), [])

    def test_invalid_arguments(self):
        provider = self.provider(lambda request: self.fail("Unexpected HTTP"))
        for code, day in (("bad", date(2026, 10, 8)), ("s9602494", "2026-10-08")):
            with self.assertRaises((TypeError, ValueError)):
                provider.get_arrivals(code, day)
        for timeout in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                YandexRaspProvider(timeout=timeout)

    def test_cli_json_and_debug_without_key(self):
        provider = self.provider(lambda request: httpx.Response(200, json=page([row()])), debug=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("external_data.railway.cli.YandexRaspProvider", return_value=provider), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["--hub", "Московский вокзал", "--timestamp", "2026-10-08T18:00:00+03:00", "--debug"]), 0)
        result = RailwayFeatures.model_validate_json(stdout.getvalue())
        self.assertEqual(result.arrivals_next_15m, 1)
        self.assertIn("HTTP status", stderr.getvalue())
        self.assertIn("s9602494", stderr.getvalue())
        self.assertNotIn(self.key, stdout.getvalue() + stderr.getvalue())
        self.assertTrue(provider._client.is_closed)

    def test_unknown_hub_cli_no_requests(self):
        with patch("external_data.railway.cli.YandexRaspProvider") as factory, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--hub", "Нет", "--timestamp", "2026-10-08T18:00:00+03:00"]), 1)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
