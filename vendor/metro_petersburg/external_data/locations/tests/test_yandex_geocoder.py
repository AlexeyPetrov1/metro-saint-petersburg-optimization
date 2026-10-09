import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import httpx

from external_data.locations import GeocoderError, LocationResolution, YandexGeocoderProvider, clear_station_cache
from external_data.locations.cli import main
from external_data.locations.geocoder import ENDPOINT


def payload(pos="30.3351 59.9343", address="Россия, Санкт-Петербург", name="место"):
    return {"response": {"GeoObjectCollection": {"featureMember": [{"GeoObject": {
        "Point": {"pos": pos}, "name": name,
        "metaDataProperty": {"GeocoderMetaData": {"Address": {"formatted": address}}},
    }}]}}}


class YandexGeocoderTests(unittest.TestCase):
    def setUp(self):
        self.key = "unit-test-only-geocoder-key"
        env = patch.dict(os.environ, {"YANDEX_GEOCODER_API_KEY": self.key}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        clear_station_cache()
        self.addCleanup(clear_station_cache)

    def provider(self, handler, *, debug=False):
        client = httpx.Client(transport=httpx.MockTransport(handler), timeout=7.0)
        self.addCleanup(client.close)
        with patch("external_data.locations.geocoder.httpx.Client", return_value=client) as factory:
            provider = YandexGeocoderProvider(timeout=7.0, debug=debug)
        factory.assert_called_once_with(timeout=7.0, follow_redirects=False)
        return provider

    def test_request_and_coordinate_order(self):
        def handler(request):
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url).split("?")[0], ENDPOINT)
            self.assertEqual(dict(request.url.params), {
                "apikey": self.key, "geocode": "БКЗ Октябрьский, Санкт-Петербург",
                "format": "json", "results": "5", "lang": "ru_RU",
            })
            self.assertEqual(request.extensions["timeout"]["read"], 7.0)
            return httpx.Response(200, json=payload(name="БКЗ Октябрьский"))
        result = self.provider(handler).geocode("БКЗ Октябрьский, Санкт-Петербург")
        self.assertEqual(result.latitude, 59.9343)
        self.assertEqual(result.longitude, 30.3351)
        self.assertEqual(result.formatted_address, "Россия, Санкт-Петербург")

    def test_missing_key_does_not_use_other_yandex_keys(self):
        with patch.dict(os.environ, {"YANDEX_API_KEY": "gpt", "YANDEX_WEATHER_API_KEY": "weather"}, clear=True):
            with self.assertRaisesRegex(GeocoderError, "YANDEX_GEOCODER_API_KEY"):
                YandexGeocoderProvider()

    def test_blank_key(self):
        with patch.dict(os.environ, {"YANDEX_GEOCODER_API_KEY": "  "}):
            with self.assertRaises(GeocoderError):
                YandexGeocoderProvider()

    def check_status(self, status):
        provider = self.provider(lambda request: httpx.Response(status, json={"message": self.key}), debug=True)
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaisesRegex(GeocoderError, f"HTTP {status}") as caught:
            provider.geocode("БКЗ Октябрьский")
        self.assertNotIn(self.key, str(caught.exception) + output.getvalue())
        self.assertIn(str(status), output.getvalue())

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
            raise httpx.ReadTimeout(f"secret url {request.url}", request=request)
        with self.assertRaisesRegex(GeocoderError, "timeout") as caught:
            self.provider(handler).geocode("место")
        self.assertNotIn(self.key, str(caught.exception))

    def test_network_error(self):
        def handler(request):
            raise httpx.ConnectError(str(request.url), request=request)
        with self.assertRaisesRegex(GeocoderError, "ошибка сети"):
            self.provider(handler).geocode("место")

    def test_empty_result(self):
        provider = self.provider(lambda request: httpx.Response(200, json={
            "response": {"GeoObjectCollection": {"featureMember": []}},
        }))
        self.assertIsNone(provider.geocode("Неизвестное место"))

    def test_missing_address_is_null(self):
        data = payload()
        del data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["metaDataProperty"]
        result = self.provider(lambda request: httpx.Response(200, json=data)).geocode("место")
        self.assertIsNone(result.formatted_address)

    def test_metadata_text_address_fallback(self):
        data = payload()
        data["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["metaDataProperty"] = {
            "GeocoderMetaData": {"text": "Адрес из text"},
        }
        result = self.provider(lambda request: httpx.Response(200, json=data)).geocode("место")
        self.assertEqual(result.formatted_address, "Адрес из text")

    def test_non_json_response(self):
        with self.assertRaisesRegex(GeocoderError, "не является JSON"):
            self.provider(lambda request: httpx.Response(200, text="<html>error</html>")).geocode("место")

    def test_malformed_response(self):
        for data in (None, {}, {"response": []}, payload("broken"), payload("181 95"), payload("nan inf")):
            with self.subTest(data=data), self.assertRaisesRegex(GeocoderError, "структура ответа или координаты"):
                self.provider(lambda request: httpx.Response(200, content=json.dumps(data))).geocode("место")

    def test_debug_redacts_echoed_key(self):
        provider = self.provider(lambda request: httpx.Response(200, json=payload(address=self.key)), debug=True)
        output = io.StringIO()
        with redirect_stderr(output):
            provider.geocode(f"место {self.key}")
        self.assertNotIn(self.key, output.getvalue())
        self.assertIn("[REDACTED]", output.getvalue())
        self.assertIn("latitude", output.getvalue())
        self.assertIn("HTTP status", output.getvalue())

    def test_debug_disabled(self):
        provider = self.provider(lambda request: httpx.Response(200, json=payload()))
        output = io.StringIO()
        with redirect_stderr(output):
            provider.geocode("место")
        self.assertEqual(output.getvalue(), "")

    def test_http_multiple_candidates_selects_second_with_debug(self):
        data = payload(name="Октябрьский путепровод")
        data["response"]["GeoObjectCollection"]["featureMember"] += payload(
            pos="30.0 60.0", name="БКЗ Октябрьский",
        )["response"]["GeoObjectCollection"]["featureMember"]
        provider = self.provider(lambda request: httpx.Response(200, json=data), debug=True)
        output = io.StringIO()
        with redirect_stderr(output):
            result = provider.geocode("БКЗ Октябрьский, Санкт-Петербург")
        self.assertEqual((result.longitude, result.latitude), (30.0, 60.0))
        for label in ("candidate 1", "candidate 2", "candidate score", "selected candidate"):
            self.assertIn(f"DEBUG {label}", output.getvalue())
        self.assertNotIn(self.key, output.getvalue())

    def test_http_fallback_request_count_bounded(self):
        queries = []
        def handler(request):
            queries.append(request.url.params["geocode"])
            return httpx.Response(200, json=payload(name="Октябрьский путепровод"))
        result = self.provider(handler).geocode("БКЗ Октябрьский, Санкт-Петербург")
        self.assertIsNone(result)
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(set(queries)), 3)

    def test_invalid_timeout(self):
        for value in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                YandexGeocoderProvider(timeout=value)

    def test_cli_mock_http_json_and_debug(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=payload(name=request.url.params["geocode"]))
        provider = self.provider(handler, debug=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("external_data.locations.cli.YandexGeocoderProvider", return_value=provider), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["--location", "БКЗ Октябрьский", "--top-k", "3", "--debug"]), 0)
        result = LocationResolution.model_validate_json(stdout.getvalue())
        self.assertEqual(len(result.nearest_stations), 3)
        self.assertEqual(len(requests), 20)
        self.assertNotIn(self.key, stdout.getvalue() + stderr.getvalue())
        self.assertTrue(provider._client.is_closed)

    def test_cli_http_error_exit(self):
        provider = self.provider(lambda request: httpx.Response(403, text=self.key), debug=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("external_data.locations.cli.YandexGeocoderProvider", return_value=provider), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["--location", "БКЗ Октябрьский", "--debug"]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("HTTP 403", stderr.getvalue())
        self.assertNotIn(self.key, stderr.getvalue())
        self.assertTrue(provider._client.is_closed)

    def test_address_http_uses_locality_metadata_and_not_venue_score(self):
        wrong = payload(name="6", address="Россия, Санкт-Петербург, Лиговский проспект, 6")
        obj = wrong["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
        obj["metaDataProperty"]["GeocoderMetaData"].update({
            "kind": "house", "Address": {"formatted": "Россия, Санкт-Петербург, Лиговский проспект, 6",
                "country_code": "RU", "Components": [{"kind": "locality", "name": "Москва"}]},
        })
        right = json.loads(json.dumps(wrong))
        right_obj = right["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
        right_obj["Point"]["pos"] = "30.0 60.0"
        right_obj["metaDataProperty"]["GeocoderMetaData"]["Address"]["Components"][0]["name"] = "Санкт-Петербург"
        wrong["response"]["GeoObjectCollection"]["featureMember"] += right["response"]["GeoObjectCollection"]["featureMember"]
        provider = self.provider(lambda request: httpx.Response(200, json=wrong))
        result = provider.geocode_address("Лиговский проспект, 6, Санкт-Петербург")
        self.assertEqual((result.latitude, result.longitude), (60.0, 30.0))

    def test_address_cli_debug_source_and_query(self):
        queries = []
        def handler(request):
            query = request.url.params["geocode"]
            queries.append(query)
            return httpx.Response(200, json=payload(name="6" if len(queries) == 1 else query,
                                                   address="Россия, Санкт-Петербург, Лиговский проспект, 6"))
        provider = self.provider(handler, debug=True)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("external_data.locations.cli.YandexGeocoderProvider", return_value=provider), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["--location", "БКЗ Октябрьский", "--address", "Лиговский проспект, 6", "--debug"]), 0)
        self.assertEqual(queries[0], "Лиговский проспект, 6, Санкт-Петербург")
        self.assertEqual(len(queries), 20)
        self.assertNotIn("БКЗ Октябрьский, Санкт-Петербург", queries)
        self.assertIn("DEBUG resolution source: address", stderr.getvalue())
        self.assertNotIn(self.key, stderr.getvalue() + stdout.getvalue())
        self.assertEqual(len(LocationResolution.model_validate_json(stdout.getvalue()).nearest_stations), 3)


if __name__ == "__main__":
    unittest.main()
