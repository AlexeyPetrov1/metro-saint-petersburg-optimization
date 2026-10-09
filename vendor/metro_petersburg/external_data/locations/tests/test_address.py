import unittest

from external_data.locations import (
    FakeGeocoderProvider, GeocodedLocation, LINE1_STATIONS, LocationResolutionError,
    clear_station_cache, resolve_location_to_line1,
)
from external_data.locations.candidates import GeocoderCandidate


class AddressResolutionTests(unittest.TestCase):
    def setUp(self):
        clear_station_cache()
        self.addCleanup(clear_station_cache)
        self.name = "БКЗ Октябрьский"
        self.name_query = self.name + ", Санкт-Петербург"
        self.address = "Лиговский проспект, 6"
        self.query = self.address + ", Санкт-Петербург"
        # Synthetic points only; do not represent real station coordinates.
        self.fixtures = {f"метро {name}, Санкт-Петербург": GeocodedLocation(
            query=name, latitude=0.0, longitude=float(index),
        ) for index, name in enumerate(LINE1_STATIONS)}
        self.candidate = GeocoderCandidate(
            name="6", latitude=0.0, longitude=9.0, kind="house",
            locality="Санкт-Петербург", country_code="RU",
            formatted_address="Россия, Санкт-Петербург, Лиговский просп., 6",
        )
        self.fixtures[self.query] = [self.candidate]
        self.fixtures[self.name_query] = GeocodedLocation(query=self.name_query, latitude=0.0, longitude=18.0)

    def test_address_priority_and_haversine_top_k(self):
        geocoder = FakeGeocoderProvider(self.fixtures)
        result = resolve_location_to_line1(self.name, geocoder, address=self.address)
        self.assertEqual(geocoder.queries[0], self.query)
        self.assertNotIn(self.name_query, geocoder.queries)
        self.assertEqual(result.nearest_station.station_name, "Площадь Восстания")
        self.assertEqual(result.nearest_station.distance_m, 0.0)
        self.assertEqual(len(result.nearest_stations), 3)
        self.assertEqual(result.location.query, self.query)
        self.assertEqual(result.location_name, self.name)

    def test_top_k_one(self):
        result = resolve_location_to_line1(self.name, FakeGeocoderProvider(self.fixtures), 1, self.address)
        self.assertEqual(len(result.nearest_stations), 1)

    def test_address_none_preserves_old_path(self):
        geocoder = FakeGeocoderProvider(self.fixtures)
        result = resolve_location_to_line1(self.name, geocoder, address=None)
        self.assertEqual(geocoder.queries[0], self.name_query)
        self.assertNotIn(self.query, geocoder.queries)
        self.assertEqual(result.location.longitude, 18.0)

    def test_address_not_found_uses_venue_fallback(self):
        self.fixtures[self.query] = []
        geocoder = FakeGeocoderProvider(self.fixtures)
        result = resolve_location_to_line1(self.name, geocoder, address=self.address)
        self.assertEqual(geocoder.queries[:2], [self.query, self.name_query])
        self.assertEqual(result.location.longitude, 18.0)

    def test_wrong_city_is_rejected_even_if_text_mentions_petersburg(self):
        self.fixtures[self.query] = [self.candidate.model_copy(update={"locality": "Москва"})]
        self.fixtures[self.name_query] = None
        with self.assertRaises(LocationResolutionError):
            resolve_location_to_line1(self.name, FakeGeocoderProvider(self.fixtures), address=self.address)

    def test_only_city_result_rejected(self):
        self.fixtures[self.query] = [self.candidate.model_copy(update={"kind": "locality"})]
        self.fixtures[self.name_query] = None
        with self.assertRaises(LocationResolutionError):
            resolve_location_to_line1(self.name, FakeGeocoderProvider(self.fixtures), address=self.address)

    def test_formatted_city_fallback_requires_city_component(self):
        for address, accepted in (("Россия, Санкт-Петербург, Лиговский проспект, 6", True),
                                  ("Россия, Москва, улица Санкт-Петербург, 6", False),
                                  (None, False)):
            with self.subTest(address=address):
                candidate = self.candidate.model_copy(update={"locality": None, "formatted_address": address})
                geocoder = FakeGeocoderProvider({self.query: [candidate]})
                self.assertEqual(geocoder.geocode_address(self.query) is not None, accepted)

    def test_city_not_appended_twice(self):
        geocoder = FakeGeocoderProvider(self.fixtures)
        resolve_location_to_line1(self.name, geocoder, address=self.query)
        self.assertEqual(geocoder.queries[0], self.query)

    def test_existing_geocode_only_provider_supported(self):
        fixtures = dict(self.fixtures)
        fixtures[self.query] = GeocodedLocation(query=self.query, latitude=0.0, longitude=9.0,
                                               formatted_address="Санкт-Петербург, Лиговский проспект, 6")
        class LegacyProvider:
            def geocode(self, query):
                return fixtures.get(query)
        result = resolve_location_to_line1(self.name, LegacyProvider(), address=self.address)
        self.assertEqual(result.location.longitude, 9.0)

    def test_blank_address_rejected_without_requests(self):
        geocoder = FakeGeocoderProvider(self.fixtures)
        for address in ("", "   ", 123):
            with self.subTest(address=address), self.assertRaises(ValueError):
                resolve_location_to_line1(self.name, geocoder, address=address)
        self.assertEqual(geocoder.queries, [])


if __name__ == "__main__":
    unittest.main()
