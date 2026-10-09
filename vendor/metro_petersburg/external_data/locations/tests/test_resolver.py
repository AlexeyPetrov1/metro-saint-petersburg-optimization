import math
import unittest

from pydantic import ValidationError

from external_data.locations import (
    FakeGeocoderProvider, GeocodedLocation, GeoPoint, LINE1_STATIONS,
    LocationResolution, LocationResolutionError, clear_station_cache,
    haversine_distance, resolve_location_to_line1,
)


class ResolverTests(unittest.TestCase):
    def setUp(self):
        clear_station_cache()
        self.addCleanup(clear_station_cache)
        self.query = "БКЗ Октябрьский, Санкт-Петербург"
        # Deliberately synthetic geometry, NOT real station coordinates.
        self.results = {
            f"метро {name}, Санкт-Петербург": GeocodedLocation(
                query=f"метро {name}, Санкт-Петербург", latitude=0.0, longitude=float(19 - i),
            ) for i, name in enumerate(LINE1_STATIONS)
        }
        self.results[self.query] = GeocodedLocation(query=self.query, latitude=0.0, longitude=0.0)
        self.geocoder = FakeGeocoderProvider(self.results)

    def test_canonical_stations(self):
        self.assertEqual(LINE1_STATIONS, (
            "Девяткино", "Гражданский проспект", "Академическая", "Политехническая",
            "Площадь Мужества", "Лесная", "Выборгская", "Площадь Ленина",
            "Чернышевская", "Площадь Восстания", "Владимирская", "Пушкинская",
            "Технологический институт", "Балтийская", "Нарвская", "Кировский завод",
            "Автово", "Ленинский проспект", "Проспект Ветеранов",
        ))

    def test_haversine_one_degree_equator(self):
        a = GeoPoint(latitude=0.0, longitude=0.0)
        b = GeoPoint(latitude=0.0, longitude=1.0)
        self.assertAlmostEqual(haversine_distance(a, b), 111_195.08, delta=0.01)
        self.assertEqual(haversine_distance(a, b), haversine_distance(b, a))

    def test_haversine_antipodes_and_date_line(self):
        a = GeoPoint(latitude=0.0, longitude=0.0)
        b = GeoPoint(latitude=0.0, longitude=180.0)
        self.assertAlmostEqual(haversine_distance(a, b), math.pi * 6_371_008.8)
        self.assertAlmostEqual(haversine_distance(
            GeoPoint(latitude=0.0, longitude=179.5),
            GeoPoint(latitude=0.0, longitude=-179.5),
        ), 111_195.08, delta=0.01)

    def test_sorting_all_stations(self):
        result = resolve_location_to_line1("БКЗ Октябрьский", self.geocoder, top_k=19)
        self.assertEqual([s.station_name for s in result.nearest_stations], list(reversed(LINE1_STATIONS)))
        distances = [s.distance_m for s in result.nearest_stations]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(result.nearest_station, result.nearest_stations[0])

    def test_top_k_one(self):
        result = resolve_location_to_line1("БКЗ Октябрьский", self.geocoder, top_k=1)
        self.assertEqual(len(result.nearest_stations), 1)
        self.assertEqual(result.nearest_station.station_name, "Проспект Ветеранов")
        self.assertEqual(len(self.geocoder.queries), 20)  # Ranking still considers all 19.

    def test_top_k_three_default(self):
        result = resolve_location_to_line1("БКЗ Октябрьский", self.geocoder)
        self.assertEqual([s.station_name for s in result.nearest_stations], [
            "Проспект Ветеранов", "Ленинский проспект", "Автово",
        ])

    def test_exactly_on_station(self):
        self.results[self.query] = GeocodedLocation(query=self.query, latitude=0.0, longitude=10.0)
        result = resolve_location_to_line1("БКЗ Октябрьский", FakeGeocoderProvider(self.results))
        self.assertEqual(result.nearest_station.station_name, "Площадь Восстания")
        self.assertAlmostEqual(result.nearest_station.distance_m, 0.0, places=8)

    def test_equal_distances_preserve_canonical_order(self):
        for query in self.results:
            self.results[query] = GeocodedLocation(query=query, latitude=0.0, longitude=0.0)
        result = resolve_location_to_line1("БКЗ Октябрьский", FakeGeocoderProvider(self.results))
        self.assertEqual([s.station_name for s in result.nearest_stations], list(LINE1_STATIONS[:3]))

    def test_location_not_found(self):
        with self.assertRaisesRegex(LocationResolutionError, "Место не найдено"):
            resolve_location_to_line1("Неизвестное место", self.geocoder)
        self.assertEqual(self.geocoder.queries, ["Неизвестное место, Санкт-Петербург"])

    def test_one_station_not_found(self):
        query = "метро Лесная, Санкт-Петербург"
        self.results[query] = None
        geocoder = FakeGeocoderProvider(self.results)
        for _ in range(2):
            with self.assertRaisesRegex(LocationResolutionError, "Лесная"):
                resolve_location_to_line1("БКЗ Октябрьский", geocoder)
        self.assertEqual(geocoder.queries.count(query), 1)  # Empty station result is cached too.

    def test_station_geocoding_cached(self):
        for _ in range(2):
            resolve_location_to_line1("БКЗ Октябрьский", self.geocoder)
        self.assertEqual(len(self.geocoder.queries), 21)
        for name in LINE1_STATIONS:
            self.assertEqual(self.geocoder.queries.count(f"метро {name}, Санкт-Петербург"), 1)
        self.assertEqual(self.geocoder.queries.count(self.query), 2)

    def test_cache_isolated_by_provider_and_clearable(self):
        another = FakeGeocoderProvider(self.results)
        resolve_location_to_line1("БКЗ Октябрьский", self.geocoder)
        resolve_location_to_line1("БКЗ Октябрьский", another)
        self.assertEqual(len(another.queries), 20)
        clear_station_cache()
        resolve_location_to_line1("БКЗ Октябрьский", self.geocoder)
        self.assertEqual(len(self.geocoder.queries), 40)

    def test_queries_have_city_context(self):
        resolve_location_to_line1("  БКЗ Октябрьский  ", self.geocoder)
        self.assertEqual(self.geocoder.queries[0], self.query)
        self.assertEqual(self.geocoder.queries[1:], [f"метро {name}, Санкт-Петербург" for name in LINE1_STATIONS])
        resolve_location_to_line1(self.query, self.geocoder)
        self.assertEqual(self.geocoder.queries[-1], self.query)

    def test_invalid_inputs_do_not_geocode(self):
        for value in (0, -1, 20, 1.5, True, None):
            with self.subTest(top_k=value), self.assertRaises(ValueError):
                resolve_location_to_line1("БКЗ Октябрьский", self.geocoder, top_k=value)
        for name in ("", "   ", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_location_to_line1(name, self.geocoder)
        self.assertEqual(self.geocoder.queries, [])

    def test_coordinate_validation(self):
        for lat, lon in ((91, 0), (0, 181), (float("nan"), 0), (0, float("inf")), (True, 0)):
            with self.subTest(lat=lat, lon=lon), self.assertRaises(ValidationError):
                GeoPoint(latitude=lat, longitude=lon)

    def test_json_contract_has_no_time_fields(self):
        result = resolve_location_to_line1("БКЗ Октябрьский", self.geocoder)
        self.assertEqual(LocationResolution.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(set(result.model_dump()), {"location_name", "location", "nearest_station", "nearest_stations"})
        self.assertEqual(set(result.location.model_dump()), {"query", "latitude", "longitude", "formatted_address"})


if __name__ == "__main__":
    unittest.main()
