import unittest

from external_data.locations import (
    FakeGeocoderProvider, GeocodedLocation, LINE1_STATIONS, LocationResolutionError,
    clear_station_cache, resolve_location_to_line1,
)
from external_data.locations.candidates import (
    GeocoderCandidate, candidate_score, query_variants, significant_tokens,
)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        clear_station_cache()
        self.addCleanup(clear_station_cache)
        self.query = "БКЗ Октябрьский, Санкт-Петербург"
        self.wrong = GeocoderCandidate(
            name="Октябрьский путепровод", formatted_address="Россия, Санкт-Петербург, Октябрьский путепровод",
            latitude=0.0, longitude=1.0, kind="street",
        )
        self.correct = GeocoderCandidate(
            name="БКЗ Октябрьский", formatted_address="Тестовый адрес зала",
            latitude=0.0, longitude=0.0, kind="other",
        )
        self.stations = {
            f"метро {name}, Санкт-Петербург": GeocodedLocation(
                query=f"метро {name}, Санкт-Петербург", latitude=0.0, longitude=float(i),
            ) for i, name in enumerate(LINE1_STATIONS)
        }

    def test_resolver_rejects_overpass_and_selects_second_candidate(self):
        geocoder = FakeGeocoderProvider({**self.stations, self.query: [self.wrong, self.correct]})
        result = resolve_location_to_line1("БКЗ Октябрьский", geocoder)
        self.assertEqual(result.location.formatted_address, "Тестовый адрес зала")
        self.assertEqual(result.location.longitude, 0.0)
        self.assertEqual(result.nearest_station.distance_m, 0.0)
        self.assertEqual(len(geocoder.queries), 20)

    def test_first_candidate_correct(self):
        geocoder = FakeGeocoderProvider({self.query: [self.correct, self.wrong]})
        self.assertEqual(geocoder.geocode(self.query).longitude, 0.0)
        self.assertEqual(geocoder.queries, [self.query])

    def test_third_candidate_correct(self):
        geocoder = FakeGeocoderProvider({self.query: [self.wrong, self.wrong, self.correct]})
        self.assertEqual(geocoder.geocode(self.query).longitude, 0.0)

    def test_no_match_rejected_before_station_requests(self):
        variants = query_variants(self.query)
        geocoder = FakeGeocoderProvider({**self.stations, **{q: [self.wrong] for q in variants}})
        with self.assertRaisesRegex(LocationResolutionError, "достаточной уверенностью"):
            resolve_location_to_line1("БКЗ Октябрьский", geocoder)
        self.assertEqual(geocoder.queries, variants)
        self.assertEqual(len(geocoder.queries), 3)

    def test_query_fallback_stops_on_match(self):
        variants = query_variants(self.query)
        self.assertEqual(variants, [self.query,
            "БКЗ Октябрьский концертный зал, Санкт-Петербург",
            "концертный зал БКЗ Октябрьский, Санкт-Петербург",
        ])
        geocoder = FakeGeocoderProvider({variants[0]: [self.wrong], variants[1]: [self.correct]})
        result = geocoder.geocode(self.query)
        self.assertEqual(result.query, self.query)
        self.assertEqual(result.longitude, 0.0)
        self.assertEqual(geocoder.queries, variants[:2])

    def test_empty_candidates_can_use_last_fallback(self):
        variants = query_variants(self.query)
        geocoder = FakeGeocoderProvider({variants[0]: [], variants[1]: [], variants[2]: [self.correct]})
        self.assertIsNotNone(geocoder.geocode(self.query))
        self.assertEqual(geocoder.queries, variants)

    def test_scoring_is_deterministic_and_uses_original_tokens(self):
        self.assertEqual(significant_tokens(self.query), {"бкз", "октябрьский"})
        for _ in range(3):
            self.assertEqual(candidate_score(self.query, self.wrong), 0.5)
            self.assertEqual(candidate_score(self.query, self.correct), 1.0)
        self.assertEqual(candidate_score("БКЗ «ОКТЯБРЬСКИЙ»", self.correct), 1.0)

    def test_generic_word_and_city_not_sufficient(self):
        for query, name in (
            ("Ледовый дворец, Санкт-Петербург", "Дворец культуры, Санкт-Петербург"),
            ("концертный зал, Санкт-Петербург", "концертный зал"),
            ("БКЗ Октябрьский, Санкт-Петербург", "концертный зал БКЗ"),
            ("Октябрьский, Санкт-Петербург", "Октябрьский путепровод"),
        ):
            with self.subTest(query=query):
                candidate = self.correct.model_copy(update={"name": name, "formatted_address": None})
                geocoder = FakeGeocoderProvider({q: [candidate] for q in query_variants(query)})
                self.assertIsNone(geocoder.geocode(query))

    def test_full_abbreviation_expansion(self):
        candidate = self.correct.model_copy(update={"name": "Большой концертный зал «Октябрьский»"})
        self.assertEqual(candidate_score(self.query, candidate), 1.0)
        self.assertIsNotNone(FakeGeocoderProvider({self.query: [candidate]}).geocode(self.query))

    def test_fields_can_supply_matching_tokens(self):
        for field in ("formatted_address", "description", "text"):
            with self.subTest(field=field):
                candidate = self.correct.model_copy(update={"name": None, "formatted_address": None, field: "БКЗ Октябрьский"})
                self.assertIsNotNone(FakeGeocoderProvider({self.query: [candidate]}).geocode(self.query))

    def test_station_transfer_suffix_allowed_without_extra_requests(self):
        query = "метро Технологический институт, Санкт-Петербург"
        candidate = self.correct.model_copy(update={"name": "Технологический институт-2", "kind": "metro"})
        geocoder = FakeGeocoderProvider({query: [candidate]})
        self.assertIsNotNone(geocoder.geocode(query))
        self.assertEqual(geocoder.queries, [query])

    def test_single_station_name(self):
        query = "метро Лесная, Санкт-Петербург"
        candidate = self.correct.model_copy(update={"name": "Лесная"})
        self.assertIsNotNone(FakeGeocoderProvider({query: [candidate]}).geocode(query))


if __name__ == "__main__":
    unittest.main()
