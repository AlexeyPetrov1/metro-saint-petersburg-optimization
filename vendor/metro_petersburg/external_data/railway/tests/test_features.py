import unittest
from datetime import date, datetime, timedelta

from pydantic import ValidationError

from external_data.railway import (
    FakeRailwayProvider, RailwayArrival, RailwayFeatures, build_railway_features,
    get_hub, get_railway_features,
)


class FeaturesTests(unittest.TestCase):
    def setUp(self):
        self.timestamp = datetime.fromisoformat("2026-10-08T18:00:00+03:00")
        self.hub = get_hub("Московский вокзал")

    def arrival(self, minutes, *, number="101", kind="train", **kwargs):
        values = dict(arrival_time=self.timestamp + timedelta(minutes=minutes), transport_type=kind,
                      train_number=number, station_code=self.hub.rasp_station_code)
        values.update(kwargs)
        return RailwayArrival(**values)

    def test_all_inclusive_windows_and_split(self):
        arrivals = [self.arrival(minutes, kind="suburban" if minutes == 30 else "train")
                    for minutes in (-1, 0, 15, 15.01, 30, 60, 120, 120.01)]
        result = build_railway_features(self.timestamp, self.hub, arrivals)
        self.assertEqual([result.arrivals_next_15m, result.arrivals_next_30m,
                          result.arrivals_next_60m, result.arrivals_next_120m], [2, 4, 5, 6])
        self.assertEqual(result.train_arrivals_next_30m, 3)
        self.assertEqual(result.suburban_arrivals_next_30m, 1)
        self.assertEqual(result.minutes_to_next_arrival, 0)

    def test_each_boundary_and_one_second_after(self):
        for window in (15, 30, 60, 120):
            with self.subTest(window=window):
                result = build_railway_features(self.timestamp, self.hub,
                    [self.arrival(window), self.arrival(window + 1 / 60)])
                self.assertEqual(getattr(result, f"arrivals_next_{window}m"), 1)

    def test_past_arrivals_and_empty_list(self):
        for arrivals in ([], [self.arrival(-0.01)]):
            result = build_railway_features(self.timestamp, self.hub, arrivals)
            self.assertTrue(all(value == 0 for name, value in result.model_dump().items() if "arrivals_next" in name))
            self.assertIsNone(result.minutes_to_next_arrival)

    def test_minutes_to_next_fractional_unsorted(self):
        result = build_railway_features(self.timestamp, self.hub, [self.arrival(90), self.arrival(2.5), self.arrival(-5)])
        self.assertEqual(result.minutes_to_next_arrival, 2.5)

    def test_next_arrival_can_be_outside_count_windows(self):
        result = build_railway_features(self.timestamp, self.hub, [self.arrival(180)])
        self.assertEqual(result.arrivals_next_120m, 0)
        self.assertEqual(result.minutes_to_next_arrival, 180)

    def test_naive_and_numeric_timestamps_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            build_railway_features(datetime(2026, 10, 8), self.hub, [])
        for value in ("2026-10-08T18:00:00", datetime(2026, 10, 8), 0):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.arrival(0, arrival_time=value)

    def test_timezone_normalized_and_compared_by_instant(self):
        result = build_railway_features(datetime.fromisoformat("2026-10-08T15:00:00Z"), self.hub,
            [self.arrival(0, arrival_time="2026-10-08T15:15:00Z")])
        self.assertEqual(result.timestamp.isoformat(), "2026-10-08T18:00:00+03:00")
        self.assertEqual(result.arrivals_next_15m, 1)

    def test_midnight_fetches_both_dates_and_deduplicates(self):
        self.timestamp = datetime.fromisoformat("2026-10-08T23:30:00+03:00")
        first, next_day = self.arrival(10), self.arrival(60)
        provider = FakeRailwayProvider({
            (self.hub.rasp_station_code, date(2026, 10, 8)): [first, next_day],
            (self.hub.rasp_station_code, date(2026, 10, 9)): [next_day, self.arrival(120)],
        })
        result = get_railway_features(self.timestamp, self.hub, provider)
        self.assertEqual([day for _, day in provider.calls], [date(2026, 10, 8), date(2026, 10, 9)])
        self.assertEqual(result.arrivals_next_15m, 1)
        self.assertEqual(result.arrivals_next_60m, 2)
        self.assertEqual(result.arrivals_next_120m, 3)

    def test_exact_midnight_endpoint_fetches_next_day(self):
        provider = FakeRailwayProvider({})
        get_railway_features(datetime.fromisoformat("2026-10-08T22:00:00+03:00"), self.hub, provider)
        self.assertEqual(len(provider.calls), 2)

    def test_uses_moscow_date_not_input_utc_date(self):
        provider = FakeRailwayProvider({})
        get_railway_features(datetime.fromisoformat("2026-10-08T22:00:00Z"), self.hub, provider)
        self.assertEqual(provider.calls, [(self.hub.rasp_station_code, date(2026, 10, 9))])

    def test_daytime_fetches_one_date(self):
        provider = FakeRailwayProvider({})
        get_railway_features(self.timestamp, self.hub, provider)
        self.assertEqual(provider.calls, [(self.hub.rasp_station_code, self.timestamp.date())])

    def test_deduplication_number_and_transport(self):
        arrival = self.arrival(5)
        result = build_railway_features(self.timestamp, self.hub, [arrival, arrival,
            self.arrival(5, number="102"), self.arrival(5, kind="suburban")])
        self.assertEqual(result.arrivals_next_15m, 3)

    def test_missing_number_uses_title_and_does_not_merge_other_titles(self):
        arrivals = [self.arrival(5, number=None, title=title) for title in ("Маршрут А", "маршрут  а", "Маршрут Б")]
        self.assertEqual(build_railway_features(self.timestamp, self.hub, arrivals).arrivals_next_15m, 2)

    def test_fully_anonymous_records_not_merged(self):
        arrival = self.arrival(5, number=None)
        self.assertEqual(build_railway_features(self.timestamp, self.hub, [arrival, arrival]).arrivals_next_15m, 2)

    def test_other_station_ignored(self):
        result = build_railway_features(self.timestamp, self.hub, [self.arrival(5, station_code="s9602497")])
        self.assertEqual(result.arrivals_next_120m, 0)

    def test_json_roundtrip_and_metadata(self):
        result = build_railway_features(self.timestamp, self.hub, [])
        self.assertEqual(RailwayFeatures.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(result.metro_station, "Площадь Восстания")
        self.assertEqual(result.railway_name, "Московский вокзал")


if __name__ == "__main__":
    unittest.main()
