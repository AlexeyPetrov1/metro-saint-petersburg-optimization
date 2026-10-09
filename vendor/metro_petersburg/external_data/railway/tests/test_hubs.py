import unittest

from external_data.railway import HUBS, get_hub


class HubTests(unittest.TestCase):
    def test_four_hub_mappings(self):
        self.assertEqual([(h.railway_name, h.rasp_station_code, h.metro_station) for h in HUBS], [
            ("Московский вокзал", "s9602494", "Площадь Восстания"),
            ("Финляндский вокзал", "s9602497", "Площадь Ленина"),
            ("Балтийский вокзал", "s9602498", "Балтийская"),
            ("Девяткино", "s9603876", "Девяткино"),
        ])

    def test_lookup_case_and_whitespace(self):
        self.assertEqual(get_hub("  МОСКОВСКИЙ вокзал  "), HUBS[0])

    def test_unknown_hub(self):
        with self.assertRaisesRegex(ValueError, "Неизвестный hub"):
            get_hub("Неизвестный вокзал")


if __name__ == "__main__":
    unittest.main()
