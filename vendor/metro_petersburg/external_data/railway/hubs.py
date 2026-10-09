"""The four railway hubs selected for the MVP, using the task's station codes."""

from .models import RailwayHub

HUBS = (
    RailwayHub(railway_name="Московский вокзал", rasp_station_code="s9602494", metro_station="Площадь Восстания"),
    RailwayHub(railway_name="Финляндский вокзал", rasp_station_code="s9602497", metro_station="Площадь Ленина"),
    RailwayHub(railway_name="Балтийский вокзал", rasp_station_code="s9602498", metro_station="Балтийская"),
    RailwayHub(railway_name="Девяткино", rasp_station_code="s9603876", metro_station="Девяткино"),
)


def get_hub(name: str) -> RailwayHub:
    if isinstance(name, str):
        for hub in HUBS:
            if hub.railway_name.casefold() == name.strip().casefold():
                return hub
    raise ValueError("Неизвестный hub. Доступны: " + ", ".join(hub.railway_name for hub in HUBS))
