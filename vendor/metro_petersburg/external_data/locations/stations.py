"""Canonical Line 1 names; station coordinates always come from a provider."""

from threading import RLock

from .geocoder import GeocoderProvider
from .models import GeocodedLocation

LINE1_STATIONS = (
    "Девяткино", "Гражданский проспект", "Академическая", "Политехническая",
    "Площадь Мужества", "Лесная", "Выборгская", "Площадь Ленина",
    "Чернышевская", "Площадь Восстания", "Владимирская", "Пушкинская",
    "Технологический институт", "Балтийская", "Нарвская", "Кировский завод",
    "Автово", "Ленинский проспект", "Проспект Ветеранов",
)

# Identity-based scope also works with unhashable custom provider objects.
# Strong references prevent id reuse; clear_station_cache releases them explicitly.
_station_cache: dict[int, tuple[GeocoderProvider, dict[str, GeocodedLocation | None]]] = {}
_cache_lock = RLock()


def clear_station_cache() -> None:
    """Release process-local cached station results and provider references."""
    with _cache_lock:
        _station_cache.clear()


def geocode_station(station_name: str, geocoder: GeocoderProvider) -> GeocodedLocation | None:
    if station_name not in LINE1_STATIONS:
        raise ValueError("station_name must be a canonical Line 1 station")
    with _cache_lock:
        _, cached = _station_cache.setdefault(id(geocoder), (geocoder, {}))
        if station_name not in cached:
            cached[station_name] = geocoder.geocode(f"метро {station_name}, Санкт-Петербург")
        return cached[station_name]
