"""Provider boundary and a network-free fake."""

from collections.abc import Mapping
from datetime import date
from typing import Protocol

from .models import RailwayArrival


class RailwayProvider(Protocol):
    def get_arrivals(self, station_code: str, date: date) -> list[RailwayArrival]:
        ...


class FakeRailwayProvider:
    def __init__(self, results: Mapping[tuple[str, date], list[RailwayArrival]]) -> None:
        self._results = {key: list(value) for key, value in results.items()}
        self.calls: list[tuple[str, date]] = []

    def get_arrivals(self, station_code: str, date: date) -> list[RailwayArrival]:
        self.calls.append((station_code, date))
        return list(self._results.get((station_code, date), []))
