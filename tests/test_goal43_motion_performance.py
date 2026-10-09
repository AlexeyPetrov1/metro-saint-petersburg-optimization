import pytest

from simulation import simulate


def depot_case(active=53, entry=150):
    data = dict(schema_version="1.0", source="explicit line-occupancy oracle",
        period_start="2026-09-01T06:00:00+03:00", period_end="2026-09-01T06:04:00+03:00",
        stations=[dict(station_id="A", kind="line", stop_seconds=0, turnaround_seconds=20),
                  dict(station_id="B", kind="line", stop_seconds=20, turnaround_seconds=0),
                  dict(station_id="D", kind="depot", stop_seconds=0, turnaround_seconds=0)],
        tracks=[dict(track_id="EXIT", from_station="B", to_station="D", length_km="1", travel_seconds=10),
                dict(track_id="ENTRY", from_station="D", to_station="A", length_km="1", travel_seconds=10),
                dict(track_id="RETURN", from_station="A", to_station="D", length_km="1", travel_seconds=10),
                dict(track_id="LINK", from_station="A", to_station="B", length_km="1", travel_seconds=10)],
        trains=[dict(train_id=f"T{i}", train_type="fixture", cars=8, state="stop", location="B",
                     ready_after_seconds=20) for i in range(active)], max_trains_on_line=53)
    data["trains"].append(dict(train_id="R", train_type="fixture", cars=8, state="standby", location="D", ready_after_seconds=0))
    data["plans"] = [dict(plan_id="P", trips=[
        dict(trip_id="RETURN", train_id="T0", departure_second=20, route=["EXIT"], passenger=False),
        dict(trip_id="ENTER", train_id="R", departure_second=entry, route=["ENTRY"], passenger=False),
        dict(trip_id="RETURN_R", train_id="R", departure_second=max(entry+30, 200), route=["RETURN"], passenger=False)])]
    return data


def reconstructed_occupancy(data, events):
    line = {s["station_id"] for s in data["stations"] if s["kind"] == "line"}
    track_endpoints = {t["track_id"]: {t["from_station"], t["to_station"]} for t in data["tracks"]}
    locations = {t["train_id"]: t["location"] for t in data["trains"]}

    def count():
        return sum(bool(track_endpoints[place] & line) if place in track_endpoints else place in line
                   for place in locations.values())

    counts = [(0, count())]
    for event in events:
        locations[event["train_id"]] = event["location"]
        counts.append((event["second"], count()))
    return counts


def test_stopped_trains_entry_and_return_match_independent_location_ledger():
    data = depot_case()
    out = simulate(data, "P")
    counts = reconstructed_occupancy(data, out["events"])
    assert max(n for _, n in counts) == out["checks"]["peak_trains_on_line"] == 53
    assert (20, 53) in counts  # returning train still occupies the line during its exit movement
    assert (30, 52) in counts
    assert (150, 53) in counts  # entry occupies the line from departure, before arrival
    assert counts[-1] == (210, 52)


def test_initial_54_stopped_trains_exceed_the_limit():
    with pytest.raises(ValueError, match="initial line capacity exceeded: 54 > 53"):
        simulate(depot_case(active=54), "P")


def test_reserve_entering_before_return_completes_exceeds_53():
    with pytest.raises(ValueError, match="line capacity exceeded at 10: 54 > 53"):
        simulate(depot_case(entry=10), "P")


def test_unfinished_exit_still_counts_as_line_occupancy():
    data = depot_case(entry=150)
    data["period_end"] = "2026-09-01T06:00:25+03:00"
    out = simulate(data, "P")
    assert out["checks"]["peak_trains_on_line"] == 53
    assert any(r["train_id"] == "T0" for r in out["unfinished_movements"])
    assert reconstructed_occupancy(data, out["events"])[-1][1] == 53


class ObservedStation(dict):
    def __init__(self, row, reads):
        super().__init__(row)
        self.reads = reads

    def __getitem__(self, key):
        if key == "kind": self.reads.append(key)
        return super().__getitem__(key)


def test_line_occupancy_work_scales_with_events_instead_of_events_times_fleet():
    data = depot_case()
    data["period_end"] = "2026-09-01T11:35:00+03:00"
    data["plans"][0]["trips"] = [dict(trip_id=f"ROUND{i}", train_id="T0",
        departure_second=20+200*i, route=["EXIT", "ENTRY", "LINK"], passenger=False) for i in range(100)]
    for i in range(14):
        data["trains"].append(dict(data["trains"][-1], train_id=f"D{i}"))
    reads = []
    data["stations"] = [ObservedStation(row, reads) for row in data["stations"]]
    out = simulate(data, "P")
    assert out["checks"]["peak_trains_on_line"] == max(n for _, n in reconstructed_occupancy(data, out["events"]))
    assert len(reads) <= 4*len(out["events"])+2*len(data["trains"])+100
