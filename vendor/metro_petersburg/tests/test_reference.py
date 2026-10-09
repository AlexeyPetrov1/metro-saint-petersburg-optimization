import pandas as pd
import pytest

from src import config, reference

HOURS = [*range(5, 24), 0]  # часы работы метро


@pytest.fixture(scope="module")
def stations():
    return reference.load_stations()


@pytest.fixture(scope="module")
def vestibules(stations):
    return reference.load_vestibules(stations=stations)


@pytest.fixture(scope="module")
def ops():
    return reference.load_operations()


def test_stations(stations):
    assert len(stations) == 19 and stations.has_data.sum() == 18
    assert stations.station_id.iloc[0] == "devyatkino" and stations.station_id.iloc[-1] == "prospekt_veteranov"
    tech = stations.set_index("station_id").loc["tekhnologichesky_institut"]
    assert not tech.has_data and tech.n_vestibules == 0


def test_vestibules(vestibules):
    assert len(vestibules) == 23 and vestibules.station_id.nunique() == 18
    assert vestibules.vestibule_id.tolist() == (vestibules.station_id + "_" + vestibules.vestibule_no.astype(str)).tolist()
    assert set(vestibules.loc[vestibules.closes_early, "vestibule_id"]) == {"leninsky_prospekt_2", "ploshchad_lenina_2"}
    two = vestibules.groupby("station_id").size()
    assert set(two[two == 2].index) == {"devyatkino", "ploshchad_lenina", "vosstaniya", "leninsky_prospekt",
                                        "prospekt_veteranov"}


def test_incidents(vestibules):
    inc = reference.load_incidents(vestibule_ids=vestibules.vestibule_id.tolist())
    assert len(inc) == 1
    row = inc.iloc[0]
    assert row.vestibule_id == reference.ALL_VESTIBULES
    assert (row.start_local, row.end_local) == (pd.Timestamp("2026-08-31 07:00"), pd.Timestamp("2026-08-31 11:00"))
    assert "Чернышевская — пл. Восстания" in row.description


def test_events(stations):
    ev = reference.load_events(station_ids=stations.station_id.tolist())
    raw_cols = pd.read_csv(config.SPB_EVENTS_CSV, nrows=0).columns.tolist()
    assert raw_cols == ["event", "start_local", "end_local", "venue", "nearest_station_id", "source", "verified"]
    assert not ev.verified.any()
    assert {"Парад Победы", "День города", "Алые паруса", "День знаний", "Продлённая работа метро"} <= set(ev.event)
    sails = ev[ev.event == "Алые паруса"].iloc[0]
    assert sails.start_local == pd.Timestamp("2026-06-28 01:00")   # праздник 27.06, ночь на 28.06


def test_bad_reference_raises(tmp_path, stations, vestibules):
    bad = tmp_path / "v.csv"
    pd.read_csv(config.SPB_VESTIBULES_CSV).assign(station_id="nowhere").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="station_id"):
        reference.load_vestibules(bad, stations=stations)

    bad = tmp_path / "i.csv"
    pd.read_csv(config.SPB_INCIDENTS_CSV).assign(vestibule_id="nowhere").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="неизвестные vestibule_id"):
        reference.load_incidents(bad, vestibule_ids=vestibules.vestibule_id.tolist())

    bad = tmp_path / "e.csv"
    pd.read_csv(config.SPB_EVENTS_CSV).drop(columns="venue").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="колонки"):
        reference.load_events(bad)


def test_operations_constants(ops):
    assert ops["turnaround_min"] == 99
    assert ops["pairs"]["max"] == 32 and ops["pairs"]["theoretical_max"] == 34
    assert ops["reserve"]["trains"] == sum(ops["reserve"]["depots"].values()) == 4
    cap = ops["train_capacity"]
    assert cap["used"] == cap["models"]["yubileyny_81_722"] == 1458
    assert 2 * 168 + 6 * 187 == cap["used"]
    assert ops["pairs"]["max"] * cap["used"] == 46_656


def test_pairs_plan(ops):
    plan = ops["pairs_plan"]
    assert plan["verified"] is False
    regimes = [k for k in plan if isinstance(plan[k], dict)]
    assert len(regimes) == 4
    for k in regimes:
        assert sorted(plan[k]) == sorted(HOURS), k
    pairs, trains = plan["weekday_from_2026_09_01"], plan["trains_weekday_from_2026_09_01"]
    for h in HOURS:  # составов ≈ парность × оборот / 60
        assert abs(trains[h] - pairs[h] * ops["turnaround_min"] / 60) <= 0.5, h
    assert max(pairs.values()) == ops["pairs"]["max"]


def test_operations_match_mo1(ops):
    if not config.MO1_OUT.exists():
        pytest.skip("нет mo1_reports.parquet")
    mo1 = pd.read_parquet(config.MO1_OUT)
    l1 = mo1[(mo1.line == "1") & (mo1.period == "2025")].set_index("indicator").value
    o = ops["operations_2025"]
    assert o["trains_fact"] == l1["Поездов по факту"]
    assert o["train_km"] == pytest.approx(l1["Поездо/км"])
    assert o["car_km"] == pytest.approx(l1["Общие ваг/км"])
    assert o["train_hours"] == pytest.approx(l1["Поездо/час"])
