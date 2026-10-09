from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from goal43_verify import verify


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(encoded) if path.suffix == ".gz" else encoded)


def tiny_journals(tmp_path):
    output, folder = tmp_path / "output", tmp_path / "output" / "tiny"
    train = dict(train_id="B001", train_type="Baltiets", cars=8, state="standby", location="A")
    movement = dict(period_start="2026-09-01T06:00:00+03:00", period_end="2026-09-01T06:10:00+03:00",
                    stations=[dict(station_id="A", kind="line", turnaround_seconds=0),
                              dict(station_id="B", kind="line", turnaround_seconds=180),
                              dict(station_id="D", kind="depot", turnaround_seconds=0)],
                    tracks=[dict(track_id="AB", from_station="A", to_station="B", length_km="1.5", travel_seconds=140),
                            dict(track_id="BD", from_station="B", to_station="D", length_km="0.2", travel_seconds=50)],
                    trains=[train], previous_departures=[], max_trains_on_line=53,
                    plans=[dict(plan_id="tiny", trips=[dict(trip_id="passenger", train_id="B001", departure_second=30,
                                                            route=["AB"], passenger=True),
                                                       dict(trip_id="withdraw", train_id="B001", departure_second=400,
                                                            route=["BD"], passenger=False)])])
    event_values = [(30, "departure", "AB", "standby", "moving", "passenger"),
                    (170, "arrival", "B", "moving", "turnaround", "passenger"),
                    (350, "turnaround_complete", "B", "turnaround", "standby", "passenger"),
                    (400, "departure", "BD", "standby", "moving", "withdraw"),
                    (450, "arrival", "D", "moving", "standby", "withdraw")]
    events = [dict(event_id=f"E{i}", second=second, event_type=kind, location=location,
                   previous_state=previous, next_state=next_state, trip_id=trip, train_id="B001")
              for i, (second, kind, location, previous, next_state, trip) in enumerate(event_values, 1)]
    work = dict(train_id="B001", train_type="Baltiets", cars=8,
                passenger_train_km="1.5", nonpassenger_train_km="0.2", train_km="1.7",
                passenger_car_km="12", nonpassenger_car_km="1.6", car_km="13.6",
                state_seconds=dict(moving=190, stop=0, turnaround=180, standby=230))
    motion = dict(duration_seconds=600, period_start=movement["period_start"], period_end=movement["period_end"],
                  initial_state=[train], events=events, transport_work=[work],
                  final_state=[dict(train_id="B001", location="D", state="standby")],
                  checks=dict(peak_trains_on_line=1), release_authorized=False)
    groups = [dict(group_id="G1", origin="A", turnstile_second=0, platform_second=10, count=4,
                   destinations=[dict(station_id="B", direction=1, count=4)]),
              dict(group_id="G2", origin="A", turnstile_second=5, platform_second=20, count=3,
                   destinations=[dict(station_id="B", direction=1, count=3)]),
              dict(group_id="G3", origin="A", turnstile_second=590, platform_second=610, count=2,
                   destinations=[dict(station_id="B", direction=1, count=2)])]
    profiles = {"B001": [dict(car_id=f"C{i}", normal_threshold=120, max_capacity=160) for i in range(8)]}
    queue = dict(group_id="G2", origin="A", destination="B", direction=1,
                 turnstile_second=5, platform_second=20, count=3)
    loads = [dict(second=second, train_id="B001", car_id=f"C{i}", load=int(second == 30 and i < 4),
                  motion_event_id=event_id, event_type=kind)
             for second, event_id, kind in [(30, "E1", "departure"), (170, "E2", "arrival"), (450, "E5", "arrival")]
             for i in range(8)]
    checkpoint = dict(second=600, entered=9, platform_arrived=7, boarded=4, completed=4,
                      approaching=2, queued=3, onboard_count=0, waiting_passenger_seconds=1820,
                      wait_histogram={"20": 4}, queues={"A:1": [queue]}, queue_counts={"A:1": 3},
                      onboard={"B001": [[] for _ in range(8)]}, loads={"B001": [0] * 8},
                      max_car_load=1, load_events=loads,
                      board_events=[dict(second=30, count=4, wait_seconds=20, group_id="G1", motion_event_id="E1")])
    metrics = dict(max_car_load=1, train_km=1.7,
                   policy_exposure={"global": dict(passenger_car_seconds=560, exceptional_car_seconds=0,
                                                    comfort_car_seconds=0, hard_limit_car_seconds=0,
                                                    exceptional_time_share=0.0, max_load=1)},
                   resources={key: value for key, value in work.items() if key.endswith("_km")},
                   cohort=dict(arrived=9, boarded=4, unboarded=5,
                               waiting_passenger_seconds_lower_bound=1820, complete=False,
                               mean_wait_seconds=None, p95_wait_seconds=None,
                               boarded_mean_wait_seconds=20.0, boarded_p95_wait_seconds=20))
    summary = dict(metrics=metrics)
    objects = {"movement_input.json": movement, "motion.json": motion,
               "groups.json.gz": groups, "profiles.json": profiles,
               "passenger_checkpoint.json.gz": checkpoint, "summary.json": summary}
    for name, value in objects.items():
        write_json(folder / name, value)
    write_json(output / "runs.json", [dict(run_id="tiny", folder="tiny", scenario_id="control",
                                          strategy="KEEP", od="S1", seed=0, metrics=metrics, full_day=False)])
    return output, folder, objects


def test_verifier_reconstructs_kilometers_conservation_wait_and_car_time(tmp_path):
    output, _, _ = tiny_journals(tmp_path)
    result = verify(output)
    assert result["status"] == "passed"
    assert result["run_count"] == 1
    metrics = result["runs"][0]["reconstructed_metrics"]
    assert metrics["resources"]["passenger_train_km"] == 1.5
    assert metrics["resources"]["nonpassenger_train_km"] == 0.2
    assert metrics["waiting_passenger_seconds"] == 1820
    assert metrics["passenger_car_seconds"] == 560
    assert metrics["peak_trains_on_line"] == 1
    assert (output / "verification_report.json").exists()


@pytest.mark.parametrize("filename,change,reason", [
    ("motion.json", lambda value: value["transport_work"][0].update(passenger_train_km="9"), "mileage"),
    ("passenger_checkpoint.json.gz", lambda value: value.update(waiting_passenger_seconds=1819), "waiting"),
    ("passenger_checkpoint.json.gz", lambda value: value.update(entered=10), "entered"),
    ("passenger_checkpoint.json.gz", lambda value: value["wait_histogram"].update({"20": 5}), "histogram"),
    ("passenger_checkpoint.json.gz", lambda value: value["loads"]["B001"].__setitem__(0, 1), "load"),
    ("summary.json", lambda value: value["metrics"]["policy_exposure"]["global"].update(passenger_car_seconds=999), "exposure"),
])
def test_tampering_is_detected_and_reported(tmp_path, filename, change, reason):
    output, folder, objects = tiny_journals(tmp_path)
    value = deepcopy(objects[filename])
    change(value)
    write_json(folder / filename, value)
    with pytest.raises(ValueError, match=reason):
        verify(output)
    report = json.loads((output / "verification_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"


def test_offpeak_headway_includes_previous_boundary_departure(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    value = deepcopy(objects["movement_input.json"])
    value["previous_departures"] = [dict(track_id="AB", departure_second=-119)]
    write_json(folder / "movement_input.json", value)
    with pytest.raises(ValueError, match="headway"):
        verify(output)


def test_coordinated_load_and_metric_tampering_still_fails_boarding_identity(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    state = deepcopy(objects["passenger_checkpoint.json.gz"])
    state["load_events"][0]["load"] = 2
    state["max_car_load"] = 2
    summary = deepcopy(objects["summary.json"])
    summary["metrics"]["max_car_load"] = 2
    summary["metrics"]["policy_exposure"]["global"].update(passenger_car_seconds=560, max_load=2)
    write_json(folder / "passenger_checkpoint.json.gz", state)
    write_json(folder / "summary.json", summary)
    runs = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    runs[0]["metrics"] = summary["metrics"]
    write_json(output / "runs.json", runs)
    with pytest.raises(ValueError, match="boarding load"):
        verify(output)


def test_long_incident_gap_is_reported_without_rewriting_it(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    value = deepcopy(objects["movement_input.json"])
    value["previous_departures"] = [dict(track_id="AB", departure_second=-600)]
    value["incidents"] = [dict(start_second=0, end_second=30, station_id="A")]
    write_json(folder / "movement_input.json", value)
    result = verify(output)
    violation = result["runs"][0]["violations"][0]
    assert violation["kind"] == "maximum_headway"
    assert violation["actual_seconds"] == 630
    assert violation["cause"] == "incident_blocking"


def test_missing_run_file_is_an_integrity_failure(tmp_path):
    output, folder, _ = tiny_journals(tmp_path)
    (folder / "groups.json.gz").unlink()
    with pytest.raises(ValueError, match="groups"):
        verify(output)


def test_line_capacity_includes_every_initial_train(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    value = deepcopy(objects["movement_input.json"])
    value["trains"] = [dict(value["trains"][0], train_id=f"B{i:03}") for i in range(54)]
    write_json(folder / "movement_input.json", value)
    with pytest.raises(ValueError, match="line capacity"):
        verify(output)


def test_cohort_and_resource_scopes_are_independent(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    summary = deepcopy(objects["summary.json"])
    summary["cohort_scope"] = dict(start_second=0, end_second=100)
    summary["resources_scope"] = dict(start_second=300, end_second=600)
    summary["metrics"]["cohort"].update(arrived=7, unboarded=3)
    summary["metrics"]["train_km"] = 0.2
    summary["metrics"]["resources"] = dict(passenger_train_km=0, nonpassenger_train_km=0.2,
        passenger_car_km=0, nonpassenger_car_km=1.6, train_km=0.2, car_km=1.6,
        state_seconds=dict(moving=50, turnaround=50, standby=200, holding=0))
    write_json(folder / "summary.json", summary)
    runs = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    runs[0]["metrics"] = summary["metrics"]
    write_json(output / "runs.json", runs)
    result = verify(output)
    assert result["runs"][0]["reconstructed_metrics"]["resources"]["train_km"] == 0.2
    assert result["runs"][0]["reconstructed_metrics"]["cohort"]["arrived"] == 7


def test_closing_arrival_counts_mileage_while_passenger_boundary_is_pending(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    movement, motion = deepcopy(objects["movement_input.json"]), deepcopy(objects["motion.json"])
    movement["period_end"] = motion["period_end"] = "2026-09-01T06:07:30+03:00"
    motion["duration_seconds"] = 450
    motion["transport_work"][0]["state_seconds"]["standby"] = 80
    state = deepcopy(objects["passenger_checkpoint.json.gz"])
    state.update(second=450, entered=7, approaching=0, waiting_passenger_seconds=1370)
    state["load_events"] = [row for row in state["load_events"] if row["second"] < 450]
    summary = deepcopy(objects["summary.json"])
    summary["metrics"]["cohort"].update(arrived=7, unboarded=3, waiting_passenger_seconds_lower_bound=1370)
    write_json(folder / "movement_input.json", movement)
    write_json(folder / "motion.json", motion)
    write_json(folder / "passenger_checkpoint.json.gz", state)
    write_json(folder / "summary.json", summary)
    runs = json.loads((output / "runs.json").read_text(encoding="utf-8"))
    runs[0]["metrics"] = summary["metrics"]
    write_json(output / "runs.json", runs)
    result = verify(output)
    assert result["runs"][0]["reconstructed_metrics"]["resources"]["train_km"] == 1.7
    assert result["runs"][0]["reconstructed_metrics"]["waiting_passenger_seconds"] == 1370


def test_raw_files_remain_immutable_and_changed_source_is_verified_in_archive(tmp_path):
    output, _, _ = tiny_journals(tmp_path)
    workspace = tmp_path / "workspace"
    raw, source = workspace / "data/raw/old.bin", workspace / "simulation.py"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"original raw")
    source.write_bytes(b"original source")
    hashes = {str(path.relative_to(workspace)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (raw, source)}
    archive = output / "baseline_source"
    archive.mkdir()
    (archive / "simulation.py").write_bytes(source.read_bytes())
    source.write_bytes(b"adapted source")
    write_json(output / "sourcebaseline_manifest.json", dict(workspace_root=str(workspace),
               source_files=["simulation.py"], files=hashes))
    assert verify(output)["manifest_checks"]["status"] == "passed"
    raw.write_bytes(b"changed raw")
    with pytest.raises(ValueError, match="immutable"):
        verify(output)


def test_archive_tampering_is_detected(tmp_path):
    output, _, _ = tiny_journals(tmp_path)
    archive = output / "baseline_source"
    archive.mkdir()
    (archive / "simulation.py").write_bytes(b"incorrect")
    write_json(output / "sourcebaseline_manifest.json", dict(workspace_root=str(tmp_path),
               source_files=["simulation.py"], files={"simulation.py": hashlib.sha256(b"original").hexdigest()}))
    with pytest.raises(ValueError, match="archived"):
        verify(output)


def test_relative_output_path_resolves_default_archive_once(tmp_path, monkeypatch):
    output, _, _ = tiny_journals(tmp_path)
    archive = output / "baseline_source"
    archive.mkdir()
    (archive / "simulation.py").write_bytes(b"original")
    write_json(output / "sourcebaseline_manifest.json", dict(workspace_root=str(tmp_path),
               source_files=["simulation.py"], files={"simulation.py": hashlib.sha256(b"original").hexdigest()}))
    monkeypatch.chdir(tmp_path)
    assert verify("output")["manifest_checks"]["status"] == "passed"


def test_depot_return_gaps_do_not_violate_passenger_service_maximum(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    data = deepcopy(objects["movement_input.json"])
    data["previous_departures"] = [dict(track_id="BD", departure_second=-500)]
    write_json(folder / "movement_input.json", data)
    result = verify(output)
    assert not any(row.get("track_id") == "BD" and row["kind"] == "maximum_headway"
                   for row in result["runs"][0]["violations"])


def test_completed_gap_in_return_tail_is_outside_explicit_service_scope(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    data = deepcopy(objects["movement_input.json"])
    data["previous_departures"] = [dict(track_id="AB", departure_second=-451)]
    summary = deepcopy(objects["summary.json"])
    summary["service_scope"] = dict(start_second=0, end_second=20)
    write_json(folder / "movement_input.json", data)
    write_json(folder / "summary.json", summary)
    assert not verify(output)["runs"][0]["violations"]


def test_service_boundary_age_is_checked_even_without_next_departure(tmp_path):
    output, folder, objects = tiny_journals(tmp_path)
    summary = deepcopy(objects["summary.json"])
    summary["service_scope"] = dict(start_second=0, end_second=600)
    write_json(folder / "summary.json", summary)
    result = verify(output)
    assert any(row["kind"] == "maximum_headway" and row.get("boundary") == "end"
               and row["actual_seconds"] == 570 for row in result["runs"][0]["violations"])
