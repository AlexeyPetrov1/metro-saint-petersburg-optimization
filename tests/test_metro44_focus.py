import copy
import gzip
import json

import pytest

import metro44_focus as focus
from simulation import digest


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source_case(tmp_path):
    root = tmp_path / "old"
    case = root / "cases" / "2026-09-30_peak_S1"
    run = root / "runs" / "original_KEEP"
    data = dict(period_start="2026-09-30T08:15:00+03:00", period_end="2026-09-30T14:15:00+03:00",
                plans=[{"plan_id": "original", "trips": []}], headway_policy={},
                stations=[{"station_id": "A", "kind": "line"}], trains=[])
    groups = [{"group_id": "actual", "count": 10}]
    checkpoint = {"second": 3600, "seen": [], "entered": 10}
    roster = [{"train_id": f"T{i}", "type": "Baltiets" if i < 37 else "Yubileyny"} for i in range(68)]
    contract = dict(case_id=case.name, od="S1", initial_count=53, service_seconds=14400,
                    movement_digest=digest(data), actual_groups_digest=digest(groups),
                    initial_checkpoint_digest=digest(checkpoint), policy_digest=digest({}))
    for name, value in {"experiment_manifest.json": {"contract": contract},
                        "actual_groups.json": groups, "warmup_checkpoint.json": checkpoint,
                        "fleet_registry.json": roster, "policy.json": {},
                        "keep_result.json": {"run_folder": "runs/original_KEEP", "metrics": {}},
                        "median_forecasts.json": [], "ml_forecasts.json": []}.items():
        dump(case / name, value)
    dump(run / "movement_input.json", data)
    dump(run / "profiles.json", {})
    return case


def test_case_keeps_the_real_0915_clock_and_twelve_quarters(source_case):
    shared, contract = focus.prepare_case(source_case)
    assert shared["checkpoint"]["second"] == 3600
    assert contract["decision_times"] == list(range(3600, 14400, 900))
    assert contract["first_decision_at"] == "2026-09-30T09:15:00+03:00"
    assert contract["last_decision_at"] == "2026-09-30T12:00:00+03:00"
    assert contract["service_end_at"] == "2026-09-30T12:15:00+03:00"
    assert contract["initial_count"] == 53
    assert len(shared["roster"]) == 68


def test_changed_actual_checkpoint_is_rejected_before_any_replay(source_case):
    dump(source_case / "warmup_checkpoint.json", {"second": 2700, "entered": 10})
    with pytest.raises(ValueError, match="checkpoint"):
        focus.prepare_case(source_case)


def test_changed_movement_cannot_silently_rebuild_another_warmup(source_case):
    path = source_case.parent.parent / "runs/original_KEEP/movement_input.json"
    data = json.loads(path.read_text())
    data["period_start"] = "2026-09-30T08:00:00+03:00"
    dump(path, data)
    with pytest.raises(ValueError, match="movement"):
        focus.prepare_case(source_case)


def test_cohort_clipping_never_claims_an_operational_closure_or_extends_truth():
    bundle = dict(source_versions={"model": "frozen"}, rows=[
        {"interval_start": "2026-09-30T12:00:00+03:00", "predicted_count": 120},
        {"interval_start": "2026-09-30T12:15:00+03:00", "predicted_count": 130}])
    before = copy.deepcopy(bundle)
    clipped = focus.cohort_forecast(bundle, "2026-09-30T12:15:00+03:00")
    assert bundle == before
    assert [row["predicted_count"] for row in clipped["rows"]] == [120, 0]
    scope = clipped["source_versions"]["finite_cohort_boundary"]
    assert scope["at"] == "2026-09-30T12:15:00+03:00"
    assert scope["boundary_end_is_operational_closure"] is False
    assert scope["observed_truth_extended"] is False
    assert scope["economic_claim_requires_common_continuation"] is True


def test_resume_grid_comes_from_the_original_checkpoint_and_requires_every_receipt(tmp_path):
    record = dict(contract_identity="paired", next_index=2, second=4500, state={"second": 4500}, data={})
    focus.write_gzip_once(tmp_path / "resume_04500.json.gz", record)
    with pytest.raises(ValueError, match="prefix"):
        focus.load_resume(tmp_path, "paired", [3600, 4500, 5400])
    record.update(next_index=1, second=3600, state={"second": 3600})
    focus.write_gzip_once(tmp_path / "resume_03600.json.gz", record)
    assert focus.load_resume(tmp_path, "paired", [3600, 4500, 5400])["second"] == 4500
    with pytest.raises(ValueError, match="contract"):
        focus.load_resume(tmp_path, "different", [3600, 4500, 5400])


def test_executed_prefix_detects_a_rewritten_past_but_allows_future_changes():
    old = {"events": [{"second": 1, "train_id": "A"}, {"second": 4000, "train_id": "B"}]}
    future = copy.deepcopy(old)
    future["events"][1]["train_id"] = "C"
    assert focus.execution_prefix(old, 3600) == focus.execution_prefix(future, 3600)
    future["events"][0]["train_id"] = "C"
    assert focus.execution_prefix(old, 3600) != focus.execution_prefix(future, 3600)


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []

    class Runner:
        def __init__(self, data, plan, groups, profiles):
            self.duration = 21600
            self.motion = {"events": [{"second": 0, "train_id": "A"}],
                           "checks": {"peak_trains_on_line": 53}, "obligations_closed": True}

        def run(self, *, start_second, checkpoint, end_second=None):
            assert start_second == checkpoint["second"] > 0
            end = self.duration if end_second is None else end_second
            assert end > start_second
            state = copy.deepcopy(checkpoint)
            state.update(second=end, seen=state["seen"] + [end])
            calls.append((start_second, end))
            return {"checkpoint": state}

        def branch_checkpoint(self, checkpoint, original):
            return copy.deepcopy(checkpoint)

    def metrics(data, runner, state, roster, policy, start, end):
        assert (start, end) == (3600, 14400)
        return dict(train_km=8271.934, mean_wait_seconds=69.0246, p95_wait_seconds=136,
                    actual_service_admissible=True, all_service_arrivals_completed=True,
                    movement_obligations_closed=True, cohort={"complete": True},
                    seen=state["seen"])

    monkeypatch.setattr(focus, "GroupedPassengerSimulation", Runner)
    monkeypatch.setattr(focus, "_metrics", metrics)
    monkeypatch.setattr(focus, "save_run", lambda *args, **kwargs: {"folder": "runs/test"})
    monkeypatch.setattr(focus, "source_hashes", lambda *args: {})
    return calls


def test_keep_replays_all_quarters_from_saved_checkpoint_and_refuses_overwrite(source_case, tmp_path, fake_runtime):
    output = tmp_path / "focus"
    result = focus.run_focus(output, "KEEP", source_case=source_case)
    assert result["metrics"]["fully_service_admissible"] is True
    assert result["boundary_end_is_operational_closure"] is False
    assert result["economic_claim_requires_common_continuation"] is True
    assert result["economic_improvement_claimed"] is False
    assert result["groups_digest"] == digest([{ "group_id": "actual", "count": 10}])
    assert result["checkpoint_digest"] == digest({"second": 3600, "seen": [], "entered": 10})
    assert result["metrics"]["seen"] == list(range(4500, 14400, 900)) + [21600]
    assert len(result["decisions"]) == 12
    assert all(d["applied_candidate_id"] == "KEEP" for d in result["decisions"])
    assert (output / "keep_result.json").exists()
    before = (output / "keep_result.json").read_bytes()
    with pytest.raises(ValueError, match="exists|overwrite"):
        focus.run_focus(output, "KEEP", source_case=source_case)
    assert (output / "keep_result.json").read_bytes() == before


def test_interrupted_focus_resumes_the_same_passengers_instead_of_resetting(source_case, tmp_path, fake_runtime):
    output = tmp_path / "focus"
    count = 0

    def calculate(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
                  candidate_limit, deadline, cancelled):
        nonlocal count
        count += 1
        assert state["second"] == second
        assert planning_end == 14400
        if count == 2:
            raise RuntimeError("interrupted")
        return {"decision": {"candidate_table": [], "selection": {"selected_candidate_id": None,
                       "fallback_candidate_id": "KEEP", "status": "keep"}},
                "candidates": [{"candidate_id": "KEEP", "_data": data}]}

    with pytest.raises(RuntimeError, match="interrupted"):
        focus.run_focus(output, "STANDARD", source_case=source_case, calculate=calculate,
                        forecast_cache=None)
    count = 2
    result = focus.run_focus(output, "STANDARD", source_case=source_case, calculate=calculate,
                             forecast_cache=None, resume=True)
    assert len(result["decisions"]) == 12
    assert result["metrics"]["seen"] == list(range(4500, 14400, 900)) + [21600]
    assert len(list((output / "controller/STANDARD").glob("resume_*.json.gz"))) == 12


def test_source_change_during_calculation_prevents_a_completed_result(source_case, tmp_path, fake_runtime):
    output = tmp_path / "focus"

    def calculate(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
                  candidate_limit, deadline, cancelled):
        if second == 3600:
            dump(source_case / "policy.json", {"changed": True})
        return {"decision": {"candidate_table": [], "selection": {"status": "keep"}},
                "candidates": [{"candidate_id": "KEEP", "_data": data}]}

    with pytest.raises(ValueError, match="input changed"):
        focus.run_focus(output, "STANDARD", source_case=source_case, calculate=calculate,
                        forecast_cache=None)
    assert not (output / "standard_result.json").exists()


def test_custom_catalog_still_receives_the_frozen_forecast_cache(source_case, tmp_path, fake_runtime, monkeypatch):
    prepared = []

    def prepare(output, source, shared, paired):
        prepared.append((output, source))
        return {}

    def calculate(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
                  candidate_limit, deadline, cancelled):
        assert source == (tmp_path / "focus").resolve()
        return {"decision": {"candidate_table": [], "selection": {"status": "keep"}},
                "candidates": [{"candidate_id": "KEEP", "_data": data}]}

    monkeypatch.setattr(focus, "prepare_forecasts", prepare)
    focus.run_focus(tmp_path / "focus", "STANDARD", source_case=source_case,
                    calculate=calculate, forecast_cache=tmp_path / "cache", pilot_steps=1)
    assert prepared == [((tmp_path / "focus").resolve(), tmp_path / "cache")]


def test_output_cannot_write_inside_the_original_experiment(source_case):
    with pytest.raises(ValueError, match="separate"):
        focus.run_focus(source_case.parent.parent / "new", "KEEP", source_case=source_case)


def test_freeze_covers_transitive_top_level_modules(tmp_path, monkeypatch):
    (tmp_path / "metro44_focus.py").write_text("# runner", encoding="utf-8")
    (tmp_path / "replay_metrics.py").write_text("# dependency", encoding="utf-8")
    monkeypatch.setattr(focus, "ROOT", tmp_path)
    monkeypatch.setattr(focus, "SOURCES", ())
    assert set(focus.source_hashes()) == {"metro44_focus.py", "replay_metrics.py"}


def test_cached_forecast_issue_is_0900_when_first_decision_is_0915(source_case, tmp_path, monkeypatch):
    bundle = dict(issued_at="2026-09-30T09:00:00+03:00", source_versions={"model": "frozen"},
                  rows=[dict(station_id="A", interval_start="2026-09-30T09:00:00+03:00",
                             interval_end="2026-09-30T09:15:00+03:00", predicted_count=10)])
    for mode in ("median", "ml"):
        dump(source_case / f"{mode}_forecasts.json", [bundle])
    cache = tmp_path / "cache"
    for mode in ("MEDIAN", "ML", "ORACLE"):
        for hour in (9, 10, 11, 12):
            dump(cache / "forecasts" / f"{mode}_{hour:02}.json", bundle)
    shared, paired = focus.prepare_case(source_case)
    monkeypatch.setattr(focus, "read_forecast", lambda *args: copy.deepcopy(bundle))
    receipt = focus.prepare_forecasts(tmp_path / "out", cache, shared, paired)
    assert receipt["first_issue_at"] == "2026-09-30T09:00:00+03:00"
    assert receipt["first_decision_at"] == "2026-09-30T09:15:00+03:00"
