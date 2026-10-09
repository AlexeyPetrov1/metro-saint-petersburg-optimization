import copy
from types import SimpleNamespace

import pytest

import goal43_actions as actions
from goal43_policy import load_policy


def candidate_fixture(monkeypatch):
    data = {"plans": [{"plan_id": "P", "trips": []}], "period_start": "2026-09-30T11:00:00+03:00"}
    calls = []
    monkeypatch.setattr(actions, "compile_operational_timing", lambda source, *a, **k: (copy.deepcopy(source), {}))
    monkeypatch.setattr(actions, "_withdrawal_proxy", lambda *a: ([], 3))
    monkeypatch.setattr(actions, "solve_withdrawals", lambda *a, **k: SimpleNamespace(
        status="optimal", search_complete=True, elapsed_seconds=0, selected_train_ids=("B001",), saving_units=1000))
    monkeypatch.setattr(actions, "expand_operational_action", lambda source, *a, **k: (copy.deepcopy(source), {"kind": "WITHDRAW"}))

    def regulate(source, *args, **kwargs):
        target = kwargs.get("target_headway_seconds")
        calls.append(target)
        return copy.deepcopy(source), {"status": "movement_checked", "target_headway_seconds": target}

    monkeypatch.setattr(actions, "regulate_plan", regulate)
    return data, calls


def test_same_headway_grid_is_checked_with_and_without_withdrawal(monkeypatch):
    data, calls = candidate_fixture(monkeypatch)
    original = copy.deepcopy(data)
    rows, audit = actions.generate_strategy_candidates(data, 0, load_policy(), {}, minimum_remaining=1,
        include_recovery=False, headway_targets_seconds=[240, 180, 240])
    assert data == original
    assert calls == [None, 180, 240, None, 180, 240]
    assert len({row["candidate_id"] for row in rows}) == len(rows) == 8
    assert audit["headway_targets_seconds"] == [180, 240]
    for kind in ("REGULATE_ONLY", "COMBINED"):
        assert sorted(row["action"]["target_headway_seconds"] for row in rows
            if row["strategy"] == kind and "target_headway_seconds" in row["action"]) == [180, 240]


def test_unspecified_grid_preserves_existing_four_candidate_families(monkeypatch):
    data, calls = candidate_fixture(monkeypatch)
    rows, audit = actions.generate_strategy_candidates(data, 0, load_policy(), {}, minimum_remaining=1,
        include_recovery=False)
    assert calls == [None, None]
    assert [row["candidate_id"] for row in rows] == ["KEEP", "REGULATE_ONLY", "WITHDRAW_ONLY:B001", "COMBINED:B001"]
    assert "headway_targets_seconds" not in audit


@pytest.mark.parametrize("target", [112, 481, True, 150.5])
def test_invalid_headway_targets_are_rejected_before_any_movement(monkeypatch, target):
    def forbidden(*a, **k):
        raise AssertionError("invalid target must fail before compiling movement")
    monkeypatch.setattr(actions, "compile_operational_timing", forbidden)
    with pytest.raises(ValueError, match="headway"):
        actions.generate_strategy_candidates({}, 0, load_policy(), {}, minimum_remaining=1,
            headway_targets_seconds=[target])
