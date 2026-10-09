from time import monotonic

import pytest

import metro45_joint_controller as control


def test_producer_receives_frozen_fleet_and_common_deadline(monkeypatch):
    calls = []
    changed = {'plan': 'verified'}

    def producer(data, second, ids, **options):
        calls.append((data, second, ids, options))
        return changed, {'movement_passed': True, 'status': 'verified'}

    monkeypatch.setattr(control, 'propose_joint_dispatch', producer)
    selected = ['B005', 'Y025']
    builder = control.make_joint_builder(selected, solver_seconds=12)
    selected.append('B006')
    deadline = monotonic() + 60
    result, audit = builder({'source': 'current'}, 6300, 22200,
                            deadline=deadline, phase_seconds=0)
    assert result is changed and audit['movement_passed']
    assert calls[0][2] == ('B005', 'Y025')
    assert calls[0][3]['deadline'] == deadline
    assert calls[0][3]['service_end_second'] == 22200
    assert calls[0][3]['time_limit_seconds'] == 12
    assert audit['passenger_validation_required'] is True


@pytest.mark.parametrize('status,error', [('physical_rejected', ValueError),
                                        ('infeasible', ValueError),
                                        ('timeout', TimeoutError)])
def test_unverified_plan_cannot_enter_catalogue(monkeypatch, status, error):
    monkeypatch.setattr(control, 'propose_joint_dispatch',
                        lambda *a, **k: (None, {'status': status,
                                               'movement_passed': False,
                                               'reason': 'terminal conflict'}))
    with pytest.raises(error, match='terminal conflict'):
        control.make_joint_builder(['B005'])({}, 6300, 22200,
                                            deadline=monotonic()+60)


def test_expired_budget_never_starts_solver(monkeypatch):
    monkeypatch.setattr(control, 'propose_joint_dispatch',
                        lambda *a, **k: pytest.fail('expired solve started'))
    with pytest.raises(TimeoutError):
        control.make_joint_builder(['B005'])({}, 6300, 22200,
                                            deadline=monotonic()-1)


def test_clock_stability_is_explicit_and_reaches_producer(monkeypatch):
    options = {}

    def producer(*args, **kwargs):
        options.update(kwargs)
        return {}, {'movement_passed': True}

    monkeypatch.setattr(control, 'propose_joint_dispatch', producer)
    control.make_joint_builder(['B005'], clock_stability=True)(
        {}, 6300, 22200, deadline=monotonic()+60)
    assert options['clock_stability'] is True


def test_future_empty_exit_permission_is_explicit(monkeypatch):
    options = {}

    def producer(*args, **kwargs):
        options.update(kwargs)
        return {}, {'movement_passed': True}

    monkeypatch.setattr(control, 'propose_joint_dispatch', producer)
    control.make_joint_builder(['B005'], allow_future_exit_retiming=True)(
        {}, 6300, 22200, deadline=monotonic()+60)
    assert options['allow_future_exit_retiming'] is True


@pytest.mark.parametrize('fleet', [[], ['B005', 'B005']])
def test_empty_or_duplicate_fleet_is_rejected(fleet):
    with pytest.raises(ValueError):
        control.make_joint_builder(fleet)
