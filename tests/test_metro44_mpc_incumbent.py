import copy

import metro44_mpc as mpc


def slot():
    return dict(start_second=0, end_second=900, up_rate=1, down_rate=1,
                up_entries=100, down_entries=100, ready_capacity=53)


def staged_solver(monkeypatch, stages):
    real_solver = mpc.cp_model.CpSolver
    clock = [0.0]

    class Solver:
        def __init__(self):
            self.delegate = real_solver()
            self.parameters = self.delegate.parameters
            self.index = 0
            self.readable = False

        def solve(self, model):
            status, elapsed = stages[self.index]
            self.index += 1
            self.readable = status in (mpc.cp_model.OPTIMAL, mpc.cp_model.FEASIBLE)
            if self.readable:
                assert self.delegate.solve(model) in (mpc.cp_model.OPTIMAL, mpc.cp_model.FEASIBLE)
            clock[0] += elapsed
            return status

        def value(self, expression):
            if not self.readable:
                raise AssertionError('UNKNOWN has no usable assignment')
            return self.delegate.value(expression)

        def status_name(self, status):
            return self.delegate.status_name(status)

    monkeypatch.setattr(mpc.cp_model, 'CpSolver', Solver)
    monkeypatch.setattr(mpc, 'monotonic', lambda: clock[0])


def test_default_keeps_discarding_feasible_solution_after_subsolver_budget(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.FEASIBLE, 31)])
    result = mpc.solve_joint_profile([slot()], time_limit_seconds=30, deadline=470)
    assert result['slots'] == []
    assert result['status'] == 'evaluation_timeout'
    assert result['solver_status'] == 'FEASIBLE'


def test_opt_in_preserves_feasible_incumbent_within_outer_decision_budget(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.FEASIBLE, 31)])
    rows = [slot()]
    original = copy.deepcopy(rows)
    result = mpc.solve_joint_profile(rows, headway_choices=[150, 180],
        mean_wait_budget_seconds=90, time_limit_seconds=30, deadline=470, allow_incumbent=True)
    assert result['slots'] and result['status'] == 'search_incomplete'
    assert result['estimated_mean_wait_seconds'] <= 90
    assert result['incumbent_recovered'] and not result['optimality_proven']
    assert not result['search_complete']
    assert result['incumbent_solver_status'] == 'FEASIBLE'
    assert result['optimized_stages'] == 0
    assert not result['passenger_admissible'] and not result['release_authorized']
    assert rows == original


def test_later_unknown_keeps_previous_optimal_stage_assignment(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.OPTIMAL, 1), (mpc.cp_model.UNKNOWN, 30)])
    result = mpc.solve_joint_profile([slot()], headway_choices=[150, 180],
        mean_wait_budget_seconds=90, time_limit_seconds=30, deadline=470, allow_incumbent=True)
    assert result['status'] == 'search_incomplete' and result['slots']
    assert result['solver_status'] == 'UNKNOWN'
    assert result['incumbent_solver_status'] == 'OPTIMAL'
    assert result['optimized_stages'] == 1 and not result['optimality_proven']


def test_saved_work_optimum_survives_an_unknown_waiting_stage(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.OPTIMAL, 1), (mpc.cp_model.OPTIMAL, 1),
                               (mpc.cp_model.UNKNOWN, 30)])
    result = mpc.solve_joint_profile([slot()], headway_choices=[150, 180],
        mean_wait_budget_seconds=90, time_limit_seconds=30, deadline=470, allow_incumbent=True)
    assert result['slots'][0]['up_headway_seconds'] == result['slots'][0]['down_headway_seconds'] == 180
    assert result['optimized_stages'] == 2 and not result['optimality_proven']


def test_default_still_discards_an_earlier_incumbent_after_unknown(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.OPTIMAL, 1), (mpc.cp_model.UNKNOWN, 1)])
    result = mpc.solve_joint_profile([slot()], time_limit_seconds=30, deadline=470)
    assert result['slots'] == [] and result['status'] == 'evaluation_timeout'


def test_outer_deadline_discards_incumbent_even_with_opt_in(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.FEASIBLE, 471)])
    result = mpc.solve_joint_profile([slot()], time_limit_seconds=30, deadline=470,
                                    allow_incumbent=True)
    assert result['slots'] == [] and result['status'] == 'evaluation_timeout'


def test_unknown_without_an_incumbent_returns_no_candidate(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.UNKNOWN, 31)])
    result = mpc.solve_joint_profile([slot()], time_limit_seconds=30, deadline=470,
                                    allow_incumbent=True)
    assert result['slots'] == [] and result['status'] == 'evaluation_timeout'


def test_recovered_capacity_shortfall_remains_only_a_diagnostic(monkeypatch):
    staged_solver(monkeypatch, [(mpc.cp_model.FEASIBLE, 31)])
    row = slot()
    row['up_rate'] = 20
    result = mpc.solve_joint_profile([row], headway_choices=[150, 180],
        time_limit_seconds=30, deadline=470, allow_incumbent=True)
    assert result['slots'] and result['status'] == 'capacity_shortfall'
    assert not result['forecast_capacity_passed']
    assert not result['passenger_admissible'] and not result['release_authorized']
    assert not result['optimality_proven']


def test_finished_opt_in_search_keeps_proven_optimality():
    result = mpc.solve_joint_profile([slot()], headway_choices=[150, 180],
        mean_wait_budget_seconds=90, time_limit_seconds=3, allow_incumbent=True)
    assert result['status'] == 'optimal' and result['search_complete']
    assert result['optimality_proven'] and result['optimized_stages'] == 3
    assert not result['incumbent_recovered']
