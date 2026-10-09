from time import monotonic

import pytest

from metro47_withdrawal_control import refine_withdrawals


def decision(selected, status='evaluated', admissible=True):
    return dict(decision=dict(selection=dict(selected_candidate_id=selected),
        candidate_table=[dict(candidate_id='COST_SOLVER', status=status,
            service_passed=admissible, policy_passed=admissible)]), candidates=[])


def test_forecast_failure_tightens_the_joint_withdrawal_limit():
    calls = []
    def solve(**constraints):
        calls.append(constraints)
        return dict(selected_train_ids=['A', 'B'] if constraints['max_withdrawals'] is None else ['A'])
    results = iter([decision('KEEP', admissible=False), decision('COST_SOLVER')])
    result = refine_withdrawals(solve, lambda ids: next(results), deadline=monotonic()+10)
    assert calls[1]['max_withdrawals'] == 1
    assert result['decision']['selection']['selected_candidate_id'] == 'COST_SOLVER'
    assert result['decision']['withdrawal_search']['attempt_count'] == 2


def test_physical_failure_excludes_only_that_exact_assignment():
    calls = []
    def solve(**constraints):
        calls.append(constraints)
        return dict(selected_train_ids=['A'] if not constraints['no_good_cuts'] else ['B'])
    results = iter([decision('KEEP', status='rejected'), decision('COST_SOLVER')])
    refine_withdrawals(solve, lambda ids: next(results), deadline=monotonic()+10)
    assert calls[1]['no_good_cuts'] == [('A',)]
    assert calls[1]['max_withdrawals'] is None


def test_empty_assignment_is_evaluated_once_as_keep():
    calls = []
    result = refine_withdrawals(lambda **kw: dict(selected_train_ids=[]),
        lambda ids: calls.append(ids) or decision('KEEP'), deadline=monotonic()+10)
    assert calls == [[]]
    assert result['decision']['withdrawal_search']['attempt_count'] == 1


def test_capacity_feedback_reduces_a_large_failed_assignment_quickly():
    calls = []
    def solve(**constraints):
        calls.append(constraints)
        return dict(selected_train_ids=list('ABCDEFGHIJKLMNO') if constraints['max_withdrawals'] is None else [])
    refine_withdrawals(solve, lambda ids: decision('KEEP', admissible=False), deadline=monotonic()+10)
    assert calls[1]['max_withdrawals'] == 7


def test_expired_deadline_does_not_start_a_search():
    with pytest.raises(TimeoutError):
        refine_withdrawals(lambda **kw: None, lambda ids: None, deadline=monotonic()-1)
