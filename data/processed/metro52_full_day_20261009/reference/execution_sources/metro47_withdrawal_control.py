"""Refine a joint train assignment using exact forecast feasibility feedback."""

from time import monotonic


def refine_withdrawals(solve, evaluate, *, deadline, cancelled=None, max_attempts=6):
    cuts, count_limit, history, answer = [], None, [], None
    for _ in range(max_attempts):
        if monotonic() >= deadline or cancelled is not None and cancelled.is_set():
            raise TimeoutError('withdrawal control deadline')
        solved = solve(no_good_cuts=list(cuts), max_withdrawals=count_limit)
        selected = list(solved.get('selected_train_ids', []))
        answer = evaluate(selected)
        decision = answer['decision']
        rows = [r for r in decision['candidate_table'] if r['candidate_id'] == 'COST_SOLVER']
        row = rows[0] if rows else {}
        history.append(dict(solver=solved, selected_train_ids=selected,
            exact_status=row.get('status'), service_passed=row.get('service_passed'),
            policy_passed=row.get('policy_passed'), reason=row.get('reason')))
        if not selected or decision['selection'].get('selected_candidate_id') == 'COST_SOLVER':
            break
        if row.get('status') == 'evaluated':
            if row.get('service_passed') and row.get('policy_passed'):
                break
            count_limit = len(selected)//2
        else:
            cuts.append(tuple(sorted(selected)))
    answer['decision']['withdrawal_search'] = dict(attempt_count=len(history), attempts=history,
        fixed_headway_grid_used=False, future_actual_demand_used=False,
        feedback='exact forecast trials; physical no-good assignments; conservative cardinality cuts',
        operational_global_optimum_proven=False, waiting_objective_used=False)
    return answer
