from copy import deepcopy
from time import monotonic

import pytest

from metro48_temporary_solver import solve_temporary_options, solve_temporary_withdrawals


def case():
    rows = [dict(train_id=t, cancel_service_ids=[t+'0', t+'1'], cancel_entry_ids=[], cancel_exit_ids=[],
                 resume_trip_id=t+'2', resume_second=1050+150*i, return_by_second=1400)
            for i, t in enumerate(('A', 'B', 'C'))]
    clocks = [dict(track_id='P', direction='up', terminal_offset_seconds=0, second=0, trip_id='past')]
    clocks += [dict(track_id='P', direction='up', terminal_offset_seconds=0,
                   second=150*(i+1)+450*j, trip_id=t+str(j))
               for i, t in enumerate(('A', 'B', 'C')) for j in range(3)]
    return rows, clocks, dict(A=920, B=1840, C=740)


def solve(**kwargs):
    rows, clocks, values = case()
    return solve_temporary_options(rows, clocks, option_values=values, rate_slots=kwargs.pop('rate_slots', []),
        as_of=150, service_end_second=1500, **kwargs)


def test_solver_keeps_planned_returns_and_selects_highest_net_TO1_saving():
    result = solve()
    assert result['selected_train_ids'] == ['A', 'B']
    assert result['estimated_to1_resource_saving_rub'] == 2760
    assert result['resume_trip_ids'] == dict(A='A2', B='B2')
    assert result['waiting_in_objective'] is False and result['cash_saving_rub'] is None
    assert result['exact_movement_and_passenger_checks_required'] is True


def test_capacity_and_exact_feedback_use_same_fixed_clock_constraints():
    rates = [dict(start_second=150, end_second=1050, up_rate=6, down_rate=0)]
    assert solve(rate_slots=rates)['selected_train_ids'] == []
    assert solve(no_good_cuts=[['A', 'B']])['selected_train_ids'] == ['B', 'C']
    assert solve(max_withdrawals=1)['selected_train_ids'] == ['B']


def test_return_is_never_in_the_cancelled_subset_and_requires_real_clock():
    rows, clocks, values = case()
    rows[0]['cancel_service_ids'].append('A2')
    with pytest.raises(ValueError, match='return'):
        solve_temporary_options(rows, clocks, option_values=values, rate_slots=[], as_of=150, service_end_second=1500)
    rows, clocks, values = case()
    rows[0]['resume_second'] = 1051
    with pytest.raises(ValueError, match='return'):
        solve_temporary_options(rows, clocks, option_values=values, rate_slots=[], as_of=150, service_end_second=1500)


def test_solver_preserves_inputs_and_rejects_infeasible_existing_capacity():
    rows, clocks, values = case()
    old = deepcopy((rows, clocks, values))
    result = solve_temporary_options(rows, clocks, option_values=values,
        rate_slots=[dict(start_second=150, end_second=1500, up_rate=20, down_rate=0)],
        as_of=150, service_end_second=1500)
    assert result['status'] == 'no_feasible_candidate' and result['passenger_admissible'] is False
    assert (rows, clocks, values) == old


def test_expired_budget_does_not_select_a_train():
    result = solve(deadline=monotonic()-1)
    assert result['status'] == 'evaluation_timeout' and result['selected_train_ids'] == []


def wrapper_case(monkeypatch):
    import metro48_temporary_solver as module
    rows, clocks, values = case()
    for r, kind, km in zip(rows, ('Yubileyny', 'Baltiets', 'Baltiets'), (100, 50, 10)):
        r.update(train_type=kind, saved_km=str(km), to1_resource_saving_rub=str(values[r['train_id']]))
    data = dict(period_start='2026-09-30T06:00:00+03:00',
        stations=[dict(station_id=s, kind='line') for s in ('X', 'Y')],
        tracks=[dict(track_id='P_up', from_station='X', to_station='Y')],
        trains=[dict(train_id=r['train_id'], train_type=r['train_type']) for r in rows],
        plans=[dict(trips=[dict(trip_id=r['trip_id'], route=['P_up']) for r in clocks if r['trip_id'] != 'past'])],
        previous_departures=[dict(track_id='P_up', departure_second=0)],
        leg_timing=[dict(train_id=r['trip_id'][0], trip_id=r['trip_id'], leg_index=0,
                        departure_second=r['second']) for r in clocks if r['trip_id'] != 'past'])
    monkeypatch.setattr(module, 'temporary_options', lambda *a, **k: rows)
    monkeypatch.setattr(module, 'validate_bundle', lambda *a, **k: {})
    monkeypatch.setattr(module, 'station_offsets', lambda *a: {'up': [0, 0], 'down': [0, 0]})
    monkeypatch.setattr(module, 'forecast_section_rates', lambda *a, **k: [])
    return data


def test_opt_in_solver_ranks_full_maintenance_and_restores_true_TO1_metadata(monkeypatch):
    data = wrapper_case(monkeypatch)
    source = deepcopy(data)
    legacy = solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
                                        return_by_second=1400, max_withdrawals=1)
    result = solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
        return_by_second=1400, max_withdrawals=1,
        maintenance_rates={'Yubileyny': 104, 'Baltiets': 113})
    assert legacy['selected_train_ids'] == ['B']
    assert result['selected_train_ids'] == ['A']
    assert result['estimated_maintenance_resource_saving_rub'] == 10400
    assert result['estimated_to1_resource_saving_rub'] == 920
    assert result['known_maintenance_saving_by_train'] == dict(A=10400, B=5650, C=1130)
    assert 'TO1+TO2+TR1+TR2+TR3' in result['objective']
    assert result['resume_trip_ids'] == dict(A='A2')
    assert data == source


@pytest.mark.parametrize('rates', [{'Baltiets': 113}, {'Yubileyny': -1, 'Baltiets': 113},
    {'Yubileyny': 104, 'Baltiets': float('inf')}])
def test_invalid_opt_in_rates_rejected_before_solver(monkeypatch, rates):
    data = wrapper_case(monkeypatch)
    with pytest.raises(ValueError, match='maintenance'):
        solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
                                   return_by_second=1400, maintenance_rates=rates)


def test_peak_waiting_proxy_opens_proposal_without_changing_actual_capacity_or_demand(monkeypatch):
    import metro48_temporary_solver as module
    data = wrapper_case(monkeypatch)
    data['period_start'] = '2026-09-30T08:00:00+03:00'
    rates = [dict(start_second=150, end_second=1050, up_rate=4.1, down_rate=0)]
    monkeypatch.setattr(module, 'forecast_section_rates', lambda *a, **k: rates)
    bundle = {'immutable_forecast': [1000]}
    old = deepcopy((data, bundle, rates))
    strict = solve_temporary_withdrawals(data, bundle, {'second': 150}, 150, 1500, return_by_second=1400)
    captures = []
    solve_options = module.solve_temporary_options
    def capture(*a, **kw):
        captures.append(deepcopy(kw))
        return solve_options(*a, **kw)
    monkeypatch.setattr(module, 'solve_temporary_options', capture)
    result = solve_temporary_withdrawals(data, bundle, {'second': 150}, 150, 1500,
                                       return_by_second=1400, peak_missed_share_limit=.1)
    assert strict['selected_train_ids'] == []
    assert result['selected_train_ids'] == ['B']
    assert captures[0]['capacity'] == 1200
    assert captures[0]['rate_slots'][0]['up_rate'] == pytest.approx(3.69)
    assert (data, bundle, rates) == old
    assert result['peak_waiting_proxy']['calibrated'] is False
    assert result['peak_waiting_proxy']['exact_forecast_trials_unmodified'] is True
    assert result['peak_waiting_proxy']['actual_capacity_changed'] is False
    assert result['exact_movement_and_passenger_checks_required'] is True
    assert result['cash_saving_rub'] is None


@pytest.mark.parametrize('origin, expected', [
    ('2026-09-30T07:57:30+03:00', 3.6), # slot begins exactly08:00.
    ('2026-09-30T09:57:30+03:00', 4), # slot begins exactly10:00.
    ('2026-09-30T16:57:30+03:00', 3.6),
    ('2026-09-30T18:57:30+03:00', 4),
    ('2026-10-03T08:00:00+03:00', 4), # Saturday.
    ('2026-09-30T04:57:30+00:00', 3.6), # Moscow08:00.
])
def test_waiting_proxy_uses_moscow_weekday_peak_boundaries(monkeypatch, origin, expected):
    import metro48_temporary_solver as module
    data = wrapper_case(monkeypatch)
    data['period_start'] = origin
    rates = [dict(start_second=150, end_second=1050, up_rate=4, down_rate=4)]
    monkeypatch.setattr(module, 'forecast_section_rates', lambda *a, **k: rates)
    captured = []
    solve_options = module.solve_temporary_options
    def capture(*a, **kw):
        captured.extend(kw['rate_slots'])
        return solve_options(*a, **kw)
    monkeypatch.setattr(module, 'solve_temporary_options', capture)
    solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
                               return_by_second=1400, peak_missed_share_limit=.1)
    assert captured[0]['up_rate'] == pytest.approx(expected)
    assert captured[0]['down_rate'] == pytest.approx(expected)


@pytest.mark.parametrize('limit', [-.1, 1, 1.1, True, '0.1', float('nan'), float('inf')])
def test_waiting_proxy_rejects_invalid_fraction(monkeypatch, limit):
    data = wrapper_case(monkeypatch)
    with pytest.raises(ValueError, match='missed'):
        solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
                                   return_by_second=1400, peak_missed_share_limit=limit)


def test_zero_waiting_proxy_preserves_strict_proposal(monkeypatch):
    import metro48_temporary_solver as module
    data = wrapper_case(monkeypatch)
    data['period_start'] = '2026-09-30T08:00:00+03:00'
    monkeypatch.setattr(module, 'forecast_section_rates', lambda *a, **k: [
        dict(start_second=150, end_second=1050, up_rate=4.1, down_rate=0)])
    strict = solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500, return_by_second=1400)
    result = solve_temporary_withdrawals(data, {}, {'second': 150}, 150, 1500,
        return_by_second=1400, peak_missed_share_limit=0)
    assert result['selected_train_ids'] == strict['selected_train_ids']
    assert result['peak_waiting_proxy']['discounted_slot_count'] == 0
