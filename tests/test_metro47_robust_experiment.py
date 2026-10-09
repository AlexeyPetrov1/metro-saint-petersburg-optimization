import pytest

from metro47_robust_experiment import calculate_robust_catalog, run


def test_solver_and_simulator_use_the_same_guarded_forecast():
    raw = {'rows': [{'predicted_count': 100}]}
    guarded = {'rows': [{'predicted_count': 125}]}
    calls = []
    data = {'period_start': '2026-09-30T08:15:00+03:00'}
    args = (data, 'runner', 'state', 'profiles', 'roster', None, None, None, 3600, 22200, None, 99, None)

    def margin(bundle, second):
        assert bundle is raw and second == '2026-09-30T09:15:00+03:00'
        return guarded

    def solve(data, bundle, state, second, end, **kwargs):
        calls.append(('solve', bundle))
        return {'selected_train_ids': ['B001']}

    def catalog(*values, **kwargs):
        calls.append(('catalog', values[5], kwargs.get('proposal')))
        return {'accepted': True}

    def refine(solver, evaluator, **kwargs):
        return evaluator(solver()['selected_train_ids'])

    result = calculate_robust_catalog(*args, bundle_override=raw,
        margin=margin, solve=solve, catalog=catalog, refine=refine, builder='builder')
    assert result == {'accepted': True}
    assert calls == [('solve', guarded), ('catalog', guarded, {'selected_train_ids': ['B001']})]
    assert raw['rows'][0]['predicted_count'] == 100


def test_previous_run_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path)


def test_oracle_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / 'new', arm='ORACLE')
    assert not (tmp_path / 'new').exists()
