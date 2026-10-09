import numpy as np
import pandas as pd
import pytest

from metro48_gru_evaluation import benchmark_contract, matched_median_count, reserve_coverage


def test_temporal_dates_are_predeclared_and_disjoint():
    contract = benchmark_contract()
    assert contract['calibration_dates'] == ['2026-09-02', '2026-09-07', '2026-09-10']
    assert contract['evaluation_dates'] == ['2026-09-16', '2026-09-21', '2026-09-28']
    assert contract['development_dates'] == ['2026-09-30']
    assert set(contract['calibration_dates']).isdisjoint(contract['evaluation_dates'])
    assert contract['old_ml_evaluation_dates'] == ['2026-09-30']
    assert not contract['statistical_holdout_pristine']


def test_matched_median_uses_only_closed_hours_and_shared_profile():
    times = pd.date_range('2026-09-01 09:00', periods=4, freq='7D', tz='Europe/Moscow')
    history = pd.DataFrame(dict(station_id=['L1_01']*4, hour_start=times, total=[100, 200, 300, 1000000]))
    cutoff = pd.Timestamp('2026-09-22 09:15', tz='Europe/Moscow')
    profile = np.array([.1, .2, .3, .4])
    assert matched_median_count(history, 'L1_01', cutoff, profile) == 40


def test_median_missing_history_is_not_zero_demand():
    empty = pd.DataFrame(columns=['station_id', 'hour_start', 'total'])
    empty['hour_start'] = pd.to_datetime(empty.hour_start, utc=True)
    with pytest.raises(ValueError, match='history'):
        matched_median_count(empty, 'L1_01', pd.Timestamp('2026-09-22 09:15', tz='Europe/Moscow'), [.25]*4)


def test_reserve_coverage_uses_declared_additive_formula_at_zero_prediction():
    rows = pd.DataFrame(dict(predicted_count=[0, 100, 100], actual_count=[2, 110, 120]))
    result = reserve_coverage(rows, dict(normalized_residual=.1, factor=1.1))
    assert result['covered_rows'] == 1
    assert result['coverage_percent'] == pytest.approx(100/3)
    assert result['zero_prediction_rows'] == 1
    assert not result['operational_capacity_guarantee']
