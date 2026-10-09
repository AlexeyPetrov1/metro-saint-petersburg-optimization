import numpy as np
import pandas as pd

from scripts.metro_anomaly_audit import causal_reference, corrected_hourly, adjacent_windows


def test_reference_uses_only_prior_eligible_observations():
    y = pd.Series([100., 110., 90., 100., 0., 9999.])
    use = pd.Series([True, True, True, True, False, True])
    b, _, n = causal_reference(y, use)
    assert np.isnan(b[3])
    assert b[4] == 100 and b[5] == 100 and n[5] == 4


def test_reference_never_fills_missing_history():
    b, _, n = causal_reference(pd.Series([10., np.nan, 20.]), pd.Series([True]*3))
    assert np.isnan(b).all() and n[-1] == 1


def test_service_labels_shift_only_midnight_hours():
    t = pd.to_datetime(['2026-06-27 01:00', '2026-06-27 23:00']).tz_localize('Europe/Moscow')
    d = pd.DataFrame({'interval_start': t, 'interval_end': t + pd.Timedelta(hours=1)})
    result = corrected_hourly(d)
    assert result.interval_start.iloc[0].day == 28
    assert result.interval_start.iloc[1].day == 27
    assert d.interval_start.iloc[0].day == 27


def test_adjacency_does_not_bridge_absent_station():
    assert adjacent_windows(['L1_05', 'L1_06', 'L1_08', 'L1_09', 'L1_10']) == [('L1_08', 'L1_09', 'L1_10')]
