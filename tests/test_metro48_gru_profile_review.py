import pandas as pd
import pytest

from metro48_gru_profile_review import reprofile_old_ml


def _rows():
    times = pd.date_range('2026-09-30 09:00', periods=4, freq='15min', tz='Europe/Moscow')
    return pd.DataFrame(dict(model=['OLD_ML']*4, station_id=['L1_01']*4,
        interval_start=times, issued_at=[times[0]]*4, predicted_count=[1, 2, 3, 4], actual_count=[3, 3, 3, 3]))


def test_common_profile_preserves_frozen_hour_total_and_actuals():
    rows = _rows()
    changed = reprofile_old_ml(rows, lambda station, target: [.25]*4)
    assert changed.predicted_count.tolist() == [3, 3, 2, 2]
    assert changed.predicted_count.sum() == rows.predicted_count.sum()
    assert changed.actual_count.equals(rows.actual_count)
    assert changed.issued_at.equals(rows.issued_at)
    assert changed.model.eq('OLD_ML_SHARED_PROFILE').all()


def test_reprofile_refuses_partial_hour_or_different_issue_times():
    rows = _rows()
    with pytest.raises(ValueError):
        reprofile_old_ml(rows.iloc[:3], lambda station, target: [.25]*4)
    rows.loc[1, 'issued_at'] = rows.loc[1, 'interval_start']
    with pytest.raises(ValueError):
        reprofile_old_ml(rows, lambda station, target: [.25]*4)
