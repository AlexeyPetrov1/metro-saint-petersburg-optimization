import copy

import pandas as pd
import pytest

from forecasting import validate_bundle
from metro48_forecasts import profiled_median, complete_tail, score_rows, calibrate_reserve


def history():
    rows = []
    for day in range(1, 5):
        for hour in range(9, 16):
            for quarter, count in enumerate([10, 20, 30, 40]):
                start = pd.Timestamp(f'2026-09-{day:02}T{hour:02}:00:00+03:00') + pd.Timedelta(minutes=15*quarter)
                rows.append(dict(station_id='L1_01', interval_start=start,
                    interval_end=start+pd.Timedelta(minutes=15), entry_count=count,
                    is_complete=True, source_id='uploaded-spb-line1-15min'))
    return pd.DataFrame(rows)


def test_median_uses_past_profile_and_refreshes_at_quarter():
    frame = history()
    at = '2026-09-05T09:15:00+03:00'
    bundle = profiled_median(frame, ['L1_01'], at)
    assert [r['predicted_count'] for r in bundle['rows']] == [20, 30, 40, 10, 20, 30, 40, 10]
    assert bundle['as_of'] == at and bundle['refresh_seconds'] == 900
    validate_bundle(bundle, ['L1_01'], at)
    future = frame.copy()
    next_day = frame[frame.interval_start.dt.day.eq(4)].copy()
    next_day.interval_start += pd.Timedelta(days=1)
    next_day.interval_end += pd.Timedelta(days=1)
    next_day.entry_count = 100000
    future = pd.concat([future, next_day], ignore_index=True)
    assert profiled_median(future, ['L1_01'], at)['rows'] == bundle['rows']


def test_tail_preserves_model_counts_and_labels_historical_extension():
    frame = history(); at = '2026-09-05T09:15:00+03:00'
    raw = profiled_median(frame, ['L1_01'], at)
    raw['rows'][0]['predicted_count'] = 999
    before = copy.deepcopy(raw)
    extended = complete_tail(raw, frame, ['L1_01'], at, '2026-09-05T14:15:00+03:00')
    assert raw == before
    assert len(extended['rows']) == 20 and extended['rows'][0]['predicted_count'] == 999
    assert extended['rows'][8]['predicted_count'] == 20
    assert extended['source_versions']['historical_tail']['future_actual_used'] is False
    validate_bundle(extended, ['L1_01'], at)


def test_explicit_shared_profile_changes_shape_and_preserves_hour_total():
    bundle = profiled_median(history(), ['L1_01'], '2026-09-05T09:00:00+03:00',
        profile_provider=lambda station,hour: [.4,.3,.2,.1])
    assert [r['predicted_count'] for r in bundle['rows'][:4]] == [40,30,20,10]


def test_scoring_rejects_synthetic_truth_and_future_issued_predictions():
    frame = history()
    row = dict(station_id='L1_01', interval_start=frame.iloc[0].interval_start.isoformat(),
               predicted_count=9, issued_at=frame.iloc[0].interval_start.isoformat())
    scores = score_rows([row], frame)
    assert scores['wape_percent'] == 10 and scores['underpredicted_entries'] == 1
    synthetic = frame.copy(); synthetic.source_id = 'uploaded-hourly-uniform-warmup'
    with pytest.raises(ValueError, match='real quarter'):
        score_rows([row], synthetic)
    row['issued_at'] = frame.iloc[0].interval_end.isoformat()
    with pytest.raises(ValueError, match='issued'):
        score_rows([row], frame)


def test_calibration_uses_disjoint_completed_days_and_finite_sample_rank():
    rows = pd.DataFrame(dict(issued_at=['2026-09-02T09:00:00+03:00']*3,
        interval_start=['2026-09-02T09:00:00+03:00']*3,
        interval_end=['2026-09-02T09:15:00+03:00']*3,
        actual_count=[10, 12, 14], predicted_count=[10, 10, 10]))
    result = calibrate_reserve(rows, '2026-09-16T06:00:00+03:00', coverage=.5)
    assert result['factor'] == 1.2
    assert result['evaluation_actual_used'] is False
    with pytest.raises(ValueError, match='before evaluation'):
        calibrate_reserve(rows, '2026-09-02T09:00:00+03:00', coverage=.5)
