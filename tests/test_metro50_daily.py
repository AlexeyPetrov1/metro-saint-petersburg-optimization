import copy

from metro50_daily import control_times, return_boundary, admission_forecast, closed_tail


def test_day_contains_both_peaks_and_all_68_decisions():
    times = control_times(64800)
    assert len(times) == 68
    assert times[0] == 3600 and times[-1] == 63900
    assert 7200 in times and 39600 in times and 45900 in times


def test_return_uses_available_horizon_and_preserves_closing_service():
    assert return_boundary(7200, 64800) == 21000
    assert return_boundary(63900, 64800) == 64800


def test_no_arrivals_forecast_after_real_admission_close():
    bundle = {'rows': [dict(interval_start='2026-10-01T00:00:00+03:00', predicted_count=20),
                       dict(interval_start='2026-09-30T23:45:00+03:00', predicted_count=10)],
              'source_versions': {}}
    original = copy.deepcopy(bundle)
    answer = admission_forecast(bundle, '2026-10-01T00:00:00+03:00')
    assert [r['predicted_count'] for r in answer['rows']] == [0, 10]
    assert bundle == original


def test_closing_tail_is_zero_and_keeps_all_station_quarters():
    bundle = {'rows': [dict(station_id=s, interval_start='2026-09-30T23:45:00+03:00',
        interval_end='2026-10-01T00:00:00+03:00', predicted_count=10, coverage='complete') for s in ('A', 'B')]}
    answer = closed_tail(bundle, '2026-10-01T00:30:00+03:00')
    assert len(answer['rows']) == 6
    assert sum(r['predicted_count'] for r in answer['rows']) == 20
    assert answer['rows'][-1]['interval_end'] == '2026-10-01T00:30:00+03:00'
