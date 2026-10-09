import pandas as pd
import pytest

from forecasting import validate_bundle
from metro_calendar import ProductionCalendar
from metro_regime_forecast import regime_hourly_median


class Registry:
    def state_at(self, station_id, interval_start, as_of, **kwargs):
        at = pd.Timestamp(interval_start)
        closed = at.month == 7 and at.minute >= 15
        return dict(entrance_state='closed' if closed else 'open', movement_state='normal',
                    event_ids=['repair'] if closed else [], event_kinds=[], conflicts=[], unknown_timing=[])


def history():
    rows = []
    for date, values in [('2026-07-01', [100, 0, 0, 0]),
                         ('2026-09-02', [10, 20, 30, 40]),
                         ('2026-09-09', [20, 40, 60, 80]),
                         ('2026-09-16', [999, 999, 999, 999])]:
        for quarter, count in enumerate(values):
            start = pd.Timestamp(f'{date}T08:{quarter*15:02}:00+03:00')
            rows.append(dict(station_id='L1_05', interval_start=start,
                interval_end=start+pd.Timedelta(minutes=15), entry_count=count,
                is_complete=True, source_id='fixture'))
    return pd.DataFrame(rows)


def test_post_repair_forecast_excludes_closed_regime_and_preserves_quarter_shape():
    bundle = regime_hourly_median(history(), ['L1_05'], '2026-09-16T08:00:00+03:00',
                                 event_registry=Registry(), horizon_hours=1, lookback_days=100)
    assert [row['predicted_count'] for row in bundle['rows']] == [15, 30, 45, 60]
    assert bundle['rows'][0]['historical_hours'] == 2
    assert validate_bundle(bundle, ['L1_05'], bundle['as_of'])['status'] == 'passed'
    assert bundle['source_versions']['event_knowledge_mode'] == 'strict'


def test_future_observations_and_unavailable_quality_do_not_change_prediction():
    frame = history()
    frame['quality_status'] = 'ok'
    frame['quality_available_at'] = frame.interval_end
    mask = frame.interval_start.dt.day.eq(9)
    frame.loc[mask, 'quality_status'] = 'source_mismatch'
    frame.loc[mask, 'quality_available_at'] = pd.Timestamp('2026-09-17T00:00:00+03:00')
    first = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                                event_registry=Registry(), horizon_hours=1)
    assert sum(row['predicted_count'] for row in first['rows']) == 150
    frame.loc[mask, 'quality_available_at'] = pd.Timestamp('2026-09-10T00:00:00+03:00')
    second = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                                 event_registry=Registry(), horizon_hours=1)
    assert sum(row['predicted_count'] for row in second['rows']) == 100
    assert frame.loc[mask, 'entry_count'].sum() == 200


def test_missing_matching_regime_is_explicit_and_retrospective_mode_rejected():
    with pytest.raises(ValueError, match='matching regime'):
        regime_hourly_median(history(), ['L1_05'], '2026-08-01T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1, lookback_days=100)
    with pytest.raises(ValueError, match='knowledge'):
        regime_hourly_median(history(), ['L1_05'], '2026-09-16T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1, knowledge_mode='retrospective')


def test_history_digest_changes_when_counts_change_even_if_median_does_not():
    frame = history()
    a = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1)
    frame.loc[frame.interval_start.dt.day.eq(2), 'entry_count'] += 1
    b = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1)
    assert a['source_versions']['history_digest'] != b['source_versions']['history_digest']


def test_profile_changes_with_identical_hourly_totals_change_digest():
    frame = history()
    a = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1)
    selected = frame.index[frame.interval_start.dt.day.eq(2)]
    frame.loc[selected[0], 'entry_count'] += 10
    frame.loc[selected[1], 'entry_count'] -= 10
    b = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                            event_registry=Registry(), horizon_hours=1)
    assert a['source_versions']['history_digest'] != b['source_versions']['history_digest']


def test_calendar_baseline_never_uses_holiday_zeros_for_ordinary_weekday():
    frame = history()
    frame.loc[frame.interval_start.dt.day.eq(9), 'entry_count'] = 0
    calendar = ProductionCalendar(pd.DataFrame(dict(date=['2026-09-02', '2026-09-09', '2026-09-16'],
        is_workday=[True, False, True], holiday_name=['', 'Holiday', ''])))
    bundle = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
        event_registry=Registry(), horizon_hours=1, calendar=calendar)
    assert sum(row['predicted_count'] for row in bundle['rows']) == 100
    assert bundle['rows'][0]['historical_hours'] == 1
    assert bundle['source_versions']['calendar_targets'][0]['day_kind'] == 'workday'


def test_calendar_fallback_uses_matching_day_kind_on_another_weekday():
    frame = history()
    keep = frame[frame.interval_start.dt.day.eq(2)].copy()
    keep['interval_start'] += pd.Timedelta(days=1)
    keep['interval_end'] += pd.Timedelta(days=1)
    calendar = ProductionCalendar(pd.DataFrame(dict(date=['2026-09-03', '2026-09-16'], is_workday=[True, True])))
    bundle = regime_hourly_median(keep, ['L1_05'], '2026-09-16T08:00:00+03:00',
        event_registry=Registry(), horizon_hours=1, calendar=calendar)
    assert sum(row['predicted_count'] for row in bundle['rows']) == 100
    assert bundle['source_versions']['calendar_targets'][0]['selection'] == 'same_calendar_kind'


def test_missing_matching_calendar_kind_is_explicit():
    calendar = ProductionCalendar(pd.DataFrame(dict(date=['2026-09-02', '2026-09-09', '2026-09-16'],
                                                   is_workday=[False, False, True])))
    with pytest.raises(ValueError, match='calendar'):
        regime_hourly_median(history(), ['L1_05'], '2026-09-16T08:00:00+03:00',
            event_registry=Registry(), horizon_hours=1, calendar=calendar)


def test_receipt_timestamp_is_included_in_history_digest_and_latest_availability():
    frame = history()
    frame['received_at'] = frame.interval_end
    first = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                                event_registry=Registry(), horizon_hours=1)
    frame.loc[frame.interval_start.dt.day.eq(9), 'received_at'] += pd.Timedelta(days=1)
    second = regime_hourly_median(frame, ['L1_05'], '2026-09-16T08:00:00+03:00',
                                 event_registry=Registry(), horizon_hours=1)
    assert first['source_versions']['history_digest'] != second['source_versions']['history_digest']
    assert second['source_versions']['latest_observation_available_at'] == '2026-09-10T09:00:00+03:00'
