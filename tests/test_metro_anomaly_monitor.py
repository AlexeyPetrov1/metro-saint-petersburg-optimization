import pandas as pd
import pytest

from metro_anomaly_monitor import calibrate_alert_budget, monitor_anomalies


NOW = pd.Timestamp('2026-09-17T08:00:00+03:00')


class Registry:
    def __init__(self, entrance='open', available_at=None):
        self.entrance, self.available_at = entrance, available_at

    def state_at(self, station_id, interval_start, as_of, **kwargs):
        known = self.available_at is None or pd.Timestamp(as_of) >= self.available_at
        entrance = self.entrance if known else 'open'
        return dict(entrance_state=entrance, movement_state='normal',
                    demand_censored=entrance != 'open', event_ids=['repair'] if entrance != 'open' else [],
                    candidate_event_ids=[], conflicts=[], unknown_timing=False)


def history(stations=('L1_01',), current=200, count=1000):
    rows = []
    for sid in stations:
        for weeks in (4, 3, 2, 1, 0):
            start = NOW-pd.Timedelta(weeks=weeks, minutes=15)
            rows.append(dict(station_id=sid, interval_start=start,
                interval_end=start+pd.Timedelta(minutes=15), available_at=start+pd.Timedelta(minutes=15),
                entry_count=current if weeks == 0 else count, is_complete=True,
                quality_status='ok', quality_available_at=start+pd.Timedelta(minutes=15)))
    return pd.DataFrame(rows)


def test_unexplained_flow_drop_is_only_a_candidate_with_available_reference():
    result = monitor_anomalies(history(), Registry(), NOW)
    row = result['alert_rows'][0]
    assert row['classification'] == 'flow_anomaly_candidate'
    assert row['observed_count'] == 200
    assert row['reference_count'] == 1000
    assert row['residual_count'] == -800
    assert row['history_count'] == 4
    assert row['causal_claim'] is None
    assert row['movement_state'] == 'unknown'


def test_neighbour_consensus_does_not_bridge_missing_station():
    result = monitor_anomalies(history(('L1_01', 'L1_02', 'L1_04')), Registry(), NOW)
    rows = {row['station_id']: row for row in result['alert_rows']}
    assert rows['L1_01']['adjacent_station_ids'] == ['L1_01', 'L1_02']
    assert rows['L1_02']['adjacent_station_ids'] == ['L1_01', 'L1_02']
    assert rows['L1_04']['adjacent_station_ids'] == []
    assert rows['L1_01']['causal_claim'] is None


def test_repair_censorship_and_quality_mismatch_have_separate_classifications():
    repair = monitor_anomalies(history(), Registry('closed'), NOW)
    assert repair['alert_rows'][0]['classification'] == 'censored_admissions'
    assert repair['alert_rows'][0]['event_ids'] == ['repair']
    data = history()
    data.loc[data.interval_end.eq(NOW), 'quality_status'] = 'source_mismatch'
    quality = monitor_anomalies(data, Registry(), NOW)
    assert quality['alert_rows'][0]['classification'] == 'data_quality'
    assert quality['alert_rows'][0]['causal_claim'] is None


def test_future_data_late_quality_and_future_publication_do_not_leak():
    data = history()
    data.loc[data.interval_end.eq(NOW), 'quality_status'] = 'source_mismatch'
    data.loc[data.interval_end.eq(NOW), 'quality_available_at'] = NOW+pd.Timedelta(hours=1)
    future = data.iloc[[-1]].copy()
    future['interval_start'] = NOW
    future['interval_end'] = NOW+pd.Timedelta(minutes=15)
    future['available_at'] = future.interval_end
    future['entry_count'] = 999999
    result = monitor_anomalies(pd.concat([data, future]), Registry('closed', NOW+pd.Timedelta(hours=1)), NOW)
    assert result['alert_rows'][0]['classification'] == 'flow_anomaly_candidate'
    assert result['alert_rows'][0]['observed_count'] == 200
    assert result['alert_rows'][0]['event_ids'] == []


def test_known_historical_source_mismatch_is_excluded_without_filling_baseline():
    data = history()
    old = data.interval_end.lt(NOW)
    data.loc[old, 'quality_status'] = 'source_mismatch'
    result = monitor_anomalies(data, Registry(), NOW)
    assert result['alert_rows'] == []
    assert result['context']['residual_rows'][0]['reference_count'] is None
    assert result['context']['residual_rows'][0]['history_count'] == 0


def test_absolute_gate_suppresses_tiny_zero_flow_fluctuation():
    result = monitor_anomalies(history(current=0, count=50), Registry(), NOW)
    assert result['alert_rows'] == []


def test_measured_movement_disruption_is_visible_even_with_normal_entries():
    movement = pd.DataFrame([dict(line_id='L1', station_id='L1_01', observed_at=NOW-pd.Timedelta(minutes=1),
                                 available_at=NOW, movement_state='disrupted')])
    result = monitor_anomalies(history(current=1000), Registry(), NOW, movement=movement)
    row = result['alert_rows'][0]
    assert row['classification'] == 'measured_movement_disruption'
    assert row['causal_claim'] is None
    assert 'measured_motion' in row['signal_sources']


def test_optional_vestibule_residual_preserves_station_mapping():
    vestibules = history(current=100)
    vestibules['vestibule_id'] = 'V01'
    result = monitor_anomalies(history(current=1000), Registry(), NOW, vestibules=vestibules)
    assert result['alert_rows'][0]['vestibule_id'] == 'V01'
    assert result['alert_rows'][0]['station_id'] == 'L1_01'


def test_unknown_entries_and_stale_history_are_not_flow_alerts():
    data = history()
    data.loc[data.interval_end.eq(NOW), 'entry_count'] = None
    missing = monitor_anomalies(data, Registry(), NOW)
    assert missing['alert_rows'] == []
    stale = monitor_anomalies(history(), Registry(), NOW+pd.Timedelta(hours=2))
    assert stale['alert_rows'] == []
    assert 'entry_observation_stale' in stale['context']['unknowns']


def test_received_at_blocks_late_current_and_reference_values():
    data = history()
    data['received_at'] = data.interval_end
    data.loc[data.interval_end.eq(NOW), 'received_at'] = NOW+pd.Timedelta(hours=1)
    result = monitor_anomalies(data, Registry(), NOW)
    assert result['alert_rows'] == []
    assert 'entry_observation_stale' in result['context']['unknowns']


def test_uniform_hourly_warmup_cannot_supply_real_quarter_baseline():
    data = history()
    data['source_id'] = 'uploaded-hourly-uniform-warmup'
    data.loc[data.interval_end.eq(NOW), 'source_id'] = 'uploaded-spb-line1-15min'
    result = monitor_anomalies(data, Registry(), NOW)
    assert result['alert_rows'] == []
    assert result['context']['residual_rows'][0]['history_count'] == 0


def test_motion_signal_survives_missing_fresh_entry_observation():
    movement = pd.DataFrame([dict(line_id='L1', station_id='L1_01', observed_at=NOW-pd.Timedelta(minutes=1),
                                 available_at=NOW, movement_state='disrupted')])
    data = history().loc[lambda x: x.interval_end.lt(NOW)]
    result = monitor_anomalies(data, Registry(), NOW, movement=movement)
    assert result['alert_rows'][0]['classification'] == 'measured_movement_disruption'
    assert result['alert_rows'][0]['observed_count'] is None


def test_motion_and_adjacent_flow_signals_are_combined():
    movement = pd.DataFrame([dict(line_id='L1', station_id='L1_01', observed_at=NOW-pd.Timedelta(minutes=1),
                                 available_at=NOW, movement_state='disrupted')])
    result = monitor_anomalies(history(('L1_01', 'L1_02')), Registry(), NOW, movement=movement)
    row = next(row for row in result['alert_rows'] if row['station_id'] == 'L1_01')
    assert row['adjacent_station_ids'] == ['L1_01', 'L1_02']
    assert {'turnstile_residual', 'measured_motion', 'adjacent_stations'} <= set(row['signal_sources'])


def test_calibration_enforces_per_station_day_budget_without_incident_claims():
    at = pd.Timestamp('2026-09-01T00:00:00+03:00')
    residuals = pd.DataFrame([dict(station_id='L1_01', interval_start=at+pd.Timedelta(hours=i),
                                   residual_score=score, residual_count=500)
                              for i, score in enumerate([2., 4., 6.])])
    exposure = dict(start=at, end=at+pd.Timedelta(days=1), station_ids=['L1_01'], ordinary_reference=True)
    result = calibrate_alert_budget(residuals, exposure, max_alerts_per_station_day=1)
    assert result['threshold'] == 6
    assert result['maximum_alerts_per_station_day'] <= 1
    assert result['incident_precision'] is None
    zero = calibrate_alert_budget(residuals, exposure, max_alerts_per_station_day=0)
    assert zero['threshold'] > 6
    assert zero['alerts'] == 0


def test_calibration_rejects_unspecified_reference_and_conflicting_duplicates():
    at = pd.Timestamp('2026-09-01T00:00:00+03:00')
    data = pd.DataFrame([dict(station_id='L1_01', interval_start=at, residual_score=2, residual_count=500)])
    with pytest.raises(ValueError, match='ordinary'):
        calibrate_alert_budget(data, dict(start=at, end=at+pd.Timedelta(days=1), station_ids=['L1_01']))
    with pytest.raises(ValueError, match='duplicate'):
        calibrate_alert_budget(pd.concat([data, data]),
            dict(start=at, end=at+pd.Timedelta(days=1), station_ids=['L1_01'], ordinary_reference=True))
