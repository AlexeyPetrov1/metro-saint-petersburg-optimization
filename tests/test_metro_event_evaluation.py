import pandas as pd
import pytest

from metro_event_evaluation import evaluate_event_alerts, evaluate_forecasts


def frames():
    starts = pd.date_range('2026-09-09 08:00', periods=3, freq='15min', tz='Europe/Moscow')
    actuals = pd.DataFrame(dict(station_id=['L1_16'] * 3, interval_start=starts,
                               interval_end=starts + pd.Timedelta(minutes=15), entry_count=[100, 0, 200],
                               quality_flag=['ok', 'closure', 'anomaly'], quality_status=['ok', 'ok', 'source_mismatch'],
                               evaluation_slice=['ordinary', 'repair', 'disruption']))
    predictions = actuals[['station_id', 'interval_start', 'interval_end']].copy()
    predictions['predicted_count'] = [90, 50, 100]
    predictions['q10'], predictions['q90'] = [80, 0, 80], [110, 60, 120]
    predictions['as_of'] = starts[0]
    predictions['available_at'] = starts[0]
    predictions['features_available_at'] = starts[0]
    return predictions, actuals


def test_evaluation_keeps_flagged_actuals_and_overlapping_quality_slice():
    predictions, actuals = frames()
    result = evaluate_forecasts(predictions, actuals)
    assert result['all']['n'] == 3
    assert result['all']['wape'] == pytest.approx(160 / 300)
    assert result['all']['q10_q90_coverage'] == pytest.approx(2 / 3)
    assert result['slices']['repair']['n'] == 1
    assert result['slices']['repair']['wape'] is None
    assert result['slices']['data_quality']['n'] == 1
    assert result['slices']['disruption']['n'] == 1
    assert result['slices']['holiday']['n'] == 0
    assert result['slices']['holiday']['q10_q90_coverage'] is None
    assert result['event_detection']['precision'] is None


def test_partial_actuals_remain_visible_but_are_ineligible_for_forecast_error():
    predictions, actuals = frames()
    actuals['is_complete'] = [True, False, True]
    result = evaluate_forecasts(predictions, actuals)
    assert result['all']['rows'] == 3 and result['all']['n'] == 2
    assert result['all']['wape'] == pytest.approx(110 / 300)
    assert result['all']['excluded_incomplete'] == 1
    assert result['rows'][1]['actual_count'] == 0
    assert not result['rows'][1]['evaluation_eligible']
    assert result['slices']['repair']['n'] == 0


@pytest.mark.parametrize('value', [-1, float('inf')])
def test_invalid_predicted_counts_are_rejected(value):
    predictions, actuals = frames()
    predictions['predicted_count'] = predictions.predicted_count.astype(float)
    predictions.loc[0, 'predicted_count'] = value
    with pytest.raises(ValueError, match='predicted_count'):
        evaluate_forecasts(predictions, actuals)


def test_predictions_cannot_use_later_features():
    predictions, actuals = frames()
    predictions['features_available_at'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='as_of'):
        evaluate_forecasts(predictions, actuals)


def test_later_forecast_publication_is_valid_before_target_and_decision_time():
    predictions, actuals = frames()
    predictions = predictions.iloc[1:].copy()
    predictions['available_at'] += pd.Timedelta(minutes=5)
    assert evaluate_forecasts(predictions, actuals)['all']['n'] == 2
    with pytest.raises(ValueError, match='decision'):
        evaluate_forecasts(predictions, actuals, decision_time=predictions.as_of.iloc[0])
    predictions['available_at'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='target'):
        evaluate_forecasts(predictions, actuals)


def test_recovery_requires_exact_confirmed_disruption_end():
    class Registry:
        records = [dict(event_id='e', version=1, confidence='confirmed', kind='train_disruption',
                        line_id='L1', scope={'level': 'station', 'station_ids': ['L1_16']},
                        timing={'observed': {'end': '2026-09-09T08:20:00+03:00'}})]

        def label_at(self, station_id, interval_start, vestibule_id=None):
            return dict(event_kinds=['planned_entrance_closure'] if interval_start.minute == 0 else [])

    predictions, actuals = frames()
    actuals = actuals.drop(columns='evaluation_slice')
    result = evaluate_forecasts(predictions, actuals, Registry())
    assert result['slices']['repair']['n'] == 1
    assert result['slices']['recovery']['n'] == 1
    Registry.records[0]['timing']['observed']['end'] = None
    assert evaluate_forecasts(predictions, actuals, Registry())['slices']['recovery']['n'] == 0


def test_actuals_are_scored_after_forecast_as_of_and_are_not_features():
    predictions, actuals = frames()
    actuals['available_at'] = actuals.interval_end + pd.Timedelta(days=1)
    result = evaluate_forecasts(predictions, actuals)
    assert result['all']['n'] == 3
    assert result['rows'][2]['actual_count'] == 200


def test_registry_event_groups_do_not_duplicate_rows():
    class Registry:
        def label_at(self, station_id, interval_start, vestibule_id=None):
            return dict(event_kinds=['repair', 'repair'], event_ids=['a', 'b'])
    predictions, actuals = frames()
    actuals = actuals.drop(columns='evaluation_slice')
    result = evaluate_forecasts(predictions, actuals, Registry())
    assert result['all']['n'] == result['slices']['repair']['n'] == 3


def test_event_metrics_require_explicit_exhaustive_truth_and_exposure():
    ts = pd.Timestamp('2026-09-09 08:00', tz='Europe/Moscow')
    alerts = pd.DataFrame([dict(alert_id='a', station_id='L1_16', alerted_at=ts + pd.Timedelta(minutes=5))])
    events = pd.DataFrame([dict(event_id='e', station_id='L1_16', start=ts, end=ts + pd.Timedelta(minutes=20))])
    assert evaluate_event_alerts(alerts, events, None)['recall'] is None
    exposure = dict(truth_exhaustive=True, station_ids=['L1_16'], start=ts, end=ts + pd.Timedelta(hours=1))
    result = evaluate_event_alerts(alerts, events, exposure)
    assert result['precision'] == result['recall'] == 1
    assert result['mean_detection_delay_seconds'] == 300
    exposure['truth_exhaustive'] = False
    assert evaluate_event_alerts(alerts, events, exposure)['false_alerts_per_station_hour'] is None


def test_multiple_alerts_detect_one_event_once():
    ts = pd.Timestamp('2026-09-09 08:00', tz='Europe/Moscow')
    alerts = pd.DataFrame([dict(alert_id=k, station_id='L1_16', alerted_at=ts + pd.Timedelta(minutes=m))
                           for k, m in [('a', 5), ('b', 10), ('c', 40)]])
    events = pd.DataFrame([dict(event_id='e', station_id='L1_16', start=ts, end=ts + pd.Timedelta(minutes=20))])
    exposure = dict(truth_exhaustive=True, station_ids=['L1_16'], start=ts, end=ts + pd.Timedelta(hours=1))
    result = evaluate_event_alerts(alerts, events, exposure)
    assert result['detected_events'] == 1
    assert result['false_alerts'] == 1
    assert result['mean_detection_delay_seconds'] == 300


def test_partial_exposure_event_is_not_a_false_alert_or_eligible_delay():
    ts = pd.Timestamp('2026-09-09 08:00', tz='Europe/Moscow')
    alerts = pd.DataFrame([dict(alert_id='a', station_id='L1_16', alerted_at=ts + pd.Timedelta(minutes=5))])
    events = pd.DataFrame([dict(event_id='e', station_id='L1_16', start=ts - pd.Timedelta(minutes=10),
                                end=ts + pd.Timedelta(minutes=20))])
    exposure = dict(truth_exhaustive=True, station_ids=['L1_16'], start=ts, end=ts + pd.Timedelta(hours=1))
    result = evaluate_event_alerts(alerts, events, exposure)
    assert result['precision'] == 1
    assert result['false_alerts'] == result['eligible_events'] == 0
    assert result['recall'] is result['mean_detection_delay_seconds'] is None


def test_empty_truth_can_measure_false_alerts_with_exhaustive_exposure():
    ts = pd.Timestamp('2026-09-09 08:00', tz='Europe/Moscow')
    alerts = pd.DataFrame([dict(alert_id='a', station_id='L1_16', alerted_at=ts + pd.Timedelta(minutes=5))])
    events = pd.DataFrame(columns=['event_id', 'station_id', 'start', 'end'])
    exposure = dict(truth_exhaustive=True, station_ids=['L1_16'], start=ts, end=ts + pd.Timedelta(hours=1))
    result = evaluate_event_alerts(alerts, events, exposure)
    assert result['false_alerts_per_station_hour'] == 1
    assert result['precision'] == 0 and result['recall'] is None


def test_duplicate_truth_and_invalid_quantiles_fail():
    predictions, actuals = frames()
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_forecasts(predictions, pd.concat([actuals, actuals.iloc[:1]]))
    predictions['q90'] = predictions.q10 - 1
    with pytest.raises(ValueError, match='quantile'):
        evaluate_forecasts(predictions, actuals)
