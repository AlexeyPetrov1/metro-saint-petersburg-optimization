from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from metro_events import EventRegistry
from metro_ml import FrozenMetroForecaster


def event(**changes):
    value = dict(event_id='schedule', version_id='schedule:v1', version=1,
        line_id='L1', scope={'level': 'vestibule', 'station_ids': ['L1_10'], 'vestibule_ids': ['V12']},
        published_at='2026-05-08T10:00:00+03:00', ingested_at='2026-05-08T10:01:00+03:00',
        confidence='confirmed', sources=[{'url': 'https://metro.spb.ru/schedule'}],
        kind='planned_event_entrance_closure', entrance_state='closed', movement_state='normal',
        timing={'planned': {'start': '2026-05-09T11:00:00+03:00', 'end': '2026-05-09T13:00:00+03:00'}})
    value.update(changes)
    return value


def forecaster(records):
    model = FrozenMetroForecaster.__new__(FrozenMetroForecaster)
    model.event_registry = EventRegistry.from_records(records)
    model.knowledge_mode = 'strict'
    model.inverse = {'vosstaniya': 'L1_10'}
    model.data = SimpleNamespace(vestibules=pd.DataFrame([
        dict(station_id='vosstaniya', vestibule_id='vosstaniya_1', pos=0),
        dict(station_id='vosstaniya', vestibule_id='vosstaniya_2', pos=1)]))
    return model


def test_known_vestibule_schedule_only_changes_future_hours_and_matching_entrance():
    model = forecaster([event()])
    ts = pd.date_range('2026-05-09T10:00:00+03:00', periods=3, freq='h').tz_convert('UTC')
    closed, flagged = np.zeros((3, 2), bool), np.zeros((3, 2), bool)
    ids = model._apply_known_schedules(ts, closed, flagged, pd.Timestamp('2026-05-09T11:00:00+03:00'))
    assert ids == ['schedule']
    assert closed.tolist() == [[False, False], [True, False], [True, False]]
    assert flagged.tolist() == closed.tolist()


def test_late_publication_or_ingestion_cannot_change_known_future_schedule():
    model = forecaster([event(ingested_at='2026-05-09T14:00:00+03:00')])
    ts = pd.date_range('2026-05-09T11:00:00+03:00', periods=2, freq='h').tz_convert('UTC')
    closed, flagged = np.zeros((2, 2), bool), np.zeros((2, 2), bool)
    model._apply_known_schedules(ts, closed, flagged, pd.Timestamp('2026-05-09T11:00:00+03:00'))
    assert not closed.any() and not flagged.any()
    model.knowledge_mode = 'publication_replay'
    model._apply_known_schedules(ts, closed, flagged, pd.Timestamp('2026-05-09T11:00:00+03:00'))
    assert closed[:, 0].all() and not closed[:, 1].any()


def test_partial_hour_restriction_does_not_claim_full_hour_closure():
    model = forecaster([event(timing={'planned': {'start': '2026-05-09T11:10:00+03:00',
                                                 'end': '2026-05-09T11:15:00+03:00'}})])
    ts = pd.date_range('2026-05-09T11:00:00+03:00', periods=1, freq='h').tz_convert('UTC')
    closed, flagged = np.zeros((1, 2), bool), np.zeros((1, 2), bool)
    model._apply_known_schedules(ts, closed, flagged, pd.Timestamp('2026-05-09T11:00:00+03:00'))
    assert not closed.any() and flagged[0, 0] and not flagged[0, 1]


def test_explicit_night_service_overrides_static_night_closure():
    model = forecaster([event(line_id=None, scope={'level': 'network'},
        entrance_state='open', kind='planned_night_service',
        timing={'planned': {'start': '2026-05-30T01:00:00+03:00', 'end': '2026-05-30T05:00:00+03:00'}})])
    ts = pd.date_range('2026-05-30T01:00:00+03:00', periods=2, freq='h').tz_convert('UTC')
    closed, flagged = np.ones((2, 2), bool), np.ones((2, 2), bool)
    model._apply_known_schedules(ts, closed, flagged, pd.Timestamp('2026-05-30T01:00:00+03:00'))
    assert not closed.any() and not flagged.any()


def test_event_knowledge_mode_is_validated_before_loading_weights():
    with pytest.raises(ValueError, match='knowledge_mode'):
        FrozenMetroForecaster(knowledge_mode='retrospective')


@dataclass
class Panel:
    thresholds: object = None


def cached_forecaster():
    model = forecaster([])
    model.event_registry = None
    model._cache = {}
    model._known_event_ids = []
    model.mapping = {'L1_10': 'vosstaniya'}
    model.model_ids = {'vosstaniya'}
    model.hashes = {}
    model.history_source = 'simulator_observations'
    model.allow_training_period = True
    model.use_retrospective_history_flags = True
    model.meta = dict(thresholds=[], train_start='2026-01-01', train_end='2026-05-01')
    model.calls = 0
    cutoff = pd.Timestamp('2026-05-09T11:00:00+03:00').tz_convert('UTC')
    data = SimpleNamespace(forecast=pd.DataFrame({'ts_utc': [cutoff], 'precipitation': [0]}))

    def data_at(frame, when):
        model.calls += 1
        return data, None

    model._data_at = data_at
    model.BT = SimpleNamespace(build_panel=lambda data, final: Panel(),
        make_rows=lambda panel, for_export: pd.DataFrame({'t': [cutoff-pd.Timedelta(hours=1)], 'tau': [cutoff]}),
        station_frame=lambda rows, prediction, panel: pd.DataFrame({'station_id': ['vosstaniya'],
            'tau': [cutoff], 'q50': [100], 'n': [2], 'n_open': [2]}))
    model.F = SimpleNamespace(add_features=lambda rows, panel: rows)
    model.model = SimpleNamespace(predict=lambda rows: SimpleNamespace(q50=np.array([100])))
    model.station_quantiles = SimpleNamespace(predict=lambda rows: pd.DataFrame(index=rows.index))
    frame = pd.DataFrame([dict(station_id='L1_10',
        interval_start=pd.Timestamp('2026-05-09T10:00:00+03:00'),
        interval_end=pd.Timestamp('2026-05-09T10:15:00+03:00'),
        available_at=pd.Timestamp('2026-05-09T10:20:00+03:00'),
        entry_count=10, is_complete=True, source_id='fixture')])
    return model, frame


def test_changed_observation_availability_invalidates_cache_without_changing_count_digest():
    model, frame = cached_forecaster()
    first = model._compute(frame, '2026-05-09T11:00:00+03:00')
    changed = frame.copy()
    changed['available_at'] += pd.Timedelta(minutes=5)
    second = model._compute(changed, '2026-05-09T11:00:00+03:00')
    assert model.calls == 2
    assert first['observation_digest'] == second['observation_digest']
    assert second['latest_observation_available_at'] == pd.Timestamp('2026-05-09T10:25:00+03:00')
    model._compute(changed, '2026-05-09T11:00:00+03:00')
    assert model.calls == 2


def test_feature_availability_covers_decision_time_and_reports_observation_receipt_separately():
    model, frame = cached_forecaster()
    bundle = model.forecast_bundle(frame, ['L1_10'], '2026-05-09T11:00:00+03:00', horizon_hours=1)
    assert bundle['features_available_at'] == bundle['as_of']
    assert pd.Timestamp(bundle['source_versions']['latest_observation_available_at']) == frame.available_at.max()


@pytest.mark.parametrize('received_at', ['2026-05-09T11:01:00+03:00', None])
def test_observed_history_excludes_late_or_unknown_receipt_despite_past_publication(received_at):
    from metro_ml import observed_history
    _, frame = cached_forecaster()
    frame['received_at'] = pd.Series(pd.to_datetime([received_at], utc=True), index=frame.index)
    assert observed_history(frame, pd.Timestamp('2026-05-09T11:00:00+03:00')).empty


def test_observed_history_rejects_timezone_naive_receipt():
    from metro_ml import observed_history
    _, frame = cached_forecaster()
    frame['received_at'] = pd.Timestamp('2026-05-09T10:20:00')
    with pytest.raises(ValueError, match='timezone'):
        observed_history(frame, pd.Timestamp('2026-05-09T11:00:00+03:00'))


@pytest.mark.parametrize('legacy', [True, False])
def test_unknown_provenance_historical_flags_are_only_used_in_explicit_legacy_policy(legacy):
    model = forecaster([])
    model.use_retrospective_history_flags = legacy
    ts = pd.date_range('2026-05-09T09:00:00+03:00', periods=3, freq='h').tz_convert('UTC')
    historical_closed = np.array([[True, False], [False, True], [True, True]])
    historical_flagged = np.ones((3, 2), bool)
    model.data.grid = SimpleNamespace(ts=ts, closed=historical_closed, flagged=historical_flagged)
    closed, flagged = np.zeros((3, 2), bool), np.zeros((3, 2), bool)
    model._restore_history_flags(ts, closed, flagged, pd.Timestamp('2026-05-09T11:00:00+03:00'))
    if legacy:
        assert np.array_equal(closed[:2], historical_closed[:2])
        assert flagged[:2].all()
    else:
        assert not closed.any() and not flagged.any()
    assert not closed[2].any() and not flagged[2].any()
    assert historical_flagged.all()


def test_anomaly_aware_bundle_declares_disabled_retrospective_history_flags():
    model, frame = cached_forecaster()
    model.use_retrospective_history_flags = False
    bundle = model.forecast_bundle(frame, ['L1_10'], '2026-05-09T11:00:00+03:00', horizon_hours=1)
    assert bundle['source_versions']['history_flag_policy'] == 'public_schedule_and_available_quality_only'


def test_retrospective_history_flag_policy_must_be_explicit_boolean():
    with pytest.raises(ValueError, match='retrospective'):
        FrozenMetroForecaster(use_retrospective_history_flags='false')
