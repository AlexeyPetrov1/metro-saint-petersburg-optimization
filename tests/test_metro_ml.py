from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from metro_ml import (FrozenMetroForecaster, causal_quarter_profile,
                      safe_extract, station_mapping)
from forecasting import validate_bundle


@pytest.mark.parametrize('name', ['../outside.txt', '/outside.txt', 'C:/outside.txt', '..\\outside.txt'])
def test_archive_path_traversal_is_rejected_before_any_write(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('good.txt', 'ok')
        z.writestr(name, 'bad')
    with pytest.raises(ValueError, match='unsafe'):
        safe_extract(archive, tmp_path / 'out')
    assert not (tmp_path / 'out' / 'good.txt').exists()


def test_extract_preserves_existing_files_and_detects_changed_bytes(tmp_path):
    archive = tmp_path / 'ok.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('nested/a.txt', 'first')
    safe_extract(archive, tmp_path / 'out')
    safe_extract(archive, tmp_path / 'out')
    (tmp_path / 'out/nested/a.txt').write_text('different')
    with pytest.raises(ValueError, match='existing'):
        safe_extract(archive, tmp_path / 'out')


def test_station_mapping_uses_names_and_reversed_line_order():
    mapping = station_mapping()
    assert mapping['L1_01'] == 'prospekt_veteranov'
    assert mapping['L1_07'] == 'tekhnologichesky_institut'
    assert mapping['L1_19'] == 'devyatkino'
    assert len(set(mapping.values())) == 19


def profile_history():
    rows = []
    for date, values in [('2026-09-23', [10, 20, 30, 40]),
                         ('2026-09-30', [99999, 0, 0, 0])]:
        for q, value in enumerate(values):
            start = pd.Timestamp(f'{date}T08:{q*15:02}:00+03:00')
            rows.append(dict(station_id='L1_01', interval_start=start,
                interval_end=start+pd.Timedelta(minutes=15), entry_count=value,
                is_complete=True, source_id='fixture'))
    return pd.DataFrame(rows)


def test_quarter_profile_uses_completed_historical_hours_and_preserves_shape():
    p = causal_quarter_profile(profile_history(), 'L1_01',
        pd.Timestamp('2026-09-30T08:00:00+03:00'), pd.Timestamp('2026-09-30T08:00:00+03:00'))
    np.testing.assert_allclose(p, [.1, .2, .3, .4])


def test_unavailable_published_history_cannot_enter_profile():
    data = profile_history()
    data['available_at'] = data.interval_end + pd.Timedelta(days=20)
    np.testing.assert_allclose(causal_quarter_profile(data, 'L1_01',
        pd.Timestamp('2026-09-30T08:00:00+03:00'), pd.Timestamp('2026-09-30T08:00:00+03:00')), [.25]*4)


def test_evaluation_reports_total_weighted_error_and_missing_truth_separately():
    from scripts.evaluate_metro_ml import metrics
    result = metrics(pd.DataFrame(dict(actual_count=[100,900,np.nan],predicted_count=[200,800,999999])))
    assert result == dict(n=2,actual_total=1000.0,mae=100.0,rmse=100.0,wape=.2)
    empty = metrics(pd.DataFrame(dict(actual_count=[np.nan],predicted_count=[1])))
    assert empty['n']==0 and empty['wape'] is None


def test_profile_control_preserves_median_hour_and_ignores_future_shape(tmp_path):
    from scripts.evaluate_metro_ml import add_profile_control
    profile_history().to_parquet(tmp_path/'simulator_history.parquet',index=False)
    predictions = []
    for q,actual in enumerate([40,30,20,10]):
        predictions.append(dict(date='2026-09-30',origin='2026-09-30T08:00:00+03:00',mode='median',
            station_id='L1_01',target_start=f'2026-09-30T08:{q*15:02}:00+03:00',horizon_hours=1,
            predicted_count=25,actual_count=actual,source='median',temporal_holdout=True))
    pd.DataFrame(predictions).to_csv(tmp_path/'quarter_predictions.csv',index=False)
    pd.DataFrame([dict(date='2026-09-30',origin='2026-09-30T08:00:00+03:00',mode='simulator_observations',
        station_id='L1_01',target_hour='2026-09-30T08:00:00+03:00',horizon_hours=1,predicted_count=100,
        actual_original_hourly=100,actual_quarter_sum=100,temporal_holdout=True)]).to_csv(tmp_path/'hourly_quantiles.csv',index=False)
    (tmp_path/'manifest.json').write_text('{}')
    add_profile_control(tmp_path)
    result = pd.read_csv(tmp_path/'median_profile_control_predictions.csv')
    assert result.predicted_count.tolist()==[10,20,30,40]
    assert result.predicted_count.sum()==100
    with pytest.raises(FileExistsError,match='preserved'):
        add_profile_control(tmp_path)


def artifacts_available():
    return Path('data/processed/metro_ml_inputs/lgbm_final/meta.json').exists()


@pytest.mark.skipif(not artifacts_available(), reason='user model archive not prepared')
def test_actual_frozen_bundle_is_causal_and_hourly_refresh_has_no_third_hour():
    forecaster = FrozenMetroForecaster()
    history = forecaster.load_simulator_frame()
    when = '2026-09-30T08:45:00+03:00'
    ids = list(station_mapping())
    bundle = forecaster.forecast_bundle(history, ids, when)
    assert validate_bundle(bundle, ids, when)['available_horizon_seconds'] == 4500
    assert bundle['as_of'] == '2026-09-30T08:00:00+03:00'
    assert len(bundle['rows']) == 19*8
    assert bundle['source_versions']['train_end'] == '2026-09-29'
    assert bundle['source_versions']['temporal_holdout'] is True
    assert bundle['source_versions']['upstream_embargo_holdout'] is False
    assert bundle['source_versions']['fallback_station_ids'] == ['L1_07']
    changed = history.copy()
    changed.loc[changed.interval_end > pd.Timestamp(bundle['as_of']), 'entry_count'] = 999999
    cutoff = pd.Timestamp(bundle['as_of']).tz_convert('UTC')
    future = forecaster.data.grid.ts+pd.Timedelta(hours=1) > cutoff
    forecaster.data.grid.entries[future] = 999999
    forecaster.data.grid.closed[future] = True
    forecaster.data.grid.flagged[future] = True
    forecaster.quarters.loc[forecaster.quarters.ts_utc+pd.Timedelta(minutes=15)>cutoff,'entries'] = 999999
    forecaster._cache.clear()
    again = forecaster.forecast_bundle(changed, ids, when)
    assert bundle['rows'] == again['rows']
    assert bundle['source_versions']['observation_digest'] == again['source_versions']['observation_digest']


@pytest.mark.skipif(not artifacts_available(), reason='user model archive not prepared')
def test_future_late_vestibule_schedules_are_public_and_missing_weather_stays_nan():
    forecaster = FrozenMetroForecaster()
    frame = forecaster.load_simulator_frame()
    rows = forecaster.feature_rows(frame,'2026-09-30T22:00:00+03:00')
    assert not rows.vestibule_id.eq('leninsky_prospekt_2').any()
    assert not rows[(rows.tau.dt.tz_convert('Europe/Moscow').dt.hour==23)].vestibule_id.eq('ploshchad_lenina_2').any()
    assert rows.fc_precip_tau.isna().all()
    assert rows.fc_precip_3h_tau.isna().all()


@pytest.mark.skipif(not artifacts_available(), reason='user model archive not prepared')
def test_actual_frozen_weights_reject_unknown_line_and_show_observation_cutoff():
    forecaster = FrozenMetroForecaster()
    frame = forecaster.load_simulator_frame()
    with pytest.raises(ValueError, match='station'):
        forecaster.forecast_bundle(frame, ['L2_01'], '2026-09-30T08:00:00+03:00')
    features = forecaster.feature_rows(frame, '2026-09-30T08:00:00+03:00')
    assert features.t.max() == pd.Timestamp('2026-09-30T04:00:00Z')
    changed = frame.copy()
    m = ((changed.station_id == 'L1_01') &
         (changed.interval_start >= pd.Timestamp('2026-09-30T07:00:00+03:00')) &
         (changed.interval_end <= pd.Timestamp('2026-09-30T08:00:00+03:00')))
    changed.loc[m, 'entry_count'] += 100
    shifted = forecaster.feature_rows(changed, '2026-09-30T08:00:00+03:00')
    original = features[features.station_id == 'prospekt_veteranov'].drop_duplicates('vestibule_id').y_t.sum()
    updated = shifted[shifted.station_id == 'prospekt_veteranov'].drop_duplicates('vestibule_id').y_t.sum()
    assert updated == pytest.approx(original + 400)


@pytest.mark.skipif(not artifacts_available(), reason='user model archive not prepared')
def test_loader_uses_hourly_history_only_where_observed_quarters_are_absent():
    forecaster = FrozenMetroForecaster()
    frame = forecaster.load_simulator_frame()
    assert frame.interval_start.min() == pd.Timestamp('2026-01-01T03:00:00+03:00')
    february = frame[frame.interval_start.between(
        pd.Timestamp('2026-02-01T03:00:00+03:00'), pd.Timestamp('2026-03-01T02:45:00+03:00'))]
    assert february.source_id.eq('uploaded-spb-line1-15min').all()
    assert not frame.duplicated(['station_id','interval_start']).any()
