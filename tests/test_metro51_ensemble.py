import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from metro51_ensemble import (
    ENSEMBLE_INPUTS, FrozenQuarterEnsemble, persistence_point,
    prepare_ensemble_inputs, slot_hour_parts, stack_predictions,
)


def test_extraction_checks_every_path_before_writing(tmp_path):
    archive = tmp_path / 'models.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('models/a.json', '{}')
        z.writestr('../escape.json', '{}')
    with pytest.raises(ValueError, match='unsafe'):
        prepare_ensemble_inputs(archive, archive, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_persistence_uses_only_last_two_completed_quarters():
    now = pd.Timestamp('2026-09-30T09:15:00+03:00')
    h = pd.DataFrame({'ts': pd.date_range(now-pd.Timedelta(hours=1), periods=8, freq='15min'),
                      'vestibule_id': ['a']*8, 'entries': [1, 1, 20, 40, 999, 999, 999, 999],
                      'norm': [10]*8})
    point = persistence_point(h, ['a'], now, np.array([[100., 200., 300., 400.]]))
    np.testing.assert_allclose(point, np.log((np.array([[100.,200.,300.,400.]])*3+1)/
                                            (np.array([[100.,200.,300.,400.]])+1)))
    h.loc[h.ts >= now, 'entries'] = 10**9
    np.testing.assert_array_equal(persistence_point(h, ['a'], now, np.array([[100.,200.,300.,400.]])), point)


def test_persistence_missing_history_uses_ratio_one_and_clips_extremes():
    now = pd.Timestamp('2026-09-30T09:00:00+03:00')
    h = pd.DataFrame({'ts': [now-pd.Timedelta(minutes=15)], 'vestibule_id': ['a'],
                      'entries': [100000], 'norm': [10.]})
    result = persistence_point(h, ['a','b'], now, np.full((2,4), 10.))
    np.testing.assert_allclose(result[0], np.log(31/11))
    np.testing.assert_array_equal(result[1], np.zeros(4))


def test_shifted_native_slot_is_split_across_hour_boundary_without_duplicate_norm():
    parts = slot_hour_parts('2026-09-30T09:45:00+03:00', np.array([[1.,3.,2.,4.,5.,5.,1.,1.]]), ['a'])
    first = parts[parts.k.eq(1)]
    assert first.hour_ts.nunique() == 2
    assert first.quarter_norm.sum() == 4
    assert first.h_hour.tolist() == [1,2]
    assert first.quarter_in_hour.tolist() == [3,0]


def _params():
    return {'weights': {'names':['b1','b2','b3'],'by':['k'],
        'w':{q:{} for q in ['10','50','90']},
        'w_k':{q:{str(k):[.2,.3,.5] for k in range(1,5)} for q in ['10','50','90']}},
        'conformal': {'cell':{},'k':{str(k):[.1,.2] for k in range(1,5)}}}


def test_stack_matches_weighted_log_mix_and_conformal():
    rows = pd.DataFrame({'k':[1,2], 'band4':['day','day']})
    for base, value in [('b1',1.),('b2',2.),('b3',3.)]:
        for q, delta in [('10',-.2),('50',0),('90',.2)]:
            rows[f'{base}_{q}'] = value+delta
    result = stack_predictions(rows, _params())
    np.testing.assert_allclose(result[['stack_10','stack_50','stack_90']], [[2.,2.3,2.7]]*2)


@pytest.fixture(scope='module')
def model():
    if not (ENSEMBLE_INPUTS/'models/lgbm_holdout/meta.json').exists():
        pytest.skip('new complete ensemble inputs not prepared')
    return FrozenQuarterEnsemble()


@pytest.fixture(scope='module')
def frame(model):
    return model.load_source_frame()


def test_actual_contract_uses_all_native_bases_and_true_cutoffs(model):
    c = model.contract()
    assert c['components'] == ['b1_lightgbm','b2_persistence','b3_gru_seeds012','frozen_meta_stack']
    assert c['b1_train_end'] == '2026-08-24'
    assert c['b3_train_end'] == '2026-07-31'
    assert c['meta_fit_months'] == [5,7]
    assert c['b2_calibration_months'] == [2,5,7]
    assert c['uncovered_station_ids'] == ['L1_07']
    assert c['native_update_seconds'] == 1800


def test_bundle_complete_quarter_refresh_and_future_mutation_invariance(model, frame):
    now = pd.Timestamp('2026-09-30T09:15:00+03:00')
    ids = list(model.mapping)
    a = model.forecast_bundle(frame, ids, now, horizon_steps=8)
    changed = frame.copy()
    changed.loc[changed.interval_end > now, 'entry_count'] = 10**8
    b = model.forecast_bundle(changed, ids, now)
    assert a['rows'] == b['rows']
    assert len(a['rows']) == 19*8
    assert a['refresh_seconds'] == 900
    assert a['source_versions']['shifted_quarter_inference']
    assert not a['source_versions']['future_truth_used']
    assert {row['coverage'] for row in a['rows']} == {'complete'}
    assert all(row['q10'] <= row['q50'] <= row['q90'] for row in a['source_versions']['quantile_rows'])
    q90 = model.forecast_bundle(frame,ids,now,quantile='q90')
    assert [r['predicted_count'] for r in q90['rows']] == [r['q90'] for r in a['source_versions']['quantile_rows']]


def test_available_at_prevents_using_late_observations(model, frame):
    now = pd.Timestamp('2026-09-30T09:30:00+03:00')
    delayed = frame.copy()
    delayed['available_at'] = delayed.interval_end
    mask = delayed.interval_end.eq(now)
    delayed.loc[mask,'available_at'] = now+pd.Timedelta(minutes=15)
    a = model.forecast_bundle(delayed, list(model.mapping), now)
    delayed.loc[mask,'entry_count'] = 10**8
    b = model.forecast_bundle(delayed, list(model.mapping), now)
    assert a['rows'] == b['rows']


def test_recent_quarter_changes_persistence_and_full_stack(model, frame):
    now = pd.Timestamp('2026-09-30T09:15:00+03:00')
    a = model.model_slots(frame, now)
    changed = frame.copy()
    changed.loc[changed.interval_end.eq(now), 'entry_count'] *= 2
    b = model.model_slots(changed, now)
    assert not np.allclose(a['rows'].b2_50, b['rows'].b2_50)
    assert not np.allclose(a['slots'], b['slots'])


def test_training_period_and_unsupported_horizon_are_rejected(model, frame):
    with pytest.raises(ValueError, match='training|embargo'):
        model.forecast_bundle(frame, list(model.mapping), '2026-08-24T09:00:00+03:00')
    with pytest.raises(ValueError, match='horizon'):
        model.forecast_bundle(frame, list(model.mapping), '2026-09-30T09:00:00+03:00', horizon_steps=16)


def test_b3_output_matches_saved_native_networks_on_same_input(model, frame):
    import torch
    now = pd.Timestamp('2026-09-30T09:00:00+03:00')
    prediction = model.model_slots(frame, now)
    window = prediction['window']
    with torch.no_grad():
        x = torch.from_numpy(model.gru.scaler.transform(window))
        codes = torch.from_numpy(prediction['codes'])
        native = np.mean([net(x,codes).numpy().astype(float) for net in model.gru.nets],axis=0)
    np.testing.assert_allclose(prediction['b3_z'], native, rtol=0, atol=0)


def test_source_frame_has_all_red_stations_and_same_day_truth(model, frame):
    assert set(frame.station_id) == set(model.mapping)
    day = frame[frame.interval_start.ge(pd.Timestamp('2026-09-30T06:00:00+03:00')) &
                frame.interval_start.lt(pd.Timestamp('2026-10-01T00:00:00+03:00'))]
    assert day.entry_count.sum() == 758936
    assert not frame.duplicated(['station_id','interval_start']).any()


def test_month_end_forecast_covers_closed_future_hours_without_fake_demand(model,frame):
    bundle = model.forecast_bundle(frame,list(model.mapping),'2026-09-30T23:45:00+03:00')
    closed = [r for r in bundle['rows'] if pd.Timestamp(r['interval_start']).hour in [1,2]]
    assert closed
    assert all(r['predicted_count']==0 for r in closed)


def test_feature_evidence_uses_measured_lightgbm_contributions(model,frame):
    result = model.feature_evidence(frame,'2026-09-30T09:00:00+03:00',max_features=3)
    assert result['method'] == 'LightGBM pred_contrib in raw q50 log-ratio space'
    assert result['rows']
    assert all(len(r['features'])==3 for r in result['rows'])
    assert result['max_sum_error'] < 1e-8
    assert result['interpretation'] == 'model contribution, not causal effect'
