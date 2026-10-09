import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from metro48_gru import MODEL_INPUTS, FrozenQuarterGRU, build_window, decode_slots, extract_model, split_slots


def test_extract_refuses_existing_destination(tmp_path):
    archive = tmp_path / 'a.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('model/meta.json', '{}')
    destination = tmp_path / 'out'
    destination.mkdir()
    with pytest.raises(FileExistsError):
        extract_model(archive, destination)


def test_extract_validates_all_paths_before_writing(tmp_path):
    archive = tmp_path / 'a.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('model/meta.json', '{}')
        z.writestr('../escape.pt', 'bad')
    with pytest.raises(ValueError):
        extract_model(archive, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_decode_uses_offset_log_target_and_preserves_quantiles():
    z = np.array([[[-1., 0., 1.]] * 4])
    n = np.array([[100., 200., 300., 0.]])
    y = decode_slots(z, n)
    np.testing.assert_allclose(y[..., 1], n)
    assert np.all(y[..., 0] <= y[..., 1])
    assert np.all(y[..., 1] <= y[..., 2])
    assert y[0, 0, 2] == pytest.approx(101 * np.e - 1)


def test_decode_rejects_nonfinite_predictions():
    with pytest.raises(ValueError):
        decode_slots(np.full((1, 4, 3), np.nan), np.ones((1, 4)))


def test_split_preserves_native_totals_and_causal_profile():
    out = split_slots(np.array([[11., 13., 15., 17.]]), np.array([[1., 3.] * 4]))
    assert out.shape == (1, 8)
    np.testing.assert_array_equal(out.reshape(1, 4, 2).sum(2), [[11, 13, 15, 17]])
    assert out[0, 1] > out[0, 0]


def test_split_zero_norm_uses_equal_shares():
    assert split_slots(np.array([[3., 3., 3., 3.]]), np.zeros((1, 8))).tolist() == [[2, 1] * 4]


def _history():
    now = pd.Timestamp('2026-09-30 09:15', tz='Europe/Moscow')
    times = pd.date_range(now - pd.Timedelta(hours=4), periods=24, freq='15min')
    records = [dict(ts=t, vestibule_id=v, entries=10., norm=10.) for t in times for v in ['a', 'b']]
    return now, pd.DataFrame(records)


def test_window_uses_16_completed_quarters_and_line_ratio():
    now, history = _history()
    x = build_window(history, ['a', 'b'], now)
    assert x.shape == (2, 16, 4)
    np.testing.assert_array_equal(x, np.zeros_like(x))


def test_future_target_mutation_cannot_change_window():
    now, history = _history()
    expected = build_window(history, ['a', 'b'], now)
    history.loc[history.ts >= now, ['entries', 'norm']] = 1e8
    np.testing.assert_array_equal(build_window(history, ['a', 'b'], now), expected)


def test_window_missing_values_are_explicitly_masked():
    now, history = _history()
    history.loc[history.vestibule_id.eq('a'), 'entries'] = np.nan
    x = build_window(history, ['a', 'b'], now)
    assert np.isnan(x[0, :, 0]).all()
    assert (x[0, :, 1] == 1).all()
    assert (x[:, :, 3] == 0).all()


def test_window_rejects_duplicate_or_unfinished_intervals():
    now, history = _history()
    with pytest.raises(ValueError):
        build_window(pd.concat([history, history.iloc[:1]]), ['a', 'b'], now)
    with pytest.raises(ValueError):
        build_window(history, ['a', 'b'], now + pd.Timedelta(minutes=1))


@pytest.fixture(scope='module')
def actual_model():
    if not (MODEL_INPUTS / 'stack_gru_2026-09/meta.json').exists():
        pytest.skip('uploaded GRU inputs not extracted')
    return FrozenQuarterGRU()


@pytest.fixture(scope='module')
def actual_frame():
    return pd.read_parquet('data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet')


def test_uploaded_contract_has_original_embedding_order_and_training_cutoff(actual_model):
    contract = actual_model.contract()
    assert contract['meta']['info']['train_end'] == '2026-07-31'
    assert contract['vestibule_order'] == sorted(contract['vestibule_order'])
    assert len(contract['vestibule_order']) == 23
    assert contract['uncovered_station_ids'] == ['L1_07']
    assert contract['native_training_update_seconds'] == 1800
    assert not contract['full_stack_weights_in_archive']


def test_uploaded_bundle_marks_shifted_quarter_and_complete_red_line(actual_model, actual_frame):
    from forecasting import validate_bundle
    now = '2026-09-30T09:15:00+03:00'
    stations = list(actual_model.mapping)
    bundle = actual_model.forecast_bundle(actual_frame, stations, now)
    assert len(bundle['rows']) == 152
    assert bundle['refresh_seconds'] == 900
    assert bundle['source_versions']['shifted_quarter_inference']
    assert not bundle['source_versions']['retrospective_history_flags']
    assert validate_bundle(bundle, stations, now)['status'] == 'passed'


def test_actual_future_mutation_cannot_change_prediction(actual_model, actual_frame):
    now = pd.Timestamp('2026-09-30T09:15:00+03:00')
    a = actual_model.forecast_bundle(actual_frame, list(actual_model.mapping), now)
    changed = actual_frame.copy()
    changed.loc[changed.interval_end > now, 'entry_count'] = 10000000
    b = actual_model.forecast_bundle(changed, list(actual_model.mapping), now)
    assert a['rows'] == b['rows']
    assert a['source_versions']['preprocessing_input_digest'] == b['source_versions']['preprocessing_input_digest']


def test_newly_completed_quarter_changes_gru_prediction(actual_model, actual_frame):
    now = pd.Timestamp('2026-09-30T09:15:00+03:00')
    a = actual_model.model_slots(actual_frame, now)
    changed = actual_frame.copy()
    changed.loc[changed.interval_end.eq(now), 'entry_count'] *= 2
    b = actual_model.model_slots(changed, now)
    assert not np.allclose(a['slots'], b['slots'])


def test_training_period_is_rejected(actual_model, actual_frame):
    with pytest.raises(ValueError, match='training'):
        actual_model.model_slots(actual_frame, '2026-07-31T09:00:00+03:00')
