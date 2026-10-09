import json

import pandas as pd
import pytest

from metro_ml_verify import check_preserved, score_forecast, check_case_sources, check_model_archive


def test_error_join_uses_station_and_time_and_preserves_zero_truth():
    at = pd.Timestamp('2026-09-30T08:00:00+03:00')
    frame = pd.DataFrame(dict(station_id=['B', 'A'], interval_start=[at, at],
        entry_count=[0, 100], is_complete=[True, True]))
    bundle = {'rows': [dict(station_id='A', interval_start=at.isoformat(), predicted_count=80),
                       dict(station_id='B', interval_start=at.isoformat(), predicted_count=10)]}
    score = score_forecast(bundle, frame, at.isoformat())
    assert score == dict(rows=2, actual_entries=100, absolute_error=30.0, wape_percent=30.0)
    frame.loc[0, 'is_complete'] = False
    with pytest.raises(ValueError, match='coverage'):
        score_forecast(bundle, frame, at.isoformat())


def test_preservation_rejects_modified_bytes_even_when_length_is_unchanged(tmp_path):
    import hashlib
    path = tmp_path/'weights.txt'
    path.write_bytes(b'1234')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'files': [dict(path='weights.txt', size=4,
        sha256=hashlib.sha256(b'1234').hexdigest())]}))
    assert check_preserved(manifest, tmp_path) == 1
    path.write_bytes(b'4321')
    with pytest.raises(ValueError, match='changed'):
        check_preserved(manifest, tmp_path)


def test_archived_experiment_source_is_verified_after_a_later_code_fix(tmp_path):
    import hashlib
    workspace, output = tmp_path/'workspace', tmp_path/'experiment'
    workspace.mkdir(); output.mkdir()
    (workspace/'runner.py').write_bytes(b'new guarded code')
    (output/'old.py').write_bytes(b'old experiment code')
    saved = dict(contract=dict(source_hashes={'runner.py': hashlib.sha256(b'old experiment code').hexdigest()}),
        archived_sources={'runner.py': 'old.py'})
    check_case_sources(saved, workspace, output)
    (output/'old.py').write_bytes(b'tampered experiment code')
    with pytest.raises(ValueError, match='changed'):
        check_case_sources(saved, workspace, output)


def test_loaded_model_bytes_must_match_uploaded_archive(tmp_path):
    import zipfile
    archive = tmp_path/'model.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('lgbm_final/q50.txt', b'original weights')
    folder = tmp_path/'lgbm_final'; folder.mkdir()
    (folder/'q50.txt').write_bytes(b'original weights')
    assert check_model_archive(archive, tmp_path) == 1
    (folder/'q50.txt').write_bytes(b'other weights')
    with pytest.raises(ValueError, match='weights changed'):
        check_model_archive(archive, tmp_path)
