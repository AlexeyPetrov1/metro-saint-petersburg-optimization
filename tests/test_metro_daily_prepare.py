from metro_daily_prepare import prepare

import pytest


def test_preparation_refuses_to_overwrite_even_a_partial_experiment(tmp_path):
    existing = tmp_path / 'observations.parquet'
    existing.write_bytes(b'preserve this dataset')
    with pytest.raises(ValueError, match='empty'):
        prepare(tmp_path)
    assert existing.read_bytes() == b'preserve this dataset'
