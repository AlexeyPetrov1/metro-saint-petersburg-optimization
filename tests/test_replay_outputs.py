import pytest

from metro_replay import safe_output


def test_replay_cannot_overwrite_previous_outputs(tmp_path):
    with pytest.raises(ValueError): safe_output(tmp_path)
