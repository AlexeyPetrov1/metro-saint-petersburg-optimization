import pytest

from metro47_experiment import run


def test_existing_artifacts_are_never_overwritten(tmp_path):
    sentinel = tmp_path / 'result.json'
    sentinel.write_text('preserved', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert sentinel.read_text(encoding='utf-8') == 'preserved'


@pytest.mark.parametrize('steps', [0, 21, True])
def test_incomplete_or_invalid_control_contract_is_rejected(tmp_path, steps):
    with pytest.raises(ValueError):
        run(tmp_path / 'new', steps=steps)
    assert not (tmp_path / 'new').exists()


def test_future_oracle_cannot_be_selected(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / 'new', arm='ORACLE')
