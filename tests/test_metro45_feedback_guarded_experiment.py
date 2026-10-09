import json

import pytest

import metro45_feedback_guarded_experiment as guarded


def setup(monkeypatch, tmp_path, change=False, fail=False):
    watched = tmp_path / 'ML_08.json'
    watched.write_bytes(b'past-original-model')
    monkeypatch.setattr(guarded, 'watched_paths', lambda guard: [watched])

    def replay(output, **options):
        assert not output.exists()
        output.mkdir()
        if change:
            watched.write_bytes(b'changed-during-replay')
        if fail:
            raise RuntimeError('replay failed')

    monkeypatch.setattr(guarded, 'run_feedback', replay)
    return watched


def test_prior_forecast_hashes_are_captured_before_replay(monkeypatch, tmp_path):
    watched = setup(monkeypatch, tmp_path)
    output = tmp_path / 'result'
    guarded.run(output, guard=True, steps=1)
    receipt = json.loads((output / 'startup_history_guard.json').read_text())
    assert receipt['startup_hashes_captured_before_replay'] is True
    assert receipt['sources_preserved'] is True
    assert receipt['before'] == receipt['after']
    assert str(watched) in receipt['before']


def test_mutated_history_is_rejected_and_receipt_is_kept(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, change=True)
    output = tmp_path / 'result'
    with pytest.raises(ValueError, match='history changed'):
        guarded.run(output, guard=True)
    receipt = json.loads((output / 'startup_history_guard.json').read_text())
    assert receipt['sources_preserved'] is False


def test_failed_replay_keeps_history_receipt(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, fail=True)
    output = tmp_path / 'result'
    with pytest.raises(RuntimeError, match='replay failed'):
        guarded.run(output, guard=True)
    assert json.loads((output / 'startup_history_guard.json').read_text())['sources_preserved']


def test_existing_result_is_preserved(monkeypatch, tmp_path):
    output = tmp_path / 'result'
    output.mkdir()
    monkeypatch.setattr(guarded, 'run_feedback', lambda *a, **k: pytest.fail('replay started'))
    with pytest.raises(FileExistsError):
        guarded.run(output)
