import json
from pathlib import Path

import pytest

from scripts import forecast_metro_ml as cli


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    inputs = tmp_path / 'inputs'
    (inputs / 'lgbm_final').mkdir(parents=True)
    (inputs / 'lgbm_final/meta.json').write_text('{}')
    (inputs / 'data/interim').mkdir(parents=True)
    (inputs / 'data/interim/spb_line1_hourly.parquet').write_bytes(b'ready')
    (inputs / 'data/interim/spb_line1_15min.parquet').write_bytes(b'ready')
    calls = []

    class Model:
        def __init__(self, *, allow_training_period):
            self.diagnostic = allow_training_period
            calls.append(('load', allow_training_period))

        def load_simulator_frame(self):
            return 'observed_history'

        def forecast_bundle(self, frame, stations, at):
            calls.append(('forecast', frame, stations, at))
            if at.startswith('2026-09-29') and not self.diagnostic:
                raise ValueError('weights saw this training period; enable diagnostic mode explicitly')
            if 'L2_01' in stations:
                raise ValueError('station coverage belongs to Line1, not Line2')
            return dict(target='turnstile_entries', unit='passengers_per_15m',
                evaluation_mode='training_period_diagnostic' if self.diagnostic else 'temporal_holdout_no_embargo',
                rows=[dict(station_id=sid, predicted_count=12) for sid in stations],
                source_versions=dict(train_end='2026-09-29', note='Технологический институт'))

    monkeypatch.setattr(cli, 'INPUTS', inputs)
    monkeypatch.setattr(cli, 'FrozenMetroForecaster', Model)
    return calls


def test_as_of_required_before_loading_any_model(fake_model):
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 2
    assert fake_model == []


def test_stdout_defaults_to_all_stations_and_strict_holdout(fake_model, capsys):
    assert cli.main(['--as-of', '2026-09-30T09:00:00+03:00']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['target'] == 'turnstile_entries'
    assert len(result['rows']) == 19
    assert fake_model[0] == ('load', False)
    assert fake_model[1][1] == 'observed_history'


def test_seen_training_dates_require_explicit_diagnostic_flag(fake_model, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['--as-of', '2026-09-29T09:00:00+03:00'])
    assert error.value.code == 2
    assert 'training period' in capsys.readouterr().err
    assert cli.main(['--as-of', '2026-09-29T09:00:00+03:00', '--diagnostic']) == 0
    assert json.loads(capsys.readouterr().out)['evaluation_mode'] == 'training_period_diagnostic'


def test_selected_stations_write_utf8_json_to_nested_path(fake_model, tmp_path, capsys):
    output = tmp_path / 'results with spaces/forecast.json'
    assert cli.main(['--as-of', '2026-09-30T09:00:00+03:00', '--stations', 'L1_01', 'L1_07',
        '--output', str(output)]) == 0
    text = output.read_text(encoding='utf-8')
    result = json.loads(text)
    assert [row['station_id'] for row in result['rows']] == ['L1_01', 'L1_07']
    assert 'Технологический институт' in text
    assert json.loads(capsys.readouterr().out)['output'] == str(output.resolve())


def test_missing_prepared_inputs_trigger_archive_preparation(fake_model, monkeypatch):
    (cli.INPUTS / 'lgbm_final/meta.json').unlink()
    prepared = []
    monkeypatch.setattr(cli, 'prepare_inputs', lambda: prepared.append(True))
    assert cli.main(['--as-of', '2026-09-30T09:00:00+03:00', '--stations', 'L1_01']) == 0
    assert prepared == [True]


def test_unknown_line_is_rejected_without_creating_output(fake_model, tmp_path):
    output = tmp_path / 'bad.json'
    with pytest.raises(SystemExit) as error:
        cli.main(['--as-of', '2026-09-30T09:00:00+03:00', '--stations', 'L2_01', '--output', str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_different_forecast_file_is_preserved(fake_model, tmp_path):
    output = tmp_path / 'prior.json'
    output.write_text('prior result', encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        cli.main(['--as-of', '2026-09-30T09:00:00+03:00', '--output', str(output)])
    assert error.value.code == 2
    assert output.read_text(encoding='utf-8') == 'prior result'
