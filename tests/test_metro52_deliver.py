import hashlib
import importlib
import json
import subprocess
import zipfile

import pytest


@pytest.fixture
def delivery():
    return importlib.import_module('scripts.metro52_deliver')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'campaign'
    write_json(root/'status.json', dict(stage='completed', expected_arms=['reference','balanced'],
        pending_arms=[], jobs={a:dict(exit_code=0) for a in ['reference','balanced']}))
    for arm in ['reference','balanced']:
        write_json(root/arm/'result.json', dict(contract=dict(arm=arm.upper(), stage='full_day'), full_day=True))
    pdf = root/'review/report.pdf'
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b'%PDF-1.7\nactual-test-pdf')
    write_json(root/'review/comparison.json', dict(full_day=True, audits={a:{} for a in ['reference','balanced']},
                                                goal_achieved=False, all_results_admissible=False))
    write_json(root/'review/verification.json', dict(pdf_created=True,
        artifacts={'report.pdf':hashlib.sha256(pdf.read_bytes()).hexdigest(),
                   'comparison.json':hashlib.sha256((root/'review/comparison.json').read_bytes()).hexdigest()}))
    write_json(root/'forecast_review/report.json', {'wape':0.1})
    return root


def test_complete_failed_economics_can_be_delivered(delivery, campaign):
    assert delivery.completed(campaign) is True


def test_incomplete_does_not_trigger_delivery(delivery, campaign):
    status = json.loads((campaign/'status.json').read_text())
    status['stage'] = 'reporting'
    write_json(campaign/'status.json', status)
    assert delivery.completed(campaign) is False


@pytest.mark.parametrize('failure', ['missing_pdf','changed_pdf','missing_arm','failed_scope'])
def test_completion_guards(delivery, campaign, failure):
    if failure == 'missing_pdf':
        (campaign/'review/report.pdf').unlink()
    elif failure == 'changed_pdf':
        (campaign/'review/report.pdf').write_bytes(b'%PDF-replaced')
    elif failure == 'missing_arm':
        (campaign/'balanced/result.json').unlink()
    else:
        write_json(campaign/'status.json', {'stage':'failed'})
    with pytest.raises((ValueError, FileNotFoundError)):
        delivery.completed(campaign)


def test_archive_crc_hashes_and_exact_public_copy(delivery, campaign, tmp_path):
    original = (campaign/'reference/result.json').read_bytes()
    for name in ['.env','.ENV']:
        (campaign/name).write_bytes(b'not-read-or-delivered')
    export, output = tmp_path/'export', tmp_path/'delivery'
    manifest = delivery.prepare(campaign, export, output)
    with zipfile.ZipFile(output/'campaign.zip') as archive:
        assert archive.testzip() is None
        assert '.env' not in archive.namelist()
        assert '.ENV' not in archive.namelist()
        for row in manifest['files']:
            content = archive.read(row['path'])
            assert hashlib.sha256(content).hexdigest() == row['sha256']
            copied = export/'data/processed/campaign'/row['path']
            assert copied.read_bytes() == content
    assert (export/'data/processed/campaign/reference/result.json').read_bytes() == original
    assert not list(export.rglob('*.zip'))
    assert manifest['readme_update'] == {'exists':False, 'changed':False}


def test_only_pending_readme_phrase_changes_after_completed_campaign(delivery, campaign, tmp_path):
    export = tmp_path/'export'
    export.mkdir()
    phrase = 'Новый суточный прогон выполняется; итоговых результатов ещё нет.'
    before = ('# Existing title\r\n\r\n'+phrase+'\r\n\r\nExisting footer.\r\n').encode('utf8')
    (export/'README.md').write_bytes(before)
    manifest = delivery.prepare(campaign, export, tmp_path/'delivery')
    after = (export/'README.md').read_bytes()
    assert phrase.encode('utf8') not in after
    assert after.startswith(b'# Existing title\r\n\r\n')
    assert after.endswith(b'\r\n\r\nExisting footer.\r\n')
    assert b'data/processed/campaign/review/report.pdf' in after
    assert b'data/processed/campaign/review/report.html' in after
    assert 'Суточный прогон завершён.'.encode('utf8') in after
    assert manifest['readme_update'] == dict(exists=True, changed=True,
        previous_sha256=hashlib.sha256(before).hexdigest(), new_sha256=hashlib.sha256(after).hexdigest())
    comparison = json.loads((campaign/'review/comparison.json').read_text())
    assert comparison['goal_achieved'] is False


def test_current_readme_is_preserved(delivery, campaign, tmp_path):
    export = tmp_path/'export'
    export.mkdir()
    before = b'# Current README\r\nEverything remains.\r\n'
    (export/'README.md').write_bytes(before)
    manifest = delivery.prepare(campaign, export, tmp_path/'delivery')
    assert (export/'README.md').read_bytes() == before
    assert manifest['readme_update']['changed'] is False


def test_updated_readme_is_in_scoped_publication(delivery, campaign, tmp_path, monkeypatch):
    export = tmp_path/'export'
    export.mkdir()
    (export/'README.md').write_text('Новый суточный прогон выполняется; итоговых результатов ещё нет.', encoding='utf8')
    calls = []
    def publish(path, paths):
        calls.append((path, paths))
        return 'a'*40
    monkeypatch.setattr(delivery, 'publish', publish)
    status = delivery.watch(campaign, export, tmp_path/'delivery', should_publish=True, interval=0, quiet_seconds=0)
    assert status['published'] is True
    assert calls == [(export, ['data/processed/campaign','verification_delivery/campaign','README.md'])]


def test_secret_and_large_file_block_before_public_copy(delivery, campaign, tmp_path, monkeypatch):
    export = tmp_path/'export'
    secret = campaign/'unsafe.log'
    secret.write_bytes(b'AQVN' + b'a'*40)
    with pytest.raises(ValueError, match='secret'):
        delivery.prepare(campaign, export, tmp_path/'delivery')
    assert not export.exists()
    secret.unlink()
    monkeypatch.setattr(delivery, 'MAX_FILE_BYTES', 10)
    with pytest.raises(ValueError, match='100 MB'):
        delivery.prepare(campaign, export, tmp_path/'delivery')
    assert not export.exists()


def test_publish_is_noninteractive_scoped_and_verifies_remote(delivery, tmp_path):
    calls = []
    sha = 'f'*40
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        output = sha+'\n' if command[1:3] == ['rev-parse','HEAD'] else ''
        if command[1] == 'ls-remote':
            output = sha+'\trefs/heads/main\n'
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
    paths = ['data/processed/campaign','verification_delivery/campaign']
    assert delivery.publish(tmp_path, paths, execute=execute) == sha
    assert ['git','add','--',*paths] in [c for c,_ in calls]
    assert ['git','push','origin','HEAD:main'] in [c for c,_ in calls]
    assert all('--force' not in c for c,_ in calls)
    assert all(k['env']['GIT_TERMINAL_PROMPT'] == '0' and k['env']['GCM_INTERACTIVE'] == 'never' for _,k in calls)


def test_failed_campaign_records_failure_without_process_commands(delivery, campaign, tmp_path):
    write_json(campaign/'status.json', {'stage':'failed'})
    target = tmp_path/'delivery'
    with pytest.raises(ValueError):
        delivery.watch(campaign, tmp_path/'export', target)
    assert json.loads((target/'status.json').read_text())['stage'] == 'failed'


@pytest.mark.parametrize('failure', ['unrelated_index','remote_mismatch'])
def test_publication_failure_is_explicit(delivery, tmp_path, failure):
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        output = ''
        if command[1:3] == ['diff','--cached'] and failure == 'unrelated_index':
            output = 'unrelated.py\n'
        elif command[1] == 'rev-parse':
            output = 'a'*40
        elif command[1] == 'ls-remote':
            output = 'b'*40+'\trefs/heads/main\n'
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')
    with pytest.raises(ValueError):
        delivery.publish(tmp_path, ['data/processed/campaign'], execute=execute)
    if failure == 'unrelated_index':
        assert len(calls) == 1
