"""Deliver an already completed campaign; never start or stop simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from scripts.metro52_package import TOKENS, allowed, long_path

MAX_FILE_BYTES = 100_000_000
ARMS = ['reference', 'balanced']


def read(path):
    return json.loads(long_path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with long_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def included(relative):
    return (allowed(relative) and not any(part.lower().startswith('.env') for part in relative.parts)
            and relative.name.lower() != 'secrets.toml'
            and 'pdf-browser-profile' not in relative.parts and not relative.name.endswith('.tmp'))


def completed(root):
    root = Path(root)
    try:
        status = read(root/'status.json')
    except FileNotFoundError:
        return False
    if status.get('stage') == 'failed':
        raise ValueError('Campaign failed; completion cannot be delivered')
    if status.get('stage') not in ('completed', 'completed_partial'):
        return False
    partial = status['stage'] == 'completed_partial'
    if status.get('expected_arms') != ARMS or status.get('pending_arms'):
        raise ValueError('Completed campaign has an unexpected scope')
    for arm in ARMS:
        if status.get('jobs', {}).get(arm, {}).get('exit_code') != 0:
            raise ValueError('Required model arm did not exit successfully')
        result = read(root/arm/'result.json')
        contract = result.get('contract', {})
        if partial:
            if (contract.get('arm') != arm.upper() or contract.get('stage') != 'partial_60'
                    or contract.get('full_day') is not False or result.get('full_day') is not False
                    or contract.get('stop_after_steps') != 60 or read(root/arm/'contract.json') != contract):
                raise ValueError('Required partial result and matching materialized contract are missing')
        elif contract.get('arm') != arm.upper() or contract.get('stage') != 'full_day' or result.get('full_day') is not True:
            raise ValueError('Required full-day result is missing')
    pdf = root/'review/report.pdf'
    with long_path(pdf).open('rb') as stream:
        if stream.read(4) != b'%PDF':
            raise ValueError('Required report is not a PDF')
    comparison = read(root/'review/comparison.json')
    if partial:
        audits = comparison.get('audits', {})
        if (comparison.get('full_day') is not False or comparison.get('stage') != 'partial_60'
                or comparison.get('annual_scenario_emitted') is not False
                or comparison.get('paired_inputs_passed') is not True or set(audits) != set(ARMS)
                or comparison.get('cash_saving_rub') is not None or comparison.get('goal_achieved') is True
                or not all(isinstance(a, dict) and isinstance(a.get('checks'), dict)
                           and a.get('checkpoint_verified_against_recompiled_runner') is True for a in audits.values())):
            raise ValueError('Partial comparison requires paired audits without full-day, annual or cash claims')
    elif comparison.get('full_day') is not True or set(comparison.get('audits', {})) != set(ARMS):
        raise ValueError('Comparison must cover both completed full-day arms')
    verification = read(root/'review/verification.json')
    artifacts = verification.get('artifacts', {})
    if verification.get('pdf_created') is not True or not {'report.pdf','comparison.json'} <= set(artifacts):
        raise ValueError('Report verification is incomplete')
    for name, sha in artifacts.items():
        if Path(name).name != name or digest(root/'review'/name) != sha:
            raise ValueError('Report artifact SHA verification failed')
    if not isinstance(read(root/'forecast_review/report.json'), dict):
        raise ValueError('Forecast audit is missing')
    return True


def selected(root):
    rows, omitted = [], []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not included(relative):
            if path.is_file():
                omitted.append(relative.as_posix())
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Campaign symlinks are outside delivery contract')
        if path.is_file():
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('An individual campaign file exceeds the public 100 MB limit')
            rows.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=digest(path)))
    return rows, omitted


def scan(root, rows):
    findings = []
    for row in rows:
        tail = b''
        with long_path(root/row['path']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                content = tail+chunk
                variants = [content]
                if b'\x00' in content[:512]:
                    variants.append(content.decode('utf-16-le', errors='ignore').encode('utf8'))
                for kind, pattern in TOKENS:
                    if any(pattern.search(value) for value in variants):
                        finding = dict(path=row['path'], kind=kind)
                        if finding not in findings:
                            findings.append(finding)
                tail = content[-4096:]
    return findings


def update_readme(export, relative, *, partial=False):
    path = export/'README.md'
    if not path.exists():
        return dict(exists=False, changed=False)
    before = path.read_bytes()
    pending = ('Новый суточный прогон выполняется; итоговых результатов ещё нет.',
               'Выполняется сокращённый прогон до 60-го решения (21:00); итоговых результатов ещё нет.')
    headline = 'Сокращённый опыт 06:00–21:00 завершён после 60 решений; полный день не выполнялся.' if partial else 'Суточный прогон завершён.'
    complete = (f'{headline} [PDF]({relative}/review/report.pdf), '
                f'[HTML]({relative}/review/report.html), [отчёты и данные]({relative}). '
                'Результаты проверок безопасности и экономики приведены в отчётах.').encode('utf8')
    after = before
    for marker in pending:
        after = after.replace(marker.encode('utf8'), complete)
    if after != before:
        path.write_bytes(after)
    return dict(exists=True, changed=after != before,
                previous_sha256=hashlib.sha256(before).hexdigest(), new_sha256=hashlib.sha256(after).hexdigest())


def prepare(root, export, delivery):
    root, export, delivery = (Path(p).resolve() for p in (root, export, delivery))
    if delivery.is_relative_to(root) or export.is_relative_to(root) or delivery.is_relative_to(export):
        raise ValueError('Archive and public export must be separate from campaign')
    if not completed(root):
        raise ValueError('Campaign is not completed')
    partial = read(root/'status.json')['stage'] == 'completed_partial'
    rows, omitted = selected(root)
    findings = scan(root, rows)
    if findings:
        save(delivery/'secret_findings.json', findings)
        raise ValueError('Campaign secret scan failed; values were not recorded')
    delivery.mkdir(parents=True, exist_ok=True)
    archive = delivery/'campaign.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for row in rows:
            output.write(long_path(root/row['path']), row['path'])
    with zipfile.ZipFile(temporary) as output:
        if output.testzip() is not None or set(output.namelist()) != {r['path'] for r in rows}:
            raise ValueError('ZIP CRC or inventory verification failed')
        for row in rows:
            with output.open(row['path']) as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha != row['sha256'] or digest(root/row['path']) != row['sha256']:
                raise ValueError('Campaign changed during archive creation')
    temporary.replace(archive)
    relative = Path('data/processed')/root.name
    for row in rows:
        source, target = root/row['path'], export/relative/row['path']
        if not target.resolve().is_relative_to(export):
            raise ValueError('Public destination escapes the export root')
        if digest(source) != row['sha256']:
            raise ValueError('Campaign changed before public copy')
        if target.exists() and digest(target) != row['sha256']:
            raise ValueError('Public destination contains different existing data')
        long_path(target.parent).mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(long_path(source), long_path(target))
        if digest(target) != row['sha256']:
            raise ValueError('Public file SHA verification failed')
    manifest = dict(schema_version=1, campaign=root.name, public_path=relative.as_posix(), files=rows,
        omitted=omitted, secret_findings=[], zip_crc_verified=True, zip_sha256=digest(archive),
        zip_bytes=archive.stat().st_size, zip_published=False, source_bytes_preserved=True,
        economic_goal_required_for_delivery=False, scope='partial_60' if partial else 'full_day',
        full_day_completed=not partial, annual_scenario_emitted=not partial, accepted_cash_saving_rub=None)
    manifest['readme_update'] = update_readme(export, relative.as_posix(), partial=partial)
    save(delivery/'MANIFEST.json', manifest)
    heading = ('# Partial campaign delivery\n\n'
               'Paired REFERENCE and BALANCED records cover 06:00–21:00 after 60 decisions. '
               'Materialization completed; the original full-day process did not complete. '
               'Unclosed journeys are censored; no daily or annual extrapolation is included.\n\n' if partial else
               '# Completed campaign delivery\n\nBoth REFERENCE and BALANCED completed. '
               'Reports include safety, forecast, incident and economic findings.\n\n')
    note = (heading+
            'Delivery does not certify operational safety, economic improvement or real cash savings. '
            'Failed acceptance checks remain visible in the original reports.\n\n'
            f'Public data: `{relative.as_posix()}`. Local complete archive: `{archive}`. '
            f'Archive SHA256: `{manifest["zip_sha256"]}`. ZIP is not uploaded to GitHub.\n\n'
            'Original report/data bytes and provenance paths are preserved. MANIFEST.json lists every included file '
            'and the excluded credential files, caches and temporary browser files.\n')
    (delivery/'DELIVERY.md').write_text(note, encoding='utf8')
    receipt = export/'verification_delivery'/root.name
    receipt.mkdir(parents=True, exist_ok=True)
    for name in ['MANIFEST.json','DELIVERY.md']:
        shutil.copyfile(delivery/name, receipt/name)
    return manifest


def publish(export, paths, *, execute=subprocess.run):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    def git(*arguments):
        return execute(['git',*arguments], cwd=export, env=env, check=True, timeout=600,
                       capture_output=True, text=True).stdout.strip()
    staged = git('diff','--cached','--name-only').splitlines()
    if any(not any(name == p or name.startswith(p+'/') for p in paths) for name in staged):
        raise ValueError('Unrelated staged changes block scoped publication')
    git('add','--',*paths)
    git('commit','--only','-m','Publish completed metro campaign reports and data','--',*paths)
    sha = git('rev-parse','HEAD')
    git('push','origin','HEAD:main')
    remote = git('ls-remote','origin','refs/heads/main').split()
    if not remote or remote[0] != sha:
        raise ValueError('Remote commit SHA verification failed')
    return sha


def fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob('*') if p.is_file() and included(p.relative_to(root)))


def watch(root, export, delivery, *, should_publish=False, interval=10, quiet_seconds=20, max_seconds=86400):
    root, export, delivery = (Path(p).resolve() for p in (root, export, delivery))
    delivery.mkdir(parents=True, exist_ok=True)
    (delivery/'DELIVERY.md').write_text(
        '# Campaign delivery pending\n\nWaiting for both completed arms and verified reports.\n\n'
        f'Expected archive: `{delivery/"campaign.zip"}`. It has not been created yet.\n', encoding='utf8')
    status = dict(stage='waiting', campaign=str(root), publish_requested=should_publish)
    started, stable_since, previous = time.monotonic(), None, None
    try:
        while time.monotonic()-started < max_seconds:
            status['heartbeat_at'] = time.time()
            save(delivery/'status.json', status)
            if completed(root):
                current = fingerprint(root)
                if current != previous:
                    previous, stable_since = current, time.monotonic()
                elif time.monotonic()-stable_since >= quiet_seconds:
                    break
            time.sleep(interval)
        else:
            raise TimeoutError('Completed stable campaign was not available before delivery timeout')
        status.update(stage='packaging', heartbeat_at=time.time())
        save(delivery/'status.json', status)
        manifest = prepare(root, export, delivery)
        status.update(stage='copied', zip_sha256=manifest['zip_sha256'], heartbeat_at=time.time())
        save(delivery/'status.json', status)
        if should_publish:
            paths = [manifest['public_path'], 'verification_delivery/'+root.name]
            if manifest['readme_update']['changed']:
                paths.append('README.md')
            status['commit'] = publish(export, paths)
        status.update(stage='completed', published=should_publish, finished_at=time.time())
        save(delivery/'status.json', status)
        return status
    except Exception as error:
        status.update(stage='failed', error_type=type(error).__name__, failed_at=time.time())
        if isinstance(error, ValueError):
            status['reason'] = str(error)
        save(delivery/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','export','delivery']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--quiet-seconds', type=float, default=20)
    parser.add_argument('--max-seconds', type=float, default=86400)
    args = parser.parse_args()
    if args.interval <= 0 or args.quiet_seconds < 15 or args.max_seconds <= 0:
        parser.error('Positive polling/timeout and at least 15 quiet seconds are required')
    watch(args.root, args.export, args.delivery, should_publish=args.publish,
          interval=args.interval, quiet_seconds=args.quiet_seconds, max_seconds=args.max_seconds)
