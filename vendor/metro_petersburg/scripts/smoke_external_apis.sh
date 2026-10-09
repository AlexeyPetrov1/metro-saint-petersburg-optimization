#!/usr/bin/env bash
# Run from any directory; all CLI commands execute at the project root.
# .env supports KEY=value, optional export, quotes and comments. Values are
# literal: no command substitution, interpolation or shell code is executed.
# stdout/stderr artifacts are redacted BEFORE writing. No --debug API calls.
set +x
set -eu
umask 077
smoke_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$smoke_script_dir/.."

exec python3 - <<'PY'
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import quote, quote_plus
from zoneinfo import ZoneInfo

REQUIRED = (
    "YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL",
    "YANDEX_WEATHER_API_KEY", "YANDEX_GEOCODER_API_KEY", "YANDEX_RASP_API_KEY",
)


def load_env():
    path = Path('.env')
    if not path.exists():
        return None
    # Parse first, then export atomically; never source an executable file.
    values = {}
    try:
        for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            match = re.fullmatch(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
            if not match:
                return f'Invalid .env assignment at line {line_number}; existing environment retained.'
            name, raw = match.groups()
            tokens = shlex.split(raw, comments=True, posix=True)
            if len(tokens) > 1:
                return f'Quote values containing spaces in .env at line {line_number}; existing environment retained.'
            value = tokens[0] if tokens else ''
            if '\x00' in value:
                return f'Invalid .env value at line {line_number}; existing environment retained.'
            values[name] = value
    except (OSError, UnicodeError, ValueError):
        return 'Cannot parse .env safely; existing environment retained.'
    os.environ.update(values)
    return None


def main():
    config_error = load_env()
    secret_values = {
        value for name, value in os.environ.items()
        if value and (name in REQUIRED or re.search(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL', name, re.I))
    }
    variants = set()
    for value in secret_values:
        for secret in {value, value.strip()} - {''}:
            variants.update((secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1],
                             quote(secret, safe=''), quote_plus(secret)))
    variants = sorted(variants, key=len, reverse=True)

    def redact(value):
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='replace')
        value = value or ''
        for secret in variants:
            value = value.replace(secret, '[REDACTED]')
        # Keep newlines/tabs readable; suppress terminal control characters.
        return re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', lambda m: f'\\x{ord(m[0]):02x}', value)

    states = {name: 'SET' if os.environ.get(name, '').strip() else 'MISSING' for name in REQUIRED}
    for name, state in states.items():
        print(f'{name}: {state}', flush=True)
    if config_error:
        print('Configuration: FAIL (could not load .env safely)', flush=True)

    artifacts = Path('artifacts')
    artifacts.mkdir(exist_ok=True, mode=0o700)
    for _ in range(10):
        started = datetime.now(ZoneInfo('Europe/Moscow'))
        stem = 'external_api_smoke_' + started.strftime('%Y-%m-%d_%H%M%S')
        run_dir = artifacts / stem
        try:
            run_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            time.sleep(1)
    else:
        print('Cannot create unique smoke artifacts directory.', file=sys.stderr)
        return 1

    checks = [
        ('events', 'Events / YandexGPT', 'external_data.events.cli', [
            '--provider', 'yandex', '--text',
            '12 октября в 19:30 в БКЗ Октябрьский по адресу Лиговский проспект, 6 состоится концерт Barcelona Flamenco Ballet',
            '--published-at', '2026-10-07T12:00:00+03:00', '--source-type', 'telegram']),
        ('weather', 'Weather / Yandex Weather', 'external_data.weather.cli', ['--lat', '59.9343', '--lon', '30.3351']),
        ('locations', 'Locations / Yandex Geocoder', 'external_data.locations.cli', [
            '--location', 'БКЗ Октябрьский', '--address', 'Лиговский проспект, 6', '--top-k', '3']),
        ('railway', 'Railway / Yandex Rasp', 'external_data.railway.cli', [
            '--hub', 'Московский вокзал', '--timestamp', '2026-10-08T18:00:00+03:00']),
        ('calendar', 'Calendar / Local demo', 'external_data.calendar.cli', [
            '--timestamp', '2026-10-10T18:00:00+03:00',
            '--calendar-file', 'external_data/calendar/examples/demo_calendar.json']),
        ('features', 'Features / Local demo', 'external_data.features.cli', []),
    ]
    report_path = artifacts / (stem + '.md')
    results = []

    def fenced(content, language):
        length = max((len(m[0]) for m in re.finditer(r'`+', content)), default=0)
        fence = '`' * max(3, length + 1)
        return f'{fence}{language}\n{content.rstrip()}\n{fence}\n'

    def write_report():
        lines = ['# External API Smoke Test', f'Timestamp: {started.isoformat()}', '',
                 '| Check | Status |', '| --- | --- |']
        statuses = {item['name']: item['status'] for item in results}
        lines += [f'| {name} | {statuses.get(name, "PENDING")} |' for _, name, _, _ in checks]
        lines += ['', '## Configuration', '']
        lines += [f'- {name}: {state}' for name, state in states.items()]
        if config_error:
            lines += ['', config_error]
        lines += ['', 'Calendar and Features use local demo data. PASS means CLI exit code 0.',
                  'Secrets are redacted in saved output. Each check has a 300-second timeout.', '']
        for item in results:
            lines += [f'## {item["name"]}', f'Exit code: {item["code"]}', '',
                      f'Stdout: [{item["slug"]}.stdout.json]({stem}/{item["slug"]}.stdout.json)',
                      f'Exit code: [{item["slug"]}.exitcode]({stem}/{item["slug"]}.exitcode)', '',
                      fenced(item['stdout'], 'json')]
            if item['stderr']:
                lines += [f'Stderr: [{item["slug"]}.stderr.txt]({stem}/{item["slug"]}.stderr.txt)', '',
                          fenced(item['stderr'], 'text')]
        report_path.write_text('\n'.join(lines), encoding='utf-8')

    write_report()
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for slug, name, module, args in checks:
        print(f'{name}: RUNNING', flush=True)
        try:
            process = subprocess.run([sys.executable, '-m', module, *args],
                                     capture_output=True, timeout=300, env=child_env)
            code, stdout, stderr = process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            code, stdout = 124, error.stdout
            stderr = (error.stderr or b'') + b'\nSmoke check timed out after 300 seconds.\n'
        except OSError:
            code, stdout, stderr = 127, '', 'Could not start CLI process.'
        stdout, stderr = redact(stdout), redact(stderr)
        (run_dir / f'{slug}.stdout.json').write_text(stdout, encoding='utf-8')
        (run_dir / f'{slug}.exitcode').write_text(f'{code}\n', encoding='utf-8')
        if code != 0 or stderr:
            (run_dir / f'{slug}.stderr.txt').write_text(stderr, encoding='utf-8')
        status = 'PASS' if code == 0 else 'FAIL'
        results.append(dict(slug=slug, name=name, status=status, code=code, stdout=stdout, stderr=stderr))
        write_report()
        print(f'{name}: {status}', flush=True)
    print(f'Report: {report_path}', flush=True)
    return int(bool(config_error) or any(item['code'] != 0 for item in results))


try:
    raise SystemExit(main())
except Exception:
    # Never leak a raw exception that might contain credentials or child output.
    print('Smoke runner failed; check local Python and artifact write permissions.', file=sys.stderr)
    raise SystemExit(1)
PY
