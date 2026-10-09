"""Run the tested full-day comparison and keep its progress on disk."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_suites(paths):
    results = []
    for path in paths:
        path = Path(path)
        tree = ET.parse(path)
        suites = list(tree.iter('testsuite'))
        counts = {key: sum(int(s.get(key, 0)) for s in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        if counts['failures'] or counts['errors'] or not counts['tests']:
            raise ValueError(f'Unsuccessful readiness suite: {path}')
        results.append(dict(path=str(path.resolve()), sha256=sha256(path), **counts))
    if not results:
        raise ValueError('Readiness evidence is required')
    return results


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_commands(python, output):
    return {arm.lower(): [python, str(ROOT/'metro51_run.py'), '--output', str(Path(output)/arm.lower()),
                         '--arm', arm, '--stage', 'full_day', '--seed', '7',
                         '--control-start-second', '900', '--max-attempts', '2']
            for arm in ('REFERENCE', 'OLD', 'ENSEMBLE_OLD', 'BALANCED')}


def can_start_model(running, limit, available_bytes):
    threshold = 2_000_000_000 if running == 0 else 4_000_000_000
    return running < limit and available_bytes >= threshold


def run_incidents(output):
    from metro51_incident_probe import run as incidents
    return incidents(Path(output)/'incidents', source=Path(output)/'reference',
                     total_budget_seconds=480, minutes=(3, 8, 15))


def available_memory():
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Unable to read available memory')
        return status.available
    import os
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    temporary.replace(path)


def model_proof(output, junit):
    inputs = ROOT/'data/processed/metro51_inputs_20261009'
    sources = [ROOT/name for name in ('metro51_ensemble.py', 'metro49_predict.py', 'metro50_ensemble.py',
                                      'tests/test_metro51_ensemble.py')]
    sources = [p for p in sources if p.is_file()]
    save(output/'model_verification/evidence.json', dict(
        exit_code=0, command=[sys.executable, '-m', 'pytest', 'tests'],
        junit_xml=str(Path(junit).resolve()), junit_sha256=sha256(junit),
        source_sha256={str(p.resolve()): sha256(p) for p in sources},
        input_manifest_sha256=sha256(inputs/'input_manifest.json')))


def run(output, suites, *, max_models=2):
    readiness = check_suites(suites)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Campaign already exists: {output}')
    output.mkdir(parents=True)
    model_proof(output, suites[0])
    from metro51_report import verify_model_evidence
    verification = verify_model_evidence(ROOT/'data/processed/metro51_inputs_20261009',
                                          test_evidence=output/'model_verification/evidence.json')
    save(output/'model_verification/verification.json', verification)
    if not verification['passed']:
        raise ValueError(f"Model verification failed: {verification['failed_checks']}")
    commands = arm_commands(sys.executable, output)
    status = dict(stage='running', readiness=readiness, started_at=time.time(), jobs={},
                  features=['full_frozen_ensemble', 'causal_quarter_forecasts', 'joint_optimizer',
                            'stochastic_boarding', 'local_service_gates', 'stabling_and_depot_mileage',
                            'yandexgpt_explanations', 'forecast_audit', 'synthetic_stops', 'economics_pdf'])
    processes, streams = {}, []

    def start(name, command):
        stream = (output/f'{name}.log').open('x', encoding='utf8')
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        streams.append(stream)
        processes[name] = process
        status['jobs'][name] = dict(pid=process.pid, command=command, started_at=time.time(), exit_code=None)
        print(json.dumps(dict(stage='started', name=name, pid=process.pid)), flush=True)

    try:
        start('reference', commands['reference'])
        start('explanations', [sys.executable, '-m', 'scripts.metro51_explain_campaign', '--root', str(output),
                               '--watch', '--timeout', '15', '--max-seconds', '43200'])
        queue = ['balanced', 'old', 'ensemble_old']
        waiting_since = None
        while queue or any(p.poll() is None for n, p in processes.items() if n != 'explanations'):
            for name, process in processes.items():
                if process.poll() is not None:
                    status['jobs'][name]['exit_code'] = process.returncode
            failed = [n for n, j in status['jobs'].items() if j['exit_code'] not in (None, 0)]
            if failed:
                raise RuntimeError(f'Campaign process failed: {failed}')
            running = sum(p.poll() is None for n, p in processes.items() if n in commands and n != 'reference')
            memory = available_memory()
            if queue and processes['reference'].poll() == 0 and can_start_model(running, max_models, memory):
                name = queue.pop(0)
                start(name, commands[name])
                waiting_since = None
            elif queue and running == 0 and processes['reference'].poll() == 0:
                waiting_since = waiting_since or time.monotonic()
                if time.monotonic()-waiting_since > 300:
                    raise MemoryError('Not enough memory to start a model after five minutes')
            status.update(updated_at=time.time(), available_memory_bytes=memory, pending_arms=queue.copy())
            save(output/'status.json', status)
            time.sleep(5)
        processes['explanations'].wait()
        status['jobs']['explanations']['exit_code'] = processes['explanations'].returncode
        if processes['explanations'].returncode:
            raise RuntimeError('Explanation worker failed')
        status['stage'] = 'reporting'
        save(output/'status.json', status)
        subprocess.run([sys.executable, str(ROOT/'metro51_forecast_audit.py'), '--root', str(output)], cwd=ROOT, check=True)
        incident_output = output/'incidents'
        run_incidents(output)
        subprocess.run([sys.executable, str(ROOT/'metro51_report.py'), str(output),
                        '--output', str(output/'review'), '--incidents', str(incident_output)], cwd=ROOT, check=True)
        status.update(stage='completed', finished_at=time.time())
    except Exception as exc:
        status.update(stage='failed', error=f'{type(exc).__name__}: {exc}', failed_at=time.time())
        raise
    finally:
        save(output/'status.json', status)
        for stream in streams:
            stream.close()
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--junit', type=Path, nargs='+', required=True)
    parser.add_argument('--max-models', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.output, args.junit, max_models=args.max_models)
