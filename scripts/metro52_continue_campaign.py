"""Adopt the existing BALANCED and explanation workers without starting model arms."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.metro52_campaign import run_incidents, save


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_scope(root):
    scope = read(Path(root)/'campaign_scope.json')
    if (scope.get('expected_arms') != ['reference', 'balanced']
            or set(scope.get('skipped_arms', [])) != {'old', 'ensemble_old'}):
        raise ValueError('Continuation requires the explicit two-arm user scope')
    return scope


def windows_records(pids):
    if any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError('Positive integer PIDs required')
    query = ' OR '.join(f'ProcessId={pid}' for pid in pids)
    code = ("[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
            f"$p=@(Get-CimInstance Win32_Process -Filter '{query}' | ForEach-Object {{"
            "[pscustomobject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;"
            "created=$_.CreationDate.ToString('o');command=$_.CommandLine;exe=$_.ExecutablePath}});"
            'ConvertTo-Json -InputObject $p -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', code],
                            capture_output=True, text=True, encoding='utf-8', check=True)
    return {row['pid']: row for row in json.loads(result.stdout)}


class WindowsProcess:
    def __init__(self, pid, *, kernel=None):
        if kernel is None:
            if sys.platform != 'win32':
                raise OSError('Live process adoption requires Windows')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            signatures = {
                'OpenProcess': (wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
                'WaitForSingleObject': (wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]),
                'GetExitCodeProcess': (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]),
                'GetProcessTimes': (wintypes.BOOL, [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4),
                'TerminateProcess': (wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
                'CloseHandle': (wintypes.BOOL, [wintypes.HANDLE]),
            }
            for name, (result, arguments) in signatures.items():
                function = getattr(kernel, name)
                function.restype, function.argtypes = result, arguments
        self.pid, self.kernel = pid, kernel
        self.handle = kernel.OpenProcess(0x101000, False, pid)
        if not self.handle:
            raise OSError(f'Cannot adopt process {pid}')
        self.created = self.creation(self.handle)

    def creation(self, handle):
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise OSError(f'Cannot read process creation time: {self.pid}')
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime

    def poll(self):
        wait = self.kernel.WaitForSingleObject(self.handle, 0)
        if wait == 258:
            return None
        if wait != 0:
            raise OSError(f'Cannot poll process {self.pid}: {wait}')
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise OSError(f'Cannot read exit code: {self.pid}')
        return code.value

    def terminate(self):
        current = self.poll()
        if current is not None:
            return current
        handle = self.kernel.OpenProcess(0x101001, False, self.pid)
        if not handle:
            raise OSError(f'Cannot finish explanation process {self.pid}')
        try:
            if self.creation(handle) != self.created:
                raise ValueError('Process identity changed before scoped completion')
            if not self.kernel.TerminateProcess(handle, 125):
                raise OSError(f'Cannot finish explanation process {self.pid}')
            if self.kernel.WaitForSingleObject(handle, 5000) != 0:
                raise TimeoutError('Explanation process did not exit')
            return self.poll()
        finally:
            self.kernel.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def adopt(pid, role, root, *, reader=windows_records, factory=WindowsProcess):
    if role not in ('balanced', 'llm'):
        raise ValueError('Only the requested workers may be adopted')
    before = reader([pid]).get(pid)
    if not before:
        raise ValueError(f'Missing live process identity: {pid}')
    command = before['command'].replace('"', '').replace('\\', '/').lower()
    path = str(Path(root).resolve()).replace('\\', '/').lower()
    source = str(ROOT/'metro51_run.py').replace('\\', '/').lower()
    marker = (source, '--arm balanced', f'--output {path}/balanced') if role == 'balanced' else (
        'scripts.metro51_explain_campaign', '--watch', f'--root {path}')
    if any(not re.search(re.escape(value)+r'(?=\s|$)', command) for value in marker):
        raise ValueError(f'Unexpected {role} process command: {pid}')
    process = factory(pid)
    if reader([pid]).get(pid) != before:
        process.close()
        raise ValueError(f'Process identity changed during adoption: {pid}')
    process.identity = before
    return process


def check_parent(identity, parent_pid):
    if identity['parent_pid'] != parent_pid:
        raise ValueError('Actual worker has an unexpected parent PID')


def require_model_result(root, processes):
    codes = [processes[name].poll() for name in ('balanced_child', 'balanced_stub')]
    if codes != [0, 0]:
        raise RuntimeError(f'BALANCED requires two successful exit codes: {codes}')
    result = read(Path(root)/'balanced/result.json')
    contract = result.get('contract', {})
    if contract.get('arm') != 'BALANCED' or contract.get('stage') != 'full_day':
        raise ValueError('Completed full-day BALANCED result required')
    return result


def explanation_coverage(root, *, validator=None, result=None):
    if validator is None:
        from metro51_explanations import validate_explanation
        validator = validate_explanation
    root = Path(root)
    facts = sorted(path for arm in ('reference', 'balanced')
                   for path in (root/arm/'explanations').glob('facts_*.json'))
    pending = []
    for path in facts:
        answer = path.with_name(path.name.replace('facts_', 'explanation_'))
        try:
            saved = read(answer)
            if saved.get('metadata', {}).get('error_code') == 'disabled':
                raise ValueError('Explanation watcher has not processed this packet')
            validator(read(path), saved)
        except (OSError, ValueError, KeyError) as error:
            pending.append(dict(path=str(path), reason=type(error).__name__))
    if result is not None:
        for decision in result.get('decisions', []):
            second = decision['decision_second']
            path = root/'balanced/explanations'/f'facts_{second:05d}.json'
            if path not in facts:
                pending.append(dict(path=str(path), reason='missing_decision_packet'))
    return dict(facts=len(facts), pending=pending, valid=len(facts)-sum(p['reason'] != 'missing_decision_packet' for p in pending))


def finish_explanations(processes, coverage, *, sleeper=time.sleep):
    if coverage['pending']:
        raise ValueError('Cannot finish watcher before all explanations are valid')
    receipts = {}
    for name in ('llm_child', 'llm_stub'):
        process = processes[name]
        code = process.poll()
        intentional = code is None
        if intentional:
            code = process.terminate()
        receipts[name] = dict(exit_code=code, intentional=intentional,
                              reason='user_scope_completion' if intentional else 'natural_exit')
        if name == 'llm_child':
            for _ in range(20):
                if processes['llm_stub'].poll() is not None:
                    break
                sleeper(.25)
    return receipts


def postprocess(root, *, execute=subprocess.run):
    root = Path(root)
    execute([sys.executable, str(ROOT/'metro51_forecast_audit.py'), '--root', str(root)], cwd=ROOT, check=True)
    run_incidents(root)
    execute([sys.executable, str(ROOT/'metro51_report.py'), str(root), '--output', str(root/'review'),
             '--incidents', str(root/'incidents')], cwd=ROOT, check=True)


def run(root, balanced_pid, balanced_child_pid, llm_pid, llm_child_pid, *, poll_seconds=5):
    root = Path(root).resolve()
    load_scope(root)
    status = read(root/'status.json')
    if status['jobs']['reference']['exit_code'] != 0 or not (root/'reference/result.json').is_file():
        raise ValueError('Completed REFERENCE must be preserved')
    reference_bytes = (root/'reference/result.json').read_bytes()
    processes = {}
    try:
        for name, pid, role in [('balanced_stub', balanced_pid, 'balanced'), ('balanced_child', balanced_child_pid, 'balanced'),
                                ('llm_stub', llm_pid, 'llm'), ('llm_child', llm_child_pid, 'llm')]:
            processes[name] = adopt(pid, role, root)
        check_parent(processes['balanced_child'].identity, balanced_pid)
        check_parent(processes['llm_child'].identity, llm_pid)
        backup = root/'logs'/f'status.before_continuation.{time.time_ns()}.json'
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes((root/'status.json').read_bytes())
        status.update(stage='running', expected_arms=['reference', 'balanced'],
                      skipped_arms=['old', 'ensemble_old'], pending_arms=[],
                      scheduler_handover='Existing worker handles adopted; no model arms started',
                      continuation=dict(pid=os.getpid(), started_at=time.time(),
                                        identities={name:p.identity for name,p in processes.items()}))
        while True:
            codes = {name: p.poll() for name,p in processes.items()}
            if any(codes[name] not in (None, 0) for name in ('balanced_child', 'balanced_stub')):
                raise RuntimeError('Adopted BALANCED process failed')
            result = require_model_result(root, processes) if codes['balanced_child'] == codes['balanced_stub'] == 0 else None
            coverage = explanation_coverage(root, result=result)
            status['jobs']['balanced'].update(exit_code=codes['balanced_stub'], child_pid=balanced_child_pid,
                                               child_exit_code=codes['balanced_child'])
            status.update(updated_at=time.time(), explanation_coverage=coverage)
            save(root/'status.json', status)
            if result is not None and not coverage['pending']:
                break
            if codes['llm_child'] is not None:
                raise RuntimeError('Explanation watcher ended before required coverage')
            time.sleep(poll_seconds)
        balanced_bytes = (root/'balanced/result.json').read_bytes()
        completion = finish_explanations(processes, coverage)
        status['jobs']['explanations'].update(
            exit_code=completion['llm_stub']['exit_code'], child_pid=llm_child_pid,
            child_exit_code=completion['llm_child']['exit_code'], completion=completion,
            completion_reason='user_scope_completion')
        if not (root/'logs/explanations.status.json').exists():
            save(root/'logs/explanations.status.json', dict(
                ended=True, completion_reason='user_scope_completion', expected_arms=['reference','balanced'],
                facts=coverage['facts'], valid=coverage['valid'], completion=completion,
                all_simulations_finished_for_selected_scope=True, all_original_arms_finished=False))
        status.update(stage='reporting', updated_at=time.time())
        save(root/'status.json', status)
        postprocess(root)
        if (root/'reference/result.json').read_bytes() != reference_bytes:
            raise ValueError('Completed REFERENCE changed during continuation')
        if (root/'balanced/result.json').read_bytes() != balanced_bytes:
            raise ValueError('Completed BALANCED changed during postprocessing')
        status.update(stage='completed', finished_at=time.time())
        save(root/'status.json', status)
        return status
    except Exception as error:
        status.update(stage='failed', error=type(error).__name__, failed_at=time.time())
        save(root/'status.json', status)
        raise
    finally:
        for process in processes.values():
            process.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--balanced-pid', type=int, required=True)
    parser.add_argument('--balanced-child-pid', type=int, required=True)
    parser.add_argument('--llm-pid', type=int, required=True)
    parser.add_argument('--llm-child-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    run(args.root, args.balanced_pid, args.balanced_child_pid, args.llm_pid, args.llm_child_pid,
        poll_seconds=args.poll_seconds)
