"""Stop after the sixtieth committed decision and report a paired partial window."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.metro52_campaign import save, run_incidents
from scripts.metro52_continue_campaign import (
    adopt, check_parent, explanation_coverage, finish_explanations, read)


def selected_output(root):
    root = Path(root).resolve()
    scope = read(root / 'campaign_scope.json')
    if scope.get('stop_after_steps') != 60 or scope.get('stop_at_second') != 54000:
        raise ValueError('Explicit stop-after-60 scope is required')
    if scope.get('expected_arms') != ['reference', 'balanced']:
        raise ValueError('The paired reference and balanced scope is required')
    if read(root / 'balanced/contract.json').get('control_start_second') != 900:
        raise ValueError('The existing decision clock must start at 06:15')
    output = Path(scope['comparison_root']).resolve()
    if output == root or output.parent != root.parent:
        raise ValueError('Partial comparison must use a separate sibling folder')
    return output


def cutoff_ready(root):
    arm = Path(root) / 'balanced'
    names = ['controller/decision_54000.json', 'controller/resume_54000.json.gz',
             'explanations/facts_54000.json', 'explanations/explanation_54000.json']
    if not all((arm / name).is_file() for name in names):
        return False
    try:
        if read(arm / names[0]).get('decision_second') != 54000:
            return False
        lines = (arm / 'events.jsonl').read_text(encoding='utf-8-sig').splitlines()
    except (OSError, ValueError):
        return False
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (event.get('stage'), event.get('index'), event.get('second')) == ('decision_completed', 60, 54000):
            return True
    return False


def stop_balanced(root, processes, *, sleeper=time.sleep):
    if not cutoff_ready(root):
        raise ValueError('Cannot stop before the sixtieth decision is committed')
    receipts = {}
    for name in ('balanced_child', 'balanced_stub'):
        process = processes[name]
        code = process.poll()
        intentional = code is None
        if intentional:
            code = process.terminate()
        receipts[name] = dict(exit_code=code, intentional=intentional)
        if name == 'balanced_child':
            for _ in range(20):
                if processes['balanced_stub'].poll() is not None:
                    break
                sleeper(.25)
    return dict(reason='user_requested_stop_after_60', cutoff_second=54000,
                committed_steps=60, processes=receipts, full_day_finished=False)


def run(root, balanced_pid, balanced_child_pid, llm_pid, llm_child_pid, *, poll_seconds=.25):
    root = Path(root).resolve()
    output = selected_output(root)
    status = read(root / 'status.json')
    processes = {}
    try:
        for name, pid, role in [('balanced_stub', balanced_pid, 'balanced'),
                                ('balanced_child', balanced_child_pid, 'balanced'),
                                ('llm_stub', llm_pid, 'llm'), ('llm_child', llm_child_pid, 'llm')]:
            processes[name] = adopt(pid, role, root)
        check_parent(processes['balanced_child'].identity, balanced_pid)
        check_parent(processes['llm_child'].identity, llm_pid)
        status.update(stage='running', stop_after_steps=60, stop_at_second=54000,
                      pending_arms=[], expected_arms=['reference', 'balanced'],
                      comparison_root=str(output), original_full_day_finished=False,
                      scheduler_handover='Committed checkpoint60 stop monitor active',
                      stop_monitor_pid=os.getpid())
        heartbeat = 0
        while not cutoff_ready(root):
            if any(processes[n].poll() is not None for n in ('balanced_stub', 'balanced_child')):
                raise RuntimeError('Model ended before the requested checkpoint')
            if time.monotonic() - heartbeat >= 5:
                status['updated_at'] = time.time()
                save(root / 'status.json', status)
                heartbeat = time.monotonic()
            time.sleep(poll_seconds)
        receipt = stop_balanced(root, processes)
        save(root / 'stop60_receipt.json', receipt)
        status['jobs']['balanced'].update(exit_code=receipt['processes']['balanced_stub']['exit_code'],
            child_exit_code=receipt['processes']['balanced_child']['exit_code'],
            completion_reason=receipt['reason'])
        status.update(stage='stopped_after_60', updated_at=time.time(), stop_receipt=receipt)
        save(root / 'status.json', status)
        deadline = time.monotonic() + 480
        while True:
            coverage = explanation_coverage(root)
            if not coverage['pending']:
                break
            if time.monotonic() >= deadline or processes['llm_child'].poll() is not None:
                raise RuntimeError('Committed explanation coverage did not finish')
            time.sleep(1)
        completion = finish_explanations(processes, coverage)
        status['jobs']['explanations'].update(exit_code=completion['llm_stub']['exit_code'],
            completion=completion, completion_reason='user_scope_completion')
        save(root / 'logs/explanations.status.json', dict(
            ended=True, expected_arms=['reference', 'balanced'], stop_after_steps=60,
            facts=coverage['facts'], valid=coverage['valid'], completion=completion,
            completion_reason='user_scope_completion', full_day_finished=False))
        from scripts.metro52_partial import materialize
        materialize(root, output, steps=60)
        status.update(stage='reporting', updated_at=time.time())
        save(root / 'status.json', status)
        subprocess.run([sys.executable, str(ROOT / 'metro51_forecast_audit.py'), '--root', str(output)], cwd=ROOT, check=True)
        run_incidents(output)
        subprocess.run([sys.executable, str(ROOT / 'metro51_report.py'), str(output),
                        '--output', str(output / 'review'), '--incidents', str(output / 'incidents')], cwd=ROOT, check=True)
        partial_status = read(output / 'status.json')
        partial_status.update(stage='completed_partial', finished_at=time.time(), source_stop_receipt=receipt)
        save(output / 'status.json', partial_status)
        status.update(stage='completed_partial', finished_at=time.time())
        save(root / 'status.json', status)
        return status
    except Exception as error:
        status.update(stage='failed', error=type(error).__name__, failed_at=time.time())
        save(root / 'status.json', status)
        raise
    finally:
        for process in processes.values():
            process.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    for name in ('balanced-pid', 'balanced-child-pid', 'llm-pid', 'llm-child-pid'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.balanced_pid, args.balanced_child_pid, args.llm_pid, args.llm_child_pid)
