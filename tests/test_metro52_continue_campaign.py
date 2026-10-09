import ctypes
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/metro52_continue_campaign.py'
spec = importlib.util.spec_from_file_location('continue_campaign_tested', PATH)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_scope_accepts_bom_and_never_requeues_skipped_models(tmp_path):
    scope = dict(expected_arms=['reference', 'balanced'], skipped_arms=['old', 'ensemble_old'])
    (tmp_path/'campaign_scope.json').write_text(json.dumps(scope), encoding='utf-8-sig')
    assert continuation.load_scope(tmp_path) == scope
    scope['expected_arms'].append('old')
    write(tmp_path/'campaign_scope.json', scope)
    with pytest.raises(ValueError):
        continuation.load_scope(tmp_path)


class Kernel:
    def __init__(self):
        self.rights = []
        self.running = True
        self.code = 0
        self.creation = 123
        self.closed = []

    def OpenProcess(self, rights, inherit, pid):
        self.rights.append(rights)
        return len(self.rights)

    def GetProcessTimes(self, handle, created, exited, kernel, user):
        created._obj.dwLowDateTime = self.creation
        return 1

    def WaitForSingleObject(self, handle, timeout):
        return 258 if self.running else 0

    def GetExitCodeProcess(self, handle, code):
        code._obj.value = self.code
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1

    def TerminateProcess(self, handle, code):
        self.running = False
        self.code = code
        return 1


def test_handle_poll_uses_wait_state_and_actual_exit_code():
    kernel = Kernel()
    process = continuation.WindowsProcess(17, kernel=kernel)
    assert kernel.rights == [0x101000]
    assert process.poll() is None
    kernel.running = False
    kernel.code = 259
    assert process.poll() == 259
    process.close()
    assert kernel.closed


def test_termination_rejects_reused_pid_without_touching_process():
    kernel = Kernel()
    process = continuation.WindowsProcess(17, kernel=kernel)
    kernel.creation = 124
    with pytest.raises(ValueError, match='identity'):
        process.terminate()
    assert kernel.running


def test_adoption_requires_same_creation_command_and_real_parent(tmp_path):
    record = dict(pid=17, parent_pid=10, created='original',
                  command=f'python {continuation.ROOT}/metro51_run.py --output {tmp_path}/balanced --arm BALANCED')
    changed = dict(record, created='replacement')
    snapshots = iter([{17: record}, {17: changed}])
    with pytest.raises(ValueError, match='identity'):
        continuation.adopt(17, 'balanced', tmp_path, reader=lambda p: next(snapshots),
                           factory=lambda pid: continuation.WindowsProcess(pid, kernel=Kernel()))
    with pytest.raises(ValueError, match='parent'):
        continuation.check_parent(dict(parent_pid=99), 10)


class Process:
    def __init__(self, code=None):
        self.code = code
        self.terminations = 0
        self.identity = {}

    def poll(self):
        return self.code

    def terminate(self):
        self.terminations += 1
        self.code = 125
        return 125

    def close(self):
        pass


def test_completed_model_needs_both_zero_exits_and_full_result(tmp_path):
    processes = {'balanced_stub': Process(0), 'balanced_child': Process(0)}
    with pytest.raises(FileNotFoundError):
        continuation.require_model_result(tmp_path, processes)
    write(tmp_path/'balanced/result.json', {'contract':{'arm':'BALANCED','stage':'full_day'},'decisions':[]})
    assert continuation.require_model_result(tmp_path, processes)['contract']['arm'] == 'BALANCED'
    processes['balanced_child'].code = 7
    with pytest.raises(RuntimeError):
        continuation.require_model_result(tmp_path, processes)


def test_disabled_or_invalid_explanation_keeps_live_watcher(tmp_path):
    packet = tmp_path/'balanced/explanations/facts_00900.json'
    write(packet, {'decision_id':'D'})
    answer = packet.with_name('explanation_00900.json')
    write(answer, {'metadata':{'error_code':'disabled'}})
    assert continuation.explanation_coverage(tmp_path, validator=lambda a,b:None)['pending']
    write(answer, {'metadata':{'error_code':'invalid_response'}})
    def reject(a,b):
        raise ValueError('bad context')
    assert continuation.explanation_coverage(tmp_path, validator=reject)['pending']
    assert not continuation.explanation_coverage(tmp_path, validator=lambda a,b:None)['pending']


def test_scope_stop_requires_valid_coverage_and_records_real_termination():
    processes = {'llm_stub':Process(), 'llm_child':Process()}
    with pytest.raises(ValueError):
        continuation.finish_explanations(processes, {'pending':[{}]}, sleeper=lambda _:None)
    assert all(p.terminations == 0 for p in processes.values())
    receipts = continuation.finish_explanations(processes, {'pending':[]}, sleeper=lambda _:None)
    assert receipts['llm_child']['exit_code'] == 125
    assert receipts['llm_child']['reason'] == 'user_scope_completion'
    assert receipts['llm_stub']['exit_code'] == 125


def test_postprocess_uses_existing_reference_incident_runner(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(continuation, 'run_incidents', lambda root:calls.append(('incidents',root)))
    continuation.postprocess(tmp_path, execute=lambda args, **kwargs:calls.append((args,kwargs)))
    assert ('incidents', tmp_path) in calls
    commands = [args for args,kwargs in calls if isinstance(args,list)]
    assert len(commands) == 2
    assert all('metro51_run.py' not in ' '.join(command) for command in commands)
    assert '--root' in commands[0]
    assert '--incidents' in commands[1]


def test_continuation_preserves_results_and_waits_for_model_before_scoped_completion(tmp_path, monkeypatch):
    write(tmp_path/'campaign_scope.json', dict(expected_arms=['reference','balanced'], skipped_arms=['old','ensemble_old']))
    write(tmp_path/'status.json', {'jobs':{'reference':{'exit_code':0},'balanced':{},'explanations':{}},'pending_arms':['old']})
    write(tmp_path/'reference/result.json', {'contract':{'arm':'REFERENCE','stage':'full_day'}})
    write(tmp_path/'balanced/result.json', {'contract':{'arm':'BALANCED','stage':'full_day'},'decisions':[{'decision_second':900}]})
    write(tmp_path/'balanced/explanations/facts_00900.json', {'decision_id':'D'})
    write(tmp_path/'balanced/explanations/explanation_00900.json', {'metadata':{'error_code':'invalid_response'}})
    before = {arm:(tmp_path/arm/'result.json').read_bytes() for arm in ['reference','balanced']}
    ticks, objects, posted = [0], {}, []
    def adopted(pid, role, root):
        process=Process()
        process.identity={'parent_pid':10 if pid==11 else 20 if pid==21 else 1}
        if role=='balanced':
            process.poll=lambda:0 if ticks[0] else None
        objects[pid]=process
        return process
    def sleep(_):
        assert objects[21].terminations==0 or ticks[0]>0
        ticks[0]+=1
    coverage=continuation.explanation_coverage
    monkeypatch.setattr(continuation,'adopt',adopted)
    monkeypatch.setattr(continuation,'explanation_coverage',lambda root, result=None:coverage(root,result=result,validator=lambda a,b:None))
    monkeypatch.setattr(continuation.time,'sleep',sleep)
    monkeypatch.setattr(continuation,'postprocess',lambda root:posted.append(root))
    result=continuation.run(tmp_path,10,11,20,21,poll_seconds=.01)
    assert result['stage']=='completed'
    assert result['pending_arms']==[]
    assert result['expected_arms']==['reference','balanced']
    assert result['jobs']['explanations']['completion']['llm_child']['exit_code']==125
    assert result['jobs']['explanations']['exit_code']==125
    receipt=continuation.read(tmp_path/'logs/explanations.status.json')
    assert receipt['all_simulations_finished_for_selected_scope'] is True
    assert receipt['all_original_arms_finished'] is False
    assert receipt['completion_reason']=='user_scope_completion'
    assert posted==[tmp_path]
    assert all((tmp_path/arm/'result.json').read_bytes()==raw for arm,raw in before.items())
