import json
from pathlib import Path

import pytest

from scripts import metro52_stop60 as stop


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'running'
    write(root / 'balanced/contract.json', dict(control_start_second=900))
    write(root / 'campaign_scope.json', dict(stop_after_steps=60, stop_at_second=54000,
        comparison_root=str(tmp_path / 'paired'), expected_arms=['reference', 'balanced']))
    for name in ('controller/decision_54000.json', 'controller/resume_54000.json.gz',
                 'explanations/facts_54000.json', 'explanations/explanation_54000.json'):
        write(root / 'balanced' / name, dict(decision_second=54000))
    return root


def commit(root):
    (root / 'balanced/events.jsonl').write_text(json.dumps(dict(
        stage='decision_completed', second=54000, index=60)) + '\n', encoding='utf8')


def test_checkpoint_alone_cannot_trigger_stop_before_fact_packet(campaign):
    assert not stop.cutoff_ready(campaign)
    commit(campaign)
    assert stop.cutoff_ready(campaign)
    (campaign / 'balanced/explanations/facts_54000.json').unlink()
    assert not stop.cutoff_ready(campaign)


def test_incomplete_or_other_journal_events_do_not_trigger_stop(campaign):
    events = campaign / 'balanced/events.jsonl'
    events.write_text('{"stage":"decision_started","second":54000}\n{')
    assert not stop.cutoff_ready(campaign)
    commit(campaign)
    (campaign / 'balanced/controller/decision_54000.json').write_text('{')
    assert not stop.cutoff_ready(campaign)


class Process:
    def __init__(self):
        self.code = None
        self.signals = 0

    def poll(self):
        return self.code

    def terminate(self):
        self.signals += 1
        self.code = 125
        return 125


def test_only_balanced_processes_stop_after_committed_milestone(campaign):
    processes = {name: Process() for name in ('balanced_child', 'balanced_stub', 'llm_child', 'llm_stub')}
    with pytest.raises(ValueError):
        stop.stop_balanced(campaign, processes, sleeper=lambda _: None)
    assert all(p.signals == 0 for p in processes.values())
    commit(campaign)
    receipt = stop.stop_balanced(campaign, processes, sleeper=lambda _: None)
    assert receipt['reason'] == 'user_requested_stop_after_60'
    assert receipt['cutoff_second'] == 54000
    assert processes['balanced_child'].signals == processes['balanced_stub'].signals == 1
    assert processes['llm_child'].signals == processes['llm_stub'].signals == 0
    assert receipt['processes']['balanced_child']['exit_code'] == 125


def test_scope_rejects_other_stop_target_and_unsafe_output(campaign):
    assert stop.selected_output(campaign) == campaign.parent / 'paired'
    scope = json.loads((campaign / 'campaign_scope.json').read_text())
    scope['stop_after_steps'] = 59
    write(campaign / 'campaign_scope.json', scope)
    with pytest.raises(ValueError):
        stop.selected_output(campaign)
    scope['stop_after_steps'] = 60
    scope['comparison_root'] = str(campaign / 'nested')
    write(campaign / 'campaign_scope.json', scope)
    with pytest.raises(ValueError):
        stop.selected_output(campaign)


def test_run_adopts_workers_preserves_stop_codes_and_original_contracts(campaign, monkeypatch):
    import subprocess
    from scripts import metro52_partial

    commit(campaign)
    reference_contract = campaign/'reference/contract.json'
    write(reference_contract,dict(arm='REFERENCE',stage='full_day',full_day=True,control_start_second=900))
    original = {p:p.read_bytes() for p in [reference_contract,campaign/'balanced/contract.json']}
    write(campaign/'status.json',dict(stage='running',jobs={
        'reference':{'exit_code':0},'balanced':{'exit_code':None},'explanations':{'exit_code':None}}))
    actions,processes = [],{}

    class Handle(Process):
        def __init__(self,pid,parent):
            super().__init__()
            self.identity = dict(pid=pid,parent_pid=parent)
            self.closed = False

        def terminate(self):
            actions.append(('stop',self.identity['pid']))
            code = super().terminate()
            if self.identity['pid'] in [2,4]:
                processes[self.identity['parent_pid']].code = code
            return code

        def close(self):
            self.closed = True

    processes.update({pid:Handle(pid,parent) for pid,parent in [(1,999),(2,1),(3,999),(4,3)]})
    def adopt(pid,role,root):
        assert root == campaign
        actions.append(('adopt',pid,role))
        return processes[pid]
    monkeypatch.setattr(stop,'adopt',adopt)
    monkeypatch.setattr(stop.time,'sleep',lambda _:None)
    checks = []
    def coverage(root):
        assert processes[2].poll() == 125
        assert processes[4].poll() is None
        checks.append(len(checks))
        return dict(facts=60,valid=59 if len(checks)==1 else 60,
                    pending=[{'reason':'disabled'}] if len(checks)==1 else [])
    monkeypatch.setattr(stop,'explanation_coverage',coverage)

    output = campaign.parent/'paired'
    def materialize(source,target,*,steps):
        assert source == campaign and target == output and steps == 60
        status = json.loads((source/'status.json').read_text())
        assert status['jobs']['balanced']['exit_code'] == status['jobs']['balanced']['child_exit_code'] == 125
        assert all(p.read_bytes()==raw for p,raw in original.items())
        assert processes[4].poll() == 125 and len(checks)==2
        actions.append(('materialize',steps))
        write(target/'status.json',dict(stage='materialized_partial',full_day=False,stop_after_steps=60))
    monkeypatch.setattr(metro52_partial,'materialize',materialize)

    def execute(command,**kwargs):
        assert command[1] in [str(stop.ROOT/'metro51_forecast_audit.py'),str(stop.ROOT/'metro51_report.py')]
        assert kwargs['check'] is True
        assert str(output) in command
        actions.append(('report',Path(command[1]).name))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(stop.subprocess,'run',execute)
    def no_spawn(*args,**kwargs):
        pytest.fail('Stop monitor must not spawn a model arm')
    monkeypatch.setattr(stop.subprocess,'Popen',no_spawn)
    monkeypatch.setattr(stop,'run_incidents',lambda root:actions.append(('incidents',str(root))))

    status = stop.run(campaign,1,2,3,4,poll_seconds=0)
    assert status['stage'] == 'completed_partial'
    assert status['jobs']['balanced']['exit_code'] == status['jobs']['balanced']['child_exit_code'] == 125
    assert status['jobs']['balanced']['completion_reason'] == 'user_requested_stop_after_60'
    assert status['jobs']['explanations']['exit_code'] == 125
    assert status['original_full_day_finished'] is False
    assert json.loads((output/'status.json').read_text())['stage'] == 'completed_partial'
    assert all(p.read_bytes()==raw for p,raw in original.items())
    assert all(p.closed for p in processes.values())
    assert [a for a in actions if a[0]=='adopt'] == [('adopt',1,'balanced'),('adopt',2,'balanced'),('adopt',3,'llm'),('adopt',4,'llm')]
    assert [a for a in actions if a[0]=='stop'] == [('stop',2),('stop',4)]
    assert [a for a in actions if a[0]=='report'] == [('report','metro51_forecast_audit.py'),('report','metro51_report.py')]
