import pytest

from metro48_experiment import decision_times, calculate, run


def test_incident_triggers_immediate_decision_between_quarters():
    assert decision_times(3600,6300,event_times=[4013,4029]) == [3600,4013,4029,4500,5400]
    with pytest.raises(ValueError):
        decision_times(3600,6300,event_times=[3599])


def test_solver_and_exact_trials_share_guarded_tail_and_return_bound():
    calls = []; raw = {'raw':True}; tail = {'tail':True}; guarded = {'guarded':True}
    data = {'period_start':'2026-09-30T08:15:00+03:00','stations':[{'station_id':'L1_01','kind':'line'}]}
    def extend(bundle,*args):
        assert bundle is raw
        calls.append(('tail',args[-1])); return tail
    def guard(bundle,at):
        assert bundle is tail
        return guarded
    def solve(*args,**kwargs):
        assert args[1] is guarded and kwargs['return_by_second'] == 21600
        return {'selected_train_ids':['B001']}
    def catalog(*args,**kwargs):
        assert args[5] is guarded
        assert kwargs['proposal']['return_by_second'] == 21600
        return {'decision':{'accepted':True},'candidates':[]}
    def refine(solver,evaluate,**kwargs):
        return evaluate(solver()['selected_train_ids'])
    result = calculate(data,'runner','state','profiles','roster',raw,'frame',3600,22200,
        100,None,mode='temporary',extend=extend,guard=guard,solver=solve,catalog=catalog,
        refine=refine,builder='builder')
    assert result['decision']['accepted'] is True
    assert calls == [('tail','2026-09-30T14:15:00+03:00')]


def test_existing_run_and_invalid_arm_are_rejected_before_writes(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path,arm='GRU')
    with pytest.raises(ValueError):
        run(tmp_path/'new',arm='ORACLE')
    assert not (tmp_path/'new').exists()
