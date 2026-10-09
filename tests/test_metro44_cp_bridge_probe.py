import copy

import pytest

import metro44_cp_bridge_probe as probe


def test_same_profile_and_existing_chains_are_passed_without_new_releases():
    data={'plans':[{'trips':[]}],'marker':'original'}
    profile=[dict(terminal='L1_01',start_second=6300,end_second=7200,headway_seconds=160),
             dict(terminal='L1_19',start_second=6300,end_second=7200,headway_seconds=150)]
    original=copy.deepcopy((data,profile));calls=[]
    def build(candidate,second,end,**kwargs):
        calls.append((copy.deepcopy(candidate),second,end,copy.deepcopy(kwargs)))
        candidate['marker']='changed'
        return candidate,dict(movement_passed=True)
    def gate(changed,second,end,**kwargs):
        assert kwargs['reference']==original[0]
        return dict(movement_passed=True,obligations_closed=True)
    result=probe.compile_profile(data,profile,second=6300,planning_end=22200,
                                  deadline=10**12,builder=build,gate=gate)
    assert (data,profile)==original
    assert calls[0][1:3]==(6300,22200)
    assert calls[0][3]==dict(headway=profile,bridge_seconds=480,phase_seconds=0,
                           wear_priority=False,deadline=10**12)
    assert result['status']=='physical_passed' and result['candidate']['marker']=='changed'
    assert result['profile_unchanged']


def test_independent_physical_failure_hides_candidate_despite_builder_claim():
    result=probe.compile_profile({},[{'headway_seconds':160}],second=6300,planning_end=22200,
        deadline=10**12,builder=lambda *a,**k:({'changed':True},{'movement_passed':True}),
        gate=lambda *a,**k:dict(movement_passed=False,terminal_total=3))
    assert result['status']=='physical_rejected' and result['candidate']is None
    assert not result['audit']['movement_passed']


def test_constructor_failure_is_reported_without_calling_gate():
    def build(*args,**kwargs):raise ValueError('fixed chain conflict')
    def gate(*args,**kwargs):raise AssertionError('must not validate nonexistent candidate')
    result=probe.compile_profile({},[],second=6300,planning_end=22200,
        deadline=10**12,builder=build,gate=gate)
    assert result['status']=='constructor_rejected' and result['reason']=='fixed chain conflict'
    assert result['candidate']is None


def test_expired_outer_deadline_does_not_start_compilation(monkeypatch):
    monkeypatch.setattr(probe,'monotonic',lambda:470)
    def build(*args,**kwargs):raise AssertionError('expired work must not start')
    with pytest.raises(TimeoutError):
        probe.compile_profile({},[],second=6300,planning_end=22200,deadline=470,builder=build)


def test_late_compilation_does_not_deliver_or_validate_a_candidate(monkeypatch):
    clock=[0]
    monkeypatch.setattr(probe,'monotonic',lambda:clock[0])
    def build(*args,**kwargs):
        clock[0]=471
        return {},{}
    def gate(*args,**kwargs):raise AssertionError('late candidate must not proceed')
    with pytest.raises(TimeoutError):
        probe.compile_profile({},[],second=6300,planning_end=22200,deadline=470,builder=build,gate=gate)
