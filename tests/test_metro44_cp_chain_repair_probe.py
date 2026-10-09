import copy
import json
from pathlib import Path

import pytest

import metro44_cp_chain_repair_probe as probe


def data(at=6350):
    return dict(plans=[dict(trips=[dict(trip_id='A',train_id='T',route=['P'],passenger=True)])],
                leg_timing=[dict(trip_id='A',train_id='T',leg_index=0,departure_second=at)])


def test_repair_is_one_bounded_call_and_preserves_inputs_and_bridge():
    original=data();candidate=data();before=copy.deepcopy((original,candidate));calls=[]
    def repair(value,second,end,**kwargs):
        calls.append((second,end,kwargs));return value,dict(movement_passed=True)
    result=probe.repair_profile(original,candidate,second=6300,planning_end=22200,deadline=10**12,
        repair=repair,gate=lambda *a,**k:dict(movement_passed=True))
    assert (original,candidate)==before and len(calls)==1
    assert calls[0]==(6300,22200,dict(standby_repairs=5,max_gap_repairs=5,deadline=10**12))
    assert result['status']=='physical_passed' and result['bridge_preserved'] and result['chains_preserved']


def test_changed_nearby_departure_rejects_even_when_physical_gate_claims_pass():
    def repair(value,*args,**kwargs):
        value['leg_timing'][0]['departure_second']+=1
        return value,dict(movement_passed=True)
    result=probe.repair_profile(data(),data(),second=6300,planning_end=22200,deadline=10**12,
        repair=repair,gate=lambda *a,**k:dict(movement_passed=True))
    assert result['status']=='physical_rejected' and not result['bridge_preserved']
    assert result['candidate']is None


def test_changed_train_assignment_rejects_even_when_physical_gate_claims_pass():
    def repair(value,*args,**kwargs):
        value['plans'][0]['trips'][0]['train_id']='NEW'
        return value,dict(movement_passed=True)
    result=probe.repair_profile(data(),data(),second=6300,planning_end=22200,deadline=10**12,
        repair=repair,gate=lambda *a,**k:dict(movement_passed=True))
    assert result['status']=='physical_rejected' and not result['chains_preserved']


def test_repair_refusal_is_terminal_and_does_not_retry():
    calls=[]
    def repair(*args,**kwargs):
        calls.append(1);raise ValueError('started approach cannot be repaired')
    result=probe.repair_profile(data(),data(),second=6300,planning_end=22200,deadline=10**12,repair=repair)
    assert len(calls)==1 and result['candidate']is None
    assert result['reason']=='started approach cannot be repaired'


def test_expired_deadline_does_not_start_or_deliver_repair(monkeypatch):
    monkeypatch.setattr(probe,'monotonic',lambda:470)
    def repair(*args,**kwargs):raise AssertionError('must not run')
    with pytest.raises(TimeoutError):
        probe.repair_profile(data(),data(),second=6300,planning_end=22200,deadline=470,repair=repair)


def test_saved_cp_conflict_is_reproduced_with_same_past_and_nearby_starts():
    from metro44_catalog import physical_gate
    from metro44_planner_bridge import rebuild_bridge_timetable
    root=Path(__file__).resolve().parent.parent
    source=root/'data/processed/metro44_continuation_6h_closed_20261009/runs/CONT6H_CLOSED_KEEP_S1_7/movement_input.json'
    profile=root/'data/processed/metro44_catalog_cp30_incumbent_20261009/directional_profile.json'
    original=json.loads(source.read_text());before=copy.deepcopy(original)
    changed,audit=rebuild_bridge_timetable(original,6300,22200,headway=json.loads(profile.read_text()),bridge_seconds=480)
    checked=physical_gate(changed,6300,22200,reference=original)
    assert original==before and checked['executed_prefix_preserved'] and checked['current_movements_preserved']
    assert probe.bridge_starts(original,6300)==probe.bridge_starts(changed,6300)
    assert checked['headways_passed'] and checked['obligations_closed'] and checked['peak_trains_on_line']==53
    assert checked['terminal_resources']['maximum_standby_by_terminal']=={'L1_01':2,'L1_19':2}
    assert not checked['movement_passed']
