import copy
from time import monotonic

import pytest

from goal43_policy import load_policy
from goal43_replay import build_operational_day
from metro44_chain_repair import repair_chain_timetable
from metro44_assignment_plan import _current_movements_preserved,terminal_resources
from simulation import simulate


def test_valid_known_plan_is_not_rewritten_and_all_physics_are_checked():
    source=build_operational_day(load_policy(),count=53,start_at='2026-09-30T08:15:00+03:00',service_hours=4)[0]
    before=copy.deepcopy(source)
    changed,audit=repair_chain_timetable(source,6300,14400,deadline=monotonic()+90)
    assert source==before and changed==source
    assert audit['movement_passed'] and not audit['standby_repairs']and not audit['max_gap_repairs']
    motion=simulate(changed,changed['plans'][0]['plan_id'])
    assert terminal_resources(changed,motion)['passed']and motion['obligations_closed']
    assert _current_movements_preserved(source,changed,6300)
    assert audit['peak_trains_on_line']<=53 and audit['maximum_headway']<=480


def test_expired_deadline_prevents_repair():
    with pytest.raises(TimeoutError):repair_chain_timetable({},3600,65400,deadline=monotonic()-1)


@pytest.mark.parametrize('limit',[True,-1,6,1.5])
def test_repair_iterations_are_bounded(limit):
    with pytest.raises(ValueError):repair_chain_timetable({},3600,65400,max_gap_repairs=limit)
