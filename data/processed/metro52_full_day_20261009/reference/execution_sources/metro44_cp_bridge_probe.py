"""Compile one frozen CP proposal through the existing vehicle chains."""

import copy
from time import monotonic

from metro44_catalog import physical_gate
from metro44_planner_bridge import rebuild_bridge_timetable


def compile_profile(data, profile, *, second, planning_end, deadline, builder=None, gate=None):
    def check():
        if monotonic()>=deadline:raise TimeoutError('CP bridge probe deadline')
    check()
    build=builder or rebuild_bridge_timetable
    validate=gate or physical_gate
    try:
        changed,audit=build(copy.deepcopy(data),second,planning_end,headway=copy.deepcopy(profile),
            bridge_seconds=480,phase_seconds=0,wear_priority=False,deadline=deadline)
    except ValueError as error:
        check()
        return dict(status='constructor_rejected',candidate=None,reason=str(error),profile_unchanged=True)
    check()
    audit={**audit,**validate(changed,second,planning_end,reference=data,deadline=deadline)}
    check()
    passed=audit.get('movement_passed')is True
    return dict(status='physical_passed'if passed else'physical_rejected',
                candidate=changed if passed else None,audit=audit,profile_unchanged=True)
