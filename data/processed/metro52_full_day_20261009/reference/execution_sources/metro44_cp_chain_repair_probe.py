"""One bounded repair of a compiled CP chain; preserve the original bridge."""

import copy
from time import monotonic

from metro44_catalog import physical_gate
from metro44_chain_repair import repair_chain_timetable


def chains(data):
    return {t['trip_id']:(t['train_id'],tuple(t['route']),t['passenger'])for t in data['plans'][0]['trips']}


def bridge_starts(data,second):
    starts={(r['trip_id'],r['leg_index']):r['departure_second']for r in data['leg_timing']}
    return {t['trip_id']:(t['train_id'],tuple(t['route']),starts[t['trip_id'],0])
            for t in data['plans'][0]['trips']if t['passenger']and second<=starts[t['trip_id'],0]<second+480}


def repair_profile(original,candidate,*,second,planning_end,deadline,repair=None,gate=None):
    def check():
        if monotonic()>=deadline:raise TimeoutError('CP chain repair deadline')
    check()
    try:
        changed,audit=(repair or repair_chain_timetable)(copy.deepcopy(candidate),second,planning_end,
            standby_repairs=5,max_gap_repairs=5,deadline=deadline)
    except ValueError as error:
        check()
        return dict(status='repair_rejected',candidate=None,reason=str(error))
    check()
    same_bridge=bridge_starts(original,second)==bridge_starts(changed,second)
    same_chains=chains(candidate)==chains(changed)
    audit={**audit,**(gate or physical_gate)(changed,second,planning_end,reference=original,deadline=deadline)}
    check()
    passed=audit.get('movement_passed')is True and same_bridge and same_chains
    return dict(status='physical_passed'if passed else'physical_rejected',candidate=changed if passed else None,
                bridge_preserved=same_bridge,chains_preserved=same_chains,audit=audit)
