"""Try one peak phase with the existing dispatcher's exact physical checks."""

import copy
from types import FunctionType


def replace_peak_phases(calendars, peak_start_second, phases):
    if set(phases) != set(calendars):
        raise ValueError('phases must name every terminal')
    out = copy.deepcopy(calendars)
    for terminal, phase in phases.items():
        rows = [r for r in out[terminal] if r['start_second'] == peak_start_second]
        if len(rows) != 1:
            raise ValueError('selected peak must exist exactly once per terminal')
        if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase < rows[0]['headway_seconds']:
            raise ValueError('phase must fit the peak headway')
        rows[0]['phase_seconds'] = phase
    return out


def rebuild_with_peak_phase(data, as_of, service_end, phases, *, peak_start_second,
                            headway, deadline=None, builder=None):
    if builder is None:
        from metro44_planner_wait import rebuild_wait_timetable
        builder = rebuild_wait_timetable
    context = dict(builder.__globals__)
    calendar_builder = context['_peak_calendars']

    def calendars(*args, **kwargs):
        return replace_peak_phases(calendar_builder(*args, **kwargs), peak_start_second, phases)

    context['_peak_calendars'] = calendars
    worker = FunctionType(builder.__code__, context, builder.__name__, builder.__defaults__, builder.__closure__)
    worker.__kwdefaults__ = copy.copy(builder.__kwdefaults__)
    return worker(data, as_of, service_end, headway=headway, deadline=deadline)
