"""Continuous reference from the real initial scenario, with advance preparation."""

import math
from datetime import datetime, timedelta
from time import monotonic

from goal43_policy import minimum_headway, parameter
from metro_daily import _geometry, reference_counts
from metro44_assignment_plan import _current_movements_preserved, terminal_resources
from metro44_chain_repair import repair_chain_timetable
from metro44_linked_gap_repair import release_constraints
from metro44_planner_calendar import rebuild_calendar_timetable
from rolling import operational_headways
from simulation import digest, simulate


def preparation_profile(data, service_end):
    """Source clock counts, with a full vehicle-cycle preparation lead before peaks.

    This is a declared reference strategy, not a fitted economic optimum. Preparing
    for a peak must precede its first quarter; future passenger labels are excluded.
    """
    _, _, _, half = _geometry(data)
    policy = data['headway_policy']
    origin = datetime.fromisoformat(data['period_start'])
    maximum = parameter(policy, 'headway_max_s')
    preparation = 2 * half + max(parameter(policy, 'depot_release_s').values())
    counts = reference_counts()
    rows = []
    for second in range(0, service_end, 900):
        at = origin + timedelta(seconds=second)
        minimum = minimum_headway(at, policy)
        peak = minimum == 113
        upcoming = any(minimum_headway(at + timedelta(seconds=offset), policy) == 113
                       for offset in range(0, preparation + 900, 900))
        count = parameter(policy, 'line_train_limit') if peak else counts[at.hour]
        target = math.ceil(2 * half / count)
        if upcoming and not peak:
            target = minimum
        rows.append(dict(start_second=second, end_second=min(second+900, service_end),
                         headway_seconds=min(maximum, max(minimum, target))))
    return rows


def validate_readiness(data, roster):
    origin = datetime.fromisoformat(data['period_start'])
    ready = {r['train_id']: r for r in roster}
    if set(ready) != {t['train_id'] for t in data['trains']}:
        raise ValueError('explicit same-fleet readiness required')
    for train in data['trains']:
        if not train['location'].startswith('DEPOT_'):
            continue
        row = ready[train['train_id']]
        if (not row.get('ready_at') or row.get('readiness_status') in (None, 'unknown')
                or datetime.fromisoformat(row['ready_at']) > origin + timedelta(seconds=train['ready_after_seconds'])):
            raise ValueError('readiness must be explicit and agree with the source')


def build_reference(data, roster, service_end, *, deadline=None):
    def budget():
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError('continuous reference deadline')
    budget()
    validate_readiness(data, roster)
    profile = preparation_profile(data, service_end)
    changed, constructor = rebuild_calendar_timetable(data, 0, service_end,
        headway=profile, initial=True, wear_priority=True, deadline=deadline)
    if not constructor['movement_passed']:
        changed, repair = repair_chain_timetable(changed, 0, service_end,
            standby_repairs=5, max_gap_repairs=5, deadline=deadline)
    else:
        repair = None
    budget()
    motion = simulate(changed, changed['plans'][0]['plan_id'])
    headways = operational_headways(changed, motion, 0, service_end)
    resources = terminal_resources(changed, motion)
    release = release_constraints(changed, 0)
    current = _current_movements_preserved(data, changed, 0)
    if changed['trains'] != data['trains'] or changed['tracks'] != data['tracks'] or not current:
        raise ValueError('continuous reference changes initial geometry or current movements')
    audit = dict(movement_passed=headways['passed'] and resources['passed'] and release['passed']
        and motion['obligations_closed'], initial_fleet_preserved=True, current_movements_preserved=current,
        initial_second=0, source_clock_counts=reference_counts(), preparation_lead_seconds=
        2*_geometry(data)[3]+max(parameter(data['headway_policy'],'depot_release_s').values()),
        profile=profile, constructor=constructor, repair=repair,
        headways={k:v for k,v in headways.items() if k not in ('intervals','boundary_ages')},
        terminal_resources=resources, releases=release,
        peak_trains_on_line=motion['checks']['peak_trains_on_line'], obligations_closed=motion['obligations_closed'],
        passenger_validation_required=True, reference_cost_is_optimal=False, cash_saving_rub=None,
        source_digest=digest(data), result_digest=digest(changed),
        scope='Line 1 continuous reference from initial 33 and same explicitly assumed 68-train readiness')
    budget()
    return changed, audit
