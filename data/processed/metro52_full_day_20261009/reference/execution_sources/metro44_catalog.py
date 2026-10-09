"""Small causal catalogue; exact terminal and passenger gates precede selection."""

import copy
from datetime import datetime, timedelta
from time import monotonic

from forecasting import validate_bundle
from goal43_policy import minimum_headway, parameter
from metro_daily_run import read_forecast
from metro44_assignment_plan import _current_movements_preserved, terminal_resources
from metro44_mpc import solve_joint_profile
from metro44_planner import headway_seconds
from metro44_planner_bridge import rebuild_bridge_timetable
from metro44_planner_noidle import terminal_total_audit
from metro44_planner_terminal import rebuild_terminal_timetable
from metro44_run import (_slots, closed_per_train, directional_profile,
                         estimated_keep_wait_budget, keep_wait_budgets)
from metro44_selection import choose_candidate
from rolling import decide_operational, operational_headways
from simulation import digest, simulate


def _check(deadline, cancelled=None):
    if monotonic() >= deadline or cancelled is not None and cancelled.is_set():
        raise TimeoutError('catalog decision deadline')


def _mode(strategy):
    modes = {'MPC_ML': 'ML', 'ML': 'ML', 'MPC_BASELINE_FORECAST': 'MEDIAN',
             'BASELINE_FORECAST': 'MEDIAN', 'STANDARD': 'MEDIAN'}
    if strategy not in modes:
        raise ValueError('catalog supports causal ML or BASELINE_FORECAST only')
    return modes[strategy]


def _policy(mode):
    if mode not in ('strict', 'pareto'):
        raise ValueError('unknown quality policy')
    return dict(mode=mode, user_requirement=False, posthoc_development=mode == 'pareto',
                mean_seconds=90 if mode == 'pareto' else None,
                p95_seconds=180 if mode == 'pareto' else None,
                scope='same forecast KEEP mean/p95' if mode == 'strict' else 'assumed development tradeoff')


def terminal_gate(audit):
    total = audit.get('maximum_total_by_terminal', audit.get('maximum_visits_by_terminal', {}))
    standby = audit.get('maximum_standby_by_terminal', {})
    return (audit.get('passed') is True and bool(total) and set(total) == set(standby)
            and all(0 <= n <= 2 for n in total.values()) and all(0 <= n <= 1 for n in standby.values()))


def physical_gate(data, second, service_end, *, reference=None, reference_motion=None, deadline=float('inf')):
    """Check the complete movement plan independently of a builder's claims."""
    _check(deadline)
    motion = simulate(data, data['plans'][0]['plan_id'])
    _check(deadline)
    total, visits = terminal_total_audit(data, motion), terminal_resources(data, motion)
    gaps = operational_headways(data, motion, second, service_end)
    preserved = current = True
    if reference is not None:
        before = reference_motion if reference_motion else simulate(reference, reference['plans'][0]['plan_id'])
        preserved = ([e for e in before['events'] if e['second'] < second]
                     == [e for e in motion['events'] if e['second'] < second])
        current = _current_movements_preserved(reference, data, second)
        preserved &= all(data[k] == reference[k] for k in ('stations', 'tracks', 'trains', 'headway_policy'))
    preparation = _preparation_ok(data, reference, second)
    supported = motion['checks']['supported_constraints_status'] == 'passed'
    passed = (supported and preserved and current and preparation and motion['obligations_closed']
              and gaps['passed'] and motion['checks']['peak_trains_on_line'] <= 53
              and terminal_gate(total) and terminal_gate(visits))
    _check(deadline)
    return dict(movement_passed=passed, terminal_resources=total, terminal_visits=visits,
        executed_prefix_preserved=preserved, current_movements_preserved=current,
        depot_preparation_passed=preparation, obligations_closed=motion['obligations_closed'],
        headways_passed=gaps['passed'], minimum_headway=gaps['minimum_seconds'],
        maximum_headway=gaps['maximum_seconds'], headway_violations=gaps['violations'][:10],
        peak_trains_on_line=motion['checks']['peak_trains_on_line'],
        infrastructure_scope='assumed shared terminal TOTAL2/standby1; real layout unverified',
        _data_digest=digest(data), _motion_digest=digest(motion), _transport_work=motion['transport_work'])


def _preparation_ok(data, reference, second):
    if reference is None:
        return True
    old = {t['trip_id']: t for t in reference['plans'][0]['trips']}
    timings = {(r['trip_id'], r['leg_index']): r['departure_second'] for r in data['leg_timing']}
    original = {(r['trip_id'], r['leg_index']): r['departure_second'] for r in reference['leg_timing']}
    tracks = {r['track_id']: r for r in data['tracks']}
    lead = parameter(data['headway_policy'], 'depot_release_s')
    trips = data['plans'][0]['trips']
    for trip in trips:
        track = trip['route'][0]
        if not track.startswith('ENTRY_') or timings[trip['trip_id'], 0] < second:
            continue
        if trip['trip_id'] in old and timings[trip['trip_id'], 0] == original[trip['trip_id'], 0]:
            continue
        starts = [timings[t['trip_id'], 0] for t in trips if t['passenger'] and t['train_id'] == trip['train_id']
                  and timings[t['trip_id'], 0] >= timings[trip['trip_id'], 0]]
        depot_id = tracks[track]['from_station']
        if next(s['kind'] for s in data['stations'] if s['station_id'] == depot_id) == 'siding':
            if not starts:
                return False
            continue
        depot = {'DEPOT_AVTOVO': 'Avtovo', 'DEPOT_NORTHERN': 'Northern'}.get(depot_id, depot_id)
        if depot not in lead or not starts or min(starts) < second + lead[depot]:
            return False
    return True


def peak_offpeak_profile(data, second, end, *, peak_seconds=125, offpeak_seconds=160):
    origin, policy = datetime.fromisoformat(data['period_start']), data['headway_policy']
    return [dict(terminal='*', start_second=s, end_second=min(end, s + 900),
                 headway_seconds=peak_seconds if minimum_headway(origin + timedelta(seconds=s), policy) < 150
                 else offpeak_seconds) for s in range(second, end, 900)]


def catalog_profiles(data, bundle, state, second, *, quality_policy='strict', deadline=float('inf'), service_end=None):
    policy = _policy(quality_policy)
    _check(deadline)
    at = datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)
    slots = _slots(data, bundle, at.isoformat(), state)
    surrogate = estimated_keep_wait_budget(data, bundle, second)
    search_budget = policy['mean_seconds'] if quality_policy == 'pareto' else surrogate['mean_wait_budget_seconds']
    solved = solve_joint_profile(slots, mean_wait_budget_seconds=search_budget,
        time_limit_seconds=min(3, max(0.01, deadline - monotonic())), deadline=deadline)
    _check(deadline)
    end = service_end if service_end is not None else max(r['end_second'] for r in slots)
    fixed = peak_offpeak_profile(data, second, end)
    specs = []
    if solved.get('slots'):
        directional = directional_profile(data, solved)
        horizon = max(r['end_second'] for r in directional)
        # Common policy beyond the forecast; it never reads future actual entries.
        continuation = peak_offpeak_profile(data, horizon, end)
        specs.append(dict(candidate_id='CP_DIRECTIONAL', builder='stable',
                          headway=directional + continuation, wear_priority=True))
    specs.append(dict(candidate_id='H125_OFF160_WEAR', builder='stable', headway=fixed, wear_priority=True))
    if minimum_headway(at, data['headway_policy']) >= 150:
        specs.append(dict(candidate_id='INTERVAL160_BRIDGE', builder='bridge',
                          headway=160, bridge_seconds=480, wear_priority=False))
    return specs, dict(cp_search=solved, cp_waiting_surrogate=surrogate,
        cp_search_budget_seconds=search_budget, outside_forecast_profile='common clock-aware H125/OFF160 assumption',
        exact_validator_required=True, physical_scope='shared TOTAL2/standby1')


def calculate_catalog(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
                      candidate_limit, deadline, cancelled, *, quality_policy='strict', builders=None,
                      profile_specs=None, bundle_override=None):
    """Return a reviewable decision; the caller advances its own immutable state."""
    began = monotonic()
    deadline = min(deadline, began + 470)
    quality, mode = _policy(quality_policy), _mode(strategy)
    _check(deadline, cancelled)
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or not 0 <= candidate_limit <= 3:
        raise ValueError('catalog candidate limit must be between zero and three')
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    bundle = copy.deepcopy(bundle_override) if bundle_override is not None else read_forecast(source, mode, ids, at)
    versions = bundle.get('source_versions', {})
    if versions.get('future_truth_intentionally_used') or versions.get('operationally_available') is False:
        raise ValueError('causal catalogue cannot use unavailable future truth')
    validate_bundle(bundle, ids, at)
    _check(deadline, cancelled)
    read_seconds = monotonic() - began
    build = builders if builders is not None else {'stable': rebuild_terminal_timetable, 'bridge': rebuild_bridge_timetable}
    specifications, search = (catalog_profiles(data, bundle, state, second, quality_policy=quality_policy,
        deadline=deadline, service_end=planning_end) if profile_specs is None and candidate_limit else
        (profile_specs or [], dict(cp_search=None, injected_profiles=profile_specs is not None)))
    candidates, rejected, seen = [], [], set()

    def record(name, changed, audit, action):
        _check(deadline, cancelled)
        checked = physical_gate(changed, second, planning_end, reference=data,
            reference_motion=getattr(runner, 'motion', None), deadline=deadline)
        audit = {**audit, **checked}
        identity = digest(changed)
        if not audit['movement_passed']:
            row = dict(candidate_id=name, strategy='REPLAN', status='rejected',
                       reason='independent physical/terminal gate failed', audit=audit)
            rejected.append(row); candidates.append(row)
        elif identity in seen:
            row = dict(candidate_id=name, strategy='REPLAN', status='duplicate', reason='same complete movement plan')
            rejected.append(row); candidates.append(row)
        else:
            seen.add(identity)
            candidates.append(dict(candidate_id=name, strategy='KEEP' if name == 'KEEP' else 'REPLAN',
                status='movement_checked', _data=changed, audit=audit, action=action,
                timing=dict(search_complete=True)))

    record('KEEP', data, {}, dict(kind='KEEP', decision_second=second))
    for spec in specifications[:candidate_limit]:
        _check(deadline, cancelled)
        name = spec['candidate_id']
        try:
            if spec['builder'] == 'bridge' and minimum_headway(datetime.fromisoformat(at), data['headway_policy']) < 150:
                raise ValueError('fixed-chain bridge is available only offpeak')
            options = {k: v for k, v in spec.items() if k not in ('candidate_id', 'builder')}
            changed, audit = build[spec['builder']](copy.deepcopy(data), second, planning_end,
                deadline=deadline, phase_seconds=0, **options)
            record(name, changed, audit, dict(kind='REPLAN', decision_second=second, **options))
        except ValueError as error:
            row = dict(candidate_id=name, strategy='REPLAN', status='rejected', reason=str(error))
            rejected.append(row); candidates.append(row)
    _check(deadline, cancelled)
    generation_seconds = monotonic() - began - read_seconds
    valid = [r for r in candidates if r['status'] == 'movement_checked']
    validation_began = monotonic()
    decision = (decide_operational(data, {'S1': runner}, {'S1': state}, profiles, {}, bundle, data['headway_policy'],
        candidates=valid, budget_seconds=max(0, deadline - monotonic() - 5), include_recovery=False)
        if valid else dict(candidate_table=[], search_complete=True))
    _check(deadline, cancelled)
    validation_seconds = monotonic() - validation_began
    decision['candidate_table'].extend(rejected)
    by_id = {r['candidate_id']: r for r in candidates}
    accounting_began = monotonic()
    for row in decision['candidate_table']:
        if row.get('status') == 'evaluated' and row.get('passengers_checked') and row.get('physical_passed'):
            _check(deadline, cancelled)
            candidate = by_id[row['candidate_id']]
            work = candidate['audit'].get('_transport_work')
            row['committed_per_train'] = closed_per_train(candidate['_data'], roster, transport_work=work)
    _check(deadline, cancelled)
    mean, p95 = keep_wait_budgets(decision['candidate_table']) if quality_policy == 'strict' else (90, 180)
    selection = choose_candidate(decision['candidate_table'], mean_wait_budget_s=mean, p95_wait_budget_s=p95,
                                 search_incomplete=not decision.get('search_complete', False))
    # The existing application helper may apply mitigation: keep it diagnostic here.
    selection['diagnostic_mitigation_candidate_id'] = selection.get('mitigation_candidate_id')
    selection['mitigation_candidate_id'] = None
    _check(deadline, cancelled)
    decision.update(selection=selection, quality_policy=quality, forecast_digest=digest(bundle),
        source_mode='ML' if mode == 'ML' else 'BASELINE_FORECAST', search44=search,
        forecast_issued_at=bundle['issued_at'], decision_time=at, decision_second=second,
        wait_budget_reference='same forecast KEEP' if quality_policy == 'strict' else 'assumed development 90/180',
        cash_saving_rub=None, lost_revenue_rub=None, economic_claim_requires_common_continuation=True,
        forecast_inference_in_timing=False, elapsed_seconds=monotonic() - began,
        latency_breakdown=dict(cached_read_seconds=read_seconds, generation_seconds=generation_seconds,
            validation_seconds=validation_seconds, accounting_and_selection_seconds=monotonic() - accounting_began))
    return dict(decision=decision, candidates=candidates)
