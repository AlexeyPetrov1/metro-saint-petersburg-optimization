"""Accept paired Line 1 resource savings without a waiting-time budget.

Cases declare matching passenger, service and full movement scopes. Transport
work contains all 68 trains and separates passenger from empty mileage. The
caller must independently audit the supplied movement/passenger evidence;
this module checks its arithmetic and acceptance conditions, not the kernel.
"""

from decimal import Decimal


RATES = {'baltiets': Decimal('9.2'), 'yubileyny': Decimal('7.4')}
REQUIRED_PHYSICAL = ('movement_passed', 'headways_passed', 'terminal_passed',
    'depot_preparation_passed', 'executed_prefix_preserved', 'current_movements_preserved')
CONTEXT = ('cohort_digest', 'profiles_digest', 'common_start_digest',
           'cohort_scope', 'service_scope', 'resource_scope')


def _number(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a quantity')
    try:
        value = Decimal(str(value))
    except Exception as exc:
        raise ValueError('invalid quantity') from exc
    if not value.is_finite() or value < 0:
        raise ValueError('quantity must be finite and nonnegative')
    return value


def _ledger(case):
    rows = case['transport_work']
    result = {}
    for row in rows:
        tid, kind = row['train_id'], row['train_type'].casefold()
        if not tid or tid in result or kind not in RATES:
            raise ValueError('invalid fixed fleet identity/type')
        km = _number(row['passenger_train_km']) + _number(row['nonpassenger_train_km'])
        if 'train_km' in row and _number(row['train_km']) != km:
            raise ValueError('train mileage components disagree')
        result[tid] = dict(train_type=kind, km=km, resource=km * RATES[kind])
    if len(result) != 68:
        raise ValueError('fixed fleet requires all 68 train rows')
    if sum(r['train_type'] == 'baltiets' for r in result.values()) != 37:
        raise ValueError('fixed fleet requires 37 Baltiets and 31 Yubileyny')
    return result


def _tail(case, ledger):
    raw = case.get('endpoint_km_upper_bound')
    if raw is None:
        return (Decimal(0), Decimal(0)) if case.get('movement_obligations_closed') is True else (None, None)
    if set(raw) != set(ledger):
        raise ValueError('endpoint bounds require every fixed train')
    km = {tid: _number(value) for tid, value in raw.items()}
    if case.get('movement_obligations_closed') is True and any(km.values()):
        raise ValueError('closed movement cannot have endpoint mileage')
    return sum(km.values(), Decimal(0)), sum((value * RATES[ledger[tid]['train_type']]
                                           for tid, value in km.items()), Decimal(0))


def _capacity(exposure):
    if exposure.get('admissible') is not True:
        return False
    global_row = exposure.get('global_summary', exposure.get('global'))
    collections = [exposure.get(k) for k in ('windows', 'per_train', 'per_car')]
    if not isinstance(global_row, dict) or any(not rows for rows in collections):
        return False
    collections = [list(rows.values()) if isinstance(rows, dict) else rows for rows in collections]
    try:
        summaries = [(global_row, True),
                     *((row, i == 0) for i, rows in enumerate(collections) for row in rows)]
        for row, share_required in summaries:
            volume, exceptional, hard, load = (_number(row[k]) for k in
                ('passenger_car_seconds', 'exceptional_car_seconds', 'hard_limit_car_seconds', 'max_load'))
            if exceptional > volume or hard or load > 160:
                return False
            share = row.get('exceptional_time_share')
            if not volume:
                if exceptional or load or share is not None:
                    return False
            elif (share is None or abs(_number(share) - exceptional / volume) > Decimal('1e-12')
                  or share_required and _number(share) > Decimal('.01')):
                return False
        for rows in collections:
            for key in ('passenger_car_seconds', 'exceptional_car_seconds', 'hard_limit_car_seconds'):
                if sum((_number(row[key]) for row in rows), Decimal(0)) != _number(global_row[key]):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _physical(case):
    gate = case.get('physical', {})
    peak = gate.get('peak_trains_on_line')
    return (all(gate.get(key) is True for key in REQUIRED_PHYSICAL)
            and type(peak) is int and 0 <= peak <= 53)


def assess_pair(reference, candidate, *, threshold='0.01'):
    """Return conservative paired savings and explicit reasons for rejection.

    End progress is bounded independently, without trajectory coupling or
    interpolation. Shared initial progress cancels under common-start evidence.
    Arrival mileage charges the whole crossing initial track; reference cost is
    therefore an upper bound, giving a conservative saving fraction. Open ends
    remain diagnostic and never satisfy the full closed-plan requirement.
    """
    threshold = _number(threshold)
    if not 0 < threshold <= 1:
        raise ValueError('saving threshold must be in (0,1]')
    old, new = _ledger(reference), _ledger(candidate)
    if {k: r['train_type'] for k, r in old.items()} != {k: r['train_type'] for k, r in new.items()}:
        raise ValueError('fixed fleet identity and types must match')
    reasons = []
    if reference.get('simulated_line') != 1 or candidate.get('simulated_line') != 1:
        reasons.append('line_one_required')
    if (not reference.get('period_start') or reference.get('period_start') != candidate.get('period_start')
            or any(reference.get(k) is None or reference.get(k) != candidate.get(k) for k in CONTEXT)):
        reasons.append('pair_context_mismatch')
    for label, case in (('reference', reference), ('candidate', candidate)):
        scope = case.get('resource_scope', {})
        if scope.get('includes_entry_exit') is not True:
            reasons.append('entry_exit_scope_unverified')
        if (scope.get('start_second') != 0 or scope.get('end_second') != case.get('closure_second')
                or type(case.get('closure_second')) is not int or case['closure_second'] <= 0):
            reasons.append(f'{label}_full_horizon_unverified')
        boundaries = [case.get(k, {}) for k in ('cohort_scope', 'service_scope')]
        if (any(type(row.get(k)) is not int for row in boundaries for k in ('start_second', 'end_second'))
                or any(not 0 <= row['start_second'] < row['end_second'] <= scope.get('end_second', -1) for row in boundaries)
                or boundaries[0]['end_second'] > boundaries[1]['end_second']):
            reasons.append(f'{label}_full_horizon_unverified')
        if not _physical(case):
            reasons.append(f'{label}_physical')
        if case.get('movement_obligations_closed') is not True:
            reasons.append(f'{label}_movement_unclosed')
        stocks = [case.get(k) for k in ('entered', 'completed', 'queued', 'approaching', 'onboard')]
        if (any(type(x) is not int or x < 0 for x in stocks)
                or stocks[0] != stocks[1] or any(stocks[2:])
                or case.get('all_service_arrivals_completed') is not True):
            reasons.append(f'{label}_passengers_unfinished')
        if not _capacity(case.get('exposure', {})):
            reasons.append(f'{label}_passenger_capacity')
    if reference.get('entered') != candidate.get('entered'):
        reasons.append('passenger_total_mismatch')
    km_delta = sum((old[k]['km'] - new[k]['km'] for k in old), Decimal(0))
    wear_delta = sum((old[k]['resource'] - new[k]['resource'] for k in old), Decimal(0))
    old_tail, old_wear = _tail(reference, old)
    new_tail, new_wear = _tail(candidate, new)
    lower = None if new_tail is None else km_delta - new_tail
    upper = None if old_tail is None else km_delta + old_tail
    resource_lower = None if new_wear is None else wear_delta - new_wear
    resource_upper = None if old_wear is None else wear_delta + old_wear
    baseline = sum((r['resource'] for r in old.values()), Decimal(0))
    baseline_upper = None if old_wear is None else baseline + old_wear
    if 'pair_context_mismatch' in reasons:
        lower = upper = resource_lower = resource_upper = None
    fraction = None if resource_lower is None or not baseline_upper else resource_lower / baseline_upper
    if fraction is None:
        reasons.append('saving_fraction_unverified')
    elif resource_lower < threshold * baseline_upper:
        reasons.append('saving_below_threshold')
    number = lambda value: None if value is None else float(value)
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), threshold=float(threshold),
        fixed_fleet_size=68, reference_to1_resource_rub_upper_bound=number(baseline_upper),
        candidate_arrival_to1_resource_rub=number(sum((r['resource'] for r in new.values()), Decimal(0))),
        saving_train_km_lower_bound=number(lower), saving_train_km_upper_bound=number(upper),
        saving_km_per_fixed_train_lower_bound=number(None if lower is None else lower / 68),
        saving_to1_resource_rub_lower_bound=number(resource_lower),
        saving_to1_resource_rub_upper_bound=number(resource_upper), saving_fraction_lower_bound=number(fraction),
        per_train=[dict(train_id=k, train_type=old[k]['train_type'], reference_km=float(old[k]['km']),
            candidate_km=float(new[k]['km']), arrival_saving_km=float(old[k]['km'] - new[k]['km']),
            arrival_to1_resource_saving_rub=float(old[k]['resource'] - new[k]['resource'])) for k in sorted(old)],
        waiting_budget_applied=False, position_interpolation_assumed=False,
        cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None,
        scope='paired full closed Line 1 plan; TO1 resource equivalent, not cash; caller must independently audit journals')
