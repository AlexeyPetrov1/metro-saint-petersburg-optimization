"""Integer demand-constrained frequency optimization; exact acceptance is separate."""

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
from math import ceil, isfinite
from time import monotonic

from ortools.sat.python import cp_model

from forecasting import validate_bundle
from goal43_policy import minimum_headway, parameter
from metro44_mpc import forecast_section_rates
from metro44_run import directional_profile, station_offsets
from metro_daily import _geometry
from simulation import digest


SCALE = 1_000_000


def _number(value, name, minimum=0):
    if type(value) not in (int, float) or not isfinite(value) or value < minimum:
        raise ValueError('invalid ' + name)
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid ' + name)
    return value


def _scaled_rate(value):
    return int((Decimal(str(value))*SCALE).to_integral_value(rounding=ROUND_CEILING))


def solve_demand_profile(slots, *, one_way_km, cost_rub_per_km,
                         half_cycle_seconds=3040, line_capacity=53, cars=8,
                         normal_limit=150, car_concentration=1, floor_rates=None,
                         time_limit_seconds=3, deadline=None):
    """Minimize only TO1 resource cost over every integer headway in the user bounds.

    Queue stock belongs in the supplied crossing rates. Fleet is an aggregate
    bound, not a claim about terminal positions, preparation, or complete duties.
    """
    started = monotonic()
    _number(time_limit_seconds, 'search budget', 1e-6)
    end = min(started+time_limit_seconds, deadline if deadline is not None else float('inf'))
    empty = dict(slots=[], passenger_admissible=False, release_authorized=False,
        cash_saving_rub=None, objective='to1_resource_cost_only', waiting_in_objective=False,
        mean_wait_budget_seconds=None,
        scope='forecast frequency/capacity surrogate; exact motion/passengers separate',
        finite_catalog=False, initial_and_final_obligations_checked=False,
        depot_preparation_checked=False, terminal_positions_checked=False)
    if started >= end:
        return dict(empty, status='evaluation_timeout', search_complete=False)
    _number(one_way_km, 'one-way km', 1e-9)
    _number(cost_rub_per_km, 'TO1 rate', 1e-9)
    for value, name in ((half_cycle_seconds, 'half cycle'), (line_capacity, 'line capacity'),
                        (cars, 'car count'), (normal_limit, 'normal car limit')):
        _integer(value, name)
    if line_capacity > 53 or cars != 8 or normal_limit != 150:
        raise ValueError('user fleet/car/normal limits required')
    _number(car_concentration, 'car concentration', 1)
    capacity = int((Decimal(cars*normal_limit)/Decimal(str(car_concentration)))
                   .to_integral_value(rounding=ROUND_FLOOR))
    rows = deepcopy(list(slots))
    if not rows:
        raise ValueError('empty demand horizon')
    feedback, seen = {}, set()
    for row in floor_rates or []:
        clock = _integer(row['start_second'], 'feedback clock', 0)
        observed = _integer(row['observed_as_of_second'], 'feedback availability', 0)
        if (clock in seen or observed > rows[0]['start_second']
                or row.get('future_actual_used') is not False
                or not isinstance(row.get('source_id'), str) or not row['source_id'].strip()):
            raise ValueError('unavailable or unprovenance feedback')
        seen.add(clock)
        feedback[clock] = [_number(row[d+'_rate'], 'feedback rate') for d in ('up', 'down')]
    if set(feedback) - {r['start_second'] for r in rows}:
        raise ValueError('feedback clock outside demand horizon')
    model, variables, work = cp_model.CpModel(), [], []
    for index, row in enumerate(rows):
        first = _integer(row['start_second'], 'slot clock', 0)
        last = _integer(row['end_second'], 'slot end', 0)
        if last-first != 900 or index and first != rows[index-1]['end_second']:
            raise ValueError('contiguous complete 15-minute slots required')
        minimum = row.get('minimum_headway_seconds', 150)
        maximum = row.get('maximum_headway_seconds', 480)
        if minimum not in (113, 150) or type(minimum) is not int or maximum != 480 or type(maximum) is not int:
            raise ValueError('user peak/offpeak/maximum headway bounds required')
        ready = min(line_capacity, _integer(row.get('ready_capacity', line_capacity), 'ready capacity'))
        rates = [max(_number(row[d+'_rate'], 'crossing rate'), feedback.get(first, [0, 0])[k])
                 for k, d in enumerate(('up', 'down'))]
        upper = [min(maximum, capacity*SCALE//_scaled_rate(r)) if r else maximum for r in rates]
        if min(upper) < minimum:
            return dict(empty, status='no_feasible_candidate', search_complete=True,
                        reason='strict per-car crossing capacity', infeasible_slot=index)
        headways = [model.new_int_var(minimum, upper[k], f'h{index}_{d}')
                    for k, d in enumerate(('up', 'down'))]
        frequencies = []
        for k, h in enumerate(headways):
            model.add(_scaled_rate(rates[k])*h <= capacity*SCALE)
            frequency = model.new_int_var(900*SCALE//upper[k], 900*SCALE//minimum, f'f{index}_{k}')
            model.add_division_equality(frequency, 900*SCALE, h)
            model.add_hint(h, upper[k])
            model.add_hint(frequency, 900*SCALE//upper[k])
            frequencies.append(frequency)
        product = model.new_int_var(minimum**2, upper[0]*upper[1], f'product{index}')
        model.add_multiplication_equality(product, headways)
        model.add_hint(product, upper[0]*upper[1])
        model.add(half_cycle_seconds*sum(headways) <= ready*product)
        variables.append((headways, rates))
        work.extend(frequencies)
    # Cost has one confirmed positive rate. Maximizing intervals minimizes cost;
    # type assignment and the true finite-window ledger remain exact-validator work.
    unit_cost = max(1, round(one_way_km*cost_rub_per_km*10_000))
    model.minimize(unit_cost*sum(work))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    solver.parameters.max_time_in_seconds = max(1e-6, end-monotonic())
    if monotonic() >= end:
        return dict(empty, status='evaluation_timeout', search_complete=False)
    status = solver.solve(model)
    if monotonic() >= end or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(empty, status='no_feasible_candidate' if status == cp_model.INFEASIBLE
                    else 'evaluation_timeout', search_complete=status == cp_model.INFEASIBLE,
                    solver_status=solver.status_name(status), elapsed_seconds=monotonic()-started)
    extracted, km = [], 0.0
    for row, (hs, rates) in zip(rows, variables):
        up, down = [solver.value(h) for h in hs]
        needed = ceil(Fraction(half_cycle_seconds, up)+Fraction(half_cycle_seconds, down))
        extracted.append(dict(start_second=row['start_second'], end_second=row['end_second'],
            up_headway_seconds=up, down_headway_seconds=down, train_count=needed,
            up_rate=rates[0], down_rate=rates[1]))
        km += one_way_km*900*(1/up+1/down)
    return dict(empty, slots=extracted, status='optimal' if status == cp_model.OPTIMAL else 'search_incomplete',
        search_complete=status == cp_model.OPTIMAL, solver_status=solver.status_name(status),
        forecast_capacity_passed=True, capacity_passengers=capacity,
        car_concentration=car_concentration, car_concentration_scope='supplied allocation proxy; actual car loads checked separately',
        cost_rub_per_km=cost_rub_per_km, estimated_train_km=km, estimated_to1_resource_rub=km*cost_rub_per_km,
        objective_division_scale=SCALE, objective_rounding_scope='frequency floor error below one millionth of a departure per direction/slot',
        feedback_digest=digest(floor_rates) if floor_rates else None, elapsed_seconds=monotonic()-started)


def optimize_demand_profile(data, bundle, state, as_of, *, horizon_seconds=7200,
                            one_way_km=None, cost_rub_per_km=9.2, car_concentration=1,
                            floor_rates=None, ready_capacity=None, scenario='S1',
                            service_end_second=None, continuation_headway=None,
                            time_limit_seconds=3, deadline=None):
    """Adapt a causal entry bundle/current queues to bridge-compatible terminal rows."""
    origin = datetime.fromisoformat(data['period_start'])
    if isinstance(as_of, str):
        moment = datetime.fromisoformat(as_of)
        if moment.tzinfo is None or origin.tzinfo is None:
            raise ValueError('aware decision clock required')
        seconds = (moment-origin).total_seconds()
        if not seconds.is_integer():
            raise ValueError('integer decision clock required')
        as_of = int(seconds)
    _integer(as_of, 'decision clock', 0)
    _integer(horizon_seconds, 'forecast horizon')
    if as_of % 900 or horizon_seconds % 900 or state.get('second') != as_of:
        raise ValueError('15-minute decision/current checkpoint required')
    at = (origin+timedelta(seconds=as_of)).isoformat()
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    versions = bundle.get('source_versions', {})
    if versions.get('future_truth_intentionally_used') or versions.get('operationally_available') is False:
        raise ValueError('future actual demand unavailable to optimizer')
    validate_bundle(bundle, ids, at)
    tracks, _, routes, half = _geometry(data)
    if one_way_km is None:
        lengths = [sum(float(tracks[t]['length_km']) for t in route) for route in routes.values()]
        if not lengths or max(lengths)-min(lengths) > 1e-6:
            raise ValueError('symmetric source one-way length required')
        one_way_km = lengths[0]
    slots = forecast_section_rates(bundle, ids, at, period_start=data['period_start'],
        scenario=scenario, checkpoint=state, station_offsets_seconds=station_offsets(data))
    slots = [r for r in slots if as_of <= r['start_second'] < as_of+horizon_seconds]
    forecast_end = max((r['end_second'] for r in slots), default=as_of)
    expected = list(range(as_of, forecast_end, 900))
    actual_horizon = forecast_end-as_of
    if not slots or [r['start_second'] for r in slots] != expected:
        raise ValueError('complete available future forecast horizon required')
    policy = data['headway_policy']
    for row in slots:
        row['minimum_headway_seconds'] = max(minimum_headway(origin+timedelta(seconds=s), policy)
            for s in (row['start_second'], row['end_second']-1))
        row['maximum_headway_seconds'] = parameter(policy, 'headway_max_s')
        if ready_capacity is not None:
            row['ready_capacity'] = ready_capacity.get(row['start_second'], 0) if isinstance(ready_capacity, dict) else ready_capacity
    result = solve_demand_profile(slots, one_way_km=one_way_km, cost_rub_per_km=cost_rub_per_km,
        half_cycle_seconds=half, line_capacity=parameter(policy, 'line_train_limit'),
        cars=parameter(policy, 'cars_per_train'), normal_limit=parameter(policy, 'normal_limit'),
        car_concentration=car_concentration, floor_rates=floor_rates,
        time_limit_seconds=time_limit_seconds, deadline=deadline)
    profile = directional_profile(data, result)
    tail = []
    service_end = forecast_end if service_end_second is None else service_end_second
    _integer(service_end, 'continuation end', 0)
    if service_end < forecast_end:
        raise ValueError('continuation must end at or beyond the complete forecast horizon')
    common = ceil(2*half/parameter(policy, 'line_train_limit')) if continuation_headway is None else continuation_headway
    _integer(common, 'common continuation headway')
    if common > 480:
        raise ValueError('continuation exceeds maximum interval')
    if profile:
        for first in range(forecast_end, service_end, 900):
            last = min(service_end, first+900)
            floor = max(minimum_headway(origin+timedelta(seconds=s), policy) for s in (first, last-1))
            tail.append(dict(terminal='*', start_second=first, end_second=last,
                             headway_seconds=max(floor, common)))
    return {**result, 'profile': profile+tail, 'forecast_digest': digest(bundle),
        'decision_second': as_of, 'forecast_horizon_seconds': actual_horizon,
        'requested_horizon_seconds': horizon_seconds,
        'future_actual_used': False, 'queue_stock_scope': 'known checkpoint crossing stock amortized in first forecast quarter',
        'rate_clock_scope': 'existing spatial shift by station travel/stop offsets; nearest forecast edge held',
        'continuation_profile': tail, 'continuation_scope': 'common causal clock/geometry assumption; demand not forecast or optimized beyond horizon',
        'readiness_scope': 'aggregate supplied ready_capacity or line ceiling; concrete train readiness and preparation require exact gate',
        'type_cost_scope': 'supplied confirmed positive TO1 rate; actual train types and km are separate exact accounting'}
