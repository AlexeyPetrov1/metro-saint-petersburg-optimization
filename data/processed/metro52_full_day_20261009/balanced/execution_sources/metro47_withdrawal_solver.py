"""Choose concrete terminal withdrawals; keep every other known service clock exact."""

from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from math import isfinite
from time import monotonic

from ortools.sat.python import cp_model

from dwell import timing_key
from forecasting import validate_bundle
from goal43_policy import parameter
from metro44_mpc import forecast_section_rates
from metro44_run import station_offsets
from simulation import digest


def solve_withdrawal_options(options, departures, *, option_values, rate_slots,
                             as_of, service_end_second, capacity=1200,
                             no_good_cuts=(), max_withdrawals=None,
                             time_limit_seconds=5, deadline=None):
    """Maximize exact known cancelled-duty TO1 value, subject to fixed-clock coverage.

    Demand bounds use the largest supplied directional crossing rate overlapping
    each possible retained gap. No passenger truth or waiting value is consulted.
    """
    started = monotonic()
    end = min(started+time_limit_seconds, deadline if deadline is not None else float('inf'))
    empty = dict(selected_train_ids=[], withdrawal_options=[], withdrawal_count=0,
        estimated_to1_resource_saving_rub=0, passenger_admissible=False,
        release_authorized=False, cash_saving_rub=None, waiting_in_objective=False,
        objective='cancelled_known_train_km_times_confirmed_TO1_rates',
        tail_demand_scope='no supplied forecast: fixed-clock maximum-gap coverage only',
        exact_movement_and_passenger_checks_required=True)
    if started >= end:
        return dict(empty, status='evaluation_timeout', search_complete=False)
    if (type(as_of) is not int or as_of < 0 or type(service_end_second) is not int
            or service_end_second <= as_of or type(capacity) is not int or capacity <= 0):
        raise ValueError('invalid clock or capacity')
    if max_withdrawals is not None and (type(max_withdrawals) is not int or max_withdrawals < 0):
        raise ValueError('invalid withdrawal feedback cardinality')
    ids = [r['train_id'] for r in options]
    if len(ids) != len(set(ids)) or set(option_values) != set(ids):
        raise ValueError('unique concrete train values required')
    if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in option_values.values()):
        raise ValueError('invalid known TO1 value')
    owner = {}
    for row in options:
        for key in ('cancel_service_ids', 'cancel_entry_ids', 'cancel_exit_ids'):
            for identity in row.get(key, []):
                if identity in owner:
                    raise ValueError('overlapping cancelled trip ownership')
                owner[identity] = row['train_id']
    for row in rate_slots:
        if (row['end_second'] <= row['start_second']
                or any(type(row[d+'_rate']) not in (int, float)
                       or not isfinite(row[d+'_rate']) or row[d+'_rate'] < 0 for d in ('up', 'down'))):
            raise ValueError('invalid forecast crossing rate')
    model = cp_model.CpModel()
    removed = {tid: model.new_bool_var('withdraw_'+tid) for tid in ids}
    for tid in ids:
        if option_values[tid] == 0:
            model.add(removed[tid] == 0)
    if max_withdrawals is not None:
        model.add(sum(removed.values()) <= max_withdrawals)
    for cut in no_good_cuts:
        chosen = set(cut)
        if chosen-set(ids):
            raise ValueError('unknown cut train')
        model.add(sum(removed[t].Not() for t in chosen)
                  + sum(removed[t] for t in ids if t not in chosen) >= 1)
    by_track = defaultdict(list)
    immutable = 0
    for row in departures:
        if row['trip_id'] not in owner:
            immutable += 1
        if row['second'] <= service_end_second:
            by_track[row['track_id']].append(row)
    if not by_track:
        raise ValueError('no directional line-track clocks')
    coverage_constraints = capacity_constraints = 0
    seen = set()

    def keep(row):
        return removed[owner[row['trip_id']]].Not() if row['trip_id'] in owner else 1

    def require(candidates, left=None):
        if any(r['trip_id'] not in owner for r in candidates):
            return
        candidates_ids = tuple(sorted(owner[r['trip_id']] for r in candidates))
        left_id = owner.get(left['trip_id']) if left else None
        key = candidates_ids, left_id
        if key not in seen:
            model.add(sum(keep(r) for r in candidates) >= (keep(left) if left else 1))
            seen.add(key)

    def within_capacity(left, right):
        offset = left.get('terminal_offset_seconds', 0)
        a, b = left['second']-offset, right['second']-offset
        maximum = max((r[left['direction']+'_rate'] for r in rate_slots
                       if r['start_second'] < b and r['end_second'] > a), default=0)
        return Decimal(str(maximum))*(right['second']-left['second']) <= capacity

    for track, rows in by_track.items():
        rows.sort(key=lambda r: r['second'])
        if len({r['second'] for r in rows}) != len(rows):
            raise ValueError('duplicate directional departure clock')
        history = [r for r in rows if r['second'] < as_of]
        future = [r for r in rows if r['second'] >= as_of]
        require([r for r in rows if as_of-480 <= r['second'] <= as_of])
        end_row = dict(rows[-1], second=service_end_second)
        require([r for r in rows if service_end_second-480 <= r['second'] <= service_end_second
                 and within_capacity(r, end_row)])
        relevant = history[-1:]+future
        for index, left in enumerate(relevant):
            if monotonic() >= end:
                return dict(empty, status='evaluation_timeout', search_complete=False)
            if left['second'] >= service_end_second-480 and within_capacity(left, end_row):
                continue
            possible = []
            for right in relevant[index+1:]:
                if right['second']-left['second'] > 480:
                    break
                possible.append(right)
            allowed = [r for r in possible if within_capacity(left, r)]
            before = len(seen)
            require(allowed, left)
            if len(seen) > before:
                coverage_constraints += 1
                if len(allowed) < len(possible):
                    capacity_constraints += 1
    model.maximize(sum(round(option_values[tid]*1_000_000)*removed[tid] for tid in ids))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    solver.parameters.max_time_in_seconds = max(1e-6, end-monotonic())
    if monotonic() >= end:
        return dict(empty, status='evaluation_timeout', search_complete=False)
    status = solver.solve(model)
    common = dict(fixed_clock_coverage_constraints=coverage_constraints,
        forecast_capacity_constraints=capacity_constraints, immutable_departure_count=immutable,
        option_count=len(options), feedback_exact_no_good_count=len(no_good_cuts),
        feedback_max_withdrawals=max_withdrawals, elapsed_seconds=monotonic()-started,
        solver_status=solver.status_name(status), objective_precision_rub=1e-6)
    if monotonic() >= end or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(empty, **common, status='no_feasible_candidate' if status == cp_model.INFEASIBLE
                    else 'evaluation_timeout', search_complete=status == cp_model.INFEASIBLE)
    chosen = sorted(tid for tid in ids if solver.value(removed[tid]))
    return dict(empty, **common, status='optimal' if status == cp_model.OPTIMAL else 'search_incomplete',
        search_complete=status == cp_model.OPTIMAL, selected_train_ids=chosen,
        withdrawal_options=[r for r in options if r['train_id'] in chosen], withdrawal_count=len(chosen),
        estimated_to1_resource_saving_rub=sum(option_values[t] for t in chosen),
        demand_bound_scope='max supplied quarter rate times entire possible retained gap; terminal-clock nominal offset')


def solve_withdrawals(data, bundle, state, as_of, service_end_second, *, deadline=None,
                      no_good_cuts=(), max_withdrawals=None, time_limit_seconds=5,
                      car_concentration=1, scenario='S1'):
    """Use canonical producer options and the complete causal known movement schedule."""
    from metro47_withdrawal import withdrawal_options

    if state.get('second') != as_of:
        raise ValueError('current checkpoint clock mismatch')
    origin = datetime.fromisoformat(data['period_start'])
    at = (origin+timedelta(seconds=as_of)).isoformat()
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    versions = bundle.get('source_versions', {})
    if versions.get('future_truth_intentionally_used') or versions.get('operationally_available') is False:
        raise ValueError('unavailable future passenger truth')
    validate_bundle(bundle, ids, at)
    options = withdrawal_options(data, as_of, service_end_second, deadline=deadline)
    tracks = {r['track_id']: r for r in data['tracks']}
    stations = {r['station_id']: r for r in data['stations']}
    trips = {r['trip_id']: r for r in data['plans'][0]['trips']}
    trains = {r['train_id']: r for r in data['trains']}
    offsets = station_offsets(data)
    # The canonical producer computes these from exact source Decimal distances,
    # including removed future ENTRY/EXIT movements and the one new early EXIT.
    km_by_train = {r['train_id']: float(r['saved_km']) for r in options}
    values = {r['train_id']: float(r['to1_resource_saving_rub']) for r in options}
    clock_rows = {}
    for row in data.get('previous_departures', []):
        track = tracks[row['track_id']]
        if all(stations[track[s]]['kind'] == 'line' for s in ('from_station', 'to_station')):
            direction = 'up' if row['track_id'].endswith('_up') else 'down'
            clock_rows[row['track_id'], row['departure_second']] = dict(track_id=row['track_id'], direction=direction,
                second=row['departure_second'], trip_id='@history',
                terminal_offset_seconds=offsets[direction][ids.index(track['from_station'])])
    for row in data['leg_timing']:
        track_id = (trains[row['train_id']]['location'] if row['trip_id'] == '@initial'
                    else trips[row['trip_id']]['route'][row['leg_index']])
        track = tracks[track_id]
        if not all(stations[track[s]]['kind'] == 'line' for s in ('from_station', 'to_station')):
            continue
        direction = 'up' if track_id.endswith('_up') else 'down'
        clock_rows[track_id, row['departure_second']] = dict(track_id=track_id, direction=direction,
            second=row['departure_second'], trip_id=row['trip_id'],
            terminal_offset_seconds=offsets[direction][ids.index(track['from_station'])])
    rates = forecast_section_rates(bundle, ids, at, period_start=data['period_start'],
        scenario=scenario, checkpoint=state, station_offsets_seconds=offsets)
    if type(car_concentration) not in (int, float) or not isfinite(car_concentration) or car_concentration < 1:
        raise ValueError('invalid car concentration')
    result = solve_withdrawal_options(options, list(clock_rows.values()), option_values=values,
        rate_slots=rates, as_of=as_of, service_end_second=service_end_second,
        capacity=int(1200/car_concentration), no_good_cuts=no_good_cuts,
        max_withdrawals=max_withdrawals, time_limit_seconds=time_limit_seconds, deadline=deadline)
    result.update(forecast_digest=digest(bundle), decision_second=as_of,
        future_actual_used=False, known_source_km_saving_by_train=km_by_train,
        selected_known_train_km_saving=sum(km_by_train[t] for t in result['selected_train_ids']),
        retained_clock_scope='all preserved SERVICE/initial/past/current leg clocks remain exact',
        preparation_scope='canonical producer options only, no invented depot readiness',
        car_capacity_scope='150 per car with supplied concentration proxy; actual160/1percent checked separately')
    return result
