"""Small joint frequency/fleet search; the existing event engine validates its plans."""

from datetime import datetime, timedelta
from math import ceil, isfinite
from time import monotonic

from ortools.sat.python import cp_model


def forecast_section_rates(bundle, station_ids, as_of, *, period_start=None, scenario='S1',
                           checkpoint=None, demand_scale=1.0, station_offsets_seconds=None):
    """Convert entry forecasts to crossing rates under the explicitly assumed OD model."""
    now = datetime.fromisoformat(as_of)
    origin = datetime.fromisoformat(period_start or as_of)
    if datetime.fromisoformat(bundle['issued_at']) > now:
        raise ValueError('future forecast issue')
    if len(set(station_ids)) != len(station_ids) or len(station_ids) < 2:
        raise ValueError('distinct ordered stations required')
    if scenario not in ('S1', 'S2', 'S3') or not isfinite(demand_scale) or demand_scale <= 0:
        raise ValueError('invalid demand scenario or scale')
    positions = {s: i for i, s in enumerate(station_ids)}
    quarters, seen = {}, set()
    for row in bundle['rows']:
        at = datetime.fromisoformat(row['interval_start'])
        key = row['station_id'], at
        if key in seen:
            raise ValueError('duplicate station quarter')
        seen.add(key)
        if row['station_id'] not in positions:
            raise ValueError('unknown forecast station')
        count = float(row['predicted_count'])
        if not isfinite(count) or count < 0:
            raise ValueError('invalid entry forecast')
        if at >= now:
            quarters.setdefault(at, {})[row['station_id']] = count
    result = []
    offsets = station_offsets_seconds or {d: [0]*len(station_ids) for d in ('up', 'down')}
    if set(offsets) != {'up', 'down'} or any(len(v) != len(station_ids) for v in offsets.values()):
        raise ValueError('directional station offsets must cover the ordered stations')
    times = sorted(quarters)
    for at, counts in sorted(quarters.items()):
        if set(counts) != set(station_ids):
            raise ValueError('incomplete station quarter')
        flows = {d: [0.0]*(len(station_ids)-1) for d in (1, -1)}
        entries = {d: 0.0 for d in (1, -1)}
        for station, count in counts.items():
            i = positions[station]
            destinations = [j for j in range(len(station_ids)) if j != i]
            weights = [4 if scenario == 'S2' and 6 <= j < 10 else
                       abs(j-i) if scenario == 'S3' else 1 for j in destinations]
            for j, weight in zip(destinations, weights):
                direction = 1 if j > i else -1
                boarding_at = at+timedelta(seconds=offsets['up' if direction == 1 else 'down'][i])
                quarter = boarding_at.replace(minute=boarding_at.minute//15*15, second=0, microsecond=0)
                # Forecast edges are approximations, never observations supplied from the future.
                shifted = quarters.get(quarter, quarters[times[0] if quarter < times[0] else times[-1]])
                amount = shifted[station]*weight/sum(weights)*demand_scale
                entries[direction] += amount
                for section in range(min(i, j), max(i, j)):
                    flows[direction][section] += amount/900
        # Queued stock is a separate obligation, never a predicted turnstile count.
        if checkpoint is not None and not result:
            for queue in checkpoint['queues'].values():
                for row in queue:
                    i, j = positions[row['origin']], positions[row['destination']]
                    direction = 1 if j > i else -1
                    for section in range(min(i, j), max(i, j)):
                        flows[direction][section] += row['count']/900
        result.append(dict(start_second=int((at-origin).total_seconds()),
            end_second=int((at-origin).total_seconds())+900,
            up_rate=max(flows[1]), down_rate=max(flows[-1]),
            up_entries=entries[1], down_entries=entries[-1],
            up_section_rates=flows[1], down_section_rates=flows[-1],
            spatial_shift_applied=station_offsets_seconds is not None,
            horizon_edge_policy='hold nearest available forecast quarter; exact simulator checks tail',
            od=scenario, entry_to_load_conversion='assumed OD crossing rate; exact event check required'))
    return result


def solve_joint_profile(slots, *, headway_choices=(115, 118, 120, 125, 130, 135, 140,
                       150, 160, 170, 180, 200, 220, 240, 270, 300, 360, 420, 480),
                       half_cycle_seconds=3040, line_capacity=53, cars=8, normal_limit=150,
                       mean_wait_budget_seconds=None, time_limit_seconds=3.0, deadline=None,
                       allow_incumbent=False):
    """Jointly choose directional frequencies and necessary fleet for each 15m slot.

    Capacity uses a crossing-rate approximation. Ready capacity must be supplied by
    the caller from known train/depot commitments. Assignments, turnbacks, queues,
    transitions, and end-of-horizon obligations remain subject to exact validation.
    Opt-in incumbents survive the search budget, never the caller's hard deadline.
    """
    started = monotonic()
    if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError('positive finite search budget required')
    finish = min(started+time_limit_seconds, deadline if deadline is not None else float('inf'))
    empty = dict(slots=[], passenger_admissible=False, release_authorized=False,
                 cash_saving_rub=None, validated_scope='aggregate_demand_frequency_fleet_only')
    if started >= finish:
        return dict(empty, status='evaluation_timeout', search_complete=False)
    choices = sorted(set(headway_choices))
    if not choices or any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in choices):
        raise ValueError('positive integer headway choices required')
    if line_capacity <= 0 or half_cycle_seconds <= 0 or cars <= 0 or normal_limit <= 0:
        raise ValueError('positive vehicle and cycle limits required')
    if mean_wait_budget_seconds is not None and (not isfinite(mean_wait_budget_seconds) or mean_wait_budget_seconds < 0):
        raise ValueError('nonnegative finite service budget required')
    model = cp_model.CpModel()
    variables, options, wait_terms, work_terms, risk_terms = [], [], [], [], []
    total_entries = 0
    for index, row in enumerate(slots):
        minimum = row.get('minimum_headway_seconds', 150)
        maximum = row.get('maximum_headway_seconds', 480)
        ready = min(line_capacity, row.get('ready_capacity', line_capacity))
        rates = [float(row['up_rate']), float(row['down_rate'])]
        entries = [float(row[f'{direction}_wait_entries'] if f'{direction}_wait_entries' in row
                         else row[f'{direction}_entries'])
                   for direction in ('up', 'down')]
        if any(not isfinite(e) or e < 0 for e in entries):
            raise ValueError('invalid waiting cohort')
        weights = [round(e * 100) for e in entries]
        if any(not isfinite(r) or r < 0 for r in rates) or any(w < 0 for w in weights):
            raise ValueError('invalid demand')
        total_entries += sum(weights)
        available = []
        for up in choices:
            for down in choices:
                if not minimum <= up <= maximum or not minimum <= down <= maximum:
                    continue
                needed = ceil(half_cycle_seconds*(1/up+1/down)-1e-10)
                if needed > ready:
                    continue
                excess = round(100*sum(max(0, rate*h-cars*normal_limit)
                                      for rate, h in zip(rates, (up, down))))
                work = round(1_000_000_000/up)+round(1_000_000_000/down)
                wait = weights[0]*up+weights[1]*down
                available.append((up, down, needed, excess, work, wait))
        if not available:
            return dict(empty, status='no_feasible_candidate', search_complete=True,
                        infeasible_slot=index, reason='frequency/fleet bounds')
        selected = [model.new_bool_var(f'q{index}_choice{i}') for i in range(len(available))]
        model.add_exactly_one(selected)
        variables.append(selected); options.append(available)
        risk_terms.extend(v*o[3] for v, o in zip(selected, available))
        work_terms.extend(v*o[4] for v, o in zip(selected, available))
        wait_terms.extend(v*o[5] for v, o in zip(selected, available))
        if row.get('previous_up_headway_seconds') is not None:
            # Reuse an admissible warm start without making it an instruction.
            hint = next((i for i, o in enumerate(available) if o[:2] ==
                         (row['previous_up_headway_seconds'], row['previous_down_headway_seconds'])), None)
            if hint is not None:
                for i, v in enumerate(selected):
                    model.add_hint(v, int(i == hint))
    if mean_wait_budget_seconds is not None:
        model.add(sum(wait_terms) <= int(2*mean_wait_budget_seconds*total_entries))
    # Separate solves avoid overflowing integer coefficients on a busy line.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    status = cp_model.UNKNOWN
    complete = True
    incumbent, incumbent_status, optimized_stages = None, None, 0
    for objective in (sum(risk_terms), sum(work_terms), sum(wait_terms)):
        model.minimize(objective)
        solver.parameters.max_time_in_seconds = max(1e-6, finish-monotonic())
        status = solver.solve(model)
        if allow_incumbent and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            incumbent = [[i for i, v in enumerate(selected) if solver.value(v)]
                         for selected in variables]
            incumbent_status = status
        if status != cp_model.OPTIMAL:
            complete = False
            break
        optimized_stages += 1
        model.add(objective == solver.value(objective))
        if allow_incumbent and monotonic() >= finish:
            complete = False
            break
    checked_at = monotonic()
    outer_expired = deadline is not None and checked_at >= deadline
    recovered = allow_incumbent and (status == cp_model.UNKNOWN or checked_at >= finish)
    valid = (incumbent is not None and not outer_expired and
             status not in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID)) if allow_incumbent else (
             status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and checked_at < finish)
    if not valid:
        return dict(empty, status='no_feasible_candidate' if status == cp_model.INFEASIBLE else
                    'model_invalid' if status == cp_model.MODEL_INVALID else 'evaluation_timeout',
                    search_complete=status == cp_model.INFEASIBLE,
                    solver_status=solver.status_name(status), elapsed_seconds=monotonic()-started)
    extracted, weighted_wait, excess, frequency = [], 0, 0, 0
    for index, (row, selected, available) in enumerate(zip(slots, variables, options)):
        indices = incumbent[index] if allow_incumbent else [i for i, v in enumerate(selected) if solver.value(v)]
        if len(indices) != 1:
            raise RuntimeError('incomplete aggregate assignment')
        up, down, trains, risk, work, wait = available[indices[0]]
        if trains > min(line_capacity, row.get('ready_capacity', line_capacity)):
            raise RuntimeError('aggregate fleet assignment exceeds readiness')
        extracted.append(dict(start_second=row['start_second'], end_second=row['end_second'],
            up_headway_seconds=up, down_headway_seconds=down, train_count=trains,
            forecast_excess_passengers=risk/100))
        weighted_wait += wait; excess += risk; frequency += work
    mean = weighted_wait/(2*total_entries) if total_entries else None
    if mean_wait_budget_seconds is not None and mean is not None and mean > mean_wait_budget_seconds+1e-6:
        raise RuntimeError('aggregate service budget exceeded')
    if allow_incumbent and deadline is not None and monotonic() >= deadline:
        return dict(empty, status='evaluation_timeout', search_complete=False,
                    solver_status=solver.status_name(status), elapsed_seconds=monotonic()-started)
    provenance = dict(incumbent_recovered=recovered, optimality_proven=complete,
                      incumbent_solver_status=solver.status_name(incumbent_status),
                      optimized_stages=optimized_stages) if allow_incumbent else {}
    return dict(empty, slots=extracted,
        status='capacity_shortfall' if excess else 'optimal' if complete else 'search_incomplete',
        search_complete=complete, solver_status=solver.status_name(status),
        forecast_capacity_passed=excess == 0, estimated_mean_wait_seconds=mean,
        expected_frequency_work_units=frequency, elapsed_seconds=monotonic()-started,
        finite_catalog=True, terminal_positions_and_final_queue_verified=False,
        waiting_weights_scope='explicit waiting cohort' if any('up_wait_entries' in r or 'down_wait_entries' in r
                                                              for r in slots) else 'legacy directional entries',
        **provenance)
