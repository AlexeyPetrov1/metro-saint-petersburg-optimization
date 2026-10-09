"""Small exact dispatch model. Every requested job is mandatory; validation is separate."""

from time import monotonic

from ortools.sat.python import cp_model


RATES = {'baltiets': 9.2, 'yubileyny': 7.4}


def solve_joint_dispatch(jobs, availability, *, as_of=0, fixed_terminal_intervals=(),
                         fixed_standby_intervals=(), fixed_line_intervals=(),
                         fixed_departures=(), line_limit=53, deadline=None,
                         time_limit_seconds=5, clock_stability=False):
    """Choose bounded clocks and concrete complete vehicle paths, with no skipped jobs.

    run_seconds includes the 180-second destination turn. Depot readiness must be
    explicit. Waiting is a supplied local weight, not a passenger simulation.
    """
    began = monotonic()
    if deadline is not None and began >= deadline:
        raise TimeoutError('joint dispatch deadline')
    jobs, fleet = list(jobs), list(availability)
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate mandatory job')
    if len({t['train_id'] for t in fleet}) != len(fleet):
        raise ValueError('duplicate train')
    for j in jobs:
        if j['earliest_second'] < as_of or j['latest_second'] < j['earliest_second']:
            raise ValueError('invalid future departure window')
        if j['run_seconds'] < 180 or j.get('minimum_headway_seconds', 150) < 113:
            raise ValueError('physical turn/headway bound')
        if j.get('max_run_seconds', j['run_seconds']) < j['run_seconds']:
            raise ValueError('invalid route-duration window')
        if j['km'] < 0 or j.get('waiting_weight', 1) < 0:
            raise ValueError('negative objective input')
    ready = []
    for t in fleet:
        if t['train_type'].lower() not in RATES:
            raise ValueError('unknown train type')
        if t.get('on_line', True):
            ready.append(max(as_of, int(t['ready_second'])))
        else:
            if t.get('depot_ready_second') is None:
                raise ValueError('explicit depot readiness required')
            if t.get('release_seconds') not in (720, 780):
                raise ValueError('depot release must be 720 or 780 seconds')
            ready.append(max(as_of, int(t['depot_ready_second'])) + t['release_seconds'])
    empty = dict(assignments={}, departures={}, ready_seconds={}, reserve_entries={}, mandatory_jobs=len(jobs),
                 mandatory_jobs_covered=0, train_km=None, to1_resource_equivalent_rub=None,
                 physical_validation_required=True, waiting_is_proxy=True,
                 scope='mandatory bounded service jobs and concrete train paths')
    if not jobs:
        return dict(empty, status='optimal', mandatory_jobs_covered=0, train_km=0,
                    to1_resource_equivalent_rub=0, seconds=monotonic()-began)
    horizon = max([as_of+1] + ready + [j['latest_second']+j.get('max_run_seconds',j['run_seconds'])+300 for j in jobs]
                  + [b for a,b,d in fixed_line_intervals]) + 300
    model = cp_model.CpModel()
    dep = [model.new_int_var(j['earliest_second'], j['latest_second'], f'd{i}')
           for i,j in enumerate(jobs)]
    finish = [model.new_int_var(j['earliest_second']+j['run_seconds'],
              j['latest_second']+j.get('max_run_seconds',j['run_seconds']), f'r{i}') for i,j in enumerate(jobs)]
    for i,j in enumerate(jobs):
        model.add(finish[i] >= dep[i]+j['run_seconds'])
        model.add(finish[i] <= dep[i]+j.get('max_run_seconds',j['run_seconds']))
    terminals, standby, line = {}, {}, []

    def interval(start, end, present, name):
        size = model.new_int_var(0, horizon, name+'size')
        if present is None:
            model.add(size == end-start)
            return model.new_interval_var(start, size, end, name)
        model.add(size == end-start).only_enforce_if(present)
        return model.new_optional_interval_var(start, size, end, present, name)

    def visit(location, start, end, present, name, parking=False):
        table = standby if parking else terminals
        table.setdefault(location, []).append(interval(start, end, present, name))

    for location,a,b in fixed_terminal_intervals:
        if b > max(a, as_of): visit(location, max(a, as_of), b, None, 'fixedvisit')
    for location,a,b in fixed_standby_intervals:
        if b > max(a, as_of): visit(location, max(a, as_of), b, None, 'fixedpark', True)
    for a,b,demand in fixed_line_intervals:
        if b > max(a, as_of): line.append((interval(max(a, as_of), b, None, 'fixedline'), demand))
    assigned, starts, ends, edges, used = {}, {}, {}, {}, {}
    for n,t in enumerate(fleet):
        if deadline is not None and monotonic() >= deadline: raise TimeoutError('joint dispatch deadline')
        used[n] = model.new_bool_var(f'used{n}')
        service = []
        for i,j in enumerate(jobs):
            assigned[n,i] = model.new_bool_var(f'x{n}_{i}')
            service.append(interval(dep[i], finish[i], assigned[n,i], f's{n}_{i}'))
            ends[n,i] = model.new_bool_var(f'end{n}_{i}')
            if t['location'] == j['origin'] and ready[n] <= j['latest_second']:
                starts[n,i] = model.new_bool_var(f'start{n}_{i}')
                model.add(dep[i] >= ready[n]).only_enforce_if(starts[n,i])
                if t.get('on_line', True):
                    visit(t['location'], max(as_of, t.get('visit_start_second', ready[n]-180)),
                          dep[i], starts[n,i], f'first{n}_{i}')
                    visit(t['location'], ready[n], dep[i], starts[n,i], f'park{n}_{i}', True)
                else:
                    visit(t['location'], dep[i]-180, dep[i], starts[n,i], f'entry{n}_{i}')
            visit(j['destination'], finish[i]-180, finish[i], ends[n,i], f'last{n}_{i}')
        for i,a in enumerate(jobs):
            for k,b in enumerate(jobs):
                if i == k or a['destination'] != b['origin']:
                    continue
                if a['earliest_second']+a['run_seconds'] > b['latest_second']:
                    continue
                edge = edges[n,i,k] = model.new_bool_var(f'e{n}_{i}_{k}')
                model.add(dep[k] >= finish[i]).only_enforce_if(edge)
                visit(a['destination'], finish[i]-180,
                      dep[k], edge, f'visit{n}_{i}_{k}')
                visit(a['destination'], finish[i],
                      dep[k], edge, f'idle{n}_{i}_{k}', True)
        for i in range(len(jobs)):
            incoming = [e for (v,a,b),e in edges.items() if v == n and b == i]
            outgoing = [e for (v,a,b),e in edges.items() if v == n and a == i]
            model.add(sum(incoming)+starts.get((n,i), 0) == assigned[n,i])
            model.add(sum(outgoing)+ends[n,i] == assigned[n,i])
        model.add(sum(e for (v,i),e in starts.items() if v == n) == used[n])
        model.add(sum(e for (v,i),e in ends.items() if v == n) == used[n])
        model.add_no_overlap(service)
        exit_end = model.new_int_var(as_of, horizon, f'exit{n}')
        for i,j in enumerate(jobs):
            model.add(exit_end == finish[i]+t.get('exit_seconds', 120)+1).only_enforce_if(ends[n,i])
        model.add(exit_end == ready[n]+t.get('exit_seconds', 120)+1).only_enforce_if(used[n].Not())
        if t.get('on_line', True):
            visit(t['location'], max(as_of, t.get('visit_start_second', ready[n]-180)),
                  ready[n], used[n].Not(), f'unused{n}')
            line.append((interval(as_of, exit_end, None, f'line{n}'), 1))
        else:
            entry = model.new_int_var(as_of, horizon, f'entry{n}')
            for (v,i),start in starts.items():
                if v == n: model.add(entry == dep[i]-t.get('entry_lead_seconds', 300)).only_enforce_if(start)
            line.append((interval(entry, exit_end, used[n], f'reserve{n}'), 1))
    for i in range(len(jobs)):
        model.add(sum(assigned[n,i] for n in range(len(fleet))) == 1)

    def gap(left, right, minimum, enabled=None):
        constraint = model.add(right-left >= minimum)
        if enabled is not None: constraint.only_enforce_if(enabled)

    def ordered_gap(a, aw, amin, b, bw, bmin):
        if aw[1] <= bw[0]: gap(a, b, amin)
        elif bw[1] <= aw[0]: gap(b, a, bmin)
        else:
            order = model.new_bool_var('order')
            gap(a, b, amin, order); gap(b, a, bmin, order.Not())
    for i,a in enumerate(jobs):
        window = a['earliest_second'], a['latest_second']
        minimum = a.get('minimum_headway_seconds', 150)
        for k in range(i):
            b = jobs[k]
            if a['origin'] == b['origin']:
                ordered_gap(dep[i], window, minimum, dep[k],
                            (b['earliest_second'], b['latest_second']), b.get('minimum_headway_seconds', 150))
        for location,clock,other_minimum in fixed_departures:
            if location == a['origin']:
                ordered_gap(dep[i], window, minimum, clock, (clock, clock), other_minimum)
    for spans in terminals.values(): model.add_cumulative(spans, [1]*len(spans), 2)
    for spans in standby.values(): model.add_cumulative(spans, [1]*len(spans), 1)
    if line: model.add_cumulative([i for i,d in line], [d for i,d in line], line_limit)
    resource = sum(round(j['km']*RATES[t['train_type'].lower()]*10000)*assigned[n,i]
                   for n,t in enumerate(fleet) for i,j in enumerate(jobs))
    resource += sum(round((t.get('entry_km', 0)+t.get('exit_km', 0))*RATES[t['train_type'].lower()]*10000)*used[n]
                    for n,t in enumerate(fleet) if not t.get('on_line', True))
    max_wait = sum(round(j.get('waiting_weight', 1))*(j['latest_second']-j['earliest_second']) for j in jobs)
    waiting = sum(round(j.get('waiting_weight', 1))*(dep[i]-j['earliest_second']) for i,j in enumerate(jobs))
    if clock_stability:
        deviations,max_wait = [],0
        for i,j in enumerate(jobs):
            anchor = j.get('original_departure_second',(j['earliest_second']+j['latest_second'])//2)
            bound = max(abs(j['earliest_second']-anchor),abs(j['latest_second']-anchor))
            deviation = model.new_int_var(0,bound,f'change{i}')
            model.add_abs_equality(deviation,dep[i]-anchor)
            weight = round(j.get('waiting_weight',1))
            deviations.append(weight*deviation);max_wait += weight*bound
        waiting = sum(deviations)
    max_wait += sum(j.get('max_run_seconds',j['run_seconds'])-j['run_seconds'] for j in jobs)
    waiting += sum(finish[i]-dep[i]-j['run_seconds'] for i,j in enumerate(jobs))
    model.minimize(resource*(max_wait+1)+waiting)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(0.001, min(time_limit_seconds,
        deadline-monotonic() if deadline is not None else time_limit_seconds))
    if deadline is not None and monotonic() >= deadline: raise TimeoutError('joint dispatch deadline')
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(empty, status='infeasible' if status == cp_model.INFEASIBLE else 'timeout',
                    search_complete=status == cp_model.INFEASIBLE, seconds=monotonic()-began,
                    solver_status=solver.status_name(status))
    assignments = {j['job_id']: fleet[n]['train_id'] for i,j in enumerate(jobs)
                   for n in range(len(fleet)) if solver.value(assigned[n,i])}
    departures = {j['job_id']: solver.value(dep[i]) for i,j in enumerate(jobs)}
    reserves = {t['train_id']: departures[jobs[i]['job_id']]-t.get('entry_lead_seconds', 300)
                for (n,i),start in starts.items() for t in [fleet[n]]
                if not t.get('on_line', True) and solver.value(start)}
    by_id = {t['train_id']:t for t in fleet}
    used_reserves = [by_id[tid] for tid in reserves]
    cost = sum(j['km']*RATES[by_id[assignments[j['job_id']]]['train_type'].lower()] for j in jobs)
    cost += sum((t.get('entry_km', 0)+t.get('exit_km', 0))*RATES[t['train_type'].lower()] for t in used_reserves)
    return dict(empty, assignments=assignments, departures=departures,
                ready_seconds={j['job_id']:solver.value(finish[i]) for i,j in enumerate(jobs)}, reserve_entries=reserves,
                mandatory_jobs_covered=len(assignments), status='optimal' if status == cp_model.OPTIMAL else 'feasible',
                search_complete=status == cp_model.OPTIMAL, seconds=monotonic()-began,
                train_km=sum(j['km'] for j in jobs)+sum(t.get('entry_km', 0)+t.get('exit_km', 0) for t in used_reserves),
                to1_resource_equivalent_rub=cost, waiting_proxy=solver.value(waiting),
                secondary_objective='clock_stability_and_route_padding' if clock_stability else 'earliest_departure_and_route_padding',
                secondary_upper_bound=max_wait,resource_lex_multiplier=max_wait+1,
                solver_status=solver.status_name(status))


def _fixed_resources(data, motion, excluded):
    """Exact occupied spans of untouched trains, from their known movement plan."""
    stations = {r['station_id']:r for r in data['stations']}
    tracks = {r['track_id']:r for r in data['tracks']}
    terminals = data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    opened, parking, online = {}, {}, {}
    visits, waits, lines = [], [], []
    def active(state, location):
        return any(stations[tracks[location][side]]['kind'] == 'line' for side in ('from_station','to_station')) \
            if state == 'moving' else stations[location]['kind'] == 'line'
    def close(table, output, tid, second):
        if tid in table:
            location,start = table.pop(tid)
            if start < second: output.append((location,start,second))
    for t in data['trains']:
        tid,location = t['train_id'],t['location']
        if tid in excluded: continue
        if location in terminals and t['state'] != 'moving': opened[tid] = location,0
        if location in terminals and t['state'] == 'standby': parking[tid] = location,0
        if active(t['state'], location): online[tid] = 0
    for e in motion['events']:
        tid,second,location = e['train_id'],e['second'],e['location']
        if tid in excluded: continue
        if e['event_type'] == 'departure': close(opened, visits, tid, second)
        if e['event_type'] == 'arrival' and location in terminals: opened[tid] = location,second
        close(parking, waits, tid, second)
        if e['next_state'] == 'standby' and location in terminals: parking[tid] = location,second
        if active(e['next_state'], location): online.setdefault(tid, second)
        elif tid in online: lines.append((online.pop(tid),second,1))
    end = motion['duration_seconds']
    for table,output in ((opened,visits),(parking,waits)):
        for tid in list(table): close(table,output,tid,end)
    lines.extend((a,end,1) for a in online.values() if a < end)
    return visits,waits,lines


def build_joint_problem(data, as_of, selected_train_ids, *, window_seconds=30, source_motion=None):
    """Free all unstarted SERVICE jobs of selected active trains; other duties stay fixed.

    Readiness and durations come from the known plan, not future passenger truth.
    Begun whole trips and existing preparation obligations remain committed.
    Only explicitly declared reserves may enter; depot returns cannot re-enter.
    """
    from datetime import timedelta
    from dwell import timing_key
    from goal43_policy import minimum_headway
    from metro44_assignment_plan import build_assignment_problem
    from simulation import instant
    if not 0 <= window_seconds <= 60: raise ValueError('time flex must be 0..60 seconds')
    selected = set(selected_train_ids)
    problem = build_assignment_problem(data, as_of)
    motion = source_motion or problem['source_motion']
    availability = [dict(t) for t in problem['availability'] if t['train_id'] in selected]
    if {t['train_id'] for t in availability} != selected or any(t.get('locked_exit') for t in availability):
        raise ValueError('first joint adapter requires selected active terminal-ready chains')
    declared = {r['train_id']:r for r in data['headway_policy']['demo']['reserves']}
    source_trains = {r['train_id']:r for r in data['trains']}
    for t in availability:
        if t['on_line']: t['visit_start_second'] = t['ready_second']-180
        else:
            record = declared.get(t['train_id'])
            if not record or record.get('status') == 'unknown': raise ValueError('undeclared reserve readiness')
            t['depot_ready_second'] = source_trains[t['train_id']]['ready_after_seconds']
            t['readiness_status'] = record['status']
    tracks = {t['track_id']:t for t in data['tracks']}
    stations = {s['station_id']:s for s in data['stations']}
    trips = {t['trip_id']:t for t in data['plans'][0]['trips']}
    jobs = []
    start = instant(data['period_start']);policy = data['headway_policy']
    for row in problem['jobs']:
        if row['original_train_id'] not in selected: continue
        clock = row['departure_second']
        earliest,latest = max(as_of,clock-window_seconds),clock+window_seconds
        minimum = minimum_headway(start+timedelta(seconds=earliest),policy)
        if minimum != minimum_headway(start+timedelta(seconds=latest),policy):
            earliest = latest = clock
            minimum = minimum_headway(start+timedelta(seconds=clock),policy)
        nominal = sum(tracks[track]['travel_seconds'] + (stations[tracks[track]['to_station']]['turnaround_seconds']
                      or stations[tracks[track]['to_station']]['stop_seconds']) for track in trips[row['job_id']]['route'])
        elapsed = row['ready_second']-clock
        jobs.append(dict(job_id=row['job_id'], original_train_id=row['original_train_id'],
            original_departure_second=clock, earliest_second=earliest, latest_second=latest,
            run_seconds=nominal,max_run_seconds=max(nominal,elapsed+window_seconds),source_elapsed_seconds=elapsed,
            source_ready_second=row['ready_second'],origin=row['origin'], destination=row['destination'],
            km=row['km'], minimum_headway_seconds=minimum))
    if not jobs: raise ValueError('no selected future SERVICE jobs')
    visits,waits,lines = _fixed_resources(data,motion,selected)
    job_ids = {j['job_id'] for j in jobs}
    tracks = {t['track_id']:t for t in data['tracks']}
    timing = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    fixed = [(tracks[t['route'][0]]['from_station'],
              timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second'],
              minimum_headway(start+timedelta(seconds=timing[timing_key(t['train_id'],t['trip_id'],0)]['departure_second']),policy))
             for t in data['plans'][0]['trips'] if t['passenger'] and t['trip_id'] not in job_ids]
    return dict(jobs=jobs, availability=availability, as_of=as_of, job_ids=sorted(job_ids),
                retained_ids=problem['retained_ids'], selected_train_ids=sorted(selected),
                source_motion=motion, fixed_terminal_intervals=visits, fixed_standby_intervals=waits,
                fixed_line_intervals=lines, fixed_departures=fixed,
                scope='selected complete vehicle chains and declared reserves; all source SERVICE jobs preserved')


def apply_joint_dispatch(data, problem, solution, *, service_end_second=None, deadline=None,
                         profile='fixed', seed=7, repair_terminal_holds=False,allow_future_exit_retiming=False):
    """Compile a complete joint assignment and refuse any violated physical contract."""
    import copy
    from dwell import timing_key
    from metro44_assignment_plan import terminal_resources, _current_movements_preserved
    from metro44_terminal_timing import compile_terminal_timing
    from rolling import operational_headways
    from simulation import simulate, digest
    def check_deadline():
        if deadline is not None and monotonic() >= deadline: raise TimeoutError('joint adapter deadline')
    check_deadline()
    jobs = {j['job_id']:j for j in problem['jobs']}
    assignments,departures = solution['assignments'],solution['departures']
    if set(assignments) != set(jobs) or set(departures) != set(jobs):
        raise ValueError('joint solution must cover every mandatory job')
    available = {t['train_id']:t for t in problem['availability']}
    if set(assignments.values())-set(available): raise ValueError('unavailable train')
    for identity,j in jobs.items():
        if not j['earliest_second'] <= departures[identity] <= j['latest_second']:
            raise ValueError('joint departure outside declared window')
    as_of = problem['as_of'];selected = set(problem['selected_train_ids'])
    retained = set(problem['retained_ids']);out = copy.deepcopy(data)
    tracks = {r['track_id']:r for r in data['tracks']}
    old = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in data['leg_timing']}
    # Selected unstarted clocks have no inherited lower bounds. Committed duties do.
    removed = {t['trip_id'] for t in data['plans'][0]['trips'] if t['train_id'] in selected
               and t['trip_id'] not in retained and not t['passenger']}
    out['plans'][0]['trips'] = [t for t in out['plans'][0]['trips'] if t['trip_id'] not in removed]
    out['leg_timing'] = [r for r in out['leg_timing'] if r['trip_id'] not in jobs and r['trip_id'] not in removed]
    bounds = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r['departure_second']
              for r in out['leg_timing'] if r['departure_second'] >= 0}
    for t in out['plans'][0]['trips']:
        identity = t['trip_id']
        if identity not in jobs: continue
        t.update(train_id=assignments[identity], departure_second=departures[identity])
        bounds[timing_key(t['train_id'],identity,0)] = departures[identity]
        # Preserve known route timing shape, shifted within the free window.
        source = jobs[identity];delta = departures[identity]-source['original_departure_second']
        for i in range(len(t['route'])):
            row = copy.deepcopy(old[timing_key(source['original_train_id'],identity,i)])
            row.update(train_id=t['train_id'], departure_second=row['departure_second']+delta)
            if i == len(t['route'])-1:
                arrival = solution.get('ready_seconds', {}).get(identity,source['source_ready_second']+delta)-180
                travel = tracks[t['route'][i]]['travel_seconds']
                row['departure_second'] = min(row['departure_second'], arrival-travel)
                row['approach_holding_seconds'] = arrival-row['departure_second']-travel
            out['leg_timing'].append(row)
    for tid,state in available.items():
        owned = sorted((jobs[j] for j,train in assignments.items() if train == tid),
                       key=lambda j:departures[j['job_id']])
        location,ready = state['location'],state['ready_second']
        if owned and not state['on_line']:
            first = departures[owned[0]['job_id']]
            identity = f'G45:{as_of}:{tid}:ENTRY';entry = first-state['entry_lead_seconds']
            out['plans'][0]['trips'].append(dict(trip_id=identity,train_id=tid,departure_second=entry,
                                               route=[state['entry_track']],passenger=False))
            bounds[timing_key(tid,identity,0)] = entry
        for j in owned:
            if location != j['origin'] or ready > departures[j['job_id']]:
                raise ValueError('joint path is not physically ready')
            location = j['destination']
            ready = solution.get('ready_seconds', {}).get(j['job_id'], j['source_ready_second']+
                    departures[j['job_id']]-j['original_departure_second'])
        if not state['on_line'] and not owned: continue
        exit_track = next(t['track_id'] for t in tracks.values()
                          if t['from_station'] == location and t['track_id'].startswith('EXIT_'))
        identity = f'G45:{as_of}:{tid}:EXIT'
        out['plans'][0]['trips'].append(dict(trip_id=identity,train_id=tid,departure_second=ready,
                                           route=[exit_track],passenger=False))
        bounds[timing_key(tid,identity,0)] = ready
    check_deadline()
    out,timing = compile_terminal_timing(out,out['plans'][0]['plan_id'],profile,seed,
        policy=data['headway_policy'],as_of=as_of,departure_not_before=bounds)
    check_deadline();motion = simulate(out,out['plans'][0]['plan_id'])
    hold_audit = dict(passed=True,repairs=[],enabled=repair_terminal_holds)
    if repair_terminal_holds:
        out,hold_audit,motion = repair_joint_holds(out,as_of,deadline=deadline)
    before = problem['source_motion']
    prefix = [e for e in before['events'] if e['second'] < as_of] == [e for e in motion['events'] if e['second'] < as_of]
    current = _current_movements_preserved(data,out,as_of)
    actual = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in out['leg_timing']}
    unchanged = departure_scope(data,out,as_of,jobs,removed,allow_future_exit_retiming=allow_future_exit_retiming)
    preserved = unchanged['preserved']
    passenger_ids = lambda value: {t['trip_id'] for t in value['plans'][0]['trips'] if t['passenger']}
    coverage = passenger_ids(data) == passenger_ids(out)
    clocks = {identity:actual[timing_key(assignments[identity],identity,0)]['departure_second'] for identity in jobs}
    windows = all(j['earliest_second'] <= clocks[identity] <= j['latest_second'] for identity,j in jobs.items())
    resources = terminal_resources(out,motion)
    gaps = operational_headways(out,motion,as_of,service_end_second or motion['duration_seconds'])
    if service_end_second is None:
        gaps['violations'] = [r for r in gaps['violations'] if r['kind'] != 'boundary_end']
        gaps['passed'] = not gaps['violations'] and not gaps['unknown_tracks']
    audit = dict(movement_passed=prefix and current and preserved and coverage and windows
                 and resources['passed'] and gaps['passed'] and motion['obligations_closed'],
        executed_prefix_preserved=prefix,current_movements_preserved=current,
        untouched_departures_preserved=preserved,mandatory_jobs_preserved=coverage,
        future_exit_retiming_allowed=allow_future_exit_retiming,departure_scope=unchanged,
        actual_departures=clocks,departure_windows_passed=windows,terminal_resources=resources,
        headways={k:v for k,v in gaps.items() if k not in ('intervals','boundary_ages')},
        obligations_closed=motion['obligations_closed'],peak_trains_on_line=motion['checks']['peak_trains_on_line'],
        selected_train_ids=sorted(selected), scope=problem['scope'],
        terminal_holding_repair=hold_audit,
        past_and_current_frozen=True,unstarted_departures_may_advance=True,
        passenger_checked=False,release_authorized=False,cash_saving_rub=None,
        _data_digest=digest(out),_motion_digest=digest(motion),_transport_work=motion['transport_work'])
    check_deadline();return out,audit


def repair_joint_holds(data, as_of, *, max_repairs=12, deadline=None):
    """Delay an unstarted inbound arrival until its real terminal place frees.

    Departures and vehicle assignment stay fixed. No overtaking or committed
    arrival may change; every hold is checked by the shared movement validator.
    """
    import copy
    from metro44_assignment_plan import terminal_resources
    from simulation import simulate
    out = copy.deepcopy(data);repairs = []
    trips = {t['trip_id']:t for t in out['plans'][0]['trips']}
    tracks = {t['track_id']:t for t in out['tracks']}
    terminals = data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    for attempt in range(max_repairs+1):
        if deadline is not None and monotonic() >= deadline: raise TimeoutError('joint holding deadline')
        motion = simulate(out,out['plans'][0]['plan_id']);resources = terminal_resources(out,motion)
        if resources['passed']: return out,dict(passed=True,repairs=repairs),motion
        failures = [(r,'visit') for r in resources['visit_violations']] + [(r,'standby') for r in resources['parking_violations']]
        failure,kind = min(failures,key=lambda value:value[0]['second'])
        if attempt == max_repairs: raise ValueError('joint terminal holding budget exhausted')
        at,terminal = failure['second'],failure['terminal']
        visits,parking = {},{}
        arrivals = {}
        for t in out['trains']:
            if t['location'] in terminals and t['state'] != 'moving': visits[t['train_id']] = t['location']
            if t['location'] in terminals and t['state'] == 'standby': parking[t['train_id']] = t['location']
        for e in motion['events']:
            if e['second'] > at: break
            tid = e['train_id']
            if e['event_type'] == 'departure': visits.pop(tid,None)
            if e['event_type'] == 'arrival' and e['location'] in terminals:
                visits[tid] = e['location'];arrivals[tid] = e
            parking.pop(tid,None)
            if e['next_state'] == 'standby' and e['location'] in terminals: parking[tid] = e['location']
        occupants = [tid for tid,location in (visits if kind == 'visit' else parking).items() if location == terminal]
        candidates = sorted((arrivals[tid] for tid in occupants if tid in arrivals),key=lambda e:e['second'],reverse=True)
        changed = False
        for arrival in candidates:
            tid,identity = arrival['train_id'],arrival['trip_id']
            trip = trips[identity];last = len(trip['route'])-1
            leg = next(r for r in out['leg_timing'] if r['train_id']==tid and r['trip_id']==identity and r['leg_index']==last)
            if leg['departure_second'] < as_of: continue
            other_release = min(e['second'] for e in motion['events'] if e['event_type']=='departure'
                                and e['train_id'] in occupants and e['train_id'] != tid and e['second'] > at)
            ready = arrival['second']+leg['service_seconds']
            delay = other_release+1-(arrival['second'] if kind == 'visit' else ready)
            own_release = next(e['second'] for e in motion['events'] if e['event_type']=='departure'
                               and e['train_id']==tid and e['second'] > arrival['second'])
            if delay <= 0 or ready+delay > own_release: continue
            track = trip['route'][last]
            follower = [e['second'] for e in motion['events'] if e['event_type']=='arrival' and e['second'] > arrival['second']
                        and e['location']==terminal and e['trip_id'] in trips and trips[e['trip_id']]['route'][-1] == track]
            if follower and arrival['second']+delay >= min(follower): continue
            leg['approach_holding_seconds'] = leg.get('approach_holding_seconds',0)+delay
            repairs.append(dict(train_id=tid,trip_id=identity,leg_index=last,kind=kind,terminal=terminal,
                                departure_second=leg['departure_second'],old_arrival_second=arrival['second'],
                                new_arrival_second=arrival['second']+delay,holding_added_seconds=delay))
            changed = True;break
        if not changed: raise ValueError('terminal hold cannot change committed movement or overtake its follower')


def propose_joint_dispatch(data, as_of, selected_train_ids, *, window_seconds=60,
                           time_limit_seconds=60, service_end_second=None, deadline=None,
                           profile='fixed', seed=7, clock_stability=False,allow_future_exit_retiming=False):
    """Bounded controller producer: verified candidate, or an explicit failure."""
    began = monotonic();deadline = min(deadline, began+470) if deadline is not None else began+470
    result = None
    try:
        if monotonic() >= deadline: raise TimeoutError('joint producer deadline')
        problem = build_joint_problem(data,as_of,selected_train_ids,window_seconds=window_seconds)
        result = solve_joint_dispatch(problem['jobs'],problem['availability'],as_of=as_of,
            fixed_terminal_intervals=problem['fixed_terminal_intervals'],
            fixed_standby_intervals=problem['fixed_standby_intervals'],
            fixed_line_intervals=problem['fixed_line_intervals'],fixed_departures=problem['fixed_departures'],
            time_limit_seconds=time_limit_seconds,deadline=deadline,clock_stability=clock_stability)
        if not result['assignments']:
            return None,dict(status=result['status'],solver=result,movement_passed=False,
                             reason='no complete mandatory dispatch found',seconds=monotonic()-began,
                             cash_saving_rub=None,release_authorized=False)
        out,audit = apply_joint_dispatch(data,problem,result,deadline=deadline,profile=profile,seed=seed,
            service_end_second=service_end_second,repair_terminal_holds=True,
            allow_future_exit_retiming=allow_future_exit_retiming)
        audit.update(solver=result,status='verified' if audit['movement_passed'] else 'physical_rejected',
                     seconds=monotonic()-began)
        return (out if audit['movement_passed'] else None),audit
    except (ValueError,TimeoutError) as error:
        return None,dict(status='timeout' if isinstance(error,TimeoutError) else 'physical_rejected',
                         movement_passed=False,solver=result,reason=str(error),seconds=monotonic()-began,
                         cash_saving_rub=None,release_authorized=False)


def departure_scope(source, candidate, as_of, job_ids=(), removed_ids=(), *, allow_future_exit_retiming=False):
    """Allow a declared future empty return exception, preserving every SERVICE clock."""
    from dwell import timing_key
    actual = {timing_key(r['train_id'],r['trip_id'],r['leg_index']):r for r in candidate['leg_timing']}
    trips = {t['trip_id']:t for t in source['plans'][0]['trips']}
    first = {r['trip_id']:r['departure_second'] for r in source['leg_timing'] if r['leg_index']==0}
    excluded = set(job_ids)|set(removed_ids);changes,allowed = [],[]
    for row in source['leg_timing']:
        identity = row['trip_id']
        if identity in excluded: continue
        clock = actual.get(timing_key(row['train_id'],identity,row['leg_index']),{}).get('departure_second')
        if clock == row['departure_second']: continue
        trip = trips.get(identity,{})
        change = dict(train_id=row['train_id'],trip_id=identity,leg_index=row['leg_index'],
                      source_departure_second=row['departure_second'],candidate_departure_second=clock,
                      delay_seconds=clock-row['departure_second'] if clock is not None else None)
        legal = allow_future_exit_retiming and not trip.get('passenger',True) and trip.get('route') \
            and all(track.startswith('EXIT_') for track in trip['route']) \
            and first[identity] >= as_of and clock is not None and clock >= as_of
        (allowed if legal else changes).append(change)
    return dict(preserved=not changes,violations=changes,allowed_exit_changes=allowed,
                scope='free wholly-unstarted nonpassenger EXIT only; ENTRY/SERVICE/past/current protected')
