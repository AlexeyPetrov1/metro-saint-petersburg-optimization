"""Assign known service jobs to trains without changing their clocks or demand."""

from math import isfinite

from ortools.sat.python import cp_model


RATES = {'baltiets': 9.2, 'yubileyny': 7.4}


def solve_fixed_assignments(jobs, availability, *, time_limit_seconds=5, line_train_limit=None):
    empty = dict(assignments={}, release_authorized=False, passenger_checked=False,
                 scope='fixed service clocks; type-weighted TO1 resource, not cash')
    if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError('positive search budget required')
    if len({j['job_id'] for j in jobs}) != len(jobs) or len({t['train_id'] for t in availability}) != len(availability):
        raise ValueError('distinct jobs and trains required')
    for j in jobs:
        if (not isfinite(j['km']) or j['km'] < 0 or j['ready_second'] <= j['departure_second']
                or j['origin'] == j['destination']):
            raise ValueError('invalid service job')
    for t in availability:
        if t['train_type'].lower() not in RATES: raise ValueError('unknown train type')
    model = cp_model.CpModel()
    kinds = list(RATES)
    incoming, outgoing, edges, starts, unused = {}, {}, {}, {}, {}
    exit_delay = {k: max((t.get('exit_seconds', 0) for t in availability if t['train_type'].lower() == k), default=0) for k in kinds}
    present = {(k,i): model.new_bool_var(f'job_{k}_{i}') for k in kinds for i in range(len(jobs))}
    for k in kinds:
        for i in range(len(jobs)):
            incoming[k,i], outgoing[k,i] = [], []
            edge = model.new_bool_var(f'end_{k}_{i}')
            edges[k,i,None] = edge; outgoing[k,i].append(edge)
        for i, a in enumerate(jobs):
            for j, b in enumerate(jobs):
                if a['destination'] == b['origin'] and a['ready_second'] <= b['departure_second']:
                    edge = model.new_bool_var(f'next_{k}_{i}_{j}')
                    edges[k,i,j] = edge; outgoing[k,i].append(edge); incoming[k,j].append(edge)
    for n, t in enumerate(availability):
        k = t['train_type'].lower()
        unused[n] = model.new_bool_var(f'unused_{n}')
        options = [unused[n]]
        for j, b in enumerate(jobs):
            if t['location'] == b['origin'] and t['ready_second'] <= b['departure_second']:
                edge = model.new_bool_var(f'start_{n}_{j}')
                starts[n,j] = edge; options.append(edge); incoming[k,j].append(edge)
        model.add_exactly_one(options)
    for i in range(len(jobs)):
        model.add_exactly_one(present[k,i] for k in kinds)
        for k in kinds:
            model.add(sum(incoming[k,i]) == present[k,i])
            model.add(sum(outgoing[k,i]) == present[k,i])
    if line_train_limit is not None:
        if isinstance(line_train_limit, bool) or not isinstance(line_train_limit, int) or line_train_limit <= 0:
            raise ValueError('positive integer line limit required')
        initial_count = sum(t.get('on_line', True) for t in availability)
        if initial_count > line_train_limit: raise ValueError('initial fleet exceeds line capacity')
        times = {j['ready_second']+exit_delay[k] for j in jobs for k in kinds}
        times.update(t['ready_second']+t.get('exit_seconds',0) for t in availability if t.get('on_line',True))
        times.update(jobs[j]['departure_second']-availability[n].get('entry_lead_seconds',0)
                     for n,j in starts if not availability[n].get('on_line',True))
        for at in times:
            entries = [e for (n,j),e in starts.items() if not availability[n].get('on_line',True)
                       and jobs[j]['departure_second']-availability[n].get('entry_lead_seconds',0) <= at]
            returns = [e for (k,i,j),e in edges.items() if j is None and jobs[i]['ready_second']+exit_delay[k] < at]
            idle_returns = [unused[n] for n,t in enumerate(availability) if t.get('on_line',True)
                            and t['ready_second']+t.get('exit_seconds',0) < at]
            model.add(initial_count+sum(entries)-sum(returns)-sum(idle_returns) <= line_train_limit)
    reserve_cost = sum(round((availability[n].get('entry_km',0)+availability[n].get('exit_km',0))
                              *RATES[availability[n]['train_type'].lower()]*10000)*e
                       for (n,j),e in starts.items() if not availability[n].get('on_line',True))
    model.minimize(sum(round(j['km']*RATES[k]*10000)*present[k,i]
                       for i,j in enumerate(jobs) for k in kinds)+reserve_cost)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(empty, status='infeasible' if status == cp_model.INFEASIBLE else 'search_incomplete',
                    search_complete=status == cp_model.INFEASIBLE)
    assignments, reserve_starts, used_reserves = {}, {}, []
    for n, t in enumerate(availability):
        k = t['train_type'].lower()
        at = next((j for (a,j), e in starts.items() if a == n and solver.value(e)), None)
        if at is not None and not t.get('on_line',True):
            reserve_starts[t['train_id']] = jobs[at]['departure_second']-t.get('entry_lead_seconds',0)
            used_reserves.append(t)
        while at is not None:
            identity = jobs[at]['job_id']
            if identity in assignments: raise ValueError('assignment paths overlap')
            assignments[identity] = t['train_id']
            at = next((j for (kind,i,j),e in edges.items() if kind == k and i == at and solver.value(e)), None)
    if set(assignments) != {j['job_id'] for j in jobs}: raise ValueError('incomplete assignment')
    types = {t['train_id']:t['train_type'].lower() for t in availability}
    return dict(empty, assignments=assignments, status='optimal' if status == cp_model.OPTIMAL else 'search_incomplete',
                search_complete=status == cp_model.OPTIMAL, reserve_starts=reserve_starts,
                train_km=sum(j['km'] for j in jobs)+sum(t.get('entry_km',0)+t.get('exit_km',0) for t in used_reserves),
                to1_resource_equivalent_rub=sum(j['km']*RATES[types[assignments[j['job_id']]]] for j in jobs)
                    +sum((t.get('entry_km',0)+t.get('exit_km',0))*RATES[t['train_type'].lower()] for t in used_reserves),
                seconds=solver.wall_time)
