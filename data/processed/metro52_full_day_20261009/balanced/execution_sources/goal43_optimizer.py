"""Bounded aggregate CP-SAT plans; passenger and infrastructure checks stay in the simulator."""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class DepartureSolution:
    times: Mapping[str, int]
    status: str
    solver_status: str
    elapsed_seconds: float
    search_complete: bool
    objective: tuple[int, int] | None = None
    validated_scope: str = "aggregate_departures_only"


@dataclass(frozen=True)
class WithdrawalSolution:
    selected_train_ids: tuple[str, ...]
    remaining_count: int | None
    saving_units: int | None
    status: str
    solver_status: str
    elapsed_seconds: float
    search_complete: bool
    validated_scope: str = "aggregate_fleet_only"


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _budget(started, time_limit_seconds, deadline):
    if not isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be finite and positive")
    if deadline is not None and not isfinite(deadline):
        raise ValueError("deadline must be a finite monotonic-clock value")
    return min(started + time_limit_seconds, deadline if deadline is not None else float("inf"))


def _solver(end):
    remaining = end - monotonic()
    if remaining <= 0:
        return None
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    return solver


def _status(status):
    return {cp_model.OPTIMAL: ("optimal", True), cp_model.FEASIBLE: ("search_incomplete", False),
            cp_model.INFEASIBLE: ("no_feasible_candidate", True),
            cp_model.UNKNOWN: ("evaluation_timeout", False)}.get(status, ("model_invalid", False))


def solve_departures(departures, *, min_headway_seconds=113, max_headway_seconds=480,
                     max_hold_seconds=480, fixed_departures=None, links=(),
                     horizon_end_seconds=None, target_headway_seconds=None,
                     headway_windows=(), time_limit_seconds=10.0, deadline=None):
    """Return complete aggregate departure times, or no times on infeasibility/timeout.

    Rows contain event_id/train_id/station_id/direction/scheduled_seconds/earliest_seconds.
    Optional latest_seconds and min_headway_seconds further constrain the row. Times never
    advance before the schedule. Fixed rows anchor committed departures. Links contain
    (before_id, after_id, minimum_elapsed_seconds), e.g. travel plus a terminal turnaround.
    Platform order follows the scheduled order, not a newly sorted optimized timetable.
    Windows contain (start, exclusive_end, minimum_headway), selected by actual departure.
    The optional target minimizes total gap deviation first, then total holding time.
    This model knows neither passengers nor full infrastructure, so it is not a recommendation.
    """
    started = monotonic()
    end = _budget(started, time_limit_seconds, deadline)
    minimum = _integer(min_headway_seconds, "min_headway_seconds", 1)
    maximum = _integer(max_headway_seconds, "max_headway_seconds", minimum)
    hold = _integer(max_hold_seconds, "max_hold_seconds", 0)
    horizon = None if horizon_end_seconds is None else _integer(horizon_end_seconds, "horizon_end_seconds")
    target = None if target_headway_seconds is None else _integer(target_headway_seconds, "target_headway_seconds", 1)
    windows = sorted(tuple(_integer(v, "headway_window") for v in window) for window in headway_windows)
    for i, window in enumerate(windows):
        if len(window) != 3 or window[0] >= window[1] or window[2] < 1:
            raise ValueError("headway_windows require (start, end, positive minimum)")
        if i and windows[i-1][1] > window[0]:
            raise ValueError("headway_windows overlap")
    rows, groups = {}, defaultdict(list)
    for raw in departures:
        row = dict(raw)
        event = _identifier(row["event_id"], "event_id")
        if event in rows:
            raise ValueError(f"duplicate event_id {event}")
        for key in ("train_id", "station_id", "direction"):
            row[key] = _identifier(row[key], key)
        row["scheduled_seconds"] = _integer(row["scheduled_seconds"], "scheduled_seconds")
        row["earliest_seconds"] = _integer(row["earliest_seconds"], "earliest_seconds")
        row["min_headway_seconds"] = _integer(row.get("min_headway_seconds", minimum), "min_headway_seconds", minimum)
        if "latest_seconds" in row:
            row["latest_seconds"] = _integer(row["latest_seconds"], "latest_seconds")
        rows[event] = row
        groups[(row["station_id"], row["direction"])].append(event)
    fixed = dict(fixed_departures or {})
    if fixed.keys() - rows.keys():
        raise ValueError("unknown event in fixed_departures")
    fixed = {event: _integer(value, "fixed_departure_seconds") for event, value in fixed.items()}
    links = tuple((before, after, _integer(elapsed, "link_elapsed_seconds", 0)) for before, after, elapsed in links)
    if any(before not in rows or after not in rows for before, after, _ in links):
        raise ValueError("unknown event in links")

    def result(status, solver_status, complete=False, times=None, objective=None):
        return DepartureSolution(MappingProxyType(dict(times or {})), status, solver_status,
                                 monotonic()-started, complete, objective)

    if monotonic() >= end:
        return result("evaluation_timeout", "NOT_STARTED")
    model, variables, bounds = cp_model.CpModel(), {}, {}
    for event, row in rows.items():
        lower = max(row["scheduled_seconds"], row["earliest_seconds"])
        upper = min(row["scheduled_seconds"]+hold, row.get("latest_seconds", row["scheduled_seconds"]+hold))
        if horizon is not None:
            upper = min(upper, horizon)
        if event in fixed:
            if not lower <= fixed[event] <= upper:
                return result("no_feasible_candidate", "INFEASIBLE", True)
            lower = upper = fixed[event]
        if lower > upper:
            return result("no_feasible_candidate", "INFEASIBLE", True)
        bounds[event] = lower, upper
        variables[event] = model.new_int_var(lower, upper, event)
    gaps, deviations = [], []
    for events in groups.values():
        events.sort(key=lambda event: (rows[event]["scheduled_seconds"], event))
        for before, after in zip(events, events[1:]):
            gap = variables[after] - variables[before]
            minimum_row = rows[after]["min_headway_seconds"]
            model.add(gap >= minimum_row)
            model.add(gap <= maximum)
            if windows:
                selectors = []
                for index, (start, finish, window_minimum) in enumerate(windows):
                    if finish <= bounds[after][0] or start > bounds[after][1]:
                        continue
                    selected = model.new_bool_var(f"{after}_window_{index}")
                    model.add(variables[after] >= start).only_enforce_if(selected)
                    model.add(variables[after] < finish).only_enforce_if(selected)
                    model.add(gap >= window_minimum).only_enforce_if(selected)
                    selectors.append(selected)
                model.add_bool_or(selectors)
            gaps.append((before, after))
            if target is not None:
                deviation = model.new_int_var(0, max(target, maximum), f"{after}_gap_deviation")
                model.add_abs_equality(deviation, gap-target)
                deviations.append(deviation)
    for before, after, elapsed in links:
        model.add(variables[after] >= variables[before]+elapsed)
    holdings = sum(variables[event]-row["scheduled_seconds"] for event, row in rows.items())
    model.minimize(sum(deviations)*(len(rows)*hold+1)+holdings)
    solver = _solver(end)
    if solver is None:
        return result("evaluation_timeout", "NOT_STARTED")
    raw_status = solver.solve(model)
    status, complete = _status(raw_status)
    if raw_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(status, solver.status_name(raw_status), complete)
    times = {event: solver.value(variable) for event, variable in variables.items()}
    # Independently check the complete extracted assignment before returning it.
    valid = all(lower <= times[event] <= upper for event, (lower, upper) in bounds.items())
    valid &= all(minimum <= times[after]-times[before] <= maximum for before, after in gaps)
    valid &= all(times[after]-times[before] >= rows[after]["min_headway_seconds"] for before, after in gaps)
    valid &= all(times[after] >= times[before]+elapsed for before, after, elapsed in links)
    valid &= all(times[event] == value for event, value in fixed.items())
    if windows:
        valid &= all(any(start <= times[after] < finish and times[after]-times[before] >= value
                         for start, finish, value in windows) for before, after in gaps)
    if not valid:
        raise RuntimeError("CP-SAT assignment failed independent aggregate validation")
    if monotonic() >= end:
        return result("evaluation_timeout", solver.status_name(raw_status))
    deviation = sum(abs(times[after]-times[before]-target) for before, after in gaps) if target is not None else 0
    total_hold = sum(times[event]-row["scheduled_seconds"] for event, row in rows.items())
    return result(status, solver.status_name(raw_status), complete, times, (deviation, total_hold))


def solve_withdrawals(candidates, *, initial_count, minimum_remaining, max_withdrawals=None,
                      line_capacity=53, time_limit_seconds=10.0, deadline=None, excluded_subsets=()):
    """Select eligible IDs using caller-supplied integer savings and an explicit fleet floor.

    Candidates are already possible withdrawal opportunities, not trains teleported to a depot.
    Passenger admissibility, actual exit times and saved kilometres require detailed simulation.
    """
    started = monotonic()
    end = _budget(started, time_limit_seconds, deadline)
    capacity = _integer(line_capacity, "line_capacity", 1)
    initial = _integer(initial_count, "initial_count", 0)
    floor = _integer(minimum_remaining, "minimum_remaining", 0)
    if initial > capacity:
        raise ValueError(f"initial_count exceeds line capacity {capacity}")
    if floor > initial:
        raise ValueError("minimum_remaining exceeds initial_count")
    rows = {}
    for raw in candidates:
        train = _identifier(raw["train_id"], "train_id")
        if train in rows:
            raise ValueError(f"duplicate train_id {train}")
        if not isinstance(raw.get("eligible", True), bool):
            raise ValueError("eligible must be boolean")
        rows[train] = (_integer(raw["saving_units"], "saving_units"), raw.get("eligible", True))
    if len(rows) > initial:
        raise ValueError("candidate count exceeds initial_count")
    cuts = len(rows) if max_withdrawals is None else _integer(max_withdrawals, "max_withdrawals", 0)

    def result(status, solver_status, complete=False, selected=(), remaining=None, saving=None):
        return WithdrawalSolution(selected, remaining, saving, status, solver_status, monotonic()-started, complete)

    if monotonic() >= end:
        return result("evaluation_timeout", "NOT_STARTED")
    model = cp_model.CpModel()
    variables = {train: model.new_bool_var(train) for train in sorted(rows)}
    for train, (saving, eligible) in rows.items():
        if not eligible or saving <= 0:
            model.add(variables[train] == 0)
    model.add(sum(variables.values()) <= min(cuts, initial-floor))
    for subset in excluded_subsets:
        subset = frozenset(subset)
        if subset - variables.keys():
            raise ValueError("unknown train in excluded_subsets")
        model.add(sum(1-variable if train in subset else variable for train, variable in variables.items()) >= 1)
    # Lexicographic preference: saved units, then fewer withdrawals; one worker is repeatable.
    model.maximize(sum(variable*(rows[train][0]*(len(rows)+1)-1) for train, variable in variables.items()))
    solver = _solver(end)
    if solver is None:
        return result("evaluation_timeout", "NOT_STARTED")
    raw_status = solver.solve(model)
    status, complete = _status(raw_status)
    if raw_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(status, solver.status_name(raw_status), complete)
    selected = tuple(train for train, variable in variables.items() if solver.value(variable))
    if len(selected) > min(cuts, initial-floor) or any(not rows[train][1] or rows[train][0] <= 0 for train in selected):
        raise RuntimeError("CP-SAT assignment failed independent aggregate fleet validation")
    if monotonic() >= end:
        return result("evaluation_timeout", solver.status_name(raw_status))
    return result(status, solver.status_name(raw_status), complete, selected, initial-len(selected),
                  sum(rows[train][0] for train in selected))
