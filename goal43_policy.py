"""Versioned dispatch policy, explicit demo fleet and exact car-time exposure."""

from collections import Counter, defaultdict
from datetime import datetime, time
from fractions import Fraction
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_POLICY = Path(__file__).resolve().parent / "config/decision_engine/goal4_3.json"


def parameter(policy, name):
    return policy["parameters"][name]["value"]


def load_policy(path=None):
    policy = json.loads(Path(path or DEFAULT_POLICY).read_text(encoding="utf-8"))
    validate_policy(policy)
    return policy


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
        raise ValueError(f"{name} must be an integer >= {int(not allow_zero)}")
    return value


def _aware_datetime(at):
    parsed = datetime.fromisoformat(at) if isinstance(at, str) else at
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return parsed


def validate_policy(policy):
    if policy.get("profile_id") != "goal4_3" or policy.get("schema_version") != 1:
        raise ValueError("Unsupported policy profile or schema")
    for name, entry in policy["parameters"].items():
        if not {"value", "unit", "source", "status"} <= entry.keys() or not entry["unit"] or not entry["source"]:
            raise ValueError(f"Missing provenance for {name}")
    p = lambda name: parameter(policy, name)
    if p("timezone") != "Europe/Moscow":
        raise ValueError("This profile uses Europe/Moscow")
    for name in ("physical_headway_min_s", "peak_headway_min_s", "offpeak_headway_min_s",
                 "headway_max_s", "turnaround_s", "line_train_limit", "cars_per_train",
                 "comfort_limit", "normal_limit", "hard_limit", "decision_step_s",
                 "notification_budget_s", "to1_interval_km"):
        _positive_integer(p(name), name)
    if not p("physical_headway_min_s") <= p("peak_headway_min_s") <= p("offpeak_headway_min_s") <= p("headway_max_s"):
        raise ValueError("Invalid headway limits")
    if not p("comfort_limit") < p("normal_limit") < p("hard_limit"):
        raise ValueError("Invalid per-car occupancy limits")
    share = p("exceptional_time_share_max")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("Invalid exceptional-time share")
    for window in p("peak_windows"):
        if not set(window["weekdays"]) <= set(range(7)) or time.fromisoformat(window["start"]) >= time.fromisoformat(window["end"]):
            raise ValueError("Invalid peak window")
    for depot, seconds in p("depot_release_s").items():
        _positive_integer(seconds, f"depot release {depot}")
    for train_type, count in p("fleet_by_type").items():
        _positive_integer(count, f"fleet {train_type}")
    if p("line_train_limit") > sum(p("fleet_by_type").values()):
        raise ValueError("Line limit exceeds assigned fleet")
    if p("reference_roundtrip_s") is not None:
        raise ValueError("Artificial roundtrip requirement was withdrawn")
    return policy


def minimum_headway(at, policy=None):
    policy = policy or load_policy()
    local = _aware_datetime(at).astimezone(ZoneInfo(parameter(policy, "timezone")))
    for window in parameter(policy, "peak_windows"):
        if local.weekday() in window["weekdays"] and time.fromisoformat(window["start"]) <= local.time() < time.fromisoformat(window["end"]):
            return parameter(policy, "peak_headway_min_s")
    return parameter(policy, "offpeak_headway_min_s")


def depot_release_seconds(depot, policy=None):
    return parameter(policy or load_policy(), "depot_release_s")[depot]


def build_train_registry(policy=None, initial_on_line=None, reserves=None):
    policy = policy or load_policy()
    initial = policy["demo"]["initial_on_line"] if initial_on_line is None else initial_on_line
    reserves = policy["demo"]["reserves"] if reserves is None else reserves
    fleet = parameter(policy, "fleet_by_type")
    if set(initial) != set(fleet):
        raise ValueError("Initial allocation must explicitly specify each train type")
    rows = []
    for train_type, prefix in (("Baltiets", "B"), ("Yubileyny", "Y")):
        count = _positive_integer(initial[train_type], f"initial {train_type}", allow_zero=True)
        if count > fleet[train_type]:
            raise ValueError("Initial allocation exceeds assigned fleet")
        for number in range(1, fleet[train_type] + 1):
            on_line = number <= count
            rows.append({"train_id": f"{prefix}{number:03}", "type": train_type,
                         "state": "on_line" if on_line else "unavailable",
                         "location": {"kind": "line", "line_id": "L1"} if on_line else {"kind": "unknown"},
                         "ready_at": None, "readiness_status": "not_applicable" if on_line else "unknown",
                         "odometer_km": None, "km_since_to1": None, "status": "assumed"})
    by_id = {row["train_id"]: row for row in rows}
    for reserve in reserves:
        if reserve["train_id"] not in by_id:
            raise ValueError("Reserve train is not in assigned fleet")
        row = by_id[reserve["train_id"]]
        if row["state"] != "unavailable":
            raise ValueError("Reserve train is already assigned")
        if reserve["depot"] not in parameter(policy, "depot_release_s"):
            raise ValueError("Unknown depot")
        ready_at = reserve.get("ready_at")
        if ready_at is not None:
            _aware_datetime(ready_at)
        row.update(state="reserve", location={"kind": "depot", "depot": reserve["depot"]},
                   ready_at=ready_at, readiness_status="assumed" if ready_at is not None else "unknown")
    validate_train_registry(rows, policy)
    return rows


def validate_train_registry(registry, policy=None):
    policy = policy or load_policy()
    fleet = parameter(policy, "fleet_by_type")
    if len({row["train_id"] for row in registry}) != len(registry):
        raise ValueError("duplicate train_id")
    if Counter(row["type"] for row in registry) != Counter(fleet):
        raise ValueError("Registry must contain the entire assigned fleet by type")
    on_line_states = {"on_line", "stopped", "approach_hold", "turnaround", "entering", "withdrawing"}
    states = on_line_states | {"reserve", "unavailable", "maintenance"}
    count = 0
    for row in registry:
        if row["state"] not in states:
            raise ValueError("Unknown train state")
        location = row["location"]
        on_line = location["kind"] == "line"
        if (row["state"] in on_line_states) != on_line:
            raise ValueError("Train state disagrees with line occupancy")
        count += on_line
        if row["state"] == "reserve" and (location["kind"] != "depot" or location.get("depot") not in parameter(policy, "depot_release_s")):
            raise ValueError("Reserve requires an explicit known depot")
        if row.get("ready_at") is not None:
            _aware_datetime(row["ready_at"])
        for name in ("km_since_to1", "odometer_km"):
            value = row.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {name}")
    if count > parameter(policy, "line_train_limit"):
        raise ValueError(f"Line occupancy exceeds {parameter(policy, 'line_train_limit')}")
    return {"train_count": len(registry), "on_line_count": count, "fleet_by_type": dict(fleet)}


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Interval times must be finite numeric seconds")
    if isinstance(value, int) or value.is_integer():
        return int(value)
    return Fraction(str(value))


def _accumulated_summary(counts, limit):
    occupied, exceptional, comfort, hard, maximum = counts
    share = Fraction(exceptional, occupied) if occupied else None
    status = "unknown" if share is None else "fail" if hard or share > limit else "pass"
    return {"status": status, "passenger_car_seconds": float(occupied),
            "exceptional_car_seconds": float(exceptional), "comfort_car_seconds": float(comfort),
            "hard_limit_car_seconds": float(hard), "exceptional_time_share": float(share) if share is not None else None,
            "hard_limit_ok": not bool(hard) if share is not None else None, "max_load": maximum}


def occupancy_exposure(intervals, policy=None, window_start_s=None, window_end_s=None):
    """Integrate half-open load intervals in seconds; car_index is zero-based.

    Each occupied interval represents passenger running, station service or holding.
    Step boundaries start at window_start_s (or floor(first time / decision_step_s)).
    Per-car and per-train shares are reported; admissibility applies fleet-wide
    over the whole supplied episode and every decision step.
    """
    policy = policy or load_policy()
    groups = defaultdict(list)
    rows = []
    for value in intervals:
        row = dict(value)
        for name in ("start", "end"):
            alias = f"{name}_s"
            if alias in row:
                if name in row and row[name] != row[alias]:
                    raise ValueError(f"Conflicting {name} aliases")
                row[name] = row[alias]
        row["start"], row["end"] = _seconds(row["start"]), _seconds(row["end"])
        if row["end"] <= row["start"]:
            raise ValueError("Interval end must follow start")
        _positive_integer(row["load"], "load", allow_zero=True)
        _positive_integer(row["car_index"], "car_index", allow_zero=True)
        if row["car_index"] >= parameter(policy, "cars_per_train") or not isinstance(row["train_id"], str) or not row["train_id"]:
            raise ValueError("Invalid car or train identity")
        groups[(row["train_id"], row["car_index"])].append(row)
        rows.append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["start"])
        if any(previous["end"] > current["start"] for previous, current in zip(group, group[1:])):
            raise ValueError("overlapping intervals for the same car")
    step = parameter(policy, "decision_step_s")
    start = _seconds(window_start_s) if window_start_s is not None else (min(row["start"] for row in rows) // step) * step if rows else 0
    end = _seconds(window_end_s) if window_end_s is not None else max((row["end"] for row in rows), default=start)
    if end < start:
        raise ValueError("Coverage end precedes start")

    total = [0, 0, 0, 0, 0]
    window_count = -((start - end) // step)
    bins = [[0, 0, 0, 0, 0] for _ in range(window_count)]
    car_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    train_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    comfort = parameter(policy, "comfort_limit")
    normal = parameter(policy, "normal_limit")
    hard = parameter(policy, "hard_limit")
    limit = Fraction(str(parameter(policy, "exceptional_time_share_max")))

    def accumulate(counts, duration, load):
        if load > 0:
            counts[0] += duration
        if load > normal:
            counts[1] += duration
        if load > comfort:
            counts[2] += duration
        if load > hard:
            counts[3] += duration
        counts[4] = max(counts[4], load)

    for row in rows:
        left, right = max(start, row["start"]), min(end, row["end"])
        if right <= left:
            continue
        duration, load = right - left, row["load"]
        accumulate(total, duration, load)
        accumulate(car_counts[row["train_id"], row["car_index"]], duration, load)
        accumulate(train_counts[row["train_id"]], duration, load)
        slot = int((left - start) // step)
        while left < right:
            stop = min(right, start + (slot + 1) * step)
            accumulate(bins[slot], stop - left, load)
            left = stop
            slot += 1
    global_summary = _accumulated_summary(total, limit)
    windows = [dict(_accumulated_summary(counts, limit), start=float(start + i * step), end=float(min(end, start + (i + 1) * step)))
               for i, counts in enumerate(bins)]
    per_car = [dict(_accumulated_summary(counts, limit), train_id=train_id, car_index=car_index)
               for (train_id, car_index), counts in sorted(car_counts.items())]
    per_train = [dict(_accumulated_summary(counts, limit), train_id=train_id) for train_id, counts in sorted(train_counts.items())]
    status = "fail" if global_summary["status"] == "fail" or any(window["status"] == "fail" for window in windows) else global_summary["status"]
    worst = max(per_car, key=lambda row: (row["max_load"], row["exceptional_time_share"] or 0,
                                         row["exceptional_car_seconds"], row["train_id"], -row["car_index"]), default=None)
    return {"status": status, "admissible": None if status == "unknown" else status == "pass",
            "global": global_summary, "windows": windows, "per_train": per_train,
            "per_car": per_car, "worst_car": worst,
            "unit": "car-seconds", "policy_scope": parameter(policy, "exceptional_time_scope")}
