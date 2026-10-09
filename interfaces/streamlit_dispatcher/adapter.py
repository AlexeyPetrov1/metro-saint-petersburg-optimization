"""Read-only bridge from saved metro51 logs to the dispatcher UI."""

from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

NAMES = (
    "Проспект Ветеранов", "Ленинский проспект", "Автово", "Кировский завод",
    "Нарвская", "Балтийская", "Технологический институт", "Пушкинская",
    "Владимирская", "Площадь Восстания", "Чернышевская", "Площадь Ленина",
    "Выборгская", "Лесная", "Площадь Мужества", "Политехническая",
    "Академическая", "Гражданский проспект", "Девяткино",
)


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class Series:
    def __init__(self, rows=(), *, cumulative=False):
        merged = defaultdict(int)
        if cumulative:
            for t, value in rows:
                merged[t] += value
            total = 0
            values = []
            for t, value in sorted(merged.items()):
                total += value
                values.append((t, total))
        else:
            values = sorted(dict(rows).items())
        self.times = [t for t, _ in values]
        self.values = [v for _, v in values]

    def at(self, second):
        i = bisect_right(self.times, second) - 1
        return self.values[i] if i >= 0 else 0

    def pairs(self):
        return list(zip(self.times, self.values))


def planned_motion(data):
    """Geometry for an unfinished run; passenger state still comes from its checkpoint."""
    tracks = {r["track_id"]: r for r in data["tracks"]}
    trips = {t["trip_id"]: t for p in data.get("plans", []) for t in p["trips"]}
    events = []
    for train in data["trains"]:
        track = tracks.get(train["location"])
        initial = train.get("initial_movement")
        if track and initial:
            events.append(dict(second=initial["arrival_second"], train_id=train["train_id"],
                               event_type="arrival", location=track["to_station"]))
    for row in data.get("leg_timing", []):
        if row["trip_id"] not in trips:
            continue
        trip = trips[row["trip_id"]]
        track = tracks[trip["route"][row["leg_index"]]]
        departure = row["departure_second"]
        events += [
            dict(second=departure, train_id=row["train_id"], event_type="departure",
                 location=track["track_id"]),
            dict(second=departure + track["travel_seconds"] + row.get("approach_holding_seconds", 0),
                 train_id=row["train_id"], event_type="arrival", location=track["to_station"]),
        ]
    return {"events": sorted(events, key=lambda r: r["second"])}


class Replay:
    def __init__(self, data, groups, state, motion, forecasts=(), decisions=(), explanations=None):
        self.data, self.groups, self.state = data, groups, state
        self.start = datetime.fromisoformat(data["period_start"])
        self.end = int(state["second"])
        self.partial = False
        self.path = None
        self.station_ids = sorted(r["station_id"] for r in data["stations"] if r.get("kind", "line") == "line")
        self.names = {s: NAMES[int(s.split("_")[-1]) - 1] for s in self.station_ids}
        self.tracks = {r["track_id"]: r for r in data["tracks"]}
        self.explanations = explanations or {}
        self.forecasts = sorted(forecasts, key=lambda f: self.forecast_second(f))
        self.decisions = sorted(decisions, key=lambda d: d["decision_second"])
        self.forecast_times = [self.forecast_second(f) for f in self.forecasts]
        self.decision_times = [d["decision_second"] for d in self.decisions]
        self.entered = Series(((g["turnstile_second"], g["count"]) for g in groups), cumulative=True)
        self.platform = Series(((g["platform_second"], g["count"]) for g in groups), cumulative=True)
        queue_rows = defaultdict(list)
        by_group = {g["group_id"]: g for g in groups}
        for g in groups:
            for dest in g["destinations"]:
                queue_rows[g["origin"], dest["direction"]].append((g["platform_second"], dest["count"]))
        boarded = []
        for row in state.get("board_events", []):
            g = by_group[row["group_id"]]
            dest = next(d for d in g["destinations"] if d["station_id"] == row["destination"])
            queue_rows[g["origin"], dest["direction"]].append((row["second"], -row["count"]))
            boarded.append((row["second"], row["count"]))
        self.queues = {key: Series(rows, cumulative=True) for key, rows in queue_rows.items()}
        self.boarded = Series(boarded, cumulative=True)
        car_rows = defaultdict(list)
        for row in state.get("load_events", []):
            car_rows[row["train_id"], row["car_id"]].append((row["second"], row["load"]))
        self.cars = {key: Series(rows) for key, rows in car_rows.items()}
        events = defaultdict(list)
        for train in data["trains"]:
            events[train["train_id"]].append(dict(second=-1, location=train["location"], event_type="initial"))
        for row in motion.get("events", []):
            if row["event_type"] in ("departure", "arrival"):
                events[row["train_id"]].append(row)
        self.motion = {key: sorted(rows, key=lambda r: r["second"]) for key, rows in events.items()}
        self.motion_times = {key: [r["second"] for r in rows] for key, rows in self.motion.items()}

    @classmethod
    def from_arm(cls, path):
        path = Path(path).resolve()
        common = read(path / "common_scenario.json.gz")
        result_path = path / "result.json"
        if result_path.exists():
            run = path / read(result_path)["run"]["folder"]
            data = read(run / "movement_input.json")
            state = read(run / "passenger_checkpoint.json.gz")
            motion = read(run / "motion.json")
            partial = False
        else:
            checkpoints = sorted((path / "controller").glob("resume_*.json.gz"))
            if not checkpoints:
                raise ValueError("No recorded passenger checkpoint in this arm")
            checkpoint = read(checkpoints[-1])
            data, state = checkpoint["data"], checkpoint["state"]
            motion, partial = planned_motion(data), True
        forecasts = [read(p) for p in sorted((path / "controller").glob("raw_forecast_*.json"))]
        decisions = [read(p) for p in sorted((path / "controller").glob("decision_*.json"))]
        explanations = {int(p.stem.split("_")[-1]): read(p)
                        for p in (path / "explanations").glob("explanation_*.json")}
        replay = cls(data, common["groups"], state, motion, forecasts, decisions, explanations)
        replay.path, replay.partial = path, partial
        return replay

    def seconds(self, iso):
        return (datetime.fromisoformat(iso.replace("Z", "+00:00")) - self.start).total_seconds()

    def forecast_second(self, f):
        return max(self.seconds(f[key]) for key in
                   ("issued_at", "available_at", "features_available_at") if f.get(key))

    def forecast_at(self, second):
        i = bisect_right(self.forecast_times, second) - 1
        if i < 0:
            return None
        f = self.forecasts[i]
        quantiles = {(r["station_id"], r["interval_start"]): r
                     for r in f.get("source_versions", {}).get("quantile_rows", [])}
        fallbacks = f.get("source_versions", {}).get("fallback_station_ids", [])
        rows = []
        for r in f["rows"]:
            q = quantiles.get((r["station_id"], r["interval_start"]), {})
            rows.append({**r, "q10": q.get("q10"), "q50": q.get("q50", r["predicted_count"]),
                         "q90": q.get("q90"), "fallback": r["station_id"] in fallbacks})
        return {"issued_at": f["issued_at"], "model_version": f["model_version"],
                "rows": rows, "stale": not any(self.seconds(r["interval_end"]) > second for r in rows)}

    def train_at(self, train, second):
        tid = train["train_id"]
        events, times = self.motion[tid], self.motion_times[tid]
        i = max(0, bisect_right(times, second) - 1)
        event = events[i]
        location = event["location"]
        track = self.tracks.get(location)
        fraction, origin, dest = 0., location, location
        if track:
            origin, dest = track["from_station"], track["to_station"]
            if event["event_type"] == "initial":
                arrival = train.get("initial_movement", {}).get("arrival_second", track["travel_seconds"])
                departure = arrival - track["travel_seconds"]
            else:
                departure = event["second"]
                arrival = next((r["second"] for r in events[i + 1:] if r["event_type"] == "arrival"),
                               departure + track["travel_seconds"])
            fraction = min(1., max(0., (second - departure) / max(1, arrival - departure)))
        cars = [self.cars.get((tid, f"C{j + 1}"), Series()).at(second) for j in range(train.get("cars", 8))]
        return {"train_id": tid, "train_type": train.get("train_type", "unknown"), "location": location,
                "from": origin, "to": dest, "fraction": fraction, "moving": bool(track),
                "on_line": origin in self.station_ids and dest in self.station_ids, "cars": cars}

    def frame(self, second):
        if not 0 <= second <= self.end:
            raise ValueError(f"Time outside recorded passenger interval [0, {self.end}]")
        cutoff = second - .000001  # saved snapshots contain events strictly before their timestamp
        stations = [{"station_id": s, "name": self.names[s],
                     "north": self.queues.get((s, 1), Series()).at(cutoff),
                     "south": self.queues.get((s, -1), Series()).at(cutoff)} for s in self.station_ids]
        trains = [self.train_at(t, cutoff) for t in self.data["trains"]]
        onboard = sum(sum(t["cars"]) for t in trains)
        entered, platform = self.entered.at(cutoff), self.platform.at(cutoff)
        j = bisect_right(self.decision_times, second) - 1
        decision = self.decisions[j] if j >= 0 else None
        return {"second": second, "time": (self.start + timedelta(seconds=second)).isoformat(),
                "stations": stations, "trains": trains, "entered": entered,
                "approaching": entered - platform, "waiting": sum(s["north"] + s["south"] for s in stations),
                "onboard": onboard, "completed": self.boarded.at(cutoff) - onboard,
                "forecast": self.forecast_at(second), "decision": decision,
                "explanation": self.explanations.get(decision["decision_second"]) if decision else None}

    def station_history(self, station_id, second):
        slots = defaultdict(int)
        for g in self.groups:
            if g["origin"] == station_id and g["turnstile_second"] < min(second, self.end):
                slots[g["turnstile_second"] // 900 * 900] += g["count"]
        return [{"interval_start": (self.start + timedelta(seconds=t)).isoformat(), "count": value,
                 "complete": t + 900 <= second} for t, value in sorted(slots.items())]

    def player_payload(self, baseline=None):
        baseline = baseline or self
        end = min(self.end, baseline.end)
        seconds = list(range(0, end + 1, 900))
        order = list(reversed(self.station_ids))
        variants = {}
        for key, replay in [("baseline", baseline), ("policy", self)]:
            frames = [replay.frame(t) for t in seconds]
            ratios = {d: [[] for _ in range(len(order) - 1)] for d in ("north", "south")}
            queues = {d: [[next(s[d] for s in f["stations"] if s["station_id"] == sid)
                           for f in frames] for sid in order] for d in ("north", "south")}
            for f in frames:
                for i, (up, down) in enumerate(zip(order, order[1:])):
                    for d, a, b in [("north", down, up), ("south", up, down)]:
                        loads = [max(t["cars"], default=0) / 150 for t in f["trains"]
                                 if t["moving"] and t["from"] == a and t["to"] == b]
                        ratios[d][i].append(max(loads) if loads else None)
            zeros = {d: [[None] * len(seconds) for _ in range(len(order) - 1)] for d in ("north", "south")}
            variants[key] = {"fill": ratios, "ratio": ratios, "left": zeros,
                             "queues": queues,
                             "kpi": {"waiting": [f["waiting"] for f in frames],
                                     "denied": [0] * len(seconds), "wait_pax_min": [0] * len(seconds),
                                     "trains_in_service": [sum(t["on_line"] for t in f["trains"]) for f in frames]}}
        actions = []
        for d in self.decisions:
            if d["decision_second"] > end:
                continue
            selection = d.get("selection", {})
            actions.append({"k": d["decision_second"] / 900,
                            "time": (self.start + timedelta(seconds=d["decision_second"])).strftime("%H:%M"),
                            "what": d.get("applied_candidate_id", "KEEP"), "target": "Красная линия",
                            "status": selection.get("status", "unknown"), "reason": "",
                            "seg": None, "dir": None, "applied": selection.get("release_authorized", False),
                            "error": False, "critical": False, "ba": None, "win": ""})
        return {"banner": None, "anomaly": [[False] * len(seconds) for _ in order],
                "reasons": [[[] for _ in seconds] for _ in order],
                "scenario": "сохранённый прогон", "date": self.start.strftime("%d.%m.%Y"),
                "policy": self.path.name if self.path else "Replay",
                "times": [(self.start + timedelta(seconds=t)).strftime("%H:%M") for t in seconds],
                "first": 0, "last": len(seconds) - 1,
                "stations": [self.names[s] for s in order], "station_ids": order,
                "segs": [{"label": self.names[a], "tip": f"{self.names[a]} – {self.names[b]}"}
                         for a, b in zip(order, order[1:])], "data": variants, "actions": actions,
                "adapter": self.browser_data(), "baseline_adapter": baseline.browser_data()}

    def browser_data(self):
        """Compact event histories; the browser moves the clock without rerunning Python."""
        return {"start": self.start.isoformat(), "end": self.end,
                "entered": self.entered.pairs(), "boarded": self.boarded.pairs(),
                "platform": self.platform.pairs(),
                "trains": self.data["trains"], "tracks": self.data["tracks"],
                "motion": {t: [[e["second"], e["location"]] for e in events] for t, events in self.motion.items()},
                "cars": {f"{t}:{c}": s.pairs() for (t, c), s in self.cars.items()},
                "queues": {f"{s}:{d}": series.pairs() for (s, d), series in self.queues.items()},
                "forecasts": [{"second": t, **self.forecast_at(t)} for t in self.forecast_times if t <= self.end],
                "decisions": [{"second": d["decision_second"], "candidate": d.get("applied_candidate_id", "KEEP"),
                               "status": d.get("selection", {}).get("status", "unknown"),
                               "release_authorized": d.get("selection", {}).get("release_authorized", False),
                               "explanation": self.explanations.get(d["decision_second"])}
                              for d in self.decisions if d["decision_second"] <= self.end]}
