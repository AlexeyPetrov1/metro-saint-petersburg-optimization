"""FIFO passenger accounting driven exclusively by audited movement events."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from economics import number, ratio, write_json, write_text
from simulation import digest, fields_only, identifier, indexed, integer, simulate


MODEL_VERSION = "passengers-v1"
ROOT = Path(__file__).resolve().parent
OUTPUT_NAMES = {"service.passengers.json", "no_trains.passengers.json", "split_run.json",
                "empty_demand.json", "movement.simulation.json", "control_check.json",
                "assumptions.json", "summary.md"}
ASSUMPTIONS = {
    "mode": "educational_fixture", "initial_passenger_state": "empty queues and empty trains",
    "arrival_source": "explicit individual arrivals and directed routes; no inferred turnstile OD",
    "routing": "direct ride on a prefix of the remaining current trip; no transfers",
    "boarding": "at movement departure; alighting at movement arrival; fixed dwell times",
    "fifo": "arrival_second then passenger_id among eligible passengers",
    "same_instant": "demand arrivals, train arrivals/alighting, wait completions, departures/boarding",
    "windows": "[start,end); final movement arrivals/waits included at horizon end; demand at end excluded",
    "p95": "nearest rank ceil(0.95*N), boarded passengers only",
    "waiting_area": "integral of queue size, including unserved residual waiting",
    "control": "5 arrivals at 0; departures 60 and 180; max_capacity=3; educational normal_threshold=2",
    "capacity_scope": "whole train; no distribution by car and no real regulatory admission check",
    "feedback_to_motion": False, "real_OD_verified": False,
    "operational_applicability": "unknown", "release_authorized": False,
}


class PassengerSimulation:
    def __init__(self, movement, plan_id, demand):
        self.motion = simulate(movement, plan_id)
        fields_only(demand, ("schema_version", "source", "arrivals", "capacities"))
        if demand["schema_version"] != "1.0":
            raise ValueError("unsupported passenger schema")
        identifier(demand["source"], "passenger source")
        self.source = demand["source"]
        self.arrivals = copy.deepcopy(indexed(demand["arrivals"], "passenger_id"))
        self.capacities = copy.deepcopy(indexed(demand["capacities"], "train_id"))
        self.tracks = copy.deepcopy(indexed(movement["tracks"], "track_id"))
        stations = indexed(movement["stations"], "station_id")
        trains = indexed(movement["trains"], "train_id")
        if set(self.capacities) != set(trains):
            raise ValueError("exactly one capacity record required for each train")
        for cap in self.capacities.values():
            fields_only(cap, ("train_id", "max_capacity", "normal_threshold", "source"))
            integer(cap["max_capacity"], "max_capacity", 1)
            integer(cap["normal_threshold"], "normal_threshold", 1)
            identifier(cap["source"], "capacity source")
            if cap["normal_threshold"] > cap["max_capacity"]:
                raise ValueError("normal_threshold exceeds max_capacity")
        for person in self.arrivals.values():
            fields_only(person, ("passenger_id", "arrival_second", "route"))
            integer(person["arrival_second"], "arrival_second")
            route = person["route"]
            if not isinstance(route, list) or not route:
                raise ValueError("passenger route must be a nonempty list")
            if any(not isinstance(t, str) or t not in self.tracks for t in route):
                raise ValueError("unknown passenger route track")
            legs = [self.tracks[t] for t in route]
            if any(a["to_station"] != b["from_station"] for a, b in zip(legs, legs[1:])):
                raise ValueError("discontinuous passenger route")
            if any(stations[leg[k]]["kind"] != "line" for leg in legs for k in ("from_station", "to_station")):
                raise ValueError("passenger routes must stay on line stations")
        self.duration = self.motion["duration_seconds"]
        self.context_id = digest({"model_version": MODEL_VERSION, "motion": self.motion,
                                  "arrivals": list(self.arrivals.values()),
                                  "capacities": list(self.capacities.values()), "source": self.source})
        trips = indexed(indexed(movement["plans"], "plan_id")[plan_id]["trips"], "trip_id")
        next_leg = {trip: 0 for trip in trips}
        stream = []
        for person in self.arrivals.values():
            stream.append((person["arrival_second"], 0, person["passenger_id"],
                           {"kind": "demand", "second": person["arrival_second"],
                            "passenger_id": person["passenger_id"]}))
        for order, event in enumerate(self.motion["events"]):
            row = {"kind": "motion", **copy.deepcopy(event)}
            if event["event_type"] == "departure":
                trip = trips[event["trip_id"]]
                i = next_leg[event["trip_id"]]
                if trip["route"][i] != event["location"]:
                    raise ValueError("movement departure route mismatch")
                next_leg[event["trip_id"]] += 1
                row.update(passenger_service=trip["passenger"], remaining_route=trip["route"][i:])
            priority = {"arrival": 1, "departure": 3}.get(event["event_type"], 2)
            stream.append((event["second"], priority, f"{order:012d}", row))
        self.stream = [row for _, _, _, row in sorted(stream)]

    def _initial_state(self):
        return {"model_version": MODEL_VERSION, "context_id": self.context_id, "second": 0,
                "next_event_index": 0, "waiting_passenger_seconds": 0,
                "max_queue": 0, "max_queue_by_station": {}, "denied_boarding_attempts": 0,
                "denied_passenger_ids": [], "load_events": [], "passenger_events": [],
                "passengers": [{**copy.deepcopy(p), "state": "not_entered", "train_id": None,
                                "route_position": 0, "boarded_second": None, "completed_second": None,
                                "waiting_seconds": None} for p in self.arrivals.values()]}

    @staticmethod
    def _seal(state):
        return digest({k: v for k, v in state.items() if k != "sha256"})

    def _restore(self, checkpoint, start):
        if checkpoint is None:
            if start != 0:
                raise ValueError("nonzero start requires the previous checkpoint")
            return self._initial_state()
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("checkpoint must be a nonempty object")
        fields_only(checkpoint, (*self._initial_state(), "sha256"))
        if checkpoint.get("sha256") != self._seal(checkpoint):
            raise ValueError("checkpoint checksum mismatch")
        if checkpoint.get("context_id") != self.context_id or checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError("checkpoint belongs to another movement, demand or capacity context")
        if integer(checkpoint["second"], "checkpoint second") != start or start == 0:
            raise ValueError("windows must be consecutive without gaps, overlap or replay")
        cursor = integer(checkpoint["next_event_index"], "checkpoint cursor")
        if cursor != sum(e["second"] < start for e in self.stream):
            raise ValueError("checkpoint cursor does not match the window boundary")
        people = indexed(checkpoint["passengers"], "passenger_id")
        if set(people) != set(self.arrivals):
            raise ValueError("checkpoint passenger set differs from demand")
        for p in people.values():
            fields_only(p, ("passenger_id", "arrival_second", "route", "state", "train_id",
                            "route_position", "boarded_second", "completed_second", "waiting_seconds"))
            original = self.arrivals[p["passenger_id"]]
            if any(p[k] != original[k] for k in original):
                raise ValueError("checkpoint changed passenger identity, arrival or route")
            if p["state"] not in {"not_entered", "queued", "onboard", "completed"}:
                raise ValueError("unknown checkpoint passenger state")
            position = integer(p["route_position"], "route_position")
            boarded = p["state"] in {"onboard", "completed"}
            if (p["state"] == "not_entered") != (p["arrival_second"] >= start):
                raise ValueError("checkpoint entry time is inconsistent")
            if boarded:
                board = integer(p["boarded_second"], "boarded_second")
                wait = integer(p["waiting_seconds"], "waiting_seconds")
                if not p["arrival_second"] <= board < start or wait != board - p["arrival_second"]:
                    raise ValueError("checkpoint boarding time or waiting age is inconsistent")
            elif p["boarded_second"] is not None or p["waiting_seconds"] is not None or position:
                raise ValueError("unboarded passenger has boarding history")
            if p["state"] == "completed":
                done = integer(p["completed_second"], "completed_second")
                if position != len(p["route"]) or not p["boarded_second"] <= done < start:
                    raise ValueError("checkpoint completion is inconsistent")
            elif p["completed_second"] is not None or position >= len(p["route"]):
                raise ValueError("checkpoint route position is inconsistent")
            if (p["state"] == "onboard" and p["train_id"] not in self.capacities
                    or p["state"] != "onboard" and p["train_id"] is not None):
                raise ValueError("checkpoint train reference is inconsistent")
        for key in ("load_events", "passenger_events", "denied_passenger_ids"):
            if not isinstance(checkpoint[key], list):
                raise ValueError(f"checkpoint {key} must be a list")
        if any(not isinstance(row, dict) for key in ("load_events", "passenger_events") for row in checkpoint[key]):
            raise ValueError("checkpoint event records must be objects")
        for key in ("waiting_passenger_seconds", "max_queue", "denied_boarding_attempts"):
            integer(checkpoint[key], key)
        wait = sum(p["waiting_seconds"] or 0 for p in people.values())
        residual = sum(start - p["arrival_second"] for p in people.values() if p["state"] == "queued")
        if checkpoint["waiting_passenger_seconds"] != wait + residual:
            raise ValueError("checkpoint waiting-time conservation failed")
        state = copy.deepcopy(checkpoint)
        del state["sha256"]
        return state

    def _advance(self, state, second):
        queued = sum(p["state"] == "queued" for p in state["passengers"])
        state["waiting_passenger_seconds"] += queued * (second - state["second"])
        state["second"] = second

    def _queue_peak(self, state):
        counts = {}
        for p in state["passengers"]:
            if p["state"] == "queued":
                station = self.tracks[p["route"][0]]["from_station"]
                counts[station] = counts.get(station, 0) + 1
        state["max_queue"] = max(state["max_queue"], sum(counts.values()))
        for station, count in counts.items():
            state["max_queue_by_station"][station] = max(state["max_queue_by_station"].get(station, 0), count)

    @staticmethod
    def _log(state, kind, person, event):
        state["passenger_events"].append({"event_type": kind, "second": state["second"],
            "passenger_id": person["passenger_id"], "train_id": event.get("train_id"),
            "motion_event_id": event.get("event_id")})

    def _process(self, state, event):
        if event["kind"] == "demand":
            person = next(p for p in state["passengers"] if p["passenger_id"] == event["passenger_id"])
            if person["state"] != "not_entered":
                raise ValueError("passenger arrival processed twice")
            person["state"] = "queued"
            self._log(state, "enter", person, event)
            self._queue_peak(state)
            return
        train = event["train_id"]
        onboard = [p for p in state["passengers"] if p["state"] == "onboard" and p["train_id"] == train]
        if event["event_type"] == "arrival":
            for person in onboard:
                track = self.tracks[person["route"][person["route_position"]]]
                if track["to_station"] != event["location"]:
                    raise ValueError("passenger arrival does not match directed route")
                person["route_position"] += 1
                if person["route_position"] == len(person["route"]):
                    person.update(state="completed", train_id=None, completed_second=state["second"])
                    self._log(state, "alight", person, event)
        elif event["event_type"] == "departure":
            cap = self.capacities[train]
            if any(p["route"][p["route_position"]] != event["location"] for p in onboard):
                raise ValueError("onboard passenger route differs from train departure")
            eligible = []
            if event["passenger_service"]:
                eligible = sorted((p for p in state["passengers"] if p["state"] == "queued"
                    and p["route"] == event["remaining_route"][:len(p["route"])]),
                    key=lambda p: (p["arrival_second"], p["passenger_id"]))
            places = cap["max_capacity"] - len(onboard)
            if places < 0:
                raise ValueError("passenger capacity exceeded")
            selected, denied = eligible[:places], eligible[places:]
            for person in selected:
                person.update(state="onboard", train_id=train, boarded_second=state["second"],
                              waiting_seconds=state["second"] - person["arrival_second"])
                self._log(state, "board", person, event)
            state["denied_boarding_attempts"] += len(denied)
            state["denied_passenger_ids"] = sorted(set(state["denied_passenger_ids"]) |
                                                   {p["passenger_id"] for p in denied})
            load = len(onboard) + len(selected)
            state["load_events"].append({"motion_event_id": event["event_id"], "second": state["second"],
                "train_id": train, "track_id": event["location"], "load": load,
                "boarded": len(selected), "denied_boarding": len(denied),
                "max_capacity": cap["max_capacity"], "normal_threshold": cap["normal_threshold"],
                "normal_threshold_exceeded": load > cap["normal_threshold"],
                "load_to_normal_ratio": number(ratio(Decimal(load), Decimal(cap["normal_threshold"]))),
                "source": cap["source"], "applicability": "educational_only"})

    def _result(self, state, start, previous_area):
        people, second = state["passengers"], state["second"]
        counts = {s: sum(p["state"] == s for p in people)
                  for s in ("not_entered", "queued", "onboard", "completed")}
        entered = len(people) - counts["not_entered"]
        waits = sorted(p["waiting_seconds"] for p in people if p["waiting_seconds"] is not None)
        queues = [{"passenger_id": p["passenger_id"], "arrival_second": p["arrival_second"],
                   "station_id": self.tracks[p["route"][0]]["from_station"],
                   "route": copy.deepcopy(p["route"]), "waiting_age_seconds": second - p["arrival_second"]}
                  for p in sorted(people, key=lambda p: (p["arrival_second"], p["passenger_id"])) if p["state"] == "queued"]
        residual = sum(p["waiting_age_seconds"] for p in queues)
        balance = entered == counts["queued"] + counts["onboard"] + counts["completed"]
        identity = state["waiting_passenger_seconds"] == sum(waits) + residual
        if not balance or not identity:
            raise ValueError("passenger balance or waiting-time conservation failed")
        metrics = {"entered": entered, "boarded": len(waits), "completed": counts["completed"],
            "queued": counts["queued"], "onboard": counts["onboard"], "future_arrivals": counts["not_entered"],
            "waiting_passenger_seconds": state["waiting_passenger_seconds"],
            "boarded_wait_passenger_seconds": sum(waits), "residual_wait_passenger_seconds": residual,
            "boarded_waits_seconds": waits,
            "mean_wait_seconds": number(ratio(Decimal(sum(waits)), Decimal(len(waits)))) if waits else None,
            "p95_wait_seconds": waits[(95 * len(waits) + 99) // 100 - 1] if waits else None,
            "p95_method": "nearest_rank_boarded_only", "max_queue": state["max_queue"],
            "max_queue_by_station": copy.deepcopy(state["max_queue_by_station"]),
            "denied_boarding_attempts": state["denied_boarding_attempts"],
            "unique_denied_passengers": len(state["denied_passenger_ids"]),
            "normal_threshold_exceedance_events": sum(r["normal_threshold_exceeded"] for r in state["load_events"]),
            "balance": {"initial_passengers": 0, "entered": entered, "completed": counts["completed"],
                        "queued": counts["queued"], "onboard": counts["onboard"], "passed": balance},
            "waiting_identity_passed": identity}
        state["sha256"] = self._seal(state)
        return {"schema_version": "1.0", "model_version": MODEL_VERSION,
            "movement_model_version": self.motion["model_version"], "evaluation_mode": "fixture",
            "source": self.source, "context_id": self.context_id, "plan_id": self.motion["plan_id"],
            "window_start_second": start, "window_end_second": second,
            "window_waiting_passenger_seconds": state["waiting_passenger_seconds"] - previous_area,
            "metrics_scope": "cumulative from horizon start", "metrics": metrics, "queues": queues,
            "load_events": copy.deepcopy(state["load_events"]),
            "passenger_events": copy.deepcopy(state["passenger_events"]), "checkpoint": state,
            "passenger_obligations_closed": not (counts["queued"] or counts["onboard"]),
            "transport_work": copy.deepcopy(self.motion["transport_work"]),
            "transport_work_scope": "full movement horizon, unchanged by passenger windows",
            "release_authorized": False}

    def run(self, *, start_second=0, end_second=None, checkpoint=None):
        start = integer(start_second, "window start")
        end = self.duration if end_second is None else integer(end_second, "window end", 1)
        if not start < end <= self.duration:
            raise ValueError("window must have positive duration and stay within movement horizon")
        state = self._restore(checkpoint, start)
        previous_area = state["waiting_passenger_seconds"]
        while state["next_event_index"] < len(self.stream):
            event = self.stream[state["next_event_index"]]
            second = event["second"]
            closing_motion = end == self.duration and event["kind"] == "motion" and second == end
            if second > end or (second == end and not closing_motion):
                # Demand at horizon end sorts before closing movement events.
                if end == self.duration and second == end and event["kind"] == "demand":
                    state["next_event_index"] += 1
                    continue
                break
            if second < start:
                raise ValueError("checkpoint event cursor precedes window start")
            self._advance(state, second)
            self._process(state, event)
            state["next_event_index"] += 1
        self._advance(state, end)
        return self._result(state, start, previous_area)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_output(output, inputs):
    output = Path(output).resolve()
    protected = [ROOT / p for p in ("data/raw", "data/processed/metro", "data/processed/metro_repeat",
        "data/processed/economics", "data/processed/economics_repeat", "data/processed/economics_audit",
        "data/processed/simulation", "data/processed/simulation_repeat", "data/processed/simulation_audit",
        "docs", "examples", "tests", "config", "metro_data", ".venv")]
    if ROOT.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output overlaps protected project files")
    if any(Path(p).resolve().is_relative_to(output) for p in inputs):
        raise ValueError("output contains an input or program file")
    if output.exists() and any(output.iterdir()):
        manifest = output / "build_manifest.json"
        if not manifest.is_file():
            raise ValueError("output is not an owned passenger artifact directory")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("model_version") != MODEL_VERSION
                or data.get("status") not in {"success", "error"}
                or {p.name for p in output.iterdir()} - OUTPUT_NAMES - {"build_manifest.json"}):
            raise ValueError("output is not an owned passenger artifact directory")
    return output


def run_example(movement_path, demand_path, output):
    inputs = [Path(p).resolve() for p in (movement_path, demand_path, __file__, ROOT / "simulation.py", ROOT / "economics.py")]
    output = _safe_output(output, inputs)
    output.mkdir(parents=True, exist_ok=True)
    try:
        data, demand = [json.loads(p.read_text(encoding="utf-8-sig")) for p in inputs[:2]]
        service = PassengerSimulation(data, "service", demand)
        continuous = service.run()
        none = PassengerSimulation(data, "no_trains", demand).run()
        empty_demand = copy.deepcopy(demand)
        empty_demand["arrivals"] = []
        empty_engine = PassengerSimulation(data, "service", empty_demand)
        empty = empty_engine.run()
        start, cp, windows = 0, None, []
        for end in (60, 180, service.duration):
            part = service.run(start_second=start, end_second=end, checkpoint=cp)
            windows.append({"start": start, "end": end, "waiting_passenger_seconds": part["window_waiting_passenger_seconds"],
                            "queued": part["metrics"]["queued"], "onboard": part["metrics"]["onboard"]})
            cp, start = part["checkpoint"], end
        expected = {"entered": 5, "completed": 5, "queued": 0, "onboard": 0,
                    "waiting_passenger_seconds": 540, "mean_wait_seconds": "108", "p95_wait_seconds": 180,
                    "normal_threshold_exceedance_events": 1}
        actual = {k: continuous["metrics"][k] for k in expected}
        control = {"manual_expected": expected, "actual": actual, "manual_matches": actual == expected,
                   "no_trains_expected": {"queued": 5, "waiting_passenger_seconds": 3000},
                   "no_trains_actual": {k: none["metrics"][k] for k in ("queued", "waiting_passenger_seconds")},
                   "continuous_equals_split": cp == continuous["checkpoint"],
                   "continuous_equals_split_metrics": part["metrics"] == continuous["metrics"],
                   "continuous_equals_split_events": part["passenger_events"] == continuous["passenger_events"],
                   "continuous_equals_split_loads": part["load_events"] == continuous["load_events"],
                   "window_waiting_sum_matches": sum(w["waiting_passenger_seconds"] for w in windows) == 540,
                   "transport_work_equal_empty_and_loaded": service.motion == empty_engine.motion,
                   "motion_result_sha256": digest(service.motion),
                   "empty_motion_result_sha256": digest(empty_engine.motion), "release_authorized": False}
        control["no_trains_matches"] = control["no_trains_expected"] == control["no_trains_actual"]
        control["passed"] = all(control[k] for k in ("manual_matches", "no_trains_matches", "continuous_equals_split",
            "continuous_equals_split_metrics", "continuous_equals_split_events", "continuous_equals_split_loads",
            "window_waiting_sum_matches", "transport_work_equal_empty_and_loaded"))
        if not control["passed"]:
            raise ValueError("passenger fixture differs from the GOAL 3.2 manual controls")
        with TemporaryDirectory(prefix="passengers-") as temporary:
            stage = Path(temporary)
            values = {"service.passengers.json": continuous, "no_trains.passengers.json": none,
                      "empty_demand.json": empty, "movement.simulation.json": service.motion,
                      "split_run.json": {"windows": windows, "final": part},
                      "control_check.json": control, "assumptions.json": ASSUMPTIONS}
            for name, value in values.items():
                write_json(stage / name, value)
            write_text(stage / "summary.md", """# Пассажирская модель GOAL 3.2

Учебный пример: 5 пассажиров приходят в момент 0; отправления поездов происходят на 60-й и 180-й секундах. Вместимость каждого поезда равна 3, учебный штатный порог равен 2.

| Показатель | Ручной расчёт | Модель |
|---|---:|---:|
| Ожидание, пассажиро-секунды | 3 × 60 + 2 × 180 = 540 | 540 |
| Среднее ожидание, секунды | 540 / 5 = 108 | 108 |
| 95-й процентиль, секунды | 180 | 180 |
| Превышение штатного порога первого поезда | 3 > 2 | обнаружено |
| Очередь без поездов через 600 секунд | 5 | 5 |
| Ожидание без поездов, пассажиро-секунды | 5 × 600 = 3000 | 3000 |

Баланс выполнен: 5 вошедших = 5 завершивших поездку. Без поездов: 5 вошедших = 5 оставшихся в очереди. Среднее и P95 севших пассажиров без поездов неизвестны: посадок нет.

Непрерывный запуск и окна [0,60), [60,180), [180,600] дают одинаковые состояние, журнал и показатели. Ожидание по окнам: 300 + 240 + 0 = 540 пассажиро-секунд. Возраст очереди сохраняется, события на границе обрабатываются однократно.

При пустом и ненулевом спросе совпадает весь результат движения simulation-v1.1, включая транспортную работу. Пассажиры используют его события прибытия и отправления; время движения и остановок фиксировано.

Вместимость и порог относятся к учебному примеру. Настоящие OD, распределение по вагонам и применимость порога неизвестны. Пассажирский расчёт не подтверждает допуск состава к выпуску и не даёт денежной экономии.
""")
            manifest = {"model_version": MODEL_VERSION, "status": "success",
                        "inputs": {str(p): _sha(p) for p in inputs},
                        "outputs": {name: _sha(stage / name) for name in sorted(OUTPUT_NAMES)}}
            for name in sorted(OUTPUT_NAMES):
                (output / name).write_bytes((stage / name).read_bytes())
            write_json(output / "build_manifest.json", manifest)
        return control
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        write_json(output / "build_manifest.json", {"model_version": MODEL_VERSION, "status": "error", "error": str(exc)})
        raise


def check_output(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    fields_only(manifest, ("model_version", "status", "inputs", "outputs"))
    if manifest["model_version"] != MODEL_VERSION or manifest["status"] != "success":
        raise ValueError("passenger run has no current successful manifest")
    if (not isinstance(manifest["inputs"], dict) or len(manifest["inputs"]) != 5
            or not isinstance(manifest["outputs"], dict) or set(manifest["outputs"]) != OUTPUT_NAMES
            or {p.name for p in output.iterdir()} != OUTPUT_NAMES | {"build_manifest.json"}):
        raise ValueError("incomplete passenger manifest or artifact set")
    required = {str(Path(__file__).resolve()), str(ROOT / "simulation.py"), str(ROOT / "economics.py")}
    if not required <= set(manifest["inputs"]):
        raise ValueError("manifest lacks current program sources")
    for raw, expected in manifest["inputs"].items():
        path = Path(raw)
        if not path.is_absolute() or _sha(path) != expected:
            raise ValueError(f"input hash mismatch: {raw}")
    for name, expected in manifest["outputs"].items():
        if _sha(output / name) != expected:
            raise ValueError(f"output hash mismatch: {name}")
    return {"status": "passed", "files": len(OUTPUT_NAMES) + 1, "release_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--movement", type=Path, default=ROOT / "examples/passengers/movement.json")
    run.add_argument("--demand", type=Path, default=ROOT / "examples/passengers/demand.json")
    run.add_argument("--output", type=Path, default=ROOT / "data/processed/passengers")
    check = commands.add_parser("check")
    check.add_argument("--output", type=Path, default=ROOT / "data/processed/passengers")
    args = parser.parse_args(argv)
    try:
        result = run_example(args.movement, args.demand, args.output) if args.command == "run" else check_output(args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        print(f"passengers: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
