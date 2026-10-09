"""Profile a short passenger window using a saved historical replay input."""

import argparse
import cProfile
import gzip
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from passenger_groups import GroupedPassengerSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/goal4_3/runs/station_stop_3_KEEP_S1_7"))
    parser.add_argument("--start", type=int, default=1843)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--budget", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/processed/goal4_3_profile/historical_n3_window"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name in ("movement_input.json", "profiles.json", "groups.json.gz"):
        path = args.input / name
        raw = path.read_bytes()
        inputs[name] = dict(path=path.as_posix(), sha256=hashlib.sha256(raw).hexdigest())
        payload = gzip.decompress(raw) if name.endswith(".gz") else raw
        inputs[name]["value"] = json.loads(payload)
    timings = []

    def stage(name, function):
        print(json.dumps(dict(stage=name, status="started")), flush=True)
        profiler = cProfile.Profile()
        began = time.perf_counter()
        try:
            return profiler.runcall(function)
        finally:
            seconds = time.perf_counter() - began
            profiler.dump_stats(str(args.output / (name + ".prof")))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
            (args.output / (name + ".txt")).write_text(stream.getvalue(), encoding="utf-8")
            timings.append(dict(stage=name, seconds=seconds))
            (args.output / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(json.dumps(timings[-1]), flush=True)

    data, profiles, groups = (inputs[name]["value"] for name in ("movement_input.json", "profiles.json", "groups.json.gz"))
    runner = stage("constructor", lambda: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], groups, profiles))
    checkpoint = stage("warmup", lambda: runner.run(end_second=args.start, deadline=time.monotonic() + args.budget)["checkpoint"])
    stage("checkpoint", lambda: runner._validate_checkpoint(checkpoint, args.start))
    out = stage("resume", lambda: runner.run(start_second=args.start, end_second=args.start + args.seconds,
        checkpoint=checkpoint, deadline=time.monotonic() + args.budget))
    summary = dict(inputs={name: {k:v for k,v in row.items() if k != "value"} for name,row in inputs.items()},
        start_second=args.start, duration_seconds=args.seconds, timings=timings,
        groups=len(groups), motion_events=len(runner.motion["events"]), metrics=out["metrics"],
        queue_cohorts=sum(len(rows) for rows in out["checkpoint"]["queues"].values()),
        board_events=len(out["checkpoint"]["board_events"]), load_events=len(out["checkpoint"]["load_events"]),
        code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("simulation.py", "passenger_groups.py")})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("inputs", "code_sha256", "metrics")}), flush=True)


if __name__ == "__main__":
    main()
