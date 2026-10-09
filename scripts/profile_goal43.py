"""Isolated stage timings and cProfile traces for the full GOAL 4.3 line."""

import argparse
import cProfile
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import pstats
import sys
from threading import Event
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import hourly_median
from goal43_actions import generate_strategy_candidates
from goal43_policy import load_policy
from goal43_replay import actual_groups, build_operational_day, fixture_history
from passenger_groups import GroupedPassengerSimulation
from rolling import _decide_operational


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=38)
    parser.add_argument("--warmup", type=int, default=21600)
    parser.add_argument("--od", choices=("S1", "S2", "S3"), default="S1")
    parser.add_argument("--through", choices=("warmup", "checkpoint", "candidates", "decision"), default="candidates")
    parser.add_argument("--budget", type=float, default=120)
    parser.add_argument("--profile-stages", nargs="+", default=["build", "groups", "constructor", "warmup", "checkpoint", "candidates", "decision"])
    parser.add_argument("--output", type=Path, default=Path("data/processed/goal4_3_profile"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings = []

    def stage(name, function):
        print(json.dumps(dict(stage=name, status="started")), flush=True)
        profiler = cProfile.Profile()
        began = time.perf_counter()
        try:
            value = profiler.runcall(function) if name in args.profile_stages else function()
        finally:
            seconds = time.perf_counter()-began
            if name in args.profile_stages:
                profiler.dump_stats(str(args.output / (name+".prof")))
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
                (args.output / (name+".txt")).write_text(stream.getvalue(), encoding="utf-8")
            timings.append(dict(stage=name, seconds=seconds))
            (args.output / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(json.dumps(timings[-1]), flush=True)
        return value

    policy = load_policy()
    data, roster, ready, profiles, meta = stage("build", lambda: build_operational_day(policy, count=args.count))
    policy = data["headway_policy"]
    ids = [s["station_id"] for s in data["stations"] if s["kind"] == "line"]
    frame = fixture_history(ids)
    groups = stage("groups", lambda: actual_groups(frame, data, meta["service_end_second"])[args.od])
    source = stage("constructor", lambda: GroupedPassengerSimulation(data, data["plans"][0]["plan_id"], groups, profiles))
    snapshot = stage("warmup", lambda: source.run(end_second=args.warmup)["checkpoint"])
    print(json.dumps(dict(groups=len(groups), motion_events=len(source.motion["events"]),
        passenger_stream=len(source.stream), load_events=len(snapshot["load_events"]),
        board_events=len(snapshot["board_events"]), queued=snapshot["queued"])), flush=True)
    if args.through == "warmup": return
    stage("checkpoint", lambda: source._validate_checkpoint(snapshot, args.warmup))
    if args.through == "checkpoint": return
    candidates, search = stage("candidates", lambda: generate_strategy_candidates(data, args.warmup,
        policy, ready, minimum_remaining=args.count-2, include_recovery=False,
        time_limit_seconds=5, deadline=time.monotonic()+args.budget))
    print(json.dumps(dict(candidates=[dict(candidate_id=c["candidate_id"], status=c["status"]) for c in candidates],
                          search=search)), flush=True)
    if args.through == "candidates": return
    now = datetime.fromisoformat(data["period_start"])+timedelta(seconds=args.warmup)
    bundle = hourly_median(frame, ids, now.isoformat())
    began = time.monotonic()
    decision = stage("decision", lambda: _decide_operational(data, {args.od: source}, {args.od: snapshot},
        profiles, ready, bundle, policy, candidates=candidates, budget_seconds=args.budget,
        incidents=(), profile="fixed", seed=7, include_recovery=False, minimum_remaining=args.count-2,
        began=began, deadline=began+args.budget, cancelled=Event()))
    print(json.dumps(dict(selection=decision["selection"], elapsed_seconds=decision["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
