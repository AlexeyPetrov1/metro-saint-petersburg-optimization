"""One isolated physical pilot without the artificial terminal idle cutoff."""

import gzip
import hashlib
import importlib.util
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic


def remove_idle_cutoff(source):
    old = "max_idle=max(0,2*h-parameter(policy,'turnaround_s')-2)"
    if source.count(old) != 1:
        raise ValueError('exactly one idle cutoff required')
    return source.replace(old, "max_idle=float('inf') if repair_future_platforms else max(0,2*h-parameter(policy,'turnaround_s')-2)")


def terminal_total_audit(data, motion):
    caps = data['headway_policy']['demo']['terminal_parking_capacity_by_station']
    states = {t['train_id']: (t['state'], t['location']) for t in data['trains']}
    totals, parking, bad = dict.fromkeys(caps, 0), dict.fromkeys(caps, 0), []
    def inspect(second):
        for terminal, cap in caps.items():
            total = sum(s != 'moving' and loc == terminal for s, loc in states.values())
            idle = sum(s == 'standby' and loc == terminal for s, loc in states.values())
            totals[terminal] = max(totals[terminal], total)
            parking[terminal] = max(parking[terminal], idle)
            if total > 2 or idle > cap:
                bad.append(dict(second=second, terminal=terminal, total=total, standby=idle))
    inspect(0)
    for second, events in itertools.groupby(motion['events'], key=lambda e: e['second']):
        for event in events:
            states[event['train_id']] = (event['next_state'], event['location'])
        inspect(second)
    return dict(passed=not bad, maximum_total_by_terminal=totals,
                maximum_standby_by_terminal=parking, violations=bad[:20], half_open=True)


def pilot():
    root = Path(__file__).resolve().parent
    frozen = root / 'data/processed/metro44_search_execution_v2'
    source = frozen / 'metro44_planner_search.py'
    output = root / 'data/processed/metro44_noidle_20261009'
    output.mkdir(exist_ok=False)
    modified = output / 'experimental_planner.py'
    modified.write_text(remove_idle_cutoff(source.read_text('utf-8')), encoding='utf-8')
    sys.path.insert(0, str(frozen))
    spec = importlib.util.spec_from_file_location('_noidle_physical_pilot', modified)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    from simulation import simulate, digest
    with gzip.open(root / 'data/processed/metro_daily_experiments_20261009/baseline.json.gz', 'rt', encoding='utf-8') as stream:
        baseline = json.load(stream)
    start = monotonic()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), peak_headway=125,
        phase_seconds={'L1_01': 0, 'L1_19': 40}, occupancy_margin=2, as_of_second=3600,
        service_end_second=64800, no_passenger_simulation=True, existing_sources_modified=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        experimental_source_sha256=hashlib.sha256(modified.read_bytes()).hexdigest(),
        copied_from=str(source), input_data_digest=digest(baseline['data']))
    try:
        changed, audit = planner.rebuild_timetable(baseline['data'], 3600, 64800, headway=125,
            phase_seconds=result['phase_seconds'], occupancy_margin=2,
            repair_future_platforms=True, deadline=start + 455)
        motion = simulate(changed, changed['plans'][0]['plan_id'])
        terminal = terminal_total_audit(changed, motion)
        effective = planner.effective_peak_headways(changed, motion)
        physical = audit['movement_passed'] and terminal['passed'] and audit['peak_trains_on_line'] <= 53
        result.update(status='movement_checked' if physical else 'rejected',
            audit={k: v for k, v in audit.items() if not k.startswith('_') and k != 'departures'},
            terminal_total=terminal, effective_peak_headways=effective,
            passenger_validation_eligible=physical and audit['peak_trains_on_line'] > 42
                and effective['under_150_count'] > 0)
        if physical:
            for name, value in [('movement_input.json.gz', changed), ('motion.json.gz', motion)]:
                with gzip.open(output / name, 'wt', encoding='utf-8') as stream:
                    json.dump(value, stream, separators=(',', ':'))
    except (ValueError, TimeoutError) as error:
        result.update(status='evaluation_timeout' if isinstance(error, TimeoutError) else 'rejected',
                      reason=str(error), passenger_validation_eligible=False)
    result['elapsed_seconds'] = monotonic() - start
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    pilot()
