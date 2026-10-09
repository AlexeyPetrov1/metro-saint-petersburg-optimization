"""One isolated pilot combining unrestricted idle and shared terminal slots."""

import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from metro44_planner_noidle import remove_idle_cutoff, terminal_total_audit


def bind_terminal_adapter(source):
    old = 'from dwell import compile_operational_timing, timing_key'
    if source.count(old) != 1:
        raise ValueError('exactly one compiler import required')
    return source.replace(old, 'from dwell import timing_key\nfrom metro44_terminal_timing import compile_terminal_timing as compile_operational_timing')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot():
    root = Path(__file__).resolve().parent
    frozen = root / 'data/processed/metro44_search_execution_v2'
    manifest = json.loads((frozen / 'execution_manifest.json').read_text('utf-8'))
    if any(sha(frozen / name) != value for name,value in manifest['source_hashes'].items()):
        raise ValueError('frozen physical snapshot changed')
    output = root / 'data/processed/metro44_noidle_terminal_20261009'
    output.mkdir(exist_ok=False)
    source = frozen / 'metro44_planner_search.py'
    modified = output / 'experimental_planner.py'
    modified.write_text(bind_terminal_adapter(remove_idle_cutoff(source.read_text('utf-8'))), 'utf-8')
    shutil.copy2(root / 'metro44_terminal_timing.py', output / 'metro44_terminal_timing.py')
    shutil.copy2(root / 'metro44_planner_noidle.py', output / 'metro44_planner_noidle.py')
    ledger = dict(frozen_snapshot=str(frozen), frozen_source_hashes=manifest['source_hashes'],
        experimental_source_hashes={name: sha(output / name) for name in
            ('experimental_planner.py', 'metro44_terminal_timing.py', 'metro44_planner_noidle.py')},
        runner_sha256=sha(Path(__file__)), original_sources_modified=False)
    (output / 'source_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n', 'utf-8')
    sys.path.insert(0, str(frozen))
    sys.path.insert(0, str(output))
    spec = importlib.util.spec_from_file_location('_noidle_terminal_pilot', modified)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    from simulation import simulate, digest
    with gzip.open(root / 'data/processed/metro_daily_experiments_20261009/baseline.json.gz', 'rt', encoding='utf-8') as stream:
        baseline = json.load(stream)
    started = monotonic()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), peak_headway=125,
        phase_seconds={'L1_01': 0, 'L1_19': 40}, occupancy_margin=2, as_of_second=3600,
        service_end_second=64800, input_data_digest=digest(baseline['data']),
        no_passenger_simulation=True, existing_sources_modified=False,
        terminal_resource_assumption='same two slots for incoming service and entry; signalling unverified')
    try:
        changed, audit = planner.rebuild_timetable(baseline['data'], 3600, 64800, headway=125,
            phase_seconds=result['phase_seconds'], occupancy_margin=2,
            repair_future_platforms=True, deadline=started + 455)
        motion = simulate(changed, changed['plans'][0]['plan_id'])
        terminal = terminal_total_audit(changed, motion)
        effective = planner.effective_peak_headways(changed, motion)
        physical = audit['movement_passed'] and terminal['passed'] and audit['peak_trains_on_line'] <= 53
        result.update(status='movement_checked' if physical else 'rejected',
            audit={k:v for k,v in audit.items() if not k.startswith('_') and k != 'departures'},
            terminal_total=terminal, effective_peak_headways=effective,
            passenger_validation_eligible=physical and audit['peak_trains_on_line'] > 42 and effective['under_150_count'] > 0)
        if physical:
            for name,value in [('movement_input.json.gz', changed), ('motion.json.gz', motion)]:
                with gzip.open(output / name, 'wt', encoding='utf-8') as stream:
                    json.dump(value, stream, separators=(',', ':'))
    except (ValueError, TimeoutError) as error:
        result.update(status='evaluation_timeout' if isinstance(error, TimeoutError) else 'rejected',
                      reason=str(error), passenger_validation_eligible=False)
    result['elapsed_seconds'] = monotonic() - started
    result['source_ledger_preserved'] = all(sha(frozen / name) == value for name,value in ledger['frozen_source_hashes'].items())
    result['source_ledger_preserved'] &= all(sha(output / name) == value for name,value in ledger['experimental_source_hashes'].items())
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    pilot()
