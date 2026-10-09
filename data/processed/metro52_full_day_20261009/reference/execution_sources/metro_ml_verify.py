"""Recompute forecast scores, paired identities and saved movement ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import pandas as pd

from forecasting import validate_bundle
from simulation import digest

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_preserved(manifest, workspace=ROOT):
    rows = read(manifest)['files']
    for row in rows:
        path = Path(workspace)/row['path']
        require(path.is_file() and path.stat().st_size == row['size'] and
            file_hash(path) == row['sha256'], f"preserved bytes changed: {row['path']}")
    return len(rows)


def score_forecast(bundle, frame, decision_time):
    predictions = pd.DataFrame(bundle['rows'])
    predictions['interval_start'] = pd.to_datetime(predictions.interval_start, utc=True)
    predictions = predictions[predictions.interval_start.ge(pd.Timestamp(decision_time))]
    joined = predictions.merge(frame[['station_id', 'interval_start', 'entry_count', 'is_complete']],
        on=['station_id', 'interval_start'], how='left', validate='one_to_one')
    require(len(joined) > 0 and joined.is_complete.eq(True).all() and joined.entry_count.notna().all(),
        'forecast actual coverage is incomplete')
    error = float((joined.predicted_count-joined.entry_count).abs().sum())
    total = int(joined.entry_count.sum())
    return dict(rows=len(joined), actual_entries=total, absolute_error=error,
        wape_percent=100*error/total if total else None)


def check_case_sources(saved, workspace, output):
    archived = saved.get('archived_sources', {})
    for name, expected in saved['contract']['source_hashes'].items():
        path = Path(output)/archived[name] if name in archived else Path(workspace)/name
        require(file_hash(path) == expected, f'case source changed: {name}')


def check_model_archive(archive, inputs):
    count = 0
    with zipfile.ZipFile(archive) as stream:
        for item in stream.infolist():
            if item.is_dir() or not item.filename.startswith('lgbm_final/'):
                continue
            expected = hashlib.sha256(stream.read(item)).hexdigest()
            require(file_hash(Path(inputs)/item.filename) == expected, f'loaded weights changed: {item.filename}')
            count += 1
    require(count > 0, 'uploaded archive contains no model files')
    return count


def verify_case(folder, frame):
    saved = read(folder/'experiment_manifest.json')
    contract = saved['contract']
    case = read(folder/'metrics.json')
    require(digest(contract) == saved['experiment_identity'] == case['experiment_identity'], 'case identity changed')
    require(digest(read(folder/'warmup_checkpoint.json')) == contract['initial_checkpoint_digest'], 'warmup snapshot changed')
    require(digest(read(folder/'actual_groups.json')) == contract['actual_groups_digest'], 'actual demand changed')
    check_case_sources(saved, ROOT, folder.parent.parent)
    require(set(case['strategies']) == {'KEEP', 'MEDIAN', 'ML'}, 'unpaired strategies')
    first_run = next(iter(case['strategies'].values()))
    movement = read(folder.parent.parent/first_run['run_folder']/'movement_input.json')
    origin = pd.Timestamp(movement['period_start'])
    actual = frame[frame.station_id.isin(contract['station_ids']) & frame.interval_start.ge(origin) &
        frame.interval_start.lt(origin+pd.Timedelta(seconds=contract['service_seconds']))]
    require(len(actual) == contract['actual_rows'] and actual.is_complete.eq(True).all() and
        int(actual.entry_count.sum()) == contract['actual_entries'], 'case actual coverage differs from original quarter observations')
    canonical = actual.sort_values(['station_id', 'interval_start']).assign(
        interval_start=lambda data: data.interval_start.map(lambda x: x.isoformat()),
        interval_end=lambda data: data.interval_end.map(lambda x: x.isoformat()))
    require(digest(canonical[['station_id', 'interval_start', 'interval_end', 'entry_count']].to_dict('records')) ==
        contract['actual_interval_digest'], 'actual target values differ from original quarter observations')
    scored = 0
    for strategy, result in case['strategies'].items():
        require(result['initial_checkpoint_digest'] == contract['initial_checkpoint_digest'] and
            result['actual_groups_digest'] == contract['actual_groups_digest'], 'strategies use different inputs')
        decisions = read(folder/f'{strategy.lower()}_decisions.json')
        bundles = read(folder/f'{strategy.lower()}_forecasts.json')
        require(len(decisions) == len(bundles) == len(result['forecast_errors']), 'forecast or decision missing')
        for decision, bundle, recorded in zip(decisions, bundles, result['forecast_errors']):
            validate_bundle(bundle, contract['station_ids'], decision['decision_time'])
            actual = score_forecast(bundle, frame, decision['decision_time'])
            for key, value in actual.items():
                require(recorded[key] == value if value is None else math.isclose(recorded[key], value, abs_tol=1e-9),
                    f'forecast score differs: {strategy}/{key}')
            scored += actual['rows']
    return dict(case_id=case['case_id'], strategies=3, forecast_rows_checked=scored)


def verify(output):
    output = Path(output).resolve()
    frame = pd.read_parquet(ROOT/'data/processed/metro/entries_station_15m.parquet')
    cases = sorted((output/'cases').glob('*/metrics.json'))
    require(bool(cases), 'no completed comparisons')
    checks = [verify_case(path.parent, frame) for path in cases]
    records = []
    for path in cases:
        for result in read(path)['strategies'].values():
            summary = read(output/result['run_folder']/'summary.json')
            records.append(dict(summary, folder=result['run_folder']))
    (output/'runs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    from goal43_verify import verify as verify_ledgers
    ledgers = verify_ledgers(output)
    preserved = check_preserved(ROOT/'data/processed/metro_ml_verification/preserved_before.json')
    models = check_model_archive(ROOT/'data/raw/metro/lgbm_final.zip', ROOT/'data/processed/metro_ml_inputs')
    result = dict(status='passed', cases=checks, run_count=ledgers['run_count'], preserved_files=preserved,
        unchanged_model_files=models,
        ledger_verification='verification_report.json',
        operational_violation_count=sum(len(row['violations']) for row in ledgers['runs']),
        integrity_pass_does_not_mean_policy_admissibility=True)
    (output/'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(verify(parser.parse_args().output), ensure_ascii=False))
