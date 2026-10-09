"""Prepare immutable entry-quality inputs and replay the anomaly-aware pipeline."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from forecasting import hourly_median
from metro_anomaly_monitor import monitor_anomalies
from metro_calendar import DEFAULT_CALENDAR_SOURCE, calendar_day, load_calendar
from metro_event_evaluation import evaluate_forecasts
from metro_events import DEFAULT_REGISTRY, EventRegistry, VESTIBULE_STATIONS
from metro_ml import file_hash
from metro_monitoring import control_guard, guard_candidates, load_movement
from metro_quality import DEFAULT_HOURLY_SOURCE, annotate_quality, reconcile_sources
from metro_regime_forecast import regime_hourly_median


ROOT = Path(__file__).resolve().parent
QUARTERS = ROOT/'data/processed/metro/entries_vestibule_15m.parquet'
PIPELINE_CODE = ('metro_pipeline.py','metro_events.py','metro_quality.py','metro_monitoring.py',
    'metro_regime_forecast.py','metro_event_evaluation.py','metro_anomaly_monitor.py','metro_calendar.py',
    'forecasting.py','metro_ml.py','rolling.py','metro_daily_prepare.py','metro_daily_run.py')


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is pd.NaT or value is pd.NA or isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _empty(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('pipeline requires an empty output directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _verify(files):
    return all(Path(row['path']).is_file() and file_hash(row['path']) == row['sha256'] for row in files)


def _snapshot_code(output):
    folder = output/'source_versions'
    folder.mkdir()
    hashes = {}
    for name in PIPELINE_CODE:
        source = ROOT/name
        if source.exists():
            hashes[name] = file_hash(source)
            shutil.copy2(source, folder/f'{hashes[name]}_{name}')
    return hashes


def load_monitoring_context(output):
    output = Path(output)
    path = output/'monitoring_manifest.json'
    if not path.exists():
        path = output/'pipeline_manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if any(file_hash(output/name) != sha for name, sha in manifest['artifact_hashes'].items()):
        raise ValueError('monitoring snapshot changed')
    return dict(observations=pd.read_parquet(output/'observations.parquet'),
        event_registry=EventRegistry.load(output/'events.json'),
        movement=pd.read_parquet(output/'movement.parquet') if (output/'movement.parquet').exists() else None,
        knowledge_mode=manifest.get('knowledge_mode','strict'),
        enable_anomaly_monitor=True,
        vestibules=pd.read_parquet(output/'vestibule_monitoring.parquet') if (output/'vestibule_monitoring.parquet').exists() else None,
        require_movement_coverage=manifest.get('require_movement_coverage',False))


def vestibule_monitoring(quarters, reconciliation):
    keys = ['station_id','vestibule_id','hour_start']
    frame = quarters.assign(hour_start=quarters.interval_start.dt.floor('h'))
    frame = frame.merge(reconciliation[[*keys,'quality_status','quality_available_at']],
        on=keys,how='left',validate='many_to_one').drop(columns='hour_start')
    frame['is_complete'] = frame.entry_count.notna() & frame.get('is_complete',True)
    return frame


def annotate_calendar(frame, calendar):
    result = frame.copy()
    dates = result.interval_start.dt.tz_convert('Europe/Moscow').dt.date
    kinds = {day:calendar_day(day,calendar)['day_kind'] for day in dates.unique()}
    result['day_kind'] = dates.map(kinds)
    result['is_holiday'] = result.day_kind.eq('holiday')
    return result


def prepare_pipeline(output, *, quarters_path=QUARTERS, hourly_path=DEFAULT_HOURLY_SOURCE,
                     registry_path=DEFAULT_REGISTRY, movement_path=None, calendar_path=DEFAULT_CALENDAR_SOURCE):
    output = _empty(output)
    sources = [Path(p).resolve() for p in (quarters_path, hourly_path, registry_path)]
    if movement_path:
        sources.append(Path(movement_path).resolve())
    if calendar_path is not None:
        sources.append(Path(calendar_path).resolve())
    preserved = sources+[p for p in (ROOT/'data/raw/metro/data.zip', ROOT/'data/raw/metro/lgbm_final.zip',
        ROOT/'data/processed/metro_anomaly_audit_20261009_v2/summary.json') if p.exists()]
    files = [dict(path=str(p), sha256=file_hash(p), size=p.stat().st_size) for p in preserved]
    _write(output/'INCOMPLETE.json', dict(status='preparation_incomplete', preserved_files=files))
    registry = EventRegistry.load(registry_path)
    calendar = load_calendar(calendar_path) if calendar_path is not None else None
    code_hashes = _snapshot_code(output)
    q, h = pd.read_parquet(quarters_path), pd.read_parquet(hourly_path)
    reconciled = reconcile_sources(q, h, service_hour_labels=True)
    q.to_parquet(output/'raw_vestibule_observations.parquet', index=False)
    vestibule_monitoring(q, reconciled).to_parquet(output/'vestibule_monitoring.parquet',index=False)
    reconciled.to_parquet(output/'source_reconciliation.parquet', index=False)
    if not q.vestibule_id.map(VESTIBULE_STATIONS).eq(q.station_id).all():
        raise ValueError('pipeline requires the declared Line 1 vestibule/station mapping')
    expected = pd.Series(VESTIBULE_STATIONS).value_counts()
    station = q.groupby(['station_id', 'interval_start', 'interval_end'], observed=True).agg(
        entry_count=('entry_count', 'sum'), present=('vestibule_id', 'nunique'),
        observed=('entry_count', 'count')).reset_index()
    station['is_complete'] = station.present.eq(station.station_id.map(expected)) & station.observed.eq(station.present)
    totals = q.groupby(['station_id', 'interval_start', 'interval_end'], observed=True).entry_count.sum(min_count=1)
    station['entry_count'] = totals.reindex(pd.MultiIndex.from_frame(station[['station_id','interval_start','interval_end']])).to_numpy()
    station['entry_count'] = station.entry_count.astype('Int64')
    if 'is_complete' in q:
        complete = q.groupby(['station_id','interval_start','interval_end'], observed=True).is_complete.all()
        station['is_complete'] &= complete.reindex(pd.MultiIndex.from_frame(station[['station_id','interval_start','interval_end']])).to_numpy()
    station['source_id'] = 'uploaded-spb-line1-15min'
    station['available_at'] = station.interval_end
    # Source publication/receipt, when supplied, may delay observations.
    available = q.interval_end.copy()
    for field in ('available_at', 'received_at'):
        if field in q:
            available = pd.concat([available, q[field]], axis=1).max(axis=1).where(available.notna() & q[field].notna())
    metadata = q.assign(_available=available).groupby(['station_id', 'interval_start'], observed=True).agg(
        latest=('_available', 'max'), known=('_available', lambda s: s.notna().all())).reset_index()
    station = station.merge(metadata, on=['station_id', 'interval_start'], validate='one_to_one')
    station['available_at'] = station.latest.where(station.known)
    station = annotate_quality(station.drop(columns=['latest', 'known']), reconciled)
    station = annotate_calendar(station,calendar)
    station.to_parquet(output/'observations.parquet', index=False)
    shutil.copy2(registry_path, output/'events.json')
    if calendar_path is not None:
        shutil.copy2(calendar_path,output/'calendar.parquet')
    if movement_path:
        load_movement(movement_path).to_parquet(output/'movement.parquet', index=False)
    day = reconciled.hour_start.dt.tz_convert('Europe/Moscow').dt.hour.between(6,23)
    manifest = dict(schema_version=1, pipeline_version='metro-anomaly-aware-v1',
        code_hashes=code_hashes,
        calendar=calendar.manifest() if calendar is not None else None,
        source_files=files, line_ids=sorted(set(station.station_id.str.split('_').str[0])),
        observation_rows=len(station), observation_total=int(station.entry_count.sum()),
        complete_rows=int(station.is_complete.sum()), event_versions=len(registry.records),
        missing_station_ids=sorted(set(expected.index)-set(station.station_id)),
        source_quality=reconciled.quality_status.value_counts().to_dict(),
        comparable_daytime_hours=int((day & reconciled.quality_status.ne('unknown')).sum()),
        mismatched_daytime_hours=int((day & reconciled.quality_status.eq('source_mismatch')).sum()),
        raw_counts_preserved=True, unknown_source_coverage_retained=True,
        entry_availability='provided publication/receipt or explicit assumption at interval end',
        service_hour_correction='hourly labels 00:00–02:00 shifted one calendar day; raw labels retained',
        event_ingestion='curated evidence ingested 2026-10-09; historical strict replay cannot use it earlier',
        movement_telemetry='provided' if movement_path else 'unavailable', line2_flow='unavailable',
        frozen_weights_changed=False, preserved_inputs_verified=_verify(files))
    if not manifest['preserved_inputs_verified']:
        raise ValueError('source changed during preparation')
    manifest['artifact_hashes'] = {p.name: file_hash(p) for p in output.iterdir() if p.suffix in ('.parquet', '.json') and p.name != 'INCOMPLETE.json'}
    _write(output/'pipeline_manifest.json', manifest)
    (output/'INCOMPLETE.json').unlink()
    return manifest


def default_cases():
    cases = [dict(as_of=f'2026-09-30T{h:02}:00:00+03:00', station_ids=[f'L1_{i:02}' for i in range(1,20)],
                  horizon_hours=1, purpose='temporal_holdout') for h in (8,9,12,17,18,22)]
    cases += [dict(as_of=at, station_ids=ids, horizon_hours=1, purpose=purpose) for at, ids, purpose in (
        ('2026-07-29T08:00:00+03:00', ['L1_05'], 'repair_diagnostic'),
        ('2026-07-15T08:00:00+03:00', ['L1_18'], 'repair_diagnostic'),
        ('2026-07-17T18:00:00+03:00', ['L1_18','L1_19'], 'disruption_diagnostic'),
        ('2026-05-09T12:00:00+03:00', ['L1_10'], 'conflicting_timing_diagnostic'))]
    return cases


def replay_pipeline(output, *, cases=None, knowledge_mode='strict', replay_name='replay'):
    output = Path(output).resolve()
    if Path(replay_name).name != replay_name or replay_name in ('', '.', '..'):
        raise ValueError('replay_name must be a single directory name')
    manifest = json.loads((output/'pipeline_manifest.json').read_text(encoding='utf-8'))
    if not _verify(manifest['source_files']) or any(file_hash(output/name) != sha for name,sha in manifest['artifact_hashes'].items()):
        raise ValueError('prepared sources changed before replay')
    if any(file_hash(ROOT/name) != sha for name, sha in manifest.get('code_hashes',{}).items()):
        raise ValueError('pipeline code changed; prepare a new isolated run')
    folder = _empty(output/replay_name)
    _write(folder/'INCOMPLETE.json', dict(status='replay_incomplete'))
    registry = EventRegistry.load(output/'events.json')
    frame = pd.read_parquet(output/'observations.parquet')
    calendar = load_calendar(output/'calendar.parquet') if (output/'calendar.parquet').exists() else None
    vestibules = pd.read_parquet(output/'vestibule_monitoring.parquet') if (output/'vestibule_monitoring.parquet').exists() else None
    movement = pd.read_parquet(output/'movement.parquet') if (output/'movement.parquet').exists() else None
    forecasts, errors, guards, monitors = dict(legacy=[], regime=[]), [], [], []
    selected_cases = list(cases if cases is not None else default_cases())
    for case_index, case in enumerate(selected_cases):
        at, ids = case['as_of'], case['station_ids']
        horizon = case.get('horizon_hours', 1)
        guard = control_guard(frame, registry, at, ids, movement, knowledge_mode=knowledge_mode,
                              horizon_seconds=horizon*3600)
        decisions = guard_candidates([dict(candidate_id='KEEP', action=dict(kind='KEEP'), status='movement_checked'),
            dict(candidate_id='REDUCE', action=dict(kind='REPLAN', count_delta=-2), status='movement_checked')], guard)
        guards.append(dict(case_index=case_index, case=case, guard=guard, candidates=decisions))
        monitored = monitor_anomalies(frame[frame.station_id.isin(ids)], registry, at,
            vestibules=vestibules[vestibules.station_id.isin(ids)] if vestibules is not None else None,
            movement=movement,knowledge_mode=knowledge_mode)
        monitors.append(dict(case_index=case_index,case=case,**monitored))
        for name in forecasts:
            try:
                bundle = (hourly_median(frame.drop(columns=['quality_status','quality_available_at']), ids, at, horizon)
                    if name == 'legacy' else regime_hourly_median(frame, ids, at, event_registry=registry,
                        horizon_hours=horizon, knowledge_mode=knowledge_mode,calendar=calendar))
                _write(folder/f'{name}_{case_index:04}_{pd.Timestamp(at).strftime("%Y%m%d_%H%M")}.json', bundle)
                forecasts[name].extend(dict(**row, as_of=pd.Timestamp(bundle['as_of']),
                    available_at=pd.Timestamp(bundle['available_at']), features_available_at=pd.Timestamp(bundle['features_available_at']),
                    model_version=bundle['model_version'], purpose=case.get('purpose','diagnostic'), case_index=case_index) for row in bundle['rows'])
            except ValueError as error:
                errors.append(dict(model=name, case_index=case_index, case=case, reason=str(error)))
    models, eligible_keys, prepared, raw_predictions, repeated_keys = {}, {}, {}, {}, {}
    keys = ['station_id', 'interval_start', 'as_of']

    def score(data):
        result = evaluate_forecasts(data.drop_duplicates(keys), frame, registry)
        result.pop('rows')
        return result

    def by_purpose(data):
        return {str(purpose): score(part) for purpose, part in data.groupby('purpose', sort=True)}

    for name, rows in forecasts.items():
        eligible_keys[name], repeated_keys[name] = set(), 0
        if not rows:
            models[name] = dict(status='no_scorable_forecasts', by_purpose={})
            continue
        predicted = pd.DataFrame(rows)
        for field in ('interval_start', 'interval_end', 'as_of', 'available_at', 'features_available_at'):
            predicted[field] = pd.to_datetime(predicted[field], utc=True)
        repeated_keys[name] = int(predicted.duplicated(keys).sum())
        # Overlapping requests can repeat an identical forecast. Contradictory
        # predictions for one model/key are reported and excluded, never averaged.
        fields = ['predicted_count', 'interval_end', *[c for c in ('q10', 'q90') if c in predicted]]
        conflicts = predicted.groupby(keys, observed=True)[fields].nunique(dropna=False).gt(1).any(axis=1)
        conflicting = set(conflicts.index[conflicts].tolist())
        if conflicting:
            errors.append(dict(model=name, reason='conflicting predictions for repeated forecast keys',
                               forecast_keys=[dict(zip(keys, key)) for key in sorted(conflicting)]))
            predicted = predicted.loc[~pd.MultiIndex.from_frame(predicted[keys]).isin(conflicting)].copy()
        raw_predictions[name] = predicted
        prepared[name] = predicted.drop_duplicates(keys).reset_index(drop=True)
        scored = evaluate_forecasts(prepared[name], frame, registry)
        scored_rows = scored.pop('rows')
        eligible_keys[name] = {tuple(row[c] for c in keys) for row in scored_rows if row['evaluation_eligible']}
        _write(folder/f'{name}_scored_rows.json', scored_rows)
        scored['by_purpose'] = by_purpose(predicted)
        models[name] = scored
    common = set.intersection(*eligible_keys.values())
    comparison_models = {}
    template = next(iter(prepared.values()), None)
    if template is None:
        template = pd.DataFrame({c: pd.Series(dtype='datetime64[ns, UTC]') for c in
            ['interval_start', 'interval_end', 'as_of', 'available_at', 'features_available_at']})
        template['station_id'], template['purpose'] = pd.Series(dtype=str), pd.Series(dtype=str)
        template['predicted_count'] = pd.Series(dtype=float)
    for name in forecasts:
        predicted = prepared.get(name, template.iloc[:0])
        paired = predicted.loc[pd.MultiIndex.from_frame(predicted[keys]).isin(common)]
        comparison_models[name] = score(paired)
        purposes = raw_predictions.get(name, template.iloc[:0])
        purposes = purposes.loc[pd.MultiIndex.from_frame(purposes[keys]).isin(common)]
        comparison_models[name]['by_purpose'] = by_purpose(purposes)
    origins = sorted({pd.Timestamp(case['as_of']).isoformat() for case in selected_cases})
    all_eligible = set.union(*eligible_keys.values())
    observations = {}
    for station, target, origin in all_eligible:
        observations.setdefault((station, target), set()).add(origin)
    overlapping = sum(len(origins) > 1 for origins in observations.values())
    warnings = []
    if any(repeated_keys.values()):
        warnings.append('Repeated station/target/origin forecast keys were deduplicated for aggregate metrics.')
    if overlapping:
        warnings.append('Some observed targets occur under several forecast origins; their errors are correlated.')
    if any(len(values) != len(common) for values in eligible_keys.values()):
        warnings.append('Model metrics have unequal coverage; comparative metrics use common eligible keys only.')
    scope = dict(requested_case_count=len(selected_cases), cases=selected_cases, origins=origins,
        dates=sorted({pd.Timestamp(at).tz_convert('Europe/Moscow').date().isoformat() for at in origins}),
        purposes=sorted({case.get('purpose','diagnostic') for case in selected_cases}),
        model_eligible_rows={name: len(values) for name, values in eligible_keys.items()},
        unique_observed_targets=len(observations), observed_targets_with_multiple_origins=overlapping,
        repeated_forecast_keys=repeated_keys, warnings=warnings)
    comparison = dict(status='available' if common else 'unavailable', key_columns=keys,
        common_eligible_rows=len(common), models=comparison_models,
        scope='Intersection of complete observed station/target/origin keys scored by every model.')
    summary = dict(knowledge_mode=knowledge_mode, models=models, comparison=comparison, unscorable_cases=errors,
        code_hashes=manifest.get('code_hashes',{}),
        preserved_inputs_verified=_verify(manifest['source_files']),
        missing_event_truth='incident recall/delay and false alerts unavailable without exhaustive incident log and motion telemetry',
        evaluation_scope=scope,
        latent_demand_estimated=False, line2_metrics=None)
    _write(folder/'guard_decisions.json', guards)
    _write(folder/'monitor_alerts.json',monitors)
    _write(folder/'summary.json', summary)
    (folder/'INCOMPLETE.json').unlink()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'replay'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--movement', type=Path)
    parser.add_argument('--knowledge-mode', choices=['strict','publication_replay'], default='strict')
    parser.add_argument('--replay-name', default='replay')
    args = parser.parse_args()
    result = (prepare_pipeline(args.output, movement_path=args.movement) if args.command == 'prepare' else
        replay_pipeline(args.output, knowledge_mode=args.knowledge_mode, replay_name=args.replay_name))
    concise = (dict(observation_rows=result['observation_rows'],source_quality=result['source_quality'],
        preserved_inputs_verified=result['preserved_inputs_verified']) if args.command == 'prepare' else
        dict(common_eligible_rows=result['comparison']['common_eligible_rows'],
            model_wape={name:value['all']['wape'] for name,value in result['comparison']['models'].items()},
            unscorable_cases=len(result['unscorable_cases']),preserved_inputs_verified=result['preserved_inputs_verified']))
    print(json.dumps(dict(stage=args.command,output=str(args.output),**_clean(concise)),ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
