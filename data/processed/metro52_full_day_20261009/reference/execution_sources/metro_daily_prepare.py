"""Prepare frozen, causal inputs for a new isolated daily experiment."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from economics import write_json
from metro_daily import oracle_bundle, proposal_specs
from metro_daily_run import ROOT, bootstrap
from metro_ml import FrozenMetroForecaster, station_mapping
from metro_ml_compare import median_provider
from metro_ml_verify import file_hash
from metro_events import DEFAULT_REGISTRY, EventRegistry
from metro_monitoring import load_movement
from metro_pipeline import QUARTERS, annotate_calendar, vestibule_monitoring
from metro_calendar import DEFAULT_CALENDAR_SOURCE, load_calendar
from metro_quality import DEFAULT_HOURLY_SOURCE, annotate_quality, reconcile_sources
from metro_regime_forecast import regime_hourly_median


def prepare(output, *, anomaly_aware=True, knowledge_mode='strict', movement_path=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('preparation requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    previous = ROOT/'data/processed/metro_daily_experiments_20261009/preserved_before.json'
    if previous.exists():
        shutil.copy2(previous, output/'preserved_before.json')
    else:
        paths = [ROOT/'data/raw/metro'/name for name in ('data.zip','lgbm_final.zip')]
        write_json(output/'preserved_before.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),
            size=p.stat().st_size,sha256=file_hash(p)) for p in paths]))
    if knowledge_mode not in ('strict','publication_replay'):
        raise ValueError('online event knowledge mode is required')
    registry = EventRegistry.load() if anomaly_aware else None
    calendar = load_calendar() if anomaly_aware else None
    model = FrozenMetroForecaster(allow_training_period=False, event_registry=registry, knowledge_mode=knowledge_mode,
                                 use_retrospective_history_flags=not anomaly_aware)
    frame = model.load_simulator_frame()
    if anomaly_aware:
        quarters = pd.read_parquet(QUARTERS)
        reconciliation = reconcile_sources(quarters, pd.read_parquet(DEFAULT_HOURLY_SOURCE),
            service_hour_labels=True)
        vestibule_monitoring(quarters,reconciliation).to_parquet(output/'vestibule_monitoring.parquet',index=False)
        frame = annotate_quality(frame, reconciliation)
        frame = annotate_calendar(frame,calendar)
        reconciliation.to_parquet(output/'source_reconciliation.parquet', index=False)
        shutil.copy2(DEFAULT_REGISTRY, output/'events.json')
        shutil.copy2(DEFAULT_CALENDAR_SOURCE,output/'calendar.parquet')
        if movement_path:
            load_movement(movement_path).to_parquet(output/'movement.parquet', index=False)
    frame.to_parquet(output/'observations.parquet',index=False)
    if anomaly_aware:
        names = ['observations.parquet','events.json','source_reconciliation.parquet','vestibule_monitoring.parquet','calendar.parquet']
        if movement_path:
            names.append('movement.parquet')
        write_json(output/'monitoring_manifest.json',dict(schema_version=1,knowledge_mode=knowledge_mode,
            require_movement_coverage=False,line2_flow='unavailable',
            movement_telemetry='provided' if movement_path else 'unavailable',
            calendar=calendar.manifest(),
            raw_counts_preserved=True,frozen_weights_changed=False,
            artifact_hashes={name:file_hash(output/name) for name in names}))
    ids = list(station_mapping())
    folder = output/'forecasts'
    folder.mkdir()
    end = '2026-10-01T00:00:00+03:00'
    for hour in range(7,24):
        at = f'2026-09-30T{hour:02}:00:00+03:00'
        horizon = 1 if hour == 23 else 2
        bundles = dict(ML=model.forecast_bundle(frame,ids,at,horizon_hours=horizon),
            MEDIAN=(regime_hourly_median(frame,ids,at,event_registry=registry,horizon_hours=horizon,
                knowledge_mode=knowledge_mode,calendar=calendar) if anomaly_aware else median_provider(frame,ids,at,horizon_hours=horizon)),
            ORACLE=oracle_bundle(frame,ids,at,end))
        for strategy,bundle in bundles.items():
            write_json(folder/f'{strategy}_{hour:02}.json',bundle)
        print(json.dumps(dict(stage='daily_forecast_inputs',hour=hour,strategies=3)),flush=True)
    shared = bootstrap(output)
    manifest = json.loads((output/'experiment_manifest.json').read_text(encoding='utf-8'))
    campaign = dict(date='2026-09-30',strategies=['BASELINE','MEDIAN','ML','ORACLE'],
        baseline_identity=manifest['identity'],controller_decisions_per_arm=68,
        candidate_specs=proposal_specs(),bounded_search=True,no_retraining=True,actual_line=1,
        blue_line_transfer_performed=False,service_end=end,
        known_closure_tail='zero new entries after service end; existing passengers retain clearance horizon',
        source_hashes=manifest['contract']['source_hashes'],
        preparation_source_sha256=file_hash(Path(__file__)))
    campaign['anomaly_pipeline'] = dict(enabled=anomaly_aware,knowledge_mode=knowledge_mode,
        monitoring_manifest_sha256=file_hash(output/'monitoring_manifest.json') if anomaly_aware else None)
    write_json(output/'campaign_manifest.json',campaign)
    return shared['meta']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--legacy-pipeline',action='store_true')
    parser.add_argument('--event-knowledge-mode',choices=['strict','publication_replay'],default='strict')
    parser.add_argument('--movement',type=Path)
    args=parser.parse_args()
    prepare(args.output,anomaly_aware=not args.legacy_pipeline,knowledge_mode=args.event_knowledge_mode,
        movement_path=args.movement)
