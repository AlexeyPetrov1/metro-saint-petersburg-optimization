"""Reallocate saved hourly ML totals to the exact GRU comparison profile."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from line1 import largest_remainders
from metro48_forecasts import score_rows
from metro48_gru import FrozenQuarterGRU
from metro_ml import file_hash


def reprofile_old_ml(rows, profile_provider):
    result = rows.copy()
    result['hour'] = result.interval_start.dt.floor('h')
    for (station, hour), group in result.groupby(['station_id', 'hour']):
        if len(group) != 4 or group.interval_start.dt.minute.nunique() != 4 or not group.issued_at.eq(hour).all():
            raise ValueError('reprofiling requires four forecasts issued at the same hour boundary')
        profile = np.asarray(profile_provider(station, hour))
        weights = np.maximum(0, np.rint(profile * 1_000_000_000)).astype(np.int64).tolist()
        counts = largest_remainders(int(group.predicted_count.sum()), weights)
        for index, row in group.iterrows():
            result.loc[index, 'predicted_count'] = counts[row.interval_start.minute // 15]
    result['model'] = 'OLD_ML_SHARED_PROFILE'
    return result.drop(columns='hour')


def run(benchmark, output):
    benchmark, output = Path(benchmark), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    path = benchmark / 'next_quarter_predictions.parquet'
    before = file_hash(path)
    data = pd.read_parquet(path)
    model = FrozenQuarterGRU()
    old = data[data.model.eq('OLD_ML')].copy()
    shared = reprofile_old_ml(old, model.station_profile)
    all_rows = pd.concat([data[data.stage.eq('development')], shared], ignore_index=True)
    frame = pd.read_parquet('data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet')
    scores = {}
    for name, rows in all_rows.groupby('model'):
        scores[name] = score_rows(rows.to_dict('records'), frame)
    shared.to_parquet(output / 'old_ml_shared_profile.parquet', index=False)
    result = dict(source_benchmark=str(benchmark.resolve()), source_predictions_sha256=before,
        source_predictions_unchanged=file_hash(path) == before,
        preserves_old_ml_hourly_totals=True, source_weights_unchanged=True,
        profile='sameoriginalcausalGRUstationnormalizationaggregate', scores=scores,
        comparison_scope='all19stationsSep30nextquarter06-23:45; reuseddevelopmentday',
        held_out_dates_other_models_not_available=True, operational_saving=None)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(scores), flush=True)
    return result
