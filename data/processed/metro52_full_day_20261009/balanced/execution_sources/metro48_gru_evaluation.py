"""Predeclared chronological GRU benchmark; next genuine quarter targets only."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from line1 import largest_remainders
from metro48_forecasts import calibrate_reserve, score_rows
from metro48_gru import FrozenQuarterGRU, MODEL_INPUTS
from metro_ml import FrozenMetroForecaster, INPUTS, file_hash

OUTPUT = Path(__file__).resolve().parent / 'data/processed/metro48_forecast_benchmark_20261009'


def benchmark_contract():
    return dict(calibration_dates=['2026-09-02', '2026-09-07', '2026-09-10'],
        evaluation_dates=['2026-09-16', '2026-09-21', '2026-09-28'],
        development_dates=['2026-09-30'], old_ml_evaluation_dates=['2026-09-30'],
        decision_hours='06:00-23:45 every15minutes', target='next genuine station quarter only',
        stations='all19red_line_stations', native_gru_update_minutes=[0, 30], shifted_minutes=[15, 45],
        peak_hours='weekday08:00-10:00and17:00-19:00', reserve_coverage_target=.99,
        optimization_parameters_selected_on_evaluation=False, statistical_holdout_pristine=False,
        evaluation_limit='Historical September results were previously inspected; these dates test frozen user weights, not a pristine project holdout. Old final LightGBM trained throughSep29 is evaluated onlySep30 without embargo.',
        models=dict(GRU='frozenstandaloneB3; causal historyflagsFalse; shiftedupdatesexperimental',
                    MEDIAN_SHARED_PROFILE='completepast stationhourmedian; samecausalGRUnormalizationprofile',
                    OLD_ML='savedfinalLightGBM hourlyupdate; causalhistoryflagsFalse; Sep30only'))


def matched_median_count(history, station_id, target, profile):
    now = pd.Timestamp(target)
    sample = history[history.station_id.eq(station_id) & history.hour_start.dt.hour.eq(now.hour) &
                     (history.hour_start + pd.Timedelta(hours=1) <= now.floor('h'))].total
    if sample.empty:
        raise ValueError(f'missing complete hourly history for {station_id}/{now.hour}')
    total = int(np.floor(sample.median() + .5))
    weights = np.maximum(0, np.rint(np.asarray(profile) * 1_000_000_000)).astype(np.int64).tolist()
    return largest_remainders(total, weights)[now.minute // 15]


def reserve_coverage(rows, reserve):
    residual = reserve['normalized_residual']
    if residual is None:
        return dict(rows=len(rows), coverage_percent=None, operational_capacity_guarantee=False)
    upper = rows.predicted_count.to_numpy(float) + np.ceil(residual * np.maximum(rows.predicted_count.to_numpy(float), 1))
    covered = int((rows.actual_count.to_numpy(float) <= upper).sum())
    return dict(rows=len(rows), covered_rows=covered, coverage_percent=100*covered/len(rows),
                zero_prediction_rows=int(rows.predicted_count.eq(0).sum()),
                operational_capacity_guarantee=False, exchangeability_under_temporal_drift='unverified')


def _hour_history(frame):
    data = frame[frame.is_complete.eq(True)].copy()
    data['hour_start'] = data.interval_start.dt.floor('h')
    hours = data.groupby(['station_id', 'hour_start']).agg(total=('entry_count', 'sum'),
        n=('entry_count', 'size'), sources=('source_id', 'nunique')).reset_index()
    return hours[hours.n.eq(4) & hours.sources.eq(1)]


def _scores(predictions, frame):
    records = []
    groups = [('all', []), ('date', ['date']), ('station', ['station_id']),
              ('update_mode', ['update_mode']), ('period', ['period']),
              ('date_update', ['date', 'update_mode']), ('date_period', ['date', 'period'])]
    for scope, extra in groups:
        keys = ['model', 'stage', *extra]
        for values, rows in predictions.groupby(keys, sort=True):
            labels = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
            records.append(dict(scope=scope, **labels, **score_rows(rows.to_dict('records'), frame)))
    return records


def _plot(output, predictions, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'GRU': '#166534', 'MEDIAN_SHARED_PROFILE': '#c2410c', 'OLD_ML': '#1d4ed8'}
    labels = {'GRU': 'GRU', 'MEDIAN_SHARED_PROFILE': 'Медиана, общий профиль', 'OLD_ML': 'Старая LightGBM'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    daily = [r for r in scores if r['scope'] == 'date' and r['stage'] != 'calibration']
    for model in predictions.model.unique():
        rows = [r for r in daily if r['model'] == model]
        axes[0].plot([r['date'][5:] for r in rows], [r['wape_percent'] for r in rows], marker='o',
                     color=colors[model], label=labels[model])
    axes[0].set(ylabel='WAPE, %', title='Следующая четверть: все 19 станций')
    axes[0].legend()
    rows = predictions[predictions.date.eq('2026-09-30')]
    for model, subset in rows.groupby('model'):
        totals = subset.groupby('interval_start').predicted_count.sum()
        axes[1].plot(totals.index.tz_convert('Europe/Moscow').hour + totals.index.minute/60,
                     totals.to_numpy(), label=labels[model], color=colors[model])
    actual = rows.drop_duplicates(['station_id', 'interval_start']).groupby('interval_start').actual_count.sum()
    axes[1].plot(actual.index.tz_convert('Europe/Moscow').hour + actual.index.minute/60,
                 actual.to_numpy(), color='black', alpha=.6, label='Фактические входы')
    axes[1].set(xlabel='Время, МСК', ylabel='Входы за 15 минут', title='30 сентября: общий входной поток')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'forecast_benchmark.png', dpi=160)
    plt.close(fig)


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = benchmark_contract()
    (output / 'contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf8')
    frame_path = Path('data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet')
    protected = [frame_path, Path('data/raw/metro/data.zip'), Path('data/raw/metro/lgbm_final.zip'),
                 *sorted((MODEL_INPUTS / 'stack_gru_2026-09').glob('*')),
                 Path(__file__), Path('metro48_gru.py'), Path('metro48_forecasts.py')]
    hashes = {str(p.resolve()): file_hash(p) for p in protected}
    start = time.perf_counter()
    gru = FrozenQuarterGRU()
    old = FrozenMetroForecaster(use_retrospective_history_flags=False)
    frame = pd.read_parquet(frame_path)
    stations = list(gru.mapping)
    hours = _hour_history(frame)
    actual = frame.set_index(['station_id', 'interval_start'])
    predictions, receipts = [], []
    old_cache, median_cache, profile_cache = {}, {}, {}
    for stage, dates in [('calibration', contract['calibration_dates']),
                         ('evaluation', contract['evaluation_dates']),
                         ('development', contract['development_dates'])]:
        for date in dates:
            for now in pd.date_range(date + ' 06:00', periods=72, freq='15min', tz='Europe/Moscow'):
                tick = time.perf_counter()
                bundle = gru.forecast_bundle(frame, stations, now.isoformat())
                receipt = dict(as_of=now.isoformat(), model='GRU', elapsed_seconds=time.perf_counter()-tick,
                               feature_digest=bundle['source_versions']['preprocessing_input_digest'],
                               shifted_quarter_inference=bundle['source_versions']['shifted_quarter_inference'])
                receipts.append(receipt)
                model_rows = {'GRU': {r['station_id']: r for r in bundle['rows'] if pd.Timestamp(r['interval_start']) == now}}
                if stage == 'development':
                    floor = now.floor('h')
                    if floor not in old_cache:
                        old_cache[floor] = old.forecast_bundle(frame, stations, floor.isoformat())
                    model_rows['OLD_ML'] = {r['station_id']: r for r in old_cache[floor]['rows'] if pd.Timestamp(r['interval_start']) == now}
                native = now.minute % 30 == 0
                peak = now.dayofweek < 5 and (8 <= now.hour < 10 or 17 <= now.hour < 19)
                for sid in stations:
                    key = sid, now.floor('h')
                    if key not in profile_cache:
                        profile_cache[key] = gru.station_profile(sid, now)
                    if key not in median_cache:
                        sample = hours[hours.station_id.eq(sid) & hours.hour_start.dt.hour.eq(now.hour) &
                                       (hours.hour_start + pd.Timedelta(hours=1) <= now.floor('h'))].total
                        if sample.empty:
                            raise ValueError(f'missing median history: {sid}/{now}')
                        total = int(np.floor(sample.median() + .5))
                        weights = np.rint(profile_cache[key] * 1_000_000_000).astype(np.int64).tolist()
                        median_cache[key] = largest_remainders(total, weights)
                    counts = {model: rowmap[sid]['predicted_count'] for model, rowmap in model_rows.items()}
                    counts['MEDIAN_SHARED_PROFILE'] = median_cache[key][now.minute // 15]
                    target = actual.loc[(sid, now)]
                    if not target.is_complete or target.source_id != 'uploaded-spb-line1-15min':
                        raise ValueError(f'target lacks genuine complete quarter: {sid}/{now}')
                    for model, count in counts.items():
                        issued = now.floor('h') if model == 'OLD_ML' else now
                        predictions.append(dict(model=model, stage=stage, date=date, station_id=sid,
                            issued_at=issued.isoformat(), interval_start=now.isoformat(),
                            interval_end=(now+pd.Timedelta(minutes=15)).isoformat(), predicted_count=int(count),
                            actual_count=int(target.entry_count), update_mode='native' if native else 'shifted',
                            period='peak' if peak else 'nonpeak', fallback=sid == 'L1_07'))
                if date == '2026-09-30' and pd.Timestamp('2026-09-30 09:15', tz='Europe/Moscow') <= now <= pd.Timestamp('2026-09-30 14:00', tz='Europe/Moscow'):
                    cache_dir = output / 'finite_episode_gru_cache'
                    cache_dir.mkdir(exist_ok=True)
                    (cache_dir / f'forecast_{now.strftime("%H%M")}.json').write_text(json.dumps(bundle, ensure_ascii=False), encoding='utf8')
            subset = pd.DataFrame(predictions)
            subset = subset[subset.date.eq(date)]
            print(json.dumps(dict(date=date, stage=stage, quarters=72,
                                  metrics={model: score_rows(rows.to_dict('records'), frame)
                                           for model, rows in subset.groupby('model')})), flush=True)
    data = pd.DataFrame(predictions)
    for column in ['issued_at', 'interval_start', 'interval_end']:
        data[column] = pd.to_datetime(data[column], utc=True)
    data.to_parquet(output / 'next_quarter_predictions.parquet', index=False)
    data.to_csv(output / 'next_quarter_predictions.csv', index=False, encoding='utf8')
    scores = _scores(data, frame)
    reserve_results = {}
    for model in ['GRU', 'MEDIAN_SHARED_PROFILE']:
        cal = data[data.model.eq(model) & data.stage.eq('calibration')]
        reserve = calibrate_reserve(cal, '2026-09-16T06:00:00+03:00', coverage=.99)
        evaluation = data[data.model.eq(model) & data.stage.eq('evaluation')]
        reserve_results[model] = dict(calibration=reserve, evaluation_coverage=reserve_coverage(evaluation, reserve))
    unchanged = all(file_hash(path) == expected for path, expected in hashes.items())
    if not unchanged:
        raise RuntimeError('protected source or weight changed during evaluation')
    result = dict(contract=contract, scores=scores, reserves=reserve_results, inference_receipts=receipts,
                  elapsed_seconds=time.perf_counter()-start, protected_hashes=hashes,
                  protected_inputs_unchanged=unchanged, weight_contract=gru.contract(),
                  old_ml_train_end=old.meta['train_end'], cash_saving=None, capacity_guarantee=False)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    _plot(output, data, scores)
    lines = ['# Проверка модели GRU', '',
             'Прогнозируем входы на станцию за следующую четверть. Целевые аномалии включены в оценку.', '',
             'GRU выдаёт четыре получасовых слота. Обновления в :15 и :45 проверены отдельно как перенос на новую сетку.', '',
             '| Модель | Этап | WAPE, % | Смещение, % |', '|---|---|---:|---:|']
    for row in scores:
        if row['scope'] == 'all':
            lines.append(f"| {row['model']} | {row['stage']} | {row['wape_percent']:.3f} | {row['signed_bias_percent']:.3f} |")
    lines.extend(['', 'Медиана использует тот же причинный профиль распределения внутри часа. Обучение GRU заканчивается 31 июля; веса не изменялись.',
                  'Старая LightGBM оценена только 30 сентября: её обучение заканчивается 29 сентября, временного зазора нет.',
                  'Сентябрь уже исследовался в проекте. Это проверка замороженных весов на заранее выбранных датах, а не новая нетронутая выборка.',
                  'Резерв 99% рассчитан на отдельных ранних датах. Его покрытие не является гарантией вместимости составов или денежной экономии.', '',
                  '![График прогнозов](forecast_benchmark.png)'])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf8')
    return result


if __name__ == '__main__':
    run()
